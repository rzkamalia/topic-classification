import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model import (
    get_device,
    get_input_limit,
    load_pretrained_model,
    load_saved_model,
    resolve_model_name,
)
from saving import (
    save_best_model,
    save_history,
    save_test_results,
    save_training_setup,
)
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_frames(frames):
    if not {"train", "val"}.issubset(frames):
        raise ValueError("frames must contain train and val splits")
    prepared = {}
    seen_texts = set()
    for name, original in frames.items():
        if name not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {name}")
        if original.empty:
            raise ValueError(f"{name}: split is empty")
        required = {"headline", "short_description", "category"}
        if missing := required - set(original.columns):
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        frame = original.copy().reset_index(drop=True)
        for column in sorted(required):
            frame[column] = (
                frame[column].astype("string").str.normalize("NFKC")
                .str.replace(r"\s+", " ", regex=True).str.strip()
            )
            invalid = (
                frame[column].isna() | frame[column].eq("")
                | frame[column].str.casefold().eq("null")
            ).fillna(True)
            if invalid.any():
                raise ValueError(f"{name}: clean missing values in {column} before training")
        frame["category"] = frame["category"].str.upper()
        frame["text"] = frame["headline"] + " " + frame["short_description"]
        text_pairs = set(zip(
            frame["headline"].str.casefold(), frame["short_description"].str.casefold()
        ))
        if seen_texts & text_pairs:
            raise ValueError(f"{name}: duplicate headline/description pairs overlap another split")
        seen_texts.update(text_pairs)
        prepared[name] = frame

    label_to_id = {
        label: index
        for index, label in enumerate(sorted(prepared["train"]["category"].unique()))
    }
    for name, frame in prepared.items():
        unseen = set(frame["category"]) - label_to_id.keys()
        if unseen:
            raise ValueError(f"{name}: labels not present in train: {sorted(unseen)}")
    return prepared, label_to_id


def determine_max_length(train_texts, tokenizer, model_config):
    context_limit = get_input_limit(tokenizer, model_config)
    # Inspect train only, counting pretrained subword tokens including special tokens.
    train_encoding = tokenizer(train_texts, truncation=False, padding=False, verbose=False)
    train_max_length = max(len(ids) for ids in train_encoding["input_ids"])
    max_length = min(train_max_length, context_limit)
    return max_length, train_max_length, context_limit


class NewsDataset(Dataset):

    def __init__(self, frame, tokenizer, label_to_id, max_length):
        self.encodings = tokenizer(
            frame["text"].tolist(), truncation=True, max_length=max_length, padding=False
        )
        self.labels = [label_to_id[label] for label in frame["category"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def build_loaders(frames, tokenizer, label_to_id, max_length, *, batch_size=8, seed=42):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    return {
        name: DataLoader(
            NewsDataset(frame, tokenizer, label_to_id, max_length),
            batch_size=batch_size,
            shuffle=name == "train",
            collate_fn=collator,
            generator=torch.Generator().manual_seed(seed),
        )
        for name, frame in frames.items()
    }


def calculate_metrics(targets, predictions, label_ids):
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(
            targets, predictions, labels=label_ids, average="macro", zero_division=0
        )),
    }


def run_epoch(
    model, loader, device, label_ids, optimizer=None, scheduler=None, scaler=None,
    gradient_clip=1.0, use_amp=True,
):
    training = optimizer is not None
    model.train(training)
    total_loss, total_examples = 0.0, 0
    targets, predictions = [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch["labels"]
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type, enabled=use_amp and device.type == "cuda"
            ):
                output = model(**batch)
                loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Advance the schedule only if AMP performed the optimizer step.
                if scheduler is not None and scaler.get_scale() >= previous_scale:
                    scheduler.step()
        total_loss += loss.detach().item() * labels.size(0)
        total_examples += labels.size(0)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(output.logits.detach().argmax(dim=1).cpu().tolist())
    if not total_examples:
        raise ValueError("Cannot evaluate an empty loader")
    return {
        "loss": total_loss / total_examples,
        **calculate_metrics(targets, predictions, label_ids),
    }, targets, predictions


def train_model(
    frames, model_name, output_dir, epochs=50, batch_size=16, patience=2,
    seed=42, learning_rate=2e-5, weight_decay=0.01,
    gradient_clip=1.0, warmup_ratio=0.1, use_amp=True,
):
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    checkpoint = resolve_model_name(model_name)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output_dir: {output_dir}")
    frames, label_to_id = prepare_frames(frames)
    set_seed(seed)
    device = get_device()
    label_names = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    label_ids = list(range(len(label_names)))
    model, tokenizer = load_pretrained_model(checkpoint, label_to_id)
    max_length, train_max_length, context_limit = determine_max_length(
        frames["train"]["text"].tolist(), tokenizer, model.config
    )
    # Save the training input limit with the tokenizer for consistent inference.
    tokenizer.model_max_length = max_length
    loaders = build_loaders(
        frames, tokenizer, label_to_id, max_length, batch_size=batch_size, seed=seed
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_updates = len(loaders["train"]) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * warmup_ratio),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    best_dir = save_training_setup(
        output_dir, checkpoint=checkpoint, label_to_id=label_to_id,
        optimizer=optimizer, epochs=epochs, batch_size=batch_size,
        patience=patience, device=device, max_length=max_length,
        train_max_length=train_max_length, context_limit=context_limit,
        seed=seed, gradient_clip=gradient_clip,
        warmup_ratio=warmup_ratio, use_amp=use_amp,
    )
    print(f"Model: {checkpoint}; device: {device}; input limit: {max_length} tokens", flush=True)
    history, best_score, best_epoch, stale_epochs = [], -1.0, 0, 0
    best_validation_metrics = None
    for epoch in range(1, epochs + 1):
        train_metrics, _, _ = run_epoch(
            model, loaders["train"], device, label_ids,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            gradient_clip=gradient_clip, use_amp=use_amp,
        )
        val_metrics, targets, predictions = run_epoch(
            model, loaders["val"], device, label_ids, use_amp=use_amp
        )
        history.append({
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        })
        save_history(output_dir, history)
        print(
            f"Epoch {epoch}/{epochs} | train loss {train_metrics['loss']:.4f} | "
            f"train accuracy {train_metrics['accuracy']:.4f} | train macro-F1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | val accuracy {val_metrics['accuracy']:.4f} | "
            f"val macro-F1 {val_metrics['macro_f1']:.4f}", flush=True,
        )
        if val_metrics["macro_f1"] > best_score:
            best_score, best_epoch, stale_epochs = val_metrics["macro_f1"], epoch, 0
            best_validation_metrics = dict(val_metrics)
            save_best_model(
                output_dir, model, tokenizer, epoch, val_metrics,
                targets, predictions, label_names,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Release the final-epoch model and optimizer before loading the best checkpoint.
    del optimizer, scheduler, scaler, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    model, tokenizer = load_saved_model(best_dir, device=device)
    test_metrics = None
    if "test" in loaders:
        # Evaluate test once, after checkpoint selection is complete.
        test_metrics, targets, predictions = run_epoch(
            model, loaders["test"], device, label_ids, use_amp=use_amp
        )
        save_test_results(output_dir, test_metrics, targets, predictions, label_names)
    print(f"Best epoch: {best_epoch}; validation macro-F1: {best_score:.4f}")
    print("Artifacts saved to:", output_dir.resolve())
    return {
        "model": model, "tokenizer": tokenizer, "history": pd.DataFrame(history),
        "best_epoch": best_epoch, "best_validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics, "output_dir": output_dir,
    }
