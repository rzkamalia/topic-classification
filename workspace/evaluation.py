import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from tqdm.auto import tqdm

from model import (
    get_device,
    get_input_limit,
    get_label_mapping,
    load_saved_model,
    resolve_model_dir,
)


@torch.inference_mode()
def evaluate_model(model_dir, test_csv, output_dir, batch_size=8, device="auto"):
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output_dir: {output_dir}")
    frame = pd.read_csv(test_csv, keep_default_na=False)
    if frame.empty:
        raise ValueError("Test dataset is empty")
    required = {"headline", "short_description", "category"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing test columns: {sorted(missing)}")

    # Match training normalization; do not silently drop or relabel test examples.
    normalized = frame.copy()
    for column in sorted(required):
        normalized[column] = (
            normalized[column].astype("string").str.normalize("NFKC")
            .str.replace(r"\s+", " ", regex=True).str.strip()
        )
        invalid = (
            normalized[column].isna() | normalized[column].eq("")
            | normalized[column].str.casefold().eq("null")
        ).fillna(True)
        if invalid.any():
            raise ValueError(f"Test data contains missing values in {column}")
    normalized["category"] = normalized["category"].str.upper()
    texts = (normalized["headline"] + " " + normalized["short_description"]).tolist()

    model_dir = resolve_model_dir(model_dir)
    device = get_device(device)
    model, tokenizer = load_saved_model(model_dir, device=device)
    id_to_label = get_label_mapping(model.config)
    label_ids = list(range(model.config.num_labels))
    label_names = [id_to_label[index] for index in label_ids]
    label_to_id = {label: index for index, label in id_to_label.items()}
    if unknown := set(normalized["category"]) - label_to_id.keys():
        raise ValueError(f"Test categories are not in the saved label mapping: {sorted(unknown)}")
    targets = [label_to_id[label] for label in normalized["category"]]

    max_length = get_input_limit(tokenizer, model.config)
    print(f"Evaluating {len(frame):,} test articles on {device}", flush=True)

    total_loss = 0.0
    predictions, scores = [], []
    progress = tqdm(range(0, len(texts), batch_size), desc="Test evaluation", unit="batch")
    for start in progress:
        inputs = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = torch.tensor(targets[start:start + batch_size], dtype=torch.long, device=device)
        logits = model(**inputs).logits.float()
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite test loss detected")
        total_loss += loss.item()
        batch_scores, batch_predictions = logits.softmax(dim=-1).max(dim=-1)
        predictions.extend(batch_predictions.cpu().tolist())
        scores.extend(batch_scores.cpu().tolist())
        progress.set_postfix(loss=f"{total_loss / len(predictions):.4f}", refresh=False)

    # Macro F1 uses the complete saved label set, matching the training metric.
    metrics = {
        "loss": total_loss / len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(
            targets, predictions, labels=label_ids, average="macro", zero_division=0
        )),
    }
    per_class_precision = pd.DataFrame({
        "precision": precision_score(
            targets, predictions, labels=label_ids, average=None, zero_division=0
        ),
    }, index=label_names)
    matrix = pd.DataFrame(
        confusion_matrix(targets, predictions, labels=label_ids),
        index=label_names, columns=label_names,
    )
    predicted_frame = frame.copy()
    predicted_frame["true_category"] = [id_to_label[index] for index in targets]
    predicted_frame["predicted_category"] = [id_to_label[index] for index in predictions]
    predicted_frame["prediction_score"] = scores
    predicted_frame["is_correct"] = [target == prediction for target, prediction in zip(targets, predictions)]

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "evaluation_config.json").write_text(json.dumps({
        "model_dir": str(model_dir.resolve()), "test_csv": str(Path(test_csv).resolve()),
        "rows": len(frame), "batch_size": batch_size, "max_length": max_length,
        "device": str(device), "precision": "float32", "macro_f1_labels": "all_saved_labels",
    }, indent=2), encoding="utf-8")
    per_class_precision.to_csv(output_dir / "test_report.csv", index_label="label")
    matrix.to_csv(output_dir / "test_confusion_matrix.csv", index_label="true_label")
    predicted_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print("Evaluation saved to:", output_dir.resolve())
    return {
        "metrics": metrics, "per_class_precision": per_class_precision,
        "confusion_matrix": matrix, "predictions": predicted_frame,
        "output_dir": output_dir,
    }
