import json
from pathlib import Path

import pandas as pd
from model import model_loading_options
from sklearn.metrics import confusion_matrix, precision_score


def write_json(path, values):
    Path(path).write_text(
        json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_training_setup(
    output_dir, *, checkpoint, label_to_id, optimizer, epochs, batch_size,
    patience, device, max_length, train_max_length, context_limit,
    seed, gradient_clip, warmup_ratio, use_amp,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "label_to_id.json", label_to_id)
    write_json(output_dir / "config.json", {
        "checkpoint": checkpoint,
        "batch_size": batch_size,
        "optimizer": type(optimizer).__name__,
        "optimizer_defaults": optimizer.defaults,
        "learning_rate": optimizer.defaults["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "manual_seed": True,
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "learning_rate_schedule": "linear_decay_with_warmup",
        "gradient_clip": gradient_clip,
        "warmup_ratio": warmup_ratio,
        "amp_enabled": use_amp and device.type == "cuda",
        "selection_metric": "validation_macro_f1",
        "macro_f1_labels": "all_training_labels",
        "max_length": max_length,
        "untruncated_train_max_length": train_max_length,
        "model_context_limit": context_limit,
        "device": str(device),
        "input_format": "headline + space + short_description",
        "input_normalization": "NFKC and whitespace; case and punctuation preserved",
        "loading_options": model_loading_options(checkpoint),
    })
    return output_dir / "best_model"


def save_history(output_dir, history):
    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(output_dir) / "history.csv", index=False)
    return history_df


def save_report(output_dir, phase, targets, predictions, label_names):
    output_dir = Path(output_dir)
    label_ids = list(range(len(label_names)))
    precision = precision_score(
        targets, predictions, labels=label_ids, average=None, zero_division=0
    )
    pd.DataFrame({"precision": precision}, index=label_names).to_csv(
        output_dir / f"{phase}_report.csv", index_label="label"
    )
    matrix = confusion_matrix(targets, predictions, labels=label_ids)
    pd.DataFrame(matrix, index=label_names, columns=label_names).to_csv(
        output_dir / f"{phase}_confusion_matrix.csv", index_label="true_label"
    )


def save_best_model(
    output_dir, model, tokenizer, epoch, validation_metrics,
    targets, predictions, label_names,
):
    output_dir = Path(output_dir)
    best_dir = output_dir / "best_model"
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    write_json(
        output_dir / "best_validation_metrics.json", {"epoch": epoch, **validation_metrics}
    )
    save_report(output_dir, "validation", targets, predictions, label_names)


def save_test_results(output_dir, metrics, targets, predictions, label_names):
    output_dir = Path(output_dir)
    write_json(output_dir / "test_metrics.json", metrics)
    save_report(output_dir, "test", targets, predictions, label_names)
    pd.DataFrame({
        "row": range(len(targets)),
        "true_label": [label_names[index] for index in targets],
        "predicted_label": [label_names[index] for index in predictions],
    }).to_csv(output_dir / "test_predictions.csv", index=False)
