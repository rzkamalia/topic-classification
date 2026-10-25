from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAMES = {
    "modernbert-base": "answerdotai/ModernBERT-base",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "distilbert-base": "distilbert/distilbert-base-uncased",
}


def get_device(device="auto"):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_name(model_name):
    checkpoint = MODEL_NAMES.get(model_name, model_name)
    if checkpoint not in MODEL_NAMES.values():
        raise ValueError(f"Choose one of: {', '.join(MODEL_NAMES)}")
    return checkpoint


def model_loading_options(model_name):
    # Other encoders use their supported default attention implementation.
    if model_name in {"modernbert", MODEL_NAMES["modernbert-base"]}:
        return {"attn_implementation": "sdpa"}
    return {}


def load_pretrained_model(model_name, label_to_id):
    checkpoint = resolve_model_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(label_to_id),
        label2id=label_to_id,
        id2label={index: label for label, index in label_to_id.items()},
        **model_loading_options(checkpoint),
    )
    return model, tokenizer


def resolve_model_dir(model_dir):
    model_dir = Path(model_dir)
    if (model_dir / "best_model").is_dir():
        model_dir = model_dir / "best_model"
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            "model_dir must point to a saved best_model folder or its experiment folder"
        )
    return model_dir


def get_label_mapping(config):
    id_to_label = {int(index): label for index, label in config.id2label.items()}
    if set(id_to_label) != set(range(config.num_labels)):
        raise ValueError("Saved model has an incomplete category mapping")
    return id_to_label


def get_input_limit(tokenizer, model_config):
    limits = [
        value for value in (
            tokenizer.model_max_length,
            getattr(model_config, "max_position_embeddings", None),
        )
        if isinstance(value, int) and 0 < value < 1_000_000
    ]
    if not limits:
        raise ValueError("Cannot determine the model's maximum input length")
    return min(limits)


def load_saved_model(model_dir, device="auto"):
    model_dir = resolve_model_dir(model_dir)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    get_label_mapping(config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        **model_loading_options(config.model_type),
    ).to(get_device(device))
    model.eval()
    return model, tokenizer
