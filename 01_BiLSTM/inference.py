"""Load a trained NER checkpoint and predict BIO tags and entities.

Example:
    python inference.py --checkpoint checkpoints/bilstm_crf_best.pt --text "患者发热三天"
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config import AppConfig
from model import NERBiLSTM, NERBiLSTM_CRF


MODELS: dict[str, type[nn.Module]] = {
    "BiLSTM": NERBiLSTM,
    "BiLSTM_CRF": NERBiLSTM_CRF,
}
PAD_TOKEN = "PAD"
UNK_TOKEN = "UNK"


def get_device(device_name: str) -> torch.device:
    """Return the requested device, falling back to CPU when unavailable."""

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return device


def _mapping(value: Any, name: str) -> dict[str, int]:
    """Validate and normalize one checkpoint mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint field {name!r} must be a mapping.")
    try:
        mapping = {str(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"Checkpoint field {name!r} must contain integer IDs.") from error
    return mapping


def _id_to_tag(tag2id: Mapping[str, int]) -> dict[int, str]:
    """Build a validated reverse tag mapping for prediction output."""

    if "PAD" in tag2id:
        raise ValueError("Checkpoint tag2id must not define 'PAD'.")
    id2tag = {tag_id: tag for tag, tag_id in tag2id.items()}
    if len(id2tag) != len(tag2id) or set(id2tag) != set(range(len(tag2id))):
        raise ValueError("Checkpoint tag IDs must be unique and contiguous from 0.")
    return id2tag


def load_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore the model and mappings stored by ``train.py``."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping saved by train.py.")

    model_name = checkpoint.get("model_name")
    if model_name not in MODELS:
        raise ValueError(
            f"Unsupported checkpoint model {model_name!r}; expected one of {list(MODELS)}."
        )
    word2id = _mapping(checkpoint.get("word2id"), "word2id")
    tag2id = _mapping(checkpoint.get("tag2id"), "tag2id")
    if PAD_TOKEN not in word2id or UNK_TOKEN not in word2id:
        raise ValueError("Checkpoint word2id must define 'PAD' and 'UNK'.")
    id2tag = _id_to_tag(tag2id)

    try:
        embedding_dim = int(checkpoint["embedding_dim"])
        hidden_dim = int(checkpoint["hidden_dim"])
        dropout = float(checkpoint["dropout"])
        state_dict = checkpoint["state_dict"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Checkpoint is missing required model configuration.") from error
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint state_dict must be a mapping.")

    model = MODELS[model_name](embedding_dim, hidden_dim, dropout, word2id, tag2id)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    metadata = {
        "checkpoint": checkpoint_path,
        "model_name": model_name,
        "word2id": word2id,
        "tag2id": tag2id,
        "id2tag": id2tag,
        "epoch": checkpoint.get("epoch"),
    }
    return model, metadata


def _parse_bio_tag(tag: str) -> tuple[str, str | None]:
    """Return a BIO prefix and entity label; malformed tags are treated as O."""

    prefix, separator, label = tag.partition("-")
    if separator and prefix in {"B", "I"} and label:
        return prefix, label
    return "O", None


def extract_entities(text: str, tags: Sequence[str]) -> list[dict[str, Any]]:
    """Merge character-level BIO tags into entities with exclusive end offsets."""

    if len(text) != len(tags):
        raise ValueError("text and tags must have the same length.")

    entities: list[dict[str, Any]] = []
    start: int | None = None
    label: str | None = None

    def close_entity(end: int) -> None:
        nonlocal start, label
        if start is not None and label is not None:
            entities.append(
                {"text": text[start:end], "start": start, "end": end, "label": label}
            )
        start, label = None, None

    for index, tag in enumerate(tags):
        prefix, next_label = _parse_bio_tag(tag)
        if prefix == "I" and next_label == label and start is not None:
            continue
        if prefix in {"B", "I"}:
            close_entity(index)
            start, label = index, next_label
        else:
            close_entity(index)
    close_entity(len(text))
    return entities


@torch.no_grad()
def predict_text(
    model: nn.Module,
    metadata: Mapping[str, Any],
    text: str,
    device: torch.device,
) -> dict[str, Any]:
    """Predict one BIO tag per character and merge the resulting entities."""

    if not text:
        raise ValueError("text must not be empty.")

    word2id = _mapping(metadata.get("word2id"), "word2id")
    id2tag = metadata.get("id2tag")
    if not isinstance(id2tag, Mapping):
        raise ValueError("metadata must contain an id2tag mapping from load_model().")
    if UNK_TOKEN not in word2id:
        raise ValueError("word2id must define 'UNK'.")

    input_ids = torch.tensor(
        [[word2id.get(character, word2id[UNK_TOKEN]) for character in text]],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    model_name = metadata.get("model_name")
    if model_name == "BiLSTM":
        if not isinstance(model, NERBiLSTM):
            raise TypeError("BiLSTM checkpoint requires an NERBiLSTM model.")
        tag_ids = model(input_ids, attention_mask).argmax(dim=-1)[0].tolist()
    elif model_name == "BiLSTM_CRF":
        if not isinstance(model, NERBiLSTM_CRF):
            raise TypeError("BiLSTM_CRF checkpoint requires an NERBiLSTM_CRF model.")
        tag_ids = model.predict(input_ids, attention_mask)[0]
    else:
        raise ValueError(f"Unsupported model_name in metadata: {model_name!r}.")

    try:
        tags = [str(id2tag[tag_id]) for tag_id in tag_ids]
    except KeyError as error:
        raise RuntimeError("Model predicted a tag ID absent from the checkpoint mapping.") from error

    return {
        "model_name": model_name,
        "checkpoint": str(metadata.get("checkpoint", "")),
        "epoch": metadata.get("epoch"),
        "text": text,
        "tags": [
            {"index": index, "character": character, "tag": tag}
            for index, (character, tag) in enumerate(zip(text, tags))
        ],
        "entities": extract_entities(text, tags),
    }


def print_result(result: Mapping[str, Any], show_tags: bool = False) -> None:
    """Print a compact, readable prediction summary."""

    print(
        f"model={result['model_name']}  epoch={result.get('epoch')}  "
        f"checkpoint={result['checkpoint']}"
    )
    print("\nENTITIES")
    print("START    END  LABEL        TEXT")
    print("-" * 52)
    entities = result["entities"]
    if entities:
        for entity in entities:
            print(
                f"{entity['start']:>5}  {entity['end']:>5}  "
                f"{entity['label']:<12}  {entity['text']}"
            )
    else:
        print("(no entities)")

    if show_tags:
        print("\nBIO TAGS")
        print("INDEX  CHARACTER  TAG")
        print("-" * 32)
        for item in result["tags"]:
            character = json.dumps(item["character"], ensure_ascii=False)
            print(f"{item['index']:>5}  {character:<9}  {item['tag']}")


def parse_args() -> argparse.Namespace:
    """Parse command-line inference options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint saved by train.py")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="text to predict")
    input_group.add_argument("--input-file", type=Path, help="UTF-8 text file to predict")
    parser.add_argument("--device", default=AppConfig().device, help="for example: cpu, mps, cuda")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument("--show-tags", action="store_true", help="also print character-level BIO tags")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text if args.text is not None else args.input_file.read_text(encoding="utf-8")
    device = get_device(args.device)
    model, metadata = load_model(args.checkpoint, device)
    result = predict_text(model, metadata, text, device)
    print_result(result, args.show_tags)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nJSON result: {args.output}")


if __name__ == "__main__":
    main()
