"""Dataset and batching utilities for character-level NER training."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "PAD"
UNK_TOKEN = "UNK"
LABEL_PAD_ID = -100

Sample = tuple[list[str], list[int]]


class NERDataset(Dataset[Sample]):
    """A thin PyTorch dataset wrapper around token/tag sequences."""

    def __init__(self, data: Sequence[Sample]):
        self.data = list(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Sample:
        return self.data[index]


def _load_tag2id(path: str | Path) -> dict[str, int]:
    """Load and validate the tag-to-ID mapping used by the processed corpus."""

    try:
        with Path(path).open("r", encoding="utf-8") as file:
            raw_mapping = json.load(file)
    except (OSError, TypeError) as error:
        raise ValueError(f"Unable to read tag2id mapping: {path}") from error

    if not isinstance(raw_mapping, Mapping):
        raise ValueError("tag2id_file must contain a JSON object.")

    try:
        tag2id = {str(tag): int(tag_id) for tag, tag_id in raw_mapping.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("Every tag ID in tag2id_file must be an integer.") from error

    if "O" not in tag2id:
        raise ValueError("tag2id_file must define the outside tag 'O'.")
    if "PAD" in tag2id:
        raise ValueError("tag2id_file must not define 'PAD'; label padding uses -100 internally.")
    return tag2id


def _resolve_train_path(config: Any) -> Path:
    """Use an explicit training path, falling back to the preprocessing output."""

    train_path = getattr(config, "train_path", None)
    if train_path is None:
        train_path = getattr(config, "processed_path", None)
    if train_path is None:
        raise AttributeError("Config must define train_path or processed_path.")
    return Path(train_path)


def _resolve_vocab_path(config: Any, train_path: Path) -> Path:
    """Use an explicit vocabulary path or save it beside the training corpus."""

    vocab_path = getattr(config, "vocab_path", None)
    if vocab_path is not None:
        return Path(vocab_path)
    return train_path.with_name("vocab.txt")


def _resolve_metadata_path(config: Any, train_path: Path) -> Path:
    """Return the preprocessing sidecar that maps samples to case IDs."""

    metadata_path = getattr(config, "sample_metadata_path", None)
    if metadata_path is not None:
        return Path(metadata_path)
    return train_path.with_name("sample_metadata.jsonl")


def _label_to_id(
    label: str | int, tag2id: Mapping[str, int], *, line_number: int | None = None
) -> int:
    """Convert an integer label or a BIO tag to its numeric ID."""

    if isinstance(label, int):
        return label

    try:
        return int(label)
    except ValueError:
        if label in tag2id:
            return tag2id[label]

    location = f" at line {line_number}" if line_number is not None else ""
    raise ValueError(f"Unknown label {label!r}{location}.")


def _write_vocab(vocab: Sequence[str], vocab_path: Path) -> None:
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")


def _read_processed_data(config: Any) -> tuple[list[Sample], Path]:
    """Read the processed BIO corpus without constructing a vocabulary.

    The preprocessor writes one ``character<TAB>label_id`` pair per line and a
    blank line between source documents.  Those blank lines are retained as
    sequence boundaries so no samples are silently joined or discarded.
    """

    train_path = _resolve_train_path(config)
    if not train_path.is_file():
        raise FileNotFoundError(f"Processed training file does not exist: {train_path}")

    tag2id_path = getattr(config, "tag2id_file", None)
    if tag2id_path is None:
        raise AttributeError("Config must define tag2id_file.")
    tag2id = _load_tag2id(tag2id_path)
    valid_label_ids = set(tag2id.values())

    data: list[Sample] = []
    characters: list[str] = []
    labels: list[int] = []

    def finish_sample() -> None:
        nonlocal characters, labels
        if characters:
            data.append((characters, labels))
            characters, labels = [], []

    with train_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                finish_sample()
                continue

            character, separator, label = line.rpartition("\t")
            if not separator or not character:
                raise ValueError(
                    f"Malformed training record at {train_path}:{line_number}; "
                    "expected a character/token followed by a tab and label."
                )

            label_id = _label_to_id(label, tag2id, line_number=line_number)
            if label_id not in valid_label_ids:
                raise ValueError(
                    f"Label ID {label_id} at {train_path}:{line_number} is not "
                    "defined in tag2id_file."
                )
            characters.append(character)
            labels.append(label_id)

    finish_sample()
    if not data:
        raise ValueError(f"No NER samples found in {train_path}.")

    return data, train_path


def _load_group_ids(config: Any, train_path: Path, sample_count: int) -> list[str]:
    """Load one case ID per processed sample and validate the sidecar file."""

    metadata_path = _resolve_metadata_path(config, train_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Sample metadata does not exist: {metadata_path}. "
            "Run utils/process.py to regenerate the processed corpus and metadata."
        )

    group_ids: list[str] = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed sample metadata at {metadata_path}:{line_number}."
                ) from error
            if not isinstance(record, Mapping) or not record.get("group_id"):
                raise ValueError(
                    f"Sample metadata at {metadata_path}:{line_number} must define group_id."
                )
            group_ids.append(str(record["group_id"]))

    if len(group_ids) != sample_count:
        raise ValueError(
            f"Sample metadata count ({len(group_ids)}) does not match processed "
            f"sample count ({sample_count}). Regenerate both files with utils/process.py."
        )
    return group_ids


def _build_vocab(
    train_data: Sequence[Sample], config: Any, train_path: Path
) -> dict[str, int]:
    """Build and persist a vocabulary from training samples only."""

    vocab = [PAD_TOKEN, UNK_TOKEN]
    known_characters = set(vocab)
    for tokens, _ in train_data:
        for character in tokens:
            if character not in known_characters:
                known_characters.add(character)
                vocab.append(character)

    _write_vocab(vocab, _resolve_vocab_path(config, train_path))
    return {character: index for index, character in enumerate(vocab)}


def calculate_class_weights(
    train_data: Sequence[Sample], tag2id: Mapping[str, int], power: float = 0.5
) -> torch.Tensor:
    """Return normalized inverse-frequency weights from training labels only.

    ``power=0.5`` applies square-root inverse frequency, which reduces the
    dominance of ``O`` without letting very rare BIO tags dominate optimization.
    """

    if power < 0:
        raise ValueError("class weight power must be non-negative.")

    active_ids = sorted(int(tag_id) for tag_id in tag2id.values())
    if active_ids != list(range(len(active_ids))):
        raise ValueError("Non-padding tag IDs must be contiguous from 0.")

    counts = torch.zeros(len(active_ids), dtype=torch.float)
    for _, labels in train_data:
        for label_id in labels:
            if label_id not in active_ids:
                raise ValueError(f"Unknown label ID in training data: {label_id}.")
            counts[label_id] += 1

    present = counts > 0
    if not torch.any(present):
        raise ValueError("Cannot calculate class weights from an empty training set.")

    weights = torch.ones_like(counts)
    inverse_frequency = (counts[present].sum() / counts[present]).pow(power)
    weights[present] = inverse_frequency / inverse_frequency.mean()
    return weights


def build_data(config: Any) -> tuple[list[Sample], dict[str, int]]:
    """Read all samples and return a vocabulary built from the train split only."""

    data, train_path = _read_processed_data(config)
    group_ids = _load_group_ids(config, train_path, len(data))
    train_data, _ = _split_data(data, group_ids, config)
    return data, _build_vocab(train_data, config, train_path)


def collate_fn(
    batch: Sequence[Sample],
    word2id: Mapping[str, int],
    tag2id: Mapping[str, int] | None = None,
    label_pad_id: int = LABEL_PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Numerically encode and pad a batch of NER sequences.

    Labels at padding positions use ``-100`` by default, matching PyTorch's
    default ``CrossEntropyLoss(ignore_index=-100)`` behavior.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    if PAD_TOKEN not in word2id or UNK_TOKEN not in word2id:
        raise ValueError(f"word2id must define {PAD_TOKEN!r} and {UNK_TOKEN!r}.")

    tag2id = tag2id or {}
    input_sequences: list[torch.Tensor] = []
    label_sequences: list[torch.Tensor] = []
    for tokens, labels in batch:
        if not tokens or len(tokens) != len(labels):
            raise ValueError(
                "Every NER sample must have matching non-empty token and label lists."
            )

        input_sequences.append(
            torch.tensor(
                [word2id.get(token, word2id[UNK_TOKEN]) for token in tokens],
                dtype=torch.long,
            )
        )
        label_sequences.append(
            torch.tensor(
                [_label_to_id(label, tag2id) for label in labels], dtype=torch.long
            )
        )

    input_ids_padded = pad_sequence(
        input_sequences, batch_first=True, padding_value=word2id[PAD_TOKEN]
    )
    labels_padded = pad_sequence(
        label_sequences, batch_first=True, padding_value=label_pad_id
    )
    attention_mask = input_ids_padded.ne(word2id[PAD_TOKEN]).long()
    return input_ids_padded, labels_padded, attention_mask


def _split_data(
    data: Sequence[Sample], group_ids: Sequence[str], config: Any
) -> tuple[list[Sample], list[Sample]]:
    """Split samples by case ID, preventing a case from leaking into validation."""

    train_ratio = float(getattr(config, "train_ratio", 0.8))
    if not 0 < train_ratio <= 1:
        raise ValueError("train_ratio must be in the interval (0, 1].")
    if len(data) != len(group_ids):
        raise ValueError("data and group_ids must contain the same number of samples.")

    unique_group_ids = list(dict.fromkeys(group_ids))
    if len(unique_group_ids) == 1 or train_ratio == 1:
        return list(data), []

    if getattr(config, "split_shuffle", True):
        random.Random(getattr(config, "seed", 42)).shuffle(unique_group_ids)

    train_group_count = max(
        1, min(len(unique_group_ids) - 1, int(len(unique_group_ids) * train_ratio))
    )
    train_group_ids = set(unique_group_ids[:train_group_count])
    train_data = [
        sample for sample, group_id in zip(data, group_ids) if group_id in train_group_ids
    ]
    dev_data = [
        sample for sample, group_id in zip(data, group_ids) if group_id not in train_group_ids
    ]
    return train_data, dev_data


def build_dataloader(config: Any) -> tuple[DataLoader, DataLoader]:
    """Build reproducible train and validation data loaders from ``config``."""

    data, train_path = _read_processed_data(config)
    group_ids = _load_group_ids(config, train_path, len(data))
    tag2id = _load_tag2id(config.tag2id_file)
    train_data, dev_data = _split_data(data, group_ids, config)
    word2id = _build_vocab(train_data, config, train_path)

    batch_size = int(getattr(config, "batch_size", 32))
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    num_workers = int(getattr(config, "num_workers", 0))
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    collator = partial(
        collate_fn,
        word2id=word2id,
        tag2id=tag2id,
        label_pad_id=int(getattr(config, "label_pad_id", LABEL_PAD_ID)),
    )
    loader_options = {
        "batch_size": batch_size,
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": bool(getattr(config, "pin_memory", False)),
        "drop_last": bool(getattr(config, "drop_last", False)),
    }
    train_dataloader = DataLoader(
        NERDataset(train_data),
        shuffle=bool(getattr(config, "train_shuffle", True)),
        **loader_options,
    )
    dev_dataloader = DataLoader(NERDataset(dev_data), shuffle=False, **loader_options)

    # Keep the mappings available to callers constructing an embedding or head
    # while preserving the original two-loader return signature.
    train_dataloader.word2id = word2id
    train_dataloader.tag2id = tag2id
    if bool(getattr(config, "use_class_weights", True)):
        train_dataloader.class_weights = calculate_class_weights(
            train_data, tag2id, float(getattr(config, "class_weight_power", 0.5))
        )
    else:
        train_dataloader.class_weights = None
    dev_dataloader.word2id = word2id
    dev_dataloader.tag2id = tag2id

    return train_dataloader, dev_dataloader


if __name__ == "__main__":
    import sys

    project_dir = Path(__file__).resolve().parents[1]
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    from config import AppConfig

    train_dataloader, dev_dataloader = build_dataloader(AppConfig())

    first_batch = next(iter(train_dataloader))
    print(
        f"train batches: {len(train_dataloader)}, "
        f"validation batches: {len(dev_dataloader)}"
    )
    print([tensor.shape for tensor in first_batch])
