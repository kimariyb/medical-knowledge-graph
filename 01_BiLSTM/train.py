"""Train and evaluate the BiLSTM and BiLSTM-CRF NER models."""

from __future__ import annotations

import argparse
import csv
import random
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.optim import AdamW

from config import AppConfig
from model import NERBiLSTM, NERBiLSTM_CRF
from utils.dataloader import build_dataloader


MODELS: dict[str, type[nn.Module]] = {
    "BiLSTM": NERBiLSTM,
    "BiLSTM_CRF": NERBiLSTM_CRF,
}
MODEL_ALIASES = {
    "bilstm": "BiLSTM",
    "bilstm-crf": "BiLSTM_CRF",
    "bilstm_crf": "BiLSTM_CRF",
}
CSV_FIELDS = ["timestamp", "model", "split", "epoch", "loss", "precision", "recall", "f1"]


class MetricsCSV:
    """Write one training row and one validation row for every epoch."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.file.flush()

    def write(
        self,
        model_name: str,
        split: str,
        epoch: int,
        metrics: Mapping[str, Any],
    ) -> None:
        self.writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model_name,
                "split": split,
                "epoch": epoch,
                "loss": f"{metrics['loss']:.6f}",
                "precision": f"{metrics['precision']:.6f}",
                "recall": f"{metrics['recall']:.6f}",
                "f1": f"{metrics['f1']:.6f}",
            }
        )
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "MetricsCSV":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class EarlyStopping:
    """Stop training after validation F1 stops improving by ``min_delta``."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("early stopping patience must be at least 1.")
        if min_delta < 0:
            raise ValueError("early stopping min_delta must be non-negative.")
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = float("-inf")
        self.epochs_without_improvement = 0
        self.last_improved = False

    def step(self, f1: float) -> bool:
        """Update state and return whether training should stop."""

        self.last_improved = f1 > self.best_f1 + self.min_delta
        if self.last_improved:
            self.best_f1 = f1
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    """Use the requested device when available, otherwise use CPU."""

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return device


def select_models(selection: str) -> list[str]:
    if selection.lower() == "all":
        return list(MODELS)
    model_name = MODEL_ALIASES.get(selection.lower(), selection)
    if model_name not in MODELS:
        raise ValueError("model must be one of: all, bilstm, bilstm-crf")
    return [model_name]


def active_labels(tag2id: Mapping[str, int]) -> tuple[list[int], list[str]]:
    """Return real label IDs and names in the order used for sklearn metrics."""

    labels = sorted((int(index), tag) for tag, index in tag2id.items())
    return [index for index, _ in labels], [tag for _, tag in labels]


def calculate_metrics(
    gold: Sequence[int],
    predicted: Sequence[int],
    label_ids: Sequence[int],
    label_names: Sequence[str],
) -> dict[str, Any]:
    """Calculate macro precision, recall, F1, and a printable sklearn report."""

    if not gold:
        raise ValueError("Cannot calculate metrics for an empty label sequence.")
    options = {"labels": label_ids, "average": "macro", "zero_division": 0}
    return {
        "precision": precision_score(gold, predicted, **options),
        "recall": recall_score(gold, predicted, **options),
        "f1": f1_score(gold, predicted, **options),
        "report": classification_report(
            gold,
            predicted,
            labels=label_ids,
            target_names=label_names,
            digits=4,
            zero_division=0,
        ),
    }


def loss_and_predictions(
    model: nn.Module,
    model_name: str,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | list[list[int]]]:
    """Calculate one batch loss and its predictions without padding tokens."""

    if model_name == "BiLSTM":
        assert isinstance(model, NERBiLSTM)
        logits = model(input_ids, attention_mask)
        loss = functional.cross_entropy(logits[attention_mask.bool()], labels[attention_mask.bool()])
        return loss, logits.argmax(dim=-1)

    assert isinstance(model, NERBiLSTM_CRF)
    emissions = model.emissions(input_ids, attention_mask)
    loss = model.crf(emissions, labels, attention_mask)
    return loss, model.crf.viterbi_decode(emissions, attention_mask)


def calculate_loss(
    model: nn.Module,
    model_name: str,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate a training loss, optionally rebalancing labels from train data."""

    if model_name == "BiLSTM":
        assert isinstance(model, NERBiLSTM)
        logits = model(input_ids, attention_mask)
        return functional.cross_entropy(
            logits[attention_mask.bool()],
            labels[attention_mask.bool()],
            weight=class_weights,
        )

    assert isinstance(model, NERBiLSTM_CRF)
    if class_weights is None:
        return model(input_ids, labels, attention_mask)

    emissions = model.emissions(input_ids, attention_mask)
    sequence_losses = model.crf.neg_log_likelihood(
        emissions, labels, attention_mask, reduction="none"
    )
    safe_labels = labels.masked_fill(~attention_mask.bool(), 0)
    token_weights = class_weights[safe_labels]
    sequence_weights = (token_weights * attention_mask).sum(dim=1)
    sequence_weights = sequence_weights / attention_mask.sum(dim=1).clamp_min(1)
    return (sequence_losses * sequence_weights).sum() / sequence_weights.sum()


def flatten_predictions(
    predictions: torch.Tensor | list[list[int]],
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[list[int], list[int]]:
    """Flatten predictions and labels after removing padded positions."""

    lengths = attention_mask.long().sum(dim=1).tolist()
    gold = [
        label_row[:length].detach().cpu().tolist()
        for label_row, length in zip(labels, lengths)
    ]
    if isinstance(predictions, torch.Tensor):
        predicted = [
            prediction_row[:length].detach().cpu().tolist()
            for prediction_row, length in zip(predictions, lengths)
        ]
    else:
        predicted = predictions

    if [len(row) for row in predicted] != lengths:
        raise RuntimeError("Prediction lengths do not match the attention mask.")
    return [tag for row in gold for tag in row], [tag for row in predicted for tag in row]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_name: str,
    data_loader: Any,
    device: torch.device,
    label_ids: Sequence[int],
    label_names: Sequence[str],
) -> dict[str, Any]:
    """Evaluate all batches once and return aggregate sklearn metrics."""

    model.eval()
    losses: list[float] = []
    gold: list[int] = []
    predicted: list[int] = []
    for input_ids, labels, attention_mask in data_loader:
        input_ids, labels, attention_mask = (
            input_ids.to(device),
            labels.to(device),
            attention_mask.to(device),
        )
        loss, predictions = loss_and_predictions(
            model, model_name, input_ids, labels, attention_mask
        )
        batch_gold, batch_predicted = flatten_predictions(
            predictions, labels, attention_mask
        )
        losses.append(loss.item())
        gold.extend(batch_gold)
        predicted.extend(batch_predicted)

    if not losses:
        raise ValueError("Data loader yielded no batches.")
    metrics = calculate_metrics(gold, predicted, label_ids, label_names)
    metrics["loss"] = sum(losses) / len(losses)
    return metrics


def print_header() -> None:
    print("\nMODEL        SPLIT        EPOCH        LOSS   PRECISION      RECALL          F1")
    print("-" * 82)


def print_metrics(
    model_name: str,
    split: str,
    epoch: int,
    epochs: int,
    metrics: Mapping[str, Any],
) -> None:
    epoch_text = f"{epoch}/{epochs}"
    print(
        f"{model_name:<12} {split:<12} {epoch_text:>5} "
        f"{metrics['loss']:>11.4f} {metrics['precision']:>11.4f} "
        f"{metrics['recall']:>11.4f} {metrics['f1']:>11.4f}"
    )


def save_checkpoint(
    model: nn.Module,
    model_name: str,
    checkpoint: Path,
    word2id: Mapping[str, int],
    tag2id: Mapping[str, int],
    config: Any,
    metrics: Mapping[str, Any],
    epoch: int,
    class_weights: torch.Tensor | None,
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "word2id": dict(word2id),
            "tag2id": dict(tag2id),
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "epoch": epoch,
            "class_weights": None if class_weights is None else class_weights.cpu(),
            "validation_metrics": dict(metrics),
        },
        checkpoint,
    )


def train_model(
    model_name: str,
    train_loader: Any,
    validation_loader: Any,
    config: Any,
    csv_writer: MetricsCSV,
) -> dict[str, Any]:
    """Train one model; write train and validation metrics once per epoch."""

    device = get_device(config.device)
    word2id, tag2id = train_loader.word2id, train_loader.tag2id
    label_ids, label_names = active_labels(tag2id)
    model = MODELS[model_name](
        config.embedding_dim, config.hidden_dim, config.dropout, word2id, tag2id
    ).to(device)
    optimizer = AdamW(
        model.parameters(), config.learning_rate, weight_decay=config.weight_decay
    )
    checkpoint = Path(config.checkpoint_dir) / f"{model_name.lower()}_best.pt"
    best_metrics: dict[str, Any] | None = None
    best_epoch: int | None = None
    class_weights = getattr(train_loader, "class_weights", None)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    early_stopping = EarlyStopping(
        int(getattr(config, "early_stopping_patience", 3)),
        float(getattr(config, "early_stopping_min_delta", 1e-4)),
    )
    for epoch in range(1, config.epochs + 1):
        model.train()
        for input_ids, labels, attention_mask in train_loader:
            input_ids, labels, attention_mask = (
                input_ids.to(device),
                labels.to(device),
                attention_mask.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = calculate_loss(
                model, model_name, input_ids, labels, attention_mask, class_weights
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

        train_metrics = evaluate(
            model, model_name, train_loader, device, label_ids, label_names
        )
        csv_writer.write(model_name, "train", epoch, train_metrics)
        print_metrics(model_name, "train", epoch, config.epochs, train_metrics)

        validation_metrics = evaluate(
            model, model_name, validation_loader, device, label_ids, label_names
        )
        csv_writer.write(model_name, "validation", epoch, validation_metrics)
        print_metrics(
            model_name, "validation", epoch, config.epochs, validation_metrics
        )
        print(
            f"\n{model_name} validation classification report (epoch {epoch}):\n"
            f"{validation_metrics['report']}"
        )

        should_stop = early_stopping.step(validation_metrics["f1"])
        if early_stopping.last_improved:
            best_metrics = validation_metrics
            best_epoch = epoch
            save_checkpoint(
                model,
                model_name,
                checkpoint,
                word2id,
                tag2id,
                config,
                best_metrics,
                epoch,
                class_weights,
            )
        if should_stop:
            print(
                f"{model_name} early stopping at epoch {epoch}: no validation F1 "
                f"improvement of at least {early_stopping.min_delta:.4g} for "
                f"{early_stopping.patience} epochs."
            )
            break

    assert best_metrics is not None and best_epoch is not None
    return {"checkpoint": checkpoint, "metrics": best_metrics, "epoch": best_epoch}


def run_training(config: Any, selection: str) -> dict[str, dict[str, Any]]:
    """Train each selected model with the shared data split and metrics CSV."""

    set_seed(config.seed)
    train_loader, validation_loader = build_dataloader(config)
    if not len(validation_loader.dataset):
        raise ValueError("Validation split is empty; set train_ratio below 1.0.")

    results: dict[str, dict[str, Any]] = {}
    print_header()
    with MetricsCSV(config.metrics_csv_path) as csv_writer:
        for offset, model_name in enumerate(select_models(selection)):
            set_seed(config.seed + offset)
            results[model_name] = train_model(
                model_name, train_loader, validation_loader, config, csv_writer
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="all, bilstm, or bilstm-crf")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.device is not None:
        config.device = args.device
    if args.metrics_csv is not None:
        config.metrics_csv_path = args.metrics_csv

    results = run_training(config, args.model or config.model)
    for model_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{model_name}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"checkpoint={result['checkpoint']}"
        )


if __name__ == "__main__":
    main()


"""
MODEL        SPLIT        EPOCH        LOSS   PRECISION      RECALL          F1
----------------------------------------------------------------------------------
BiLSTM       train        10/10      0.1166      0.8834      0.9720      0.9241
BiLSTM       validation   10/10      0.1406      0.8646      0.9419      0.8999

BiLSTM validation classification report (epoch 10):
              precision    recall  f1-score   support

           O     0.9880    0.9610    0.9743     33930
 B-TREATMENT     0.7256    0.8894    0.7992       217
 I-TREATMENT     0.8183    0.9041    0.8590      1001
      B-BODY     0.9154    0.9311    0.9232      2104
      I-BODY     0.8951    0.9453    0.9195      3638
     B-SIGNS     0.9334    0.9844    0.9582      1537
     I-SIGNS     0.9313    0.9828    0.9564      1684
     B-CHECK     0.9273    0.9785    0.9522      1812
     I-CHECK     0.9433    0.9663    0.9546      4062
   B-DISEASE     0.6971    0.9173    0.7922       133
   I-DISEASE     0.7362    0.9010    0.8103       505

    accuracy                         0.9590     50623
   macro avg     0.8646    0.9419    0.8999     50623
weighted avg     0.9612    0.9590    0.9597     50623

BiLSTM: precision=0.8646, recall=0.9419, f1=0.8999, checkpoint=/Volumes/KimariYB Disk/Project/medical-knowledge-graph/01_BiLSTM/checkpoints/bilstm_best.pt


MODEL        SPLIT        EPOCH        LOSS   PRECISION      RECALL          F1
----------------------------------------------------------------------------------
BiLSTM_CRF   train        10/10      8.3818      0.9510      0.9336      0.9416
BiLSTM_CRF   validation   10/10     10.8831      0.9231      0.9074      0.9147

BiLSTM_CRF validation classification report (epoch 10):
              precision    recall  f1-score   support

           O     0.9803    0.9786    0.9795     33930
 B-TREATMENT     0.9016    0.8018    0.8488       217
 I-TREATMENT     0.8796    0.8761    0.8779      1001
      B-BODY     0.9267    0.9140    0.9203      2104
      I-BODY     0.8976    0.9393    0.9179      3638
     B-SIGNS     0.9714    0.9714    0.9714      1537
     I-SIGNS     0.9692    0.9733    0.9713      1684
     B-CHECK     0.9641    0.9641    0.9641      1812
     I-CHECK     0.9584    0.9481    0.9532      4062
   B-DISEASE     0.8917    0.8045    0.8458       133
   I-DISEASE     0.8131    0.8099    0.8115       505

    accuracy                         0.9648     50623
   macro avg     0.9231    0.9074    0.9147     50623
weighted avg     0.9650    0.9648    0.9648     50623

BiLSTM_CRF: precision=0.9231, recall=0.9074, f1=0.9147, checkpoint=/Volumes/KimariYB Disk/Project/medical-knowledge-graph/01_BiLSTM/checkpoints/bilstm_crf_best.pt
"""