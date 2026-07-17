"""BiLSTM encoder and softmax classifier for character-level NER."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import encode_padded_sequences, resolve_tags, resolve_vocab


class NERBiLSTM(nn.Module):
    """Bidirectional LSTM tagger compatible with the project's data loader."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        word2id: Mapping[str, int],
        tag2id: Mapping[str, int],
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        if hidden_dim < 2 or hidden_dim % 2:
            raise ValueError("hidden_dim must be a positive even integer for a BiLSTM.")

        self.name = "BiLSTM"
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.vocab_size, self.padding_idx = resolve_vocab(word2id)
        self.tag2id = resolve_tags(tag2id)
        self.classes = len(self.tag2id)

        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx
        )
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Linear(self.hidden_dim, self.classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return per-token logits with shape ``[batch, sequence, num_tags]``."""

        encoded, mask = encode_padded_sequences(
            self.embedding, self.lstm, input_ids, attention_mask
        )
        logits = self.proj(self.dropout(encoded))
        return logits.masked_fill(~mask.unsqueeze(-1), 0.0)

    def loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Calculate masked cross-entropy loss for a batch from ``build_dataloader``."""

        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids.")

        logits = self(input_ids, attention_mask)
        targets = labels.masked_fill(~attention_mask.bool(), ignore_index)
        valid_targets = targets[targets != ignore_index]

        if torch.any(valid_targets < 0) or torch.any(valid_targets >= self.classes):
            raise ValueError("Labels at valid positions must be valid non-padding tag IDs.")

        return F.cross_entropy(
            logits.reshape(-1, self.classes),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )
