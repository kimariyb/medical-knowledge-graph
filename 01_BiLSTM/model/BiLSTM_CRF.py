"""BiLSTM-CRF model for character-level BIO named-entity recognition."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from .TorchCRF import LinearCRF, reverse_mapping
from .common import encode_padded_sequences, resolve_tags, resolve_vocab


class NERBiLSTM_CRF(nn.Module):
    """A packed BiLSTM encoder followed by a BIO-constrained CRF decoder."""

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

        self.name = "BiLSTM_CRF"
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.vocab_size, self.padding_idx = resolve_vocab(word2id)
        self.tag2id = resolve_tags(tag2id)
        self.classes = len(self.tag2id)
        self.id2tag = reverse_mapping(self.tag2id)

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
        self.crf = LinearCRF(self.classes, self.id2tag, batch_first=True)

    def emissions(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return masked BiLSTM emission scores for the CRF layer."""

        encoded, mask = encode_padded_sequences(
            self.embedding, self.lstm, input_ids, attention_mask
        )
        emissions = self.proj(self.dropout(encoded))
        return emissions.masked_fill(~mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean CRF negative log likelihood for one training batch."""

        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids.")
        return self.crf(self.emissions(input_ids, attention_mask), labels, attention_mask)

    @torch.no_grad()
    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[list[int]]:
        """Decode valid BIO tag-ID sequences without padded positions."""

        return self.crf.viterbi_decode(self.emissions(input_ids, attention_mask), attention_mask)
