"""Shared validation and encoding helpers for BiLSTM NER models."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


PAD_TOKEN = "PAD"


def resolve_vocab(word2id: Mapping[str, int]) -> tuple[int, int]:
    """Return ``(vocab_size, padding_id)`` after validating a vocabulary map."""

    if PAD_TOKEN not in word2id:
        raise ValueError(f"word2id must define the padding token {PAD_TOKEN!r}.")
    try:
        token_ids = [int(index) for index in word2id.values()]
    except (TypeError, ValueError) as error:
        raise ValueError("word2id values must be integer token IDs.") from error
    if not token_ids or min(token_ids) < 0 or len(set(token_ids)) != len(token_ids):
        raise ValueError("word2id must contain unique, non-negative token IDs.")

    padding_id = int(word2id[PAD_TOKEN])
    return max(token_ids) + 1, padding_id


def resolve_tags(tag2id: Mapping[str, int]) -> dict[str, int]:
    """Validate real NER tags with contiguous zero-based IDs."""

    try:
        active_tags = {tag: int(index) for tag, index in tag2id.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("tag2id values must be integer tag IDs.") from error
    if "PAD" in active_tags:
        raise ValueError("tag2id must not define 'PAD'; label padding uses -100 internally.")
    if not active_tags:
        raise ValueError("tag2id must define at least one NER tag.")

    tag_ids = sorted(active_tags.values())
    if tag_ids != list(range(len(active_tags))):
        raise ValueError(
            "Tag IDs must be unique and contiguous from 0; "
            f"received {tag_ids}."
        )
    return active_tags


def encode_padded_sequences(
    embedding: nn.Embedding,
    lstm: nn.LSTM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode right-padded input IDs without letting padding affect the BiLSTM."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch_size, sequence_length].")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids.")
    if input_ids.size(0) == 0 or input_ids.size(1) == 0:
        raise ValueError("input_ids must contain at least one sample and one token.")

    mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    if not torch.all(mask[:, 0]):
        raise ValueError("Every input sequence must begin with a valid token.")
    if torch.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError("attention_mask must be right-padded without gaps.")

    lengths = mask.long().sum(dim=1)
    embeddings = embedding(input_ids)
    packed_embeddings = pack_padded_sequence(
        embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    packed_output, _ = lstm(packed_embeddings)
    encoded, _ = pad_packed_sequence(
        packed_output, batch_first=True, total_length=input_ids.size(1)
    )
    return encoded, mask
