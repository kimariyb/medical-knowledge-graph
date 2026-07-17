"""A linear-chain CRF with enforced BIO transition constraints."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

NEG_INF = -10_000.0


def reverse_mapping(mapping: Mapping[str, int]) -> dict[int, str]:
    """Reverse a tag-to-ID mapping and reject ambiguous duplicate IDs."""

    reversed_mapping: dict[int, str] = {}
    for tag, index in mapping.items():
        index = int(index)
        if index in reversed_mapping:
            raise ValueError(
                f"Duplicate tag ID {index} for {reversed_mapping[index]!r} and {tag!r}."
            )
        reversed_mapping[index] = tag
    return reversed_mapping


class LinearCRF(nn.Module):
    """Linear-chain CRF for non-empty, right-padded BIO sequences.

    ``transitions[current_tag, previous_tag]`` stores the transition score.
    START and STOP are virtual states kept in the same matrix for clarity.
    Illegal transitions are masked at every forward and decode pass, so they
    remain impossible even after optimizer updates.
    """

    def __init__(
        self,
        num_tags: int,
        id2tag: Mapping[int, str],
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if num_tags < 1:
            raise ValueError("num_tags must be positive.")

        self.num_tags = int(num_tags)
        self.id2tag = {int(index): tag for index, tag in id2tag.items()}
        expected_ids = set(range(self.num_tags))
        if set(self.id2tag) != expected_ids:
            raise ValueError(
                "id2tag must contain exactly the contiguous IDs "
                f"0 through {self.num_tags - 1}."
            )
        self.batch_first = batch_first
        self.START_TAG = self.num_tags
        self.STOP_TAG = self.num_tags + 1

        self.transitions = nn.Parameter(
            torch.empty(self.num_tags + 2, self.num_tags + 2)
        )
        self.register_buffer(
            "constraint_mask",
            torch.ones(self.num_tags + 2, self.num_tags + 2, dtype=torch.bool),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize trainable scores and rebuild the fixed transition mask."""

        nn.init.xavier_uniform_(self.transitions)
        self.apply_bio_constraint()

    def apply_bio_constraint(self) -> None:
        """Build the fixed mask for legal START/STOP and BIO transitions."""

        allowed = torch.ones_like(self.constraint_mask)
        # START cannot be entered and STOP cannot be left.
        allowed[self.START_TAG, :] = False
        allowed[:, self.STOP_TAG] = False

        for previous_id, previous_tag in self.id2tag.items():
            for current_id, current_tag in self.id2tag.items():
                if not self.valid_transition(previous_tag, current_tag):
                    allowed[current_id, previous_id] = False

        # An I-* tag is invalid at the beginning of a sequence.
        for tag_id, tag in self.id2tag.items():
            if tag.startswith("I-"):
                allowed[tag_id, self.START_TAG] = False

        self.constraint_mask.copy_(allowed)

    @staticmethod
    def valid_transition(previous: str, current: str) -> bool:
        """Return whether a BIO transition from ``previous`` to ``current`` is legal."""

        if not current.startswith("I-"):
            return True
        entity_type = current[2:]
        return previous in {f"B-{entity_type}", f"I-{entity_type}"}

    def _constrained_transitions(self) -> torch.Tensor:
        """Return transition scores with illegal moves permanently disabled."""

        return self.transitions.masked_fill(~self.constraint_mask, NEG_INF)

    def _prepare_inputs(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if emissions.ndim != 3 or emissions.size(-1) != self.num_tags:
            raise ValueError(
                "emissions must have shape [batch, sequence, num_tags] (or "
                "[sequence, batch, num_tags]) with the configured num_tags."
            )

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if tags is not None:
                if tags.ndim != 2:
                    raise ValueError("tags must be a two-dimensional tensor.")
                tags = tags.transpose(0, 1)
            if mask is not None:
                if mask.ndim != 2:
                    raise ValueError("mask must be a two-dimensional tensor.")
                mask = mask.transpose(0, 1)
        elif tags is not None and tags.ndim != 2:
            raise ValueError("tags must be a two-dimensional tensor.")
        elif mask is not None and mask.ndim != 2:
            raise ValueError("mask must be a two-dimensional tensor.")

        sequence_length, batch_size, _ = emissions.shape
        if sequence_length == 0 or batch_size == 0:
            raise ValueError("CRF emissions must contain at least one token and one sample.")
        if tags is not None and tags.shape != (sequence_length, batch_size):
            raise ValueError("tags shape must match the first two emission dimensions.")

        if mask is None:
            mask = torch.ones(
                sequence_length, batch_size, dtype=torch.bool, device=emissions.device
            )
        else:
            if mask.shape != (sequence_length, batch_size):
                raise ValueError("mask shape must match the first two emission dimensions.")
            mask = mask.to(device=emissions.device, dtype=torch.bool)

        if not torch.all(mask[0]):
            raise ValueError("Every sequence must begin with a valid token.")
        if torch.any(mask[1:] & ~mask[:-1]):
            raise ValueError("CRF masks must be right-padded without gaps.")
        return emissions, tags, mask

    def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the log partition function for seq-first emissions."""

        transitions = self._constrained_transitions()
        alpha = transitions[: self.num_tags, self.START_TAG].unsqueeze(0) + emissions[0]
        # ``previous_to_current[previous, current]`` matches alpha's layout.
        previous_to_current = transitions[: self.num_tags, : self.num_tags].transpose(0, 1)

        for timestep in range(1, emissions.size(0)):
            scores = alpha.unsqueeze(2) + previous_to_current.unsqueeze(0)
            next_alpha = torch.logsumexp(scores, dim=1) + emissions[timestep]
            alpha = torch.where(mask[timestep].unsqueeze(1), next_alpha, alpha)

        alpha = alpha + transitions[self.STOP_TAG, : self.num_tags].unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def _score_gold_path(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the score of each gold BIO path for seq-first tensors."""

        valid_tags = tags[mask]
        if torch.any(valid_tags < 0) or torch.any(valid_tags >= self.num_tags):
            raise ValueError("Gold labels at valid positions must be in [0, num_tags).")
        safe_tags = tags.masked_fill(~mask, 0)
        transitions = self._constrained_transitions()

        emission_scores = emissions.gather(2, safe_tags.unsqueeze(-1)).squeeze(-1)
        emission_scores = emission_scores.masked_fill(~mask, 0.0).sum(dim=0)

        transition_scores = transitions[safe_tags[1:], safe_tags[:-1]]
        transition_scores = transition_scores.masked_fill(~mask[1:], 0.0).sum(dim=0)

        start_scores = transitions[safe_tags[0], self.START_TAG]
        lengths = mask.long().sum(dim=0)
        batch_indices = torch.arange(tags.size(1), device=tags.device)
        last_tags = safe_tags[lengths - 1, batch_indices]
        end_scores = transitions[self.STOP_TAG, last_tags]
        return start_scores + emission_scores + transition_scores + end_scores

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Return CRF negative log likelihood for padded batch data."""

        emissions, tags, mask = self._prepare_inputs(emissions, tags, mask)
        assert tags is not None
        losses = self._forward_alg(emissions, mask) - self._score_gold_path(
            emissions, tags, mask
        )
        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        if reduction == "mean":
            return losses.mean()
        raise ValueError("reduction must be one of: 'none', 'sum', 'mean'.")

    @torch.no_grad()
    def viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor | None = None
    ) -> list[list[int]]:
        """Decode the highest-scoring valid BIO path for every sequence."""

        emissions, _, mask = self._prepare_inputs(emissions, None, mask)
        transitions = self._constrained_transitions()
        score = transitions[: self.num_tags, self.START_TAG].unsqueeze(0) + emissions[0]
        previous_to_current = transitions[: self.num_tags, : self.num_tags].transpose(0, 1)
        history: list[torch.Tensor] = []

        for timestep in range(1, emissions.size(0)):
            candidate_scores = score.unsqueeze(2) + previous_to_current.unsqueeze(0)
            best_scores, best_previous_tags = candidate_scores.max(dim=1)
            best_scores = best_scores + emissions[timestep]
            score = torch.where(mask[timestep].unsqueeze(1), best_scores, score)
            history.append(best_previous_tags)

        score = score + transitions[self.STOP_TAG, : self.num_tags].unsqueeze(0)
        best_last_tags = score.argmax(dim=1)
        lengths = mask.long().sum(dim=0)
        paths: list[list[int]] = []

        for batch_index, length in enumerate(lengths.tolist()):
            path = [best_last_tags[batch_index].item()]
            for backpointers in reversed(history[: length - 1]):
                path.append(backpointers[batch_index, path[-1]].item())
            paths.append(list(reversed(path)))
        return paths

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for mean negative log likelihood, suitable for training loops."""

        return self.neg_log_likelihood(emissions, tags, mask)
