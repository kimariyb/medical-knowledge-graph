"""Tests for loading checkpoints and decoding NER predictions."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import torch

from model import NERBiLSTM


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.inference = importlib.import_module("inference")
        except ImportError as error:
            self.fail(f"inference module must be importable: {error}")

    def test_loads_checkpoint_and_returns_one_bio_tag_per_character(self) -> None:
        """Inference uses the mappings and architecture stored in its checkpoint."""

        word2id = {"PAD": 0, "UNK": 1, "甲": 2}
        tag2id = {"O": 0, "B-DISEASE": 1, "I-DISEASE": 2}
        source_model = NERBiLSTM(4, 4, 0.0, word2id, tag2id)
        checkpoint = {
            "model_name": "BiLSTM",
            "state_dict": source_model.state_dict(),
            "word2id": word2id,
            "tag2id": tag2id,
            "embedding_dim": 4,
            "hidden_dim": 4,
            "dropout": 0.0,
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pt"
            torch.save(checkpoint, checkpoint_path)
            model, metadata = self.inference.load_model(checkpoint_path, torch.device("cpu"))
            result = self.inference.predict_text(model, metadata, "甲乙", torch.device("cpu"))

        self.assertEqual(result["text"], "甲乙")
        self.assertEqual(len(result["tags"]), 2)
        self.assertEqual([item["index"] for item in result["tags"]], [0, 1])
        self.assertTrue({item["tag"] for item in result["tags"]} <= set(tag2id))

    def test_extract_entities_recovers_invalid_i_tag_as_a_new_entity(self) -> None:
        """Entity output remains usable even if a non-CRF model emits an invalid I-tag."""

        entities = self.inference.extract_entities(
            "甲乙丙丁",
            ["B-DISEASE", "I-DISEASE", "O", "I-BODY"],
        )

        self.assertEqual(
            entities,
            [
                {"text": "甲乙", "start": 0, "end": 2, "label": "DISEASE"},
                {"text": "丁", "start": 3, "end": 4, "label": "BODY"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
