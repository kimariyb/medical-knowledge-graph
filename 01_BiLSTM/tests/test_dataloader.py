"""Regression tests for leakage-free NER data preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.dataloader import PAD_TOKEN, UNK_TOKEN, _load_tag2id, build_dataloader


class GroupSplitTests(unittest.TestCase):
    def test_grouped_split_keeps_cases_together_and_vocab_uses_train_only(self) -> None:
        """A patient's documents must not leak into validation or its vocabulary."""

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            train_path = data_dir / "train.txt"
            train_path.write_text(
                "a\t0\n\n"
                "b\t0\n\n"
                "c\t0\n\n"
                "d\t0\n",
                encoding="utf-8",
            )
            metadata_path = data_dir / "sample_metadata.jsonl"
            records = [
                {"group_id": "case-1", "source": "a"},
                {"group_id": "case-1", "source": "b"},
                {"group_id": "case-2", "source": "c"},
                {"group_id": "case-2", "source": "d"},
            ]
            metadata_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            tag2id_path = data_dir / "tag2id.json"
            tag2id_path.write_text('{"O": 0}', encoding="utf-8")
            config = SimpleNamespace(
                train_path=train_path,
                sample_metadata_path=metadata_path,
                vocab_path=data_dir / "vocab.txt",
                tag2id_file=tag2id_path,
                train_ratio=0.5,
                seed=42,
                split_shuffle=True,
                batch_size=2,
                num_workers=0,
            )

            train_loader, validation_loader = build_dataloader(config)
            train_tokens = {sample[0][0] for sample in train_loader.dataset.data}
            validation_tokens = {sample[0][0] for sample in validation_loader.dataset.data}

            self.assertIn(train_tokens, ({"a", "b"}, {"c", "d"}))
            self.assertIn(validation_tokens, ({"a", "b"}, {"c", "d"}))
            self.assertTrue(train_tokens.isdisjoint(validation_tokens))
            self.assertTrue(
                all(token not in train_loader.word2id for token in validation_tokens)
            )
            self.assertEqual(train_loader.word2id[PAD_TOKEN], 0)
            self.assertEqual(train_loader.word2id[UNK_TOKEN], 1)


class TagMappingTests(unittest.TestCase):
    def test_tag_mapping_rejects_pad_label(self) -> None:
        """Label padding is internal and must not occupy an ID in tag2id.json."""

        with tempfile.TemporaryDirectory() as directory:
            tag2id_path = Path(directory) / "tag2id.json"
            tag2id_path.write_text('{"O": 0, "PAD": 1}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must not define 'PAD'"):
                _load_tag2id(tag2id_path)


if __name__ == "__main__":
    unittest.main()
