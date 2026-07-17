"""Utilities for converting the annotated corpus to BIO training data."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class BIOTransform:
    """Convert character-offset annotations to one BIO-tagged character per line.

    ``origin_path`` is expected to contain pairs named ``<name>.txt`` (the
    annotations) and ``<name>.txtoriginal.txt`` (the source text).  The output
    uses the integer IDs defined in ``tag2id_file`` and separates documents by
    a blank line.
    """

    _TEXT_FILE_SUFFIX = "original.txt"
    _CASE_ID_PATTERN = re.compile(r"-(?P<case_id>\d+)\.txtoriginal\.txt$")

    def __init__(self, config: Any):
        self.label_dict = self._load_mapping(config.labels_file, "labels_file")
        self.seq_tag_dict = self._load_mapping(config.tag2id_file, "tag2id_file")
        self.origin_path = Path(config.origin_path)
        self.processed_path = Path(config.train_path)
        self.sample_metadata_path = Path(
            getattr(
                config,
                "sample_metadata_path",
                self.processed_path.with_name("sample_metadata.jsonl"),
            )
        )

        if "O" not in self.seq_tag_dict:
            raise ValueError("tag2id_file must define the outside tag 'O'.")
        if "PAD" in self.seq_tag_dict:
            raise ValueError("tag2id_file must not define 'PAD'; label padding uses -100 internally.")

    @staticmethod
    def _load_mapping(source: Any, source_name: str) -> dict[str, Any]:
        """Load a JSON object from a path, an open file, or an in-memory map."""

        if isinstance(source, Mapping):
            mapping = dict(source)
        elif hasattr(source, "read"):
            mapping = json.load(source)
        else:
            try:
                with Path(source).open("r", encoding="utf-8") as file:
                    mapping = json.load(file)
            except (OSError, TypeError) as error:
                raise ValueError(
                    f"{source_name} must be a JSON file path, open file, or mapping."
                ) from error

        if not isinstance(mapping, Mapping):
            raise ValueError(f"{source_name} must contain a JSON object.")
        return dict(mapping)

    def transform(self) -> int:
        """Write the processed corpus and return the number of documents written."""

        if not self.origin_path.is_dir():
            raise FileNotFoundError(
                f"The origin data directory does not exist: {self.origin_path}"
            )

        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        document_count = 0

        with (
            self.processed_path.open("w", encoding="utf-8", newline="\n") as output_file,
            self.sample_metadata_path.open("w", encoding="utf-8", newline="\n") as metadata_file,
        ):
            for root, directories, files in os.walk(self.origin_path):
                directories.sort()
                for file_name in sorted(files):
                    if not file_name.endswith(self._TEXT_FILE_SUFFIX):
                        continue

                    text_path = Path(root, file_name)
                    label_path = self._label_path_for(text_path)
                    if not label_path.is_file():
                        raise FileNotFoundError(
                            f"Missing annotation file for {text_path}: {label_path}"
                        )

                    tags_by_index = self.label_dict_text(label_path)
                    # Keep leading/trailing spaces because the annotation offsets
                    # are based on the original text; only remove file terminators.
                    text = text_path.read_text(encoding="utf-8").rstrip("\r\n")
                    self._validate_spans(tags_by_index, text, label_path)

                    for index, character in enumerate(text):
                        tag = tags_by_index.get(index, "O")
                        output_file.write(f"{character}\t{self.seq_tag_dict[tag]}\n")
                    output_file.write("\n")
                    metadata_file.write(
                        json.dumps(
                            {
                                "group_id": self._group_id_for(text_path),
                                "source": str(text_path.relative_to(self.origin_path)),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    document_count += 1

        return document_count

    @classmethod
    def _label_path_for(cls, text_path: Path) -> Path:
        """Return the annotation-file path paired with ``text_path``."""

        if not text_path.name.endswith(cls._TEXT_FILE_SUFFIX):
            raise ValueError(f"Not an original-text file: {text_path}")
        return text_path.with_name(text_path.name[: -len(cls._TEXT_FILE_SUFFIX)])

    @classmethod
    def _group_id_for(cls, text_path: Path) -> str:
        """Extract the shared case number from a source file name."""

        match = cls._CASE_ID_PATTERN.search(text_path.name)
        if match is None:
            raise ValueError(
                f"Cannot extract case ID from {text_path.name!r}; expected a name "
                "ending in '-<case_id>.txtoriginal.txt'."
            )
        return match.group("case_id")

    def label_dict_text(self, label_file_path: str | Path) -> dict[int, str]:
        """Return the BIO tag for each character index in an annotation file."""

        tags_by_index: dict[int, str] = {}
        annotation_path = Path(label_file_path)

        with annotation_path.open("r", encoding="utf-8") as label_file:
            for line_number, raw_line in enumerate(label_file, start=1):
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue

                fields = line.split("\t")
                if len(fields) < 4:
                    raise ValueError(
                        f"Malformed annotation at {annotation_path}:{line_number}; "
                        "expected text, start, end, and label separated by tabs."
                    )

                _entity_text, start_text, end_text, label = fields[:4]
                try:
                    start, end = int(start_text), int(end_text)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid offsets at {annotation_path}:{line_number}: "
                        f"{start_text!r}, {end_text!r}"
                    ) from error
                if start < 0 or end < start:
                    raise ValueError(
                        f"Invalid offset range at {annotation_path}:{line_number}: "
                        f"{start}-{end}"
                    )

                label_tag = self.label_dict.get(label)
                if label_tag is None:
                    raise KeyError(
                        f"Unknown label {label!r} at {annotation_path}:{line_number}."
                    )

                for index in range(start, end + 1):
                    tag = f"B-{label_tag}" if index == start else f"I-{label_tag}"
                    if tag not in self.seq_tag_dict:
                        raise KeyError(
                            f"Tag {tag!r} from {annotation_path}:{line_number} "
                            "is missing from tag2id_file."
                        )
                    if index in tags_by_index:
                        raise ValueError(
                            f"Overlapping annotations at {annotation_path}:{line_number} "
                            f"for character offset {index}."
                        )
                    tags_by_index[index] = tag

        return tags_by_index

    @staticmethod
    def _validate_spans(
        tags_by_index: Mapping[int, str], text: str, label_path: Path
    ) -> None:
        """Ensure every annotation offset fits within its paired source text."""

        if not tags_by_index:
            return
        largest_index = max(tags_by_index)
        if largest_index >= len(text):
            raise ValueError(
                f"Annotation offset {largest_index} in {label_path} exceeds the "
                f"paired text length ({len(text)})."
            )


if __name__ == "__main__":
    # Allow both ``python utils/process.py`` from the project directory and
    # ``python 01_BiLSTM/utils/process.py`` from the repository root.
    import sys

    project_dir = Path(__file__).resolve().parents[1]
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    from config import AppConfig

    BIOTransform(AppConfig()).transform()
