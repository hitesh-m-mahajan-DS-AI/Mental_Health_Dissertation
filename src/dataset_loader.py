"""Read-only loading and auditable selection of the two frozen datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Example:
    dataset: str
    example_id: str
    source_row: int
    conversation_id: str
    utterance_id: str
    label: int
    response: str


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    examples: tuple[Example, ...]
    audit: dict[str, Any]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def _length_summary(series: pd.Series) -> dict[str, float]:
    summary = series.astype(str).str.len().describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    return {str(k): float(v) for k, v in summary.items()}


def load_motivation(path: Path) -> DatasetBundle:
    frame = pd.read_csv(path)
    required = {
        "transcript_id",
        "utterance_id",
        "therapist_response",
        "final_label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Motivation dataset missing columns: {sorted(missing)}")

    allowed = {"motivational", "non-motivational", "review"}
    found = set(frame["final_label"].dropna().astype(str).unique())
    if not found <= allowed:
        raise ValueError(f"Unexpected motivation final_label values: {sorted(found - allowed)}")
    selected = frame[frame["final_label"].isin(["motivational", "non-motivational"])]
    label_map = {"non-motivational": 0, "motivational": 1}
    examples = tuple(
        Example(
            dataset="motivation",
            example_id=f"annomi-row-{int(index)}",
            source_row=int(index),
            conversation_id=str(row.transcript_id),
            utterance_id=str(row.utterance_id),
            label=label_map[str(row.final_label)],
            response=str(row.therapist_response),
        )
        for index, row in selected.iterrows()
    )
    audit = {
        "dataset": "motivation",
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "source_rows": int(len(frame)),
        "source_columns": list(frame.columns),
        "grouping_identifier": "transcript_id",
        "utterance_identifier": "utterance_id",
        "label_field": "final_label",
        "explicit_transformations": [
            "Excluded final_label='review' from the primary binary population.",
            "Mapped final_label='motivational' to 1 and 'non-motivational' to 0.",
            "No rows were deduplicated, relabelled, or edited.",
        ],
        "final_label_counts": _plain_counts(frame["final_label"]),
        "scientific_rows": int(len(selected)),
        "scientific_label_counts": {
            str(label_map[k]): int(v)
            for k, v in selected["final_label"].value_counts().items()
        },
        "excluded_review_rows": int((frame["final_label"] == "review").sum()),
        "unique_conversations_source": int(frame["transcript_id"].nunique(dropna=True)),
        "unique_conversations_scientific": int(selected["transcript_id"].nunique(dropna=True)),
        "missing_values_by_column": {k: int(v) for k, v in frame.isna().sum().items()},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_scientific_responses": int(selected["therapist_response"].duplicated().sum()),
        "duplicate_scientific_ids": int(selected.duplicated(["transcript_id", "utterance_id"]).sum()),
        "response_character_lengths": _length_summary(selected["therapist_response"]),
    }
    return DatasetBundle("motivation", examples, audit)


def load_empathy(path: Path) -> DatasetBundle:
    frame = pd.read_csv(path)
    required = {"dialogueId", "utteranceNo", "authorRole", "utterances", "empathy"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Empathy dataset missing columns: {sorted(missing)}")
    non_null_labels = set(frame["empathy"].dropna().astype(float).unique())
    if not non_null_labels <= {0.0, 1.0}:
        raise ValueError(f"Unexpected empathy values: {sorted(non_null_labels - {0.0, 1.0})}")

    normalized_role = frame["authorRole"].fillna("").astype(str).str.strip()
    counselor = normalized_role.eq("counselor")
    valid_label = frame["empathy"].isin([0, 1])
    selected = frame[counselor & valid_label]
    examples = tuple(
        Example(
            dataset="empathy",
            example_id=f"mhlcd-row-{int(index)}",
            source_row=int(index),
            conversation_id=str(row.dialogueId),
            utterance_id=str(row.utteranceNo),
            label=int(row.empathy),
            response=str(row.utterances),
        )
        for index, row in selected.iterrows()
    )
    labeled_unknown_role = valid_label & normalized_role.eq("")
    trailing_space_roles = frame["authorRole"].fillna("").astype(str).eq("counselor ")
    audit = {
        "dataset": "empathy",
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "source_rows": int(len(frame)),
        "source_columns": list(frame.columns),
        "grouping_identifier": "dialogueId",
        "utterance_identifier": "utteranceNo",
        "label_field": "empathy",
        "response_unit": "counselor response",
        "explicit_transformations": [
            "Trimmed surrounding whitespace from authorRole for role comparison only; source values were not changed.",
            "Selected normalized authorRole='counselor' with empathy in {0,1}.",
            "Excluded labeled rows with missing authorRole rather than inferring their role.",
            "No rows were deduplicated, relabelled, or edited.",
        ],
        "author_role_counts_raw": _plain_counts(frame["authorRole"]),
        "empathy_counts_source": _plain_counts(frame["empathy"]),
        "scientific_rows": int(len(selected)),
        "scientific_label_counts": {
            str(int(k)): int(v) for k, v in selected["empathy"].value_counts().sort_index().items()
        },
        "included_trailing_space_counselor_rows": int((trailing_space_roles & valid_label).sum()),
        "excluded_labeled_missing_role_rows": int(labeled_unknown_role.sum()),
        "unique_conversations_source": int(frame["dialogueId"].nunique(dropna=True)),
        "unique_conversations_scientific": int(selected["dialogueId"].nunique(dropna=True)),
        "missing_values_by_column": {k: int(v) for k, v in frame.isna().sum().items()},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_scientific_responses": int(selected["utterances"].duplicated().sum()),
        "duplicate_scientific_ids": int(selected.duplicated(["dialogueId", "utteranceNo"]).sum()),
        "response_character_lengths": _length_summary(selected["utterances"]),
    }
    return DatasetBundle("empathy", examples, audit)


def load_both(motivation_path: Path, empathy_path: Path) -> dict[str, DatasetBundle]:
    return {
        "motivation": load_motivation(motivation_path),
        "empathy": load_empathy(empathy_path),
    }
