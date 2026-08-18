"""Build auditable same-context supportive/neutral counterfactual prompts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .causal_config import CausalPatchingConfig


@dataclass(frozen=True)
class CounterfactualPair:
    dataset: str
    selection_ordinal: int
    example_id: str
    source_row: int
    conversation_id: str
    original_label: int
    context: str
    supportive_prompt: str
    neutral_prompt: str


def _read_selection(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen activation selection not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _motivation_contexts(path: Path) -> dict[int, str]:
    frame = pd.read_csv(path)
    return {
        int(index): str(row.previous_client_utterance).strip()
        for index, row in frame.iterrows()
        if pd.notna(row.previous_client_utterance) and str(row.previous_client_utterance).strip()
    }


def _empathy_contexts(path: Path) -> dict[int, str]:
    frame = pd.read_csv(path)
    roles = frame.authorRole.fillna("").astype(str).str.strip().str.lower()
    result: dict[int, str] = {}
    last_victim: dict[str, str] = {}
    for index, row in frame.iterrows():
        dialogue = str(row.dialogueId)
        role = roles.loc[index]
        if role == "victim" and pd.notna(row.utterances):
            last_victim[dialogue] = str(row.utterances).strip()
        elif role == "counselor" and dialogue in last_victim:
            result[int(index)] = last_victim[dialogue]
    return result


def build_pairs(config: CausalPatchingConfig, dataset: str) -> tuple[CounterfactualPair, ...]:
    selection_path = (
        config.project_root / "results" / "activations" / dataset / "metadata"
        / f"selection_{config.activation_run_id}.jsonl"
    )
    records = _read_selection(selection_path)
    contexts = (
        _motivation_contexts(config.motivation_csv)
        if dataset == "motivation"
        else _empathy_contexts(config.empathy_csv)
    )
    pairs: list[CounterfactualPair] = []
    for record in records:
        source_row = int(record["source_row"])
        context = contexts.get(source_row, "")
        if not context:
            # Preserve the 200-query membership: an explicit neutral placeholder is
            # preferable to silently replacing a randomly selected observation.
            context = "The client has not provided any preceding message."
        styles = config.task_styles[dataset]
        supportive = config.prompt_template.format(style=styles["clean"], context=context)
        neutral = config.prompt_template.format(style=styles["corrupted"], context=context)
        pairs.append(CounterfactualPair(
            dataset=dataset,
            selection_ordinal=int(record["selection_ordinal"]),
            example_id=str(record["example_id"]),
            source_row=source_row,
            conversation_id=str(record["conversation_id"]),
            original_label=int(record["label"]),
            context=context,
            supportive_prompt=supportive,
            neutral_prompt=neutral,
        ))
    if len(pairs) != 200:
        raise ValueError(f"Expected exactly 200 frozen {dataset} selections, found {len(pairs)}")
    return tuple(pairs)


def validate_and_tokenize_pair(pair: CounterfactualPair, tokenizer: Any, config: CausalPatchingConfig) -> dict[str, Any]:
    clean = tokenizer(pair.supportive_prompt, add_special_tokens=True, truncation=True, max_length=config.max_length)
    corrupted = tokenizer(pair.neutral_prompt, add_special_tokens=True, truncation=True, max_length=config.max_length)
    supportive_ids = tokenizer.encode(config.supportive_target, add_special_tokens=False)
    neutral_ids = tokenizer.encode(config.neutral_target, add_special_tokens=False)
    if len(supportive_ids) != 1 or len(neutral_ids) != 1:
        raise ValueError("Supportive and neutral targets must each tokenize to exactly one token")
    if supportive_ids[0] == neutral_ids[0]:
        raise ValueError("Supportive and neutral target token IDs must differ")
    if len(clean["input_ids"]) != len(corrupted["input_ids"]):
        raise ValueError(
            f"Counterfactual token lengths differ for {pair.example_id}: "
            f"{len(clean['input_ids'])} != {len(corrupted['input_ids'])}"
        )
    differing_positions = [
        index
        for index, (clean_id, corrupted_id) in enumerate(
            zip(clean["input_ids"], corrupted["input_ids"])
        )
        if clean_id != corrupted_id
    ]
    if len(differing_positions) != 1:
        raise ValueError(
            f"Expected exactly one controlled instruction-token difference for "
            f"{pair.example_id}, found {differing_positions}"
        )
    controlled_position = differing_positions[0]
    return {
        "clean_input_ids": clean["input_ids"],
        "corrupted_input_ids": corrupted["input_ids"],
        "supportive_target_id": int(supportive_ids[0]),
        "neutral_target_id": int(neutral_ids[0]),
        "controlled_position": int(controlled_position),
        "patch_position": int(len(clean["input_ids"]) - 1),
        "clean_instruction_token_id": int(clean["input_ids"][controlled_position]),
        "corrupted_instruction_token_id": int(corrupted["input_ids"][controlled_position]),
        "truncated": len(tokenizer.encode(pair.supportive_prompt, add_special_tokens=True)) > config.max_length,
    }


def manifest_record(pair: CounterfactualPair) -> dict[str, Any]:
    record = asdict(pair)
    record["context_sha256"] = hashlib.sha256(pair.context.encode("utf-8")).hexdigest()
    record["supportive_prompt_sha256"] = hashlib.sha256(pair.supportive_prompt.encode("utf-8")).hexdigest()
    record["neutral_prompt_sha256"] = hashlib.sha256(pair.neutral_prompt.encode("utf-8")).hexdigest()
    return record
