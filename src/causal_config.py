"""Strict configuration for controlled causal activation patching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import _resolve


@dataclass(frozen=True)
class CausalPatchingConfig:
    project_root: Path
    model_path: Path
    motivation_csv: Path
    empathy_csv: Path
    activation_run_id: str
    output_root: Path
    offload_dir: Path
    dtype: str
    device_map: str
    max_memory: dict[str, str]
    max_length: int
    datasets: tuple[str, ...]
    pair_construction: str
    task_styles: dict[str, dict[str, str]]
    supportive_target: str
    neutral_target: str
    prompt_template: str
    patch_activation: str
    patch_positions: str
    score_position: str
    faithfulness_equation: str
    zero_denominator_epsilon: float
    bootstrap_resamples: int
    bootstrap_seed: int
    candidate_layer_rule: str
    completeness_random_subsets: int
    minimality_threshold_fraction_full_model: float
    batch_size: int
    smoke_test_pairs_per_dataset: int


def load_causal_config(path: Path) -> CausalPatchingConfig:
    config_path = path.resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    root = _resolve(config_path.parent, raw.get("project_root", ".."))
    required_exact = {
        "datasets": ["motivation", "empathy"],
        "pair_construction": "controlled_instruction_counterfactual",
        "patch_activation": "residual_pre",
        "patch_positions": "final_valid_prompt_token",
        "score_position": "first_generated_response_position",
        "faithfulness_equation": "(patched-corrupted)/(clean-corrupted)",
    }
    for key, expected in required_exact.items():
        if raw.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    if int(raw["max_length"]) < 8:
        raise ValueError("max_length must be at least 8")
    if float(raw["zero_denominator_epsilon"]) <= 0:
        raise ValueError("zero_denominator_epsilon must be positive")
    if int(raw["bootstrap_resamples"]) < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if int(raw["completeness_random_subsets"]) < 1:
        raise ValueError("completeness_random_subsets must be positive")
    if int(raw.get("batch_size", 1)) < 1:
        raise ValueError("batch_size must be positive")
    return CausalPatchingConfig(
        project_root=root,
        model_path=_resolve(root, raw["model_path"]),
        motivation_csv=_resolve(root, raw["motivation_csv"]),
        empathy_csv=_resolve(root, raw["empathy_csv"]),
        activation_run_id=str(raw["activation_run_id"]),
        output_root=_resolve(root, raw["output_root"]),
        offload_dir=_resolve(root, raw["offload_dir"]),
        dtype=str(raw["dtype"]),
        device_map=str(raw["device_map"]),
        max_memory={str(k): str(v) for k, v in raw.get("max_memory", {}).items()},
        max_length=int(raw["max_length"]),
        datasets=tuple(raw["datasets"]),
        pair_construction=str(raw["pair_construction"]),
        task_styles={
            dataset: {key: str(value) for key, value in styles.items()}
            for dataset, styles in raw["task_styles"].items()
        },
        supportive_target=str(raw["supportive_target"]),
        neutral_target=str(raw["neutral_target"]),
        prompt_template=str(raw["prompt_template"]),
        patch_activation=str(raw["patch_activation"]),
        patch_positions=str(raw["patch_positions"]),
        score_position=str(raw["score_position"]),
        faithfulness_equation=str(raw["faithfulness_equation"]),
        zero_denominator_epsilon=float(raw["zero_denominator_epsilon"]),
        bootstrap_resamples=int(raw["bootstrap_resamples"]),
        bootstrap_seed=int(raw["bootstrap_seed"]),
        candidate_layer_rule=str(raw["candidate_layer_rule"]),
        completeness_random_subsets=int(raw["completeness_random_subsets"]),
        minimality_threshold_fraction_full_model=float(raw["minimality_threshold_fraction_full_model"]),
        batch_size=int(raw.get("batch_size", 1)),
        smoke_test_pairs_per_dataset=int(raw["smoke_test_pairs_per_dataset"]),
    )
