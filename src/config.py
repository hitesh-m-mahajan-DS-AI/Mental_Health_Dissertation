"""Strict configuration loading for activation capture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaptureConfig:
    project_root: Path
    model_path: Path
    motivation_csv: Path
    empathy_csv: Path
    output_root: Path
    offload_dir: Path
    seed: int
    dtype: str
    device_map: str
    max_memory: dict[str, str]
    max_length: int
    max_examples_per_dataset: int | None
    token_positions: str
    add_special_tokens: bool
    input_format: str
    capture_types: tuple[str, ...]
    dataset_order: tuple[str, ...]
    smoke_test: bool


_CAPTURE_TYPES = {
    "residual_pre",
    "residual_post",
    "query",
    "key",
    "value",
    "head_output",
    "attention_output",
    "mlp_activation",
    "mlp_output",
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: Path) -> CaptureConfig:
    config_path = path.resolve()
    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    root = _resolve(config_path.parent, raw.get("project_root", ".."))

    capture_types = tuple(raw["capture_types"])
    unknown = set(capture_types) - _CAPTURE_TYPES
    if unknown:
        raise ValueError(f"Unknown capture_types: {sorted(unknown)}")
    if raw["token_positions"] not in {"all", "last_non_padding"}:
        raise ValueError("token_positions must be 'all' or 'last_non_padding'")
    if raw["input_format"] != "raw_response":
        raise ValueError("Only the explicitly implemented input_format 'raw_response' is accepted")
    order = tuple(raw.get("dataset_order", ["motivation", "empathy"]))
    if set(order) != {"motivation", "empathy"} or len(order) != 2:
        raise ValueError("dataset_order must contain motivation and empathy exactly once")

    max_examples = raw.get("max_examples_per_dataset")
    if max_examples is not None and (not isinstance(max_examples, int) or max_examples < 1):
        raise ValueError("max_examples_per_dataset must be null or a positive integer")

    return CaptureConfig(
        project_root=root,
        model_path=_resolve(root, raw["model_path"]),
        motivation_csv=_resolve(root, raw["motivation_csv"]),
        empathy_csv=_resolve(root, raw["empathy_csv"]),
        output_root=_resolve(root, raw["output_root"]),
        offload_dir=_resolve(root, raw["offload_dir"]),
        seed=int(raw["seed"]),
        dtype=str(raw["dtype"]),
        device_map=str(raw["device_map"]),
        max_memory={str(k): str(v) for k, v in raw.get("max_memory", {}).items()},
        max_length=int(raw["max_length"]),
        max_examples_per_dataset=max_examples,
        token_positions=str(raw["token_positions"]),
        add_special_tokens=bool(raw["add_special_tokens"]),
        input_format=str(raw["input_format"]),
        capture_types=capture_types,
        dataset_order=order,
        smoke_test=bool(raw.get("smoke_test", False)),
    )
