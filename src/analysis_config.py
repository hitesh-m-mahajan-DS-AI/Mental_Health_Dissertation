"""Validation for downstream scientific-method configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalysisMethodology:
    path: Path
    raw: dict[str, Any]

    @property
    def activation_run_id(self) -> str:
        return str(self.raw["activation_run_id"])

    def unresolved(self, stage: str) -> tuple[str, ...]:
        stage_config = self.raw[stage]
        required = stage_config.get("unresolved_required", {})
        return tuple(name for name, value in required.items() if value is None)

    def require_resolved(self, stage: str) -> None:
        missing = self.unresolved(stage)
        if missing:
            rendered = "\n  - ".join(missing)
            raise ValueError(
                f"{stage} is scientifically underspecified. Resolve these fields in "
                f"{self.path}:\n  - {rendered}"
            )


def load_analysis_methodology(path: Path) -> AnalysisMethodology:
    resolved = path.resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))

    if raw.get("tasks") != ["motivation", "empathy"]:
        raise ValueError("tasks must be the independent ordered tasks motivation and empathy")
    if raw.get("keep_tasks_independent") is not True:
        raise ValueError("keep_tasks_independent must be true")

    probing = raw.get("linear_probing", {})
    if probing.get("sequence_length") != 128:
        raise ValueError("linear_probing.sequence_length must be 128")
    if probing.get("activation_family") != "residual_stream":
        raise ValueError("linear_probing.activation_family must be residual_stream")
    if probing.get("split_ratios") != {"train": 0.7, "validation": 0.15, "test": 0.15}:
        raise ValueError("linear_probing split ratios must be 70/15/15")
    if probing.get("classifier") != "logistic_regression":
        raise ValueError("linear_probing.classifier must be logistic_regression")
    if probing.get("metrics") != ["accuracy", "macro_f1"]:
        raise ValueError("linear_probing metrics must be accuracy and macro_f1")

    return AnalysisMethodology(path=resolved, raw=raw)
