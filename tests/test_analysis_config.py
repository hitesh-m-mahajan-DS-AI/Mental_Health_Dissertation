import json
from pathlib import Path

import pytest

from src.analysis_config import AnalysisMethodology, load_analysis_methodology


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_methodology_records_fixed_probe_requirements() -> None:
    methodology = load_analysis_methodology(ROOT / "configs" / "analysis_methodology.json")
    probing = methodology.raw["linear_probing"]

    assert methodology.activation_run_id == "20260817T181608Z"
    assert probing["primary_token_aggregation"] == "final_valid_response_token"
    assert probing["secondary_token_aggregation"] == "mean_response_token_activation"
    assert probing["split_grouping"] == {
        "motivation": "transcript_id",
        "empathy": "dialogueId",
    }


def test_incomplete_methodology_is_an_explicit_execution_block() -> None:
    raw = json.loads((ROOT / "configs" / "analysis_methodology.json").read_text())
    raw["linear_probing"]["unresolved_required"]["grouped_split_seed"] = None
    methodology = AnalysisMethodology(path=Path("in-memory-incomplete.json"), raw=raw)

    with pytest.raises(ValueError, match="scientifically underspecified"):
        methodology.require_resolved("linear_probing")


def test_fully_resolved_copy_passes() -> None:
    raw = json.loads((ROOT / "configs" / "analysis_methodology.json").read_text())
    for key in raw["linear_probing"]["unresolved_required"]:
        raw["linear_probing"]["unresolved_required"][key] = "specified-for-test"
    methodology = AnalysisMethodology(path=Path("in-memory-test.json"), raw=raw)
    methodology.require_resolved("linear_probing")
