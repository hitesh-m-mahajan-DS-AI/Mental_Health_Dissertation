import json
from pathlib import Path

import pytest

from src.circuit_atlas import build_circuit_atlas, write_circuit_atlas


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _probe() -> dict:
    entry = {
        "significant_depths_vs_shuffled_and_layer_0": [1, 2],
        "descriptive_peak_depth": 2,
        "layer_scores": [0.1, 0.4, 0.8, 0.2],
    }
    return {"datasets": {name: {"final_valid_response_token": entry} for name in ("motivation", "empathy")}}


def _causal(records: int = 200) -> dict:
    entry = {
        "records": records,
        "candidate_layers": [2, 3],
        "candidate_layer_rule": "test rule",
        "mean_faithfulness_by_layer": [0.05, 0.2, 0.9, 0.4],
    }
    return {"datasets": {name: entry for name in ("motivation", "empathy")}}


def _sae() -> dict:
    entry = {
        "status": "complete",
        "selected_layers": [2],
        "layer_scores": {"0": 0.1, "1": 0.3, "2": 0.8, "3": 0.5},
        "selected_features": [{"layer": 2, "feature_id": 91}],
    }
    return {"datasets": {name: entry for name in ("motivation", "empathy")}}


def test_metrics_and_explicit_disagreement(tmp_path: Path) -> None:
    atlas = build_circuit_atlas(
        _write(tmp_path / "probe.json", _probe()),
        _write(tmp_path / "causal.json", _causal()),
        _write(tmp_path / "sae.json", _sae()),
    )
    comparison = atlas["datasets"]["motivation"]["comparisons"]["linear_probe_vs_causal_patching"]
    assert comparison["jaccard"]["value"] == pytest.approx(1 / 3)
    assert comparison["disagreement"] == {
        "only_linear_probe": [1],
        "only_causal_patching": [3],
        "shared": [2],
        "excluded_outside_shared_domain": {"linear_probe": [], "causal_patching": []},
    }
    assert comparison["spearman"]["available"] is True
    assert comparison["spearman"]["n"] == 4


def test_incomplete_causal_and_missing_sae_are_not_compared(tmp_path: Path) -> None:
    atlas = build_circuit_atlas(
        _write(tmp_path / "probe.json", _probe()),
        _write(tmp_path / "causal.json", _causal(records=33)),
    )
    evidence = atlas["datasets"]["motivation"]["evidence"]
    assert evidence["causal_patching"]["available"] is False
    assert "33/200" in evidence["causal_patching"]["reason"]
    assert evidence["sae"]["available"] is False
    for comparison in atlas["datasets"]["motivation"]["comparisons"].values():
        if "linear_probe" not in comparison["methods"] or "causal_patching" in comparison["methods"]:
            assert comparison["status"] == "unavailable"


def test_constant_ranking_makes_spearman_unavailable(tmp_path: Path) -> None:
    causal = _causal()
    for entry in causal["datasets"].values():
        entry["mean_faithfulness_by_layer"] = [1, 1, 1, 1]
    atlas = build_circuit_atlas(
        _write(tmp_path / "probe.json", _probe()), _write(tmp_path / "causal.json", causal)
    )
    spearman = atlas["datasets"]["empathy"]["comparisons"]["linear_probe_vs_causal_patching"]["spearman"]
    assert spearman == {"available": False, "reason": "at least one ranking is constant"}


def test_writes_json_markdown_and_chart(tmp_path: Path) -> None:
    atlas = build_circuit_atlas(
        _write(tmp_path / "probe.json", _probe()), _write(tmp_path / "causal.json", _causal())
    )
    paths = write_circuit_atlas(atlas, tmp_path / "out", "test")
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    report = paths[1].read_text(encoding="utf-8")
    assert "SAE summary is not yet available" in report
    assert "disagreement" in report


def test_probe_terminal_depth_is_excluded_when_other_method_has_no_peer(tmp_path: Path) -> None:
    probe = _probe()
    for item in probe["datasets"].values():
        item["final_valid_response_token"]["significant_depths_vs_shuffled_and_layer_0"] = [2, 32]
    atlas = build_circuit_atlas(
        _write(tmp_path / "probe.json", probe), _write(tmp_path / "causal.json", _causal())
    )
    comparison = atlas["datasets"]["motivation"]["comparisons"]["linear_probe_vs_causal_patching"]
    assert comparison["jaccard"]["union"] == [2, 3]
    assert comparison["disagreement"]["excluded_outside_shared_domain"]["linear_probe"] == [32]
