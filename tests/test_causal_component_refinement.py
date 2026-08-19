from pathlib import Path

import pytest
import torch
from torch import nn

from src.causal_component_refinement import (
    ComponentNode,
    capture_component_outputs,
    greedy_subset,
    induced_nodes,
    intervene_component_outputs,
    normalized_recovery,
    random_subsets,
    rank_candidate_nodes,
    semantic_subset,
    validate_refinement_prerequisites,
    wang_style_subset_metrics,
)


class _TupleAttention(nn.Module):
    def forward(self, value):
        return value + 1.0, "aux"


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TupleAttention()
        self.mlp = nn.Identity()

    def forward(self, value):
        value, _ = self.self_attn(value)
        return self.mlp(value * 2.0)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])


class _ToyLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()

    def forward(self, value):
        return self.model.layers[0](value)


FIXTURES = Path(__file__).parent / "fixtures" / "component_refinement"


def test_prerequisites_fail_closed_and_require_explicit_layers() -> None:
    path = FIXTURES / "ordinary_summary.json"
    with pytest.raises(RuntimeError, match="corrected_causal_patching"):
        validate_refinement_prerequisites(path, {"motivation": [1], "empathy": [2]})
    path = FIXTURES / "partial_corrected_summary.json"
    with pytest.raises(RuntimeError, match="exactly 200"):
        validate_refinement_prerequisites(path, {"motivation": [1], "empathy": [2]})
    path = FIXTURES / "corrected_summary.json"
    with pytest.raises(RuntimeError, match="empathy: selected layers"):
        validate_refinement_prerequisites(path, {"motivation": [1]})


def test_prerequisites_and_induced_nodes() -> None:
    value = validate_refinement_prerequisites(
        FIXTURES / "corrected_summary.json",
        {"motivation": [16, 10, 16], "empathy": [8]},
    )
    assert value.selected_layers["motivation"] == (10, 16)
    assert [node.key for node in induced_nodes([2])] == [
        "layer_02.attention_output", "layer_02.mlp_output"
    ]


def test_capture_restore_and_knockout_share_explicit_source_semantics() -> None:
    model = _ToyLlama()
    nodes = induced_nodes([0])
    clean = torch.tensor([[[1.0], [2.0]]])
    corrupted = torch.tensor([[[10.0], [20.0]]])
    with capture_component_outputs(model, nodes) as clean_cache:
        model(clean)
    baseline = model(corrupted)
    with intervene_component_outputs(
        model, clean_cache, [ComponentNode(0, "attention_output")], [1], mode="restore"
    ):
        restored = model(corrupted)
    assert baseline[0, 1, 0] == 42.0
    assert restored[0, 1, 0] == 6.0

    with capture_component_outputs(model, nodes) as corrupted_cache:
        model(corrupted)
    with intervene_component_outputs(
        model, corrupted_cache, [ComponentNode(0, "mlp_output")], [1], mode="knockout"
    ):
        knocked_out = model(clean)
    assert knocked_out[0, 1, 0] == baseline[0, 1, 0]


def test_ranking_and_subset_definitions_are_deterministic() -> None:
    ranking = rank_candidate_nodes([
        {"a": 0.2, "b": 0.1, "c": -0.1},
        {"a": 0.4, "b": 0.5, "c": 0.0},
    ])
    assert [item["node"] for item in ranking] == ["a", "b", "c"]
    assert semantic_subset(ranking, 2) == ("a", "b")
    assert random_subsets(["a", "b", "c"], 2, seed=7, repetitions=3) == \
        random_subsets(["c", "b", "a"], 2, seed=7, repetitions=3)
    scores = {("a",): 0.5, ("b",): 0.3, ("c",): 0.1,
              ("a", "b"): 0.8, ("a", "c"): 0.6}
    assert greedy_subset(["a", "b", "c"], 2, lambda subset: scores[tuple(sorted(subset))]) == ("a", "b")


def test_wang_style_metrics_retain_unclipped_recovery_and_drop_one_necessity() -> None:
    assert normalized_recovery(12.0, 10.0, 0.0) == 1.2
    metrics = wang_style_subset_metrics(
        candidate=["a", "b"], universe=["a", "b", "c"],
        clean_score=10.0, corrupted_score=0.0,
        candidate_restoration_score=8.0, complement_restoration_score=1.0,
        drop_one_restoration_scores={"a": 5.0, "b": 6.0},
        minimality_threshold=0.1,
    )
    assert metrics["faithfulness"] == pytest.approx(0.8)
    assert metrics["completeness"] == pytest.approx(0.9)
    assert metrics["minimality_by_node"] == pytest.approx({"a": 0.3, "b": 0.2})
    assert metrics["minimality_fraction_above_threshold"] == 1.0
