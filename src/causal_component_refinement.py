"""Post-layer causal refinement over Llama attention and MLP outputs.

This module deliberately does not choose layers.  It consumes an explicitly marked
corrected 200+200 layer-patching summary and an explicit layer list per dataset.
The induced node universe is ``(layer, attention_output|mlp_output)`` at the
controlled response position.  It is not a head-, neuron-, or position-complete
description of the model.
"""

from __future__ import annotations

import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
import torch


Component = Literal["attention_output", "mlp_output"]
Mode = Literal["restore", "knockout"]


@dataclass(frozen=True, order=True)
class ComponentNode:
    """One whole component output at one layer and controlled token position."""

    layer: int
    component: Component

    @property
    def key(self) -> str:
        return f"layer_{self.layer:02d}.{self.component}"


@dataclass(frozen=True)
class RefinementPrerequisites:
    summary_path: Path
    run_id: str
    selected_layers: dict[str, tuple[int, ...]]


def validate_refinement_prerequisites(
    summary_path: Path,
    selected_layers: Mapping[str, Sequence[int]],
    *,
    expected_records: int = 200,
    num_layers: int = 32,
) -> RefinementPrerequisites:
    """Fail closed unless the corrected summary and explicit layer sets are valid.

    The summary must carry the explicit marker
    ``summary_kind=corrected_causal_patching``.  A filename or a generic
    ``complete`` status is intentionally insufficient.
    """

    value = json.loads(summary_path.read_text(encoding="utf-8"))
    if value.get("summary_kind") != "corrected_causal_patching":
        raise RuntimeError(
            "Component refinement requires summary_kind='corrected_causal_patching'"
        )
    datasets = value.get("datasets")
    if not isinstance(datasets, dict):
        raise RuntimeError("Corrected causal summary has no datasets mapping")

    normalized: dict[str, tuple[int, ...]] = {}
    for dataset in ("motivation", "empathy"):
        item = datasets.get(dataset)
        if not isinstance(item, dict) or item.get("records") != expected_records:
            found = None if not isinstance(item, dict) else item.get("records")
            raise RuntimeError(
                f"{dataset}: corrected causal summary must contain exactly "
                f"{expected_records} records; found {found}"
            )
        if dataset not in selected_layers:
            raise RuntimeError(f"{dataset}: selected layers must be supplied explicitly")
        raw = selected_layers[dataset]
        if not raw:
            raise RuntimeError(f"{dataset}: selected layers cannot be empty")
        if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in raw):
            raise TypeError(f"{dataset}: selected layers must be integers")
        layers = tuple(sorted(set(raw)))
        if any(layer < 0 or layer >= num_layers for layer in layers):
            raise ValueError(f"{dataset}: selected layer outside [0, {num_layers - 1}]")
        normalized[dataset] = layers

    run_id = str(value.get("run_id", ""))
    if not run_id:
        raise RuntimeError("Corrected causal summary must contain a run_id")
    return RefinementPrerequisites(summary_path, run_id, normalized)


def induced_nodes(layers: Sequence[int]) -> tuple[ComponentNode, ...]:
    """Return the complete *induced* universe for the supplied selected layers."""

    return tuple(
        ComponentNode(layer, component)
        for layer in sorted(set(layers))
        for component in ("attention_output", "mlp_output")
    )


def _component_module(model: Any, node: ComponentNode) -> Any:
    layer = model.model.layers[node.layer]
    return layer.self_attn if node.component == "attention_output" else layer.mlp


def _tensor_from_output(output: Any) -> torch.Tensor:
    tensor = output[0] if isinstance(output, tuple) else output
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Component hook expected a Tensor or tuple whose first item is a Tensor")
    return tensor


def _replace_output(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (replacement,) + output[1:]
    return replacement


@contextmanager
def capture_component_outputs(
    model: Any,
    nodes: Iterable[ComponentNode],
    *,
    to_cpu: bool = True,
) -> Iterator[dict[ComponentNode, torch.Tensor]]:
    """Capture whole component outputs; caller performs the model forward pass."""

    cache: dict[ComponentNode, torch.Tensor] = {}
    handles = []
    for node in nodes:
        def hook(_module: Any, _args: tuple[Any, ...], output: Any, node: ComponentNode = node) -> None:
            tensor = _tensor_from_output(output).detach()
            cache[node] = tensor.to("cpu", copy=True) if to_cpu else tensor.clone()

        handles.append(_component_module(model, node).register_forward_hook(hook))
    try:
        yield cache
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def intervene_component_outputs(
    model: Any,
    source_cache: Mapping[ComponentNode, torch.Tensor],
    subset: Iterable[ComponentNode],
    positions: Sequence[int],
    *,
    mode: Mode,
) -> Iterator[None]:
    """Restore or knock out a node subset using activations from another run.

    ``restore`` conventionally injects clean activations into a corrupted run.
    ``knockout`` conventionally injects corrupted activations into a clean run.
    Both use the same exact hook mechanics; the named mode records causal intent
    and prevents ambiguous calls.  The source cache must be supplied explicitly.
    """

    if mode not in ("restore", "knockout"):
        raise ValueError("mode must be 'restore' or 'knockout'")
    chosen = tuple(subset)
    missing = [node.key for node in chosen if node not in source_cache]
    if missing:
        raise KeyError(f"Source cache is missing nodes: {missing}")
    handles = []
    for node in chosen:
        def hook(_module: Any, _args: tuple[Any, ...], output: Any, node: ComponentNode = node) -> Any:
            target = _tensor_from_output(output)
            if target.ndim != 3 or len(positions) != target.shape[0]:
                raise ValueError("Expected [batch, sequence, hidden] output and one position per batch item")
            source = source_cache[node].to(device=target.device, dtype=target.dtype)
            if source.shape != target.shape:
                raise ValueError(
                    f"{node.key}: source shape {tuple(source.shape)} != target shape {tuple(target.shape)}"
                )
            replacement = target.clone()
            for batch_index, position in enumerate(positions):
                if position < 0 or position >= target.shape[1]:
                    raise IndexError(f"Patch position {position} is outside the sequence")
                replacement[batch_index, position] = source[batch_index, position]
            return _replace_output(output, replacement)

        handles.append(_component_module(model, node).register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def normalized_recovery(
    intervention_score: float,
    clean_score: float,
    corrupted_score: float,
    *,
    epsilon: float = 1e-8,
) -> float | None:
    """Wang-style normalized causal recovery, intentionally not clipped."""

    denominator = clean_score - corrupted_score
    if abs(denominator) <= epsilon:
        return None
    return (intervention_score - corrupted_score) / denominator


def rank_candidate_nodes(
    per_record_recovery: Sequence[Mapping[str, float | None]],
) -> list[dict[str, Any]]:
    """Rank nodes by mean per-record normalized recovery, then stable node key."""

    keys = sorted({key for record in per_record_recovery for key in record})
    ranked = []
    for key in keys:
        values = np.asarray(
            [record[key] for record in per_record_recovery if record.get(key) is not None],
            dtype=np.float64,
        )
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"{key}: no finite recovery values")
        ranked.append({
            "node": key,
            "valid_records": int(values.size),
            "mean_recovery": float(values.mean()),
            "median_recovery": float(np.median(values)),
            "positive_fraction": float((values > 0).mean()),
        })
    ranked.sort(key=lambda item: (-item["mean_recovery"], item["node"]))
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked


def semantic_subset(ranking: Sequence[Mapping[str, Any]], size: int) -> tuple[str, ...]:
    """Top-k positive-effect nodes under the prespecified semantic component ranking."""

    eligible = [str(item["node"]) for item in ranking if float(item["mean_recovery"]) > 0]
    if size < 0 or size > len(eligible):
        raise ValueError("Semantic subset size exceeds positive-effect candidates")
    return tuple(eligible[:size])


def random_subsets(
    universe: Sequence[str], size: int, *, seed: int, repetitions: int
) -> list[tuple[str, ...]]:
    """Seeded size-matched random-node baseline subsets."""

    unique = tuple(sorted(set(universe)))
    if size < 0 or size > len(unique) or repetitions <= 0:
        raise ValueError("Invalid random subset request")
    rng = random.Random(seed)
    return [tuple(sorted(rng.sample(unique, size))) for _ in range(repetitions)]


def greedy_subset(
    universe: Sequence[str],
    size: int,
    evaluator: Callable[[tuple[str, ...]], float],
) -> tuple[str, ...]:
    """Forward-select nodes by measured marginal subset recovery."""

    remaining = set(universe)
    if size < 0 or size > len(remaining):
        raise ValueError("Invalid greedy subset size")
    selected: list[str] = []
    for _ in range(size):
        scored = [
            (float(evaluator(tuple(sorted(selected + [candidate])))), candidate)
            for candidate in remaining
        ]
        _, best = max(scored, key=lambda item: (item[0], item[1]))
        selected.append(best)
        remaining.remove(best)
    return tuple(selected)


def wang_style_subset_metrics(
    *,
    candidate: Sequence[str],
    universe: Sequence[str],
    clean_score: float,
    corrupted_score: float,
    candidate_restoration_score: float,
    complement_restoration_score: float,
    drop_one_restoration_scores: Mapping[str, float],
    minimality_threshold: float = 0.0,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Compute induced-universe faithfulness, completeness, and minimality.

    Faithfulness is candidate normalized recovery. Completeness is one minus the
    complement's recovery. Minimality is the loss in candidate faithfulness when
    each candidate node is removed. These are Wang-style causal tests over the
    induced universe only; raw scores are retained for auditability.
    """

    candidate_set = set(candidate)
    universe_set = set(universe)
    if not candidate_set <= universe_set:
        raise ValueError("Candidate contains nodes outside the induced universe")
    if set(drop_one_restoration_scores) != candidate_set:
        raise ValueError("Exactly one drop-one score is required for each candidate node")
    faithfulness = normalized_recovery(
        candidate_restoration_score, clean_score, corrupted_score, epsilon=epsilon
    )
    complement_recovery = normalized_recovery(
        complement_restoration_score, clean_score, corrupted_score, epsilon=epsilon
    )
    if faithfulness is None or complement_recovery is None:
        raise ValueError("Clean-minus-corrupted denominator is zero or too small")
    necessity: dict[str, float] = {}
    for node, score in sorted(drop_one_restoration_scores.items()):
        reduced = normalized_recovery(score, clean_score, corrupted_score, epsilon=epsilon)
        if reduced is None:
            raise AssertionError("Denominator validity changed within one evaluation")
        necessity[node] = faithfulness - reduced
    return {
        "faithfulness": faithfulness,
        "completeness": 1.0 - complement_recovery,
        "complement_recovery": complement_recovery,
        "minimality_by_node": necessity,
        "minimality_mean": float(np.mean(list(necessity.values()))) if necessity else None,
        "minimality_fraction_above_threshold": (
            float(np.mean([value > minimality_threshold for value in necessity.values()]))
            if necessity else None
        ),
        "minimality_threshold": minimality_threshold,
        "raw_scores": {
            "clean": clean_score,
            "corrupted": corrupted_score,
            "candidate_restoration": candidate_restoration_score,
            "complement_restoration": complement_restoration_score,
        },
    }


def compact_refinement_report(
    prerequisites: RefinementPrerequisites,
    dataset_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a compact, JSON-serializable report without activation tensors."""

    return {
        "source_causal_run_id": prerequisites.run_id,
        "summary_kind": "post_layer_component_refinement",
        "induced_node_definition": (
            "whole attention or MLP output vector at the controlled token position "
            "for each explicitly selected layer"
        ),
        "selected_layers": {
            key: list(value) for key, value in prerequisites.selected_layers.items()
        },
        "datasets": dict(dataset_results),
        "interpretation_boundary": (
            "Completeness is relative only to the induced layer-by-component universe; "
            "it excludes unselected layers, other positions, heads, neurons and SAE features."
        ),
        "computational_limits": [
            "Subset evaluation requires fresh model forwards; exhaustive 2^N search is not run.",
            "Greedy selection is order-dependent and is compared with seeded size-matched random subsets.",
            "Component hooks store whole hidden vectors, so batches must be sized to available memory.",
        ],
    }


def write_compact_refinement_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
