"""Cross-method circuit-atlas reporting without inventing missing evidence.

The atlas compares *layer-level* selections only.  Feature/neuron-level SAE
identities are deliberately not compared with layer-level probe or patching
results because those sets have different units.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


METHODS = ("linear_probe", "causal_patching", "sae")
DEFAULT_DATASETS = ("motivation", "empathy")


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _integer_layers(value: Any) -> list[int]:
    if value is None:
        return []
    return sorted({int(item) for item in value})


def _score_map(value: Any) -> dict[int, float]:
    """Accept either {layer: score} or a score list indexed by layer."""
    if isinstance(value, Mapping):
        return {int(layer): float(score) for layer, score in value.items()}
    if isinstance(value, list):
        return {index: float(score) for index, score in enumerate(value)}
    return {}


def _unavailable(reason: str, source: Path | None) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "source": str(source) if source else None,
        "selected_layers": [],
        "layer_scores": {},
    }


def _probe_evidence(
    document: dict[str, Any] | None,
    dataset: str,
    source: Path | None,
    aggregation: str,
) -> dict[str, Any]:
    if document is None:
        return _unavailable("linear-probe summary was not supplied", source)
    entry = document.get("datasets", {}).get(dataset, {}).get(aggregation)
    if not isinstance(entry, dict):
        return _unavailable(f"probe aggregation {aggregation!r} is absent", source)
    selected = _integer_layers(entry.get("significant_depths_vs_shuffled_and_layer_0"))
    scores = _score_map(entry.get("layer_scores") or entry.get("macro_f1_by_depth"))
    return {
        "available": True,
        "reason": None,
        "source": str(source),
        "coordinate": "residual-stream representational depth (0-32)",
        "layer_domain": list(range(33)),
        "selection_rule": "significant versus shuffled labels and layer 0",
        "selected_layers": selected,
        "layer_scores": scores,
        "descriptive_peak_layer": entry.get("descriptive_peak_depth"),
        "ranking_note": None if scores else "summary contains no complete per-layer ranking",
    }


def _causal_dataset_entry(document: dict[str, Any], dataset: str) -> dict[str, Any] | None:
    datasets = document.get("datasets")
    if isinstance(datasets, dict):
        value = datasets.get(dataset)
        return value if isinstance(value, dict) else None
    # A per-dataset layer_summary file may be supplied directly.
    if "mean_faithfulness_by_layer" in document:
        return document
    return None


def _causal_evidence(
    document: dict[str, Any] | None,
    dataset: str,
    source: Path | None,
    expected_records: int,
) -> dict[str, Any]:
    if document is None:
        return _unavailable("completed causal-patching summary was not supplied", source)
    entry = _causal_dataset_entry(document, dataset)
    if entry is None:
        return _unavailable(f"causal summary has no {dataset!r} result", source)
    records = int(entry.get("records", 0))
    if records < expected_records:
        return _unavailable(
            f"causal result is incomplete ({records}/{expected_records} records)", source
        )
    scores = _score_map(entry.get("mean_faithfulness_by_layer"))
    return {
        "available": True,
        "reason": None,
        "source": str(source),
        "coordinate": "residual-pre layer index (0-31)",
        "layer_domain": list(range(len(scores))),
        "selection_rule": entry.get("candidate_layer_rule", "reported candidate layers"),
        "selected_layers": _integer_layers(entry.get("candidate_layers")),
        "layer_scores": scores,
        "records": records,
    }


def _sae_evidence(
    document: dict[str, Any] | None, dataset: str, source: Path | None
) -> dict[str, Any]:
    if document is None:
        return _unavailable("SAE summary is not yet available", source)
    entry = document.get("datasets", {}).get(dataset)
    if not isinstance(entry, dict):
        return _unavailable(f"SAE summary has no {dataset!r} result", source)
    if entry.get("status") not in (None, "complete", "completed"):
        return _unavailable(f"SAE result status is {entry.get('status')!r}", source)
    selected = entry.get("selected_layers")
    if selected is None:
        features = entry.get("selected_features", [])
        selected = [item["layer"] for item in features if isinstance(item, dict) and "layer" in item]
    scores = _score_map(
        entry.get("layer_scores")
        or entry.get("mean_interpretability_by_layer")
        or entry.get("aggregate_feature_score_by_layer")
    )
    return {
        "available": True,
        "reason": None,
        "source": str(source),
        "coordinate": entry.get("coordinate", "residual-stream SAE layer index"),
        "layer_domain": list(range(int(entry.get("layer_count", 32)))),
        "selection_rule": entry.get("selection_rule", "reported selected SAE layers"),
        "selected_layers": _integer_layers(selected),
        "layer_scores": scores,
        "feature_level_comparison": {
            "available": False,
            "reason": "SAE feature IDs are not comparable with layer-level probe/patching sets",
        },
    }


def _compare(left_name: str, left: dict[str, Any], right_name: str, right: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"methods": [left_name, right_name]}
    if not left["available"] or not right["available"]:
        missing = [name for name, item in ((left_name, left), (right_name, right)) if not item["available"]]
        result.update({
            "status": "unavailable",
            "reason": "required completed evidence unavailable: " + ", ".join(missing),
            "jaccard": None,
            "spearman": None,
            "disagreement": None,
        })
        return result

    shared_domain = set(left.get("layer_domain", [])) & set(right.get("layer_domain", []))
    if not shared_domain:
        result.update({
            "status": "unavailable",
            "reason": "methods have no shared layer coordinate domain",
            "jaccard": None,
            "spearman": None,
            "disagreement": None,
        })
        return result
    left_all, right_all = set(left["selected_layers"]), set(right["selected_layers"])
    left_set, right_set = left_all & shared_domain, right_all & shared_domain
    union = left_set | right_set
    jaccard = None if not union else len(left_set & right_set) / len(union)
    common_scores = sorted(set(left["layer_scores"]) & set(right["layer_scores"]))
    spearman: dict[str, Any]
    if len(common_scores) < 2:
        spearman = {"available": False, "reason": "fewer than two shared ranked layers"}
    else:
        x = np.asarray([left["layer_scores"][layer] for layer in common_scores], dtype=float)
        y = np.asarray([right["layer_scores"][layer] for layer in common_scores], dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            spearman = {"available": False, "reason": "ranking contains non-finite scores"}
        elif np.ptp(x) == 0 or np.ptp(y) == 0:
            spearman = {"available": False, "reason": "at least one ranking is constant"}
        else:
            statistic, p_value = spearmanr(x, y)
            spearman = {
                "available": True,
                "rho": float(statistic),
                "p_value_two_sided": float(p_value),
                "shared_layers": common_scores,
                "n": len(common_scores),
            }
    result.update({
        "status": "comparable",
        "jaccard": {
            "available": jaccard is not None,
            "value": jaccard,
            "intersection": sorted(left_set & right_set),
            "union": sorted(union),
            "reason": None if jaccard is not None else "both selected-layer sets are empty",
        },
        "spearman": spearman,
        "disagreement": {
            f"only_{left_name}": sorted(left_set - right_set),
            f"only_{right_name}": sorted(right_set - left_set),
            "shared": sorted(left_set & right_set),
            "excluded_outside_shared_domain": {
                left_name: sorted(left_all - shared_domain),
                right_name: sorted(right_all - shared_domain),
            },
        },
    })
    return result


def build_circuit_atlas(
    probe_summary: Path,
    causal_summary: Path,
    sae_summary: Path | None = None,
    *,
    probe_aggregation: str = "final_valid_response_token",
    expected_records: int = 200,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
) -> dict[str, Any]:
    probe_doc, causal_doc, sae_doc = (
        _read_json(probe_summary), _read_json(causal_summary), _read_json(sae_summary)
    )
    atlas: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "layer-level cross-method agreement",
        "alignment": (
            "Probe representational depth k and residual-pre patch/SAE layer k are compared only "
            "on their shared integer layer indices; probe depth 32 has no residual-pre layer-32 peer."
        ),
        "datasets": {},
    }
    for dataset in datasets:
        evidence = {
            "linear_probe": _probe_evidence(probe_doc, dataset, probe_summary, probe_aggregation),
            "causal_patching": _causal_evidence(
                causal_doc, dataset, causal_summary, expected_records
            ),
            "sae": _sae_evidence(sae_doc, dataset, sae_summary),
        }
        comparisons = {
            f"{left}_vs_{right}": _compare(left, evidence[left], right, evidence[right])
            for left, right in (
                ("linear_probe", "causal_patching"),
                ("linear_probe", "sae"),
                ("causal_patching", "sae"),
            )
        }
        atlas["datasets"][dataset] = {"evidence": evidence, "comparisons": comparisons}
    return atlas


def _plot_selection_matrix(atlas: dict[str, Any], path: Path) -> None:
    datasets = list(atlas["datasets"])
    figure, axes = plt.subplots(len(datasets), 1, figsize=(11, 2.5 * len(datasets)), squeeze=False)
    for axis, dataset in zip(axes[:, 0], datasets):
        evidence = atlas["datasets"][dataset]["evidence"]
        matrix = np.full((len(METHODS), 33), np.nan)
        for row, method in enumerate(METHODS):
            if evidence[method]["available"]:
                matrix[row, :] = 0
                selected = [layer for layer in evidence[method]["selected_layers"] if 0 <= layer <= 32]
                matrix[row, selected] = 1
        cmap = matplotlib.colormaps["Blues"].copy()
        cmap.set_bad("lightgrey")
        axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        axis.set_yticks(range(len(METHODS)), [name.replace("_", " ") for name in METHODS])
        axis.set_xticks(range(33))
        axis.set_xlabel("Aligned layer/depth index")
        axis.set_title(dataset.capitalize() + " (grey = unavailable; blue = selected)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_circuit_atlas(atlas: dict[str, Any], output_directory: Path, run_label: str) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"circuit_atlas_{run_label}.json"
    report_path = output_directory / f"circuit_atlas_{run_label}.md"
    chart_path = output_directory / f"circuit_atlas_{run_label}.png"
    json_path.write_text(json.dumps(atlas, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Cross-method circuit atlas", "", atlas["alignment"], ""]
    for dataset, item in atlas["datasets"].items():
        lines.extend([f"## {dataset.capitalize()}", "", "### Evidence", ""])
        for method, evidence in item["evidence"].items():
            if evidence["available"]:
                lines.append(f"- {method}: selected layers `{evidence['selected_layers']}`")
            else:
                lines.append(f"- {method}: **unavailable** — {evidence['reason']}")
        lines.extend(["", "### Agreement and disagreement", ""])
        for name, comparison in item["comparisons"].items():
            if comparison["status"] != "comparable":
                lines.append(f"- {name}: unavailable — {comparison['reason']}")
                continue
            jac = comparison["jaccard"]
            jac_text = f"{jac['value']:.4f}" if jac["available"] else f"unavailable ({jac['reason']})"
            rho = comparison["spearman"]
            rho_text = f"{rho['rho']:.4f} (n={rho['n']})" if rho["available"] else f"unavailable ({rho['reason']})"
            lines.append(f"- {name}: Jaccard {jac_text}; Spearman {rho_text}; disagreement `{comparison['disagreement']}`")
        lines.append("")
    lines.extend([
        "## Interpretation boundary", "",
        "Agreement is descriptive and does not make a layer set a causal circuit. Feature-level SAE IDs are not compared with layer IDs. Unavailable comparisons remain explicit until their source analysis is complete.", "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _plot_selection_matrix(atlas, chart_path)
    return json_path, report_path, chart_path
