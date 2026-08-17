"""Charts and reports for completed independent linear probes."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn


AGGREGATIONS = ("final_valid_response_token", "mean_response_token_activation")
AGGREGATION_LABELS = {
    "final_valid_response_token": "Final valid response token",
    "mean_response_token_activation": "Mean response-token activation",
}


def _read_result(root: Path, dataset: str, aggregation: str, run_id: str) -> dict[str, Any]:
    path = root / dataset / f"{aggregation}_{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _series(result: dict[str, Any], dotted: str) -> np.ndarray:
    first, second = dotted.split(".")
    return np.asarray([layer[first][second] for layer in result["layers"]], dtype=float)


def _significant_depths(result: dict[str, Any]) -> list[int]:
    return [
        int(layer["depth"])
        for layer in result["layers"]
        if layer["permutation_p_values"]["max_statistic_across_depths"] <= 0.05
        and layer["permutation_p_values"]["improvement_over_layer_0"] <= 0.05
    ]


def _plot_aggregation(result: dict[str, Any], output: Path) -> None:
    depths = np.arange(len(result["layers"]))
    accuracy = _series(result, "test.accuracy")
    macro_f1 = _series(result, "test.macro_f1")
    baseline_accuracy = np.asarray(
        [layer["shuffled_label_baseline_100"]["accuracy_mean"] for layer in result["layers"]]
    )
    baseline_f1 = np.asarray(
        [layer["shuffled_label_baseline_100"]["macro_f1_mean"] for layer in result["layers"]]
    )
    lower = np.asarray(
        [
            layer["shuffled_label_baseline_100"]["macro_f1_2_5_percentile"]
            for layer in result["layers"]
        ]
    )
    upper = np.asarray(
        [
            layer["shuffled_label_baseline_100"]["macro_f1_97_5_percentile"]
            for layer in result["layers"]
        ]
    )
    significant = _significant_depths(result)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(depths, accuracy, marker="o", markersize=3, label="Observed test accuracy")
    axes[0].plot(depths, baseline_accuracy, linestyle="--", label="Shuffled-label mean")
    axes[0].axhline(accuracy[0], color="grey", linestyle=":", label="Layer-0 observed")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.03)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right")

    axes[1].fill_between(depths, lower, upper, alpha=0.18, label="Shuffled-label 95% interval")
    axes[1].plot(depths, baseline_f1, linestyle="--", label="Shuffled-label mean")
    axes[1].plot(depths, macro_f1, marker="o", markersize=3, label="Observed test macro-F1")
    axes[1].axhline(macro_f1[0], color="grey", linestyle=":", label="Layer-0 observed")
    if significant:
        axes[1].scatter(
            significant,
            macro_f1[significant],
            marker="*",
            s=70,
            color="crimson",
            label="Passes shuffled + layer-0 permutation criteria",
            zorder=4,
        )
    axes[1].set_xlabel("Representational depth (0 = residual entering block 0; 1-32 = block outputs)")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1.03)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)
    figure.suptitle(
        f"{result['dataset'].title()} - {AGGREGATION_LABELS[result['aggregation']]}"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_comparison(results: dict[str, dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for aggregation, result in results.items():
        macro_f1 = _series(result, "test.macro_f1")
        axis.plot(
            np.arange(len(macro_f1)),
            macro_f1,
            marker="o",
            markersize=3,
            label=AGGREGATION_LABELS[aggregation],
        )
    axis.set_xlabel("Representational depth (0 = residual entering block 0; 1-32 = block outputs)")
    axis.set_ylabel("Held-out test macro-F1")
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.25)
    axis.legend()
    dataset = next(iter(results.values()))["dataset"]
    axis.set_title(f"{dataset.title()} probe aggregation sensitivity")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _summary_entry(result: dict[str, Any]) -> dict[str, Any]:
    f1 = _series(result, "test.macro_f1")
    accuracy = _series(result, "test.accuracy")
    peak = int(np.argmax(f1))
    significant = _significant_depths(result)
    return {
        "layer_0_test_accuracy": float(accuracy[0]),
        "layer_0_test_macro_f1": float(f1[0]),
        "descriptive_peak_depth": peak,
        "descriptive_peak_test_accuracy": float(accuracy[peak]),
        "descriptive_peak_test_macro_f1": float(f1[peak]),
        "significant_depths_vs_shuffled_and_layer_0": significant,
        "significant_depth_count": len(significant),
        "minimum_attainable_permutation_p": 1 / (result["permutations"] + 1),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_probe_report(project_root: Path, run_id: str) -> tuple[Path, Path, Path]:
    probe_root = project_root / "results" / "probes"
    metadata_root = probe_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "activation_run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "datasets": {},
    }

    for dataset in ("motivation", "empathy"):
        results = {
            aggregation: _read_result(probe_root, dataset, aggregation, run_id)
            for aggregation in AGGREGATIONS
        }
        summary["datasets"][dataset] = {
            aggregation: _summary_entry(result) for aggregation, result in results.items()
        }
        for aggregation, result in results.items():
            _plot_aggregation(result, probe_root / dataset / f"{aggregation}_{run_id}.png")
        _plot_comparison(results, probe_root / dataset / f"aggregation_comparison_{run_id}.png")

    summary_path = metadata_root / f"linear_probe_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    motivation = summary["datasets"]["motivation"]
    empathy = summary["datasets"]["empathy"]
    report = f"""# Linear probing report

- Activation run: `{run_id}`
- Tasks: motivation and empathy analysed independently
- Examples: 200 per task, uniformly sampled without replacement
- Sequence length: 128
- Split: grouped 70/15/15, seed 42
- Probe: training-standardised L2 logistic regression, validation-selected C
- Baselines: layer 0 and 100 shuffled-label repetitions
- Significance: 1,000 full development-label permutations

## Split audit

- Motivation: 140/30/30 examples; negatives 12/2/2; 52/11/14 transcript groups.
- Empathy: 140/30/30 examples; labels 92/48, 20/10, and 20/10; 132/28/27 dialogue groups.
- No transcript or dialogue appears in more than one split.

## Motivation

The final-token probe rises from layer-0 macro-F1
`{motivation['final_valid_response_token']['layer_0_test_macro_f1']:.3f}` to a descriptive
peak of `{motivation['final_valid_response_token']['descriptive_peak_test_macro_f1']:.3f}`
at depth {motivation['final_valid_response_token']['descriptive_peak_depth']}.
Depths {motivation['final_valid_response_token']['significant_depths_vs_shuffled_and_layer_0']}
pass both the shuffled-label/max-statistic and layer-0-improvement criteria.

The mean-token sensitivity probe has a strong layer-0 baseline of
`{motivation['mean_response_token_activation']['layer_0_test_macro_f1']:.3f}` and a
descriptive peak of
`{motivation['mean_response_token_activation']['descriptive_peak_test_macro_f1']:.3f}` at
depth {motivation['mean_response_token_activation']['descriptive_peak_depth']}. Only depth
{motivation['mean_response_token_activation']['significant_depths_vs_shuffled_and_layer_0']}
passes both criteria.

The motivation test set contains only two non-motivational responses because the approved
random 200-example sample is highly imbalanced. Its accuracy and macro-F1 therefore have
coarse resolution and must not be presented as a precise population estimate. No resampling,
class weighting, or label alteration was applied.

## Empathy

The final-token probe rises from layer-0 macro-F1
`{empathy['final_valid_response_token']['layer_0_test_macro_f1']:.3f}` to a descriptive peak
of `{empathy['final_valid_response_token']['descriptive_peak_test_macro_f1']:.3f}` at depth
{empathy['final_valid_response_token']['descriptive_peak_depth']}. Every transformer-block
output depth 1-32 passes both permutation criteria.

The mean-token sensitivity probe is already strong at layer 0
(`{empathy['mean_response_token_activation']['layer_0_test_macro_f1']:.3f}`) and reaches a
descriptive peak of
`{empathy['mean_response_token_activation']['descriptive_peak_test_macro_f1']:.3f}` at depth
{empathy['mean_response_token_activation']['descriptive_peak_depth']}. It beats shuffled
labels, but no depth significantly improves over the strong layer-0 baseline under the declared
permutation criterion.

## Interpretation boundary

These probes show that motivation and empathy labels are linearly decodable from residual-
stream representations. They do not establish that the model causally uses those
representations. Candidate layers for the circuit atlas must also be supported by activation
patching and compatible SAE evidence. Descriptive test peaks are reported for orientation;
they are not treated as causal circuit selections.
"""
    report_path = metadata_root / f"linear_probe_report_{run_id}.md"
    report_path.write_text(report, encoding="utf-8")

    manifest_path = metadata_root / f"linear_probe_manifest_{run_id}.json"
    files = []
    for path in sorted(probe_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        files.append(
            {
                "path": str(path.relative_to(project_root)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "activation_run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary_path, report_path, manifest_path
