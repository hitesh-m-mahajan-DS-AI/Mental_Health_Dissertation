"""Verification and compact reporting for completed causal-patching runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .causal_config import CausalPatchingConfig
from .causal_patching import summarize_layer_results


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_verify_records(
    config: CausalPatchingConfig, dataset: str, run_id: str, expected: int = 200
) -> list[dict[str, Any]]:
    directory = config.output_root / dataset / "pairs" / run_id
    paths = sorted(directory.glob("*.json"))
    if len(paths) != expected:
        raise RuntimeError(f"{dataset}: expected {expected} pair results, found {len(paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    ordinals = [int(record["selection_ordinal"]) for record in records]
    if sorted(ordinals) != list(range(expected)):
        raise RuntimeError(f"{dataset}: result ordinals are incomplete or duplicated")
    for record in records:
        if len(record["patched_logit_differences"]) != 32 or len(record["faithfulness"]) != 32:
            raise RuntimeError(f"{dataset}/{record['example_id']}: expected 32 layer values")
        numeric = [record["clean_logit_difference"], record["corrupted_logit_difference"]]
        numeric.extend(record["patched_logit_differences"])
        if not np.isfinite(np.asarray(numeric, dtype=float)).all():
            raise RuntimeError(f"{dataset}/{record['example_id']}: non-finite logit score")
    return records


def _plot(summary: dict[str, Any], dataset: str, run_id: str, path: Path) -> None:
    layers = np.arange(len(summary["mean_faithfulness_by_layer"]))
    mean = np.asarray(summary["mean_faithfulness_by_layer"])
    low = np.asarray(summary["bootstrap_95ci_low"])
    high = np.asarray(summary["bootstrap_95ci_high"])
    raw = np.asarray(summary["mean_raw_patch_effect_by_layer"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(layers, mean, marker="o", linewidth=1.5, label="Mean normalized recovery")
    axes[0].fill_between(layers, low, high, alpha=0.25, label="95% paired bootstrap CI")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axhline(1, color="grey", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Faithfulness")
    axes[0].legend(loc="best")
    axes[0].set_title(f"{dataset.capitalize()} causal residual-stream patching")
    axes[1].bar(layers, raw, width=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Raw logit-difference recovery")
    axes[1].set_xlabel("Transformer layer (residual pre)")
    axes[1].set_xticks(layers)
    fig.suptitle(f"Run {run_id}", fontsize=9, y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_causal_report(config: CausalPatchingConfig, run_id: str) -> dict[str, Any]:
    combined: dict[str, Any] = {
        "summary_kind": "corrected_causal_patching",
        "run_id": run_id,
        "activation_run_id": config.activation_run_id,
        "datasets": {},
        "interpretation": "Instruction-conditioned first-token supportive-versus-neutral target behavior",
    }
    artifacts: list[dict[str, Any]] = []
    for dataset in config.datasets:
        records = load_and_verify_records(config, dataset, run_id)
        summary = summarize_layer_results(records, config)
        summary_path = config.output_root / dataset / f"layer_summary_{run_id}.json"
        chart_path = config.output_root / dataset / f"layer_patching_{run_id}.png"
        _write_json(summary_path, summary)
        _plot(summary, dataset, run_id, chart_path)
        combined["datasets"][dataset] = summary
        for path in (summary_path, chart_path):
            artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    summary_path = config.output_root / "metadata" / f"causal_patching_summary_{run_id}.json"
    _write_json(summary_path, combined)
    report_path = config.output_root / "metadata" / f"causal_patching_report_{run_id}.md"
    lines = [
        "# Causal activation-patching report", "", f"- Run: `{run_id}`",
        f"- Activation selection: `{config.activation_run_id}`", "- Samples: 200 motivation + 200 empathy",
        "- Intervention: clean residual-pre restoration into the neutral-instruction run, one layer at a time",
        "- Score: `logit(\" I\") - logit(\" Okay\")` at the first generated response position", "",
    ]
    for dataset in config.datasets:
        item = combined["datasets"][dataset]
        peak = int(np.nanargmax(item["mean_faithfulness_by_layer"]))
        lines.extend([
            f"## {dataset.capitalize()}", "",
            f"- Valid denominators: {item['valid_denominator_records']}/200",
            f"- Descriptive peak layer: {peak}",
            f"- Peak mean normalized recovery: {item['mean_faithfulness_by_layer'][peak]:.4f}",
            f"- Candidate layers under the prespecified rule: {item['candidate_layers']}", "",
        ])
    lines.extend([
        "## Interpretation boundary", "",
        "This experiment localizes model computation for an explicit supportive-versus-neutral style instruction and fixed opening-token contrast. It does not establish the clinical quality of a full response. Residual-layer locations are not yet an induced circuit; Wang-style completeness and minimality tests follow component/SAE refinement.", "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    artifacts.extend([
        {"path": str(summary_path), "bytes": summary_path.stat().st_size, "sha256": _sha256(summary_path)},
        {"path": str(report_path), "bytes": report_path.stat().st_size, "sha256": _sha256(report_path)},
    ])
    manifest = {"run_id": run_id, "artifacts": artifacts}
    manifest_path = config.output_root / "metadata" / f"causal_patching_artifacts_{run_id}.json"
    _write_json(manifest_path, manifest)
    return combined
