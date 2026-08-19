"""Streaming, task-separated SAE analysis over captured residual activations.

The encoder cache is sparse JSONL: dense token-by-feature tensors exist only for
one small token chunk on the accelerator and are never written to disk.  The
statistical phase uses the sparse per-example aggregates and reloads the much
smaller residual vectors for the matched random-neuron baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open
from scipy import sparse

from .probe import CaptureRecords, load_capture_records, make_grouped_split


AGGREGATIONS = ("final_valid_response_token", "mean_response_token_activation")


@dataclass(frozen=True)
class SAEAnalysisConfig:
    model: str
    sae_release: str
    sae_id: str
    repository: str
    revision: str
    layer: int
    hook_name: str
    activation_key: str
    d_in: int
    d_sae: int
    tasks: tuple[str, ...]
    bootstrap_resamples: int
    random_neuron_baseline_repetitions: int
    seeds: dict[str, int]
    local_sae_path: Path
    top_feature_count: int = 50
    top_example_count: int = 10
    encode_token_batch_size: int = 16

    @classmethod
    def load(cls, project_root: Path, path: Path) -> "SAEAnalysisConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        local_path = (
            project_root
            / ".hf-sae"
            / "l16r_8x"
            / "Llama3_1-8B-Base-L16R-8x"
        )
        return cls(
            model=str(raw.get("model", "meta-llama/Llama-3.1-8B")),
            sae_release=str(raw.get("sae_release", "llama_scope_lxr_8x")),
            sae_id=str(raw.get("sae_id", "l16r_8x")),
            repository=str(raw.get("repository", "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x")),
            revision=str(raw["revision"]),
            layer=int(raw["layer"]),
            hook_name=str(raw["hook_name"]),
            activation_key=str(raw["activation_key"]),
            d_in=int(raw["d_in"]),
            d_sae=int(raw["d_sae"]),
            tasks=tuple(raw["tasks"]),
            bootstrap_resamples=int(raw["bootstrap_resamples"]),
            random_neuron_baseline_repetitions=int(
                raw["random_neuron_baseline_repetitions"]
            ),
            seeds={str(key): int(value) for key, value in raw["seeds"].items()},
            local_sae_path=local_path,
            top_feature_count=int(raw.get("top_feature_count", 50)),
            top_example_count=int(raw.get("top_example_count", 10)),
            encode_token_batch_size=int(raw.get("encode_token_batch_size", 16)),
        )

    def validate_local_checkpoint(self) -> None:
        required = (
            self.local_sae_path / "hyperparams.json",
            self.local_sae_path / "lm_config.json",
            self.local_sae_path / "checkpoints" / "final.safetensors",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Pinned SAE is not fully available locally; downloads are disabled: "
                + ", ".join(missing)
            )
        hyperparameters = json.loads(required[0].read_text(encoding="utf-8"))
        expected = {
            "hook_point_in": self.hook_name,
            "d_model": self.d_in,
            "d_sae": self.d_sae,
        }
        actual = {key: hyperparameters.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"Local SAE metadata mismatch: expected {expected}, found {actual}")


def _pairs(indices: np.ndarray, values: np.ndarray) -> dict[str, list[Any]]:
    return {
        "indices": indices.astype(np.int32, copy=False).tolist(),
        "values": values.astype(np.float32, copy=False).tolist(),
    }


def sparse_encode_tokens(
    sae: Any,
    residual: torch.Tensor,
    *,
    token_batch_size: int,
) -> dict[str, Any]:
    """Encode tokens in chunks and retain only sparse aggregate statistics."""
    if residual.ndim != 2:
        raise ValueError(f"Expected [tokens, d_in], got {tuple(residual.shape)}")
    token_count = int(residual.shape[0])
    if token_count == 0:
        raise ValueError("Cannot encode an empty activation sequence")

    device = next(sae.parameters()).device
    feature_sums: dict[int, float] = {}
    active_token_counts: dict[int, int] = {}
    final_indices = np.empty(0, dtype=np.int64)
    final_values = np.empty(0, dtype=np.float32)
    token_l0_sum = 0

    with torch.inference_mode():
        for start in range(0, token_count, token_batch_size):
            stop = min(start + token_batch_size, token_count)
            encoded = sae.encode(residual[start:stop].to(device))
            if encoded.ndim != 2:
                raise ValueError(f"SAE returned unexpected shape {tuple(encoded.shape)}")
            nonzero = torch.nonzero(encoded, as_tuple=False)
            token_l0_sum += int(nonzero.shape[0])
            if nonzero.numel():
                rows = nonzero[:, 0]
                features = nonzero[:, 1]
                values = encoded[rows, features].float()
                unique_features = torch.unique(features)
                for feature in unique_features.tolist():
                    mask = features == feature
                    feature_sums[feature] = feature_sums.get(feature, 0.0) + float(
                        values[mask].sum().item()
                    )
                    active_token_counts[feature] = active_token_counts.get(feature, 0) + int(
                        torch.unique(rows[mask]).numel()
                    )
            if start <= token_count - 1 < stop:
                final = encoded[token_count - 1 - start].float()
                final_nz = torch.nonzero(final, as_tuple=False).flatten()
                final_indices = final_nz.cpu().numpy()
                final_values = final[final_nz].cpu().numpy()
            del encoded

    mean_indices = np.asarray(sorted(feature_sums), dtype=np.int64)
    mean_values = np.asarray(
        [feature_sums[index] / token_count for index in mean_indices], dtype=np.float32
    )
    count_indices = np.asarray(sorted(active_token_counts), dtype=np.int64)
    count_values = np.asarray(
        [active_token_counts[index] for index in count_indices], dtype=np.int32
    )
    return {
        "token_count": token_count,
        "token_l0_sum": token_l0_sum,
        "final_valid_response_token": _pairs(final_indices, final_values),
        "mean_response_token_activation": _pairs(mean_indices, mean_values),
        "active_feature_token_counts": {
            "indices": count_indices.tolist(),
            "values": count_values.tolist(),
        },
    }


def _load_sae(config: SAEAnalysisConfig, device: str) -> Any:
    config.validate_local_checkpoint()
    hf_home = config.local_sae_path.parents[1]
    revision_ref = (
        hf_home / "hub" / "models--fnlp--Llama3_1-8B-Base-LXR-8x" / "refs" / "main"
    )
    if not revision_ref.is_file() or revision_ref.read_text(encoding="utf-8").strip() != config.revision:
        raise ValueError("The offline SAELens registry cache is not pinned to the configured revision")
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Import only after offline/cache variables are fixed: huggingface_hub reads
    # several of these flags at module-import time.
    from sae_lens import SAE

    sae = SAE.from_pretrained(
        release=config.sae_release,
        sae_id=config.sae_id,
        device=device,
        dtype="bfloat16",
    )
    if int(sae.cfg.d_in) != config.d_in or int(sae.cfg.d_sae) != config.d_sae:
        raise ValueError("Loaded SAE dimensions do not match the pinned analysis configuration")
    sae.eval()
    return sae


def _existing_example_ids(path: Path, expected_provenance: dict[str, Any]) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid resume record at {path}:{line_number}") from error
        actual_provenance = {key: record.get(key) for key in expected_provenance}
        if actual_provenance != expected_provenance:
            raise ValueError(
                f"Resume cache provenance mismatch at {path}:{line_number}: "
                f"expected {expected_provenance}, found {actual_provenance}"
            )
        example_id = str(record["example_id"])
        if example_id in completed:
            raise ValueError(f"Duplicate example in SAE cache: {example_id}")
        completed.add(example_id)
    return completed


def encode_dataset(
    project_root: Path,
    records: CaptureRecords,
    config: SAEAnalysisConfig,
    run_id: str,
    sae: Any,
) -> Path:
    output = (
        project_root / "results" / "sae" / records.dataset / "cache" / f"{run_id}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "sae_id": config.sae_id,
        "sae_revision": config.revision,
        "activation_key": config.activation_key,
    }
    completed = _existing_example_ids(output, provenance)
    mode = "a" if output.exists() else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for ordinal, (example_id, path) in enumerate(
            zip(records.example_ids, records.residual_paths, strict=True)
        ):
            if example_id in completed:
                continue
            with safe_open(path, framework="pt", device="cpu") as tensors:
                if config.activation_key not in tensors.keys():
                    raise ValueError(f"{path} does not contain {config.activation_key}")
                residual = tensors.get_tensor(config.activation_key).float()
            if residual.shape[1] != config.d_in:
                raise ValueError(f"Unexpected residual shape in {path}: {tuple(residual.shape)}")
            # The capture is response-only but contains the tokenizer's BOS token.
            # Match the probing sensitivity definition by excluding BOS when present.
            response_residual = residual[1:] if residual.shape[0] > 1 else residual
            encoded = sparse_encode_tokens(
                sae, response_residual, token_batch_size=config.encode_token_batch_size
            )
            row = {
                "ordinal": ordinal,
                "dataset": records.dataset,
                "example_id": example_id,
                "label": int(records.labels[ordinal]),
                "conversation_id": str(records.groups[ordinal]),
                **provenance,
                **encoded,
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return output


def load_sparse_cache(
    path: Path, aggregation: str, d_sae: int
) -> tuple[list[dict[str, Any]], sparse.csr_matrix]:
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in rows:
        vector = row[aggregation]
        row_indices = [int(value) for value in vector["indices"]]
        if any(value < 0 or value >= d_sae for value in row_indices):
            raise ValueError("Sparse feature index outside configured SAE dimensions")
        indices.extend(row_indices)
        data.extend(float(value) for value in vector["values"])
        indptr.append(len(data))
    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), indices, indptr),
        shape=(len(rows), d_sae),
        dtype=np.float32,
    )
    return rows, matrix


def group_label_cell_matrix(
    values: np.ndarray | sparse.csr_matrix,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Collapse samples to one equally weighted cell per conversation and label."""
    keys = sorted({(str(group), int(label)) for group, label in zip(groups, labels, strict=True)})
    cell_rows: list[Any] = []
    cell_labels: list[int] = []
    cell_groups: list[str] = []
    for group, label in keys:
        selected = (groups.astype(str) == group) & (labels == label)
        cell_rows.append(values[selected].mean(axis=0))
        cell_labels.append(label)
        cell_groups.append(group)
    if sparse.issparse(values):
        cells = sparse.vstack([sparse.csr_matrix(row) for row in cell_rows], format="csr")
    else:
        cells = np.vstack(cell_rows).astype(np.float64, copy=False)
    return cells, np.asarray(cell_labels, dtype=np.int8), np.asarray(cell_groups, dtype=object)


def standardized_mean_difference(
    values: np.ndarray | sparse.csr_matrix, labels: np.ndarray
) -> np.ndarray:
    positive = values[labels == 1]
    negative = values[labels == 0]
    if positive.shape[0] < 2 or negative.shape[0] < 2:
        raise ValueError("Both classes need at least two units for standardized mean difference")

    def moments(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        squared_mean = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel() if sparse.issparse(matrix) else np.mean(matrix * matrix, axis=0)
        variance = np.maximum(squared_mean - mean * mean, 0.0)
        variance *= matrix.shape[0] / (matrix.shape[0] - 1)
        return mean, variance

    positive_mean, positive_variance = moments(positive)
    negative_mean, negative_variance = moments(negative)
    denominator = positive.shape[0] + negative.shape[0] - 2
    pooled = np.sqrt(
        ((positive.shape[0] - 1) * positive_variance + (negative.shape[0] - 1) * negative_variance)
        / denominator
    )
    difference = positive_mean - negative_mean
    return np.divide(difference, pooled, out=np.zeros_like(difference), where=pooled > 0)


def grouped_smd(
    values: np.ndarray | sparse.csr_matrix, labels: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    cells, cell_labels, _ = group_label_cell_matrix(values, labels, groups)
    return standardized_mean_difference(cells, cell_labels)


def grouped_bootstrap_selected(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> np.ndarray:
    """Cluster bootstrap conversations, returning SMDs for selected features."""
    cells, cell_labels, cell_groups = group_label_cell_matrix(values, labels, groups)
    cells = np.asarray(cells)
    unique_groups = np.unique(cell_groups)
    rng = np.random.default_rng(seed)
    output = np.empty((resamples, cells.shape[1]), dtype=np.float32)
    for repetition in range(resamples):
        # An ordinary cluster bootstrap can occasionally omit a rare class. Reject
        # only those undefined draws rather than silently changing to row sampling.
        for _attempt in range(10_000):
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            chosen = np.concatenate([np.flatnonzero(cell_groups == group) for group in sampled])
            counts = np.bincount(cell_labels[chosen], minlength=2)
            if np.all(counts >= 2):
                break
        else:
            raise ValueError("Could not draw a grouped bootstrap sample containing both classes")
        # Duplicate clusters are intentionally retained by concatenation.
        output[repetition] = standardized_mean_difference(cells[chosen], cell_labels[chosen])
    return output


def matched_random_neuron_baseline(
    residual_smd: np.ndarray,
    selected_feature_smd: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    count = len(selected_feature_smd)
    if count > len(residual_smd):
        raise ValueError("Cannot match more SAE features than available residual neurons")
    rng = np.random.default_rng(seed)
    mean_null = np.empty(repetitions, dtype=np.float32)
    max_null = np.empty(repetitions, dtype=np.float32)
    absolute = np.abs(residual_smd)
    for repetition in range(repetitions):
        chosen = rng.choice(len(absolute), size=count, replace=False)
        mean_null[repetition] = float(absolute[chosen].mean())
        max_null[repetition] = float(absolute[chosen].max())
    observed_mean = float(np.abs(selected_feature_smd).mean())
    observed_max = float(np.abs(selected_feature_smd).max())
    return {
        "definition": "same-count uniformly sampled layer residual neurons",
        "repetitions": repetitions,
        "selected_count": count,
        "observed_mean_absolute_smd": observed_mean,
        "observed_max_absolute_smd": observed_max,
        "null_mean_absolute_smd_quantiles": np.quantile(mean_null, [0.025, 0.5, 0.975]).tolist(),
        "null_max_absolute_smd_quantiles": np.quantile(max_null, [0.025, 0.5, 0.975]).tolist(),
        "mean_empirical_p": float((1 + np.sum(mean_null >= observed_mean)) / (repetitions + 1)),
        "max_empirical_p": float((1 + np.sum(max_null >= observed_max)) / (repetitions + 1)),
    }


def _load_residual_aggregates(records: CaptureRecords, activation_key: str) -> dict[str, np.ndarray]:
    output = {name: np.empty((len(records.example_ids), 4096), dtype=np.float32) for name in AGGREGATIONS}
    for index, path in enumerate(records.residual_paths):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            activation = tensors.get_tensor(activation_key).float().numpy()
        output["final_valid_response_token"][index] = activation[-1]
        response = activation[1:] if len(activation) > 1 else activation
        output["mean_response_token_activation"][index] = response.mean(axis=0)
    return output


def analyze_dataset(
    project_root: Path,
    records: CaptureRecords,
    config: SAEAnalysisConfig,
    run_id: str,
) -> dict[str, Any]:
    cache_path = project_root / "results" / "sae" / records.dataset / "cache" / f"{run_id}.jsonl"
    split = make_grouped_split(
        records.labels,
        records.groups,
        ratios=(0.70, 0.15, 0.15),
        seed=config.seeds[records.dataset],
        candidate_assignments=100_000,
    )
    residuals = _load_residual_aggregates(records, config.activation_key)
    task_result: dict[str, Any] = {
        "dataset": records.dataset,
        "run_id": run_id,
        "layer": config.layer,
        "activation_key": config.activation_key,
        "sae": {
            "model": config.model,
            "release": config.sae_release,
            "sae_id": config.sae_id,
            "repository": config.repository,
            "revision": config.revision,
            "framework": "SAELens",
            "local_only": True,
            "d_in": config.d_in,
            "d_sae": config.d_sae,
        },
        "feature_selection_population": "training split only; conversation-label cell weighted",
        "interpretability_scores": "pending_blinded_human_scoring",
        "aggregations": {},
    }
    for aggregation_index, aggregation in enumerate(AGGREGATIONS):
        rows, features = load_sparse_cache(cache_path, aggregation, config.d_sae)
        if len(rows) != len(records.example_ids):
            raise ValueError(
                f"Incomplete {records.dataset} SAE cache: expected {len(records.example_ids)}, "
                f"found {len(rows)}"
            )
        if [row["example_id"] for row in rows] != list(records.example_ids):
            raise ValueError("SAE cache order does not match capture metadata")
        train_features = features[split.train]
        train_labels = records.labels[split.train]
        train_groups = records.groups[split.train]
        effects = grouped_smd(train_features, train_labels, train_groups)
        top_count = min(config.top_feature_count, int(np.count_nonzero(np.isfinite(effects))))
        selected = np.argsort(-np.abs(effects), kind="stable")[:top_count]
        selected_values = train_features[:, selected].toarray()
        bootstrap = grouped_bootstrap_selected(
            selected_values,
            train_labels,
            train_groups,
            resamples=config.bootstrap_resamples,
            seed=config.seeds[records.dataset] + aggregation_index * 10_000,
        )
        residual_effects = grouped_smd(
            residuals[aggregation][split.train], train_labels, train_groups
        )
        baseline = matched_random_neuron_baseline(
            residual_effects,
            effects[selected],
            repetitions=config.random_neuron_baseline_repetitions,
            seed=config.seeds[records.dataset] + 50_000 + aggregation_index,
        )
        example_frequency = np.asarray(features.getnnz(axis=0)).ravel() / features.shape[0]
        selected_all = features[:, selected].toarray()
        token_active_counts = np.zeros(config.d_sae, dtype=np.int64)
        total_response_tokens = 0
        for row in rows:
            total_response_tokens += int(row["token_count"])
            counts = row["active_feature_token_counts"]
            token_active_counts[np.asarray(counts["indices"], dtype=np.int64)] += np.asarray(
                counts["values"], dtype=np.int64
            )
        feature_rows = []
        for rank, feature_id in enumerate(selected):
            order = np.argsort(-selected_all[:, rank], kind="stable")[: config.top_example_count]
            feature_rows.append(
                {
                    "rank": rank + 1,
                    "feature_id": int(feature_id),
                    "standardized_mean_difference": float(effects[feature_id]),
                    "bootstrap_ci_95": np.quantile(bootstrap[:, rank], [0.025, 0.975]).tolist(),
                    "example_activation_frequency": float(example_frequency[feature_id]),
                    "token_activation_frequency": float(
                        token_active_counts[feature_id] / total_response_tokens
                    ),
                    "interpretability_score": None,
                    "top_activating_examples": [
                        {
                            "example_id": records.example_ids[index],
                            "label": int(records.labels[index]),
                            "conversation_id": str(records.groups[index]),
                            "activation": float(selected_all[index, rank]),
                        }
                        for index in order
                    ],
                }
            )
        task_result["aggregations"][aggregation] = {
            "feature_count": config.d_sae,
            "selected_feature_count": top_count,
            "features": feature_rows,
            "random_neuron_baseline": baseline,
        }

    cache_rows = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
    total_tokens = sum(int(row["token_count"]) for row in cache_rows)
    total_active = sum(int(row["token_l0_sum"]) for row in cache_rows)
    task_result["activation_sparsity"] = {
        "tokens": total_tokens,
        "active_feature_events": total_active,
        "mean_l0_features_per_token": total_active / total_tokens,
        "global_nonzero_fraction": total_active / (total_tokens * config.d_sae),
    }
    return task_result


def write_outputs(project_root: Path, result: dict[str, Any]) -> tuple[Path, Path, Path]:
    import matplotlib.pyplot as plt

    dataset = result["dataset"]
    run_id = result["run_id"]
    output_dir = project_root / "results" / "sae" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"sae_analysis_{run_id}.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    chart_path = output_dir / f"sae_top_features_{run_id}.png"
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for axis, aggregation in zip(axes, AGGREGATIONS, strict=True):
        rows = result["aggregations"][aggregation]["features"][:20]
        axis.bar([str(row["feature_id"]) for row in rows], [row["standardized_mean_difference"] for row in rows])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(aggregation.replace("_", " "))
        axis.set_ylabel("grouped train-only SMD")
        axis.tick_params(axis="x", rotation=60)
    figure.suptitle(f"{dataset.title()} — layer {result['layer']} SAE features")
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    report_path = output_dir / f"sae_report_{run_id}.md"
    lines = [
        f"# SAE analysis: {dataset}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Layer: `{result['layer']}` (`{result['activation_key']}`)",
        f"- Mean activation L0: {result['activation_sparsity']['mean_l0_features_per_token']:.3f}",
        "- Tasks were encoded and analysed independently.",
        "- Feature selection used the grouped training split only; validation/test labels were not used.",
        "- Interpretability scores remain pending blinded human review of the recorded top examples.",
        "",
    ]
    for aggregation in AGGREGATIONS:
        section = result["aggregations"][aggregation]
        baseline = section["random_neuron_baseline"]
        lines.extend(
            [
                f"## {aggregation.replace('_', ' ').title()}",
                "",
                "Top feature IDs: " + ", ".join(str(row["feature_id"]) for row in section["features"][:10]),
                f"Matched random-neuron mean-effect p: `{baseline['mean_empirical_p']:.6f}`",
                f"Matched random-neuron max-effect p: `{baseline['max_empirical_p']:.6f}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, chart_path, report_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/sae_analysis.json"))
    parser.add_argument("--activation-run-id", default="20260817T181608Z")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=("encode", "analyze", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    root = arguments.project_root.resolve()
    config_path = arguments.config if arguments.config.is_absolute() else root / arguments.config
    config = SAEAnalysisConfig.load(root, config_path)
    records_by_task = {
        task: load_capture_records(root, task, arguments.activation_run_id) for task in config.tasks
    }
    if arguments.phase in ("encode", "all"):
        sae = _load_sae(config, arguments.device)
        for task in config.tasks:
            encode_dataset(root, records_by_task[task], config, arguments.run_id, sae)
    if arguments.phase in ("analyze", "all"):
        for task in config.tasks:
            result = analyze_dataset(root, records_by_task[task], config, arguments.run_id)
            write_outputs(root, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
