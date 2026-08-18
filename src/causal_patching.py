"""Layerwise residual-stream causal activation patching for local Llama."""

from __future__ import annotations

import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .causal_config import CausalPatchingConfig
from .causal_pairs import build_pairs, manifest_record, validate_and_tokenize_pair
from .config import CaptureConfig
from .metadata import new_run_id, sha256_file
from .model_loader import load_model, validate_local_checkpoint


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _capture_compatible_config(config: CausalPatchingConfig) -> CaptureConfig:
    return CaptureConfig(
        project_root=config.project_root,
        model_path=config.model_path,
        motivation_csv=config.motivation_csv,
        empathy_csv=config.empathy_csv,
        output_root=config.output_root,
        offload_dir=config.offload_dir,
        seed=config.bootstrap_seed,
        dtype=config.dtype,
        device_map=config.device_map,
        max_memory=config.max_memory,
        max_length=config.max_length,
        max_examples_per_dataset=200,
        sampling_strategy="uniform_random_without_replacement",
        sampling_seeds={"motivation": 42, "empathy": 43},
        token_positions="all",
        add_special_tokens=True,
        input_format="raw_response",
        capture_types=("residual_pre",),
        dataset_order=config.datasets,
        smoke_test=False,
    )


def _model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def _scores(
    logits: torch.Tensor, lengths: list[int], supportive_id: int, neutral_id: int
) -> list[float]:
    return [
        float((logits[index, length - 1, supportive_id] - logits[index, length - 1, neutral_id]).float().cpu())
        for index, length in enumerate(lengths)
    ]


@contextmanager
def _capture_residual_pre(model: Any) -> Iterator[list[torch.Tensor]]:
    cache: list[torch.Tensor] = []
    handles = []
    for layer in model.model.layers:
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            cache.append(args[0].detach().to("cpu", copy=True))
        handles.append(layer.register_forward_pre_hook(hook))
    try:
        yield cache
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _patch_residual_pre(model: Any, layer_index: int, clean: torch.Tensor) -> Iterator[None]:
    layer = model.model.layers[layer_index]

    def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        replacement = clean.to(device=args[0].device, dtype=args[0].dtype)
        return (replacement,) + args[1:]

    handle = layer.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _forward(model: Any, input_ids: list[list[int]], pad_token_id: int) -> torch.Tensor:
    lengths = [len(ids) for ids in input_ids]
    width = max(lengths)
    padded = [ids + [pad_token_id] * (width - len(ids)) for ids in input_ids]
    masks = [[1] * length + [0] * (width - length) for length in lengths]
    device = _model_input_device(model)
    tensor = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    with torch.inference_mode():
        return model(input_ids=tensor, attention_mask=attention_mask, use_cache=False).logits


def patch_batch(
    model: Any,
    encoded_batch: list[dict[str, Any]],
    num_layers: int,
    epsilon: float,
    pad_token_id: int,
) -> list[dict[str, Any]]:
    supportive_id = encoded_batch[0]["supportive_target_id"]
    neutral_id = encoded_batch[0]["neutral_target_id"]
    if any(e["supportive_target_id"] != supportive_id or e["neutral_target_id"] != neutral_id for e in encoded_batch):
        raise ValueError("All examples in a batch must use the same target IDs")
    clean_ids = [e["clean_input_ids"] for e in encoded_batch]
    corrupted_ids = [e["corrupted_input_ids"] for e in encoded_batch]
    clean_lengths = [len(ids) for ids in clean_ids]
    corrupted_lengths = [len(ids) for ids in corrupted_ids]
    with _capture_residual_pre(model) as clean_cache:
        clean_scores = _scores(
            _forward(model, clean_ids, pad_token_id), clean_lengths, supportive_id, neutral_id
        )
    if len(clean_cache) != num_layers:
        raise RuntimeError(f"Captured {len(clean_cache)} residuals, expected {num_layers}")
    corrupted_scores = _scores(
        _forward(model, corrupted_ids, pad_token_id), corrupted_lengths, supportive_id, neutral_id
    )
    patched_by_layer: list[list[float]] = []
    for layer_index in range(num_layers):
        with _patch_residual_pre(model, layer_index, clean_cache[layer_index]):
            logits = _forward(model, corrupted_ids, pad_token_id)
        patched_by_layer.append(
            _scores(logits, corrupted_lengths, supportive_id, neutral_id)
        )
    results = []
    for batch_index, (clean_score, corrupted_score) in enumerate(zip(clean_scores, corrupted_scores)):
        patched_scores = [layer[batch_index] for layer in patched_by_layer]
        denominator = clean_score - corrupted_score
        results.append({
            "clean_logit_difference": clean_score,
            "corrupted_logit_difference": corrupted_score,
            "clean_minus_corrupted": denominator,
            "patched_logit_differences": patched_scores,
            "faithfulness": [
                None if abs(denominator) <= epsilon else (score - corrupted_score) / denominator
                for score in patched_scores
            ],
            "denominator_valid": abs(denominator) > epsilon,
        })
    return results


def patch_one_pair(model: Any, encoded: dict[str, Any], num_layers: int, epsilon: float, pad_token_id: int = 128009) -> dict[str, Any]:
    """Compatibility wrapper used by focused tests and diagnostics."""
    return patch_batch(model, [encoded], num_layers, epsilon, pad_token_id)[0]


def _write_pair_manifest(config: CausalPatchingConfig, tokenizer: Any, dataset: str, run_id: str) -> dict[str, Any]:
    pairs = build_pairs(config, dataset)
    path = config.output_root / dataset / "metadata" / f"pairs_{run_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    truncated = 0
    placeholders = 0
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            encoded = validate_and_tokenize_pair(pair, tokenizer, config)
            truncated += int(encoded["truncated"])
            placeholders += int(pair.context == "The client has not provided any preceding message.")
            record = manifest_record(pair)
            record.update({
                "token_length": len(encoded["clean_input_ids"]),
                "truncated": encoded["truncated"],
                "supportive_target_id": encoded["supportive_target_id"],
                "neutral_target_id": encoded["neutral_target_id"],
            })
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "path": str(path), "sha256": sha256_file(path), "records": len(pairs),
        "truncated": truncated, "missing_context_placeholders": placeholders,
    }


def summarize_layer_results(records: list[dict[str, Any]], config: CausalPatchingConfig) -> dict[str, Any]:
    valid = [r for r in records if r["denominator_valid"]]
    if not valid:
        raise ValueError("No pair has a valid clean-minus-corrupted denominator")
    values = np.asarray([r["faithfulness"] for r in valid], dtype=np.float64)
    effects = np.asarray([
        np.asarray(r["patched_logit_differences"]) - float(r["corrupted_logit_difference"])
        for r in valid
    ])
    rng = np.random.default_rng(config.bootstrap_seed)
    means = values.mean(axis=0)
    boot = np.empty((config.bootstrap_resamples, values.shape[1]), dtype=np.float64)
    for index in range(config.bootstrap_resamples):
        sample = rng.integers(0, len(values), size=len(values))
        boot[index] = values[sample].mean(axis=0)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    candidates = [
        int(i) for i in range(values.shape[1])
        if effects[:, i].mean() > 0 and low[i] > 0
    ]
    return {
        "records": len(records),
        "valid_denominator_records": len(valid),
        "invalid_denominator_records": len(records) - len(valid),
        "mean_faithfulness_by_layer": means.tolist(),
        "bootstrap_95ci_low": low.tolist(),
        "bootstrap_95ci_high": high.tolist(),
        "mean_raw_patch_effect_by_layer": effects.mean(axis=0).tolist(),
        "candidate_layers": candidates,
        "candidate_layer_rule": config.candidate_layer_rule,
    }


def run_causal_patching(
    config: CausalPatchingConfig,
    inspect_only: bool = False,
    smoke_test: bool = False,
    resume_run_id: str | None = None,
) -> str:
    run_id = resume_run_id or new_run_id()
    capture_config = _capture_compatible_config(config)
    hf_config, tokenizer, tokenizer_info = validate_local_checkpoint(capture_config)
    manifests = {
        dataset: _write_pair_manifest(config, tokenizer, dataset, run_id)
        for dataset in config.datasets
    }
    metadata_path = config.output_root / "metadata" / f"run_{run_id}.json"
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "status": "inspection_complete" if inspect_only else "running",
        "activation_run_id": config.activation_run_id,
        "method": "controlled_instruction_counterfactual_residual_stream_patching",
        "model": str(config.model_path),
        "num_hidden_layers": int(hf_config.num_hidden_layers),
        "tokenizer": tokenizer_info,
        "pair_manifests": manifests,
        "supportive_target": config.supportive_target,
        "neutral_target": config.neutral_target,
        "faithfulness_equation": config.faithfulness_equation,
        "limitations": [
            "The intervention operationalizes supportiveness through an explicit style instruction and fixed opening-token targets.",
            "It measures instruction-conditioned first-token behavior, not the quality of a complete generated counselling response.",
        ],
    }
    _write_json(metadata_path, metadata)
    if inspect_only:
        return run_id

    model = load_model(capture_config)
    limit = config.smoke_test_pairs_per_dataset if smoke_test else 200
    for dataset in config.datasets:
        pairs = build_pairs(config, dataset)[:limit]
        result_dir = config.output_root / dataset / "pairs" / run_id
        result_dir.mkdir(parents=True, exist_ok=True)
        pending = [
            pair for pair in pairs
            if not (result_dir / f"{pair.selection_ordinal:03d}_{pair.example_id}.json").is_file()
        ]
        for start in range(0, len(pending), config.batch_size):
            pair_batch = pending[start:start + config.batch_size]
            encoded_batch = [validate_and_tokenize_pair(pair, tokenizer, config) for pair in pair_batch]
            batch_results = patch_batch(
                model, encoded_batch, int(hf_config.num_hidden_layers),
                config.zero_denominator_epsilon, int(tokenizer.eos_token_id)
            )
            for pair, encoded, result in zip(pair_batch, encoded_batch, batch_results):
                result.update({
                    "dataset": dataset,
                    "selection_ordinal": pair.selection_ordinal,
                    "example_id": pair.example_id,
                    "source_row": pair.source_row,
                    "token_length": len(encoded["clean_input_ids"]),
                    "supportive_target_id": encoded["supportive_target_id"],
                    "neutral_target_id": encoded["neutral_target_id"],
                })
                path = result_dir / f"{pair.selection_ordinal:03d}_{pair.example_id}.json"
                _write_json(path, result)
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(result_dir.glob("*.json"))]
        _write_json(config.output_root / dataset / f"layer_summary_{run_id}.json", summarize_layer_results(records, config))
    metadata["status"] = "smoke_test_complete" if smoke_test else "layer_patching_complete"
    _write_json(metadata_path, metadata)
    return run_id
