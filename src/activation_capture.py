"""Orchestrate validation, local Llama loading, hooks, capture, and reports."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .activation_hooks import ActivationCollector
from .config import CaptureConfig
from .dataset_loader import DatasetBundle, load_both
from .metadata import (
    hardware_metadata,
    model_file_metadata,
    new_run_id,
    package_versions,
    sha256_file,
)
from .model_loader import (
    discover_hook_points,
    load_model,
    model_device_map,
    resolve_dtype,
    transformer_lens_compatibility,
    validate_local_checkpoint,
)
from .storage import DatasetWriter, verify_safetensors_outputs


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_population_manifest(output_root: Path, bundle: DatasetBundle) -> dict[str, Any]:
    path = output_root / bundle.name / "metadata" / "population_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in bundle.examples:
            record = {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "source_row": example.source_row,
                "conversation_id": example.conversation_id,
                "utterance_id": example.utterance_id,
                "label": example.label,
                "response_sha256": hashlib.sha256(example.response.encode("utf-8")).hexdigest(),
                "response_character_length": len(example.response),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": len(bundle.examples),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_examples(
    bundle: DatasetBundle,
    limit: int | None,
    strategy: str,
    seed: int,
) -> tuple[Any, ...]:
    """Select examples without changing labels or the underlying populations."""
    if limit is None:
        return bundle.examples
    if limit > len(bundle.examples):
        raise ValueError(
            f"Requested {limit} {bundle.name} examples from a population of {len(bundle.examples)}"
        )
    if strategy == "source_order":
        return bundle.examples[:limit]
    if strategy == "uniform_random_without_replacement":
        selected = random.Random(seed).sample(list(bundle.examples), limit)
        # Process in source order after random membership selection. This makes logs
        # easier to audit without changing which examples were randomly selected.
        return tuple(sorted(selected, key=lambda example: example.source_row))
    raise ValueError(f"Unsupported sampling strategy {strategy!r}")


def _write_selection_manifest(
    output_root: Path,
    bundle: DatasetBundle,
    examples: tuple[Any, ...],
    run_id: str,
    strategy: str,
    seed: int,
) -> dict[str, Any]:
    path = output_root / bundle.name / "metadata" / f"selection_{run_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ordinal, example in enumerate(examples):
            record = {
                "selection_ordinal": ordinal,
                "dataset": example.dataset,
                "example_id": example.example_id,
                "source_row": example.source_row,
                "conversation_id": example.conversation_id,
                "utterance_id": example.utterance_id,
                "label": example.label,
                "response_sha256": hashlib.sha256(example.response.encode("utf-8")).hexdigest(),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    label_counts = {
        str(label): sum(example.label == label for example in examples) for label in (0, 1)
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": len(examples),
        "strategy": strategy,
        "seed": seed,
        "label_counts": label_counts,
        "unique_conversations": len({example.conversation_id for example in examples}),
    }


def _storage_estimate(
    examples_by_dataset: dict[str, tuple[Any, ...]],
    tokenizer: Any,
    hf_config: Any,
    config: CaptureConfig,
) -> dict[str, Any]:
    head_dim = int(hf_config.hidden_size // hf_config.num_attention_heads)
    dimensions = {
        "residual_pre": int(hf_config.hidden_size),
        "residual_post": int(hf_config.hidden_size),
        "query": int(hf_config.num_attention_heads * head_dim),
        "key": int(hf_config.num_key_value_heads * head_dim),
        "value": int(hf_config.num_key_value_heads * head_dim),
        "head_output": int(hf_config.hidden_size),
        "attention_output": int(hf_config.hidden_size),
        "mlp_activation": int(hf_config.intermediate_size),
        "mlp_output": int(hf_config.hidden_size),
    }
    bytes_per_element = torch.tensor([], dtype=resolve_dtype(config.dtype)).element_size()
    bytes_per_token = (
        int(hf_config.num_hidden_layers)
        * sum(dimensions[name] for name in config.capture_types)
        * bytes_per_element
    )
    result: dict[str, Any] = {
        "bytes_per_captured_token": bytes_per_token,
        "tensor_payload_only": True,
        "note": "Estimate excludes safetensors headers, JSON metadata, and filesystem overhead.",
        "datasets": {},
    }
    for name, examples in examples_by_dataset.items():
        token_lengths = [
            len(
                tokenizer(
                    example.response,
                    add_special_tokens=config.add_special_tokens,
                    truncation=False,
                )["input_ids"]
            )
            for example in examples
        ]
        if config.token_positions == "all":
            captured_tokens = sum(min(length, config.max_length) for length in token_lengths)
        else:
            captured_tokens = len(token_lengths)
        estimated_bytes = captured_tokens * bytes_per_token
        result["datasets"][name] = {
            "examples": len(token_lengths),
            "tokens_before_truncation": sum(token_lengths),
            "estimated_captured_tokens": captured_tokens,
            "truncated_examples": sum(length > config.max_length for length in token_lengths),
            "maximum_tokens_before_truncation": max(token_lengths),
            "estimated_activation_bytes": estimated_bytes,
            "estimated_activation_gib": estimated_bytes / 1024**3,
        }
    result["total_estimated_activation_gib"] = sum(
        item["estimated_activation_gib"] for item in result["datasets"].values()
    )
    return result


def _human_report(metadata: dict[str, Any], manifests: dict[str, str]) -> str:
    config = metadata["model_config"]
    hardware = metadata["hardware"]
    cuda = hardware.get("cuda", {})
    lines = [
        "# Activation capture checkpoint report",
        "",
        f"- Run ID: `{metadata['run_id']}`",
        f"- Mode: `{'smoke test' if metadata['capture_config']['smoke_test'] else 'configured capture'}`",
        f"- Resumed after interruption: `{metadata.get('resumed', False)}`",
        f"- Model: `{metadata['model_files']['model_path']}` (`{config['architectures'][0]}`)",
        f"- Shape: {config['num_hidden_layers']} layers, hidden size {config['hidden_size']}, "
        f"{config['num_attention_heads']} attention heads, {config['num_key_value_heads']} KV heads",
        f"- Configured dtype: `{metadata['capture_config']['dtype']}`",
        f"- GPU: `{cuda.get('name', 'none')}`",
        f"- Instrumentation: direct Hugging Face module hooks; TransformerLens config compatibility "
        f"is `{metadata['transformer_lens_compatibility']['compatible']}`",
        f"- Hook points discovered: {len(metadata['hook_points'])}",
        "",
        "## Random selections",
        "",
    ]
    for name, selection in metadata["selection_manifests"].items():
        lines.append(
            f"- {name}: {selection['records']} examples; labels {selection['label_counts']}; "
            f"{selection['unique_conversations']} unique conversations; seed {selection['seed']}"
        )
    estimate = metadata["configured_selection_storage_estimate"]
    lines.extend(
        [
            "",
            "## Capture and verification",
            "",
            f"- Estimated tensor payload: {estimate['total_estimated_activation_gib']:.2f} GiB",
        ]
    )
    for name, path in manifests.items():
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        output_bytes = sum(item["size_bytes"] for item in manifest["outputs"])
        verification = manifest["verification"]
        lines.append(
            f"- {name}: {manifest['captured_examples']} examples; "
            f"{len(manifest['outputs'])} files; {output_bytes / 1024**3:.2f} GiB; "
            f"{verification['verified_tensors']} verified tensors"
        )
        lines.append(f"  - Manifest: `{path}`")
    lines.extend(
        [
            "",
            "## Methodological boundary",
            "",
            "This run captures the configured raw response with BOS and every token up to 64. "
            "It does not decide the later grouped train/validation/test ratios, causal corruption "
            "construction, behavioural logit definition, or SAE checkpoint/source.",
            "",
        ]
    )
    return "\n".join(lines)


def run_capture(
    config: CaptureConfig,
    inspect_only: bool = False,
    resume_run_id: str | None = None,
) -> None:
    _seed_everything(config.seed)
    run_id = resume_run_id or new_run_id()
    config.output_root.mkdir(parents=True, exist_ok=True)

    hf_config, tokenizer, tokenizer_info = validate_local_checkpoint(config)
    hook_points = discover_hook_points(hf_config)
    tl_compatibility = transformer_lens_compatibility(
        hf_config, resolve_dtype(config.dtype), config.max_length
    )
    bundles = load_both(config.motivation_csv, config.empathy_csv)
    population_manifests = {
        name: _write_population_manifest(config.output_root, bundle)
        for name, bundle in bundles.items()
    }
    full_population_storage_estimate = _storage_estimate(
        {name: bundle.examples for name, bundle in bundles.items()},
        tokenizer,
        hf_config,
        config,
    )
    selected_examples = {
        name: select_examples(
            bundle,
            config.max_examples_per_dataset,
            config.sampling_strategy,
            config.sampling_seeds[name],
        )
        for name, bundle in bundles.items()
    }
    selection_manifests = {
        name: _write_selection_manifest(
            config.output_root,
            bundle,
            selected_examples[name],
            run_id,
            config.sampling_strategy,
            config.sampling_seeds[name],
        )
        for name, bundle in bundles.items()
    }
    configured_selection_storage_estimate = _storage_estimate(
        selected_examples, tokenizer, hf_config, config
    )

    capture_config = {
        "seed": config.seed,
        "dtype": config.dtype,
        "device_map": config.device_map,
        "max_memory": config.max_memory,
        "max_length": config.max_length,
        "max_examples_per_dataset": config.max_examples_per_dataset,
        "sampling_strategy": config.sampling_strategy,
        "sampling_seeds": config.sampling_seeds,
        "token_positions": config.token_positions,
        "add_special_tokens": config.add_special_tokens,
        "input_format": config.input_format,
        "capture_types": list(config.capture_types),
        "dataset_order": list(config.dataset_order),
        "smoke_test": config.smoke_test,
    }
    run_metadata: dict[str, Any] = {
        "run_id": run_id,
        "status": "inspected" if inspect_only else "initializing_model",
        "resumed": resume_run_id is not None,
        "capture_config": capture_config,
        "model_files": model_file_metadata(config.model_path),
        "model_config": hf_config.to_dict(),
        "tokenizer": tokenizer_info,
        "software_versions": package_versions(),
        "hardware": hardware_metadata(),
        "transformer_lens_compatibility": tl_compatibility,
        "instrumentation": "direct_huggingface_pytorch_forward_hooks",
        "hook_points": hook_points,
        "dataset_audits": {name: bundle.audit for name, bundle in bundles.items()},
        "population_manifests": population_manifests,
        "selection_manifests": selection_manifests,
        "provenance_reference_files": {
            name: {
                "path": str(config.project_root / name),
                "present": (config.project_root / name).is_file(),
            }
            for name in [
                "AnnoMI-full.csv",
                "annomi_binary_corrected.csv",
                "annomi_manual_review_corrected.csv",
            ]
        },
        "full_population_storage_estimate": full_population_storage_estimate,
        "configured_selection_storage_estimate": configured_selection_storage_estimate,
        "unresolved_scientific_choices": [
            "The configured capture uses each raw counselor/therapist response with BOS; alternative conversational context was not specified.",
            "Every token is captured up to 64; the downstream probe token aggregation remains to be specified.",
            "Grouped train/validation/test ratios and seed for probing.",
            "Causal corruption construction and behavioural logit-difference definition.",
            "SAE source/checkpoint compatibility; sae-lens is not currently installed.",
            "The three AnnoMI provenance/reference CSVs named in the handover are not present in the workspace.",
        ],
    }
    run_metadata_path = config.output_root / "metadata" / f"run_{run_id}.json"
    _write_json(run_metadata_path, run_metadata)
    for name, bundle in bundles.items():
        dataset_manifest = dict(bundle.audit)
        dataset_manifest["population_manifest"] = population_manifests[name]
        _write_json(config.output_root / name / "metadata" / "dataset_manifest.json", dataset_manifest)

    print(f"Validated local checkpoint: {config.model_path}")
    print(
        f"Datasets: motivation={len(bundles['motivation'].examples)} binary rows; "
        f"empathy={len(bundles['empathy'].examples)} counselor rows"
    )
    for name in config.dataset_order:
        selection = selection_manifests[name]
        print(
            f"Selected {name}: {selection['records']} random examples, "
            f"labels={selection['label_counts']}, seed={selection['seed']}"
        )
    print(
        "Estimated configured tensor payload: "
        f"{configured_selection_storage_estimate['total_estimated_activation_gib']:.2f} GiB"
    )
    print(f"Discovered {len(hook_points)} hook points across {hf_config.num_hidden_layers} layers")
    if inspect_only:
        print(f"Inspection metadata: {run_metadata_path}")
        return

    print("Loading exact local Llama-3.1-8B weights (no quantization or model substitution)...")
    model = load_model(config)
    run_metadata["model_device_map"] = model_device_map(model)
    run_metadata["status"] = "capturing"
    _write_json(run_metadata_path, run_metadata)
    collector = ActivationCollector(model, config.capture_types)
    collector.register()
    manifest_paths: dict[str, str] = {}
    try:
        for dataset_name in config.dataset_order:
            bundle = bundles[dataset_name]
            examples = selected_examples[dataset_name]
            existing_manifest = (
                config.output_root / dataset_name / f"manifest_{run_id}.json"
            )
            if resume_run_id and existing_manifest.is_file():
                existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
                if (
                    existing.get("captured_examples") == len(examples)
                    and existing.get("verification", {}).get("status") == "passed"
                ):
                    manifest_paths[dataset_name] = str(existing_manifest)
                    print(
                        f"Skipping completed {dataset_name}: {len(examples)} examples "
                        "already captured and verified"
                    )
                    continue
                raise ValueError(
                    f"Existing {dataset_name} manifest is not a complete verified match: "
                    f"{existing_manifest}"
                )
            writer = DatasetWriter(
                config.output_root,
                dataset_name,
                run_id,
                resume=resume_run_id is not None,
            )
            completed_ids = {record["example_id"] for record in writer.records}
            if not completed_ids <= {example.example_id for example in examples}:
                raise ValueError(
                    f"Resume metadata for {dataset_name} contains examples outside the selection"
                )
            print(
                f"Capturing {len(examples) - len(completed_ids)} remaining {dataset_name} "
                f"example(s); {len(completed_ids)} already complete..."
            )
            for ordinal, example in enumerate(examples):
                if example.example_id in completed_ids:
                    continue
                full = tokenizer(
                    example.response,
                    add_special_tokens=config.add_special_tokens,
                    truncation=False,
                    return_attention_mask=True,
                )
                encoded = tokenizer(
                    example.response,
                    add_special_tokens=config.add_special_tokens,
                    truncation=True,
                    max_length=config.max_length,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                input_device = model.get_input_embeddings().weight.device
                collector.clear()
                with torch.inference_mode():
                    model(
                        input_ids=input_ids.to(input_device),
                        attention_mask=attention_mask.to(input_device),
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                valid_length = int(attention_mask.sum().item())
                activations = collector.pop(config.token_positions, valid_length)
                expected = int(hf_config.num_hidden_layers) * len(config.capture_types)
                if len(activations) != expected:
                    raise ValueError(f"Expected {expected} activations, captured {len(activations)}")
                ids = input_ids[0, :valid_length].tolist()
                positions = (
                    list(range(valid_length))
                    if config.token_positions == "all"
                    else [valid_length - 1]
                )
                writer.write_example(
                    ordinal=ordinal,
                    example=example,
                    activations=activations,
                    input_ids=ids,
                    tokens=tokenizer.convert_ids_to_tokens(ids),
                    token_positions=positions,
                    was_truncated=len(full["input_ids"]) > config.max_length,
                )
                print(
                    f"  {dataset_name} {ordinal + 1}/{len(examples)}: "
                    f"{example.example_id}, {valid_length} token(s), {len(activations)} tensors"
                )
            verification = verify_safetensors_outputs(writer.records)
            manifest = {
                "run_id": run_id,
                "dataset": dataset_name,
                "dataset_audit": bundle.audit,
                "capture_config": capture_config,
                "model_path": str(config.model_path),
                "model_weight_index_sha256": run_metadata["model_files"]["weight_index_sha256"],
                "tokenizer": tokenizer_info,
                "instrumentation": run_metadata["instrumentation"],
                "verification": verification,
            }
            path = writer.finalize(manifest)
            manifest_paths[dataset_name] = str(path)
    finally:
        collector.remove()

    run_metadata["status"] = "completed"
    run_metadata["dataset_manifests"] = manifest_paths
    _write_json(run_metadata_path, run_metadata)
    report_path = config.output_root / "metadata" / f"checkpoint_report_{run_id}.md"
    report_path.write_text(_human_report(run_metadata, manifest_paths), encoding="utf-8")
    print(f"Capture completed: {run_metadata_path}")
    print(f"Human-readable report: {report_path}")
