"""Chunk-safe safetensors storage and integrity manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .activation_hooks import activation_group
from .dataset_loader import Example
from .metadata import sha256_file


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


class DatasetWriter:
    def __init__(
        self, output_root: Path, dataset: str, run_id: str, resume: bool = False
    ) -> None:
        self.root = output_root / dataset
        self.run_id = run_id
        for name in ["residual_stream", "attention", "mlp", "metadata"]:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / "metadata" / f"{run_id}_examples.jsonl"
        self.records: list[dict[str, Any]] = []
        if resume and self.metadata_path.is_file():
            self.records = [
                json.loads(line)
                for line in self.metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def write_example(
        self,
        ordinal: int,
        example: Example,
        activations: dict[str, torch.Tensor],
        input_ids: list[int],
        tokens: list[str],
        token_positions: list[int],
        was_truncated: bool,
    ) -> dict[str, Any]:
        safe_id = f"{self.run_id}_example_{ordinal:06d}_row_{example.source_row}"
        grouped: dict[str, dict[str, torch.Tensor]] = {
            "residual_stream": {},
            "attention": {},
            "mlp": {},
        }
        for key, tensor in activations.items():
            activation_type = key.split(".", 1)[1]
            grouped[activation_group(activation_type)][key] = tensor

        output_files: list[dict[str, Any]] = []
        for group, tensors in grouped.items():
            if not tensors:
                continue
            path = self.root / group / f"{safe_id}.safetensors"
            metadata = {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "conversation_id": example.conversation_id,
                "utterance_id": example.utterance_id,
                "label": str(example.label),
                "run_id": self.run_id,
                "activation_group": group,
            }
            save_file(tensors, path, metadata=metadata)
            output_files.append(
                {
                    "activation_group": group,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                    "tensors": {
                        key: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                        for key, tensor in tensors.items()
                    },
                }
            )

        record = {
            "ordinal": ordinal,
            "dataset": example.dataset,
            "example_id": example.example_id,
            "source_row": example.source_row,
            "conversation_id": example.conversation_id,
            "utterance_id": example.utterance_id,
            "label": example.label,
            "response_sha256": hashlib.sha256(example.response.encode("utf-8")).hexdigest(),
            "input_ids": input_ids,
            "tokens": tokens,
            "captured_token_positions": token_positions,
            "was_truncated": was_truncated,
            "output_files": output_files,
        }
        with self.metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.records.append(record)
        return record

    def finalize(self, manifest: dict[str, Any]) -> Path:
        manifest["captured_examples"] = len(self.records)
        manifest["examples_metadata_path"] = str(self.metadata_path)
        manifest["examples_metadata_sha256"] = sha256_file(self.metadata_path)
        manifest["outputs"] = [output for record in self.records for output in record["output_files"]]
        run_manifest = self.root / f"manifest_{self.run_id}.json"
        _json_dump(run_manifest, manifest)
        _json_dump(self.root / "manifest.json", manifest)
        return run_manifest


def verify_safetensors_outputs(records: list[dict[str, Any]]) -> dict[str, Any]:
    checked_files = 0
    checked_tensors = 0
    checked_values = 0
    for record in records:
        for output in record["output_files"]:
            path = Path(output["path"])
            if sha256_file(path) != output["sha256"]:
                raise ValueError(f"Checksum mismatch after write: {path}")
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                if set(keys) != set(output["tensors"]):
                    raise ValueError(f"Tensor key mismatch after reload: {path}")
                for key in keys:
                    tensor = handle.get_tensor(key)
                    if list(tensor.shape) != output["tensors"][key]["shape"]:
                        raise ValueError(f"Tensor shape mismatch after reload: {path}:{key}")
                    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                        raise ValueError(f"Non-finite activation values after reload: {path}:{key}")
                    if torch.count_nonzero(tensor).item() == 0:
                        raise ValueError(f"All-zero activation tensor after reload: {path}:{key}")
                    checked_values += int(tensor.numel())
                    checked_tensors += 1
            checked_files += 1
    return {
        "verified_files": checked_files,
        "verified_tensors": checked_tensors,
        "verified_values": checked_values,
        "finite_values": True,
        "no_all_zero_tensors": True,
        "status": "passed",
    }
