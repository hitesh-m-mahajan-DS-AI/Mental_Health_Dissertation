"""Reproducibility metadata helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    names = [
        "torch",
        "transformers",
        "transformer-lens",
        "sae-lens",
        "scikit-learn",
        "pandas",
        "numpy",
        "safetensors",
        "accelerate",
        "psutil",
        "pytest",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_metadata() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "system_memory_total_bytes": int(vm.total),
        "system_memory_available_bytes_at_start": int(vm.available),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        result["cuda"] = {
            "device": 0,
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "memory_total_bytes": int(total),
            "memory_free_bytes_at_start": int(free),
        }
    try:
        result["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = None
    return result


def model_file_metadata(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    return {
        "model_path": str(model_path),
        "model_revision": None,
        "revision_note": "No local VCS/Hugging Face revision metadata was found in the checkpoint directory.",
        "config_sha256": sha256_file(config_path),
        "tokenizer_config_sha256": sha256_file(tokenizer_config_path),
        "weight_index_sha256": sha256_file(index_path),
        "weight_index_total_size_bytes": int(index.get("metadata", {}).get("total_size", 0)),
        "weight_shards": [
            {"name": name, "size_bytes": int((model_path / name).stat().st_size)} for name in shard_names
        ],
    }


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
