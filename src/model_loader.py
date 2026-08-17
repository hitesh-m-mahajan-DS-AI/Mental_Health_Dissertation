"""Load the exact local Llama-3.1-8B checkpoint without substitution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import CaptureConfig


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(mapping)}")
    return mapping[name]


def validate_local_checkpoint(config: CaptureConfig) -> tuple[Any, Any, dict[str, Any]]:
    required = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    missing = [name for name in required if not (config.model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Local model checkpoint is incomplete; missing {missing}")

    hf_config = AutoConfig.from_pretrained(config.model_path, local_files_only=True)
    if hf_config.model_type != "llama" or hf_config.architectures != ["LlamaForCausalLM"]:
        raise ValueError(
            f"Expected LlamaForCausalLM, found model_type={hf_config.model_type}, "
            f"architectures={hf_config.architectures}"
        )
    expected = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32}
    actual = {key: int(getattr(hf_config, key)) for key in expected}
    if actual != expected:
        raise ValueError(f"Checkpoint does not match expected Llama-3.1-8B dimensions: {actual}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    if tokenizer.vocab_size != 128000 or len(tokenizer) != int(hf_config.vocab_size):
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: tokenizer base={tokenizer.vocab_size}, "
            f"tokenizer total={len(tokenizer)}, model={hf_config.vocab_size}"
        )
    tokenizer_info = {
        "class": type(tokenizer).__name__,
        "path": str(config.model_path),
        "vocab_size_base": int(tokenizer.vocab_size),
        "vocab_size_total": int(len(tokenizer)),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "model_max_length": int(tokenizer.model_max_length),
    }
    return hf_config, tokenizer, tokenizer_info


def discover_hook_points(hf_config: Any) -> list[dict[str, Any]]:
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config)
    points: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(model.model.layers):
        points.extend(
            [
                {"layer": layer_index, "activation_type": "residual_pre", "module": f"model.layers.{layer_index}"},
                {"layer": layer_index, "activation_type": "residual_post", "module": f"model.layers.{layer_index}"},
                {"layer": layer_index, "activation_type": "query", "module": f"model.layers.{layer_index}.self_attn.q_proj"},
                {"layer": layer_index, "activation_type": "key", "module": f"model.layers.{layer_index}.self_attn.k_proj"},
                {"layer": layer_index, "activation_type": "value", "module": f"model.layers.{layer_index}.self_attn.v_proj"},
                {"layer": layer_index, "activation_type": "head_output", "module": f"model.layers.{layer_index}.self_attn.o_proj (pre-hook)"},
                {"layer": layer_index, "activation_type": "attention_output", "module": f"model.layers.{layer_index}.self_attn"},
                {"layer": layer_index, "activation_type": "mlp_activation", "module": f"model.layers.{layer_index}.mlp.down_proj (pre-hook)"},
                {"layer": layer_index, "activation_type": "mlp_output", "module": f"model.layers.{layer_index}.mlp"},
            ]
        )
    del model
    return points


def transformer_lens_compatibility(hf_config: Any, dtype: torch.dtype, max_length: int) -> dict[str, Any]:
    try:
        from transformer_lens.loading_from_pretrained import get_pretrained_model_config

        tl_config = get_pretrained_model_config(
            "meta-llama/Llama-3.1-8B",
            hf_cfg=hf_config.to_dict(),
            dtype=dtype,
            n_ctx=max_length,
        )
        return {
            "compatible": True,
            "model_name": tl_config.model_name,
            "original_architecture": tl_config.original_architecture,
            "n_layers": tl_config.n_layers,
            "d_model": tl_config.d_model,
            "n_heads": tl_config.n_heads,
            "n_key_value_heads": tl_config.n_key_value_heads,
            "d_mlp": tl_config.d_mlp,
            "note": "Config conversion verified. Direct HF module hooks are used to avoid a second 8B weight conversion/copy.",
        }
    except Exception as exc:  # report the exact compatibility failure rather than substituting a model
        return {"compatible": False, "error": f"{type(exc).__name__}: {exc}"}


def load_model(config: CaptureConfig) -> Any:
    dtype = resolve_dtype(config.dtype)
    max_memory: dict[int | str, str] = {}
    for key, value in config.max_memory.items():
        max_memory[int(key) if key.isdigit() else key] = value
    config.offload_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "dtype": dtype,
        "device_map": config.device_map,
        "low_cpu_mem_usage": True,
        "offload_folder": str(config.offload_dir),
        "offload_state_dict": True,
        "attn_implementation": "eager",
    }
    if max_memory:
        kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(config.model_path, **kwargs)
    model.eval()
    return model


def model_device_map(model: Any) -> dict[str, str]:
    raw = getattr(model, "hf_device_map", {})
    return {str(k): str(v) for k, v in raw.items()}
