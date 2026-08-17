"""Exact Hugging Face Llama hook definitions and activation collection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


_GROUPS = {
    "residual_pre": "residual_stream",
    "residual_post": "residual_stream",
    "query": "attention",
    "key": "attention",
    "value": "attention",
    "head_output": "attention",
    "attention_output": "attention",
    "mlp_activation": "mlp",
    "mlp_output": "mlp",
}


def activation_group(name: str) -> str:
    return _GROUPS[name]


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"Hook output did not contain a tensor: {type(value).__name__}")


class ActivationCollector:
    def __init__(self, model: Any, capture_types: tuple[str, ...]) -> None:
        self.model = model
        self.capture_types = set(capture_types)
        self.handles: list[Any] = []
        self.values: dict[str, torch.Tensor] = {}
        self.n_heads = int(model.config.num_attention_heads)
        self.head_dim = int(model.config.hidden_size // model.config.num_attention_heads)

    def _save(self, key: str, value: Any, reshape_heads: bool = False) -> None:
        tensor = _first_tensor(value).detach()
        if tensor.ndim < 2 or tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size one for {key}; got shape {tuple(tensor.shape)}")
        tensor = tensor[0]
        if reshape_heads:
            if tensor.shape[-1] != self.n_heads * self.head_dim:
                raise ValueError(f"Cannot reshape {key} with shape {tuple(tensor.shape)} into heads")
            tensor = tensor.reshape(*tensor.shape[:-1], self.n_heads, self.head_dim)
        self.values[key] = tensor.to(device="cpu", copy=True).contiguous()

    def _pre_hook(self, key: str, reshape_heads: bool = False) -> Callable[..., None]:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            self._save(key, inputs, reshape_heads=reshape_heads)

        return hook

    def _post_hook(self, key: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            self._save(key, output)

        return hook

    def register(self) -> None:
        if self.handles:
            raise RuntimeError("Hooks are already registered")
        for layer_index, layer in enumerate(self.model.model.layers):
            prefix = f"layer_{layer_index:02d}."
            if "residual_pre" in self.capture_types:
                self.handles.append(layer.register_forward_pre_hook(self._pre_hook(prefix + "residual_pre")))
            if "residual_post" in self.capture_types:
                self.handles.append(layer.register_forward_hook(self._post_hook(prefix + "residual_post")))
            if "query" in self.capture_types:
                self.handles.append(layer.self_attn.q_proj.register_forward_hook(self._post_hook(prefix + "query")))
            if "key" in self.capture_types:
                self.handles.append(layer.self_attn.k_proj.register_forward_hook(self._post_hook(prefix + "key")))
            if "value" in self.capture_types:
                self.handles.append(layer.self_attn.v_proj.register_forward_hook(self._post_hook(prefix + "value")))
            if "head_output" in self.capture_types:
                self.handles.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        self._pre_hook(prefix + "head_output", reshape_heads=True)
                    )
                )
            if "attention_output" in self.capture_types:
                self.handles.append(
                    layer.self_attn.register_forward_hook(self._post_hook(prefix + "attention_output"))
                )
            if "mlp_activation" in self.capture_types:
                self.handles.append(
                    layer.mlp.down_proj.register_forward_pre_hook(self._pre_hook(prefix + "mlp_activation"))
                )
            if "mlp_output" in self.capture_types:
                self.handles.append(layer.mlp.register_forward_hook(self._post_hook(prefix + "mlp_output")))

    def clear(self) -> None:
        self.values.clear()

    def pop(self, token_positions: str, valid_length: int) -> dict[str, torch.Tensor]:
        result = self.values
        self.values = {}
        if token_positions == "all":
            return result
        if token_positions == "last_non_padding":
            index = valid_length - 1
            return {key: value[index : index + 1] for key, value in result.items()}
        raise ValueError(f"Unsupported token_positions {token_positions!r}")

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
