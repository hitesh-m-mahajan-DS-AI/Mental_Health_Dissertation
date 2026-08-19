import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.sae_analysis import (
    SAEAnalysisConfig,
    grouped_bootstrap_selected,
    grouped_smd,
    load_sparse_cache,
    matched_random_neuron_baseline,
    sparse_encode_tokens,
)


class _IdentitySparseSAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return torch.relu(values)


def test_sparse_encoder_aggregates_without_dense_output_storage() -> None:
    residual = torch.tensor([[-1.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    result = sparse_encode_tokens(_IdentitySparseSAE(), residual, token_batch_size=1)
    assert result["token_count"] == 2
    assert result["token_l0_sum"] == 3
    assert result["final_valid_response_token"]["indices"] == [0, 2]
    assert result["mean_response_token_activation"]["indices"] == [0, 1, 2]
    assert result["mean_response_token_activation"]["values"] == pytest.approx([1.5, 1.0, 2.0])


def test_grouped_smd_collapses_conversation_label_cells() -> None:
    values = np.asarray([[0.0], [2.0], [4.0], [6.0], [5.0], [7.0], [9.0], [11.0]])
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    # Cell means: negatives [1, 5], positives [6, 10]; pooled SD sqrt(8).
    assert grouped_smd(values, labels, groups)[0] == pytest.approx(5 / np.sqrt(8))


def test_group_bootstrap_is_seeded_and_clustered() -> None:
    values = np.asarray([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    groups = np.asarray(["a", "b", "c", "d", "e", "f"])
    first = grouped_bootstrap_selected(values, labels, groups, resamples=20, seed=7)
    second = grouped_bootstrap_selected(values, labels, groups, resamples=20, seed=7)
    np.testing.assert_array_equal(first, second)


def test_matched_random_neuron_baseline_is_reproducible() -> None:
    residual = np.linspace(-1, 1, 20)
    selected = np.asarray([1.5, -1.2, 1.1])
    first = matched_random_neuron_baseline(residual, selected, repetitions=100, seed=3)
    second = matched_random_neuron_baseline(residual, selected, repetitions=100, seed=3)
    assert first == second
    assert first["selected_count"] == 3
    assert 0 < first["mean_empirical_p"] <= 1


def test_sparse_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    rows = [
        {"example_id": "a", "final_valid_response_token": {"indices": [1], "values": [2.5]}},
        {"example_id": "b", "final_valid_response_token": {"indices": [0, 2], "values": [1.0, 3.0]}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    loaded, matrix = load_sparse_cache(path, "final_valid_response_token", 4)
    assert [row["example_id"] for row in loaded] == ["a", "b"]
    np.testing.assert_allclose(matrix.toarray(), [[0, 2.5, 0, 0], [1, 0, 3, 0]])


def test_config_rejects_mismatched_local_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "sae_analysis.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "revision": "pinned",
                "layer": 16,
                "hook_name": "blocks.16.hook_resid_post",
                "activation_key": "layer_16.residual_post",
                "d_in": 4096,
                "d_sae": 32768,
                "tasks": ["motivation", "empathy"],
                "bootstrap_resamples": 10,
                "random_neuron_baseline_repetitions": 10,
                "seeds": {"motivation": 42, "empathy": 43},
            }
        ),
        encoding="utf-8",
    )
    local = tmp_path / ".hf-sae" / "l16r_8x" / "Llama3_1-8B-Base-L16R-8x"
    (local / "checkpoints").mkdir(parents=True)
    (local / "checkpoints" / "final.safetensors").touch()
    (local / "lm_config.json").write_text("{}", encoding="utf-8")
    (local / "hyperparams.json").write_text(
        json.dumps({"hook_point_in": "wrong", "d_model": 4096, "d_sae": 32768}),
        encoding="utf-8",
    )
    config = SAEAnalysisConfig.load(tmp_path, config_path)
    with pytest.raises(ValueError, match="metadata mismatch"):
        config.validate_local_checkpoint()
