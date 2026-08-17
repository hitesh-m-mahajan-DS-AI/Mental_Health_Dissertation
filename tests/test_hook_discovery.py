from pathlib import Path

from transformers import AutoConfig

from src.model_loader import discover_hook_points


ROOT = Path(__file__).resolve().parents[1]


def test_all_expected_llama_hook_points_are_discovered() -> None:
    config = AutoConfig.from_pretrained(ROOT / "model", local_files_only=True)
    points = discover_hook_points(config)
    assert len(points) == 32 * 9
    assert {point["activation_type"] for point in points} == {
        "residual_pre",
        "residual_post",
        "query",
        "key",
        "value",
        "head_output",
        "attention_output",
        "mlp_activation",
        "mlp_output",
    }
