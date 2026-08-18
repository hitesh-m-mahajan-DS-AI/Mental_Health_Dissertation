import numpy as np

from src.causal_config import load_causal_config
from src.causal_patching import summarize_layer_results
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_summary_uses_only_valid_denominators_and_is_reproducible() -> None:
    config = load_causal_config(ROOT / "configs" / "causal_patching.json")
    records = [
        {"denominator_valid": True, "faithfulness": [0.0, 1.0],
         "patched_logit_differences": [1.0, 2.0], "corrupted_logit_difference": 1.0},
        {"denominator_valid": True, "faithfulness": [0.2, 0.8],
         "patched_logit_differences": [1.2, 1.8], "corrupted_logit_difference": 1.0},
        {"denominator_valid": False, "faithfulness": [None, None],
         "patched_logit_differences": [0.0, 0.0], "corrupted_logit_difference": 0.0},
    ]
    first = summarize_layer_results(records, config)
    second = summarize_layer_results(records, config)
    assert first == second
    assert first["valid_denominator_records"] == 2
    assert np.allclose(first["mean_faithfulness_by_layer"], [0.1, 0.9])
