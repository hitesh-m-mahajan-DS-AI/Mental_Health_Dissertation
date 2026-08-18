from pathlib import Path

from transformers import AutoTokenizer

from src.causal_config import load_causal_config
from src.causal_pairs import build_pairs, validate_and_tokenize_pair


ROOT = Path(__file__).resolve().parents[1]


def test_builds_exact_frozen_200_plus_200_pairs() -> None:
    config = load_causal_config(ROOT / "configs" / "causal_patching.json")
    motivation = build_pairs(config, "motivation")
    empathy = build_pairs(config, "empathy")
    assert len(motivation) == len(empathy) == 200
    assert [p.source_row for p in motivation] == sorted(p.source_row for p in motivation)
    assert [p.source_row for p in empathy] == sorted(p.source_row for p in empathy)
    assert all(p.supportive_prompt != p.neutral_prompt for p in motivation + empathy)


def test_targets_are_single_tokens_and_pair_lengths_align() -> None:
    config = load_causal_config(ROOT / "configs" / "causal_patching.json")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    for dataset in config.datasets:
        for pair in build_pairs(config, dataset):
            encoded = validate_and_tokenize_pair(pair, tokenizer, config)
            assert len(encoded["clean_input_ids"]) == len(encoded["corrupted_input_ids"])
            assert encoded["supportive_target_id"] != encoded["neutral_target_id"]
