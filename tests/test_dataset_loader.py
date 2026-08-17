from pathlib import Path

from src.dataset_loader import load_empathy, load_motivation


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_motivation_population() -> None:
    bundle = load_motivation(ROOT / "annomi_motivation_final_candidate_v3_2.csv")
    assert len(bundle.examples) == 2246
    assert sum(example.label == 1 for example in bundle.examples) == 2096
    assert sum(example.label == 0 for example in bundle.examples) == 150
    assert bundle.audit["excluded_review_rows"] == 2636


def test_frozen_empathy_population_and_role_anomalies() -> None:
    bundle = load_empathy(ROOT / "MHLCD.csv")
    assert len(bundle.examples) == 14408
    assert bundle.audit["included_trailing_space_counselor_rows"] == 18
    assert bundle.audit["excluded_labeled_missing_role_rows"] == 2
    assert sum(example.label == 0 for example in bundle.examples) == 10091
    assert sum(example.label == 1 for example in bundle.examples) == 4317
    assert len({example.example_id for example in bundle.examples}) == len(bundle.examples)
