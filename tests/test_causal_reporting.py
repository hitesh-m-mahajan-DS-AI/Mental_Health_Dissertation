import json
from pathlib import Path

import pytest

from src.causal_config import load_causal_config
from src.causal_reporting import load_and_verify_records


ROOT = Path(__file__).resolve().parents[1]


def test_verifier_rejects_incomplete_result_set(tmp_path: Path) -> None:
    config = load_causal_config(ROOT / "configs" / "causal_patching.json")
    object.__setattr__(config, "output_root", tmp_path)
    directory = tmp_path / "motivation" / "pairs" / "test"
    directory.mkdir(parents=True)
    (directory / "000_example.json").write_text(json.dumps({"selection_ordinal": 0}))
    with pytest.raises(RuntimeError, match="expected 200 pair results, found 1"):
        load_and_verify_records(config, "motivation", "test")
