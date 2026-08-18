"""Generate the completion-aware cross-method circuit atlas."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.circuit_atlas import build_circuit_atlas, write_circuit_atlas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--causal-summary", type=Path, required=True)
    parser.add_argument("--sae-summary", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("results/circuit_atlas"))
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--expected-records", type=int, default=200)
    parser.add_argument("--probe-aggregation", default="final_valid_response_token")
    args = parser.parse_args()
    atlas = build_circuit_atlas(
        args.probe_summary,
        args.causal_summary,
        args.sae_summary,
        expected_records=args.expected_records,
        probe_aggregation=args.probe_aggregation,
    )
    paths = write_circuit_atlas(atlas, args.output_directory, args.run_label)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

