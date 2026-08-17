"""Single user-facing entry point for motivation and empathy activation capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.activation_capture import run_capture
from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Llama-3.1-8B activations for both frozen datasets in one run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/activation_capture.json"),
        help="JSON configuration file (default: configs/activation_capture.json)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate configs/datasets/hook paths without loading the 8B weights.",
    )
    parser.add_argument(
        "--resume-run-id",
        type=str,
        default=None,
        help="Resume an interrupted run, preserving completed per-example files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_capture(config, inspect_only=args.inspect_only, resume_run_id=args.resume_run_id)


if __name__ == "__main__":
    main()
