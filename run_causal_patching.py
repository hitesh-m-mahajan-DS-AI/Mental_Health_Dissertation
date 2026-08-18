"""Command-line entry point for controlled causal activation patching."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.causal_config import load_causal_config
from src.causal_patching import run_causal_patching


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Llama-3.1-8B residual-stream causal patching")
    parser.add_argument("--config", type=Path, default=Path("configs/causal_patching.json"))
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-run-id")
    args = parser.parse_args()
    run_id = run_causal_patching(
        load_causal_config(args.config),
        inspect_only=args.inspect_only,
        smoke_test=args.smoke_test,
        resume_run_id=args.resume_run_id,
    )
    print(f"Causal patching run ID: {run_id}")


if __name__ == "__main__":
    main()
