"""Generate verified JSON, charts, and prose from a completed causal run."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.causal_config import load_causal_config
from src.causal_reporting import generate_causal_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--config", type=Path, default=Path("configs/causal_patching.json"))
    args = parser.parse_args()
    generate_causal_report(load_causal_config(args.config), args.run_id)
    print(f"Generated causal-patching report for {args.run_id}")


if __name__ == "__main__":
    main()
