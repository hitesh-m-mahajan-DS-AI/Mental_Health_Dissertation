"""Single user-facing entry point for independent motivation and empathy probes."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis_config import load_analysis_methodology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent grouped residual-stream probes for motivation and empathy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis_methodology.json"),
        help="Analysis methodology JSON (default: configs/analysis_methodology.json)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate scientific settings without reading activation tensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methodology = load_analysis_methodology(args.config)
    methodology.require_resolved("linear_probing")

    if args.validate_only:
        print(
            "Linear-probing methodology is complete for activation run "
            f"{methodology.activation_run_id}."
        )
        return

    raise NotImplementedError(
        "Probe execution will be enabled after the declared methodology passes validation."
    )


if __name__ == "__main__":
    main()
