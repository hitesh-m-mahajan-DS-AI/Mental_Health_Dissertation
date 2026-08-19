"""Generate compact charts and reports from completed linear-probe JSON results."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.probe_reporting import generate_probe_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260817T181608Z")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    summary, report, manifest = generate_probe_report(root, args.run_id)
    print(f"Summary: {summary}")
    print(f"Report: {report}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
