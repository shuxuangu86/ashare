#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Report AQuant market-state materialization")
    parser.add_argument("--artifact-directory", type=Path, default=Path("artifacts/regime_v1"))
    args = parser.parse_args()
    summary = args.artifact_directory / "state_materialization_summary.json"
    print(json.dumps(json.loads(summary.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
