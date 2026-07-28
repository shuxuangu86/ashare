#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

from aquant.factors.evaluation import write_leakage_attestation

_TESTS = (
    "tests/unit/factors/test_institutional_evaluation.py",
    "tests/unit/factors/test_l1_operators.py",
    "tests/integration/test_factor_pipeline.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and attest factor leakage controls")
    parser.add_argument("--data-release-id", required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-cov", *_TESTS],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        return completed.returncode
    summary = completed.stdout.strip().splitlines()[-1]
    payload = write_leakage_attestation(
        output=args.output,
        data_release_id=args.data_release_id,
        code_version=args.code_version,
        test_files=_TESTS,
        test_summary=summary,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
