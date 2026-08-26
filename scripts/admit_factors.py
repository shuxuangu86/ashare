#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from aquant.factors.evaluation import admit_factor_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply configured production factor gates")
    parser.add_argument("--convergence", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--leakage-attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    try:
        payload = admit_factor_candidates(
            convergence_path=args.convergence,
            report_dir=args.report_dir,
            evaluation_manifest_path=args.evaluation_manifest,
            gate_config_path=args.gate_config,
            leakage_attestation_path=args.leakage_attestation,
            output=args.output,
            horizon=args.horizon,
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "production_core_count": payload["production_core_count"],
                    "content_hash": payload["content_hash"],
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if payload["status"] == "PASS" else 2
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
