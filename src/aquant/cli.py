import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from aquant.config import load_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aquant")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_check = subparsers.add_parser(
        "config-check", help="validate configuration and print a secret-safe summary"
    )
    config_check.add_argument(
        "--config",
        action="append",
        required=True,
        type=Path,
        help="YAML file; repeat to add overlays",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "config-check":
        settings = load_settings(args.config)
        print(json.dumps(settings.safe_summary(), ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
