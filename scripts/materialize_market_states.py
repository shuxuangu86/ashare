#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import date
from pathlib import Path

from aquant.data.history.microcap import HistoryReleaseReader
from aquant.regime.catalog import core_state_registry
from aquant.regime.data_adapter import load_market_state_panel
from aquant.regime.definitions import MarketStateFamily, MarketStateScope
from aquant.regime.panel import compute_market_state_panel
from aquant.regime.publisher import publish_market_state_artifacts
from aquant.regime.registry import MarketStateRegistry


def main() -> None:
    started_at = time.perf_counter()
    args = _parser().parse_args()
    if not 1 <= args.max_workers <= 2:
        raise SystemExit("--max-workers must be between 1 and 2")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    registry = core_state_registry()
    selected = tuple(
        spec
        for spec in registry
        if (args.state_family is None or spec.family.value == args.state_family)
        and (args.scope is None or spec.scope.value == args.scope)
    )
    selected_registry = MarketStateRegistry(selected)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "registered": len(registry),
                    "selected": len(selected),
                    "families": sorted({spec.family.value for spec in selected}),
                    "scopes": sorted({spec.scope.value for spec in selected}),
                },
                indent=2,
            )
        )
        return
    output = args.output
    manifest = output / "manifest.json"
    config_hash = _config_hash(Path("configs/regime"))
    input_hash = _input_hash(args, selected_registry, config_hash)
    if manifest.exists() and args.resume and not args.force:
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        previous_hash = previous.get("summary", {}).get("input_hash")
        if previous_hash == input_hash:
            print(manifest.read_text(encoding="utf-8"))
            return
        raise SystemExit("resume input hash changed; rerun with --force")
    if manifest.exists() and not args.force:
        raise SystemExit("output already exists; use --resume or --force")
    panel = load_market_state_panel(
        args.release_directory,
        args.raw_state,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
    )
    computed = compute_market_state_panel(panel)
    selected_ids = {spec.state_id for spec in selected}
    selected_states = {
        state_id: values for state_id, values in computed.items() if state_id in selected_ids
    }
    summary = publish_market_state_artifacts(
        output,
        panel=panel,
        states=selected_states,
        registry=selected_registry,
        data_release_id=args.release_id,
        code_version=_code_version(),
        config_hash=config_hash,
        input_hash=input_hash,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        started_at=started_at,
        factor_root=Path("data/factors"),
        daily_pattern=HistoryReleaseReader(args.release_directory).parquet_pattern("daily"),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize PIT-safe AQuant market states")
    parser.add_argument("--state-family", choices=[item.value for item in MarketStateFamily])
    parser.add_argument("--scope", choices=[item.value for item in MarketStateScope])
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2007, 1, 4))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2026, 7, 17))
    parser.add_argument("--release-id", default="cn_equity_history_20260717_001")
    parser.add_argument(
        "--release-directory",
        type=Path,
        default=Path("data/standard/history-release=cn_equity_history_20260717_001"),
    )
    parser.add_argument(
        "--raw-state",
        type=Path,
        default=Path("artifacts/tushare-backfill/state.sqlite3"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/regime_v1"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _config_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.yaml")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _code_version() -> str:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relevant_paths = (
        "configs/regime",
        "scripts/materialize_market_states.py",
        "scripts/evaluate_market_states.py",
        "scripts/build_regime_feature_set.py",
        "scripts/report_market_states.py",
        "src/aquant/regime",
        "src/aquant/factors/data_loader.py",
    )
    dirty = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all", "--", *relevant_paths),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return f"{revision}-dirty" if dirty else revision


def _input_hash(
    args: argparse.Namespace,
    registry: MarketStateRegistry,
    config_hash: str,
) -> str:
    release_manifest = args.release_directory / "manifest.json"
    raw_state = args.raw_state.stat()
    payload = {
        "state_ids": sorted(spec.state_id for spec in registry),
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "release_id": args.release_id,
        "release_manifest_hash": (
            hashlib.sha256(release_manifest.read_bytes()).hexdigest()
            if release_manifest.exists()
            else None
        ),
        "raw_state_size": raw_state.st_size,
        "raw_state_mtime_ns": raw_state.st_mtime_ns,
        "config_hash": config_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


if __name__ == "__main__":
    main()
