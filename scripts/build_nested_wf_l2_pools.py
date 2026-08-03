#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from aquant.data.history.microcap import HistoryReleaseReader
from aquant.factors.selection.nested_walk_forward import build_nested_l2_pools


def main() -> None:
    args = _arguments()
    pre_manifest = _complete_manifest(args.prehistory_report_dir)
    _complete_manifest(args.current_report_dir)
    current_metadata = json.loads((args.current_cache_dir / "metadata.json").read_text())
    current_dates = tuple(date.fromisoformat(value) for value in current_metadata["trade_dates"])
    outer_dates = tuple(
        value for value in current_dates if args.backtest_start <= value <= args.backtest_end
    )
    pre_dates = _trade_dates(
        args.release_dir,
        start=date.fromisoformat(pre_manifest["start_date"]),
        end=date.fromisoformat(pre_manifest["end_date"]),
    )
    if pre_dates[-1] >= current_dates[0]:
        raise ValueError("prehistory and current evidence dates must not overlap")
    factor_records = _merge_reports(
        args.prehistory_report_dir,
        args.current_report_dir,
        pre_dates=pre_dates,
        current_dates=current_dates,
    )
    payload = build_nested_l2_pools(
        factor_records=factor_records,
        evidence_dates=pre_dates + current_dates,
        outer_dates=outer_dates,
        output=args.output,
        purge_observations=args.purge_observations,
        fold_count=args.fold_count,
        maximum_family_members=args.maximum_family_members,
        provenance={
            "data_release_id": pre_manifest["data_release_id"],
            "prehistory_manifest_hash": _file_hash(
                args.prehistory_report_dir / "evaluation_manifest.json"
            ),
            "current_manifest_hash": _file_hash(
                args.current_report_dir / "evaluation_manifest.json"
            ),
            "current_cache_hash": current_metadata["content_hash"],
        },
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "fold_count": payload["fold_count"],
                "union_factor_count": payload["union_factor_count"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe nested walk-forward L2 pools")
    parser.add_argument("--prehistory-report-dir", type=Path, required=True)
    parser.add_argument("--current-report-dir", type=Path, required=True)
    parser.add_argument("--current-cache-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--backtest-start", type=date.fromisoformat, required=True)
    parser.add_argument("--backtest-end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--purge-observations", type=int, default=6)
    parser.add_argument("--maximum-family-members", type=int, default=9)
    return parser.parse_args()


def _complete_manifest(report_dir: Path) -> dict[str, Any]:
    manifest = json.loads((report_dir / "evaluation_manifest.json").read_text())
    if manifest.get("status") != "PASS" or manifest.get("completed_count") != manifest.get(
        "factor_count"
    ):
        raise ValueError(f"evaluation is incomplete: {report_dir}")
    return manifest


def _trade_dates(release_dir: Path, *, start: date, end: date) -> tuple[date, ...]:
    release = HistoryReleaseReader(release_dir)
    release.require("daily")
    connection = duckdb.connect(":memory:")
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT trade_date
            FROM read_parquet(?)
            WHERE trade_date BETWEEN ? AND ?
              AND exchange IN ('XSHG', 'XSHE')
            ORDER BY trade_date
            """,
            [release.parquet_pattern("daily"), start, end],
        ).fetchall()
    finally:
        connection.close()
    return tuple(row[0] for row in rows)


def _merge_reports(
    prehistory_dir: Path,
    current_dir: Path,
    *,
    pre_dates: tuple[date, ...],
    current_dates: tuple[date, ...],
) -> dict[str, dict[str, Any]]:
    prehistory = _reports(prehistory_dir, expected_dates=len(pre_dates))
    current = _reports(current_dir, expected_dates=len(current_dates))
    if prehistory.keys() != current.keys():
        missing_pre = sorted(current.keys() - prehistory.keys())
        missing_current = sorted(prehistory.keys() - current.keys())
        raise ValueError(
            f"report factor mismatch: missing_pre={missing_pre[:5]}, "
            f"missing_current={missing_current[:5]}"
        )
    merged: dict[str, dict[str, Any]] = {}
    for factor_id in sorted(prehistory):
        left = prehistory[factor_id]
        right = current[factor_id]
        if left["expression_hash"] != right["expression_hash"]:
            raise ValueError(f"expression changed between evidence windows: {factor_id}")
        merged[factor_id] = {
            **right,
            "rank_ic": np.concatenate((left["rank_ic"], right["rank_ic"])),
        }
    return merged


def _reports(directory: Path, *, expected_dates: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in directory.glob("*.json"):
        if path.name == "evaluation_manifest.json":
            continue
        report = json.loads(path.read_text())
        factor = report["factor"]
        values = np.asarray(report["rank_ic"]["by_date"], dtype=np.float64)
        if len(values) != expected_dates:
            raise ValueError(f"RankIC/date mismatch in {path.name}")
        factor_id = str(factor["factor_id"])
        result[factor_id] = {
            "family": factor["family"],
            "subfamily": factor.get("subfamily", "unknown"),
            "expected_direction": factor.get("expected_direction", 0),
            "expression_hash": factor["expression_hash"],
            "complexity_score": factor.get("complexity_score", 100.0),
            "rank_ic": values,
        }
    return result


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
