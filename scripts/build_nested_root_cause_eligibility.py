#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from aquant.data.history import DuckDBMicrocapHistory  # type: ignore[import-untyped]
from aquant.strategies.microcap.experiments import (  # type: ignore[import-untyped]
    RebalanceFrequency,
    generate_rebalance_dates,
)


def main() -> None:
    args = _arguments()
    _attest_code_version(args.code_version)
    cache = json.loads((args.cache_dir / "metadata.json").read_text())
    l3 = json.loads(args.l3_metadata.read_text())
    dates = tuple(date.fromisoformat(value) for value in cache["trade_dates"])
    codes = tuple(str(value) for value in cache["ts_codes"])
    date_index = {value: index for index, value in enumerate(dates)}
    code_index = {value: index for index, value in enumerate(codes)}
    mask = np.zeros((len(dates), len(codes)), dtype=np.bool_)
    flags = np.zeros((len(dates), len(codes)), dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    with DuckDBMicrocapHistory(args.history_release) as history:
        release_id = history.release.manifest.release_id
        for fold in l3["folds"]:
            start = date.fromisoformat(fold["test_start"])
            end = date.fromisoformat(fold["test_end"])
            sessions = tuple(value for value in dates if start <= value <= end)
            selected = fold["l4_selection"]["selected"]
            frequency = RebalanceFrequency(selected["frequency"])
            for trade_date in generate_rebalance_dates(sessions, frequency):
                snapshot = history.snapshot(trade_date)
                counts = {
                    "eligible": 0,
                    "exchange": 0,
                    "new_listing": 0,
                    "st": 0,
                    "suspended": 0,
                    "delisting_risk": 0,
                    "missing_cache_code": 0,
                }
                for item in snapshot.observations:
                    code = _ts_code(item.symbol.canonical)
                    position = code_index.get(code)
                    reason = _exclusion_reason(item, trade_date)
                    if position is None:
                        counts["missing_cache_code"] += 1
                    else:
                        value = _eligibility_flags(item, trade_date)
                        flags[date_index[trade_date], position] = value
                        if reason is None:
                            mask[date_index[trade_date], position] = True
                            counts["eligible"] += 1
                        else:
                            counts[reason] += 1
                rows.append(
                    {
                        "fold": int(fold["fold"]),
                        "trade_date": trade_date.isoformat(),
                        "target_count": int(selected["target_count"]),
                        "frequency": frequency.value,
                        **counts,
                    }
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / "production_eligibility_mask.npy"
    flags_path = args.output_dir / "eligibility_flags.npy"
    _write_npy(mask_path, mask)
    _write_npy(flags_path, flags)
    counts_text = _csv_text(rows)
    _write_text(args.output_dir / "eligibility_counts.csv", counts_text)
    stable = {
        "status": "PASS",
        "stage": "NESTED_ROOT_CAUSE_ELIGIBILITY",
        "shape": list(mask.shape),
        "rebalance_dates": len(rows),
        "history_release_id": release_id,
        "minimum_listing_days": 120,
        "excluded": ["NON_SH_SZ", "NEW_LISTING", "ST", "SUSPENDED", "DELISTING_RISK"],
        "eligibility_flag_bits": {
            "PIT_UNIVERSE": 1,
            "LISTED_120_DAYS": 2,
            "NOT_ST": 4,
            "NOT_SUSPENDED": 8,
            "NOT_DELISTING_RISK": 16,
        },
        "cache_content_hash": cache["content_hash"],
        "l3_content_hash": l3["content_hash"],
        "mask_sha256": _file_hash(mask_path),
        "flags_sha256": _file_hash(flags_path),
        "counts_sha256": _text_hash(counts_text),
        "code_version": args.code_version,
        "generator_file": "scripts/build_nested_root_cause_eligibility.py",
        "generator_sha256": _file_hash(Path(__file__)),
    }
    stable["content_hash"] = _hash(stable)
    payload = {**stable, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    _write_text(args.output_dir / "eligibility_metadata.json", json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


def _attest_code_version(declared: str) -> None:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
    ).strip()
    if declared != head or dirty:
        raise ValueError("research artifacts require declared HEAD and a clean tracked worktree")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build production eligibility mask for L3 audit")
    parser.add_argument("--history-release", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--l3-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    return parser.parse_args()


def _ts_code(canonical: str) -> str:
    return canonical.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")


def _exclusion_reason(item: Any, trade_date: date) -> str | None:
    canonical = item.symbol.canonical
    if not canonical.endswith((".XSHG", ".XSHE")):
        return "exchange"
    if item.suspended:
        return "suspended"
    if item.is_st:
        return "st"
    if item.is_delisting_risk:
        return "delisting_risk"
    if (trade_date - item.list_date).days < 120:
        return "new_listing"
    return None


def _eligibility_flags(item: Any, trade_date: date) -> int:
    value = 1
    if (trade_date - item.list_date).days >= 120:
        value |= 2
    if not item.is_st:
        value |= 4
    if not item.suspended:
        value |= 8
    if not item.is_delisting_risk:
        value |= 16
    return value


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_npy(path: Path, values: np.ndarray[Any, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".eligibility-", suffix=".npy", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.save(temporary, values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


if __name__ == "__main__":
    main()
