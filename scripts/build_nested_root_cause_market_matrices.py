#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import numpy.typing as npt

from aquant.data.history.microcap import HistoryReleaseReader  # type: ignore[import-untyped]


def main() -> None:
    args = _arguments()
    cache_metadata = json.loads((args.cache_dir / "metadata.json").read_text())
    if cache_metadata.get("status") != "PASS":
        raise ValueError("convergence cache must be complete")
    dates = tuple(cache_metadata["trade_dates"])
    codes = tuple(cache_metadata["ts_codes"])
    shape = (len(dates), len(codes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "market_matrices.json"
    if args.resume and _is_complete(metadata_path, cache_metadata["content_hash"], shape):
        print(metadata_path.read_text(), end="")
        return

    release = HistoryReleaseReader(args.release_dir)
    release.require("daily", "adj_factor")
    raw_close = np.load(args.cache_dir / "close.npy", mmap_mode="r")
    if raw_close.shape != shape:
        raise ValueError("cache close matrix does not align with metadata")
    temporary_dir = Path(tempfile.mkdtemp(prefix=".nested-root-cause-", dir=args.output_dir))
    try:
        adjusted_close = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            temporary_dir / "adjusted_close.npy", mode="w+", dtype=np.float32, shape=shape
        )
        adjusted_open = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            temporary_dir / "adjusted_open.npy", mode="w+", dtype=np.float32, shape=shape
        )
        adjustment_factor = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            temporary_dir / "adjustment_factor.npy", mode="w+", dtype=np.float32, shape=shape
        )
        adjusted_close[:] = np.nan
        adjusted_open[:] = np.nan
        adjustment_factor[:] = np.nan
        _fill(
            release=release,
            start_date=dates[0],
            end_date=dates[-1],
            dates=dates,
            codes=codes,
            adjusted_close=adjusted_close,
            adjusted_open=adjusted_open,
            adjustment_factor=adjustment_factor,
        )
        expected_close = _adjusted_price(raw_close, adjustment_factor)
        valid = np.isfinite(expected_close)
        if np.count_nonzero(valid) == 0:
            raise ValueError("adjusted market matrices contain no observations")
        if not np.allclose(
            expected_close[valid], adjusted_close[valid], rtol=1e-5, atol=1e-3
        ):
            raise ValueError("adjusted close does not reconcile with cache close")
        for array in (adjusted_close, adjusted_open, adjustment_factor):
            array.flush()
            array._mmap.close()
        outputs = {}
        for name in ("adjusted_close.npy", "adjusted_open.npy", "adjustment_factor.npy"):
            source = temporary_dir / name
            target = args.output_dir / name
            os.replace(source, target)
            outputs[name] = {"sha256": _file_hash(target), "size": target.stat().st_size}
        payload: dict[str, Any] = {
            "status": "PASS",
            "stage": "NESTED_ROOT_CAUSE_MARKET_MATRICES",
            "shape": list(shape),
            "start_date": dates[0],
            "end_date": dates[-1],
            "cache_content_hash": cache_metadata["content_hash"],
            "data_release_id": cache_metadata["data_release_id"],
            "release_manifest_status": release.manifest.status,
            "method": "adjusted_price=raw_daily_price*cumulative_adj_factor",
            "outputs": outputs,
            "code_version": args.code_version,
        }
        payload["content_hash"] = _hash(payload)
        _write_json(metadata_path, payload)
        print(json.dumps(payload, ensure_ascii=False))
    finally:
        if temporary_dir.exists():
            for child in temporary_dir.iterdir():
                child.unlink(missing_ok=True)
            temporary_dir.rmdir()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned adjusted market matrices")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _fill(
    *,
    release: Any,
    start_date: str,
    end_date: str,
    dates: tuple[str, ...],
    codes: tuple[str, ...],
    adjusted_close: npt.NDArray[np.float32],
    adjusted_open: npt.NDArray[np.float32],
    adjustment_factor: npt.NDArray[np.float32],
) -> None:
    date_index = {value: index for index, value in enumerate(dates)}
    code_index = {value: index for index, value in enumerate(codes)}
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='2GB'")
        connection.execute("SET threads=2")
        batches = connection.execute(
            """
            SELECT
                CAST(daily.trade_date AS VARCHAR),
                daily.ts_code,
                CAST(daily.close AS DOUBLE) * CAST(adj.adj_factor AS DOUBLE),
                CAST(daily.open AS DOUBLE) * CAST(adj.adj_factor AS DOUBLE),
                CAST(adj.adj_factor AS DOUBLE)
            FROM read_parquet(?) AS daily
            JOIN read_parquet(?) AS adj
              ON adj.trade_date = daily.trade_date
             AND adj.ts_code = daily.ts_code
            WHERE daily.trade_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND daily.exchange IN ('XSHG', 'XSHE')
            ORDER BY daily.trade_date, daily.ts_code
            """,
            [
                release.parquet_pattern("daily"),
                release.parquet_pattern("adj_factor"),
                start_date,
                end_date,
            ],
        ).fetch_record_batch(rows_per_batch=100_000)
        for batch in batches:
            batch_dates = batch.column(0).to_pylist()
            batch_codes = batch.column(1).to_pylist()
            keep = np.fromiter(
                (
                    day in date_index and code in code_index
                    for day, code in zip(batch_dates, batch_codes, strict=True)
                ),
                dtype=bool,
            )
            if not np.any(keep):
                continue
            date_positions = np.fromiter(
                (
                    date_index[day]
                    for day, selected in zip(batch_dates, keep, strict=True)
                    if selected
                ),
                dtype=np.int64,
            )
            code_positions = np.fromiter(
                (
                    code_index[code]
                    for code, selected in zip(batch_codes, keep, strict=True)
                    if selected
                ),
                dtype=np.int64,
            )
            adjusted_close[date_positions, code_positions] = np.asarray(
                batch.column(2).to_pylist(), dtype=np.float64
            )[keep]
            adjusted_open[date_positions, code_positions] = np.asarray(
                batch.column(3).to_pylist(), dtype=np.float64
            )[keep]
            adjustment_factor[date_positions, code_positions] = np.asarray(
                batch.column(4).to_pylist(), dtype=np.float64
            )[keep]
    finally:
        connection.close()


def _adjusted_price(
    raw_price: npt.ArrayLike, adjustment_factor: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    raw = np.asarray(raw_price, dtype=np.float64)
    factor = np.asarray(adjustment_factor, dtype=np.float64)
    if raw.shape != factor.shape:
        raise ValueError("raw price and adjustment factor must align")
    with np.errstate(all="ignore"):
        result = raw * factor
    result[~np.isfinite(result)] = np.nan
    return result


def _is_complete(metadata_path: Path, cache_hash: str, shape: tuple[int, int]) -> bool:
    if not metadata_path.exists():
        return False
    payload = json.loads(metadata_path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("cache_content_hash") != cache_hash
        or payload.get("shape") != list(shape)
    ):
        return False
    for name, evidence in payload.get("outputs", {}).items():
        path = metadata_path.parent / name
        if not path.exists() or _file_hash(path) != evidence.get("sha256"):
            return False
    return True


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
