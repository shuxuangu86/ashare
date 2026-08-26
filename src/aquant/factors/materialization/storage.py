import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from aquant.domain.data_release import DataReleaseId
from aquant.factors.atomic.models import FactorPanelInput
from aquant.factors.materialization.manifest import MaterializedFile
from aquant.factors.spec import FactorSpec
from aquant.factors.types import FactorResult, FactorValue


def write_partitioned(
    root: Path,
    results: tuple[tuple[str, FactorResult], ...],
) -> tuple[MaterializedFile, ...]:
    grouped: dict[tuple[str, int, int], list[FactorValue]] = defaultdict(list)
    for family, result in results:
        for value in result.values:
            grouped[(family, value.trade_date.year, value.trade_date.month)].append(value)
    files: list[MaterializedFile] = []
    for (family, year, month), values in sorted(grouped.items()):
        directory = root / f"factor_family={family}" / f"year={year}" / f"month={month:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "data.parquet"
        ordered = sorted(
            values,
            key=lambda item: (
                item.trade_date,
                item.ts_code,
                item.factor_id,
                item.factor_version,
            ),
        )
        table = pa.table(
            {
                "trade_date": [item.trade_date for item in ordered],
                "ts_code": [item.ts_code for item in ordered],
                "factor_id": [item.factor_id for item in ordered],
                "factor_version": [item.factor_version for item in ordered],
                "value": [item.value for item in ordered],
                "is_valid": [item.is_valid for item in ordered],
                "quality_flags": [list(item.quality_flags) for item in ordered],
                "data_release_id": [str(item.data_release_id) for item in ordered],
                "computed_at": [item.computed_at for item in ordered],
            }
        )
        pq.write_table(table, path, compression="zstd")
        relative = path.relative_to(root).as_posix()
        files.append(
            MaterializedFile(
                relative,
                len(ordered),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return tuple(files)


def write_factor_array(
    root: Path,
    *,
    spec: FactorSpec,
    panel: FactorPanelInput,
    values: object,
    data_release_id: DataReleaseId,
    computed_at: datetime,
) -> tuple[MaterializedFile, ...]:
    matrix = np.asarray(values, dtype=np.float64)
    expected = (len(panel.trade_dates), len(panel.ts_codes))
    if matrix.shape != expected:
        raise ValueError(f"factor array returned {matrix.shape}, expected {expected}")
    files: list[MaterializedFile] = []
    partitions = sorted({(value.year, value.month) for value in panel.trade_dates})
    for year, month in partitions:
        date_indices = [
            index
            for index, value in enumerate(panel.trade_dates)
            if value.year == year and value.month == month
        ]
        partition_values = matrix[date_indices].reshape(-1)
        row_count = len(partition_values)
        valid = np.isfinite(partition_values)
        offsets = np.concatenate(([0], np.cumsum(~valid, dtype=np.int32)))
        flags = pa.ListArray.from_arrays(
            pa.array(offsets, type=pa.int32()),
            pa.array(["NON_FINITE"] * int(np.count_nonzero(~valid)), type=pa.string()),
        )
        table = pa.table(
            {
                "trade_date": [
                    panel.trade_dates[index] for index in date_indices for _ in panel.ts_codes
                ],
                "ts_code": list(panel.ts_codes) * len(date_indices),
                "factor_id": pa.array([spec.factor_id] * row_count, type=pa.string()),
                "factor_version": pa.array([spec.version] * row_count, type=pa.string()),
                "value": pa.array(
                    np.where(valid, partition_values, np.nan),
                    type=pa.float64(),
                    mask=~valid,
                ),
                "is_valid": pa.array(valid),
                "quality_flags": flags,
                "data_release_id": pa.array(
                    [str(data_release_id)] * row_count,
                    type=pa.string(),
                ),
                "computed_at": [computed_at] * row_count,
            }
        )
        directory = root / f"factor_family={spec.family}" / f"year={year}" / f"month={month:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"factor={spec.factor_id}-version={spec.version}.parquet"
        pq.write_table(table, path, compression="zstd")
        relative = path.relative_to(root).as_posix()
        files.append(MaterializedFile(relative, row_count, _file_hash(path)))
    return tuple(files)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
