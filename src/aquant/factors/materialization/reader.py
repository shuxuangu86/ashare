import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from aquant.factors.materialization.manifest import MaterializationManifest


@dataclass(frozen=True, slots=True)
class StoredFactorValue:
    trade_date: date
    ts_code: str
    value: float


class MaterializedFactorReader:
    """Verified, predicate-pushed reader for one immutable factor release."""

    def __init__(self, release_directory: Path) -> None:
        self.release_directory = release_directory.resolve()
        self.manifest = MaterializationManifest.read(self.release_directory / "manifest.json")
        if self.manifest.status != "PUBLISHED":
            raise ValueError("factor materialization must be published")

    def load(
        self,
        *,
        factor_id: str,
        factor_version: str,
        start_date: date,
        end_date: date,
    ) -> tuple[StoredFactorValue, ...]:
        identity = (factor_id.strip(), factor_version.strip())
        if not all(identity) or identity not in self.manifest.factor_versions:
            raise ValueError(f"factor materialization does not contain {identity}")
        if start_date > end_date:
            raise ValueError("factor read range is inverted")
        if start_date < date.fromisoformat(
            self.manifest.start_date
        ) or end_date > date.fromisoformat(self.manifest.end_date):
            raise ValueError("factor read range is outside the materialization")

        suffix = f"factor={identity[0]}-version={identity[1]}.parquet"
        selected = tuple(
            item
            for item in self.manifest.files
            if item.path.endswith(suffix)
            and _partition_overlaps(item.path, start_date=start_date, end_date=end_date)
        )
        if not selected:
            raise ValueError("factor materialization has no matching partitions")
        paths: list[str] = []
        for item in selected:
            path = self.release_directory / item.path
            if _file_hash(path) != item.sha256:
                raise ValueError(f"factor partition content hash mismatch: {item.path}")
            paths.append(str(path))

        connection = duckdb.connect(":memory:")
        try:
            rows = connection.execute(
                """
                SELECT trade_date, ts_code, value, data_release_id
                FROM read_parquet(?)
                WHERE factor_id = ?
                  AND factor_version = ?
                  AND trade_date BETWEEN ? AND ?
                  AND is_valid
                  AND isfinite(value)
                  AND right(ts_code, 2) IN ('SH', 'SZ')
                ORDER BY trade_date, ts_code
                """,
                [
                    paths,
                    identity[0],
                    identity[1],
                    start_date,
                    end_date,
                ],
            ).fetchall()
        finally:
            connection.close()
        if not rows:
            raise ValueError("factor materialization returned no valid values")
        output: list[StoredFactorValue] = []
        previous: tuple[date, str] | None = None
        for trade_date, ts_code, value, data_release_id in rows:
            key = (trade_date, str(ts_code))
            if key == previous:
                raise ValueError(f"duplicate materialized factor key: {key}")
            if str(data_release_id) != self.manifest.data_release_id:
                raise ValueError("factor row data release does not match manifest")
            output.append(StoredFactorValue(trade_date, str(ts_code), float(value)))
            previous = key
        return tuple(output)


def _partition_overlaps(path: str, *, start_date: date, end_date: date) -> bool:
    parts = Path(path).parts
    year = next(int(item.split("=", 1)[1]) for item in parts if item.startswith("year="))
    month = next(int(item.split("=", 1)[1]) for item in parts if item.startswith("month="))
    return (
        (start_date.year, start_date.month)
        <= (year, month)
        <= (
            end_date.year,
            end_date.month,
        )
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
