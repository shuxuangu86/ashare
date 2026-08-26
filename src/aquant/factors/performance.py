import hashlib
import json
import os
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aquant.factors.atomic import AtomicFactor, FactorPanelInput


@dataclass(frozen=True, slots=True)
class PerformanceResult:
    scenario: str
    data_release_id: str
    code_version: str
    config_hash: str
    elapsed_seconds: float
    cpu_seconds: float
    cpu_utilization_percent: float
    peak_memory_bytes: int
    disk_read_bytes: int
    disk_write_bytes: int
    factor_workloads: int
    unique_computations: int
    cache_hits: int
    cache_hit_rate: float
    factor_rows_per_second: float
    panel_bytes: int
    factor_store_bytes: int
    factor_seconds: tuple[tuple[str, float], ...]
    content_hash: str

    def payload(self) -> dict[str, object]:
        return asdict(self)


def benchmark_factor_workload(
    *,
    scenario: str,
    panel: FactorPanelInput,
    factors: tuple[AtomicFactor, ...],
    data_release_id: str,
    code_version: str,
    config_hash: str,
    factor_store: Path | None = None,
    output_date_count: int | None = None,
) -> PerformanceResult:
    if not scenario.strip() or not factors:
        raise ValueError("benchmark scenario and factors are required")
    before_io = _process_io()
    before_cpu = time.process_time()
    started = time.perf_counter()
    seen: set[str] = set()
    factor_seconds: list[tuple[str, float]] = []
    checksum = hashlib.sha256()
    cache_hits = 0
    for factor in factors:
        key = factor.spec.expression_hash
        if key in seen:
            cache_hits += 1
            continue
        factor_started = time.perf_counter()
        values = factor.compute_array(panel)
        factor_seconds.append((factor.spec.factor_id, time.perf_counter() - factor_started))
        checksum.update(np.nan_to_num(values, nan=0).tobytes())
        seen.add(key)
    elapsed = time.perf_counter() - started
    cpu_seconds = time.process_time() - before_cpu
    after_io = _process_io()
    resolved_output_dates = output_date_count or len(panel.trade_dates)
    if not 0 < resolved_output_dates <= len(panel.trade_dates):
        raise ValueError("benchmark output date count is invalid")
    rows = resolved_output_dates * len(panel.ts_codes) * len(factors)
    panel_bytes = sum(np.asarray(values).nbytes for values in panel.fields.values())
    store_bytes = _directory_size(factor_store) if factor_store is not None else 0
    payload = {
        "checksum": checksum.hexdigest(),
        "config_hash": config_hash,
        "data_release_id": data_release_id,
        "scenario": scenario,
    }
    content_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PerformanceResult(
        scenario=scenario,
        data_release_id=data_release_id,
        code_version=code_version,
        config_hash=config_hash,
        elapsed_seconds=elapsed,
        cpu_seconds=cpu_seconds,
        cpu_utilization_percent=(
            cpu_seconds / elapsed / max(os.cpu_count() or 1, 1) * 100 if elapsed else 0
        ),
        peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        disk_read_bytes=max(0, after_io[0] - before_io[0]),
        disk_write_bytes=max(0, after_io[1] - before_io[1]),
        factor_workloads=len(factors),
        unique_computations=len(seen),
        cache_hits=cache_hits,
        cache_hit_rate=cache_hits / len(factors),
        factor_rows_per_second=rows / elapsed if elapsed else float("inf"),
        panel_bytes=panel_bytes,
        factor_store_bytes=store_bytes,
        factor_seconds=tuple(factor_seconds),
        content_hash=content_hash,
    )


def _process_io() -> tuple[int, int]:
    path = Path("/proc/self/io")
    if not path.exists():
        return (0, 0)
    values = {
        key: int(value)
        for line in path.read_text(encoding="utf-8").splitlines()
        for key, value in [line.split(":", maxsplit=1)]
    }
    return values.get("read_bytes", 0), values.get("write_bytes", 0)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())
