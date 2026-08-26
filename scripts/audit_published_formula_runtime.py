from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from aquant.factors.atomic import alpha101_original_library, gtja191_original_library
from aquant.factors.atomic.models import FactorPanelInput


def _panel(rows: int = 320, columns: int = 20) -> FactorPanelInput:
    generator = np.random.default_rng(20260731)
    timeline = np.arange(rows, dtype=float)[:, None]
    securities = np.arange(columns, dtype=float)[None, :]
    innovations = generator.normal(0.0004, 0.015, size=(rows, columns))
    close = (10 + securities) * np.exp(np.cumsum(innovations, axis=0))
    open_ = close * (1 + generator.normal(0, 0.004, size=(rows, columns)))
    spread = generator.uniform(0.005, 0.025, size=(rows, columns))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = (
        1_000_000
        + timeline * 1_000
        + securities * 10_000
        + generator.normal(0, 100_000, size=(rows, columns))
    )
    volume = np.maximum(volume, 10_000)
    amount = volume * (open_ + high + low + close) / 4
    market_cap = close * (1_000_000_000 + securities * 10_000_000)
    start = date(2020, 1, 1)
    return FactorPanelInput(
        tuple(start + timedelta(days=index) for index in range(rows)),
        tuple(f"{index:06d}.SZ" for index in range(columns)),
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "total_market_cap": market_cap,
        },
    )


def audit() -> dict[str, Any]:
    panel = _panel()
    factors = (*alpha101_original_library(), *gtja191_original_library())
    results: list[dict[str, Any]] = []
    for factor in factors:
        try:
            values = factor.compute_array(panel)
            finite_count = int(np.count_nonzero(np.isfinite(values)))
            status = "PASS" if finite_count else "DEGENERATE_ALL_NAN"
            error = None
        except (KeyError, TypeError, ValueError, IndexError, FloatingPointError) as exc:
            finite_count = 0
            status = "IMPLEMENTATION_FAILED"
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "factor_id": factor.spec.factor_id,
                "source_id": factor.spec.source_id,
                "source_formula_id": factor.spec.source_formula_id,
                "source_faithfulness": str(factor.spec.source_faithfulness),
                "declared_status": str(factor.spec.implementation_status),
                "normalization_error": factor.spec.parameters.get("normalization_error"),
                "runtime_status": status,
                "finite_count": finite_count,
                "error": error,
            }
        )
    counts: dict[str, int] = {}
    for row in results:
        key = str(row["runtime_status"])
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema_version": "aquant.published-formula-runtime-audit.v1",
        "factor_count": len(factors),
        "counts": dict(sorted(counts.items())),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit published formula runtime coverage")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/factor_library_v2/published_formula_runtime_audit.json"),
    )
    args = parser.parse_args()
    payload = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["counts"], sort_keys=True))
    return 0 if payload["counts"].get("IMPLEMENTATION_FAILED", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
