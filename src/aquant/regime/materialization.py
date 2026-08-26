from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from aquant.regime.definitions import MarketStateSpec, MarketStateValue
from aquant.regime.protocol import Numeric


def materialize_state_values(
    spec: MarketStateSpec,
    dates: Sequence[date],
    available_dates: Sequence[date],
    raw_values: Sequence[Numeric],
    *,
    data_release_id: str,
    code_version: str,
    config_hash: str,
    normalized_values: Sequence[Numeric] | None = None,
    percentile_values: Sequence[Numeric] | None = None,
) -> tuple[MarketStateValue, ...]:
    """Bind values to lineage while requiring an explicit post-trade availability date."""
    expected = len(dates)
    resolved_normalized = normalized_values if normalized_values is not None else raw_values
    resolved_percentiles = percentile_values if percentile_values is not None else [None] * expected
    lengths = {
        "available_dates": len(available_dates),
        "raw_values": len(raw_values),
        "normalized_values": len(resolved_normalized),
        "percentile_values": len(resolved_percentiles),
    }
    mismatched = sorted(name for name, length in lengths.items() if length != expected)
    if mismatched:
        raise ValueError(f"materialization input length mismatch: {mismatched}")
    return tuple(
        MarketStateValue(
            trade_date=trade_date,
            available_date=available_date,
            state_id=spec.state_id,
            state_version=spec.version,
            scope=spec.scope,
            raw_value=None if raw is None else float(raw),
            normalized_value=None if normalized is None else float(normalized),
            percentile_value=None if percentile is None else float(percentile),
            status=spec.status,
            data_release_id=data_release_id,
            code_version=code_version,
            config_hash=config_hash,
        )
        for trade_date, available_date, raw, normalized, percentile in zip(
            dates,
            available_dates,
            raw_values,
            resolved_normalized,
            resolved_percentiles,
            strict=True,
        )
    )
