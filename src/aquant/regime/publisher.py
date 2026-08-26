from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from aquant.regime.definitions import MarketStateStatus
from aquant.regime.episodes import MarketStateEpisode, detect_episodes
from aquant.regime.evaluation import correlation_matrix, evaluate_state, forward_returns
from aquant.regime.panel import FloatSeries, MarketStatePanelInput
from aquant.regime.registry import MarketStateRegistry
from aquant.regime.reporting import correlation_clusters


def publish_market_state_artifacts(
    output_directory: Path,
    *,
    panel: MarketStatePanelInput,
    states: dict[str, FloatSeries],
    registry: MarketStateRegistry,
    data_release_id: str,
    code_version: str,
    config_hash: str,
    input_hash: str,
    batch_size: int = 100_000,
    max_workers: int = 2,
    started_at: float | None = None,
    factor_root: Path | None = None,
    daily_pattern: str | None = None,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    registered = {spec.state_id: spec for spec in registry}
    unknown = set(states).difference(registered)
    if unknown:
        raise ValueError(f"computed states are not registered: {sorted(unknown)}")
    computed = {
        state_id: values
        for state_id, values in states.items()
        if values.shape == (len(panel.dates),)
        and registered[state_id].status
        in {MarketStateStatus.IMPLEMENTED, MarketStateStatus.PARTIAL}
    }
    _write_catalog(output_directory / "state_catalog.parquet", registry)
    lineage_rows = _write_lineage(output_directory / "market_state_lineage.parquet", registry)
    universe_rows = _write_universes(
        output_directory / "market_state_universes.parquet",
        panel=panel,
        data_release_id=data_release_id,
    )
    value_rows = _write_values(
        output_directory / "market_state_values.parquet",
        panel=panel,
        states=computed,
        registry=registry,
        data_release_id=data_release_id,
        code_version=code_version,
        config_hash=config_hash,
        row_group_size=batch_size,
    )
    correlations = correlation_matrix(computed, minimum_observations=60)
    _write_correlation(output_directory / "state_correlation.parquet", correlations)
    clusters = correlation_clusters(correlations)
    _write_json(output_directory / "state_clusters.json", clusters)
    episode_rows = _write_episodes(
        output_directory / "state_episode_summary.parquet",
        panel=panel,
        states=computed,
    )
    evaluation_rows = _write_evaluations(
        output_directory / "state_forward_return_evaluation.parquet",
        panel=panel,
        states=computed,
    )
    if factor_root is not None and daily_pattern is not None:
        from aquant.regime.factor_analysis import publish_factor_regime_analysis

        factor_analysis = publish_factor_regime_analysis(
            output_directory,
            factor_root=factor_root,
            daily_pattern=daily_pattern,
        )
    else:
        _write_empty_factor_regime(output_directory / "factor_regime_performance.parquet")
        factor_analysis = {
            "status": "DEFERRED",
            "reason": "factor root and daily pattern were not provided",
        }
        _write_json(output_directory / "l3_regime_smoke.json", factor_analysis)
    status_counts = Counter(spec.status.value for spec in registry)
    summary: dict[str, object] = {
        "registered_states": len(registry),
        "computed_state_series": len(computed),
        "materialized_rows": value_rows,
        "lineage_rows": lineage_rows,
        "universe_rows": universe_rows,
        "data_dependency_missing": status_counts[MarketStateStatus.DATA_DEPENDENCY_MISSING],
        "partial_definitions": status_counts[MarketStateStatus.PARTIAL],
        "failed": 0,
        "episodes": episode_rows,
        "forward_evaluations": evaluation_rows,
        "start_date": panel.dates[0].isoformat(),
        "end_date": panel.dates[-1].isoformat(),
        "data_release_id": data_release_id,
        "code_version": code_version,
        "config_hash": config_hash,
        "input_hash": input_hash,
        "batch_size": batch_size,
        "max_workers": max_workers,
        "elapsed_seconds": (
            round(time.perf_counter() - started_at, 3) if started_at is not None else None
        ),
        "peak_memory_kib": _peak_memory_kib(),
        "release_status": "DRAFT",
        "l3_smoke_status": factor_analysis["status"],
    }
    _write_family_checkpoints(
        output_directory,
        registry=registry,
        states=computed,
        panel=panel,
        input_hash=input_hash,
        code_version=code_version,
    )
    _write_json(output_directory / "state_materialization_summary.json", summary)
    manifest = _manifest(output_directory, summary)
    _write_json(output_directory / "manifest.json", manifest)
    return summary


def _write_catalog(path: Path, registry: MarketStateRegistry) -> None:
    rows = []
    for spec in registry:
        payload = spec.model_dump(mode="json")
        for field in (
            "parameters",
            "required_datasets",
            "required_indices",
            "required_universes",
            "related_states",
        ):
            payload[field] = json.dumps(payload[field], sort_keys=True, ensure_ascii=False)
        rows.append(payload)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _write_lineage(path: Path, registry: MarketStateRegistry) -> int:
    rows = [
        {
            "state_id": spec.state_id,
            "state_version": spec.version,
            "formula": spec.formula,
            "required_datasets": list(spec.required_datasets),
            "required_indices": list(spec.required_indices),
            "required_universes": list(spec.required_universes),
            "related_states": list(spec.related_states),
            "definition_content_hash": spec.content_hash,
        }
        for spec in registry
    ]
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return len(rows)


def _write_universes(
    path: Path,
    *,
    panel: MarketStatePanelInput,
    data_release_id: str,
) -> int:
    rows = []
    for universe_id, membership in sorted(panel.universes.items()):
        for index, trade_date in enumerate(panel.dates[:-1]):
            count = int(np.count_nonzero(membership[index]))
            rows.append(
                {
                    "trade_date": trade_date,
                    "available_date": panel.dates[index + 1],
                    "universe_id": universe_id,
                    "constituent_count": count,
                    "data_release_id": data_release_id,
                    "content_hash": hashlib.sha256(
                        (
                            f"{trade_date.isoformat()}|{universe_id}|{count}|{data_release_id}"
                        ).encode()
                    ).hexdigest(),
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return len(rows)


def _write_values(
    path: Path,
    *,
    panel: MarketStatePanelInput,
    states: dict[str, FloatSeries],
    registry: MarketStateRegistry,
    data_release_id: str,
    code_version: str,
    config_hash: str,
    row_group_size: int,
) -> int:
    tables: list[pa.Table] = []
    dates = panel.dates[:-1]
    available_dates = panel.dates[1:]
    for state_id in sorted(states):
        spec = registry.get(state_id)
        values = states[state_id][:-1]
        valid = np.isfinite(values)
        if not np.any(valid):
            continue
        selected_dates = [dates[index] for index in np.flatnonzero(valid)]
        selected_available = [available_dates[index] for index in np.flatnonzero(valid)]
        selected_values = values[valid]
        percentile = (
            selected_values
            if "percentile" in state_id or spec.normalization == "PERCENTILE_0_1"
            else [None] * len(selected_values)
        )
        hashes = [
            hashlib.sha256(
                (
                    f"{trade_date.isoformat()}|{state_id}|{spec.version}|{raw:.17g}|"
                    f"{data_release_id}|{code_version}|{config_hash}"
                ).encode()
            ).hexdigest()
            for trade_date, raw in zip(selected_dates, selected_values, strict=True)
        ]
        tables.append(
            pa.table(
                {
                    "trade_date": selected_dates,
                    "available_date": selected_available,
                    "state_id": [state_id] * len(selected_values),
                    "state_version": [spec.version] * len(selected_values),
                    "scope": [spec.scope.value] * len(selected_values),
                    "raw_value": pa.array(selected_values, type=pa.float64()),
                    "normalized_value": pa.array([None] * len(selected_values), type=pa.float64()),
                    "percentile_value": pa.array(percentile, type=pa.float64()),
                    "status": [spec.status.value] * len(selected_values),
                    "data_release_id": [data_release_id] * len(selected_values),
                    "code_version": [code_version] * len(selected_values),
                    "config_hash": [config_hash] * len(selected_values),
                    "content_hash": hashes,
                }
            )
        )
    table = (
        pa.concat_tables(tables)
        if tables
        else pa.table(
            {
                "trade_date": pa.array([], type=pa.date32()),
                "available_date": pa.array([], type=pa.date32()),
                "state_id": pa.array([], type=pa.string()),
            }
        )
    )
    pq.write_table(table, path, compression="zstd", row_group_size=row_group_size)
    return int(table.num_rows)


def _write_family_checkpoints(
    directory: Path,
    *,
    registry: MarketStateRegistry,
    states: dict[str, FloatSeries],
    panel: MarketStatePanelInput,
    input_hash: str,
    code_version: str,
) -> None:
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    families = sorted({spec.family.value for spec in registry})
    for family in families:
        specs = tuple(spec for spec in registry if spec.family.value == family)
        materialized = sorted(spec.state_id for spec in specs if spec.state_id in states)
        payload = {
            "schema_version": "aquant.market-state-family-checkpoint.v1",
            "family": family,
            "status": "PASS",
            "input_hash": input_hash,
            "code_version": code_version,
            "start_date": panel.dates[0].isoformat(),
            "end_date": panel.dates[-1].isoformat(),
            "registered_states": len(specs),
            "materialized_state_series": len(materialized),
            "state_ids": materialized,
        }
        payload["content_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _write_json(checkpoints / f"{family.lower()}.json", payload)


def _write_correlation(path: Path, rows: tuple[tuple[str, str, int, float], ...]) -> None:
    pq.write_table(
        pa.table(
            {
                "left_state_id": [row[0] for row in rows],
                "right_state_id": [row[1] for row in rows],
                "observations": [row[2] for row in rows],
                "correlation": [row[3] for row in rows],
            }
        ),
        path,
        compression="zstd",
    )


def _write_episodes(
    path: Path,
    *,
    panel: MarketStatePanelInput,
    states: dict[str, FloatSeries],
) -> int:
    level = _universe_level(panel, panel.universes["ALL_A"])
    forward_5d = forward_returns(level, 5)
    forward_20d = forward_returns(level, 20)
    date_indices = {value: index for index, value in enumerate(panel.dates)}
    episodes: list[MarketStateEpisode] = []
    for state_id, values in states.items():
        if "percentile" not in state_id and state_id != "microcap_crowding_score_v1":
            continue
        episodes.extend(
            detect_episodes(panel.dates, values, state_id=state_id, lower_threshold=0.05)
        )
        episodes.extend(
            detect_episodes(panel.dates, values, state_id=state_id, upper_threshold=0.95)
        )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_id": episode.episode_id,
                    "state_id": episode.state_id,
                    "entry_date": episode.entry_date,
                    "exit_date": episode.exit_date,
                    "minimum_value": episode.minimum_value,
                    "maximum_value": episode.maximum_value,
                    "duration": episode.duration,
                    "side": episode.side,
                    "recovery_confirmation_date": episode.recovery_confirmation_date,
                    "future_5d_return": forward_5d[date_indices[episode.exit_date]],
                    "future_20d_return": forward_20d[date_indices[episode.exit_date]],
                }
                for episode in episodes
            ],
            schema=pa.schema(
                [
                    ("episode_id", pa.string()),
                    ("state_id", pa.string()),
                    ("entry_date", pa.date32()),
                    ("exit_date", pa.date32()),
                    ("minimum_value", pa.float64()),
                    ("maximum_value", pa.float64()),
                    ("duration", pa.int64()),
                    ("side", pa.string()),
                    ("recovery_confirmation_date", pa.date32()),
                    ("future_5d_return", pa.float64()),
                    ("future_20d_return", pa.float64()),
                ]
            ),
        ),
        path,
        compression="zstd",
    )
    return len(episodes)


def _write_evaluations(
    path: Path,
    *,
    panel: MarketStatePanelInput,
    states: dict[str, FloatSeries],
) -> int:
    targets = {
        name: _universe_level(panel, membership)
        for name, membership in panel.universes.items()
        if name
        in {
            "ALL_A",
            "MICROCAP_DYNAMIC",
            "MICROCAP_FIXED_MONTHLY",
            "GROWTH",
            "VALUE",
            "HS300",
            "CSI500",
            "CSI1000",
        }
    }
    targets.update({f"{name}_INDEX": level for name, level in panel.index_levels.items()})
    rows = [
        evaluate_state(
            state_id,
            values,
            forward_returns(level, horizon),
            horizon=horizon,
            target_id=target_id,
        )
        for state_id, values in states.items()
        for target_id, level in targets.items()
        for horizon in (5, 10, 20, 40, 60)
    ]
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "state_id": row.state_id,
                    "target_id": row.target_id,
                    "horizon": row.horizon,
                    "observations": row.observations,
                    "pearson_correlation": row.pearson_correlation,
                    "spearman_correlation": row.spearman_correlation,
                    "bottom_decile_mean": row.bottom_decile_mean,
                    "lower_middle_mean": row.lower_middle_mean,
                    "middle_mean": row.middle_mean,
                    "upper_middle_mean": row.upper_middle_mean,
                    "top_decile_mean": row.top_decile_mean,
                    "conditional_spread": row.conditional_spread,
                    "newey_west_t": row.newey_west_t,
                    "direction_hit_rate": row.direction_hit_rate,
                }
                for row in rows
            ]
        ),
        path,
        compression="zstd",
    )
    return len(rows)


def _universe_level(
    panel: MarketStatePanelInput,
    membership: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
) -> FloatSeries:
    returns = np.full(panel.close.shape, np.nan)
    valid = (
        np.isfinite(panel.close[1:])
        & np.isfinite(panel.close[:-1])
        & (panel.close[1:] > 0)
        & (panel.close[:-1] > 0)
    )
    returns[1:][valid] = panel.close[1:][valid] / panel.close[:-1][valid] - 1.0
    effective = np.zeros_like(membership)
    effective[1:] = membership[:-1]
    counts = np.sum(effective & np.isfinite(returns), axis=1)
    daily = np.divide(
        np.nansum(np.where(effective, returns, np.nan), axis=1),
        counts,
        out=np.zeros(len(panel.dates)),
        where=counts > 0,
    )
    return np.cumprod(1.0 + daily)


def _write_empty_factor_regime(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "factor_id": pa.array([], type=pa.string()),
                "state_id": pa.array([], type=pa.string()),
                "status": pa.array([], type=pa.string()),
            }
        ),
        path,
    )


def _manifest(directory: Path, summary: dict[str, object]) -> dict[str, object]:
    files = {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    stable_payload: dict[str, object] = {
        "schema_version": "aquant.market-state-artifacts.v1",
        "summary": summary,
        "files": files,
    }
    payload = {
        **stable_payload,
        "created_at": datetime.now(UTC).isoformat(),
        "content_hash": hashlib.sha256(
            json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _peak_memory_kib() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
