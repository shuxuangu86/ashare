from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

_FACTORS = (
    "microcap_momentum_20d",
    "growth_to_value",
    "growth_quality",
    "turnover_shock_60d",
    "residual_reversal_5d",
)
_STATES = (
    "growth_minus_value_return_40d",
    "all_a_return_20d",
    "microcap_pb_percentile_3y",
    "all_a_above_ma20_ratio",
    "microcap_crowding_score_v1",
)


def publish_factor_regime_analysis(
    artifact_directory: Path,
    *,
    factor_root: Path,
    daily_pattern: str,
) -> dict[str, object]:
    factor_glob = str(
        factor_root / "materialization=*" / "factor_family=*" / "year=*" / "month=*" / "*.parquet"
    )
    state_path = artifact_directory / "market_state_values.parquet"
    connection = duckdb.connect(":memory:")
    connection.execute("SET memory_limit = '2GB'")
    connection.execute("SET threads = 2")
    try:
        performance = connection.execute(
            _performance_query(),
            [daily_pattern, factor_glob, str(state_path), list(_STATES)],
        ).fetch_arrow_table()
        pq.write_table(
            performance,
            artifact_directory / "factor_regime_performance.parquet",
            compression="zstd",
        )
        frame = connection.execute(
            _smoke_query(),
            [
                daily_pattern,
                factor_glob,
                list(_FACTORS),
                str(state_path),
                list(_STATES),
            ],
        ).fetch_df()
    finally:
        connection.close()
    smoke = _walk_forward_smoke(frame)
    smoke["factor_regime_rows"] = int(performance.num_rows)
    (artifact_directory / "l3_regime_smoke.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return smoke


def _performance_query() -> str:
    return """
        WITH price_history AS (
            SELECT
                trade_date,
                ts_code,
                close,
                lead(close, 20) OVER (PARTITION BY ts_code ORDER BY trade_date) / close - 1
                    AS forward_20d_return
            FROM read_parquet(?)
            WHERE trade_date >= DATE '2021-01-01'
        ),
        ranked_factors AS (
            SELECT
                factor_id,
                factor.trade_date,
                factor.ts_code,
                percent_rank() OVER (
                    PARTITION BY factor_id, factor.trade_date
                    ORDER BY value
                ) AS factor_rank,
                price.forward_20d_return
            FROM read_parquet(?, hive_partitioning = true) AS factor
            JOIN price_history AS price USING (trade_date, ts_code)
            WHERE factor.is_valid
              AND factor.trade_date >= DATE '2022-01-01'
              AND price.forward_20d_return IS NOT NULL
        ),
        daily_ic AS (
            SELECT
                factor_id,
                trade_date,
                corr(factor_rank, forward_20d_return) AS rank_ic
            FROM ranked_factors
            GROUP BY factor_id, trade_date
            HAVING count(*) >= 100
        ),
        filtered_ic AS (
            SELECT *
            FROM daily_ic
            WHERE isfinite(rank_ic)
        ),
        ranked_states AS (
            SELECT
                state_id,
                trade_date,
                CASE
                    WHEN state_percentile <= 0.10 THEN '00_10'
                    WHEN state_percentile <= 0.30 THEN '10_30'
                    WHEN state_percentile <= 0.70 THEN '30_70'
                    WHEN state_percentile <= 0.90 THEN '70_90'
                    ELSE '90_100'
                END AS regime_bucket
            FROM (
                SELECT
                    state_id,
                    trade_date,
                    percent_rank() OVER (PARTITION BY state_id ORDER BY raw_value)
                        AS state_percentile
                FROM read_parquet(?)
                WHERE state_id IN (SELECT unnest(?))
            )
        )
        SELECT
            factor_id,
            state_id,
            regime_bucket,
            count(*) AS observations,
            avg(rank_ic) AS mean_rank_ic,
            stddev_samp(rank_ic) AS rank_ic_std,
            avg(rank_ic) / nullif(stddev_samp(rank_ic), 0) AS rank_icir,
            avg(CASE WHEN rank_ic > 0 THEN 1.0 ELSE 0.0 END) AS ic_hit_rate
        FROM filtered_ic
        JOIN ranked_states USING (trade_date)
        GROUP BY factor_id, state_id, regime_bucket
        ORDER BY factor_id, state_id, regime_bucket
    """


def _smoke_query() -> str:
    factor_columns = ", ".join(
        f"max(value) FILTER (factor_id = '{factor}') AS {factor}" for factor in _FACTORS
    )
    state_columns = ", ".join(
        f"max(raw_value) FILTER (state_id = '{state}') AS {state}" for state in _STATES
    )
    return f"""
        WITH price_history AS (
            SELECT
                trade_date,
                ts_code,
                lead(close, 20) OVER (PARTITION BY ts_code ORDER BY trade_date) / close - 1
                    AS target
            FROM read_parquet(?)
            WHERE trade_date >= DATE '2021-01-01'
        ),
        factor_wide AS (
            SELECT trade_date, ts_code, {factor_columns}
            FROM read_parquet(?, hive_partitioning = true)
            WHERE factor_id IN (SELECT unnest(?))
              AND is_valid
              AND trade_date >= DATE '2022-01-01'
            GROUP BY trade_date, ts_code
        ),
        state_wide AS (
            SELECT trade_date, {state_columns}
            FROM read_parquet(?)
            WHERE state_id IN (SELECT unnest(?))
              AND trade_date >= DATE '2022-01-01'
            GROUP BY trade_date
        )
        SELECT factor_wide.*, state_wide.* EXCLUDE (trade_date), price.target
        FROM factor_wide
        JOIN state_wide USING (trade_date)
        JOIN price_history AS price USING (trade_date, ts_code)
        WHERE price.target IS NOT NULL
        ORDER BY trade_date, ts_code
    """


def _walk_forward_smoke(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"status": "DATA_DEPENDENCY_MISSING", "reason": "aligned factor panel is empty"}
    dates = np.asarray(sorted(frame["trade_date"].unique()))
    boundaries = (int(len(dates) * 0.7), int(len(dates) * 0.85), len(dates))
    model_columns = {
        "A_STOCK_FACTORS": list(_FACTORS),
        "B_FACTORS_PLUS_STATES": [*_FACTORS, *_STATES],
        "C_REGISTERED_INTERACTIONS": [*_FACTORS, *_STATES],
    }
    interaction_pairs = (
        ("microcap_momentum_20d", "all_a_return_20d"),
        ("growth_to_value", "microcap_pb_percentile_3y"),
        ("residual_reversal_5d", "microcap_crowding_score_v1"),
    )
    results: dict[str, list[float]] = {name: [] for name in model_columns}
    for test_start, test_end in ((boundaries[0], boundaries[1]), (boundaries[1], boundaries[2])):
        train = frame[frame["trade_date"] < dates[test_start]].copy()
        test_mask = frame["trade_date"] >= dates[test_start]
        if test_end < len(dates):
            test_mask &= frame["trade_date"] < dates[test_end]
        test = frame[test_mask].copy()
        for model_name, columns in model_columns.items():
            train_x = train[columns].copy()
            test_x = test[columns].copy()
            if model_name == "C_REGISTERED_INTERACTIONS":
                for left, right in interaction_pairs:
                    name = f"{left}_x_{right}"
                    train_x[name] = train[left] * train[right]
                    test_x[name] = test[left] * test[right]
            medians = train_x.median()
            train_values = train_x.fillna(medians).to_numpy(dtype=np.float64)
            test_values = test_x.fillna(medians).to_numpy(dtype=np.float64)
            train_valid = np.isfinite(train["target"].to_numpy())
            test_valid = np.isfinite(test["target"].to_numpy())
            scaler = StandardScaler().fit(train_values[train_valid])
            model = Ridge(alpha=10.0).fit(
                scaler.transform(train_values[train_valid]),
                train.loc[train_valid, "target"],
            )
            predictions = model.predict(scaler.transform(test_values[test_valid]))
            evaluated = test.loc[test_valid, ["trade_date", "target"]].copy()
            evaluated["prediction"] = predictions
            daily_ic = evaluated.groupby("trade_date", sort=True).apply(
                lambda values: values["prediction"].corr(values["target"], method="spearman"),
                include_groups=False,
            )
            results[model_name].extend(daily_ic.dropna().tolist())
    models = {
        name: {
            "oos_rank_ic": float(np.mean(values)) if values else None,
            "oos_rank_icir": (
                float(np.mean(values) / np.std(values)) if values and np.std(values) > 0 else None
            ),
            "ic_hit_rate": float(np.mean(np.asarray(values) > 0)) if values else None,
            "daily_observations": len(values),
        }
        for name, values in results.items()
    }
    baseline = models["A_STOCK_FACTORS"]["oos_rank_ic"]
    for metrics in models.values():
        value = metrics["oos_rank_ic"]
        metrics["rank_ic_increment_vs_A"] = (
            float(value - baseline) if value is not None and baseline is not None else None
        )
    return {
        "status": "RESEARCH_ONLY",
        "split": "EXPANDING_WALK_FORWARD_2_FOLDS",
        "random_split": False,
        "train_only_standardization": True,
        "models": models,
        "factors": list(_FACTORS),
        "states": list(_STATES),
    }
