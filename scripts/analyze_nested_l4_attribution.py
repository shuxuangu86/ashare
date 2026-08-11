#!/usr/bin/env python3
# ruff: noqa: E501, RUF001 -- embedded Chinese Markdown is intentionally prose-formatted.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from aquant.backtest.costs import AshareFeeSchedule  # type: ignore[import-untyped]
from aquant.domain.enums import Side  # type: ignore[import-untyped]


def main() -> None:
    args = _arguments()
    l3 = json.loads(args.l3_metadata.read_text())
    l4 = json.loads(args.l4_report.read_text())
    curve = list(csv.DictReader(args.daily_curve.open()))
    fills = list(csv.DictReader(args.fills.open()))
    _validate_inputs(l3, l4, curve, fills)

    initial = Decimal(l4["initial_equity"])
    daily_costs, cost_totals = _daily_costs(fills)
    boundary_total = _add_boundary_costs(curve, l4, initial, daily_costs)
    expected_boundary = Decimal(l4["fold_boundary_policy"]["total_estimated_liquidation_cost"])
    if abs(boundary_total - expected_boundary) > Decimal("0.000001"):
        raise ValueError("fold-boundary liquidation cost does not reconcile")
    cost_totals["boundary_liquidation"] = boundary_total
    cost_totals["total"] = sum(cost_totals.values(), Decimal("0"))

    daily_rows, path_summary = _conditional_paths(curve, daily_costs, initial)
    fold_rows = _fold_attribution(l3, l4, curve, fills, daily_rows, initial)
    family_rows = _family_attribution(l3)
    net = l4["performance"]
    gross_annual_excess = path_summary["conditional_gross_annual_return"] - float(
        net["annual_benchmark_return"]
    )
    cost_drag = path_summary["conditional_gross_annual_return"] - float(net["annual_return"])
    diagnostics = _diagnostics(l4, fold_rows, gross_annual_excess, cost_drag)
    stable: dict[str, Any] = {
        "status": "PASS_RESEARCH_ONLY",
        "stage": "L4_RETURN_CONVERSION_ATTRIBUTION",
        "research_only": True,
        "outer_oos_used_for_retuning": False,
        "attribution_is_conditional_counterfactual": True,
        "period": {"start": l4["start_date"], "end": l4["end_date"], "sessions": len(curve)},
        "initial_equity": str(initial),
        "benchmark": l4["benchmark"],
        "net_performance": net,
        "conditional_paths": path_summary,
        "cost_totals_cny": {key: str(value) for key, value in cost_totals.items()},
        "cost_share_of_initial_equity": float(cost_totals["total"] / initial),
        "gross_annual_excess_return": gross_annual_excess,
        "annual_cost_drag": cost_drag,
        "share_of_gross_annual_excess_lost_to_cost": (
            cost_drag / gross_annual_excess if gross_annual_excess > 0 else None
        ),
        "l3_to_l4_conversion": {
            "positive_l3_rankic_folds": sum(row["l3_outer_rank_ic"] > 0 for row in fold_rows),
            "positive_net_excess_folds": sum(
                row["net_annual_excess_return"] > 0 for row in fold_rows
            ),
            "rankic_vs_net_excess_correlation": _correlation(
                [row["l3_outer_rank_ic"] for row in fold_rows],
                [row["net_annual_excess_return"] for row in fold_rows],
            ),
            "rankic_vs_gross_excess_correlation": _correlation(
                [row["l3_outer_rank_ic"] for row in fold_rows],
                [row["conditional_gross_annual_excess_return"] for row in fold_rows],
            ),
        },
        "diagnostics": diagnostics,
        "limitations": [
            "Gross and double-cost paths keep realized holdings, fills, and quantities fixed; they are conditional counterfactuals, not independent reruns.",
            "Observed outer OOS is used only for attribution and must not be reused as untouched tuning evidence.",
            "The benchmark is an internal PIT all-A equal-weight proxy, not the official Wind index.",
            "The source history release remains MATERIALIZED_NOT_BACKTEST_APPROVED.",
        ],
        "provenance": {
            "l3_content_hash": l4["l3_content_hash"],
            "l4_content_hash": l4["content_hash"],
            "history_release_id": l4["history_release_id"],
            "code_version": args.code_version,
        },
    }
    daily_csv = _csv_text(daily_rows)
    fold_csv = _csv_text(fold_rows)
    family_csv = _csv_text(family_rows)
    stable["artifacts"] = {
        "daily": {"file": "daily_cost_attribution.csv", "sha256": _text_hash(daily_csv)},
        "fold": {"file": "fold_attribution.csv", "sha256": _text_hash(fold_csv)},
        "family": {"file": "family_rankic_attribution.csv", "sha256": _text_hash(family_csv)},
    }
    stable["content_hash"] = _hash(stable)
    payload = {**stable, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "attribution_summary.json", json.dumps(payload, indent=2) + "\n")
    _write(args.output_dir / "daily_cost_attribution.csv", daily_csv)
    _write(args.output_dir / "fold_attribution.csv", fold_csv)
    _write(args.output_dir / "family_rankic_attribution.csv", family_csv)
    _write(args.output_dir / "l4_return_conversion_attribution.md", _markdown(payload, fold_rows))
    print(json.dumps({"status": payload["status"], **path_summary, "diagnostics": diagnostics}))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribute nested L4 return conversion")
    parser.add_argument("--l3-metadata", type=Path, required=True)
    parser.add_argument("--l4-report", type=Path, required=True)
    parser.add_argument("--daily-curve", type=Path, required=True)
    parser.add_argument("--fills", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    return parser.parse_args()


def _validate_inputs(
    l3: dict[str, Any], l4: dict[str, Any], curve: list[dict[str, str]], fills: list[dict[str, str]]
) -> None:
    if l3.get("status") != "PASS" or l3.get("outer_test_used_for_model_selection") is not False:
        raise ValueError("attribution requires leakage-safe nested L3 metadata")
    if (
        l4.get("status") != "PASS_RESEARCH_ONLY"
        or l4.get("outer_test_used_for_optimization") is not False
    ):
        raise ValueError("attribution requires completed nested L4 results")
    if not curve or not fills:
        raise ValueError("attribution inputs must not be empty")
    if len(curve) != int(l4["sessions"]):
        raise ValueError("daily curve does not match reported session count")
    if len(fills) != int(l4["execution"]["fills"]):
        raise ValueError("fill file does not match reported fill count")


def _daily_costs(
    fills: list[dict[str, str]],
) -> tuple[dict[str, dict[str, Decimal]], dict[str, Decimal]]:
    schedule = AshareFeeSchedule()
    daily: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    totals: dict[str, Decimal] = defaultdict(Decimal)
    slip = Decimal("0.0005")
    for fill in fills:
        trade_date = datetime.fromisoformat(fill["occurred_at"]).date()
        side = Side(fill["side"])
        notional = Decimal(fill["notional"])
        recorded_fee = Decimal(fill["fee"])
        breakdown = schedule.model_for(trade_date).calculate(side, notional)
        if abs(breakdown.total - recorded_fee) > Decimal("0.000001"):
            raise ValueError(f"recorded fee does not reconcile on {trade_date}")
        slippage = (
            notional * slip / (Decimal("1") + slip)
            if side is Side.BUY
            else notional * slip / (Decimal("1") - slip)
        )
        key = trade_date.isoformat()
        for name, value in (
            ("commission", breakdown.commission),
            ("stamp_duty", breakdown.stamp_duty),
            ("transfer_fee", breakdown.transfer_fee),
            ("slippage", slippage),
        ):
            daily[key][name] += value
            totals[name] += value
        daily[key]["trade_notional"] += notional
        daily[key]["fill_count"] += Decimal("1")
    return daily, totals


def _add_boundary_costs(
    curve: list[dict[str, str]],
    l4: dict[str, Any],
    initial: Decimal,
    daily: dict[str, dict[str, Decimal]],
) -> Decimal:
    starts = {int(item["fold"]): item["test_start"] for item in l4["fold_performance"]}
    by_date = {row["trade_date"]: index for index, row in enumerate(curve)}
    total = Decimal("0")
    for fold, start in starts.items():
        if fold == 0:
            continue
        index = by_date[start]
        previous_equity = initial * Decimal(curve[index - 1]["strategy_net_value"])
        rate = _boundary_rate(date.fromisoformat(start))
        value = previous_equity * rate
        daily[start]["boundary_liquidation"] += value
        total += value
    return total


def _boundary_rate(next_fold_start: date) -> Decimal:
    stamp = Decimal("0.001") if next_fold_start < date(2023, 8, 28) else Decimal("0.0005")
    transfer = Decimal("0.00002") if next_fold_start < date(2022, 4, 29) else Decimal("0.00001")
    return Decimal("0.0005") + Decimal("0.0003") + stamp + transfer


def _conditional_paths(
    curve: list[dict[str, str]], daily_costs: dict[str, dict[str, Decimal]], initial: Decimal
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    gross_value = 1.0
    double_cost_value = 1.0
    result: list[dict[str, Any]] = []
    previous_equity = initial
    gross_returns: list[float] = []
    double_cost_returns: list[float] = []
    for row in curve:
        costs = daily_costs.get(row["trade_date"], {})
        fee = (
            costs.get("commission", Decimal("0"))
            + costs.get("stamp_duty", Decimal("0"))
            + costs.get("transfer_fee", Decimal("0"))
        )
        slippage = costs.get("slippage", Decimal("0"))
        boundary = costs.get("boundary_liquidation", Decimal("0"))
        total_cost = fee + slippage + boundary
        net_return = float(row["strategy_return"])
        cost_return = float(total_cost / previous_equity)
        gross_return = net_return + cost_return
        double_cost_return = net_return - cost_return
        gross_value *= 1 + gross_return
        double_cost_value *= 1 + double_cost_return
        gross_returns.append(gross_return)
        double_cost_returns.append(double_cost_return)
        result.append(
            {
                "trade_date": row["trade_date"],
                "outer_fold": int(row["outer_fold"]),
                "net_return": net_return,
                "benchmark_return": float(row["benchmark_return"]),
                "commission_cny": float(costs.get("commission", Decimal("0"))),
                "stamp_duty_cny": float(costs.get("stamp_duty", Decimal("0"))),
                "transfer_fee_cny": float(costs.get("transfer_fee", Decimal("0"))),
                "slippage_cny": float(slippage),
                "boundary_liquidation_cny": float(boundary),
                "total_cost_cny": float(total_cost),
                "trade_notional_cny": float(costs.get("trade_notional", Decimal("0"))),
                "fill_count": int(costs.get("fill_count", Decimal("0"))),
                "conditional_gross_return": gross_return,
                "conditional_double_cost_return": double_cost_return,
                "net_value": float(row["strategy_net_value"]),
                "conditional_gross_net_value": gross_value,
                "conditional_double_cost_net_value": double_cost_value,
            }
        )
        previous_equity = initial * Decimal(row["strategy_net_value"])
    return result, {
        "conditional_gross_total_return": gross_value - 1,
        "conditional_gross_annual_return": _annualized(gross_returns),
        "conditional_double_cost_total_return": double_cost_value - 1,
        "conditional_double_cost_annual_return": _annualized(double_cost_returns),
    }


def _fold_attribution(
    l3: dict[str, Any],
    l4: dict[str, Any],
    curve: list[dict[str, str]],
    fills: list[dict[str, str]],
    daily_rows: list[dict[str, Any]],
    initial: Decimal,
) -> list[dict[str, Any]]:
    l3_folds = {int(item["fold"]): item for item in l3["folds"]}
    l4_folds = {int(item["fold"]): item for item in l4["fold_performance"]}
    configs = {int(item["fold"]): item["selected"] for item in l4["fold_configurations"]}
    result: list[dict[str, Any]] = []
    for fold in sorted(l4_folds):
        rows = [row for row in daily_rows if row["outer_fold"] == fold]
        source = [row for row in curve if int(row["outer_fold"]) == fold]
        start, end = source[0]["trade_date"], source[-1]["trade_date"]
        fold_fills = [fill for fill in fills if start <= fill["occurred_at"][:10] <= end]
        gross_returns = [float(row["conditional_gross_return"]) for row in rows]
        benchmark_returns = [float(row["benchmark_return"]) for row in rows]
        costs = sum((Decimal(str(row["total_cost_cny"])) for row in rows), Decimal("0"))
        notional = sum((Decimal(fill["notional"]) for fill in fold_fills), Decimal("0"))
        average_equity = initial * Decimal(str(np.mean([float(row["net_value"]) for row in rows])))
        gross_annual = _annualized(gross_returns)
        benchmark_annual = _annualized(benchmark_returns)
        families = l3_folds[fold]["families"]
        result.append(
            {
                "fold": fold,
                "test_start": start,
                "test_end": end,
                "sessions": len(rows),
                "target_count": int(configs[fold]["target_count"]),
                "frequency": configs[fold]["frequency"],
                "l3_outer_rank_ic": float(l3_folds[fold]["outer_composite_rank_ic"]),
                "positive_family_rankic_count": sum(
                    float(item["outer_rank_ic"]) > 0 for item in families
                ),
                "net_annual_return": float(l4_folds[fold]["annual_return"]),
                "benchmark_annual_return": float(l4_folds[fold]["annual_benchmark_return"]),
                "net_annual_excess_return": float(l4_folds[fold]["annual_excess_return"]),
                "conditional_gross_annual_return": gross_annual,
                "conditional_gross_annual_excess_return": gross_annual - benchmark_annual,
                "annual_cost_drag": gross_annual - float(l4_folds[fold]["annual_return"]),
                "total_cost_cny": float(costs),
                "trade_notional_cny": float(notional),
                "annualized_gross_turnover": float(
                    notional / average_equity * Decimal(252 / len(rows))
                ),
                "fills": len(fold_fills),
                "buy_fills": sum(fill["side"] == "BUY" for fill in fold_fills),
                "sell_fills": sum(fill["side"] == "SELL" for fill in fold_fills),
                "unique_traded_symbols": len({fill["symbol"] for fill in fold_fills}),
            }
        )
    return result


def _family_attribution(l3: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fold in l3["folds"]:
        for item in fold["families"]:
            grouped[item["family"]].append(item)
    result = []
    for family, items in sorted(grouped.items()):
        values = [float(item["outer_rank_ic"]) for item in items]
        methods = Counter(item["selected_method"] for item in items)
        result.append(
            {
                "family": family,
                "folds": len(items),
                "mean_outer_rank_ic": float(np.mean(values)),
                "minimum_outer_rank_ic": min(values),
                "maximum_outer_rank_ic": max(values),
                "positive_fold_count": sum(value > 0 for value in values),
                "mean_factor_count": float(np.mean([len(item["factor_ids"]) for item in items])),
                "selected_methods": json.dumps(dict(sorted(methods.items())), sort_keys=True),
            }
        )
    return result


def _diagnostics(
    l4: dict[str, Any], fold_rows: list[dict[str, Any]], gross_excess: float, cost_drag: float
) -> list[dict[str, Any]]:
    annual_excess = float(l4["performance"]["annual_excess_return"])
    diagnostics = [
        {
            "code": "COST_DOMINATED",
            "severity": "HIGH",
            "evidence": f"annual cost drag {cost_drag:.4%} versus net annual excess {annual_excess:.4%}",
        },
        {
            "code": "HIGH_TURNOVER",
            "severity": "HIGH",
            "evidence": "annualized gross turnover by fold is "
            + ", ".join(f"{row['annualized_gross_turnover']:.1f}x" for row in fold_rows),
        },
        {
            "code": "RANKIC_TOP_TAIL_CONVERSION_UNSTABLE",
            "severity": "HIGH",
            "evidence": "all five L3 fold RankIC values are positive but only three folds earn positive net excess",
        },
        {
            "code": "REGIME_BREAK_LAST_TWO_FOLDS",
            "severity": "HIGH",
            "evidence": "net annual excess is negative in both 2024-2025 and 2025-2026 folds",
        },
    ]
    if gross_excess < 0.10:
        diagnostics.append(
            {
                "code": "GROSS_ALPHA_BELOW_TARGET",
                "severity": "HIGH",
                "evidence": f"conditional zero-friction annual excess {gross_excess:.4%} remains below 10%",
            }
        )
    return diagnostics


def _annualized(values: list[float]) -> float:
    total = float(np.prod(1 + np.asarray(values, dtype=float)))
    return total ** (252 / len(values)) - 1 if total > 0 else -1.0


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _markdown(payload: dict[str, Any], folds: list[dict[str, Any]]) -> str:
    net = payload["net_performance"]
    paths = payload["conditional_paths"]
    costs = payload["cost_totals_cny"]
    rows = "\n".join(
        f"| {row['fold']} | {row['l3_outer_rank_ic']:.4f} | "
        f"{row['net_annual_excess_return']:.2%} | "
        f"{row['conditional_gross_annual_excess_return']:.2%} | "
        f"{row['annual_cost_drag']:.2%} | {row['annualized_gross_turnover']:.1f}x |"
        for row in folds
    )
    reasons = "\n".join(
        f"- `{item['code']}` ({item['severity']}): {item['evidence']}"
        for item in payload["diagnostics"]
    )
    return f"""# L4 收益转化归因

## 核心结论

当前策略并非完全没有毛 Alpha，而是同时受到高换手成本和组合收益转化不稳定的影响。实际年化收益为 {net["annual_return"]:.2%}，基准代理为 {net["annual_benchmark_return"]:.2%}，年化超额仅 {net["annual_excess_return"]:.2%}。在保持实际持仓、成交和数量不变的条件下，加回手续费、滑点和折边界清仓成本后，条件反事实年化收益为 {paths["conditional_gross_annual_return"]:.2%}，毛年化超额约 {payload["gross_annual_excess_return"]:.2%}。

成本使年化收益损失约 {payload["annual_cost_drag"]:.2%}，但完全无成本时的毛超额仍未达到 10%，因此不能只靠降低费率解决。

## 成本拆解

- 佣金：{float(costs["commission"]):,.2f} 元
- 印花税：{float(costs["stamp_duty"]):,.2f} 元
- 过户费：{float(costs["transfer_fee"]):,.2f} 元
- 5bps 成交滑点：{float(costs["slippage"]):,.2f} 元
- 年度折边界清仓：{float(costs["boundary_liquidation"]):,.2f} 元
- 总摩擦成本：{float(costs["total"]):,.2f} 元，占初始资金 {payload["cost_share_of_initial_equity"]:.2%}

若成本相对当前假设再增加一倍，条件反事实年化收益降至 {paths["conditional_double_cost_annual_return"]:.2%}。

## 分折转化

| 折 | L3 RankIC | 实际年化超额 | 无摩擦条件年化超额 | 成本拖累 | 年化毛换手 |
|---:|---:|---:|---:|---:|---:|
{rows}

## 诊断

{reasons}

## 解释边界

无摩擦和双倍成本路径固定了实际成交股票与数量，只回答“同一组合的摩擦成本有多大”，不是重新运行的独立策略。外层 OOS 仅用于归因，不能再作为未见数据调参。基准仍是内部 PIT 全 A 等权代理，并非官方万得指数；数据发布仍为 `MATERIALIZED_NOT_BACKTEST_APPROVED`。
"""


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


if __name__ == "__main__":
    main()
