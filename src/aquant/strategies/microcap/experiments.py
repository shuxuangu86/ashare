from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise

from aquant.backtest import (
    AshareExecutionRules,
    AshareFeeModel,
    AshareFeeSchedule,
    AshareOpenMatcher,
    BacktestMetrics,
    BacktestResult,
    EventDrivenBacktest,
    MarketSession,
    NextOpenMatcher,
    UnfilledOrderPolicy,
    calculate_metrics,
)
from aquant.strategies.microcap.models import MicrocapSnapshot
from aquant.strategies.microcap.prototypes import (
    PROTOTYPE_DEFINITIONS,
    MicrocapPrototype,
)
from aquant.strategies.microcap.strategy import MicrocapPrototypeStrategy


class RebalanceFrequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class CostScenario(StrEnum):
    FRICTIONLESS = "frictionless"
    RULES_ONLY = "rules-only"
    BASE_COST = "base-cost"
    DOUBLE_COST = "double-cost"


@dataclass(frozen=True, slots=True)
class MicrocapExperimentSpec:
    prototype: MicrocapPrototype
    frequency: RebalanceFrequency
    cost_scenario: CostScenario
    initial_equity: Decimal

    def __post_init__(self) -> None:
        equity = Decimal(self.initial_equity)
        if not equity.is_finite() or equity <= 0:
            raise ValueError("experiment initial equity must be finite and positive")
        object.__setattr__(self, "initial_equity", equity)

    @property
    def run_id(self) -> str:
        return (
            f"microcap-v1-{self.prototype.value}-{self.frequency.value}-{self.cost_scenario.value}"
        )


@dataclass(frozen=True, slots=True)
class MicrocapExperimentOutcome:
    spec: MicrocapExperimentSpec
    result: BacktestResult
    metrics: BacktestMetrics


def build_experiment_matrix(
    *,
    research_initial_equity: Decimal = Decimal("100000000"),
    executable_initial_equity: Decimal = Decimal("1000000"),
) -> tuple[MicrocapExperimentSpec, ...]:
    """Build six prototypes x three frequencies x four cost scenarios."""

    specs: list[MicrocapExperimentSpec] = []
    for prototype in MicrocapPrototype:
        definition = PROTOTYPE_DEFINITIONS[prototype]
        initial_equity = (
            executable_initial_equity if definition.executable_capital else research_initial_equity
        )
        for frequency in RebalanceFrequency:
            for scenario in CostScenario:
                specs.append(
                    MicrocapExperimentSpec(
                        prototype,
                        frequency,
                        scenario,
                        initial_equity,
                    )
                )
    return tuple(specs)


def generate_rebalance_dates(
    trading_dates: Sequence[date],
    frequency: RebalanceFrequency,
) -> frozenset[date]:
    """Return signal dates that have a following session available for T+1 fills."""

    dates = tuple(trading_dates)
    if len(dates) < 2:
        raise ValueError("rebalance schedule requires at least two trading dates")
    if any(previous >= current for previous, current in pairwise(dates)):
        raise ValueError("trading dates must be strictly increasing and unique")
    eligible = dates[:-1]
    if frequency is RebalanceFrequency.DAILY:
        return frozenset(eligible)

    last_by_period: dict[tuple[int, int], date] = {}
    for trade_date in dates:
        if frequency is RebalanceFrequency.WEEKLY:
            iso = trade_date.isocalendar()
            period = (iso.year, iso.week)
        else:
            period = (trade_date.year, trade_date.month)
        last_by_period[period] = trade_date
    return frozenset(
        trade_date for trade_date in last_by_period.values() if trade_date != dates[-1]
    )


def matcher_for_cost_scenario(
    scenario: CostScenario,
) -> NextOpenMatcher | AshareOpenMatcher:
    if scenario is CostScenario.FRICTIONLESS:
        return NextOpenMatcher()

    multiplier = Decimal("2") if scenario is CostScenario.DOUBLE_COST else Decimal("1")
    if scenario is CostScenario.RULES_ONLY:
        fee_model = AshareFeeModel(
            commission_rate=Decimal("0"),
            minimum_commission=Decimal("0"),
            sell_stamp_duty_rate=Decimal("0"),
            transfer_fee_rate=Decimal("0"),
        )
        slippage = Decimal("0")
        fee_schedule = None
    else:
        fee_model = None
        fee_schedule = AshareFeeSchedule(cost_multiplier=multiplier)
        slippage = Decimal("5") * multiplier
    return AshareOpenMatcher(
        rules=AshareExecutionRules(
            lot_size=100,
            max_volume_participation=Decimal("0.005"),
            slippage_bps=slippage,
            require_security_status=True,
            use_prior_20d_average_volume=True,
        ),
        fee_model=fee_model,
        fee_schedule=fee_schedule,
    )


class MicrocapExperimentRunner:
    """Run the frozen experiment catalog on one immutable session/snapshot release."""

    def __init__(
        self,
        *,
        sessions: tuple[MarketSession, ...],
        snapshots: Mapping[date, MicrocapSnapshot],
    ) -> None:
        if not sessions:
            raise ValueError("micro-cap experiment runner requires sessions")
        self._sessions = sessions
        self._snapshots = dict(snapshots)

    def run_one(self, spec: MicrocapExperimentSpec) -> MicrocapExperimentOutcome:
        rebalance_dates = generate_rebalance_dates(
            tuple(session.trade_date for session in self._sessions),
            spec.frequency,
        )
        strategy = MicrocapPrototypeStrategy(
            prototype=spec.prototype,
            snapshots=self._snapshots,
            rebalance_dates=rebalance_dates,
        )
        result = EventDrivenBacktest(
            run_id=spec.run_id,
            initial_cash=spec.initial_equity,
            matcher=matcher_for_cost_scenario(spec.cost_scenario),
            unfilled_order_policy=UnfilledOrderPolicy.CANCEL_AFTER_OPEN,
        ).run(self._sessions, strategy)
        return MicrocapExperimentOutcome(
            spec,
            result,
            calculate_metrics(result, initial_equity=spec.initial_equity),
        )

    def run_all(
        self,
        specs: Sequence[MicrocapExperimentSpec] | None = None,
    ) -> tuple[MicrocapExperimentOutcome, ...]:
        catalog = tuple(specs) if specs is not None else build_experiment_matrix()
        if not catalog:
            raise ValueError("experiment catalog cannot be empty")
        return tuple(self.run_one(spec) for spec in catalog)


def render_microcap_matrix_report(
    outcomes: Sequence[MicrocapExperimentOutcome],
) -> str:
    if not outcomes:
        raise ValueError("micro-cap matrix report requires outcomes")
    rows = [
        "# Micro-cap Prototype Matrix",
        "",
        "| Prototype | Rebalance | Cost scenario | Capital | Return | Max drawdown "
        "| Turnover | Fill rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for outcome in outcomes:
        spec = outcome.spec
        metrics = outcome.metrics
        rows.append(
            f"| {spec.prototype.value} | {spec.frequency.value} | "
            f"{spec.cost_scenario.value} | {spec.initial_equity} | "
            f"{metrics.total_return:.4%} | {metrics.max_drawdown:.4%} | "
            f"{metrics.gross_turnover:.4f} | {metrics.fill_rate:.2%} |"
        )
    rows.extend(
        (
            "",
            "The 100/300/400-stock portfolios use research capital. "
            "The 95/35/10-stock portfolios use the RMB 1,000,000 executable-capital profile.",
            "",
        )
    )
    return "\n".join(rows)
