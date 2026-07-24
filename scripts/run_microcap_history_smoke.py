import argparse
import json
from datetime import date, datetime
from pathlib import Path

from aquant.data.history import DuckDBMicrocapHistory
from aquant.strategies.microcap.experiments import (
    CostScenario,
    MicrocapExperimentRunner,
    MicrocapExperimentSpec,
    RebalanceFrequency,
)
from aquant.strategies.microcap.prototypes import MicrocapPrototype


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short unadjusted smoke test of the historical micro-cap chain"
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--prototype",
        choices=[value.value for value in MicrocapPrototype],
        default=MicrocapPrototype.SMALLEST_400.value,
    )
    parser.add_argument(
        "--frequency",
        choices=[value.value for value in RebalanceFrequency],
        default=RebalanceFrequency.WEEKLY.value,
    )
    parser.add_argument(
        "--cost-scenario",
        choices=[value.value for value in CostScenario],
        default=CostScenario.RULES_ONLY.value,
    )
    return parser.parse_args()


def _date(value: str, *, field: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise SystemExit(f"{field} must use YYYYMMDD") from exc


def main() -> int:
    args = _parse_args()
    start = _date(args.start_date, field="start-date")
    end = _date(args.end_date, field="end-date")
    with DuckDBMicrocapHistory(args.release_dir) as history:
        sessions = history.sessions(start, end)
        session_dates = tuple(session.trade_date for session in sessions)
        snapshots = history.snapshots(session_dates)
    prototype = MicrocapPrototype(args.prototype)
    initial_equity = (
        1_000_000
        if prototype
        in {
            MicrocapPrototype.EXECUTABLE_95,
            MicrocapPrototype.DIVIDEND_QUALITY_10,
            MicrocapPrototype.LOW_PB_LOW_TURNOVER_35,
        }
        else 100_000_000
    )
    spec = MicrocapExperimentSpec(
        prototype=prototype,
        frequency=RebalanceFrequency(args.frequency),
        cost_scenario=CostScenario(args.cost_scenario),
        initial_equity=initial_equity,
    )
    outcome = MicrocapExperimentRunner(
        sessions=sessions,
        snapshots=snapshots,
    ).run_one(spec)
    print(
        json.dumps(
            {
                "status": "SMOKE_ONLY_UNADJUSTED_RETURNS_NOT_RESEARCH_RESULT",
                "run_id": spec.run_id,
                "sessions": len(sessions),
                "observations_first_session": len(snapshots[session_dates[0]].observations),
                "orders": len(outcome.result.orders),
                "fills": len(outcome.result.fills),
                "total_return": str(outcome.metrics.total_return),
                "max_drawdown": str(outcome.metrics.max_drawdown),
                "warning": (
                    "This command validates data/PIT/execution wiring only. "
                    "Do not use its return metrics until corporate actions are ledger-applied."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
