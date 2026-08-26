import pytest
from scripts.analyze_nested_execution_bridge import _bridge_row, _stability_summary


def test_bridge_row_is_an_additive_sequential_attribution() -> None:
    paths = {
        "all_signal_gross": [0.01, 0.01],
        "eligible_signal_gross": [0.009, 0.009],
        "eligible_signal_10bps": [0.008, 0.008],
        "benchmark": [0.0, 0.0],
    }

    row = _bridge_row(
        fold=0,
        start="2020-01-01",
        end="2020-01-02",
        target_count=20,
        frequency="weekly",
        paths=paths,
        actual_gross_excess=0.5,
        actual_net_excess=0.4,
    )

    assert (
        row["universe_filter_effect"]
        + row["execution_constraint_effect"]
        + row["actual_cost_effect"]
    ) == pytest.approx(row["total_conversion_effect"])


def test_stability_summary_reports_selected_outer_rank_and_optimism() -> None:
    rows = [
        {
            "fold": 0,
            "selected": True,
            "inner_annual_excess": 0.20,
            "inner_selection_score": 0.20,
            "outer_eligible_10bps_excess": 0.05,
            "outer_rank": 2,
        },
        {
            "fold": 0,
            "selected": False,
            "inner_annual_excess": 0.10,
            "inner_selection_score": 0.10,
            "outer_eligible_10bps_excess": 0.10,
            "outer_rank": 1,
        },
    ]

    summary = _stability_summary(rows)

    assert summary["selected_outer_ranks"] == [2]
    assert summary["chosen_mean_optimism"] == pytest.approx(0.15)
