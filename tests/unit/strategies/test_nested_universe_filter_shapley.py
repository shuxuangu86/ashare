import pytest
from scripts.analyze_nested_universe_filter_shapley import _shapley_rows


def test_exact_shapley_satisfies_efficiency() -> None:
    values = {subset: 0.01 * subset.bit_count() for subset in range(32)}

    rows = _shapley_rows("ALL", values)

    assert [row["shapley_annual_excess_effect"] for row in rows] == pytest.approx([0.01] * 5)
    assert sum(row["shapley_annual_excess_effect"] for row in rows) == pytest.approx(
        values[31] - values[0]
    )
