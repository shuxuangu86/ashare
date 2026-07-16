import numpy as np
import pytest

from aquant.factors import (
    ExpressionEvaluator,
    FactorExpressionError,
    FactorPanel,
    parse_expression,
)


def _panel() -> FactorPanel:
    return FactorPanel(
        ("A", "B", "C"),
        {
            "Close": np.array(
                [[10.0, 20.0, 30.0], [11.0, 18.0, 33.0], [12.0, 21.0, 36.0], [14.0, 24.0, 39.0]]
            ),
            "Volume": np.array(
                [
                    [100.0, 200.0, 300.0],
                    [110.0, 180.0, 330.0],
                    [120.0, 210.0, 360.0],
                    [140.0, 240.0, 390.0],
                ]
            ),
        },
    )


def test_expression_has_deterministic_hash_fields_and_complexity() -> None:
    source = "RankCS(Div(Sub(Close, Mean(Close, 2)), Std(Close, 2)))"
    first = parse_expression(source)
    second = parse_expression(source)
    assert first.expression_hash == second.expression_hash
    assert len(first.expression_hash) == 64
    assert first.required_fields == ("Close",)
    assert first.complexity == 8


@pytest.mark.parametrize(
    "source",
    [
        "",
        "Unknown(Close)",
        "Close.__class__",
        "Mean(Close, window=20)",
        "Mean(Close, 0)",
        "Mean(Close, 2.5)",
        "NotAField",
        "__import__('os').system('whoami')",
        "Add(Close)",
    ],
)
def test_parser_rejects_non_whitelisted_or_invalid_syntax(source: str) -> None:
    with pytest.raises(FactorExpressionError):
        parse_expression(source)


def test_complexity_limit_is_enforced() -> None:
    with pytest.raises(FactorExpressionError, match="complexity"):
        parse_expression("Add(Add(Close, Open), High)", max_complexity=2)


def test_evaluates_rolling_zscore_formula_without_lookahead() -> None:
    expression = parse_expression("ZScore(Div(Sub(Close, Mean(Close, 2)), Std(Close, 2)))")
    result = ExpressionEvaluator().evaluate(expression, _panel())
    assert np.isnan(result[0]).all()
    assert np.allclose(result[-1], np.array([0.0, 0.0, 0.0]), equal_nan=True)


def test_ref_delta_and_cross_section_rank() -> None:
    evaluator = ExpressionEvaluator()
    panel = _panel()
    delta = evaluator.evaluate(parse_expression("Delta(Close, 1)"), panel)
    ranked = evaluator.evaluate(parse_expression("RankCS(Close)"), panel)
    assert np.isnan(delta[0]).all()
    assert np.allclose(delta[-1], [2.0, 3.0, 3.0])
    assert np.allclose(ranked[0], [1 / 3, 2 / 3, 1.0])


def test_time_series_pair_and_neutralization_operators() -> None:
    evaluator = ExpressionEvaluator()
    panel = _panel()
    correlation = evaluator.evaluate(parse_expression("Corr(Close, Volume, 3)"), panel)
    neutral = evaluator.evaluate(parse_expression("Neutralize(Close, Volume)"), panel)
    assert np.isnan(correlation[:2]).all()
    assert np.allclose(correlation[2:], 1.0)
    assert np.allclose(neutral, 0.0, atol=1e-10)


def test_division_by_zero_and_invalid_log_become_missing() -> None:
    panel = _panel()
    result = ExpressionEvaluator().evaluate(
        parse_expression("Log(Div(Close, Sub(Close, Close)))"), panel
    )
    assert np.isnan(result).all()


def test_factor_panel_validates_shape_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="shape"):
        FactorPanel(("A",), {"Close": np.array([1.0])})
    with pytest.raises(KeyError, match="missing"):
        ExpressionEvaluator().evaluate(parse_expression("Open"), _panel())
