import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aquant.factors.definitions.dsl import FactorExpressionError
from aquant.factors.formula import (
    ExpressionLimits,
    FormulaCompiler,
    analyze_expression,
    parse_formula,
)
from aquant.factors.formula.canonicalize import canonicalize_expression, expression_hash
from aquant.factors.operators import FactorPanel


def panel() -> FactorPanel:
    close = np.arange(1.0, 31.0)[:, None] * np.array([[1.0, 2.0, 3.0]])
    volume = 100 + close * 10
    return FactorPanel(
        ("A", "B", "C"),
        {
            "Close": close,
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Volume": volume,
            "Amount": volume * close,
            "VWAP": close * 0.995,
        },
    )


def test_infix_and_function_forms_have_same_stable_canonical_hash() -> None:
    infix = "Rank((Close - Mean(Close, 10)) / Std(Close, 10))"
    functional = "Rank(Div(Sub(Close, Mean(Close,10)), Std(Close,10)))"
    assert canonicalize_expression(infix) == canonicalize_expression(functional)
    assert expression_hash(infix) == expression_hash(functional)


def test_dependencies_history_complexity_and_depth_are_inferred() -> None:
    expression = parse_formula("Rank(Mean(Return(Close, 5), 20))")
    metrics = analyze_expression(expression)
    assert metrics.input_fields == ("Close",)
    assert metrics.history_requirement == 24
    assert metrics.node_count == 4
    assert metrics.depth == 4
    assert metrics.window_count == 2
    assert metrics.complexity_score > metrics.node_count


def test_condition_aliases_decay_scale_arg_and_signed_power_compile() -> None:
    source = "Scale(Where(Return(Close, 1) > 0, SignedPower(Close, 0.5), 0))"
    expression = parse_formula(source)
    result = FormulaCompiler(expression).evaluate(panel())
    assert np.isnan(result[0]).all()
    np.testing.assert_allclose(np.sum(np.abs(result[1:]), axis=1), 1)

    decay = FormulaCompiler(parse_formula("DecayLinear(Close, 3)")).evaluate(panel())
    maximum = FormulaCompiler(parse_formula("ArgMax(Close, 3)")).evaluate(panel())
    assert np.isnan(decay[:2]).all()
    np.testing.assert_allclose(maximum[2:], 3)


@pytest.mark.parametrize(
    "source",
    [
        "Lead(Close, 1)",
        "Close[0]",
        "[value for value in Close]",
        "(lambda value: value)(Close)",
        "open('/etc/passwd')",
        "__import__('os')",
        "Close > Open > Low",
    ],
)
def test_formula_parser_rejects_future_or_unsafe_syntax(source: str) -> None:
    with pytest.raises(FactorExpressionError):
        parse_formula(source)


def test_l2a_and_l2b_limits_are_enforced() -> None:
    expression = "Rank(Mean(Return(Close, 5), 20))"
    with pytest.raises(FactorExpressionError, match="depth"):
        parse_formula(expression, limits=ExpressionLimits(max_depth=3))
    assert parse_formula(expression, limits=ExpressionLimits(max_depth=6)).depth == 4
    with pytest.raises(FactorExpressionError, match="input-field"):
        parse_formula(
            "Close + Open + High + Low + Volume + Amount",
            limits=ExpressionLimits(max_fields=5),
        )


def test_alpha101_and_alpha191_style_samples_compile_without_eval() -> None:
    samples = (
        "Rank(Corr(Rank(Volume), Rank(VWAP), 6))",
        "Rank((Close - Mean(Close, 10)) / Std(Close, 10))",
        "Rank(Delta(Log(Volume), 2) * ((Close - Open) / Open))",
    )
    for source in samples:
        compiled = FormulaCompiler(parse_formula(source))
        assert compiled.dependencies
        assert compiled.evaluate(panel()).shape == panel().shape


@given(st.integers(min_value=1, max_value=250))
def test_expression_canonicalization_is_whitespace_and_parenthesis_stable(window: int) -> None:
    compact = f"Rank(Delta(Close,{window}))"
    spaced = f" Rank( (Delta(Close, {window})) ) "
    assert canonicalize_expression(compact) == canonicalize_expression(spaced)
