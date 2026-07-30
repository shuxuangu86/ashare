import hashlib
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from aquant.regime import (
    MarketStateBuilder,
    MarketStateFamily,
    MarketStateRegistry,
    MarketStateStatus,
    RegimeInteractionSpec,
    aggregate_turnover,
    amount_share,
    build_registered_interactions,
    core_state_registry,
    expanding_percentile,
    materialize_state_values,
    microcap_crowding_score,
    relative_returns,
    rolling_mean,
    rolling_percentile,
)
from aquant.regime.crowding import CROWDING_INPUTS
from aquant.regime.index_relative import compute_hs300_csi2000_states
from aquant.regime.liquidity import compute_liquidity_states
from aquant.regime.style import compute_style_states
from aquant.regime.valuation import (
    compute_dividend_yield_states,
    compute_microcap_pb_states,
)


def test_core_catalog_contains_p0_states_without_exact_duplicates() -> None:
    registry = core_state_registry()
    assert len(registry) == 41
    for state_id in (
        "growth_minus_value_return_40d",
        "hs300_minus_csi2000_return_20d",
        "hs300_minus_csi2000_return_40d",
        "hs300_minus_csi2000_return_60d",
        "hs300_amount_share_all_a",
        "csi1000_turnover_daily",
        "microcap_crowding_score_v1",
        "dividend_index_dividend_yield_ffmv_weighted",
        "microcap_pb_percentile_5y",
    ):
        assert registry.get(state_id).availability_lag == 1


def test_registry_rejects_identity_and_exact_formula_duplicates() -> None:
    spec = core_state_registry().get("growth_minus_value_return_40d")
    registry = MarketStateRegistry((spec,))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)
    duplicate = spec.model_copy(
        update={"state_id": "duplicate", "display_name": "Duplicate", "content_hash": ""}
    )
    duplicate = type(spec).model_validate(duplicate.model_dump())
    with pytest.raises(ValueError, match="exact market-state formula"):
        registry.register(duplicate)


def test_compounded_relative_return_is_not_simple_return_difference() -> None:
    result = relative_returns((100.0, 120.0), (100.0, 110.0), 1)
    assert result == (None, pytest.approx(1.2 / 1.1 - 1.0))
    style = compute_style_states((100.0, 120.0), (100.0, 110.0))
    assert style["growth_minus_value_return_5d"] == (None, None)


def test_amount_share_and_aggregate_turnover_have_explicit_denominators() -> None:
    assert amount_share((20.0, 150.0, None), (100.0, 100.0, 100.0)) == (
        0.2,
        None,
        None,
    )
    assert aggregate_turnover((10.0, 20.0, None), (100.0, 200.0, -1.0)) == 0.1
    assert aggregate_turnover((10.0,), (0.0,)) is None


def test_rolling_and_expanding_percentiles_are_future_invariant() -> None:
    history = (1.0, 2.0, 2.0, 4.0)
    baseline = rolling_percentile(history, 3, minimum_periods=2)
    extended = rolling_percentile((*history, 1000.0), 3, minimum_periods=2)
    assert extended[: len(history)] == baseline
    expanding = expanding_percentile((*history, 1000.0), minimum_periods=2)
    assert expanding[: len(history)] == expanding_percentile(history, minimum_periods=2)
    assert baseline[0] is None
    assert baseline[2] == pytest.approx(2 / 3)


def test_microcap_pb_excludes_non_positive_book_values() -> None:
    rows = tuple((2.0 + index / 1000, -3.0, 0.0, None) for index in range(252))
    states = compute_microcap_pb_states(rows)
    assert states["microcap_pb_median"][0] == 2.0
    assert states["microcap_pb_percentile_3y"][250] is None
    assert states["microcap_pb_percentile_3y"][251] is not None
    assert states["microcap_pb_percentile_expanding"][251] is not None


def test_dividend_yield_aggregations_keep_distinct_economic_meanings() -> None:
    states = compute_dividend_yield_states(
        ((0.02, 0.04, -0.01),),
        ((100.0, 300.0, 500.0),),
    )
    assert states["dividend_index_dividend_yield_mean"] == (0.03,)
    assert states["dividend_index_dividend_yield_median"] == (0.03,)
    assert states["dividend_index_dividend_yield_ffmv_weighted"] == (0.035,)


def test_crowding_score_preserves_inputs_and_requires_coverage() -> None:
    inputs = {name: (0.2 + index * 0.1, None) for index, name in enumerate(CROWDING_INPUTS)}
    result = microcap_crowding_score(inputs)
    assert result[0] == pytest.approx(0.4)
    assert result[1] is None
    with pytest.raises(ValueError, match="missing"):
        microcap_crowding_score({name: (0.1,) for name in CROWDING_INPUTS[:-1]})


def test_materialization_is_t_plus_one_and_content_addressed() -> None:
    spec = core_state_registry().get("growth_minus_value_return_40d")
    trade_dates = (date(2026, 7, 29), date(2026, 7, 30))
    available_dates = (date(2026, 7, 30), date(2026, 7, 31))
    config_hash = hashlib.sha256(b"regime-config").hexdigest()
    first = materialize_state_values(
        spec,
        trade_dates,
        available_dates,
        (0.1, 0.2),
        data_release_id="cn_equity_20260731_001",
        code_version="test",
        config_hash=config_hash,
    )
    second = materialize_state_values(
        spec,
        trade_dates,
        available_dates,
        (0.1, 0.2),
        data_release_id="cn_equity_20260731_001",
        code_version="test",
        config_hash=config_hash,
    )
    assert [value.content_hash for value in first] == [value.content_hash for value in second]
    with pytest.raises(ValidationError, match="trade_date"):
        materialize_state_values(
            spec,
            trade_dates[:1],
            trade_dates[:1],
            (0.1,),
            data_release_id="cn_equity_20260731_001",
            code_version="test",
            config_hash=config_hash,
        )


def test_builder_rejects_unsorted_and_misaligned_inputs() -> None:
    builder = MarketStateBuilder(core_state_registry())
    dates = (date(2026, 7, 30), date(2026, 7, 29))
    with pytest.raises(ValueError, match="strictly increasing"):
        builder.validate_inputs(dates, {"close": (1.0, 2.0)})
    with pytest.raises(ValueError, match="length mismatch"):
        builder.validate_inputs(
            tuple(date(2026, 7, 29) + timedelta(days=index) for index in range(2)),
            {"close": (1.0,)},
        )


def test_only_registered_interactions_are_built_with_complexity_cap() -> None:
    registration = RegimeInteractionSpec("momentum", "trend", "momentum_x_trend")
    result = build_registered_interactions(
        {"momentum": (1.0, None, 2.0)},
        {"trend": (0.5, 0.7, -0.5)},
        (registration,),
    )
    assert result == {"momentum_x_trend": (0.5, None, -1.0)}
    with pytest.raises(ValueError, match="complexity"):
        build_registered_interactions(
            {"momentum": (1.0,)},
            {"trend": (1.0,)},
            (registration,),
            maximum_interactions=0,
        )


def test_catalog_selection_and_calculators_cover_aligned_series() -> None:
    registry = core_state_registry()
    assert len(registry.select(family=MarketStateFamily.LIQUIDITY)) == 6
    assert len(registry.select(status=MarketStateStatus.IMPLEMENTED)) == 41
    with pytest.raises(KeyError, match="unknown"):
        registry.get("not_registered")

    hs300 = tuple(100.0 + index for index in range(253))
    csi2000 = tuple(100.0 + index / 2 for index in range(253))
    relative = compute_hs300_csi2000_states(hs300, csi2000)
    assert relative["hs300_minus_csi2000_return_252d"][-1] is not None

    liquidity = compute_liquidity_states(
        (20.0,) * 253,
        (100.0,) * 253,
        tuple(0.01 + index / 100_000 for index in range(253)),
    )
    assert liquidity["hs300_amount_share_20d"][19] == pytest.approx(0.2)
    assert liquidity["csi1000_turnover_percentile_3y"][-1] is not None


def test_validation_errors_are_explicit() -> None:
    spec = core_state_registry().get("microcap_pb_median")
    with pytest.raises(ValidationError, match="required dataset"):
        type(spec).model_validate(
            {**spec.model_dump(), "required_datasets": (), "content_hash": ""}
        )
    with pytest.raises(ValidationError, match="cannot exceed"):
        type(spec).model_validate({**spec.model_dump(), "minimum_periods": 2, "content_hash": ""})
    with pytest.raises(ValidationError, match="related to itself"):
        type(spec).model_validate(
            {**spec.model_dump(), "related_states": (spec.state_id,), "content_hash": ""}
        )
    with pytest.raises(ValueError, match="window"):
        rolling_mean((1.0,), 0)
    with pytest.raises(ValueError, match="minimum_periods"):
        rolling_percentile((1.0,), 2, minimum_periods=3)
    with pytest.raises(ValueError, match="equal length"):
        amount_share((1.0,), (1.0, 2.0))


def test_builder_materialization_and_interaction_fail_closed() -> None:
    builder = MarketStateBuilder(core_state_registry())
    assert builder.select_specs(("microcap_pb_median",))[0].state_id == "microcap_pb_median"
    with pytest.raises(ValueError, match="unique"):
        builder.select_specs(("microcap_pb_median", "microcap_pb_median"))
    with pytest.raises(ValueError, match="output length"):
        builder.require_output_length((date(2026, 7, 30),), {"state": ()})

    spec = core_state_registry().get("microcap_pb_median")
    with pytest.raises(ValueError, match="length mismatch"):
        materialize_state_values(
            spec,
            (date(2026, 7, 30),),
            (),
            (1.0,),
            data_release_id="cn_equity_20260731_001",
            code_version="test",
            config_hash="0" * 64,
        )

    registration = RegimeInteractionSpec("momentum", "trend", "momentum_x_trend")
    with pytest.raises(KeyError, match="missing"):
        build_registered_interactions({}, {"trend": (1.0,)}, (registration,))
    with pytest.raises(ValueError, match="equal length"):
        build_registered_interactions(
            {"momentum": (1.0, 2.0)},
            {"trend": (1.0,)},
            (registration,),
        )
