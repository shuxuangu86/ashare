import numpy as np
from tests.unit.factors.test_atomic_library import panel

from aquant.factors.atomic import (
    academic_extension_library,
    baseline_factor_library,
    china_7000_controlled_library,
    china_rule_search_space,
    han_yang_zhou_library,
    technical_sentiment_library,
)
from aquant.factors.spec import SourceFaithfulness


def _by_id(factors: tuple[object, ...]) -> dict[str, object]:
    return {factor.spec.factor_id: factor for factor in factors}  # type: ignore[attr-defined]


def test_academic_extension_counts_metadata_and_search_space_are_stable() -> None:
    han = han_yang_zhou_library()
    china = china_7000_controlled_library()
    sentiment = technical_sentiment_library()
    all_factors = academic_extension_library()
    space = china_rule_search_space()

    assert (len(han), len(china), len(sentiment), len(all_factors)) == (13, 687, 7, 707)
    assert space.source_trial_count == 7_846
    assert space.controlled_trial_count == 687
    assert space.families == (
        "filter",
        "moving_average",
        "support_resistance",
        "channel_breakout",
        "obv_average",
    )
    assert space.search_space_hash == (
        "54be09bf05a86cb170501950eedb6d67533a280114d474ff25c298fff3bfa822"
    )
    assert len({factor.spec.factor_id for factor in all_factors}) == 707
    assert len({factor.spec.expression_hash for factor in all_factors}) == 707
    assert all(factor.spec.availability_lag == 1 for factor in all_factors)


def test_han_interactions_record_both_components() -> None:
    factors = _by_id(han_yang_zhou_library())
    interaction = factors["ma_signal_x_idiosyncratic_volatility"]
    assert interaction.spec.parent_factor_ids == (  # type: ignore[attr-defined]
        "hyz_ma_signal_5_20",
        "hyz_idiosyncratic_volatility_60",
    )
    assert interaction.spec.source_faithfulness is SourceFaithfulness.DERIVED_VARIANT  # type: ignore[attr-defined]


def test_fama_french_reuses_size_and_book_to_market_without_duplicates() -> None:
    factors = {
        factor.spec.factor_id: factor.spec
        for factor in baseline_factor_library()
        if factor.spec.source_id == "SRC_FAMA_FRENCH_2015"
    }
    assert set(factors) == {"log_total_market_cap", "book_to_market"}
    assert factors["log_total_market_cap"].source_formula_id == "SIZE"
    assert factors["book_to_market"].source_formula_id == "B/M"


def test_representative_academic_factors_are_deterministic_and_future_safe() -> None:
    original = panel(cutoff=270)
    changed = panel(future_multiplier=1_000, cutoff=270)
    selected_ids = {
        "ma_signal_x_volatility",
        "cn_rule_filter_x0050_hold1",
        "cn_rule_ma_s5_l20_b0000_hold1",
        "cn_rule_sr_w20_b0000_hold1",
        "cn_rule_channel_w20_b0000_hold1",
        "cn_rule_obvma_s5_l20_b0000_hold1",
        "net_technical_sentiment",
    }
    selected = [
        factor for factor in academic_extension_library() if factor.spec.factor_id in selected_ids
    ]
    assert {factor.spec.factor_id for factor in selected} == selected_ids
    for factor in selected:
        first = factor.compute_array(original)
        second = factor.compute_array(original)
        assert np.count_nonzero(np.isfinite(first)) > 0
        np.testing.assert_allclose(first, second, equal_nan=True)
        np.testing.assert_allclose(
            first[:270],
            factor.compute_array(changed)[:270],
            equal_nan=True,
            err_msg=factor.spec.factor_id,
        )


def test_technical_sentiment_identities_and_warmup_missingness() -> None:
    outputs = {
        factor.spec.factor_id: factor.compute_array(panel())
        for factor in technical_sentiment_library()
    }
    bullish = outputs["bullish_signal_count"]
    bearish = outputs["bearish_signal_count"]
    net = outputs["net_technical_sentiment"]
    assert np.isnan(bullish[0]).all()
    assert np.isnan(bearish[0]).all()
    assert np.nanmax(bullish + bearish) <= 10
    np.testing.assert_allclose(outputs["signal_agreement"], np.abs(net), equal_nan=True)
    assert np.nanmax(np.abs(outputs["trend_signal_breadth"])) <= 1
    assert np.nanmax(np.abs(outputs["oscillator_signal_breadth"])) <= 1
