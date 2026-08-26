from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from aquant.domain.data_release import DataReleaseId
from aquant.factors.aggregation import (
    AlphaAggregator,
    AlphaOutput,
    ModelKind,
    walk_forward_predict,
)


@pytest.mark.parametrize(
    "kind",
    (
        ModelKind.EQUAL_WEIGHT,
        ModelKind.IC_WEIGHT,
        ModelKind.ICIR_WEIGHT,
        ModelKind.RIDGE,
        ModelKind.ELASTIC_NET,
        ModelKind.LIGHTGBM,
    ),
)
def test_all_l3_baselines_fit_predict_deterministically(kind: ModelKind) -> None:
    generator = np.random.default_rng(11)
    features = generator.normal(size=(200, 6))
    target = features @ np.arange(1.0, 7.0) + generator.normal(scale=0.01, size=200)
    parameters = {"n_estimators": 20}
    first = AlphaAggregator(kind, **parameters).fit(
        features[:150],
        target[:150],
        historical_ic=np.arange(1.0, 7.0),
        historical_icir=np.arange(1.0, 7.0),
    )
    second = AlphaAggregator(kind, **parameters).fit(
        features[:150],
        target[:150],
        historical_ic=np.arange(1.0, 7.0),
        historical_icir=np.arange(1.0, 7.0),
    )
    predictions = first.predict(features[150:])
    np.testing.assert_allclose(predictions, second.predict(features[150:]))
    assert predictions.shape == (50,)


def test_standard_alpha_output_validates_bounds_and_keeps_explanations() -> None:
    output = AlphaOutput(
        date(2026, 7, 28),
        "000001.SZ",
        "ridge_20d",
        "1.0.0",
        0.01,
        0.03,
        0.8,
        0.7,
        0.2,
        0.3,
        0.9,
        DataReleaseId("cn_equity_20260728_001"),
        family_contributions=(("momentum", 0.02),),
        top_factor_contributions=(("momentum_20d", 0.01),),
        exposure_summary=(("size", -0.1),),
    )
    assert output.family_contributions[0][0] == "momentum"
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        replace(output, prediction_confidence=2)


def test_walk_forward_groups_purge_dates_not_security_rows() -> None:
    generator = np.random.default_rng(19)
    dates = np.repeat(np.arange(25), 5)
    features = generator.normal(size=(125, 3))
    target = features[:, 0] + generator.normal(scale=0.01, size=125)
    result = walk_forward_predict(
        ModelKind.RIDGE,
        features,
        target,
        train_size=10,
        validation_size=5,
        step=5,
        purge=2,
        groups=dates,
    )
    assert len(result.splits) == 2
    assert np.count_nonzero(np.isfinite(result.predictions)) == 50
    assert result.coverage == pytest.approx(0.4)
    assert result.rank_correlation > 0.9
