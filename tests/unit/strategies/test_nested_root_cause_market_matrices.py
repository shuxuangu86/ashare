import numpy as np
import pytest
from scripts.build_nested_root_cause_market_matrices import _adjusted_price


def test_adjusted_price_multiplies_raw_price_and_factor() -> None:
    raw = np.asarray([[10.0, np.nan], [8.0, 5.0]])
    factor = np.asarray([[1.0, 2.0], [1.25, np.nan]])

    result = _adjusted_price(raw, factor)

    assert result[0, 0] == pytest.approx(10.0)
    assert result[1, 0] == pytest.approx(10.0)
    assert np.isnan(result[0, 1])
    assert np.isnan(result[1, 1])


def test_adjusted_price_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="must align"):
        _adjusted_price(np.ones((2, 2)), np.ones((2, 3)))
