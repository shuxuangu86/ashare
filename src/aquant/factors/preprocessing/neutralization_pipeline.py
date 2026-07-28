from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

from aquant.factors.operators.neutralization import (
    industry_exposures,
    industry_neutralize,
    multi_exposure_neutralize,
)

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PITExposurePanel:
    trade_dates: tuple[date, ...]
    industries: npt.NDArray[np.object_]
    float_market_cap: Array
    available_dates: npt.NDArray[np.object_]

    def __post_init__(self) -> None:
        shape = self.float_market_cap.shape
        if (
            self.float_market_cap.ndim != 2
            or self.industries.shape != shape
            or self.available_dates.shape != shape
            or len(self.trade_dates) != shape[0]
        ):
            raise ValueError("PIT exposure panel fields must align")
        for row, trade_date in enumerate(self.trade_dates):
            for available in self.available_dates[row]:
                if available is not None and (
                    not isinstance(available, date) or available > trade_date
                ):
                    raise ValueError("risk exposure is not available at neutralization time")


def neutralization_variants(
    values: npt.ArrayLike,
    exposures: PITExposurePanel,
    *,
    minimum_observations: int = 20,
) -> dict[str, Array]:
    raw = np.asarray(values, dtype=float)
    if raw.shape != exposures.float_market_cap.shape:
        raise ValueError("factor and PIT exposure panels must align")
    industry = np.full(raw.shape, np.nan)
    industry_size = np.full(raw.shape, np.nan)
    for row in range(len(raw)):
        groups = exposures.industries[row]
        market_cap = exposures.float_market_cap[row]
        valid_group = np.asarray(
            [value is not None and bool(str(value).strip()) for value in groups]
        )
        valid_cap = np.isfinite(market_cap) & (market_cap > 0) & valid_group
        size = np.full(market_cap.shape, np.nan)
        size[valid_cap] = np.log(market_cap[valid_cap])
        weights = np.where(valid_cap, np.sqrt(market_cap), np.nan)
        industry[row] = industry_neutralize(
            raw[row],
            groups,
            weights=weights,
            minimum_observations=minimum_observations,
        )
        design = np.column_stack((industry_exposures(groups), size))
        industry_size[row] = multi_exposure_neutralize(
            raw[row],
            design,
            weights=weights,
            minimum_observations=minimum_observations,
        )
    return {
        "raw": raw.copy(),
        "industry_neutral": industry,
        "industry_size_neutral": industry_size,
    }
