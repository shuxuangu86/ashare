from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

_SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True, slots=True)
class EvaluationTiming:
    factor_date: date
    available_at: datetime
    execution_date: date
    return_start: date
    return_end: date

    def __post_init__(self) -> None:
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError("factor availability must be timezone-aware")
        if not (
            self.factor_date < self.execution_date
            and self.execution_date == self.return_start
            and self.return_start < self.return_end
        ):
            raise ValueError("evaluation timing must not overlap factor and return periods")
        if self.available_at.date() != self.factor_date:
            raise ValueError("daily factor must become available on factor_date")

    def payload(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def pit_forward_return_labels(
    close: npt.ArrayLike,
    trade_dates: tuple[date, ...],
    horizon: int,
) -> tuple[npt.NDArray[np.float64], tuple[EvaluationTiming, ...]]:
    prices = np.asarray(close, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[0] != len(trade_dates):
        raise ValueError("PIT labels require aligned dates and close matrix")
    if horizon <= 0:
        raise ValueError("label horizon must be positive")
    if tuple(sorted(trade_dates)) != trade_dates or len(set(trade_dates)) != len(trade_dates):
        raise ValueError("label dates must be unique and ordered")
    labels = np.full(prices.shape, np.nan)
    timings: list[EvaluationTiming] = []
    for factor_index in range(max(0, len(trade_dates) - horizon - 1)):
        start_index = factor_index + 1
        end_index = start_index + horizon
        with np.errstate(all="ignore"):
            labels[factor_index] = prices[end_index] / prices[start_index] - 1
        timings.append(
            EvaluationTiming(
                factor_date=trade_dates[factor_index],
                available_at=datetime.combine(
                    trade_dates[factor_index],
                    time(15, 0),
                    _SHANGHAI,
                ),
                execution_date=trade_dates[start_index],
                return_start=trade_dates[start_index],
                return_end=trade_dates[end_index],
            )
        )
    labels[~np.isfinite(labels)] = np.nan
    return labels, tuple(timings)
