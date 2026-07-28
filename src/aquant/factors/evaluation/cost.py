import numpy as np
import numpy.typing as npt


def net_returns(
    gross_returns: npt.ArrayLike,
    turnover: npt.ArrayLike,
    *,
    cost_bps: float,
) -> npt.NDArray[np.float64]:
    if cost_bps < 0:
        raise ValueError("cost must be non-negative")
    gross = np.asarray(gross_returns, dtype=np.float64)
    traded = np.asarray(turnover, dtype=np.float64)
    if gross.shape != traded.shape:
        raise ValueError("return and turnover arrays must have equal shape")
    return gross - traded * cost_bps / 10_000


def capacity_proxy(amount: npt.ArrayLike, weights: npt.ArrayLike) -> float:
    liquidity = np.asarray(amount, dtype=float)
    positions = np.asarray(weights, dtype=float)
    if liquidity.shape != positions.shape:
        raise ValueError("amount and weight arrays must have equal shape")
    return float(np.nansum(liquidity) / max(np.nansum(np.abs(positions)), 1e-12))
