from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import numpy.typing as npt

from aquant.data.industry import IndustryPITRepository
from aquant.factors.atomic.models import FactorPanelInput
from aquant.factors.operators.neutralization import (
    industry_exposures,
    industry_neutralize,
    multi_exposure_neutralize,
)
from aquant.factors.preprocessing.cross_section import winsorize_mad, zscore

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


@dataclass(frozen=True, slots=True)
class NeutralizationDiagnostics:
    trade_date: date
    status: str
    observations: int
    industries: int
    design_rank: int
    size_correlation_before: float | None
    size_correlation_after: float | None


@dataclass(frozen=True, slots=True)
class PITNeutralizationResult:
    raw_factor_value: Array
    winsorized_value: Array
    normalized_value: Array
    neutralized_value: Array
    neutralization_status: npt.NDArray[np.object_]
    diagnostics: tuple[NeutralizationDiagnostics, ...]


def load_pit_exposures(
    panel: FactorPanelInput,
    industry_repository: IndustryPITRepository,
    *,
    industry_level: str = "L1",
) -> PITExposurePanel:
    if "float_market_cap" not in panel.fields:
        raise KeyError("float_market_cap is required for neutralization")
    industry = industry_repository.panel(
        panel.trade_dates,
        panel.ts_codes,
        level=industry_level,
    )
    return PITExposurePanel(
        trade_dates=panel.trade_dates,
        industries=industry.industries,
        float_market_cap=np.asarray(panel.fields["float_market_cap"], dtype=float),
        available_dates=industry.available_dates,
    )


def load_size_exposures(panel: FactorPanelInput) -> PITExposurePanel:
    if "float_market_cap" not in panel.fields:
        raise KeyError("float_market_cap is required for size neutralization")
    shape = (len(panel.trade_dates), len(panel.ts_codes))
    available_dates = np.empty(shape, dtype=object)
    for row, trade_date in enumerate(panel.trade_dates):
        available_dates[row] = trade_date
    return PITExposurePanel(
        trade_dates=panel.trade_dates,
        industries=np.full(shape, None, dtype=object),
        float_market_cap=np.asarray(panel.fields["float_market_cap"], dtype=float),
        available_dates=available_dates,
    )


def neutralization_variants(
    values: npt.ArrayLike,
    exposures: PITExposurePanel,
    *,
    minimum_observations: int = 20,
    minimum_industry_observations: int = 3,
) -> dict[str, Array]:
    raw = np.asarray(values, dtype=float)
    if raw.shape != exposures.float_market_cap.shape:
        raise ValueError("factor and PIT exposure panels must align")
    industry = np.full(raw.shape, np.nan)
    size_neutral = np.full(raw.shape, np.nan)
    industry_size = np.full(raw.shape, np.nan)
    for row in range(len(raw)):
        groups = _eligible_groups(
            exposures.industries[row],
            minimum_industry_observations=minimum_industry_observations,
        )
        distinct_groups = {str(value) for value in groups if value is not None}
        market_cap = exposures.float_market_cap[row]
        valid_group = np.asarray(
            [value is not None and bool(str(value).strip()) for value in groups]
        )
        valid_cap = np.isfinite(market_cap) & (market_cap > 0) & valid_group
        size = np.full(market_cap.shape, np.nan)
        valid_size = np.isfinite(market_cap) & (market_cap > 0)
        size[valid_size] = np.log(market_cap[valid_size])
        weights = np.where(valid_cap, np.sqrt(market_cap), np.nan)
        size_weights = np.where(valid_size, np.sqrt(market_cap), np.nan)
        size_neutral[row] = multi_exposure_neutralize(
            raw[row],
            size[:, None],
            weights=size_weights,
            minimum_observations=minimum_observations,
        )
        if len(distinct_groups) < 2:
            continue
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
        "size_neutral": size_neutral,
        "industry_size_neutral": industry_size,
    }


def neutralize_pit_factor(
    values: npt.ArrayLike,
    exposures: PITExposurePanel,
    *,
    method: Literal["industry", "size", "industry_size", "industry_proxy"] = "industry_size",
    style_exposures: Mapping[str, npt.ArrayLike] | None = None,
    minimum_observations: int = 20,
    minimum_industry_observations: int = 3,
    winsorize_scale: float = 3.0,
) -> PITNeutralizationResult:
    raw = np.asarray(values, dtype=float)
    if raw.shape != exposures.float_market_cap.shape:
        raise ValueError("factor and PIT exposure panels must align")
    winsorized = np.vstack([winsorize_mad(row, scale=winsorize_scale) for row in raw])
    normalized = np.vstack([zscore(row) for row in winsorized])
    proxy_designs: tuple[Array, ...] | None = None
    if method == "industry_proxy":
        neutralized, proxy_designs = _industry_style_proxy_neutralize(
            normalized,
            exposures,
            style_exposures,
            minimum_observations=minimum_observations,
            minimum_industry_observations=minimum_industry_observations,
        )
    else:
        variants = neutralization_variants(
            normalized,
            exposures,
            minimum_observations=minimum_observations,
            minimum_industry_observations=minimum_industry_observations,
        )
        neutralized = variants[f"{method}_neutral"]
    statuses = np.full(raw.shape, "MISSING_FACTOR_VALUE", dtype=object)
    diagnostics: list[NeutralizationDiagnostics] = []
    for row, trade_date in enumerate(exposures.trade_dates):
        groups = _eligible_groups(
            exposures.industries[row],
            minimum_industry_observations=minimum_industry_observations,
        )
        caps = exposures.float_market_cap[row]
        finite_factor = np.isfinite(raw[row])
        statuses[row, finite_factor] = "UNKNOWN_INDUSTRY"
        raw_known_industry = np.asarray(
            [value is not None and bool(str(value).strip()) for value in exposures.industries[row]]
        )
        statuses[row, finite_factor & raw_known_industry] = "SMALL_INDUSTRY"
        known_industry = np.asarray([value is not None for value in groups])
        if method == "industry_proxy":
            eligible = finite_factor
            statuses[row, finite_factor] = "INSUFFICIENT_CROSS_SECTION"
        elif method == "size":
            eligible = finite_factor & np.isfinite(caps) & (caps > 0)
            statuses[row, finite_factor] = "MISSING_MARKET_CAP"
        else:
            statuses[row, finite_factor & known_industry] = "MISSING_MARKET_CAP"
            eligible = finite_factor & known_industry & np.isfinite(caps) & (caps > 0)
        statuses[row, eligible] = "INSUFFICIENT_CROSS_SECTION"
        valid = np.isfinite(neutralized[row])
        statuses[row, valid] = "VALID_PROXY" if method == "industry_proxy" else "VALID"
        design = proxy_designs[row] if proxy_designs is not None else industry_exposures(groups)
        if method == "industry_size":
            size = np.full(caps.shape, np.nan)
            size[np.isfinite(caps) & (caps > 0)] = np.log(caps[np.isfinite(caps) & (caps > 0)])
            design = np.column_stack((design, size))
        diagnostics.append(
            NeutralizationDiagnostics(
                trade_date=trade_date,
                status=(
                    "PROXY_PASS"
                    if method == "industry_proxy"
                    and np.count_nonzero(valid) >= minimum_observations
                    else "PASS"
                    if np.count_nonzero(valid) >= minimum_observations
                    else "BLOCKED"
                ),
                observations=int(np.count_nonzero(valid)),
                industries=len({str(value) for value in groups if value is not None}),
                design_rank=int(np.linalg.matrix_rank(design[eligible]))
                if np.count_nonzero(eligible) and design.shape[1]
                else 0,
                size_correlation_before=_weighted_size_correlation(normalized[row], caps),
                size_correlation_after=_weighted_size_correlation(neutralized[row], caps),
            )
        )
    return PITNeutralizationResult(
        raw_factor_value=raw.copy(),
        winsorized_value=winsorized,
        normalized_value=normalized,
        neutralized_value=neutralized,
        neutralization_status=statuses,
        diagnostics=tuple(diagnostics),
    )


def _industry_style_proxy_neutralize(
    values: Array,
    exposures: PITExposurePanel,
    style_exposures: Mapping[str, npt.ArrayLike] | None,
    *,
    minimum_observations: int,
    minimum_industry_observations: int,
) -> tuple[Array, tuple[Array, ...]]:
    if not style_exposures:
        raise ValueError("industry proxy neutralization requires PIT style exposures")
    resolved_styles = {
        name: np.asarray(style_exposures[name], dtype=float) for name in sorted(style_exposures)
    }
    if any(style.shape != values.shape for style in resolved_styles.values()):
        raise ValueError("industry proxy style exposures must align with factor values")
    output = np.full(values.shape, np.nan)
    designs: list[Array] = []
    for row in range(len(values)):
        groups = _eligible_groups(
            exposures.industries[row],
            minimum_industry_observations=minimum_industry_observations,
        )
        unknown = np.asarray(
            [float(value is None or not str(value).strip()) for value in groups],
            dtype=float,
        )
        columns = [industry_exposures(groups), unknown[:, None]]
        for name, style in resolved_styles.items():
            style_row = zscore(style[row] if name == "size" else winsorize_mad(style[row]))
            missing = ~np.isfinite(style_row)
            style_row[missing] = 0
            columns.append(style_row[:, None])
            if np.any(missing) and np.any(~missing):
                columns.append(missing.astype(float)[:, None])
        design = np.column_stack(columns)
        nonconstant = np.nanstd(design, axis=0) > 1e-12
        design = design[:, nonconstant]
        designs.append(design)
        market_cap = exposures.float_market_cap[row]
        valid_cap = np.isfinite(market_cap) & (market_cap > 0)
        fallback_weight = (
            float(np.nanmedian(np.sqrt(market_cap[valid_cap]))) if np.any(valid_cap) else 1.0
        )
        weights = np.where(valid_cap, np.sqrt(market_cap), fallback_weight)
        output[row] = multi_exposure_neutralize(
            values[row],
            design,
            weights=weights,
            minimum_observations=minimum_observations,
        )
    return output, tuple(designs)


def _eligible_groups(
    groups: npt.ArrayLike,
    *,
    minimum_industry_observations: int,
) -> npt.NDArray[np.object_]:
    if minimum_industry_observations < 1:
        raise ValueError("minimum industry observations must be positive")
    resolved = np.asarray(groups, dtype=object)
    counts: dict[str, int] = {}
    for value in resolved:
        if value is not None and str(value).strip():
            counts[str(value)] = counts.get(str(value), 0) + 1
    return np.asarray(
        [
            value
            if value is not None and counts.get(str(value), 0) >= minimum_industry_observations
            else None
            for value in resolved
        ],
        dtype=object,
    )


def _weighted_size_correlation(left: Array, market_cap: Array) -> float | None:
    size = np.full(market_cap.shape, np.nan)
    valid_cap = np.isfinite(market_cap) & (market_cap > 0)
    size[valid_cap] = np.log(market_cap[valid_cap])
    valid = np.isfinite(left) & np.isfinite(size)
    if np.count_nonzero(valid) < 3:
        return None
    weights = np.sqrt(market_cap[valid])
    weights /= np.sum(weights)
    left_centered = left[valid] - np.sum(weights * left[valid])
    size_centered = size[valid] - np.sum(weights * size[valid])
    denominator = np.sqrt(
        np.sum(weights * np.square(left_centered)) * np.sum(weights * np.square(size_centered))
    )
    if np.isclose(denominator, 0.0):
        return 0.0
    correlation = float(np.sum(weights * left_centered * size_centered) / denominator)
    return correlation if np.isfinite(correlation) else None
