from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class FamilyTestingSummary:
    family: str
    trial_count: int
    rejected_count: int
    minimum_raw_p_value: float
    minimum_adjusted_p_value: float
    fdr_alpha: float


@dataclass(frozen=True, slots=True)
class SPATestResult:
    statistic: float
    p_value: float
    lower_p_value: float
    upper_p_value: float
    trial_count: int
    observation_count: int
    bootstrap_samples: int
    stationary_restart_probability: float
    seed: int


def benjamini_hochberg(
    p_values: tuple[float, ...],
    *,
    alpha: float = 0.05,
) -> tuple[tuple[float, bool], ...]:
    if not 0 < alpha < 1:
        raise ValueError("FDR alpha must be in (0, 1)")
    values = np.asarray(p_values, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite probabilities")
    if not len(values):
        return ()
    order = np.argsort(values)
    ordered = values[order]
    adjusted = np.minimum.accumulate((ordered * len(values) / np.arange(1, len(values) + 1))[::-1])[
        ::-1
    ]
    adjusted = np.minimum(adjusted, 1)
    restored = np.empty(len(values))
    restored[order] = adjusted
    return tuple((float(value), bool(value <= alpha)) for value in restored)


def family_multiple_testing_summary(
    family_p_values: dict[str, tuple[float, ...]],
    *,
    alpha: float = 0.05,
) -> tuple[FamilyTestingSummary, ...]:
    summaries = []
    for family in sorted(family_p_values):
        p_values = family_p_values[family]
        adjusted = benjamini_hochberg(p_values, alpha=alpha)
        summaries.append(
            FamilyTestingSummary(
                family=family,
                trial_count=len(p_values),
                rejected_count=sum(rejected for _, rejected in adjusted),
                minimum_raw_p_value=min(p_values, default=1.0),
                minimum_adjusted_p_value=min(
                    (value for value, _ in adjusted),
                    default=1.0,
                ),
                fdr_alpha=alpha,
            )
        )
    return tuple(summaries)


def superior_predictive_ability_test(
    performance_differentials: npt.ArrayLike,
    *,
    bootstrap_samples: int = 500,
    stationary_restart_probability: float = 0.1,
    seed: int = 0,
) -> SPATestResult:
    values = np.asarray(performance_differentials, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 20
        or values.shape[1] < 1
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("SPA input must be a finite observations-by-rules matrix")
    if bootstrap_samples < 100:
        raise ValueError("SPA requires at least 100 bootstrap samples")
    if not 0 < stationary_restart_probability <= 1:
        raise ValueError("stationary restart probability must be in (0, 1]")

    observations, trials = values.shape
    means = np.mean(values, axis=0)
    scale = np.sqrt(_stationary_long_run_variance(values, stationary_restart_probability))
    valid_scale = np.where(scale > 0, scale, np.nan)
    statistic = _max_studentized(means, valid_scale, observations)
    threshold = -valid_scale * np.sqrt(2 * np.log(np.log(observations))) / np.sqrt(observations)
    consistent_center = np.where(means >= threshold, means, 0)
    lower_center = np.maximum(means, 0)
    upper_center = means

    generator = np.random.default_rng(seed)
    consistent_statistics = np.empty(bootstrap_samples)
    lower_statistics = np.empty(bootstrap_samples)
    upper_statistics = np.empty(bootstrap_samples)
    for sample in range(bootstrap_samples):
        indices = _stationary_bootstrap_indices(
            observations,
            stationary_restart_probability,
            generator,
        )
        resampled_mean = np.mean(values[indices], axis=0)
        consistent_statistics[sample] = _max_studentized(
            resampled_mean - consistent_center,
            valid_scale,
            observations,
        )
        lower_statistics[sample] = _max_studentized(
            resampled_mean - lower_center,
            valid_scale,
            observations,
        )
        upper_statistics[sample] = _max_studentized(
            resampled_mean - upper_center,
            valid_scale,
            observations,
        )

    def p_value(samples: npt.NDArray[np.float64]) -> float:
        return float((1 + np.count_nonzero(samples >= statistic)) / (bootstrap_samples + 1))

    return SPATestResult(
        statistic=statistic,
        p_value=p_value(consistent_statistics),
        lower_p_value=p_value(lower_statistics),
        upper_p_value=p_value(upper_statistics),
        trial_count=trials,
        observation_count=observations,
        bootstrap_samples=bootstrap_samples,
        stationary_restart_probability=stationary_restart_probability,
        seed=seed,
    )


def _stationary_long_run_variance(
    values: npt.NDArray[np.float64],
    restart: float,
) -> npt.NDArray[np.float64]:
    observations = len(values)
    centered = values - np.mean(values, axis=0)
    variance = np.mean(np.square(centered), axis=0)
    weight = 1 - restart
    for lag in range(1, observations):
        covariance = np.mean(centered[:-lag] * centered[lag:], axis=0)
        kernel = ((observations - lag) / observations) * (weight**lag)
        variance += 2 * kernel * covariance
        if kernel < 1e-8:
            break
    result: npt.NDArray[np.float64] = np.maximum(variance, 0)
    return result


def _stationary_bootstrap_indices(
    observations: int,
    restart: float,
    generator: np.random.Generator,
) -> npt.NDArray[np.int64]:
    indices = np.empty(observations, dtype=np.int64)
    indices[0] = generator.integers(observations)
    for row in range(1, observations):
        if generator.random() < restart:
            indices[row] = generator.integers(observations)
        else:
            indices[row] = (indices[row - 1] + 1) % observations
    return indices


def _max_studentized(
    means: npt.NDArray[np.float64],
    scale: npt.NDArray[np.float64],
    observations: int,
) -> float:
    with np.errstate(all="ignore"):
        statistics = np.sqrt(observations) * means / scale
    finite = statistics[np.isfinite(statistics)]
    return max(0.0, float(np.max(finite))) if len(finite) else 0.0
