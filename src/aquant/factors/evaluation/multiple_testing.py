import numpy as np


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
