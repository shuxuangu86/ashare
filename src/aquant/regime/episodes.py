from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class MarketStateEpisode:
    episode_id: str
    state_id: str
    entry_date: date
    exit_date: date
    minimum_value: float
    maximum_value: float
    duration: int
    side: str
    recovery_confirmation_date: date | None


def detect_episodes(
    dates: tuple[date, ...],
    values: npt.NDArray[np.float64],
    *,
    state_id: str,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
) -> tuple[MarketStateEpisode, ...]:
    if values.shape != (len(dates),):
        raise ValueError("episode input shape mismatch")
    if (lower_threshold is None) == (upper_threshold is None):
        raise ValueError("exactly one episode threshold is required")
    if lower_threshold is not None:
        active = values < lower_threshold
    else:
        assert upper_threshold is not None
        active = values > upper_threshold
    active &= np.isfinite(values)
    episodes: list[MarketStateEpisode] = []
    entry: int | None = None
    for index, is_active in enumerate((*active.tolist(), False)):
        if is_active and entry is None:
            entry = index
        elif not is_active and entry is not None:
            sample = values[entry:index]
            episodes.append(
                MarketStateEpisode(
                    episode_id=f"{state_id}:{dates[entry].isoformat()}",
                    state_id=state_id,
                    entry_date=dates[entry],
                    exit_date=dates[index - 1],
                    minimum_value=float(np.min(sample)),
                    maximum_value=float(np.max(sample)),
                    duration=index - entry,
                    side="LOWER" if lower_threshold is not None else "UPPER",
                    recovery_confirmation_date=dates[index] if index < len(dates) else None,
                )
            )
            entry = None
    return tuple(episodes)
