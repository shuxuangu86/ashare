from dataclasses import dataclass
from datetime import date

from aquant.domain.data_release import DataReleaseId


@dataclass(frozen=True, slots=True)
class AlphaOutput:
    trade_date: date
    ts_code: str
    model_id: str
    model_version: str
    expected_alpha_5d: float
    expected_alpha_20d: float
    prediction_rank: float
    prediction_confidence: float
    downside_risk: float
    liquidity_risk: float
    signal_stability: float
    data_release_id: DataReleaseId
    family_contributions: tuple[tuple[str, float], ...] = ()
    top_factor_contributions: tuple[tuple[str, float], ...] = ()
    exposure_summary: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (self.ts_code, self.model_id, self.model_version)):
            raise ValueError("alpha output identifiers must not be blank")
        bounded = (
            self.prediction_rank,
            self.prediction_confidence,
            self.signal_stability,
        )
        if any(not 0 <= value <= 1 for value in bounded):
            raise ValueError("rank, confidence and stability must be in [0, 1]")
