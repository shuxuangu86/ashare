import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ProductionGateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    minimum_history_years: float
    minimum_cross_sections: int
    maximum_missing_rate: float
    minimum_oos_rank_ic_abs: float
    minimum_rank_icir_abs: float
    minimum_directional_ic_win_rate: float
    minimum_positive_year_ratio: float
    minimum_monotonicity_abs: float
    maximum_turnover: float
    minimum_directional_net_return: float
    maximum_peer_correlation: float
    minimum_conditional_rank_ic_abs: float
    maximum_complexity: float
    maximum_style_exposure_abs: float
    minimum_extreme_regime_rank_ic_abs: float
    score_weights: dict[str, float]

    @field_validator("version")
    @classmethod
    def _version_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("gate config version must not be blank")
        return value.strip()

    @field_validator("score_weights")
    @classmethod
    def _weights_valid(cls, value: dict[str, float]) -> dict[str, float]:
        required = {
            "predictive",
            "stability",
            "tradability",
            "coverage",
            "uniqueness",
            "complexity_penalty",
        }
        if set(value) != required or any(weight < 0 for weight in value.values()):
            raise ValueError("gate score weights are incomplete or negative")
        return value

    @model_validator(mode="after")
    def _thresholds_valid(self) -> "ProductionGateConfig":
        proportions = (
            self.maximum_missing_rate,
            self.minimum_directional_ic_win_rate,
            self.minimum_positive_year_ratio,
            self.minimum_monotonicity_abs,
            self.maximum_turnover,
            self.maximum_peer_correlation,
            self.maximum_style_exposure_abs,
        )
        if any(not 0 <= value <= 1 for value in proportions):
            raise ValueError("gate proportion thresholds must be between zero and one")
        nonnegative = (
            self.minimum_history_years,
            self.minimum_oos_rank_ic_abs,
            self.minimum_rank_icir_abs,
            self.minimum_conditional_rank_ic_abs,
            self.minimum_extreme_regime_rank_ic_abs,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("gate minimum thresholds must be nonnegative")
        if self.minimum_cross_sections <= 0 or self.maximum_complexity <= 0:
            raise ValueError("gate cross-section and complexity limits must be positive")
        if not any(self.score_weights.values()):
            raise ValueError("gate score weights must contain a positive weight")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "ProductionGateConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProductionEvidence:
    history_years: float
    cross_sections: int
    missing_rate: float
    oos_rank_ic: float
    rank_icir: float
    directional_ic_win_rate: float
    positive_year_ratio: float
    monotonicity: float
    turnover: float
    directional_net_return: float
    maximum_peer_correlation: float
    conditional_rank_ic: float
    complexity: float
    maximum_style_exposure: float
    extreme_regime_rank_ic: float
    leakage_detected: bool = False


@dataclass(frozen=True, slots=True)
class GateDecision:
    passed: bool
    overall_score: float
    components: tuple[tuple[str, float], ...]
    rejections: tuple[str, ...]
    config_hash: str
    evidence_hash: str


def evaluate_production_gate(
    evidence: ProductionEvidence,
    config: ProductionGateConfig,
) -> GateDecision:
    numeric_evidence = tuple(
        value for name, value in asdict(evidence).items() if name != "leakage_detected"
    )
    checks = {
        "NON_FINITE_EVIDENCE": all(math.isfinite(value) for value in numeric_evidence),
        "FUTURE_LEAKAGE": not evidence.leakage_detected,
        "INSUFFICIENT_HISTORY": evidence.history_years >= config.minimum_history_years,
        "INSUFFICIENT_CROSS_SECTIONS": (evidence.cross_sections >= config.minimum_cross_sections),
        "EXCESSIVE_MISSINGNESS": evidence.missing_rate <= config.maximum_missing_rate,
        "OOS_RANK_IC_FAILURE": evidence.oos_rank_ic >= config.minimum_oos_rank_ic_abs,
        "RANK_ICIR_FAILURE": evidence.rank_icir >= config.minimum_rank_icir_abs,
        "IC_WIN_RATE_FAILURE": (
            evidence.directional_ic_win_rate >= config.minimum_directional_ic_win_rate
        ),
        "YEAR_STABILITY_FAILURE": (
            evidence.positive_year_ratio >= config.minimum_positive_year_ratio
        ),
        "MONOTONICITY_FAILURE": evidence.monotonicity >= config.minimum_monotonicity_abs,
        "TURNOVER_FAILURE": evidence.turnover <= config.maximum_turnover,
        "COST_ADJUSTED_RETURN_FAILURE": (
            evidence.directional_net_return >= config.minimum_directional_net_return
        ),
        "REDUNDANCY_FAILURE": (
            evidence.maximum_peer_correlation <= config.maximum_peer_correlation
            or evidence.conditional_rank_ic >= config.minimum_conditional_rank_ic_abs
        ),
        "COMPLEXITY_FAILURE": evidence.complexity <= config.maximum_complexity,
        "STYLE_EXPOSURE_FAILURE": (
            evidence.maximum_style_exposure <= config.maximum_style_exposure_abs
        ),
        "EXTREME_REGIME_FAILURE": (
            evidence.extreme_regime_rank_ic >= config.minimum_extreme_regime_rank_ic_abs
        ),
    }
    components = {
        "coverage": max(0.0, 1 - _finite(evidence.missing_rate, 1.0)),
        "predictive": max(0.0, _finite(evidence.oos_rank_ic, 0.0)),
        "stability": (
            _finite(evidence.directional_ic_win_rate, 0.0)
            + _finite(evidence.positive_year_ratio, 0.0)
            + max(0.0, _finite(evidence.extreme_regime_rank_ic, 0.0))
        )
        / 3,
        "tradability": max(0.0, 1 - _finite(evidence.turnover, 1.0)),
        "uniqueness": max(0.0, 1 - _finite(evidence.maximum_peer_correlation, 1.0)),
        "complexity_penalty": _finite(
            evidence.complexity / max(config.maximum_complexity, 1),
            1.0,
        ),
    }
    weights = config.score_weights
    score = (
        sum(
            components[name] * weight
            for name, weight in weights.items()
            if name != "complexity_penalty"
        )
        - components["complexity_penalty"] * weights["complexity_penalty"]
    )
    rejections = tuple(name for name, passed in checks.items() if not passed)
    payload = {
        "config_hash": config.config_hash,
        "evidence": asdict(evidence),
        "rejections": rejections,
        "score": score,
    }
    evidence_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return GateDecision(
        passed=not rejections,
        overall_score=float(score),
        components=tuple(sorted((name, float(value)) for name, value in components.items())),
        rejections=rejections,
        config_hash=config.config_hash,
        evidence_hash=evidence_hash,
    )


def _finite(value: float, fallback: float) -> float:
    return value if math.isfinite(value) else fallback
