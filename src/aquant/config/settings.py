import os
from collections.abc import Mapping, MutableMapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from aquant.domain.enums import LiveTier, RunMode

ENV_PREFIX = "AQUANT_"


class StrictSettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatabaseSettings(StrictSettingsModel):
    host: str = "127.0.0.1"
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = "aquant"
    user: str = "aquant"
    password: SecretStr = SecretStr("aquant-local-only")
    pool_size: int = Field(default=10, ge=1, le=100)

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password.get_secret_value()}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class RedisSettings(StrictSettingsModel):
    host: str = "127.0.0.1"
    port: int = Field(default=6379, ge=1, le=65535)
    database: int = Field(default=0, ge=0, le=15)

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.database}"


class ServiceSettings(StrictSettingsModel):
    prefect_api_url: str = "http://127.0.0.1:4200/api"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"


class PathSettings(StrictSettingsModel):
    data: Path = Path("data")
    logs: Path = Path("logs")
    artifacts: Path = Path("artifacts")


class LiveTradingSettings(StrictSettingsModel):
    enabled: bool = False
    tier: LiveTier = LiveTier.READ_ONLY
    require_manual_confirmation: bool = True
    max_capital_cny: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def reject_unsafe_tier(self) -> Self:
        if self.tier is LiveTier.FULL:
            raise ValueError("LIVE_FULL is intentionally unavailable in the MVP")
        if self.tier is LiveTier.SMALL_CAPITAL and self.max_capital_cny <= 0:
            raise ValueError("SMALL_CAPITAL requires a positive max_capital_cny")
        return self


class TushareDataSettings(StrictSettingsModel):
    enabled: bool = False
    token_env: str = "TUSHARE_TOKEN"
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class AkshareDataSettings(StrictSettingsModel):
    enabled: bool = True


class BaostockDataSettings(StrictSettingsModel):
    enabled: bool = True


class DataSettings(StrictSettingsModel):
    market: str = "cn_equity"
    timezone: str = "Asia/Shanghai"
    raw_immutable: bool = True
    point_in_time_required: bool = True
    primary_provider: str = "akshare"
    validation_provider: str | None = "baostock"
    quarantine_on_quality_failure: bool = True
    tushare: TushareDataSettings = Field(default_factory=TushareDataSettings)
    akshare: AkshareDataSettings = Field(default_factory=AkshareDataSettings)
    baostock: BaostockDataSettings = Field(default_factory=BaostockDataSettings)

    @model_validator(mode="after")
    def require_safety_guards(self) -> Self:
        if not self.raw_immutable or not self.point_in_time_required:
            raise ValueError("raw immutability and point-in-time access cannot be disabled")
        if not self.quarantine_on_quality_failure:
            raise ValueError("quality failures must enter quarantine")
        if self.primary_provider == self.validation_provider:
            raise ValueError("primary and validation providers must be different")
        return self


class BacktestSettings(StrictSettingsModel):
    signal_time: str = "close"
    earliest_fill: str = "next_session"
    fill_price_model: str = "next_open"
    initial_cash_cny: int = Field(default=1_000_000, gt=0)
    allow_same_bar_fill: bool = False
    deterministic_seed: int = 20260716

    @model_validator(mode="after")
    def prevent_same_bar_fill(self) -> Self:
        if self.allow_same_bar_fill:
            raise ValueError("same-bar fill is disabled to prevent signal/fill leakage")
        return self


class RiskSettings(StrictSettingsModel):
    max_single_name_weight: float = Field(default=0.05, gt=0, le=1)
    max_industry_weight: float = Field(default=0.30, gt=0, le=1)
    max_daily_turnover: float = Field(default=0.20, gt=0, le=1)
    max_volume_participation: float = Field(default=0.10, gt=0, le=1)
    reject_stale_market_data: bool = True
    reject_unpublished_data_release: bool = True
    kill_switch_required: bool = True

    @model_validator(mode="after")
    def require_fail_closed_guards(self) -> Self:
        if not all(
            (
                self.reject_stale_market_data,
                self.reject_unpublished_data_release,
                self.kill_switch_required,
            )
        ):
            raise ValueError(
                "stale data, unpublished releases, and missing kill switch must fail closed"
            )
        return self


class DiscretionaryStrategySettings(StrictSettingsModel):
    enabled: bool = False
    strategy_id: str = "discretionary-fundamental-v1"
    require_human_approval: bool = True
    allow_automated_order_generation: bool = False
    compliant_sources_only: bool = True
    competing_hypothesis_count: int = Field(default=3, ge=3, le=3)
    key_variable_count: int = Field(default=3, ge=3, le=3)
    minimum_independent_non_price_clusters: int = Field(default=2, ge=2)
    observation_weight_cap: Decimal = Field(default=Decimal("0.005"), gt=0, le=1)
    evidence_weight_cap: Decimal = Field(default=Decimal("0.02"), gt=0, le=1)
    core_weight_cap: Decimal = Field(default=Decimal("0.05"), gt=0, le=1)
    minimum_probability_edge: Decimal = Field(default=Decimal("0.05"), ge=0, le=1)
    minimum_evidence_upgrade: Decimal = Field(default=Decimal("0.05"), ge=0, le=1)
    correlation_floor: Decimal = Field(default=Decimal("0.25"), ge=0, le=1)
    sizing_multiplier: Decimal = Field(default=Decimal("0.05"), gt=0)
    calibration_prior_weight: Decimal = Field(default=Decimal("4"), gt=0)

    @model_validator(mode="after")
    def require_governed_research(self) -> Self:
        if not self.require_human_approval:
            raise ValueError("discretionary strategy always requires human approval")
        if self.allow_automated_order_generation:
            raise ValueError("discretionary research cannot automatically generate broker orders")
        if not self.compliant_sources_only:
            raise ValueError("discretionary evidence must use compliant acquisition sources")
        if not (self.observation_weight_cap <= self.evidence_weight_cap <= self.core_weight_cap):
            raise ValueError("discretionary position caps must increase by evidence tier")
        return self


class BrokerSettings(StrictSettingsModel):
    name: str
    mode: RunMode
    enabled: bool = False
    live_submission_enabled: bool = False
    endpoint: str | None = None
    account_id: str | None = None

    @model_validator(mode="after")
    def validate_submission_gate(self) -> Self:
        if self.live_submission_enabled and (not self.enabled or self.mode is not RunMode.LIVE):
            raise ValueError("live broker submission requires an enabled LIVE broker")
        return self


class AppSettings(StrictSettingsModel):
    environment: str = "development"
    mode: RunMode = RunMode.BACKTEST
    timezone: str = "Asia/Shanghai"
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    services: ServiceSettings = Field(default_factory=ServiceSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    live_trading: LiveTradingSettings = Field(default_factory=LiveTradingSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    discretionary: DiscretionaryStrategySettings = Field(
        default_factory=DiscretionaryStrategySettings
    )
    broker: BrokerSettings | None = None

    @model_validator(mode="after")
    def protect_live_mode(self) -> Self:
        if self.mode is RunMode.LIVE:
            if not self.live_trading.enabled:
                raise ValueError("LIVE mode requires live_trading.enabled=true")
            if not self.live_trading.require_manual_confirmation:
                raise ValueError("LIVE mode requires manual confirmation")
        if (
            self.broker is not None
            and self.broker.live_submission_enabled
            and (self.mode is not RunMode.LIVE or not self.live_trading.enabled)
        ):
            raise ValueError("live broker submission requires the application LIVE gate")
        if self.discretionary.core_weight_cap > Decimal(str(self.risk.max_single_name_weight)):
            raise ValueError("discretionary core cap cannot exceed the global single-name cap")
        return self

    def ensure_runtime_directories(self, root: Path) -> None:
        for configured_path in (self.paths.data, self.paths.logs, self.paths.artifacts):
            path = configured_path if configured_path.is_absolute() else root / configured_path
            path.mkdir(parents=True, exist_ok=True)

    def safe_summary(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "mode": self.mode.value,
            "timezone": self.timezone,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "user": self.database.user,
                "password": "**********",
                "pool_size": self.database.pool_size,
            },
            "redis": self.redis.model_dump(),
            "services": self.services.model_dump(),
            "paths": {key: str(value) for key, value in self.paths.model_dump().items()},
            "live_trading": self.live_trading.model_dump(mode="json"),
            "data": self.data.model_dump(mode="json"),
            "backtest": self.backtest.model_dump(mode="json"),
            "risk": self.risk.model_dump(mode="json"),
            "discretionary": self.discretionary.model_dump(mode="json"),
            "broker": self.broker.model_dump(mode="json") if self.broker else None,
        }


def _deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> None:
    for key, value in overlay.items():
        existing = base.get(key)
        if isinstance(existing, MutableMapping) and isinstance(value, Mapping):
            _deep_merge(existing, value)
        else:
            base[key] = value


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return loaded


def _apply_env_overrides(config: MutableMapping[str, Any], environ: Mapping[str, str]) -> None:
    for name, raw_value in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in name[len(ENV_PREFIX) :].split("__") if part]
        if not path:
            continue
        cursor: MutableMapping[str, Any] = config
        for part in path[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, MutableMapping):
                raise ValueError(f"environment override conflicts with scalar key: {name}")
            cursor = child
        cursor[path[-1]] = yaml.safe_load(raw_value)


def load_settings(
    paths: Path | str | Sequence[Path | str],
    *,
    environ: Mapping[str, str] | None = None,
) -> AppSettings:
    path_list = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not path_list:
        raise ValueError("at least one configuration file is required")

    merged: dict[str, Any] = {}
    for raw_path in path_list:
        _deep_merge(merged, _read_yaml(Path(raw_path)))
    _apply_env_overrides(merged, os.environ if environ is None else environ)
    return AppSettings.model_validate(merged)
