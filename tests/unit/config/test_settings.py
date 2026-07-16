from pathlib import Path

import pytest
from pydantic import ValidationError

from aquant.config.settings import AppSettings, load_settings
from aquant.domain.enums import LiveTier, RunMode

PROJECT_ROOT = Path(__file__).parents[3]


def test_base_configuration_is_safe() -> None:
    settings = load_settings(PROJECT_ROOT / "config" / "base.yaml", environ={})

    assert settings.mode is RunMode.BACKTEST
    assert settings.live_trading.enabled is False
    assert settings.live_trading.tier is LiveTier.READ_ONLY
    assert settings.live_trading.require_manual_confirmation is True


def test_all_day_one_configuration_layers_validate_together() -> None:
    settings = load_settings(
        [
            PROJECT_ROOT / "config" / "base.yaml",
            PROJECT_ROOT / "config" / "data.yaml",
            PROJECT_ROOT / "config" / "backtest.yaml",
            PROJECT_ROOT / "config" / "risk.yaml",
            PROJECT_ROOT / "config" / "brokers" / "paper.yaml",
        ],
        environ={},
    )

    assert settings.data.primary_provider == "akshare"
    assert settings.data.validation_provider == "baostock"
    assert settings.data.tushare.enabled is False
    assert settings.data.raw_immutable is True
    assert settings.backtest.allow_same_bar_fill is False
    assert settings.risk.kill_switch_required is True
    assert settings.broker is not None
    assert settings.broker.name == "paper"


def test_nested_environment_override_has_highest_priority() -> None:
    settings = load_settings(
        PROJECT_ROOT / "config" / "base.yaml",
        environ={
            "AQUANT_MODE": "PAPER",
            "AQUANT_DATABASE__PORT": "55432",
            "AQUANT_LIVE_TRADING__ENABLED": "false",
            "UNRELATED": "ignored",
        },
    )

    assert settings.mode is RunMode.PAPER
    assert settings.database.port == 55432


def test_overlay_merges_without_losing_nested_base_values(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("database:\n  pool_size: 20\n", encoding="utf-8")

    settings = load_settings([PROJECT_ROOT / "config" / "base.yaml", overlay], environ={})

    assert settings.database.pool_size == 20
    assert settings.database.host == "127.0.0.1"


def test_live_mode_is_disabled_unless_explicitly_unlocked() -> None:
    with pytest.raises(ValidationError, match="enabled=true"):
        AppSettings(mode=RunMode.LIVE)


def test_live_mode_always_requires_manual_confirmation() -> None:
    with pytest.raises(ValidationError, match="manual confirmation"):
        AppSettings.model_validate(
            {
                "mode": "LIVE",
                "live_trading": {
                    "enabled": True,
                    "tier": "READ_ONLY",
                    "require_manual_confirmation": False,
                },
            }
        )


def test_live_full_is_unavailable() -> None:
    with pytest.raises(ValidationError, match="intentionally unavailable"):
        AppSettings.model_validate({"live_trading": {"tier": "FULL"}})


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"raw_immutable": False}},
        {"data": {"point_in_time_required": False}},
        {"data": {"quarantine_on_quality_failure": False}},
        {"data": {"primary_provider": "akshare", "validation_provider": "akshare"}},
        {"backtest": {"allow_same_bar_fill": True}},
        {"risk": {"reject_stale_market_data": False}},
        {"risk": {"reject_unpublished_data_release": False}},
        {"risk": {"kill_switch_required": False}},
    ],
)
def test_safety_critical_configuration_cannot_be_disabled(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AppSettings.model_validate(payload)


def test_unknown_configuration_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AppSettings.model_validate({"mod": "PAPER"})


def test_live_broker_submission_requires_all_gates() -> None:
    with pytest.raises(ValidationError, match="enabled LIVE broker"):
        AppSettings.model_validate(
            {
                "broker": {
                    "name": "qmt",
                    "mode": "LIVE",
                    "enabled": False,
                    "live_submission_enabled": True,
                }
            }
        )
    with pytest.raises(ValidationError, match="application LIVE gate"):
        AppSettings.model_validate(
            {
                "broker": {
                    "name": "qmt",
                    "mode": "LIVE",
                    "enabled": True,
                    "live_submission_enabled": True,
                }
            }
        )


def test_small_capital_requires_explicit_positive_limit() -> None:
    with pytest.raises(ValidationError, match="positive max_capital"):
        AppSettings.model_validate({"live_trading": {"tier": "SMALL_CAPITAL"}})


def test_safe_summary_masks_database_password() -> None:
    settings = AppSettings()

    assert settings.database.dsn.endswith("@127.0.0.1:5432/aquant")
    assert settings.safe_summary()["database"]["password"] == "**********"
    assert settings.database.password.get_secret_value() not in str(settings.safe_summary())


def test_runtime_directories_are_created(tmp_path: Path) -> None:
    settings = AppSettings()

    settings.ensure_runtime_directories(tmp_path)

    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "artifacts").is_dir()


def test_empty_configuration_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        load_settings([], environ={})


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("- invalid\n- root\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_settings(config, environ={})


def test_conflicting_environment_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="conflicts with scalar"):
        load_settings(
            PROJECT_ROOT / "config" / "base.yaml",
            environ={"AQUANT_MODE__NESTED": "invalid"},
        )
