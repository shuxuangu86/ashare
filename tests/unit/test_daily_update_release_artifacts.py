from pathlib import Path


def test_daily_update_scheduler_contains_only_merged_close_and_morning_tasks() -> None:
    root = Path(__file__).resolve().parents[2]
    installer = (root / "scripts/install_daily_update_tasks.ps1").read_text(encoding="utf-8")

    assert '[string]$CloseTime = "19:30"' in installer
    assert '[string]$MorningTime = "08:30"' in installer
    assert "uv run --extra data python" in installer
    assert "--workers 4 --interval 0.25" in installer
    assert "17:20" not in installer
    assert '"$TaskPrefix-Daily-Close"' in installer
    assert '"$TaskPrefix-Daily-Morning-Recheck"' in installer


def test_beijing_exchange_migration_is_forward_compatible() -> None:
    root = Path(__file__).resolve().parents[2]
    initial = (root / "infra/postgres/migrations/001_reference_and_ingestion.sql").read_text()
    migration = (root / "infra/postgres/migrations/004_beijing_stock_exchange.sql").read_text()

    assert "'XBSE'" in initial
    assert "DROP CONSTRAINT IF EXISTS instruments_exchange_check" in migration
    assert "DROP CONSTRAINT IF EXISTS trading_calendar_exchange_check" in migration
    assert migration.count("'XBSE'") == 2
