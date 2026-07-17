from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_execution_migration_contains_traceability_and_idempotency_tables() -> None:
    migration = (
        PROJECT_ROOT / "infra/postgres/migrations/002_research_and_execution.sql"
    ).read_text(encoding="utf-8")

    for table in (
        "data_releases",
        "factor_definitions",
        "experiment_runs",
        "target_portfolios",
        "order_intents",
        "broker_orders",
        "fills",
        "positions",
        "audit_events",
    ):
        assert f"CREATE TABLE aquant.{table}" in migration
    assert "idempotency_key text NOT NULL UNIQUE" in migration
    assert "data_release_id text NOT NULL REFERENCES aquant.data_releases" in migration


def test_all_committed_broker_configs_disable_live_submission() -> None:
    broker_directory = PROJECT_ROOT / "config/brokers"

    for path in broker_directory.glob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["broker"]["live_submission_enabled"] is False, path


def test_compose_installs_both_transactional_migrations() -> None:
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "001_reference_and_ingestion.sql" in compose
    assert "002_research_and_execution.sql" in compose
