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


def test_discretionary_research_template_matches_minimum_research_structure() -> None:
    template = yaml.safe_load(
        (PROJECT_ROOT / "research/templates/discretionary-case.yaml").read_text(encoding="utf-8")
    )

    assert {item["kind"] for item in template["hypotheses"]} == {
        "FUNDAMENTAL",
        "FLOW_TECHNICAL",
        "NOISE",
    }
    assert len(template["key_variables"]) == 3
    for variable in template["key_variables"]:
        assert len(variable["leading_indicators"]) >= 2
        assert all(
            len(indicator["source_clusters"]) >= 2 for indicator in variable["leading_indicators"]
        )
    assert [node["stage"] for node in template["causal_chain"]] == [
        "INDUSTRY_CHANGE",
        "CUSTOMER_BEHAVIOR",
        "COMPANY_ORDERS",
        "REVENUE",
        "MARGIN",
        "CASH_FLOW",
        "SHAREHOLDER_RETURN",
    ]


def test_discretionary_migration_contains_pit_evidence_and_human_approval_tables() -> None:
    migration = (
        PROJECT_ROOT / "infra/postgres/migrations/003_discretionary_strategy.sql"
    ).read_text(encoding="utf-8")

    for table in (
        "knowledge_candidates",
        "discretionary_radar_signals",
        "discretionary_research_cases",
        "discretionary_hypotheses",
        "discretionary_causal_nodes",
        "discretionary_evidence",
        "discretionary_belief_snapshots",
        "discretionary_market_expectations",
        "discretionary_calibration_outcomes",
        "discretionary_allocation_proposals",
        "discretionary_allocation_approvals",
        "discretionary_postmortems",
    ):
        assert f"CREATE TABLE aquant.{table}" in migration
    assert "prevent_discretionary_evidence_mutation" in migration
    assert "belief_snapshot_id text NOT NULL REFERENCES" in migration
    assert "market_expectation_id text NOT NULL REFERENCES" in migration
    assert "research_stage text NOT NULL CHECK" in migration
    assert "evidence_strength numeric(18,12) NOT NULL" in migration
    assert "validate_discretionary_proposal_lineage" in migration
    assert "validate_discretionary_approval" in migration
    assert "requires_manual_approval boolean NOT NULL DEFAULT true" in migration
    assert "manual_confirmation boolean NOT NULL CHECK (manual_confirmation)" in migration


def test_compose_installs_all_transactional_migrations() -> None:
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "001_reference_and_ingestion.sql" in compose
    assert "002_research_and_execution.sql" in compose
    assert "003_discretionary_strategy.sql" in compose
