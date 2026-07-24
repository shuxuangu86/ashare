.PHONY: bootstrap lock test quality format config-check infra-config infra-up infra-down tushare-backfill tushare-status tushare-daily-close tushare-daily-morning tushare-daily-status tushare-history-materialize microcap-history-smoke

HISTORY_END_DATE ?= 20260717
HISTORY_RELEASE_ID ?= cn_equity_history_20260717_001
SMOKE_START_DATE ?= 20260601
SMOKE_END_DATE ?= 20260717

bootstrap:
	uv sync --group dev --extra data --extra research --extra free-data --extra optimization --extra services --extra ui
	uv run pre-commit install

lock:
	uv lock

test:
	uv run pytest

quality:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src
	uv run pytest

format:
	uv run ruff check --fix .
	uv run ruff format .

config-check:
	uv run aquant config-check \
		--config config/base.yaml \
		--config config/data.yaml \
		--config config/backtest.yaml \
		--config config/risk.yaml \
		--config config/strategies/discretionary.yaml \
		--config config/brokers/paper.yaml

infra-config:
	docker compose config --quiet

infra-up:
	docker compose up -d --build

infra-down:
	docker compose down

tushare-backfill:
	uv run python -u scripts/tushare_backfill.py

tushare-status:
	uv run python -u scripts/tushare_backfill.py --status-only

tushare-daily-close:
	uv run --extra data python -u scripts/tushare_daily_update.py --mode close --workers 4 --interval 0.25

tushare-daily-morning:
	uv run --extra data python -u scripts/tushare_daily_update.py --mode morning --workers 4 --interval 0.25

tushare-daily-status:
	uv run --extra data python -u scripts/tushare_daily_update.py --status-only

tushare-history-materialize:
	uv run --extra data python -u scripts/materialize_tushare_history.py \
		--release-id $(HISTORY_RELEASE_ID) \
		--end-date $(HISTORY_END_DATE)

microcap-history-smoke:
	uv run --extra data python -u scripts/run_microcap_history_smoke.py \
		--release-dir data/standard/history-release=$(HISTORY_RELEASE_ID) \
		--start-date $(SMOKE_START_DATE) \
		--end-date $(SMOKE_END_DATE)
