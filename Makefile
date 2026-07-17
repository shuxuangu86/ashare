.PHONY: bootstrap lock test quality format config-check infra-config infra-up infra-down tushare-backfill tushare-status

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
