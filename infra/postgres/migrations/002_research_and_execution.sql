BEGIN;

CREATE TABLE aquant.data_releases (
    release_id text PRIMARY KEY CHECK (release_id ~ '^cn_equity_[0-9]{8}_[0-9]{3}$'),
    status text NOT NULL CHECK (status IN ('DRAFT', 'VALIDATING', 'PUBLISHED', 'QUARANTINED')),
    manifest_sha256 text NOT NULL CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE aquant.factor_definitions (
    factor_id text NOT NULL,
    version text NOT NULL,
    expression text NOT NULL,
    expression_hash text NOT NULL CHECK (expression_hash ~ '^[0-9a-f]{64}$'),
    status text NOT NULL CHECK (status IN ('DRAFT','COMPUTED','VALIDATED','APPROVED','PRODUCTION','DEPRECATED')),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (factor_id, version),
    UNIQUE (expression_hash)
);

CREATE TABLE aquant.experiment_runs (
    run_id text PRIMARY KEY,
    git_hash text NOT NULL,
    data_release_id text NOT NULL REFERENCES aquant.data_releases(release_id),
    config_hash text NOT NULL CHECK (config_hash ~ '^[0-9a-f]{64}$'),
    metrics jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE aquant.target_portfolios (
    strategy_id text NOT NULL,
    trade_date date NOT NULL,
    symbol text NOT NULL REFERENCES aquant.instruments(symbol),
    target_weight numeric(18,10) NOT NULL CHECK (target_weight BETWEEN 0 AND 1),
    data_release_id text NOT NULL REFERENCES aquant.data_releases(release_id),
    signal_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (strategy_id, trade_date, symbol)
);

CREATE TABLE aquant.order_intents (
    intent_id uuid PRIMARY KEY,
    idempotency_key text NOT NULL UNIQUE CHECK (idempotency_key ~ '^[0-9a-f]{64}$'),
    account_id text NOT NULL,
    strategy_id text NOT NULL,
    trade_date date NOT NULL,
    symbol text NOT NULL REFERENCES aquant.instruments(symbol),
    side text NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity bigint NOT NULL CHECK (quantity > 0),
    status text NOT NULL,
    created_at timestamptz NOT NULL
);

CREATE TABLE aquant.broker_orders (
    broker_order_id text PRIMARY KEY,
    intent_id uuid NOT NULL REFERENCES aquant.order_intents(intent_id),
    status text NOT NULL,
    submitted_at timestamptz NOT NULL,
    raw_payload jsonb NOT NULL
);

CREATE TABLE aquant.fills (
    broker_trade_id text PRIMARY KEY,
    broker_order_id text NOT NULL REFERENCES aquant.broker_orders(broker_order_id),
    symbol text NOT NULL REFERENCES aquant.instruments(symbol),
    side text NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity bigint NOT NULL CHECK (quantity > 0),
    price numeric(20,8) NOT NULL CHECK (price > 0),
    fee numeric(20,8) NOT NULL CHECK (fee >= 0),
    occurred_at timestamptz NOT NULL
);

CREATE TABLE aquant.positions (
    account_id text NOT NULL,
    symbol text NOT NULL REFERENCES aquant.instruments(symbol),
    asof_time timestamptz NOT NULL,
    quantity bigint NOT NULL CHECK (quantity >= 0),
    source text NOT NULL CHECK (source IN ('BROKER', 'LOCAL_DERIVED')),
    PRIMARY KEY (account_id, symbol, asof_time, source)
);

CREATE TABLE aquant.audit_events (
    event_id uuid PRIMARY KEY,
    event_type text NOT NULL,
    occurred_at timestamptz NOT NULL,
    payload jsonb NOT NULL,
    payload_hash text NOT NULL CHECK (payload_hash ~ '^[0-9a-f]{64}$')
);

COMMIT;
