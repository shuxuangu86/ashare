BEGIN;

CREATE SCHEMA IF NOT EXISTS aquant;

CREATE TABLE aquant.instruments (
    symbol              text PRIMARY KEY,
    security_code       text NOT NULL CHECK (security_code ~ '^[0-9]{6}$'),
    exchange            text NOT NULL CHECK (exchange IN ('XSHG', 'XSHE')),
    name                text NOT NULL CHECK (length(trim(name)) > 0),
    security_type       text NOT NULL CHECK (security_type = 'STOCK'),
    board               text NOT NULL CHECK (board IN ('MAIN', 'STAR', 'CHINEXT', 'OTHER')),
    list_date           date NOT NULL,
    delist_date         date,
    source              text NOT NULL,
    source_record_id    text NOT NULL,
    ingested_at         timestamptz NOT NULL,
    raw_batch_id        uuid NOT NULL,
    CHECK (symbol = security_code || '.' || exchange),
    CHECK (delist_date IS NULL OR delist_date >= list_date),
    UNIQUE (source, source_record_id)
);

CREATE TABLE aquant.trading_calendar (
    exchange            text NOT NULL CHECK (exchange IN ('XSHG', 'XSHE')),
    trade_date          date NOT NULL,
    is_open             boolean NOT NULL,
    source              text NOT NULL,
    source_record_id    text NOT NULL,
    ingested_at         timestamptz NOT NULL,
    raw_batch_id        uuid NOT NULL,
    PRIMARY KEY (exchange, trade_date),
    UNIQUE (source, source_record_id)
);

CREATE TABLE aquant.ingestion_batches (
    batch_id             uuid PRIMARY KEY,
    provider             text NOT NULL,
    provider_version     text NOT NULL,
    dataset              text NOT NULL,
    request_params       jsonb NOT NULL,
    source_request_id    text NOT NULL,
    response_started_at  timestamptz NOT NULL,
    response_completed_at timestamptz NOT NULL,
    payload_uri          text NOT NULL UNIQUE,
    media_type           text NOT NULL,
    transport_status     integer CHECK (transport_status BETWEEN 100 AND 599),
    sha256               text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    byte_count           bigint NOT NULL CHECK (byte_count >= 0),
    record_count         bigint CHECK (record_count >= 0),
    status               text NOT NULL CHECK (status IN ('RECEIVED', 'VALIDATED', 'QUARANTINED')),
    created_at           timestamptz NOT NULL DEFAULT now(),
    CHECK (response_completed_at >= response_started_at),
    UNIQUE (provider, dataset, source_request_id)
);

ALTER TABLE aquant.instruments
    ADD CONSTRAINT instruments_raw_batch_fk
    FOREIGN KEY (raw_batch_id) REFERENCES aquant.ingestion_batches(batch_id);

ALTER TABLE aquant.trading_calendar
    ADD CONSTRAINT trading_calendar_raw_batch_fk
    FOREIGN KEY (raw_batch_id) REFERENCES aquant.ingestion_batches(batch_id);

COMMIT;
