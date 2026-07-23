BEGIN;

ALTER TABLE aquant.instruments
    DROP CONSTRAINT IF EXISTS instruments_exchange_check;
ALTER TABLE aquant.instruments
    ADD CONSTRAINT instruments_exchange_check
    CHECK (exchange IN ('XSHG', 'XSHE', 'XBSE'));

ALTER TABLE aquant.trading_calendar
    DROP CONSTRAINT IF EXISTS trading_calendar_exchange_check;
ALTER TABLE aquant.trading_calendar
    ADD CONSTRAINT trading_calendar_exchange_check
    CHECK (exchange IN ('XSHG', 'XSHE', 'XBSE'));

COMMIT;
