# AQuant factor research source catalog

The machine-readable authority is
`configs/factors/source_registry.yaml`. The registry stores bibliographic metadata,
access status, required datasets, implementation status, and a stable registry hash.
Licensed or authorization-unclear PDFs are not committed.

## Status policy

- `IMPLEMENTED`: a tested implementation or controlled derived representation exists.
- `SOURCE_UNAVAILABLE`: metadata exists but no licensed local source supports an exact transcription.
- `FORMULA_AMBIGUOUS`: formula semantics still require a formula-by-formula audit.
- `DATA_DEPENDENCY_MISSING`: the formula is known but standardized PIT inputs are absent.
- `DEFERRED_INTRADAY`: the definition requires minute observations and is not approximated with daily data.

`SRC_ALPHA101` is registered at 101/101 and `SRC_GTJA_ALPHA191` at 191/191.
The runtime audit produces finite output for all 101 Alpha101 formulas and 190 GTJA
formulas. GTJA Alpha143 remains registered as `FORMULA_AMBIGUOUS` because the report's
`SELF` operand has no verifiable definition. Original transcriptions and adopted
formulas are both retained. OCR repairs, default-window interpretations, benchmark
proxies, VWAP derivation, and industry approximations are never labeled `EXACT`.

## Data semantics

Daily close-derived factors have `availability_lag=1`: a value computed after the close
is first usable at the next tradable session. `amount` is CNY and `volume` is shares in
the standardized bar schema. Published-formula VWAP is explicitly derived as
`amount / volume` under that validated contract and labeled `NORMALIZED_EQUIVALENT`.
Financial features must use
`announcement_date`, `available_at`, and `revision_no`; missing historical financial
fields are recorded as dependencies rather than backfilled from report-period dates.

Every new factor records `source_id`, formula identifier, faithfulness, implementation
notes, required datasets, role, complexity, variant parent, and deterministic content hash.
