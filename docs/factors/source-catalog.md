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

`SRC_ALPHA101` and `SRC_GTJA_ALPHA191` are intentionally not reported as exact
reproductions in v2.0. The repository has no prior Alpha101/191 package, and silent
VWAP approximation or unaudited public transcription would violate the source-faithfulness
contract. Their source coverage remains explicitly zero until each formula is verified.

## Data semantics

Daily close-derived factors have `availability_lag=1`: a value computed after the close
is first usable at the next tradable session. `amount` is CNY and `volume` is shares in
the standardized bar schema. No factor derives a field named VWAP from `amount / volume`
without a separate unit and source validation. Financial features must use
`announcement_date`, `available_at`, and `revision_no`; missing historical financial
fields are recorded as dependencies rather than backfilled from report-period dates.

Every new factor records `source_id`, formula identifier, faithfulness, implementation
notes, required datasets, role, complexity, variant parent, and deterministic content hash.
