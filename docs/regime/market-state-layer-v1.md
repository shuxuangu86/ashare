# AQuant Market State Layer v1

The market-state layer is an L2-R time-series feature space. It describes the
environment shared by all securities on a trade date; it is not a cross-sectional
stock-selection factor and is not a production timing signal.

## Safety contract

- Every close-derived value for date T has an explicit `available_date` after T.
- Index and universe aggregates require historical point-in-time membership.
- Rolling and expanding percentiles use only the prefix ending at the observation.
- Definitions and materialized rows are content addressed.
- Missing inputs remain missing; current membership is never backfilled.

## Implemented v1

The code catalog registers 350 states across all ten requested families. Based on
the current repository data, 307 are implemented, 14 are partial because of
incomplete CSI1000 snapshots, and 29 are explicitly dependency-missing. A full
2007-01-04 through 2026-07-17 run materialized 321 state series and 1,336,033
T+1-available rows.

The final release also writes per-family checkpoints, daily PIT universe counts,
definition lineage, six versioned feature sets, five-horizon conditional-return
evaluations against ten available market/style targets, and episode-level 5/20-day
future returns. Resume skips only when the release/config/universe input hash is
unchanged.

The raw crowding inputs are retained. The composite is an equal-weight mean of
at least three valid, past-only percentile inputs and remains `DRAFT`.

## L3 interface

`build_registered_interactions` creates only allow-listed stock-family by
state-family interactions and enforces a configurable complexity cap. It does not
generate a Cartesian product or promote any state to production.

The research smoke uses an expanding two-fold walk-forward split. Model A uses
stock factors, Model B adds market states, and Model C adds three registered
interactions. Standardization is fit on each training fold only.

On the current factor materialization, Model A/B/C OOS RankIC is respectively
0.00770, 0.00988, and 0.00895. These are smoke-test diagnostics and remain
`RESEARCH_ONLY`; they are not production timing claims.
