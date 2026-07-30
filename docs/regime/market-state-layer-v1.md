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

## Implemented v1 core

The code catalog registers 41 states across style, index-relative strength,
liquidity, valuation, and microcap crowding. It includes the requested 40-day
growth/value spread, HS300/CSI2000 spreads, HS300 amount share, CSI1000 turnover,
microcap PB history, dividend yield aggregations, and a research-only microcap
crowding smoke composite.

The raw crowding inputs are retained. The composite is an equal-weight mean of
at least three valid, past-only percentile inputs and remains `DRAFT`.

## L3 interface

`build_registered_interactions` creates only allow-listed stock-family by
state-family interactions and enforces a configurable complexity cap. It does not
generate a Cartesian product or promote any state to production.
