# Production-readiness acceptance

## Implemented controls

- Corporate-action domain events and auditable ledger application.
- Unified tradability decisions and A-share execution constraints.
- PIT-safe T+1 forward-label timing for 1/5/10/20/40 trading days.
- Institutional IC, Newey-West, grouping, cost, drawdown, regime and exposure metrics.
- Three-source redundancy convergence with residual-information checks.
- Versioned configurable production gates and lifecycle evidence hashes.
- PIT-validated neutralization operators and exposure contracts; real industry
  neutralization remains blocked until a historical industry-membership feed is released.
- Reproducible performance scenarios with resource and cache metrics.
- Immutable feature-set specifications and purged walk-forward L3 baselines.

## Measured compute scenarios

| Scenario | Compute time | Peak RSS | Notes |
|---|---:|---:|---|
| Full A-share, 1 year, 50 factors | 30.85 s | 1.23 GiB | 2.18M factor rows/s |
| Full A-share, 5 years, 50 factors | 59.21 s | 3.21 GiB | 5.86M factor rows/s |
| Full A-share, 10 years, 73 factors | 174.68 s | 7.15 GiB | 5.86M factor rows/s |
| Full A-share, 1 year, 500 workloads | 13.08 s | 1.46 GiB | 73 unique; 85.4% DAG cache hits |
| Single-day incremental, 73 factors | 14.52 s | 1.55 GiB | Includes 400-day history load |

These are local compute-only measurements. OS-cached disk reads may report zero, and
Factor Store size is zero unless a materialization benchmark is requested.

## External-data limitations

- The current release exposes implemented cash and stock dividends. Rights issues,
  split/reverse-split and delisting event feeds are not yet connected.
- Event alpha data is not connected and no event factors are synthesized.
- The current Standard release has daily PIT float market cap, but `stock_basic`
  intentionally contains no historical industry membership. Consequently
  `baseline_neutral_v1` may be published only as `DRAFT`, not promoted.
- No factor may enter `PRODUCTION` until the long-history evaluation, convergence,
  neutralized re-evaluation and configured gate all pass on a pinned data release.
