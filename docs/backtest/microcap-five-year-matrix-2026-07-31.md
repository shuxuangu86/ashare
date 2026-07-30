# Five-year micro-cap paired matrix (2026-07-31)

## Scope

- Period: 2021-07-19 through 2026-07-17 (1,211 sessions).
- Data release: `cn_equity_history_20260717_001`.
- Capital: RMB 10,000,000 for every run.
- Matrix: 6 prototypes x daily/weekly/monthly x 2 paired cost cases = 36 runs.
- Zero-cost case: A-share execution rules remain enabled; fees and slippage are zero.
- Base-cost case: identical rules plus the historical fee schedule and 5 bps slippage.

Signals use the post-close T cross-section and first execute at T+1 open. The universe
excludes Beijing exchange, listings younger than 120 calendar days, historical ST and
delisting-risk names, and sessions without a bar. Targets use equal weights, 100-share
board lots, and a 2% cash buffer.

Unfilled and partially filled differences are retried daily against the immutable
rebalance-date target. Corporate-action share ratios adjust the outstanding target.
Capacity- or limit-blocked exits can temporarily leave more distinct holdings than the
selector target; this is an execution result, not an expanded selection.

## Reproduction

```bash
uv run python scripts/run_microcap_five_year_matrix.py
```

Final artifacts:

```text
artifacts/microcap_five_year_matrix/
  microcap-5y-matrix-c9240e4e9b50.json
  microcap-5y-matrix-c9240e4e9b50.md
```

The JSON retains paired monthly and annual returns, turnover, fees, slippage, direct
friction, drawdown, realized holding counts, fill rates, run times, final state hashes,
and lineage hashes.

## Validation

- Status: `PASS`.
- PIT target hash:
  `93a5b65db859cd5cb89a04aa37325f632cacde5e78578e57e23216fa52da0d5c`.
- Result content hash:
  `c9240e4e9b50dc1fe3906aaf2deff464dc57220ea07ce825dcc27eaff3f5d5e4`.
- Runtime: 1,172.29 seconds.
- Peak RSS: 5,230.93 MiB.
- Monthly rows: 1,098 (61 months x 18 strategies).
- Annual rows: 108 (6 partial/full calendar years x 18 strategies).

The 0.5% prior-20-day-average-volume participation cap is binding for several portfolios.
The cost comparison remains controlled because both paired runs use the same capacity,
price-limit, security-status, and board-lot rules.
