# Market State Factor Catalog

| Family | Registered | Core contents |
|---|---:|---|
| Style | 15 | Growth, value, and compounded relative returns at 5/20/40/60/120 days |
| Index relative | 6 | HS300 minus CSI2000 at 5/20/40/60/120/252 days |
| Liquidity | 6 | HS300 amount share and CSI1000 aggregate turnover |
| Valuation | 8 | Microcap PB and dividend-index yield levels/percentiles |
| Crowding | 6 | Five retained inputs and one draft smoke composite |
| **Total** | **41** | All definitions have a deterministic content hash |

Multi-window states are deliberately retained. `correlation_cluster` and
`related_states` provide L3 metadata without deleting economically distinct
horizons.

All registered definitions are initially `DRAFT`. Registration means that the
formula and dependency contract exist; it does not imply that source data have
been materialized or research-validated.
