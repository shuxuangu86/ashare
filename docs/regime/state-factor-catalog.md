# Market State Factor Catalog

| Family | Registered | Core contents |
|---|---:|---|
| Style | 15 | Growth, value, and compounded relative returns at 5/20/40/60/120 days |
| Index relative | 6 | HS300 minus CSI2000 at 5/20/40/60/120/252 days |
| Liquidity | 6 | HS300 amount share and CSI1000 aggregate turnover |
| Valuation | 64 | PB level, dispersion, change and historical percentiles |
| Crowding | 6 | Five retained inputs and one draft smoke composite |
| Trend | 78 | Returns, drawdowns, moving-average distances and slopes |
| Breadth | 112 | Advances, moving-average breadth, highs/lows, limits and tails |
| Volatility | 18 | Multi-window annualized realized volatility |
| Risk appetite | 24 | Dynamic-portfolio relative returns |
| Fundamentals | 21 | PIT profit-growth and leverage aggregates plus missing-source audit |
| **Total** | **350** | All definitions have a deterministic content hash |

Multi-window states are deliberately retained. `correlation_cluster` and
`related_states` provide L3 metadata without deleting economically distinct
horizons.

All releases remain `DRAFT`. Definition status is separate: 307 are
`IMPLEMENTED`, 14 `PARTIAL`, and 29 `DATA_DEPENDENCY_MISSING`.
