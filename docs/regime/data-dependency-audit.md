# Market State Data Dependency Audit

| Dependency | Used by | Status | PIT requirement |
|---|---|---|---|
| Daily index bars | Style and index-relative returns | Partial | Bar available after close |
| Historical index membership | Amount, turnover, dividend yield | Required | Effective-date membership only |
| Daily stock amount | Liquidity and crowding | Available in project data model | Tradable ALL_A denominator |
| Free-float market value | Turnover and weighted yield | Available in project data model | Same-date visible value |
| PB and dividend yield | Valuation | Partial | Same-date released daily-basic snapshot |
| Growth/value index mapping | Style returns | Missing configuration | Must not substitute current constituents |
| 10Y government bond yield | Yield spread | Missing | State remains unregistered until PIT source exists |
| Institutional holdings | Crowding | Missing | No proxy is named as actual holdings |

The implementation supplies computation primitives and definitions. Real-data
materialization remains blocked for a state whenever its historical membership
or configured index source is absent; other families can continue independently.
