# Market State Data Dependency Audit

| Dependency | Used by | Status | PIT requirement |
|---|---|---|---|
| Daily index bars | HS300/CSI500/CSI1000/ChiNext trend | Available raw | Bar available after close |
| Historical index membership | HS300/CSI500/CSI1000 aggregates | Partial | T+1 snapshot; incomplete CSI1000 snapshots ignored |
| Daily stock amount | Liquidity and crowding | Available in project data model | Tradable ALL_A denominator |
| Free-float market value | Turnover and weighted yield | Available in project data model | Same-date visible value |
| PB and dividend yield | Valuation | Partial | Same-date released daily-basic snapshot |
| Growth/value style | Style returns | Implemented dynamic proxy | Monthly membership built from T-1 visible fields |
| 10Y government bond yield | Yield spread | Missing | Definition is registered dependency-missing; no values are fabricated |
| Institutional holdings | Crowding | Missing | No proxy is named as actual holdings |
| CSI2000, STAR50, dividend index | Requested index states | Missing | No fabricated history or current-member backfill |
| PE/EP/PS/FCF/ROE/revenue growth | Extended valuation/fundamentals | Missing | Registered as dependency-missing |

The full materialization continues around missing sources. Missing states remain
in the catalog with an explicit status and never receive fabricated values.
