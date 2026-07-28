# Industry PIT data audit

## Scope

This audit covers the AQuant Raw, Standard and PIT paths required for historical
industry neutralization. It was performed against branch
`v0.3.0-microcap-baseline`, base commit `5d66288`, and the local Tushare Raw
archive on 2026-07-29.

## Findings

| Item | Result |
|---|---|
| Classification system | Shenwan 2014 and Shenwan 2021 |
| Primary source | Tushare `index_classify` and `index_member` |
| Classification fields | `index_code`, `industry_name`, `level`, `industry_code`, `parent_code`, `src`, `is_pub` |
| Membership fields | `index_code`, `con_code`, `in_date`, `out_date`, `is_new` |
| Historical intervals | Present as inclusion and removal dates |
| Announcement timestamp | Not supplied by either endpoint |
| Existing Raw coverage | SW2021: 511 classifications and 23,236 unique membership intervals |
| Existing Raw ingestion | 2026-07-23 UTC |
| Duplicate condition | 59,156 raw membership rows collapse to 23,236 exact identities; maximum multiplicity is three |
| Instrument coverage | 5,479 distinct securities in the SW2021 archive |
| Membership date range | 1990-12-10 through 2026-04-28 |
| Existing Standard release | No industry dataset |
| Existing PIT capability | Generic PIT tables and tested neutralization operators exist, but no production industry loader |

The source client already provides immutable Raw responses, SHA-256 manifests,
SQLite checkpoints, retry/backoff, throttling and resumability. The history
release publisher provides temporary-directory construction, manifest checksums
and atomic publication.

Official Tushare documentation states that `index_classify` supports `SW2014`
and `SW2021`, while the membership endpoints expose `in_date` and `out_date`.
Neither endpoint exposes an announcement timestamp:

- <https://tushare.pro/document/2?doc_id=181>
- <https://tushare.pro/document/2?doc_id=335>

## PIT policy

Membership intervals use `[effective_from, effective_to)`. Because provider
`out_date` denotes the removal session, Standard converts it to the next trading
session before storing `effective_to`.

Classification-version boundaries are explicit:

- `SW2014`: visible from its configured launch boundary and valid until
  2021-12-13.
- `SW2021`: visible and valid from 2021-12-13.

The provider does not expose the historical announcement timestamp. AQuant must
therefore leave `announced_at` null and use the conservative
`effective-session-close` availability policy: a membership becomes visible
only after the close of `effective_from`, never before. The policy is recorded
in every release manifest. Historical revisions cannot be distinguished by the
source and remain a documented limitation; they must not be silently presented
as announcement-time evidence.

## Quality risks and controls

- Exact duplicate Raw rows are deduplicated by canonical identity and retained
  in source lineage counts.
- Conflicting names, hierarchy, intervals or simultaneous memberships enter
  quarantine; no arbitrary winner is selected.
- Version validity truncates retrospective SW2021 histories before its launch.
- Unknown codes, invalid dates, interval overlaps and orphan parents fail or
  quarantine the release according to configured severity.
- The PIT query requires both interval validity and
  `available_at <= evaluation_time`.
- A future or late revision must not alter an earlier `as_of` query.

## Update method

The focused industry backfill defaults to two workers or fewer, is checkpointed
and idempotent, and archives SW2014/SW2021 classifications before their
membership pages. Standard/PIT publication is content-addressed, resumable and
never overwrites an existing verified release.

## Audit status

`IMPLEMENTATION_REQUIRED`: genuine historical source data exists, but SW2014
must be archived and the Standard/PIT industry publisher and loader must be
implemented before `baseline_neutral_v1` can leave `DRAFT`.
