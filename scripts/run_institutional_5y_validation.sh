#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

release_dir="${1:-data/standard/history-release=cn_equity_history_20260717_001}"
report_dir="${2:-reports/institutional-5y-audited}"
cache_dir="${3:-artifacts/convergence/five_year_v1}"
code_version="${4:-working-tree}"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"

mkdir -p "$report_dir"
/usr/bin/time \
  -o "$report_dir/runtime.txt" \
  -f "wall=%e,max_rss_kb=%M" \
  "$uv_bin" run --extra data --extra research python scripts/evaluate_factors.py \
  --release-dir "$release_dir" \
  --report-dir "$report_dir" \
  --data-release-id cn_equity_20260717_001 \
  --factor-set baseline_v1 \
  --universe all_a_share \
  --start-date 20210718 \
  --end-date 20260717 \
  --horizons 1,5,10,20,40 \
  --batch-size 73 \
  --code-version "$code_version" \
  --convergence-cache "$cache_dir" \
  > "$report_dir/command.json" \
  2> "$report_dir/stderr.log"
