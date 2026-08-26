#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/gsx1339/aquant

prehistory="reports/l3-nested-prehistory-size-adjusted-2016-2021-v2"
current_reports="reports/l2-v2-all-size-neutral-adjusted-5y-v2"
release="data/standard/history-release=cn_equity_history_20260717_001"
pool="artifacts/l3_nested_walk_forward/fold_pools.json"
union_cache="/mnt/d/AQuantEvaluation/convergence/l3-nested-union-size-adjusted-2016-2026-v2"
l3_scores="/mnt/d/AQuantEvaluation/l3_nested_walk_forward/l3_scores.npy"
l3_metadata="artifacts/l3_nested_walk_forward/l3_metadata.json"
l4_output="artifacts/l3_nested_walk_forward/l4_backtest"
windows_export="/mnt/c/Users/Admin/Documents/A-share Quant/AQuant Reports/nested-walk-forward"

mkdir -p artifacts/logs artifacts/l3_nested_walk_forward \
  /mnt/d/AQuantEvaluation/l3_nested_walk_forward

json_value() {
  .venv/bin/python -c \
    'import json,sys; value=json.load(open(sys.argv[1]));
for key in sys.argv[2].split("."):
    value=value.get(key, "") if isinstance(value, dict) else ""
print(value)' "$1" "$2"
}

if [[ "${1:-}" == "--l3-worker" ]]; then
  trap 'printf "%s\tFAILED\tline=%s\n" "$(date --iso-8601=seconds)" "$LINENO" >artifacts/logs/nested-l3.status' ERR
  printf '%s\tRUNNING\n' "$(date --iso-8601=seconds)" >artifacts/logs/nested-l3.status
  /usr/bin/time -v .venv/bin/python scripts/train_nested_l3.py \
    --cache-dir "${union_cache}" \
    --history-release "${release}" \
    --fold-pools "${pool}" \
    --output-scores "${l3_scores}" \
    --output-metadata "${l3_metadata}" \
    --horizon 5 \
    --inner-validation-dates 252 \
    --inner-purge-dates 6 \
    --maximum-training-rows 200000 \
    --code-version "$(git rev-parse HEAD)"
  printf '%s\tPASS\n' "$(date --iso-8601=seconds)" >artifacts/logs/nested-l3.status
  exit 0
fi

if [[ "${1:-}" == "--l4-worker" ]]; then
  trap 'printf "%s\tFAILED\tline=%s\n" "$(date --iso-8601=seconds)" "$LINENO" >artifacts/logs/nested-l4.status' ERR
  printf '%s\tRUNNING\n' "$(date --iso-8601=seconds)" >artifacts/logs/nested-l4.status
  /usr/bin/time -v .venv/bin/python scripts/run_nested_l4_backtest.py \
    --history-release "${release}" \
    --cache-dir "${union_cache}" \
    --l3-scores "${l3_scores}" \
    --l3-metadata "${l3_metadata}" \
    --output-dir "${l4_output}" \
    --code-version "$(git rev-parse HEAD)"
  mkdir -p "${windows_export}"
  cp "${l4_output}/nested_l4_backtest.json" \
    "${l4_output}/nested_l4_backtest.md" \
    "${l4_output}/nested_l4_daily_oos.csv" \
    "${l4_output}/nested_l4_fills.csv" \
    "${l4_output}/benchmark_proxy.csv" \
    "${l3_metadata}" \
    "${pool}" \
    "${windows_export}/"
  printf '%s\tPASS\texport=%s\n' \
    "$(date --iso-8601=seconds)" "${windows_export}" \
    >artifacts/logs/nested-l4.status
  exit 0
fi

pre_manifest="${prehistory}/evaluation_manifest.json"
if [[ ! -f "${pre_manifest}" ]] || [[ "$(json_value "${pre_manifest}" status)" != "PASS" ]]; then
  echo "WAIT_PREHISTORY"
  exit 0
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "WAIT_TRACKED_WORKTREE_CLEAN"
  exit 0
fi

code_version="$(git rev-parse HEAD)"

if [[ ! -f "${pool}" ]]; then
  .venv/bin/python scripts/build_nested_wf_l2_pools.py \
    --prehistory-report-dir "${prehistory}" \
    --current-report-dir "${current_reports}" \
    --release-dir "${release}" \
    --backtest-start 2021-07-19 \
    --backtest-end 2026-07-17 \
    --output "${pool}" \
    --fold-count 5 \
    --purge-observations 6 \
    --maximum-family-members 9
fi

union_metadata="${union_cache}/metadata.json"
if [[ ! -f "${union_metadata}" ]] || [[ "$(json_value "${union_metadata}" status)" != "PASS" ]]; then
  bash scripts/launch_nested_union_materialization.sh
  echo "WAIT_UNION_CACHE"
  exit 0
fi

if [[ ! -f "${l3_metadata}" ]] \
  || [[ "$(json_value "${l3_metadata}" status)" != "PASS" ]] \
  || [[ "$(json_value "${l3_metadata}" provenance.code_version)" != "${code_version}" ]]; then
  if [[ -f artifacts/logs/nested-l3.pid ]] \
    && kill -0 "$(cat artifacts/logs/nested-l3.pid)" 2>/dev/null; then
    echo "WAIT_L3"
    exit 0
  fi
  nohup bash scripts/advance_nested_wf_pipeline.sh --l3-worker \
    >artifacts/logs/nested-l3.log 2>&1 &
  echo "$!" >artifacts/logs/nested-l3.pid
  echo "STARTED_L3"
  exit 0
fi

l4_report="${l4_output}/nested_l4_backtest.json"
if [[ ! -f "${l4_report}" ]] \
  || [[ "$(json_value "${l4_report}" status)" != "PASS_RESEARCH_ONLY" ]] \
  || [[ "$(json_value "${l4_report}" code_version)" != "${code_version}" ]]; then
  if [[ -f artifacts/logs/nested-l4.pid ]] \
    && kill -0 "$(cat artifacts/logs/nested-l4.pid)" 2>/dev/null; then
    echo "WAIT_L4"
    exit 0
  fi
  nohup bash scripts/advance_nested_wf_pipeline.sh --l4-worker \
    >artifacts/logs/nested-l4.log 2>&1 &
  echo "$!" >artifacts/logs/nested-l4.pid
  echo "STARTED_L4"
  exit 0
fi

echo "PIPELINE_COMPLETE"
