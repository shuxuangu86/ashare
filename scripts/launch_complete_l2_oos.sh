#!/usr/bin/env bash
set -euo pipefail

cd /home/gsx1339/aquant
mkdir -p artifacts/convergence artifacts/logs reports

pid_file="artifacts/logs/l2-complete-oos.pid"
log_file="artifacts/logs/l2-complete-oos.log"
status_file="artifacts/logs/l2-complete-oos.status"

if [[ "${1:-}" != "--worker" && -f "${pid_file}" ]] \
  && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "complete L2 OOS evaluation already running with pid $(cat "${pid_file}")"
  exit 0
fi

code_version="$(git rev-parse HEAD)"
common_args=(
  --release-dir data/standard/history-release=cn_equity_history_20260717_001
  --data-release-id cn_equity_20260717_001
  --universe all_a_share
  --start-date 20210719
  --end-date 20260717
  --horizons 1,5,10,20,40
  --cost-bps 10
  --batch-size 2
  --oos-fraction 0.2
  --code-version "${code_version}"
  --reload-per-batch
  --resume
)

run_evaluation() {
  local factor_set="$1"
  local report_name="$2"
  local neutralization="$3"
  shift 3
  printf '%s\tRUNNING\t%s\n' "$(date --iso-8601=seconds)" "${report_name}" >"${status_file}"
  /usr/bin/time -v .venv/bin/python scripts/evaluate_factors.py \
    "${common_args[@]}" \
    --factor-set "${factor_set}" \
    --report-dir "reports/${report_name}" \
    --convergence-cache "artifacts/convergence/${report_name}" \
    --neutralization "${neutralization}" \
    "$@"
  printf '%s\tPASS\t%s\n' "$(date --iso-8601=seconds)" "${report_name}" >"${status_file}"
}

run_worker() {
  run_evaluation alpha191_original_v1 l2-v2-alpha191-5y-v2 raw
  run_evaluation technical_classic_level_v1 l2-v2-technical-level-5y-v2 raw
  run_evaluation technical_classic_change_v1 l2-v2-technical-change-5y-v1 raw
  run_evaluation technical_classic_state_v1 l2-v2-technical-state-5y-v1 raw
  run_evaluation technical_classic_divergence_v1 l2-v2-technical-divergence-5y-v1 raw
  run_evaluation china_7000_rules_controlled_v1 l2-v2-china-rules-controlled-5y-v2 raw
  run_evaluation l2_v2_executable l2-v2-all-size-neutral-5y-v1 size
  run_evaluation l2_v2_executable l2-v2-all-industry-proxy-5y-v1 industry_proxy \
    --industry-release data/standard/industry-release=sw_industry_pit_20260717_v3 \
    --industry-quality-report \
    artifacts/data_quality/industry_pit/sw_industry_pit_20260717_v3/20210719_20260717/quality.json
  printf '%s\tCOMPLETE\tall\n' "$(date --iso-8601=seconds)" >"${status_file}"
}

if [[ "${1:-}" == "--worker" ]]; then
  trap 'printf "%s\tFAILED\tline=%s\n" "$(date --iso-8601=seconds)" "$LINENO" >"${status_file}"' ERR
  run_worker
  exit 0
fi

nohup bash scripts/launch_complete_l2_oos.sh --worker >"${log_file}" 2>&1 &
echo "$!" >"${pid_file}"
echo "started pid $(cat "${pid_file}") log ${log_file} code ${code_version}"
