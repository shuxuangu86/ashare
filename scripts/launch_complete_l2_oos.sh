#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/gsx1339/aquant
mkdir -p artifacts/convergence artifacts/logs reports

pid_file="artifacts/logs/l2-complete-oos.pid"
log_file="artifacts/logs/l2-complete-oos.log"
status_file="artifacts/logs/l2-complete-oos.status"

if [[ "${1:-}" != "--worker" && "${1:-}" != "--worker-from-divergence" \
  && "${1:-}" != "--worker-from-size" \
  && -f "${pid_file}" ]] \
  && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "complete L2 OOS evaluation already running with pid $(cat "${pid_file}")"
  exit 0
fi

code_version="$(git rev-parse HEAD)"
batch_size=4
if [[ "${1:-}" == "--worker-from-size" ]]; then
  batch_size=2
fi
common_args=(
  --release-dir data/standard/history-release=cn_equity_history_20260717_001
  --data-release-id cn_equity_20260717_001
  --universe all_a_share
  --start-date 20210719
  --end-date 20260717
  --horizons 1,5,10,20,40
  --cost-bps 10
  --batch-size "${batch_size}"
  --oos-fraction 0.2
  --code-version "${code_version}"
  --reload-per-batch
  --resume
)

run_evaluation() {
  local factor_set="$1"
  local report_name="$2"
  local neutralization="$3"
  local convergence_root="${AQUANT_CONVERGENCE_ROOT:-artifacts/convergence}"
  shift 3
  printf '%s\tRUNNING\t%s\n' "$(date --iso-8601=seconds)" "${report_name}" >"${status_file}"
  /usr/bin/time -v .venv/bin/python scripts/evaluate_factors.py \
    "${common_args[@]}" \
    --factor-set "${factor_set}" \
    --report-dir "reports/${report_name}" \
    --convergence-cache "${convergence_root}/${report_name}" \
    --neutralization "${neutralization}" \
    "$@"
  printf '%s\tPASS\t%s\n' "$(date --iso-8601=seconds)" "${report_name}" >"${status_file}"
}

run_worker() {
  run_evaluation alpha191_original_v1 l2-v2-alpha191-adjusted-5y-v3 raw
  run_evaluation technical_classic_level_v1 l2-v2-technical-level-adjusted-5y-v3 raw
  run_evaluation technical_classic_change_v1 l2-v2-technical-change-adjusted-5y-v2 raw
  run_evaluation technical_classic_state_v1 l2-v2-technical-state-adjusted-5y-v2 raw
  run_evaluation technical_classic_divergence_v1 l2-v2-technical-divergence-adjusted-5y-v2 raw
  run_evaluation china_7000_rules_controlled_v1 \
    l2-v2-china-rules-controlled-adjusted-5y-v3 raw
  AQUANT_CONVERGENCE_ROOT=/mnt/d/AQuantEvaluation/convergence \
    run_evaluation l2_v2_executable l2-v2-all-size-neutral-adjusted-5y-v2 size
  AQUANT_CONVERGENCE_ROOT=/mnt/d/AQuantEvaluation/convergence \
    run_evaluation l2_v2_executable l2-v2-all-industry-hybrid-adjusted-5y-v2 \
    industry_hybrid \
    --industry-release data/standard/industry-release=sw_industry_pit_20260717_v3 \
    --industry-quality-report \
    artifacts/data_quality/industry_pit/sw_industry_pit_20260717_v3/20210719_20260717/quality.json
  printf '%s\tCOMPLETE\tall\n' "$(date --iso-8601=seconds)" >"${status_file}"
}

run_worker_from_divergence() {
  run_evaluation technical_classic_divergence_v1 \
    l2-v2-technical-divergence-adjusted-5y-v2 raw
  run_evaluation china_7000_rules_controlled_v1 \
    l2-v2-china-rules-controlled-adjusted-5y-v3 raw
  AQUANT_CONVERGENCE_ROOT=/mnt/d/AQuantEvaluation/convergence \
    run_evaluation l2_v2_executable l2-v2-all-size-neutral-adjusted-5y-v2 size
  AQUANT_CONVERGENCE_ROOT=/mnt/d/AQuantEvaluation/convergence \
    run_evaluation l2_v2_executable l2-v2-all-industry-hybrid-adjusted-5y-v2 \
    industry_hybrid \
    --industry-release data/standard/industry-release=sw_industry_pit_20260717_v3 \
    --industry-quality-report \
    artifacts/data_quality/industry_pit/sw_industry_pit_20260717_v3/20210719_20260717/quality.json
  printf '%s\tCOMPLETE\tall\n' "$(date --iso-8601=seconds)" >"${status_file}"
}

run_worker_from_size() {
  mkdir -p /mnt/d/AQuantEvaluation/convergence
  AQUANT_CONVERGENCE_ROOT=/mnt/d/AQuantEvaluation/convergence \
    run_evaluation l2_v2_executable l2-v2-all-size-neutral-adjusted-5y-v2 size
  AQUANT_CONVERGENCE_ROOT=/mnt/d/AQuantEvaluation/convergence \
    run_evaluation l2_v2_executable l2-v2-all-industry-hybrid-adjusted-5y-v2 \
    industry_hybrid \
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

if [[ "${1:-}" == "--worker-from-divergence" ]]; then
  trap 'printf "%s\tFAILED\tline=%s\n" "$(date --iso-8601=seconds)" "$LINENO" >"${status_file}"' ERR
  run_worker_from_divergence
  exit 0
fi

if [[ "${1:-}" == "--worker-from-size" ]]; then
  trap 'printf "%s\tFAILED\tline=%s\n" "$(date --iso-8601=seconds)" "$LINENO" >"${status_file}"' ERR
  run_worker_from_size
  exit 0
fi

if [[ "${1:-}" == "--from-divergence" ]]; then
  nohup bash scripts/launch_complete_l2_oos.sh --worker-from-divergence >"${log_file}" 2>&1 &
  echo "$!" >"${pid_file}"
  echo "started pid $(cat "${pid_file}") log ${log_file} code ${code_version}"
  exit 0
fi

if [[ "${1:-}" == "--from-size" ]]; then
  nohup bash scripts/launch_complete_l2_oos.sh --worker-from-size >"${log_file}" 2>&1 &
  echo "$!" >"${pid_file}"
  echo "started pid $(cat "${pid_file}") log ${log_file} code ${code_version}"
  exit 0
fi

nohup bash scripts/launch_complete_l2_oos.sh --worker >"${log_file}" 2>&1 &
echo "$!" >"${pid_file}"
echo "started pid $(cat "${pid_file}") log ${log_file} code ${code_version}"
