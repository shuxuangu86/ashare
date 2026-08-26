#!/usr/bin/env bash
set -euo pipefail

cd /home/gsx1339/aquant
mkdir -p artifacts/convergence artifacts/logs

pid_file="artifacts/logs/academic-extension-evaluation.pid"
log_file="artifacts/logs/academic-extension-evaluation.log"
if [[ "${1:-}" != "--worker" && -f "${pid_file}" ]] \
  && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "evaluation already running with pid $(cat "${pid_file}")"
  exit 0
fi

code_version="$(git rev-parse HEAD)"
run_pack() {
  local factor_set="$1"
  local report_name="$2"
  /usr/bin/time -v .venv/bin/python scripts/evaluate_factors.py \
    --release-dir data/standard/history-release=cn_equity_history_20260717_001 \
    --report-dir "reports/${report_name}" \
    --data-release-id cn_equity_20260717_001 \
    --factor-set "${factor_set}" \
    --universe all_a_share \
    --start-date 20210718 \
    --end-date 20260717 \
    --horizons 1,5,10,20,40 \
    --cost-bps 10 \
    --batch-size 4 \
    --oos-fraction 0.2 \
    --code-version "${code_version}" \
    --convergence-cache "artifacts/convergence/${report_name}" \
    --reload-per-batch \
    --resume
}

if [[ "${1:-}" == "--worker" ]]; then
  run_pack han_yang_zhou_2013_v1 l2-v2-han-yang-zhou-5y-v1
  run_pack technical_sentiment_2023_v1 l2-v2-technical-sentiment-5y-v1
  run_pack fama_french_2015_partial_v1 l2-v2-fama-french-partial-5y-v1
  run_pack china_7000_rules_controlled_v1 l2-v2-china-rules-controlled-5y-v1
  exit 0
fi

nohup bash scripts/launch_academic_extension_evaluation.sh --worker \
  >"${log_file}" 2>&1 &
echo "$!" >"${pid_file}"
echo "started pid $(cat "${pid_file}") log ${log_file} code ${code_version}"
