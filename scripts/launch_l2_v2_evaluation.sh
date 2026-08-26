#!/usr/bin/env bash
set -euo pipefail

cd /home/gsx1339/aquant
mkdir -p artifacts/logs artifacts/convergence

pid_file="artifacts/logs/l2-v2-technical-level-evaluation.pid"
log_file="artifacts/logs/l2-v2-technical-level-evaluation.log"
if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "evaluation already running with pid $(cat "${pid_file}")"
  exit 0
fi

code_version="$(git rev-parse HEAD)"
nohup /usr/bin/time -v .venv/bin/python scripts/evaluate_factors.py \
  --release-dir data/standard/history-release=cn_equity_history_20260717_001 \
  --report-dir reports/l2-v2-technical-level-5y-v1 \
  --data-release-id cn_equity_20260717_001 \
  --factor-set technical_classic_level_v1 \
  --universe all_a_share \
  --start-date 20210718 \
  --end-date 20260717 \
  --horizons 1,5,10,20,40 \
  --cost-bps 10 \
  --batch-size 8 \
  --oos-fraction 0.2 \
  --code-version "${code_version}" \
  --convergence-cache artifacts/convergence/l2-v2-technical-level-5y-v1 \
  --reload-per-batch \
  --resume \
  > "${log_file}" 2>&1 &
echo "$!" > "${pid_file}"
echo "started pid $(cat "${pid_file}") log ${log_file} code ${code_version}"
