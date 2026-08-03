#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/gsx1339/aquant
mkdir -p artifacts/logs reports/l3-nested-prehistory-size-2016-2021-v1

pid_file="artifacts/logs/l3-nested-prehistory.pid"
log_file="artifacts/logs/l3-nested-prehistory.log"
status_file="artifacts/logs/l3-nested-prehistory.status"

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "nested prehistory evaluation already running with pid $(cat "${pid_file}")"
    exit 0
  fi
  nohup bash scripts/launch_nested_wf_pretraining.sh --worker >"${log_file}" 2>&1 &
  echo "$!" >"${pid_file}"
  echo "started pid $(cat "${pid_file}") log ${log_file}"
  exit 0
fi

trap 'printf "%s\tFAILED\tline=%s\n" "$(date --iso-8601=seconds)" "$LINENO" >"${status_file}"' ERR
printf '%s\tRUNNING\tl3-nested-prehistory-size-2016-2021-v1\n' \
  "$(date --iso-8601=seconds)" >"${status_file}"

manifest="reports/l3-nested-prehistory-size-2016-2021-v1/evaluation_manifest.json"
if [[ -f "${manifest}" ]]; then
  code_version="$('./.venv/bin/python' -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["code_version"])' "${manifest}")"
else
  code_version="$(git rev-parse HEAD)"
fi

/usr/bin/time -v .venv/bin/python scripts/evaluate_factors.py \
  --release-dir data/standard/history-release=cn_equity_history_20260717_001 \
  --data-release-id cn_equity_20260717_001 \
  --factor-set l2_v2_executable \
  --universe all_a_share \
  --start-date 20160104 \
  --end-date 20210716 \
  --horizons 5 \
  --cost-bps 10 \
  --batch-size 2 \
  --oos-fraction 0.2 \
  --code-version "${code_version}" \
  --reload-per-batch \
  --resume \
  --neutralization size \
  --report-dir reports/l3-nested-prehistory-size-2016-2021-v1

printf '%s\tPASS\tl3-nested-prehistory-size-2016-2021-v1\n' \
  "$(date --iso-8601=seconds)" >"${status_file}"
