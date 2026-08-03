#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/gsx1339/aquant

pool="artifacts/l3_nested_walk_forward/fold_pools.json"
report_dir="reports/l3-nested-union-size-2016-2026-v1"
cache_dir="/mnt/d/AQuantEvaluation/convergence/l3-nested-union-size-2016-2026-v1"
pid_file="artifacts/logs/l3-nested-union.pid"
log_file="artifacts/logs/l3-nested-union.log"
status_file="artifacts/logs/l3-nested-union.status"

mkdir -p artifacts/logs "${report_dir}"

if [[ ! -f "${pool}" ]]; then
  echo "nested L2 pool artifact is missing: ${pool}" >&2
  exit 2
fi

cache_metadata="${cache_dir}/metadata.json"
if [[ ! -f "${cache_metadata}" ]]; then
  union_factor_count="$(.venv/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["union_factor_count"])' "${pool}")"
  estimated_cache_bytes="$((union_factor_count * 64 * 1024 * 1024))"
  safety_margin_bytes="$((5 * 1024 * 1024 * 1024))"
  available_bytes="$(df --output=avail -B1 /mnt/d | tail -n 1 | tr -d ' ')"
  required_bytes="$((estimated_cache_bytes + safety_margin_bytes))"
  if (( available_bytes < required_bytes )); then
    printf '%s\tWAITING_FOR_DISK\trequired_bytes=%s\tavailable_bytes=%s\n' \
      "$(date --iso-8601=seconds)" "${required_bytes}" "${available_bytes}" \
      >"${status_file}"
    echo "WAIT_DISK required_bytes=${required_bytes} available_bytes=${available_bytes}"
    exit 0
  fi
fi

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "nested union materialization already running with pid $(cat "${pid_file}")"
    exit 0
  fi
  nohup bash scripts/launch_nested_union_materialization.sh --worker >"${log_file}" 2>&1 &
  echo "$!" >"${pid_file}"
  echo "started pid $(cat "${pid_file}") log ${log_file}"
  exit 0
fi

trap 'printf "%s\tFAILED\tline=%s\n" "$(date --iso-8601=seconds)" "$LINENO" >"${status_file}"' ERR
printf '%s\tRUNNING\tl3-nested-union-size-2016-2026-v1\n' \
  "$(date --iso-8601=seconds)" >"${status_file}"

manifest="${report_dir}/evaluation_manifest.json"
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
  --factor-ids-file "${pool}" \
  --universe all_a_share \
  --start-date 20160104 \
  --end-date 20260717 \
  --horizons 5 \
  --cost-bps 10 \
  --batch-size 2 \
  --oos-fraction 0.2 \
  --code-version "${code_version}" \
  --reload-per-batch \
  --resume \
  --neutralization size \
  --convergence-cache "${cache_dir}" \
  --report-dir "${report_dir}"

printf '%s\tPASS\tl3-nested-union-size-2016-2026-v1\n' \
  "$(date --iso-8601=seconds)" >"${status_file}"
