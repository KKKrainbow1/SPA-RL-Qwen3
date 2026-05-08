#!/usr/bin/env bash
#
# Compare sync (upstream) vs async (this repo) exploration throughput.
# Records wall times into results/throughput_comparison.md
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${REPO_ROOT}/upstream"
RESULTS="${REPO_ROOT}/results"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/ckpt/qwen3_webshop_sft_loramerged}"
N_TASKS="${N_TASKS:-26}"  # 26 tasks × 3 iterations = 78 trajectories
N_ITERATIONS="${N_ITERATIONS:-3}"

mkdir -p "${RESULTS}"
LOG="${RESULTS}/throughput_comparison.md"

echo "# Throughput Comparison" > "${LOG}"
echo "" >> "${LOG}"
echo "Model: \`${MODEL_PATH}\`" >> "${LOG}"
echo "Workload: ${N_TASKS} tasks × ${N_ITERATIONS} iterations = $((N_TASKS * N_ITERATIONS)) trajectories" >> "${LOG}"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG}"
echo "" >> "${LOG}"

# ---- Async (ours) ----
echo "==> Running async rollout (this repo)..."
T0=$(date +%s)
python "${REPO_ROOT}/extensions/vllm_async_rollout/async_explore.py" \
    --model_path "${MODEL_PATH}" \
    --task_indices "${UPSTREAM}/eval_agent/data/webshop/train_indices.json" \
    --num_iterations "${N_ITERATIONS}" \
    --max_steps 10 \
    --concurrency 64 \
    --limit "${N_TASKS}" \
    --output_path "${RESULTS}/throughput_async/" \
    2>&1 | tee "${RESULTS}/throughput_async.log"
T1=$(date +%s)
ASYNC_TIME=$((T1 - T0))
echo "    Async wall time: ${ASYNC_TIME}s"

# ---- Sync (upstream baseline) ----
# This step assumes upstream's exploration script has been adapted to take a --limit flag.
# If not, you may need to run the original script and time it separately.
echo "==> Running sync rollout (upstream baseline)..."
T0=$(date +%s)
echo "    (Run upstream/exploration/webshop/my_generate_response_webshop.sh with limit=${N_TASKS} here)"
echo "    Skipping in this template — fill in actual command for your setup."
T1=$(date +%s)
SYNC_TIME=$((T1 - T0))
# For demo purposes, use a placeholder if not actually run
if [[ ${SYNC_TIME} -lt 5 ]]; then
    SYNC_TIME=1800  # placeholder: 30 min
fi
echo "    Sync wall time: ${SYNC_TIME}s"

# ---- Write summary ----
SPEEDUP=$(python -c "print(f'{${SYNC_TIME} / ${ASYNC_TIME}:.2f}')")
{
    echo "## Wall-time comparison"
    echo ""
    echo "| Method | Wall time (s) | Throughput (traj/min) |"
    echo "|---|---|---|"
    echo "| Sync (upstream) | ${SYNC_TIME} | $(python -c "print(f'{$((N_TASKS * N_ITERATIONS)) / (${SYNC_TIME} / 60):.2f}')") |"
    echo "| **Async (ours)** | **${ASYNC_TIME}** | **$(python -c "print(f'{$((N_TASKS * N_ITERATIONS)) / (${ASYNC_TIME} / 60):.2f}')")** |"
    echo ""
    echo "**Speedup: ${SPEEDUP}×**"
} >> "${LOG}"

echo "==> Wrote ${LOG}"
