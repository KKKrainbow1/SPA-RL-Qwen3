#!/usr/bin/env bash
#
# Few-shot evaluation of Qwen3-8B on WebShop 200-task test split using
# OpenAI-style tool calls.
#
# Expected runtime: ~20 min on 1×A100 80G
# Expected cost:    ~¥10 on AutoDL
#
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qwen3_8b_fewshot}"
VLLM_PORT="${VLLM_PORT:-8000}"
K_SHOT="${K_SHOT:-1}"

mkdir -p "${OUTPUT_DIR}/logs"

VLLM_PID=""
cleanup() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "==> Tearing down vLLM (PID ${VLLM_PID})"
        kill "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..10}; do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "==> Starting vLLM OpenAI-compatible server (model=${MODEL_PATH})"
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 8192 \
    --dtype bfloat16 \
    > "${OUTPUT_DIR}/logs/vllm.log" 2>&1 &

VLLM_PID=$!
echo "    vLLM PID: ${VLLM_PID}"
echo "    Logs:    ${OUTPUT_DIR}/logs/vllm.log"

# Wait for the server to be ready
echo "==> Waiting for vLLM (up to 300s)..."
for i in {1..60}; do
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "    vLLM ready after ~$((i*5))s"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM exited during startup. Tail of log:"
        tail -40 "${OUTPUT_DIR}/logs/vllm.log" || true
        exit 1
    fi
    sleep 5
    if [[ $i -eq 60 ]]; then
        echo "ERROR: vLLM did not start within 300s"
        exit 1
    fi
done

echo "==> Running few-shot eval (k_shot=${K_SHOT})"
python extensions/tool_call_eval/fewshot_eval.py \
    --model_name "${MODEL_PATH}" \
    --vllm_port "${VLLM_PORT}" \
    --k_shot "${K_SHOT}" \
    --output_dir "${OUTPUT_DIR}"

echo "==> Done. Summary:"
python -m json.tool "${OUTPUT_DIR}/summary.json" | head -20
