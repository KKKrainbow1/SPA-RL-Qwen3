#!/usr/bin/env bash
#
# Few-shot baseline of Qwen3-8B on WebShop test 200 split.
# Wraps extensions/tool_call_eval/run_qwen3_fewshot.sh with sane defaults.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/qwen3_8b_fewshot}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export K_SHOT="${K_SHOT:-1}"
export PYTHONPATH="${REPO_ROOT}/upstream:${REPO_ROOT}:${PYTHONPATH:-}"

# pyserini's _openai.py instantiates openai.OpenAI() at module import; on
# newer openai SDK this raises if OPENAI_API_KEY is unset, even though we
# never use the OpenAI encoder. Provide a placeholder to bypass — our
# vLLM client passes api_key="EMPTY" explicitly so this does not leak.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-pyserini-bypass-not-used}"

bash extensions/tool_call_eval/run_qwen3_fewshot.sh
