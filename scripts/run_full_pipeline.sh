#!/usr/bin/env bash
#
# End-to-end SPA-RL pipeline on Qwen3-8B.
# Stages: SFT -> Exploration (async) -> PRM train -> PRM inference -> PPO -> Eval
#
# Expected runtime on 4×A100 80G: ~12 hours total.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${REPO_ROOT}/upstream"
CKPT_DIR="${REPO_ROOT}/ckpt"

mkdir -p "${CKPT_DIR}"

export PYTHONPATH="${UPSTREAM}:${REPO_ROOT}:${PYTHONPATH:-}"
export MODEL_BASE="${MODEL_BASE:-Qwen/Qwen3-8B}"

# ---------- Stage 1: SFT ----------
echo "######################################################################"
echo "# Stage 1: SFT (Qwen3-8B + LoRA on 1624 expert trajectories)         #"
echo "######################################################################"
cd "${UPSTREAM}"
bash sft/webshop_qwen3.sh    # <-- you'll need to create this from the llama3b template
python sft/merge_lora.py     # produces ${CKPT_DIR}/qwen3_webshop_sft_loramerged

# ---------- Stage 2: Exploration (async) ----------
echo "######################################################################"
echo "# Stage 2: Async exploration (vLLM + prefix caching)                  #"
echo "######################################################################"
cd "${REPO_ROOT}"
python extensions/vllm_async_rollout/async_explore.py \
    --model_path "${CKPT_DIR}/qwen3_webshop_sft_loramerged" \
    --task_indices "${UPSTREAM}/eval_agent/data/webshop/train_indices.json" \
    --num_iterations 3 \
    --max_steps 10 \
    --concurrency 64 \
    --output_path "${UPSTREAM}/exploration/webshop/exploration_outputs/explore/"

# ---------- Stage 3: PRM training ----------
echo "######################################################################"
echo "# Stage 3: PRM training                                                #"
echo "######################################################################"
cd "${UPSTREAM}"
python prm/data_org.py
deepspeed --include=localhost:0,1,2,3 prm/train_our_progress_model_lora.py

# ---------- Stage 4: PRM inference ----------
echo "######################################################################"
echo "# Stage 4: PRM inference (offline step-reward annotation)             #"
echo "######################################################################"
cd "${UPSTREAM}"
python prm/inference_prm.py
python prm/rl_data_org.py

# ---------- Stage 5: Step-wise PPO ----------
echo "######################################################################"
echo "# Stage 5: Step-wise PPO (with KL monitoring)                         #"
echo "######################################################################"
cd "${UPSTREAM}"
accelerate launch --config_file "${REPO_ROOT}/configs/accelerate_qwen3_4gpu.yaml" \
    ppo/step_ppo.py \
    --config_path "${REPO_ROOT}/configs/ppo_qwen3.json" \
    --model_path "${CKPT_DIR}/qwen3_webshop_sft_loramerged" \
    --data_file "prm/sampled_data_rl_training_webshop_flatten.json" \
    --model_type qwen3 \
    --epochs 1

# Merge PPO LoRA back into base
python ppo/merge.py

# ---------- Stage 6: Evaluation ----------
echo "######################################################################"
echo "# Stage 6: WebShop test eval                                          #"
echo "######################################################################"
cd "${REPO_ROOT}"
bash scripts/run_fewshot_baseline.sh    # also runs the trained checkpoint

echo "==> Full pipeline complete."
