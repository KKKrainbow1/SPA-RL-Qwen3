# Architecture

## Overview

The SPA-RL pipeline is six stages, each a separate process. Stages 1-4 run **once per dataset/model pair**; stages 5-6 run repeatedly during hyperparameter search.

```
Stage 1: SFT             ─►  ckpt/qwen3_webshop_sft_loramerged
Stage 2: Exploration     ─►  exploration_outputs/explore/*.json
Stage 3: PRM training    ─►  ckpt/qwen3_webshop_prm/
Stage 4: PRM inference   ─►  prm/exploration_inference_results_webshop.json
Stage 5: Step-wise PPO   ─►  ckpt/qwen3_webshop_rl_loramerged
Stage 6: Eval            ─►  results/qwen3_8b_*_summary.json
```

Each stage's output is the next's input. There's no online feedback loop in stage 5 — PPO trains on **frozen** rollout data (this is offline RL).

## Component breakdown

### Stage 1: SFT (`upstream/sft/`)

- **Backbone:** Qwen3-8B-Instruct
- **Method:** LoRA r=8, α=16, target modules = `[q_proj, k_proj, v_proj, o_proj]`
- **Data:** 1624 expert trajectories from ETO project (467 human + 1157 GPT-4)
- **Loss:** Standard next-token CE on assistant turns only
- **Output:** LoRA-merged checkpoint
- **Runtime (4×A100 80G):** ~3-4 h

### Stage 2: Exploration (`extensions/vllm_async_rollout/`)

- **Goal:** Collect 78-1500 trajectories from the SFT agent in WebShop
- **Method:** AsyncLLMEngine + asyncio.gather + prefix caching
- **Output:** JSONs containing full conversations + final environment rewards
- **Why async:** sync fastchat takes ~30 min, async takes ~5 min — matters when iterating on hyperparameters
- **Runtime (1×A100 80G, 78 traj):** ~5 min

### Stage 3: PRM training (`upstream/prm/`)

- **Architecture:** SFT-checkpoint backbone + `nn.Linear(vocab_size, 1)` head
- **Loss:** `MSE(Σ turn_values, agent_final_reward)` — completely unsupervised at the step level
- **Why a Linear over `vocab_size` not `hidden_size`:** lets the head exploit the learned next-token distribution as a progress signal; ~150K params vs ~3K, but no measurable speed difference
- **Runtime:** ~30-60 min on 4×A100

### Stage 4: PRM inference

- **Goal:** Annotate each trajectory's assistant turns with a per-turn progress score
- **Method:** Same model as stage 3, but in inference mode — yields N scalar scores per N-turn trajectory
- **Output:** `exploration_inference_results_webshop.json` (input to stage 5)
- **Runtime:** ~5-10 min

### Stage 5: Step-wise PPO (`upstream/ppo/step_ppo.py` + `step_ppotrainer.py`)

- **Trainer:** `StepPPOTrainer extends trl.PPOTrainer`
- **Key custom logic:**
  - `compute_rewards`: injects PRM step rewards at each assistant turn's last token, instead of TRL's default "all reward at the very end"
  - `frag_mask`: 0/1 mask marking which tokens are assistant outputs vs user observations — used by GAE to skip non-trainable tokens
  - Everything else (GAE, clip, KL) inherited unchanged from TRL
- **Distributed:** Accelerate + DeepSpeed ZeRO-2, optimizer offload to CPU
- **KL guard:** [`extensions/kl_monitoring/`](../extensions/kl_monitoring/) auto-halts on `clipfrac > 0.5` or `kl_coef > 2.0`
- **Runtime:** ~6-8 h on 4×A100 80G for ~200 PPO steps

### Stage 6: Eval

Two paths:

- **Upstream eval** (`upstream/eval/llama3_2_3b_eval_webshop.sh`): runs the trained checkpoint via FastChat on 200-task test split, parses `Action: search[...]` text from outputs.
- **Tool-call eval** (`extensions/tool_call_eval/`): same task list but uses Qwen3's native `<tool_call>` format. Better for benchmarking against modern function-calling models.

## Data flow during one PPO step

```
1. Dataloader yields one batch of trajectories (offline-precomputed):
       queries[i], responses[i], rewards[i] (per-step), frag_masks[i]

2. With torch.no_grad():
   2a. Forward through actor model → logprobs, values
   2b. Forward through reference model → ref_logprobs
   2c. compute_rewards(scores, logprobs, ref_logprobs, masks)
       → builds reward tensor:
         reward[t] = -kl_coef * KL[t]                (every token)
         reward[step_end] += step_score              (only at step boundaries)
   2d. compute_advantages(values, rewards, masks)
       → manual reverse GAE loop, returns advantages + returns
       → both detach()'d to be treated as constants below

3. PPO inner loop (n_epochs × n_minibatches):
   3a. Forward actor again (with grad) → new_logprobs, new_values
   3b. ratio = exp(new_logprobs - old_logprobs)
   3c. policy_loss = -A * clip(ratio, 1-ε, 1+ε)
   3d. value_loss = MSE(new_values, returns)
   3e. backward + optimizer.step

4. Update AdaptiveKLController based on observed KL.
   Log stats; KLEarlyStopCallback checks for halt conditions.
```

## Why offline (vs online)

The original SPA codebase pre-collects all rollouts before PPO begins (stage 2 → 5 is one-shot). This is **off-policy** — rollout is from θ_SFT, training is updating θ_t — but works because:

- PPO's ratio clip + KL penalty bound the off-policy bias.
- Step rewards make the per-trajectory variance much smaller than typical RLHF.
- Training horizon is short (~200 steps), so policy drift is bounded.

For longer training or multi-turn agentic settings, an online setup (e.g. verl's HybridFlow) would be preferable. See [Future Work](../README.md#future-work).
