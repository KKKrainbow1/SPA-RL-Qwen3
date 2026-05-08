# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project context

**SPA-RL-Qwen3** — reproduction and extension of the SPA-RL paper ([arXiv 2505.20732](https://arxiv.org/abs/2505.20732)) on Qwen3-8B for the WebShop benchmark. The full project rationale, architecture, and design decisions are in:

- `README.md` — project overview, results, setup
- `docs/architecture.md` — six-stage pipeline + per-PPO-step data flow
- `docs/kl_protection_mechanism.md` — dual-layer KL math
- `docs/stepppo_explained.md` — TRL StepPPOTrainer overrides

**Read these before making non-trivial changes.**

## What the user owns vs upstream

This repo only contains **extensions** to the original SPA codebase:

- `extensions/qwen3_adapter/` — Llama-3 → Qwen3 migration
- `extensions/vllm_async_rollout/` — async exploration with prefix caching
- `extensions/tool_call_eval/` — Qwen3 native tool-call eval
- `extensions/kl_monitoring/` — KL/clipfrac early-stop callback

The original SPA codebase (`upstream/`, cloned by `scripts/setup_data.sh`) is **not committed** to this repo and should not be modified directly. Patches into upstream are applied via `scripts/apply_qwen3_patches.sh`.

## Key design decisions (already locked in — don't second-guess)

1. **Offline PPO, not online.** Rollout policy is θ_SFT, training updates θ_t. PPO clip + adaptive KL keep drift bounded for ~200 steps. Migration to online (verl, GRPO) is in `Future Work`, not for this version.

2. **Tool-call format ≠ training format.** Eval uses Qwen3's native `<tool_call>` JSON; training data stays in upstream's `Action: search[...]` text form. The parser bridges them. **Do not rewrite training data into tool-call format** — that means retraining SFT/PRM/PPO from scratch.

3. **PRM at the same scale as the policy.** PRM is Qwen3-8B with `Linear(vocab_size, 1)` head, not a smaller model. The vocab-size linear (vs hidden_size) is intentional and follows the SPA paper.

4. **Thinking mode is disabled everywhere.** Qwen3's `<think>` block breaks SFT data assumptions and PRM turn-boundary detection. Always pass `enable_thinking=False`.

5. **DeepSpeed ZeRO-2, not FSDP.** Upstream and `configs/accelerate_qwen3_4gpu.yaml` use DeepSpeed; switching to FSDP is possible but means revalidating LoRA + ValueHead compatibility.

6. **KL penalty baked into reward, not loss.** This follows TRL's PPO convention. Don't refactor it into a separate loss term — that breaks the inherited GAE flow.

## Resume claim status (be honest about what's measured)

The user's resume cites:

- `65.5%` on WebShop test → **target / projected, not yet measured on Qwen3-8B**
- `+3 pp vs PPO` → reference from SPA paper's Llama-3 numbers
- `vLLM 6× throughput` → reproducible via `scripts/run_throughput_benchmark.sh`, hasn't been formally re-run on user's hardware yet

When the user asks about these numbers, refer them to `results/` (where real measured numbers should land). Don't invent ablation results — particularly avoid the previously-deleted "Reward Scale Alignment +1.5pp" claim.

## Dev workflow

- Local (this directory): write code, write docs, commit + push.
- Server (AutoDL or similar): `git pull`, run experiments, commit small `results/*.json` files, push back.
- Big files (model checkpoints, trajectory JSONs, vLLM logs) **never** go through git — see `.gitignore`.

## Commands the user runs frequently

```bash
# First-time server setup
bash scripts/setup_data.sh
bash scripts/apply_qwen3_patches.sh

# Few-shot baseline (~20 min, ~¥10 on AutoDL)
bash scripts/run_fewshot_baseline.sh

# Throughput benchmark
bash scripts/run_throughput_benchmark.sh

# Full pipeline (~12h on 4×A100 80G)
bash scripts/run_full_pipeline.sh
```

## When the user asks for new features

Common asks and the right scope:

- **"Add GRPO support"** → new `extensions/grpo/` directory, don't refactor StepPPO
- **"Try a different reward shaping"** → modify `prm/rl_data_org.py` patch in upstream, not the PPO trainer
- **"Add a new env"** → mirror `extensions/tool_call_eval/webshop_env_toolcall.py` for the new env
- **"Switch to FSDP"** → modify `configs/accelerate_qwen3_4gpu.yaml`, validate ValueHead compatibility

## Things to push back on

- Adding tool-call format to training data (huge work, no benefit — see decision #2 above)
- Trying to make PRM smaller than the policy (decision #3)
- Hardcoding vocab_size again (use `model.config.vocab_size`)
- Running PPO without KL monitoring (will silently stall — see `extensions/kl_monitoring/`)

## Memory hint

The user has accumulated context across many earlier sessions about:
- Why certain numbers were chosen as targets
- Resume strategy (what to claim, what to hedge)
- Trade-offs between TRL PPO / GRPO / verl

If the user references "what we discussed before", check the auto-memory directory first; otherwise ask them to summarize.
