# Extensions

Everything in this directory is a contribution **on top of** the original SPA-RL-Agent codebase. The upstream repo lives at `https://github.com/WangHanLinHenry/SPA-RL-Agent` and is brought in by `scripts/setup_data.sh` as `./upstream/`.

## Module overview

| Module | One-liner |
|---|---|
| [`qwen3_adapter/`](qwen3_adapter/) | Llama-3 → Qwen3-8B migration: chat template, PRM turn-boundary fix, FastChat adapter |
| [`vllm_async_rollout/`](vllm_async_rollout/) | AsyncLLMEngine-based exploration with prefix caching — ~6× throughput |
| [`tool_call_eval/`](tool_call_eval/) | OpenAI-style function-calling eval pipeline (Qwen3 native format → WebShop env) |
| [`kl_monitoring/`](kl_monitoring/) | Real-time KL/clipfrac monitoring + early-stop callback for offline PPO |

Each submodule has its own README explaining the design and how it integrates with upstream. Read those for technical details; this top-level page is just a map.

## Why a separate `extensions/` directory?

- **Clear attribution.** Upstream code is unchanged in this repo; everything novel is here. Easy for reviewers to see what's mine.
- **Surgical patching.** `scripts/apply_qwen3_patches.sh` copies/patches only the necessary files into upstream — the patches themselves are tracked here.
- **Reusability.** Modules don't depend on each other (except `tool_call_eval` references upstream's `eval_agent` package). You can lift any one into another project.

## Dependency graph

```
qwen3_adapter/  ─────► (no internal deps)
                       Used by all other modules + upstream.

vllm_async_rollout/ ──► uses qwen3_adapter.chat_template
                        Replaces upstream/exploration/

tool_call_eval/  ────► uses qwen3_adapter.chat_template
                       Wraps upstream/eval_agent/envs/WebShopEnv

kl_monitoring/  ─────► (no internal deps)
                       Hooks into upstream/ppo/step_ppo.py train loop
```
