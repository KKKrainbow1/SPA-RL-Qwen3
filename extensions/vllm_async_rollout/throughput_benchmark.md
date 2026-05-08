# Throughput Benchmark: Sync vs Async Rollout

This document describes the methodology and expected results for the **6× speedup** claim in the project's main README.

## Setup

- **Hardware**: 1× NVIDIA A100 80GB
- **Model**: Qwen3-8B-Instruct (LoRA-merged SFT checkpoint, ~16 GB bf16)
- **Workload**: 78 trajectories on WebShop (`train_indices.json` first 26 tasks × 3 iterations)
- **Decoding**: temperature=0.7, top_p=0.9, max_new_tokens=512
- **Max steps per trajectory**: 10

## Method 1: Sync baseline (upstream behaviour)

Following the upstream `exploration/webshop/my_generate_response_webshop.sh`:

1. Start FastChat controller (`fastchat.serve.controller`).
2. Start a single FastChat vLLM worker (`fastchat.serve.vllm_worker`).
3. Run `generate_response_webshop.py`, which:
   - Iterates trajectories one at a time.
   - For each step, posts to FastChat's `/worker_generate_stream` endpoint.
   - Waits for the response, calls `env.step()`, then loops.

```bash
bash upstream/exploration/webshop/my_generate_response_webshop.sh
```

**Expected wall time**: ~30 minutes (~2.6 traj/min).

GPU utilisation typically hovers at 30-40% — most time is spent on:
- HTTP serialization/deserialization between FastChat and Python
- Single-request prefill/decode (no continuous batching benefit)
- Re-attending the 200-token system prompt on every step

## Method 2: Async rollout (this module)

```bash
python extensions/vllm_async_rollout/async_explore.py \
    --model_path ckpt/qwen3_webshop_sft_loramerged \
    --task_indices upstream/eval_agent/data/webshop/train_indices.json \
    --num_iterations 3 \
    --limit 26 \
    --max_steps 10 \
    --concurrency 64 \
    --output_path exploration_outputs/qwen3_async/
```

**Expected wall time**: ~5 minutes (~15 traj/min).

GPU utilisation peaks at 80-90% because:
- 64 trajectories are simultaneously in flight; vLLM's continuous batching packs whatever is ready.
- Prefix caching reuses the system prompt's KV across all trajectories.
- No HTTP overhead — engine is in-process.

## How the speedup decomposes

Approximate breakdown (your mileage will vary by ±20%):

| Source | Sync | Async | Improvement |
|---|---|---|---|
| Per-step generate latency | ~1.2 s | ~0.4 s | ~3× via continuous batching |
| System prompt prefill | ~250 ms × 10 steps × 78 traj | reused once | ~10× via prefix caching |
| HTTP / FastChat overhead | ~50 ms / step | 0 | direct engine access |
| Trajectory parallelism | sequential | up to 64 concurrent | 64× theoretical, ~6× realised |

The realised speedup is bounded by:
- WebShop env's single-process Python lock (Lucene Java bridge).
- Memory pressure on KV cache when concurrency × longest_trajectory > available cache.
- Diminishing returns past ~32 concurrent trajectories on A100 80G.

## Reproducing the benchmark

```bash
# After setup, run both pipelines back-to-back:
bash scripts/run_throughput_benchmark.sh
```

This script runs the same 78-trajectory workload twice (sync + async), records wall times, and writes results to `results/throughput_comparison.md`.

## Why not just increase batch size in the sync version?

You can't, easily. FastChat's `/worker_generate_stream` is request-response: one HTTP call → one completion. You'd need to manually batch multiple trajectories' next-step prompts at the application layer, then re-route responses back to the right env — which is essentially what the async version does in ~150 lines.

## Caveats

- Wall times depend heavily on which 26 train tasks you select; some have ~5-step completions, others fail at max_steps. Use the provided fixed indices for fair comparison.
- WebShop's Java Lucene backend serializes search calls; with concurrency > 64, env step latency starts to dominate.
- A100 80G specifically — H100 will go faster (CUDA graphs benefit more), 24G cards will need lower concurrency.
