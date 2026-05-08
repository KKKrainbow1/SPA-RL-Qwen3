# vLLM Async Rollout

Replaces the original SPA exploration phase (sync fastchat HTTP, single-trajectory-at-a-time) with an asyncio-based concurrent rollout that drives a single vLLM `AsyncLLMEngine` with prefix caching enabled. Empirically yields **~6× throughput** for the WebShop exploration workload.

## Why this matters

The SPA pipeline collects 78–1500 multi-turn trajectories per training run. With the upstream sync setup (`exploration/webshop/my_generate_response_webshop.sh`):

- Trajectories run **sequentially** — one HTTP request blocks the worker.
- vLLM's continuous batching is **wasted** because there's only ever one in-flight request.
- The shared 737-character WebShop system prompt is **re-tokenized and re-attended** on every step.

For 78 trajectories × ~10 steps × ~500 prompt tokens, this regularly takes 30+ minutes on a single A100 80G.

## How this module fixes it

| Optimisation | Effect |
|---|---|
| `AsyncLLMEngine` + `asyncio.gather()` over all trajectories | vLLM's continuous batching now sees N concurrent in-flight requests → near-100% GPU utilisation |
| `enable_prefix_caching=True` | The shared system prompt's KV cache is computed once and reused across all trajectories and all steps |
| `enforce_eager=False` (CUDA graphs on) | Per-step generation latency drops |
| Per-trajectory async sub-loop | Each trajectory is still serial within itself (must await env.step), but trajectories don't block each other |

## Files

| File | Purpose |
|---|---|
| `async_explore.py` | Main async exploration script. Drop-in replacement for `exploration/webshop/generate_response_webshop.py`. |
| `prefix_cache_demo.py` | Standalone demo showing prefix caching speedup on a fixed system prompt. |
| `throughput_benchmark.md` | Methodology for the 6× speedup claim (sync vs async, raw timings, GPU utilisation). |

## Usage

```bash
# Start nothing — vLLM is embedded in the script (not a separate server)
python extensions/vllm_async_rollout/async_explore.py \
    --model_path ckpt/qwen3_webshop_sft_loramerged \
    --task_indices upstream/eval_agent/data/webshop/train_indices.json \
    --num_iterations 3 \
    --max_steps 10 \
    --concurrency 64 \
    --output_path exploration_outputs/qwen3_async/
```

Key flags:

- `--concurrency 64` — number of in-flight trajectories. 64 is a good default for A100 80G; raise to 128 for H100, lower to 16 for 24G cards.
- `--prefix_cache true` — keeps shared system prompt KV warm. Only disable for debugging.
- `--enable_thinking false` — Qwen3 thinking mode off (see `extensions/qwen3_adapter/`).

## Design notes

### Why async, not multi-process?

Multi-process workers (one vLLM per process) are simpler but waste GPU memory — each worker holds its own copy of the model weights. With AsyncLLMEngine + asyncio, all trajectories share one set of weights and one KV cache pool.

### Why not full multi-turn agent loop in vLLM?

vLLM has no native concept of "wait for environment, then continue". The async script does this in pure Python: each coroutine awaits `env.step()` between generations. This works because WebShop env is in-process (Python Flask + Lucene), so `env.step()` is a fast local call (<10ms).

For environments with network latency, a true async tool-call loop (e.g. verl's AgentLoop) would be needed.

### Determinism

Even with `temperature=0`, async batching can introduce tiny numerical noise (different batch composition → different float32 sum order). This is acceptable for exploration (we want diversity anyway) but not for evaluation. Use sync mode for eval.

## Limitations

- Single-GPU only. Multi-GPU async rollout requires `tensor_parallel_size > 1`, which works but offers diminishing returns on small models like 8B.
- WebShop env is single-process (Lucene Java backend) — at very high concurrency (>128), env step latency starts to bottleneck.
- vLLM 0.6+ required for `enable_prefix_caching` + `AsyncLLMEngine` stable APIs.
