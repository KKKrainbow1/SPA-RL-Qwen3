# Ablation Table — WebShop test 200 split

> Numbers populated as experiments complete. Empty rows are placeholders.

**Test setup:** WebShop `test_indices.json` (200 tasks), max_steps=10, greedy decoding (temperature=0), Qwen3-8B base.

## Main results

| # | Method | Avg Reward | Success Rate | Δ vs few-shot | Notes |
|---|---|---|---|---|---|
| 1 | Qwen3-8B zero-shot (tool_call) | _TBD_ | _TBD_ | — | No training, 0-shot |
| 2 | **Qwen3-8B 1-shot (tool_call)** | **0.644** | 0.975 (loose) | — | One in-context demo. 0 tool_call errors over 200 tasks. |
| 3 | Qwen3-8B + SFT (LoRA r=8) | _TBD_ | _TBD_ | _TBD_ | 1624 expert traj |
| 4 | Qwen3-8B + PPO baseline | _TBD_ | _TBD_ | _TBD_ | Sparse final reward |
| 5 | **Qwen3-8B + SPA (ours)** | **_TBD_** | **_TBD_** | **target +~3 pp** | PRM + grounding + KL guard |

**On the success-rate column:** WebShop's env flips `success=True` whenever the agent terminates with a Buy Now click, regardless of whether the right product was bought. `avg_reward` is the metric to compare across rows; `success_rate` is mostly format-conformance. For row 2, 29.0% of tasks earned reward = 1.0 (truly perfect) and 32.5% earned ≥ 0.8.

## Ablations on top of #5

| # | Variant | Avg Reward | Δ vs full SPA | What's removed |
|---|---|---|---|---|
| 5a | SPA without grounding signal | _TBD_ | _TBD_ | grounding `+0.5` removed |
| 5b | SPA with raw PRM (no KL guard) | _TBD_ | _TBD_ | KL early-stop disabled |
| 5c | SPA with sync rollout | _TBD_ | _TBD_ | uses upstream sync exploration (slower, same algo) |

## Reference values (from SPA paper, Llama-series)

| Method | Llama-3.2-3B | Llama-3.1-8B |
|---|---|---|
| SFT | ~58-60% | ~62% |
| PPO | 60.7% | ~64% |
| SPA | 63.7% | ~67% |

The Qwen3-8B numbers should land at or above the Llama-3.1-8B row — Qwen3 has stronger function-calling and instruction-following baselines.

## Throughput (separate axis)

See [`throughput_comparison.md`](throughput_comparison.md) for wall-time and traj/min.
