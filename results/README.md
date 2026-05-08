# Results

This directory holds reproducible experiment outputs — JSON summaries, training curves, ablation tables. **All numbers cited in the top-level README are reproducible from these files.**

## Layout

```
results/
├── README.md                          ← you are here
├── qwen3_8b_fewshot_summary.json      ← few-shot baseline (200 tasks, no training)
├── qwen3_8b_sft_summary.json          ← SFT-only baseline
├── qwen3_8b_ppo_summary.json          ← PPO baseline (sparse final reward)
├── qwen3_8b_spa_summary.json          ← SPA full (this work's main result)
├── ablation_table.md                  ← rendered comparison table
├── throughput_comparison.md           ← sync vs async wall-time + throughput
├── training_curves/                   ← W&B plot exports
│   ├── kl_coef_vs_step.png
│   ├── reward_curve.png
│   ├── clipfrac_vs_step.png
│   └── prm_vs_kl_dominance.png
└── trajectory_samples/                ← 5-10 representative trajectories per setup
    ├── fewshot_success_001.json
    ├── fewshot_failure_001.json
    └── spa_success_001.json
```

## Schema of `*_summary.json`

```json
{
  "model": "Qwen/Qwen3-8B",
  "method": "spa_full",
  "n_tasks": 200,
  "k_shot": 0,
  "max_steps": 10,
  "decoding": "greedy",
  "avg_reward": 0.000,
  "success_rate": 0.000,
  "rewards": [...],         // per-task reward
  "successes": [...],       // per-task bool (reward >= 0.99)
  "training_steps": 200,    // PPO steps if applicable
  "wall_clock_minutes": 0,
  "git_commit": "abc1234"
}
```

## How to populate this directory

Files here are **outputs**, not committed inputs. Running the scripts in `scripts/` writes here:

| Script | Output |
|---|---|
| `scripts/run_fewshot_baseline.sh` | `qwen3_8b_fewshot_summary.json` |
| `scripts/run_full_pipeline.sh` | `qwen3_8b_sft_summary.json`, `qwen3_8b_ppo_summary.json`, `qwen3_8b_spa_summary.json` |
| `scripts/run_throughput_benchmark.sh` | `throughput_comparison.md`, `throughput_async.log` |

After a run completes, **commit only the summaries** (not the per-task trajectory JSONs — those are large and not cited in the README).

## Adding new results

If you run an additional ablation or experiment:

1. Save the summary as `qwen3_8b_<your_setup>_summary.json` matching the schema above.
2. Add a row to `ablation_table.md`.
3. Update the main README's Results table if the new number changes the headline story.
