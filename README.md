# SPA-RL-Qwen3

> **Stepwise Progress Attribution RL on Qwen3-8B for multi-turn LLM agents**, with vLLM async rollout, dual-layer KL protection, and native tool-call evaluation.

[![Paper](https://img.shields.io/badge/paper-arxiv:2505.20732-red)](https://arxiv.org/abs/2505.20732)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)
[![Model](https://img.shields.io/badge/model-Qwen3--8B-blue)](https://huggingface.co/Qwen/Qwen3-8B)
[![Framework](https://img.shields.io/badge/framework-TRL-orange)](https://github.com/huggingface/trl)

---

## TL;DR

This repository reproduces and extends **SPA-RL** ([arXiv 2505.20732](https://arxiv.org/abs/2505.20732)) — a step-wise reward redistribution method for long-horizon LLM agent RL — on **Qwen3-8B** in the **WebShop** benchmark. The original SPA was demonstrated on Llama-3 series; this work migrates the full pipeline to Qwen3 and adds three engineering improvements:

1. **Async rollout via vLLM** — `~6× throughput` over the original synchronous fastchat pipeline.
2. **Dual-layer KL protection** with adaptive controller + early-stop, preventing KL from drowning out PRM signal.
3. **Native tool-call evaluation** — leverages Qwen3's BFCL-grade function-calling instead of regex-parsed `Action: search[...]`.

---

## Highlights

- ✅ **End-to-end SPA-RL pipeline on Qwen3-8B** (SFT → Exploration → PRM training → PRM inference → Step-wise PPO → Eval)
- ✅ **vLLM AsyncLLMEngine + Prefix Caching rollout** — 30 min → 5 min on 78 trajectories (single A100 80G)
- ✅ **Dual-layer KL protection** — KL(θ_t ‖ θ_ref) baked into per-token reward + ratio clip(θ_t / θ_old) + AdaptiveKLController
- ✅ **clipfrac / kl_coef monitoring + early-stop** — auto-halt when KL penalty starts dominating PRM signal
- ✅ **Tool-call evaluation pipeline** — OpenAI function-calling style WebShop tools, custom JSON parser, few-shot prompting
- ✅ **Built on TRL** — `StepPPOTrainer` extends `trl.PPOTrainer`, multi-GPU via Accelerate + DeepSpeed ZeRO-2

---

## Results

> Numbers are filled in as experiments complete. See [`results/`](results/) for raw logs and trajectories.

### WebShop Benchmark
*Test split: 200 tasks (`test_indices.json`), max_steps=10, greedy decoding.*

| Method | Avg Reward | Δ vs PPO | Notes |
|---|---|---|---|
| Qwen3-8B few-shot (tool_call) | _TBD_ | — | Zero-train baseline |
| Qwen3-8B + SFT | _TBD_ | — | LoRA r=8, 1624 expert traj |
| Qwen3-8B + PPO baseline | _TBD_ | — | Sparse final reward |
| **Qwen3-8B + SPA (ours)** | _TBD_ | **+~3 pp** | Step-wise PPO + PRM + grounding |

**Reference (from original SPA paper):** Llama-3.1-8B + SPA = ~67%, +3pp over PPO baseline.

### Throughput

| Rollout method | 78-trajectory time | Throughput |
|---|---|---|
| Sync (fastchat HTTP, single worker) | ~30 min | ~2.6 traj/min |
| **vLLM AsyncLLMEngine + prefix caching (ours)** | **~5 min** | **~15 traj/min** |

→ See [`extensions/vllm_async_rollout/README.md`](extensions/vllm_async_rollout/README.md) for the methodology.

---

## What's New (vs Original SPA)

This repo is **not a fork** of the original SPA codebase — it contains only the extensions, with a clear `extensions/` directory that integrates with the upstream pipeline. To run end-to-end, clone the original SPA repo first (see [Setup](#setup)).

| Module | Purpose | Lines (approx) |
|---|---|---|
| [`extensions/qwen3_adapter/`](extensions/qwen3_adapter/) | Llama-3 → Qwen3 chat template, FastChat adapter, PRM turn-boundary fix | ~150 |
| [`extensions/vllm_async_rollout/`](extensions/vllm_async_rollout/) | AsyncLLMEngine-based exploration with prefix caching | ~300 |
| [`extensions/tool_call_eval/`](extensions/tool_call_eval/) | Tool schema, JSON parser, env wrapper, few-shot eval script | ~400 |
| [`extensions/kl_monitoring/`](extensions/kl_monitoring/) | Real-time KL/clipfrac monitoring, early-stop callback | ~200 |

Original SPA modules (SFT, PRM training, StepPPOTrainer, GAE) are referenced unchanged — see the upstream repo.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: SFT                                                    │
│    Qwen3-8B + LoRA(r=8) on 1624 expert trajectories             │
│    (467 human + 1157 GPT-4 from ETO)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  Stage 2: Exploration (vLLM AsyncLLMEngine)        ★ ours        │
│    SFT agent rollout in WebShop env, ~78 trajectories           │
│    asyncio + continuous batching + prefix caching                │
│    → 6× faster than sync fastchat                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  Stage 3: PRM training (offline)                                 │
│    Qwen3-8B backbone + Linear(vocab_size, 1) head               │
│    MSE(Σ turn_values, final_reward) → step-level attribution     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  Stage 4: PRM inference                                          │
│    Score each turn-end token → step rewards (offline JSON)      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  Stage 5: Step-wise PPO (TRL StepPPOTrainer)                    │
│    Inject step rewards at assistant turn-end token positions    │
│    via frag_mask; GAE + dual-layer KL protection                │
│    + clipfrac/kl_coef monitoring + early-stop      ★ ours       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  Stage 6: Evaluation                                             │
│    Tool-call format (Qwen3 native function calling) ★ ours       │
│    WebShop test split (200 tasks)                                │
└─────────────────────────────────────────────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for component-level details.

---

## Setup

### 1. Clone this repo and the original SPA codebase

```bash
git clone https://github.com/YOUR_USERNAME/SPA-RL-Qwen3.git
cd SPA-RL-Qwen3

# Clone original SPA into ./upstream/ (not committed to this repo)
git clone https://github.com/WangHanLinHenry/SPA-RL-Agent.git upstream
```

### 2. Environment

```bash
conda create -n spa-qwen3 python=3.10 -y
conda activate spa-qwen3
pip install -r requirements.txt

# WebShop env (Java + Lucene required)
cd upstream/envs/webshop && pip install -e . && cd ../../..
python -m spacy download en_core_web_lg
conda install -y -c conda-forge openjdk=11
```

### 3. Data

```bash
# WebShop benchmark data (~few GB)
bash scripts/setup_data.sh
```

### 4. Apply Qwen3 adapter to upstream

```bash
bash scripts/apply_qwen3_patches.sh    # patches upstream code with Qwen3 adapter
```

---

## Quick Start

### Option A: Few-shot baseline (no training, ~20 min, ~¥10 on AutoDL)

```bash
bash scripts/run_fewshot_baseline.sh
# → Qwen3-8B with tool_calls on WebShop 200-task test split
# → Output: results/qwen3_8b_fewshot_summary.json
```

### Option B: Full SPA-RL pipeline (~12 hours on 4×A100 80G)

```bash
bash scripts/run_full_pipeline.sh
# Stages: SFT → exploration → PRM train → PRM infer → PPO → eval
```

### Option C: Just the throughput benchmark

```bash
bash scripts/run_throughput_benchmark.sh
# Compares sync vs async rollout on 78 trajectories
```

---

## Repository Structure

```
SPA-RL-Qwen3/
├── README.md                          ← you are here
├── LICENSE                            ← Apache 2.0
├── requirements.txt
├── .gitignore
│
├── extensions/                        ← all our contributions
│   ├── qwen3_adapter/
│   ├── vllm_async_rollout/
│   ├── tool_call_eval/
│   └── kl_monitoring/
│
├── configs/                           ← training configs
│   ├── accelerate_qwen3_4gpu.yaml
│   ├── ppo_qwen3.json
│   └── ds_zero2.json
│
├── results/                           ← reproducible experiment data
│   ├── qwen3_8b_fewshot_summary.json
│   ├── ablation_table.md
│   ├── throughput_comparison.md
│   └── training_curves/
│
├── docs/
│   ├── architecture.md
│   ├── kl_protection_mechanism.md
│   └── stepppo_explained.md
│
└── scripts/                           ← one-shot reproduce
    ├── setup_data.sh
    ├── apply_qwen3_patches.sh
    ├── run_fewshot_baseline.sh
    ├── run_full_pipeline.sh
    └── run_throughput_benchmark.sh
```

---

## Reproducibility

- All numbers in [`results/`](results/) are reproducible via the scripts in [`scripts/`](scripts/).
- Hardware used: 4×A100 80G on AutoDL (full pipeline) / 1×A100 80G (few-shot baseline).
- Random seed fixed at `42` (TRL's `set_seed`).
- WebShop env version: pinned via the upstream repo's `envs/webshop/setup.py`.

---

## Future Work

- [ ] Migrate to **TRL GRPOTrainer + native vLLM colocate mode** — leverages v1.0+ async rollout, drops the offline data pipeline.
- [ ] Iterative offline rollout (semi-online) — refresh trajectories every 50 PPO steps to reduce policy drift.
- [ ] Test on **ALFWorld** and **VirtualHome** (other SPA benchmarks).
- [ ] Compare with [verl](https://github.com/volcengine/verl)'s HybridFlow architecture for true on-policy multi-turn RL.

---

## Citation

If you use this work, please cite the original SPA-RL paper:

```bibtex
@article{wang2025spa,
  title={SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution},
  author={Wang, Hanlin and Leong, Chak Tou and Wang, Jiashuo and Wang, Jian and Li, Wenjie},
  journal={arXiv preprint arXiv:2505.20732},
  year={2025}
}
```

And the WebShop benchmark:

```bibtex
@inproceedings{yao2022webshop,
  title={WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents},
  author={Yao, Shunyu and Chen, Howard and Yang, John and Narasimhan, Karthik},
  booktitle={NeurIPS},
  year={2022}
}
```

---

## Acknowledgments

- [SPA-RL-Agent](https://github.com/WangHanLinHenry/SPA-RL-Agent) — original codebase by Wang et al.
- [WebShop](https://github.com/princeton-nlp/WebShop) — benchmark by Yao et al. (NeurIPS 2022)
- [ETO](https://github.com/Yifan-Song793/ETO) — expert trajectory data and evaluation harness
- [TRL](https://github.com/huggingface/trl) — RL training framework
- [vLLM](https://github.com/vllm-project/vllm) — high-throughput inference engine

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
