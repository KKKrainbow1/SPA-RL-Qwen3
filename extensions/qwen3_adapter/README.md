# Qwen3 Adapter

Drop-in adaptation layer that migrates the SPA-RL pipeline from Llama-3.2-3B-Instruct to Qwen3-8B.

## What it does

The original SPA codebase hardcodes Llama-3 conventions in three places:

1. **Chat template** — uses `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>` literally in `prm/rl_data_org.py` and `ppo/step_ppo.py::formatting_func`.
2. **PRM turn boundary detection** — `prm/inference_prm.py::preprocess` has a Llama-3-specific branch that uses `<|eot_id|>` and `<|end_header_id|>` to slice turns.
3. **FastChat conversation template** — `fastchat/model/model_adapter.py` ships adapters for Llama variants but not Qwen3.

This module replaces those three with Qwen3-compatible logic.

## Files

| File | Replaces / Patches | Purpose |
|---|---|---|
| `chat_template.py` | `prm/rl_data_org.py` template strings | Renders messages in Qwen3 ChatML format (`<\|im_start\|>{role}\n{content}<\|im_end\|>`) via `tokenizer.apply_chat_template(..., enable_thinking=False)` |
| `prm_preprocess_qwen3.py` | `prm/inference_prm.py::preprocess` Llama branch | Adds a Qwen3 branch using `<\|im_end\|>` / `<\|im_start\|>` separators with corrected turn-boundary offsets |
| `fastchat_qwen3.py` | `fastchat/model/model_adapter.py` | Defines `Qwen3Adapter(BaseModelAdapter)` and registers it via `register_model_adapter` so `vllm_worker` picks up the right conversation template |

## Why thinking mode is disabled

Qwen3 enables `<think>...</think>` reasoning blocks by default. For agent tasks like WebShop:

- Training data (SFT trajectories) **does not contain** `<think>` blocks.
- Mixing `<think>` into rollouts at inference time creates a train/eval distribution mismatch.
- PRM is trained on assistant turns whose last token is `<|im_end|>`; thinking blocks shift this position and degrade the per-turn progress estimate.

Standard practice from the Qwen3 documentation is to disable thinking for agent-style tasks via:

```python
tokenizer.apply_chat_template(messages, enable_thinking=False, ...)
```

All four pipeline stages (SFT, exploration, PRM, PPO) call the helper `render_qwen3_chat()` from `chat_template.py`, which sets this flag uniformly. Inference-time vLLM serving passes `chat_template_kwargs={"enable_thinking": False}` via `extra_body`.

## Vocab size note

The original `prm_model` class (`prm/inference_prm.py:30`) hardcodes `vocab_size=32000` (LLaMA-2 era). For Qwen3-8B this should be `model.config.vocab_size` (~151,936). The patch in `prm_preprocess_qwen3.py` reads it dynamically:

```python
vocab_size = base_model.config.vocab_size
self.LN = nn.Linear(vocab_size, 1)
```

## Verification checklist

After applying these patches, before running full training, verify:

1. `tokenizer.apply_chat_template([{...}], enable_thinking=False, tokenize=False)` produces output starting with `<|im_start|>system\n` and **not** `<|begin_of_text|>` or `<think>`.
2. PRM forward on a single trajectory yields exactly N `turn_values` for N assistant turns (off-by-one is the most common bug here).
3. FastChat `get_model_adapter("Qwen/Qwen3-8B")` returns `Qwen3Adapter`, not the default `BaseModelAdapter`.

## How to apply

```bash
# From repo root, with upstream/ already cloned:
bash scripts/apply_qwen3_patches.sh
```

The patch script copies these files into the corresponding upstream locations and adds Qwen3 branches to the relevant Python files. Original Llama-3 paths remain functional — selection is by `model_path`.
