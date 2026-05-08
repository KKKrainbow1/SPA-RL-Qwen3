# Tool-Call Evaluation

Evaluates Qwen3-8B on WebShop using **OpenAI-style function calling** instead of the upstream `Action: search[...]` text-mode protocol.

## Why this exists

Qwen3 has been heavily trained on function calling — its BFCL score is ~85, on par with much larger closed-source models. But the upstream SPA evaluation harness only knows how to parse `Action: search[keyword]` from a free-form text response (regex `r"Action:\s*(.+?)(?:\n|$)"`).

This module bridges the two:

1. Defines the WebShop action space as OpenAI-style **tool schemas** (`webshop_tools.py`).
2. Tells vLLM to use Qwen3's native tool-call output format (`<tool_call>...</tool_call>`).
3. Provides a **parser** that converts Qwen3's JSON tool calls back into the `search[...]` / `click[...]` strings that `WebAgentTextEnv.step()` expects.
4. Wraps the upstream `WebShopEnv` so it accepts either format (tool-call preferred, text fallback).
5. Provides an **evaluation script** that runs the 200-task test split end-to-end.

## When to use this vs the upstream eval

| Scenario | Use upstream eval (`Action: ...` text) | Use tool-call eval (this module) |
|---|---|---|
| Reproducing SPA paper numbers on Llama-3 | ✅ | ❌ |
| Evaluating a Qwen3 SFT/RL checkpoint | ❌ (wastes Qwen3's function-calling ability) | ✅ |
| Few-shot baseline of any modern model | ❌ | ✅ |
| Comparing tool-call vs text-mode interface | use both, ablate | ✅ |

## Files

| File | Role |
|---|---|
| `webshop_tools.py` | OpenAI tool schemas for `search` and `click` |
| `parser.py` | `parse_tool_call_output(llm_output) -> "search[...]" \| "click[...]"` |
| `webshop_env_toolcall.py` | `WebShopToolCallEnv` — drop-in replacement for `WebShopEnv` |
| `fewshot_eval.py` | End-to-end few-shot evaluation on 200-task test split |
| `run_qwen3_fewshot.sh` | Launch vLLM OpenAI server + run eval |
| `few_shot_examples.json` | One reference trajectory used as in-context demonstration |

## Usage

```bash
# 1. Make sure upstream is cloned and WebShop env is installed
# 2. Run the launch script:
bash extensions/tool_call_eval/run_qwen3_fewshot.sh
```

The script does:
1. Starts a vLLM OpenAI-compatible server with `--enable-auto-tool-choice --tool-call-parser hermes`.
2. Waits for it to be ready (~90s for Qwen3-8B).
3. Runs `fewshot_eval.py` against the 200-task `test_indices.json` split.
4. Writes per-task trajectory JSONs and a summary to `results/qwen3_8b_fewshot/`.
5. Tears down the vLLM server.

## Expected results

Without any training, just few-shot prompting:

| Setup | Expected Avg Reward |
|---|---|
| Qwen3-8B zero-shot, tool-call | 30-45% |
| Qwen3-8B 1-shot, tool-call | 40-55% |

These serve as a baseline to compare against the SPA-trained model (target: 60-69%).

## How the parser handles edge cases

`parser.py` is permissive — Qwen3 sometimes wraps tool calls in markdown, includes extra reasoning, or omits the `<tool_call>` tags entirely. It tries:

1. Find `<tool_call>{...}</tool_call>` block (Qwen3's documented format).
2. Fall back to any JSON object containing `"name": "search"|"click"`.
3. Fall back to legacy `Action: search[...]` text (in case the model regresses to text mode).

If all three fail, returns `None` and the env wrapper records "Invalid format" — the trajectory continues but that step is wasted.

## Why not retrain SFT in tool-call format?

Considered and rejected:

- The SFT data (1624 expert trajectories) is in `Action: search[...]` text. Re-rendering it as tool calls means re-tokenizing, re-masking, and verifying every example.
- Step-wise PPO's `frag_mask` construction depends on assistant turn boundaries; tool-call wrapping changes the token layout and would require updates to `formatting_func`.
- The PRM is trained on the same text-format trajectories; switching to tool-calls means retraining PRM too.

The parser-based approach lets us keep the entire training pipeline as-is and only change the **inference interface**, which is where Qwen3's tool-calling shines.
