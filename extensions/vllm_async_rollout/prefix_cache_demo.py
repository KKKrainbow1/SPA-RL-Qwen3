"""Quick standalone demo of prefix caching speedup on a fixed system prompt.

Useful as a smoke test: if this script doesn't show a clear speedup with
`enable_prefix_caching=True`, your vLLM build or hardware doesn't support it
properly, and the full async exploration won't see the expected gain either.

Run:
    python extensions/vllm_async_rollout/prefix_cache_demo.py \\
        --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import time

from vllm import LLM, SamplingParams

SHARED_SYSTEM = """You are web shopping. I will give you instructions about what to do.
You have to follow the instructions. Every round I will give you an observation, you
have to respond an action. You can use search action if search is available. You can
click one of the buttons. An action should be of the structure: search[keywords] or
click[value]. If the action is not valid, perform nothing. Your response should use:
Thought: I think ...
Action: click[something]"""

VARIED_USERS = [
    "WebShop [SEP] Instruction: i need a long clip-in hair extension under $40 [SEP] Search",
    "WebShop [SEP] Instruction: find me a black men's t-shirt size large under $25 [SEP] Search",
    "WebShop [SEP] Instruction: i want a wireless mouse with usb dongle below $30 [SEP] Search",
    "WebShop [SEP] Instruction: looking for a kids' lego set under $50 [SEP] Search",
    "WebShop [SEP] Instruction: i need a yoga mat purple color around $25 [SEP] Search",
    "WebShop [SEP] Instruction: find a mens sneaker size 10 black [SEP] Search",
    "WebShop [SEP] Instruction: looking for a coffee mug ceramic blue under $15 [SEP] Search",
    "WebShop [SEP] Instruction: i need a desk lamp led adjustable under $30 [SEP] Search",
] * 8  # 64 prompts total, all sharing the same system prefix


def run_and_time(llm: LLM, prompts: list[str], label: str) -> float:
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    t0 = time.time()
    _ = llm.generate(prompts, sp, use_tqdm=False)
    elapsed = time.time() - t0
    print(f"{label}: {elapsed:.2f}s for {len(prompts)} prompts ({len(prompts)/elapsed:.1f} prompt/s)")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    prompts = [SHARED_SYSTEM + "\n\n" + u for u in VARIED_USERS]

    print(f"\n{'=' * 60}\n  Prefix Caching: OFF\n{'=' * 60}")
    llm_off = LLM(
        model=args.model,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )
    _ = run_and_time(llm_off, prompts, "Cold (no prefix cache)")
    t_off = run_and_time(llm_off, prompts, "Warm (no prefix cache)")
    del llm_off

    print(f"\n{'=' * 60}\n  Prefix Caching: ON\n{'=' * 60}")
    llm_on = LLM(
        model=args.model,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )
    _ = run_and_time(llm_on, prompts, "Cold (prefix cache primed on first run)")
    t_on = run_and_time(llm_on, prompts, "Warm (prefix cache hit)")

    print(f"\nSpeedup: {t_off / t_on:.2f}× warm-vs-warm")


if __name__ == "__main__":
    main()
