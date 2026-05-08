"""Async multi-trajectory exploration on WebShop using vLLM AsyncLLMEngine.

Drop-in replacement for upstream's synchronous
`exploration/webshop/generate_response_webshop.py`.

Key differences:
  - All trajectories run concurrently as asyncio coroutines.
  - One vLLM AsyncLLMEngine serves the model in-process (no FastChat layer).
  - Prefix caching is on so the WebShop system prompt's KV is reused.
  - Per-trajectory loop is still serial (must await env.step) — this is fine
    because env.step is a local call (~milliseconds for WebShop's Flask + Lucene).

Expected throughput (Qwen3-8B on 1×A100 80G, 78 trajectories):
  Sync (FastChat HTTP, default):  ~30 min
  Async (this script, default):    ~5 min
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

# Upstream imports
from webshop.web_agent_site.envs import WebAgentTextEnv

logger = logging.getLogger("async_explore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation, you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
  search[keywords]
  click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:

Thought: I think ...
Action: click[something]"""


def render_messages_to_prompt(tokenizer, messages: list[dict]) -> str:
    """Render conversation history as a Qwen3 ChatML completion prompt."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_action(text: str) -> str | None:
    """Extract `search[...]` or `click[...]` from an LLM response."""
    match = re.search(r"Action:\s*(.+?)(?:\n|$)", text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


# --------------------------------------------------------------------------------------
# Per-trajectory coroutine
# --------------------------------------------------------------------------------------


async def rollout_one_trajectory(
    engine: AsyncLLMEngine,
    tokenizer,
    base_env: WebAgentTextEnv,
    session_id: int,
    iteration: int,
    sampling_params: SamplingParams,
    max_steps: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run a single trajectory to completion. Yields one record at the end."""

    async with semaphore:
        # Each coroutine resets the SHARED env to its session_id; this is OK
        # because we await between resets — coroutines never reset
        # simultaneously. (WebShop's SimServer is not thread-safe.)
        base_env.reset(session_id)
        initial_obs = base_env.observation

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_obs},
        ]

        history_for_record: list[dict] = list(messages)
        final_reward = 0.0
        done = False
        steps_taken = 0

        for step in range(max_steps):
            steps_taken += 1
            prompt = render_messages_to_prompt(tokenizer, messages)
            request_id = f"sess{session_id}_iter{iteration}_step{step}_{uuid.uuid4().hex[:6]}"

            output_text = ""
            async for output in engine.generate(prompt, sampling_params, request_id):
                # Final iteration of the async generator gives the full output
                output_text = output.outputs[0].text

            messages.append({"role": "assistant", "content": output_text})
            history_for_record.append({"role": "assistant", "content": output_text})

            action = parse_action(output_text)
            if action is None:
                obs = "Observation: Invalid format. The input must contains 'Action: '"
                done = False
            else:
                try:
                    obs, final_reward, done, _ = base_env.step(action=action)
                    obs = f"Observation:\n{obs}"
                except AssertionError:
                    obs = "Observation: Invalid action!"
                    done = False

            messages.append({"role": "user", "content": obs})
            history_for_record.append({"role": "user", "content": obs})

            if done:
                break

        return {
            "session_id": session_id,
            "iteration": iteration,
            "agent_final_reward": final_reward,
            "steps": steps_taken,
            "done": done,
            "conversations": history_for_record,
        }


# --------------------------------------------------------------------------------------
# Main async driver
# --------------------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    # 1. Load task indices
    with open(args.task_indices) as f:
        task_indices = json.load(f)
    if args.limit:
        task_indices = task_indices[: args.limit]

    logger.info(f"Loaded {len(task_indices)} tasks; running {args.num_iterations} iterations each")

    # 2. Bring up vLLM
    engine_args = AsyncEngineArgs(
        model=args.model_path,
        enable_prefix_caching=args.prefix_cache,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        dtype="bfloat16",
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = await engine.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>"],
    )

    # 3. WebShop env (single shared instance — see note in rollout_one_trajectory)
    base_env = WebAgentTextEnv(observation_mode="text", human_goals=True)

    # 4. Concurrency guard
    semaphore = asyncio.Semaphore(args.concurrency)

    # 5. Build trajectory tasks
    coros = []
    for session_id in task_indices:
        for iter_id in range(args.num_iterations):
            coros.append(
                rollout_one_trajectory(
                    engine=engine,
                    tokenizer=tokenizer,
                    base_env=base_env,
                    session_id=session_id,
                    iteration=iter_id,
                    sampling_params=sampling_params,
                    max_steps=args.max_steps,
                    semaphore=semaphore,
                )
            )

    logger.info(f"Launching {len(coros)} trajectory coroutines (concurrency={args.concurrency})")
    t0 = time.time()
    results = await asyncio.gather(*coros)
    elapsed = time.time() - t0

    # 6. Save
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "exploration.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_done = sum(1 for r in results if r["done"])
    n_success = sum(1 for r in results if r["agent_final_reward"] >= 0.99)
    avg_reward = sum(r["agent_final_reward"] for r in results) / max(1, len(results))

    logger.info("=" * 60)
    logger.info(f"Total trajectories: {len(results)}")
    logger.info(f"Done (Buy Now reached): {n_done}")
    logger.info(f"Success (reward >= 0.99): {n_success}")
    logger.info(f"Avg reward: {avg_reward:.4f}")
    logger.info(f"Wall time: {elapsed:.1f}s ({len(results) / elapsed * 60:.1f} traj/min)")
    logger.info(f"Output: {out_path}")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--task_indices",
        required=True,
        help="Path to a JSON file containing a list of WebShop session IDs",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on # tasks")

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--prefix_cache", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
