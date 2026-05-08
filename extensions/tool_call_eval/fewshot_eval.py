"""Few-shot evaluation of Qwen3-8B on WebShop using OpenAI-style tool calls.

Loads `test_indices.json` (200 tasks), runs each through the agent loop with a
single in-context demonstration, and reports avg reward + success rate.

Usage:
    # First start the vLLM OpenAI server (see run_qwen3_fewshot.sh)
    python extensions/tool_call_eval/fewshot_eval.py \\
        --model_name Qwen/Qwen3-8B \\
        --vllm_port 8000 \\
        --output_dir results/qwen3_8b_fewshot/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import openai
from tqdm import tqdm

# Upstream
from eval_agent.tasks.webshop import WebShopTask  # type: ignore
from webshop.web_agent_site.envs import WebAgentTextEnv  # type: ignore

# Ours
from extensions.tool_call_eval.webshop_env_toolcall import WebShopToolCallEnv
from extensions.tool_call_eval.webshop_tools import WEBSHOP_TOOLS

logger = logging.getLogger("qwen3_fewshot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


SYSTEM_PROMPT = """You are an autonomous shopping agent operating on WebShop, a text-based e-commerce platform.

Given a shopping instruction, you must navigate the website to find and purchase the product that best matches the user's requirements.

Workflow:
1. Read the instruction carefully and identify key attributes (product type, color, size, price limit).
2. Use the `search` function with relevant keywords to find candidate products.
3. Examine search results and use `click` to navigate to a promising product (use the ASIN ID like 'B09C337K8S').
4. On the product page, click options like color or size if needed, then click 'Buy Now' to complete the purchase.

Always reason briefly before calling a tool. Make the FEWEST tool calls possible to complete the task."""


# --------------------------------------------------------------------------------------
# Few-shot demonstration construction
# --------------------------------------------------------------------------------------


def _hashed_call_id(payload: dict) -> str:
    return f"call_{abs(hash(json.dumps(payload, sort_keys=True))) % 10**8}"


def build_fewshot_messages(
    examples: list[dict],
    initial_observation: str,
) -> list[dict]:
    """Build the message list with system prompt + few-shot examples + initial obs."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in examples:
        for step in ex["trajectory"]:
            messages.append({"role": "user", "content": step["observation"]})
            call_id = _hashed_call_id(step["tool_call"])
            messages.append(
                {
                    "role": "assistant",
                    "content": step["thought"],
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": step["tool_call"]["name"],
                                "arguments": json.dumps(step["tool_call"]["arguments"]),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "(observation continues in next turn)",
                }
            )

    messages.append({"role": "user", "content": initial_observation})
    return messages


# --------------------------------------------------------------------------------------
# Per-task driver
# --------------------------------------------------------------------------------------


def run_one_task(
    client: openai.OpenAI,
    model_name: str,
    task: WebShopTask,
    base_env: WebAgentTextEnv,
    fewshot_examples: list[dict],
    max_steps: int,
):
    env = WebShopToolCallEnv(
        task=task,
        env=base_env,
        instruction_path="upstream/eval_agent/prompt/instructions/webshop_inst.txt",
        icl_path="upstream/eval_agent/prompt/icl_examples/webshop_icl.json",
        icl_format="conversation",
        max_steps=max_steps,
    )
    observation, state = env.reset()
    initial_obs = observation

    messages = build_fewshot_messages(fewshot_examples, initial_obs)

    for _step in range(max_steps):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=WEBSHOP_TOOLS,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=512,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        except Exception as e:
            logger.error(f"task {task.task_id}: API error {e}")
            break

        msg = resp.choices[0].message

        if not msg.tool_calls:
            logger.warning(
                f"task {task.task_id}: no tool_call returned. content={msg.content!r}"
            )
            break

        tc = msg.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            logger.warning(f"task {task.task_id}: bad JSON args {tc.function.arguments!r}")
            break

        if tc.function.name == "search":
            llm_text_output = (
                f"Thought: {msg.content or ''}\nAction: search[{args.get('keyword','')}]"
            )
        elif tc.function.name == "click":
            llm_text_output = (
                f"Thought: {msg.content or ''}\nAction: click[{args.get('value','')}]"
            )
        else:
            logger.warning(f"task {task.task_id}: unknown tool {tc.function.name}")
            break

        # Send to env (parser inside the env handles either format)
        observation, state = env.step(llm_text_output)

        # Append assistant + tool to message history for next turn
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump()],
            }
        )
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})

        if state.finished:
            break

    return state


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # vLLM OpenAI-compatible client
    client = openai.OpenAI(
        base_url=f"http://{args.vllm_host}:{args.vllm_port}/v1",
        api_key="EMPTY",
    )

    # Load tasks (default: 200 from test_indices.json)
    tasks_iter, n_tasks = WebShopTask.load_tasks("test", part_num=1)
    tasks = list(tasks_iter)
    if args.limit:
        tasks = tasks[: args.limit]
    logger.info(f"Running {len(tasks)} tasks")

    # Few-shot examples (loaded from JSON)
    fewshot_path = Path(__file__).parent / "few_shot_examples.json"
    with open(fewshot_path) as f:
        fewshot_examples = json.load(f) if args.k_shot > 0 else []
    fewshot_examples = fewshot_examples[: args.k_shot]
    logger.info(f"Using {len(fewshot_examples)}-shot prompting")

    # Single shared WebShop env instance (SimServer is not thread-safe;
    # we run sequentially, so this is fine)
    base_env = WebAgentTextEnv(observation_mode="text", human_goals=True)

    rewards: list[float] = []
    successes: list[bool] = []

    for task in tqdm(tasks, desc="Eval"):
        state = run_one_task(
            client=client,
            model_name=args.model_name,
            task=task,
            base_env=base_env,
            fewshot_examples=fewshot_examples,
            max_steps=args.max_steps,
        )
        rewards.append(state.reward if state.reward is not None else 0.0)
        successes.append(bool(state.success))

        with open(output_dir / f"{task.task_id}.json", "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    avg_reward = sum(rewards) / max(1, len(rewards))
    success_rate = sum(successes) / max(1, len(successes))

    logger.warning("=" * 60)
    logger.warning(f"Model:        {args.model_name}")
    logger.warning(f"Tasks:        {len(rewards)}")
    logger.warning(f"k-shot:       {args.k_shot}")
    logger.warning(f"Avg reward:   {avg_reward:.4f}")
    logger.warning(f"Success rate: {success_rate:.4f}")
    logger.warning("=" * 60)

    summary = {
        "model": args.model_name,
        "n_tasks": len(rewards),
        "k_shot": args.k_shot,
        "max_steps": args.max_steps,
        "avg_reward": avg_reward,
        "success_rate": success_rate,
        "rewards": rewards,
        "successes": successes,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--vllm_host", default="localhost")
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--k_shot", type=int, default=1, help="Number of few-shot examples (0=zero-shot)")
    parser.add_argument("--limit", type=int, default=None)
    main(parser.parse_args())
