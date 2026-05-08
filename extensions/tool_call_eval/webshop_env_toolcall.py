"""WebShopEnv variant that accepts both Qwen3 tool-call and legacy text actions.

Drop-in replacement for upstream's `eval_agent.envs.WebShopEnv`. The only
behavioural change is in `parse_action` — everything else (history tracking,
max_steps, reward propagation, etc.) is inherited unchanged.
"""

from __future__ import annotations

import logging
from typing import Tuple

# Upstream import path. After running `scripts/apply_qwen3_patches.sh`,
# eval_agent is on the PYTHONPATH from `upstream/`.
from eval_agent.envs.webshop_env import WebShopEnv  # type: ignore

from extensions.tool_call_eval.parser import parse_tool_call_output

logger = logging.getLogger("agent_frame")


class WebShopToolCallEnv(WebShopEnv):
    """Accepts Qwen3 `<tool_call>{...}</tool_call>` output in addition to
    the upstream `Action: search[...]` text format. Useful for evaluating
    Qwen3 / GPT-4 / Claude / etc. agents that prefer function-calling APIs.
    """

    def parse_action(self, llm_output: str) -> str:  # type: ignore[override]
        action = parse_tool_call_output(llm_output)
        if action is None:
            raise ValueError(
                f"No parseable action in LLM output (tool_call/text both failed). "
                f"Output preview: {llm_output[:200]!r}"
            )
        return action
