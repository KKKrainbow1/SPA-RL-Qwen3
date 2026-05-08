"""Parse Qwen3's tool-call output into the WebShop env's expected action string.

Qwen3's expected output format (when given `tools=[...]` and `tool_choice="auto"`):

    <tool_call>
    {"name": "search", "arguments": {"keyword": "long clip-in hair extension"}}
    </tool_call>

WebAgentTextEnv.step() expects:

    "search[long clip-in hair extension]"
    "click[B09C337K8S]"

This module bridges the two with three fallback strategies, so we don't fail
the trajectory on minor formatting glitches.
"""

from __future__ import annotations

import json
import re

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.+?)\s*</tool_call>", re.DOTALL)
_INLINE_JSON = re.compile(
    r'\{[^{}]*"name"\s*:\s*"(?:search|click)"[^{}]*"arguments"[^{}]*\}', re.DOTALL
)
_LEGACY_ACTION = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.DOTALL)


def parse_tool_call_output(llm_output: str) -> str | None:
    """Convert a Qwen3 tool-call response to a WebShop action string.

    Returns the formatted action (e.g. "search[shoes]") or None if no
    parseable action is found.
    """
    if not llm_output:
        return None

    # Strategy 1: <tool_call>{...}</tool_call> wrapper (canonical Qwen3 format)
    m = _TOOL_CALL_BLOCK.search(llm_output)
    if m:
        action = _from_json(m.group(1).strip())
        if action is not None:
            return action

    # Strategy 2: bare JSON object that looks like a tool call
    m = _INLINE_JSON.search(llm_output)
    if m:
        action = _from_json(m.group(0))
        if action is not None:
            return action

    # Strategy 3: legacy `Action: search[...]` text format
    m = _LEGACY_ACTION.search(llm_output)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith(("search[", "click[")) and candidate.endswith("]"):
            return candidate

    return None


def _from_json(s: str) -> str | None:
    """Parse a single JSON tool-call object and convert to action string."""
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None

    name = obj.get("name")
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None

    if not isinstance(args, dict):
        return None

    if name == "search":
        keyword = (args.get("keyword") or "").strip()
        if not keyword:
            return None
        return f"search[{keyword}]"

    if name == "click":
        value = (args.get("value") or "").strip()
        if not value:
            return None
        return f"click[{value}]"

    return None


def has_tool_call(llm_output: str) -> bool:
    """Quick check: does this output contain a parseable action?"""
    return parse_tool_call_output(llm_output) is not None


# ---- Self-test ---------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        # Canonical Qwen3 format
        (
            '<tool_call>\n{"name":"search","arguments":{"keyword":"red shoes"}}\n</tool_call>',
            "search[red shoes]",
        ),
        # Click via tool call
        (
            '<tool_call>\n{"name":"click","arguments":{"value":"Buy Now"}}\n</tool_call>',
            "click[Buy Now]",
        ),
        # Inline JSON without wrapper
        ('{"name":"search","arguments":{"keyword":"laptop"}}', "search[laptop]"),
        # Arguments as JSON string (some models emit this)
        (
            '<tool_call>{"name":"click","arguments":"{\\"value\\":\\"B09C337K8S\\"}"}</tool_call>',
            "click[B09C337K8S]",
        ),
        # Legacy text fallback
        ("Thought: search.\nAction: search[blue mug]", "search[blue mug]"),
        # No action
        ("I am not sure what to do here.", None),
    ]
    for raw, expected in cases:
        got = parse_tool_call_output(raw)
        status = "PASS" if got == expected else "FAIL"
        print(f"[{status}] expected={expected!r:30}  got={got!r}")
