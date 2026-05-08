"""Qwen3 ChatML rendering helpers used by SFT data prep, exploration, PRM, and PPO."""

from __future__ import annotations

from typing import Iterable

from transformers import PreTrainedTokenizerBase


def render_qwen3_chat(
    messages: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Render a list of {role, content} dicts as a Qwen3 ChatML string.

    Always disables thinking mode by default — agent tasks have no <think>
    blocks in training data, and PRM turn-boundary detection assumes the
    assistant turn ends at <|im_end|> (which is shifted when thinking is on).
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def split_into_assistant_turns(rendered: str) -> list[str]:
    """Split a fully-rendered Qwen3 chat string into per-assistant-turn segments.

    Useful for `frag_mask` construction in step-wise PPO and for PRM
    turn-boundary detection. Each segment ends at a `<|im_end|>` token.
    """
    parts: list[str] = []
    cursor = 0
    while True:
        marker = rendered.find("<|im_start|>assistant\n", cursor)
        if marker < 0:
            break
        end = rendered.find("<|im_end|>", marker)
        if end < 0:
            break
        parts.append(rendered[marker : end + len("<|im_end|>")])
        cursor = end + len("<|im_end|>")
    return parts


def assistant_turn_end_token_positions(
    input_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """Return the token positions of `<|im_end|>` *within assistant turns*.

    This is what the PRM `Linear(vocab_size, 1)` head consumes — one scalar
    per assistant turn. The Llama-3 version in upstream looks for `<|eot_id|>`
    after `<|end_header_id|>`; this Qwen3 version looks for `<|im_end|>`
    after `<|im_start|>assistant\\n`.
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_marker_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    positions: list[int] = []
    inside_assistant = False
    n = len(input_ids)

    i = 0
    while i < n:
        if (
            not inside_assistant
            and input_ids[i] == im_start
            and i + len(assistant_marker_ids) <= n
            and input_ids[i : i + len(assistant_marker_ids)] == assistant_marker_ids
        ):
            inside_assistant = True
            i += len(assistant_marker_ids)
            continue

        if inside_assistant and input_ids[i] == im_end:
            positions.append(i)
            inside_assistant = False
            i += 1
            continue

        i += 1

    return positions


def messages_to_webshop_action(messages: Iterable[dict]) -> str | None:
    """Extract the last assistant turn's `Action: ...` from a message list.

    Used to bridge from chat-template-rendered messages back to the
    `search[...]` / `click[...]` action string the WebShop env expects.
    """
    import re

    last_assistant: str | None = None
    for m in messages:
        if m.get("role") == "assistant":
            last_assistant = m.get("content", "")
    if last_assistant is None:
        return None

    match = re.search(r"Action:\s*(.+?)(?:\n|$)", last_assistant, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()
