"""Qwen3 branch for `prm/inference_prm.py::preprocess`.

The upstream function has explicit branches for Llama-3.2-3B / Llama-3.1-8B
that locate assistant-turn boundaries via `<|eot_id|>` and `<|end_header_id|>`.
This file provides the equivalent Qwen3 branch using `<|im_end|>` and
`<|im_start|>` separators.

Apply by inserting the body of `preprocess_qwen3_branch` near
`prm/inference_prm.py:151` (right before the existing Llama branch).
"""

from __future__ import annotations

import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess_qwen3_branch(
    conversations: list[str],
    tokenizer: transformers.PreTrainedTokenizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> dict:
    """Mask all non-assistant tokens to IGNORE, leaving assistant tokens for PRM head.

    Returns a dict with the same keys as upstream's Llama-3 branch.
    """
    sep_end = "<|im_end|>"      # turn end marker (shared across user/assistant)
    sep_start = "<|im_start|>"  # role marker

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # Each turn ends at <|im_end|>; we walk through them, alternating
        # role markers ( user / assistant / system ).
        turns = conversation.split(sep_end)

        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID

        for i, turn in enumerate(turns):
            if turn == "":
                break

            # Identify role of this turn from the <|im_start|>{role} prefix.
            is_assistant = "<|im_start|>assistant" in turn

            turn_len = len(tokenizer(turn).input_ids)

            if not is_assistant:
                # System / user / tool turn — mask everything
                target[cur_len : cur_len + turn_len + 1] = IGNORE_TOKEN_ID
                cur_len += turn_len + 1
            else:
                # Assistant turn — mask the role-marker tokens (~3 tokens for
                # `<|im_start|>assistant\n`), keep the content tokens for PRM
                role_marker_len = len(
                    tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
                )
                target[cur_len : cur_len + role_marker_len] = IGNORE_TOKEN_ID
                cur_len += turn_len + 1  # +1 for the <|im_end|> separator

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        gpt_unmask=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def patch_prm_model_for_qwen3(prm_model_class):
    """Patch the `prm_model.__init__` so vocab_size is read from base model.

    Upstream hardcodes `vocab_size=32000`, which only works for LLaMA-2.
    For Qwen3-8B, vocab is ~151,936 — must read from `base_model.config`.

    Use as a decorator or call once at import time.
    """
    original_init = prm_model_class.__init__

    def patched_init(self, base_model, vocab_size=None):
        if vocab_size is None:
            vocab_size = base_model.config.vocab_size
        original_init(self, base_model, vocab_size=vocab_size)

    prm_model_class.__init__ = patched_init
    return prm_model_class
