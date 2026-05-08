"""FastChat adapter for Qwen3 models.

Insert into `fastchat/model/model_adapter.py` (in the upstream SPA repo's
vendored fastchat) and register via the existing `register_model_adapter`
mechanism. Without this, FastChat falls back to a default conversation
template and serves prompts with the wrong format.
"""

from __future__ import annotations

from fastchat.conversation import Conversation, SeparatorStyle, get_conv_template
from fastchat.model.model_adapter import BaseModelAdapter, register_model_adapter


class Qwen3Adapter(BaseModelAdapter):
    """Adapter for Qwen3 family (Qwen3-{0.5B, 1.7B, 4B, 8B, 14B, 32B}).

    Conversation template uses ChatML format:
        <|im_start|>{role}\n{content}<|im_end|>\n
    """

    def match(self, model_path: str) -> bool:
        path = model_path.lower()
        return "qwen3" in path or "qwen-3" in path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        # FastChat already ships a `qwen-7b-chat` template; reuse if present.
        try:
            return get_conv_template("qwen-7b-chat")
        except KeyError:
            return Conversation(
                name="qwen3",
                system_message="",
                roles=("<|im_start|>user", "<|im_start|>assistant"),
                sep_style=SeparatorStyle.CHATML,
                sep="<|im_end|>\n",
                stop_token_ids=[
                    151643,  # <|endoftext|>
                    151645,  # <|im_end|>
                ],
            )


def register():
    """Register the Qwen3 adapter with FastChat. Call once at module import."""
    register_model_adapter(Qwen3Adapter)


if __name__ == "__main__":
    register()
    from fastchat.model.model_adapter import get_model_adapter

    print(get_model_adapter("Qwen/Qwen3-8B"))
    print(get_model_adapter("./ckpt/qwen3_webshop_sft_loramerged"))
