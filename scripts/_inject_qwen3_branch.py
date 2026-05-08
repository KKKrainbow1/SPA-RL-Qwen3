#!/usr/bin/env python3
"""Helper for apply_qwen3_patches.sh — injects a Qwen3 branch into upstream's
PRM preprocess function. Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

QWEN3_IMPORT = (
    "from extensions.qwen3_adapter.prm_preprocess_qwen3 import "
    "preprocess_qwen3_branch  # qwen3_branch\n"
)

QWEN3_BRANCH = '''
    # qwen3_branch — inserted by SPA-RL-Qwen3
    if 'qwen3' in model_path.lower() or 'Qwen3' in model_path:
        return preprocess_qwen3_branch(conversations, tokenizer, input_ids, targets)
'''


def patch(path: Path) -> None:
    text = path.read_text()
    if "qwen3_branch" in text:
        return  # already patched

    # 1. Add import near the top
    if "import transformers" in text:
        text = text.replace(
            "import transformers", "import transformers\n" + QWEN3_IMPORT, 1
        )
    elif "from transformers" in text:
        idx = text.find("from transformers")
        end = text.find("\n", idx)
        text = text[: end + 1] + QWEN3_IMPORT + text[end + 1 :]
    else:
        # Fallback: just prepend
        text = QWEN3_IMPORT + text

    # 2. Insert Qwen3 branch before the Llama-3.2-3B branch
    marker = "if 'Llama-3.2-3B-Instruct' in model_path"
    if marker in text:
        text = text.replace(marker, QWEN3_BRANCH.lstrip() + "    " + marker, 1)
    else:
        # Fallback: append branch at end of file
        text = text + "\n# qwen3_branch fallback insertion\n" + QWEN3_BRANCH

    path.write_text(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <upstream_python_file>", file=sys.stderr)
        sys.exit(1)
    patch(Path(sys.argv[1]))
