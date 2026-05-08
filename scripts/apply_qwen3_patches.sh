#!/usr/bin/env bash
#
# Patch upstream SPA codebase to support Qwen3-8B.
# Idempotent — safe to re-run.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${REPO_ROOT}/upstream"

if [[ ! -d "${UPSTREAM}" ]]; then
    echo "ERROR: upstream not found. Run scripts/setup_data.sh first."
    exit 1
fi

echo "==> Patching FastChat with Qwen3 adapter..."
ADAPTER_PATH="${UPSTREAM}/fastchat/model/qwen3_adapter.py"
cp "${REPO_ROOT}/extensions/qwen3_adapter/fastchat_qwen3.py" "${ADAPTER_PATH}"

# Append registration call to fastchat's __init__ if not already present
INIT_FILE="${UPSTREAM}/fastchat/model/__init__.py"
if ! grep -q "qwen3_adapter" "${INIT_FILE}"; then
    cat >> "${INIT_FILE}" << 'EOF'

# SPA-RL-Qwen3 patch: register Qwen3 adapter
try:
    from fastchat.model.qwen3_adapter import register as _register_qwen3
    _register_qwen3()
except Exception:
    pass
EOF
    echo "    + Added Qwen3 registration to ${INIT_FILE}"
else
    echo "    = Qwen3 registration already present in ${INIT_FILE}"
fi

echo "==> Adding Qwen3 branch to prm/inference_prm.py..."
PRM_FILE="${UPSTREAM}/prm/inference_prm.py"
if ! grep -q "qwen3_branch" "${PRM_FILE}"; then
    # Inject import + Qwen3 branch just before the existing Llama-3.2-3B branch
    python "${REPO_ROOT}/scripts/_inject_qwen3_branch.py" "${PRM_FILE}"
    echo "    + Patched ${PRM_FILE}"
else
    echo "    = ${PRM_FILE} already patched"
fi

# Same patch for the training-side preprocessor
for f in "${UPSTREAM}/prm/train_our_progress_model.py" \
         "${UPSTREAM}/prm/train_our_progress_model_lora.py" \
         "${UPSTREAM}/prm/train_our_progress_model_fp16.py" \
         "${UPSTREAM}/prm/train_our_progress_model_lora_fp16.py"; do
    if [[ -f "$f" ]] && ! grep -q "qwen3_branch" "$f"; then
        python "${REPO_ROOT}/scripts/_inject_qwen3_branch.py" "$f"
        echo "    + Patched $f"
    fi
done

# Make our extensions importable from upstream's working dir
EXT_LINK="${UPSTREAM}/extensions"
if [[ ! -L "${EXT_LINK}" ]]; then
    ln -s "${REPO_ROOT}/extensions" "${EXT_LINK}"
    echo "    + Linked extensions -> ${EXT_LINK}"
fi

# Make optional eval_agent task imports lazy. Upstream's eval_agent/tasks/__init__.py
# eagerly imports every task module (alfworld, sciworld, etc.); since baseline only
# needs WebShopTask, wrap each `from .X import Y` line in try/except so a missing
# optional env package doesn't kill the WebShopTask import.
TASKS_INIT="${UPSTREAM}/eval_agent/tasks/__init__.py"
if [[ -f "${TASKS_INIT}" ]] && ! grep -q "except ImportError" "${TASKS_INIT}"; then
    python - <<PY
import re
p = "${TASKS_INIT}"
with open(p) as f: c = f.read()
c = re.sub(r'^(from \.\w+ import .+)$',
           r'try:\n    \1\nexcept ImportError:\n    pass',
           c, flags=re.MULTILINE)
with open(p, 'w') as f: f.write(c)
PY
    echo "    + Wrapped optional task imports in ${TASKS_INIT}"
else
    echo "    = ${TASKS_INIT} already patched (or missing)"
fi

echo "==> Done. Verify:"
echo "    cd upstream && python -c \"from fastchat.model.model_adapter import get_model_adapter; print(get_model_adapter('Qwen/Qwen3-8B'))\""
