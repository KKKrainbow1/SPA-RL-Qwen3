#!/usr/bin/env bash
#
# One-shot setup for the few-shot baseline on AutoDL.
#
# Idempotent — re-running skips work that's already done.
# Tested on: PyTorch 2.8 / Python 3.12 / CUDA 12.8 image, RTX 5090 (32 GB).
#
# After this finishes, kick off the baseline with:
#   MODEL_PATH=/root/autodl-tmp/models/Qwen3-8B \
#       bash scripts/run_fewshot_baseline.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
MODEL_DIR="${MODEL_DIR:-${DATA_DISK}/models/Qwen3-8B}"

step() { echo; echo "==> [$1/9] $2"; }

# -----------------------------------------------------------------------------
# 1. Cache redirection — keep heavy caches off the 30 GB system disk.
# -----------------------------------------------------------------------------
step 1 "Redirect pip / HF / ModelScope / TMP caches to ${DATA_DISK}"
mkdir -p "${DATA_DISK}/.pip-cache" "${DATA_DISK}/.hf" \
         "${DATA_DISK}/.modelscope" "${DATA_DISK}/.tmp"

if ! grep -q "AUTODL_CACHE_REDIRECT" ~/.bashrc 2>/dev/null; then
    {
        printf '\n# AUTODL_CACHE_REDIRECT — managed by scripts/autodl_one_shot_setup.sh\n'
        printf 'export PIP_CACHE_DIR=%s/.pip-cache\n' "${DATA_DISK}"
        printf 'export HF_HOME=%s/.hf\n' "${DATA_DISK}"
        printf 'export MODELSCOPE_CACHE=%s/.modelscope\n' "${DATA_DISK}"
        printf 'export TMPDIR=%s/.tmp\n' "${DATA_DISK}"
    } >> ~/.bashrc
    echo "    + appended cache exports to ~/.bashrc"
else
    echo "    = cache exports already in ~/.bashrc"
fi

export PIP_CACHE_DIR="${DATA_DISK}/.pip-cache"
export HF_HOME="${DATA_DISK}/.hf"
export MODELSCOPE_CACHE="${DATA_DISK}/.modelscope"
export TMPDIR="${DATA_DISK}/.tmp"

# -----------------------------------------------------------------------------
# 2. Free system disk — conda packages cache + leftover pip uninstall residues.
# -----------------------------------------------------------------------------
step 2 "Free system disk"
conda clean --all -y 2>&1 | tail -3 || true
rm -rf /root/miniconda3/lib/python3.12/site-packages/~* 2>/dev/null || true
df -h / | tail -1

# -----------------------------------------------------------------------------
# 3. Pull latest commits.
# -----------------------------------------------------------------------------
step 3 "git pull --ff-only"
cd "${REPO_ROOT}"
git pull --ff-only

# -----------------------------------------------------------------------------
# 4. Java for WebShop's pyserini index (needs JVM 11+).
# -----------------------------------------------------------------------------
step 4 "Install OpenJDK 21 (pyserini's anserini jar needs Java 21+)"
# class file version 65.0 in pyserini's jars ⇒ JDK 21 minimum.
NEED_JDK21=1
if command -v java >/dev/null 2>&1 && java -version 2>&1 | grep -qE 'version "(2[1-9]|[3-9])'; then
    NEED_JDK21=0
fi
if [[ "${NEED_JDK21}" == "1" ]]; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openjdk-21-jdk-headless
fi

JAVA_HOME=$(ls -d /usr/lib/jvm/java-21-openjdk-* 2>/dev/null | head -1)
if [[ -z "${JAVA_HOME}" ]]; then
    echo "ERROR: openjdk-21 install succeeded but JAVA_HOME not found in /usr/lib/jvm/"
    exit 1
fi
export JAVA_HOME
export PATH="${JAVA_HOME}/bin:${PATH}"

# Persist in ~/.bashrc so future sessions also pick up Java 21.
if ! grep -q "JAVA_HOME=/usr/lib/jvm/java-21" ~/.bashrc 2>/dev/null; then
    printf '\n# JDK 21 for pyserini (managed by autodl_one_shot_setup.sh)\nexport JAVA_HOME=%s\nexport PATH=$JAVA_HOME/bin:$PATH\n' "${JAVA_HOME}" >> ~/.bashrc
fi

java -version 2>&1 | head -1

# -----------------------------------------------------------------------------
# 5. Inference-only Python deps. Avoids the requirements.txt vllm-pin trap.
# -----------------------------------------------------------------------------
step 5 "pip install -r requirements-baseline.txt"
pip install -r requirements-baseline.txt

# -----------------------------------------------------------------------------
# 6. WebShop env + data + spacy model.
# -----------------------------------------------------------------------------
step 6 "scripts/setup_data.sh (clones upstream, downloads WebShop data)"
BASELINE_ONLY=1 bash scripts/setup_data.sh

# -----------------------------------------------------------------------------
# 7. Patch upstream FastChat / PRM for Qwen3.
# -----------------------------------------------------------------------------
step 7 "scripts/apply_qwen3_patches.sh"
bash scripts/apply_qwen3_patches.sh

# -----------------------------------------------------------------------------
# 8. Pre-download Qwen3-8B from ModelScope (much faster than HF in CN).
# -----------------------------------------------------------------------------
step 8 "Download Qwen3-8B → ${MODEL_DIR}"
if [[ -d "${MODEL_DIR}" ]] && compgen -G "${MODEL_DIR}/*.safetensors" > /dev/null; then
    echo "    = already downloaded"
else
    mkdir -p "${MODEL_DIR}"
    if command -v modelscope >/dev/null 2>&1; then
        modelscope download --model Qwen/Qwen3-8B --local_dir "${MODEL_DIR}"
    else
        python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='${MODEL_DIR}')"
    fi
fi
du -sh "${MODEL_DIR}" 2>/dev/null || true

# -----------------------------------------------------------------------------
# 9. Final state — versions, disk, GPU.
# -----------------------------------------------------------------------------
step 9 "Final state"
echo "--- versions ---"
pip show vllm torch transformers 2>/dev/null | grep -E "^(Name|Version):"
echo "--- disk ---"
df -h / | tail -1
df -h "${DATA_DISK}" | tail -1
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

cat <<EOM

==================================================================
 Setup complete. Run the baseline with:
   MODEL_PATH=${MODEL_DIR} bash scripts/run_fewshot_baseline.sh
==================================================================
EOM
