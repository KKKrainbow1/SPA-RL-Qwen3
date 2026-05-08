#!/usr/bin/env bash
#
# One-shot setup: clone upstream SPA-RL-Agent, download WebShop data,
# install env in editable mode.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/upstream"

# 1. Clone upstream if not already present
if [[ ! -d "${UPSTREAM_DIR}" ]]; then
    echo "==> Cloning upstream SPA-RL-Agent..."
    git clone https://github.com/WangHanLinHenry/SPA-RL-Agent.git "${UPSTREAM_DIR}"
else
    echo "==> Upstream already cloned at ${UPSTREAM_DIR}"
fi

# 2. WebShop env
echo "==> Patching WebShop pins for Python 3.12 compatibility..."
# Upstream WebShop's setup.py / requirements.txt are full of 2022-era exact
# pins (faiss-cpu==1.7.4, spacy==3.4.x → thinc with no py312 wheel, etc.)
# Relax every `pkg==X.Y.Z` to `pkg>=X.Y.Z` so pip picks a compatible wheel.
for f in "${UPSTREAM_DIR}/envs/webshop/setup.py" \
         "${UPSTREAM_DIR}/envs/webshop/requirements.txt"; do
    [[ -f "$f" ]] && sed -i -E \
        's/([a-zA-Z][a-zA-Z0-9_.-]*)==([0-9][0-9a-z.+-]*)/\1>=\2/g' "$f"
done

echo "==> Installing WebShop env (editable)..."
pip install -e "${UPSTREAM_DIR}/envs/webshop"
python -m spacy download en_core_web_lg

if ! command -v java >/dev/null 2>&1; then
    echo "==> Installing OpenJDK 11 via conda..."
    conda install -y -c conda-forge openjdk=11
fi

# 3. WebShop data
WEBSHOP_DIR="${UPSTREAM_DIR}/envs/webshop"
cd "${WEBSHOP_DIR}"

if [[ ! -f "data.zip" ]]; then
    echo "==> Downloading WebShop data.zip..."
    gdown 1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9
fi

if [[ ! -f "indexes.zip" ]]; then
    echo "==> Downloading WebShop indexes.zip..."
    gdown 11zOUDkJSgGhYin9NxQtG8PVpDsika86y
fi

if [[ ! -d "data" ]]; then
    echo "==> Extracting data.zip..."
    unzip -q data.zip
fi

if [[ ! -d "search_index" ]]; then
    echo "==> Extracting indexes.zip..."
    mkdir -p search_index
    unzip -q indexes.zip -d search_index/
fi

# 4. Expert SFT trajectories
cd "${UPSTREAM_DIR}"
if [[ ! -f "data.zip" ]]; then
    echo "==> Downloading expert SFT trajectories..."
    gdown 1_tBMDixZcIjKuv-LExNllha-YIRxhKIq
    unzip -q data.zip
fi

cd "${REPO_ROOT}"
echo "==> Done. Upstream is at ${UPSTREAM_DIR}"
echo "    Next step: bash scripts/apply_qwen3_patches.sh"
