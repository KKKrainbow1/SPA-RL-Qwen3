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

# 3. WebShop data + indexes. These live on Google Drive; AutoDL (CN) often
# can't reach it. If gdown fails, manually download and place the zips at
# the paths printed below, then re-run.
WEBSHOP_DIR="${UPSTREAM_DIR}/envs/webshop"
cd "${WEBSHOP_DIR}"

# Use --fuzzy to handle GDrive's redirect/virus-warning interstitials.
download_or_die() {
    local id="$1" out="$2"
    if [[ -f "${out}" ]]; then
        echo "    = ${out} already present"
        return
    fi
    echo "==> Downloading ${out} (gdown id=${id})"
    if ! gdown --fuzzy "https://drive.google.com/uc?id=${id}" -O "${out}"; then
        cat <<EOM
ERROR: gdown failed for ${out}.
Workaround:
  1. On a machine with Google Drive access, download:
       https://drive.google.com/uc?id=${id}
  2. Upload to: ${WEBSHOP_DIR}/${out}
     (via AutoDL's JupyterLab drag-drop, or scp from your laptop)
  3. Re-run: bash scripts/autodl_one_shot_setup.sh
EOM
        exit 1
    fi
}

download_or_die 1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9 data.zip
download_or_die 11zOUDkJSgGhYin9NxQtG8PVpDsika86y indexes.zip

if [[ ! -d "data" ]]; then
    echo "==> Extracting data.zip..."
    unzip -q data.zip
fi

if [[ ! -d "search_index" ]] || [[ -z "$(ls -A search_index 2>/dev/null)" ]]; then
    echo "==> Extracting indexes.zip..."
    mkdir -p search_index
    # -j: junk paths so Lucene files land directly in search_index/, not in
    # search_index/indexes/ (some indexes.zip variants wrap files in a subdir).
    unzip -q -j indexes.zip -d search_index/
fi

# 4. Expert SFT trajectories (training-only). Skip when BASELINE_ONLY=1.
cd "${UPSTREAM_DIR}"
if [[ "${BASELINE_ONLY:-0}" == "1" ]]; then
    echo "==> BASELINE_ONLY=1, skipping expert SFT trajectories"
elif [[ ! -f "data.zip" ]]; then
    echo "==> Downloading expert SFT trajectories..."
    gdown --fuzzy "https://drive.google.com/uc?id=1_tBMDixZcIjKuv-LExNllha-YIRxhKIq" -O data.zip || {
        echo "WARNING: SFT trajectories download failed — needed for training, not baseline."
        echo "         Set BASELINE_ONLY=1 to skip this step entirely."
    }
    [[ -f data.zip ]] && unzip -q data.zip
fi

cd "${REPO_ROOT}"
echo "==> Done. Upstream is at ${UPSTREAM_DIR}"
echo "    Next step: bash scripts/apply_qwen3_patches.sh"
