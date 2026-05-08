# AutoDL Few-shot Baseline Cheat Sheet

Target: Qwen3-8B on WebShop test 200, 1-shot, vLLM OpenAI server.
Hardware: 1×A100 40G. Budget: ~¥5. Wall-clock: ~20–25 min (incl. setup).
Output: `results/qwen3_8b_fewshot/summary.json` (avg_reward, success_rate).

---

## 0. Pick the AutoDL image

When renting the instance, pick a base image with **CUDA 12.1+** and **PyTorch 2.3+**.
The Miniconda + PyTorch 2.3 image works. Avoid bare Ubuntu — you'll spend ¥ on conda setup.

## 1. SSH in, clone the repo

```bash
cd ~/autodl-tmp
git clone https://github.com/<your-user>/SPA-RL-Qwen3.git
cd SPA-RL-Qwen3
```

## 2. One-shot install (≈5–7 min)

```bash
# Python deps
pip install -r requirements.txt
pip install vllm openai gdown spacy

# Upstream + WebShop sim + data + Java + spacy model
bash scripts/setup_data.sh
```

**Watch for:**
- `gdown` and the WebShop `indexes.zip` (~1 GB). If gdown errors out with "virus scan warning" or quota, run `gdown --fuzzy <id>` manually, or upload from your laptop via `scp`.
- `conda install openjdk=11` step. If conda not found, install miniconda first or `apt-get install openjdk-11-jre-headless`.

Sanity check: `ls upstream/envs/webshop/data/items_shuffle.json` should exist.

## 3. Apply Qwen3 patches

```bash
bash scripts/apply_qwen3_patches.sh
```

Verify (optional):
```bash
cd upstream && python -c "from fastchat.model.model_adapter import get_model_adapter; print(get_model_adapter('Qwen/Qwen3-8B'))"
cd ..
```

## 4. Pre-download Qwen3-8B (≈3–5 min, ~17 GB)

Done once, cached. Saves an awkward 5-min silence inside the eval run.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B')"
```

If your AutoDL region is slow on HF, set the mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 5. Run the baseline

```bash
bash scripts/run_fewshot_baseline.sh
```

What this does:
1. Sets `PYTHONPATH=upstream:.:` so `eval_agent.*` and `extensions.*` resolve.
2. Boots vLLM OpenAI server at `localhost:8000` (logs → `results/qwen3_8b_fewshot/logs/vllm.log`).
3. Waits up to 5 min for the `/v1/models` endpoint.
4. Runs `fewshot_eval.py` over 200 WebShop test tasks (k_shot=1).
5. Writes per-task JSON + `summary.json`.
6. Tears down vLLM via trap (now bug-fixed — won't orphan the process if eval crashes).

**Live monitoring** in a second SSH session:
```bash
tail -f results/qwen3_8b_fewshot/logs/vllm.log
nvidia-smi -l 5
```

## 6. Read the result

```bash
python -m json.tool results/qwen3_8b_fewshot/summary.json | head -20
```

Expected keys: `model`, `n_tasks` (=200), `avg_reward`, `success_rate`.

## 7. Commit and shut down

```bash
git add results/qwen3_8b_fewshot/summary.json
git commit -m "results: Qwen3-8B 1-shot WebShop baseline (200 tasks)"
git push
# Then in AutoDL console: shut down the instance to stop billing.
```

Per-task JSON files (`results/qwen3_8b_fewshot/<task_id>.json`) are gitignored as too large; keep on the instance until you're sure you don't need them.

---

## If something breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| `vLLM did not start within 300s` | model still downloading, or OOM | Check `logs/vllm.log` tail. If OOM, lower `--gpu-memory-utilization` to 0.80. |
| `vLLM exited during startup` | invalid `--tool-call-parser` for your vLLM version | `pip show vllm`. If <0.6, upgrade. If parser name mismatch, try `--tool-call-parser qwen` or omit it (fallback path in `fewshot_eval.py` will still work). |
| `success_rate ≈ 0.0` | tool-call parser silently dropped all calls | Check a per-task JSON — if all actions are empty, the fallback path didn't catch it. Check `vllm.log` for parser warnings. |
| `ModuleNotFoundError: eval_agent` | PYTHONPATH not set or upstream not cloned | Re-run `setup_data.sh`; ensure you launched via `scripts/run_fewshot_baseline.sh` (it sets PYTHONPATH). |
| `ModuleNotFoundError: webshop` | `pip install -e upstream/envs/webshop` failed | Re-run that step manually; check `pip list \| grep -i webshop`. |
| Java errors at env init | OpenJDK not installed | `which java` — if missing, `apt-get install openjdk-11-jre-headless` or conda install. |

## Post-baseline expectations

- **If avg_reward ~0.3–0.5 / success_rate ~0.15–0.30** → in line with raw few-shot Qwen3-8B on WebShop. This is your "no training" anchor for the resume.
- **If success_rate < 0.05** → tool-calling probably broken end-to-end. Inspect 2–3 per-task JSONs before investing in another run.
- **The 65.5% number is post-SPA training, not few-shot.** Don't compare directly.
