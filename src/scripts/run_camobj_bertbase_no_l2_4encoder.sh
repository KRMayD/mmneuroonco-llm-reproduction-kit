#!/usr/bin/env bash
# Train identical CamObj adapters for four frozen COD10K vision encoders, then
# evaluate MM-CamObj Easy/Hard with the same MC-likelihood protocol.
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_SKLEARN=1

REPO_ROOT=/home/msko021220/project/finegrainedVLM-vision-encoder-llm
PYTHON_BIN=/home/msko021220/.conda/envs/busi2/bin/python
DATA_PATH="${REPO_ROOT}/data/camobj_official_grouped_seed42"
BERT_NAME=/home/msko021220/project/hf_models/google-bert_bert-base-uncased
LLM_MODEL=/home/msko021220/project/hf_models/Qwen_Qwen3-0.6B
OPENCLIP_SRC=/home/msko021220/finegrained-vlm-training/biomedclip_finetuning/open_clip/src
MMCAMOBJ_ROOT=/home/msko021220/MM-CamObj
UPSTREAM_ROOT=/home/msko021220/finegrained-vlm-training/outputs/cod10k_mmcamobj_easyhard_leakagefree_upstream_0717_0531
BASELINE_CKPT=/home/msko021220/dataset/clip_reference_checkpoints/openai_clip_vit_b_32_quickgelu_openclip_state_dict.pt
CLIP_CKPT="${UPSTREAM_ROOT}/clip_checkpoints/cod10k_clip_1to1_exclude_mmcamobj_easyhard_2886_0717_0531/final_model.pt"
CLIPREFINE_CKPT="${UPSTREAM_ROOT}/clip_checkpoints/cod10k_cliprefine_1to1_exclude_mmcamobj_easyhard_2886_0717_0531/final_model.pt"
GMPO_CKPT="${UPSTREAM_ROOT}/gmpo_openclip_logs/cod10k_gmpo_sd_bgonly_exclude_mmcamobj_easyhard_2886_0717_0531/checkpoints/epoch_3.pt"
STAMP="${STAMP:-$(date +%m%d_%H%M%S)}"
RUN_ROOT="${REPO_ROOT}/outputs/camobj_bertbase_no_l2_4encoder_${STAMP}"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${RUN_ROOT}/metrics" "${RUN_ROOT}/launch_logs" "${LOG_DIR}"

for required in \
  "${PYTHON_BIN}" "${DATA_PATH}/train_data.json" "${DATA_PATH}/test_data.json" \
  "${BERT_NAME}/config.json" "${BERT_NAME}/model.safetensors" "${LLM_MODEL}/config.json" \
  "${BASELINE_CKPT}" "${CLIP_CKPT}" "${CLIPREFINE_CKPT}" "${GMPO_CKPT}"; do
  [[ -e "${required}" ]] || { echo "missing required path: ${required}" >&2; exit 1; }
done

"${PYTHON_BIN}" - "${DATA_PATH}/summary.json" "${RUN_ROOT}/provenance.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path, output_path = map(Path, sys.argv[1:])
summary = json.loads(summary_path.read_text())
if summary['split_unit'] != 'image_path' or summary['train_val_image_overlap'] != 0:
    raise SystemExit(f'Invalid split: {summary}')
if summary['easy_hard_benchmark_image_overlap'] != 0:
    raise SystemExit(f'Benchmark overlap: {summary["easy_hard_benchmark_image_overlap"]}')
output_path.write_text(json.dumps(summary, indent=2) + '\n')
PY

run_one() {
  local tag="$1"
  local encoder_ckpt="$2"
  local filename="camobj_${tag}_bertbase_no_l2_4gpu_${STAMP}"
  local log_path="${RUN_ROOT}/launch_logs/${tag}.train.log"

  echo "[train:start] ${tag}"
  PYTHON_BIN="${PYTHON_BIN}" \
  FILENAME="${filename}" \
  DATA_PATH="${DATA_PATH}" \
  BERT_NAME="${BERT_NAME}" \
  LLM_MODEL="${LLM_MODEL}" \
  BIOMEDCLIP_MODEL_NAME="ViT-B-32-quickgelu" \
  BIOMEDCLIP_PRETRAINED="${encoder_ckpt}" \
  BIOMEDCLIP_OPEN_CLIP_SRC="${OPENCLIP_SRC}" \
  NORMALIZE_VISUAL_FEATURES=0 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  DEVICES=4 \
  STRATEGY_NAME=ddp \
  PRECISION=bf16-mixed \
  BATCH_SIZE=2 \
  NUM_WORKERS=8 \
  MAX_EPOCHS=3 \
  SAVE_EVERY_N_EPOCHS=1 \
  CHECK_VAL_EVERY_N_EPOCH=1 \
  ACCUMULATE_GRAD_BATCHES=1 \
  LOG_EVERY_N_STEPS=50 \
  TEXT_MAX_LEN=384 \
  PATH_INPUT_DIM=512 \
  EMBED_DIM=256 \
  NUM_QUERY_TOKEN=8 \
  NUM_HIDDEN_LAYERS=12 \
  CROSS_ATTENTION_FREQ=2 \
  INIT_LR=1e-4 \
  MIN_LR=5e-6 \
  WARMUP_LR=1e-6 \
  WARMUP_STEPS=200 \
  WEIGHT_DECAY=0.05 \
  SCHEDULER=linear_warmup_cosine_lr \
  SEED=0 \
  bash "${REPO_ROOT}/scripts/train_biomedclip_qwen3_baseline.sh" >"${log_path}" 2>&1

  local checkpoint_dir
  checkpoint_dir=$(find "${REPO_ROOT}/all_checkpoints" -maxdepth 1 -type d -name "${filename}_*" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
  local checkpoint="${checkpoint_dir}/epoch=02.ckpt"
  [[ -f "${checkpoint}" ]] || { echo "missing epoch-3 checkpoint: ${checkpoint}" >&2; exit 1; }
  printf '%s\n' "${checkpoint}" >"${RUN_ROOT}/${tag}.checkpoint.txt"

  for task in easy_vqa hard_vqa; do
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_camobj_mc_likelihood.py" \
      --checkpoint "${checkpoint}" \
      --questions "${MMCAMOBJ_ROOT}/dataset/questions/${task}.jsonl" \
      --dataset-root "${MMCAMOBJ_ROOT}/dataset" \
      --output-jsonl "${RUN_ROOT}/metrics/${tag}_${task}.jsonl" \
      --device cuda:0 \
      >"${RUN_ROOT}/launch_logs/${tag}.${task}.eval.log" 2>&1
  done
  echo "[train+eval:done] ${tag}"
}

run_one baseline "${BASELINE_CKPT}"
run_one clip "${CLIP_CKPT}"
run_one cliprefine "${CLIPREFINE_CKPT}"
run_one gmpo "${GMPO_CKPT}"

"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for model in ('baseline', 'clip', 'cliprefine', 'gmpo'):
    summary[model] = {}
    for task in ('easy_vqa', 'hard_vqa'):
        source = Path(str(root / 'metrics' / f'{model}_{task}.jsonl') + '.summary.json')
        summary[model][task] = json.loads(source.read_text())
(root / 'metrics_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
PY

echo "[all:done] ${RUN_ROOT}"
