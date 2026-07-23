#!/usr/bin/env bash
# Train and evaluate one CAM model using the explicit-negative GMPO encoder.
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
GMPO_CKPT=/home/msko021220/finegrained-vlm-training/outputs/cod10k_gmpo_explicitneg_mmcamobj_easyhard_2886_0718_094623/gmpo_openclip_logs/cod10k_gmpo_sd_bgexplicitneg_exclude_mmcamobj_easyhard_2886_0718_094623/checkpoints/epoch_3.pt
STAMP="${STAMP:-$(date +%m%d_%H%M%S)}"
RUN_ROOT="${REPO_ROOT}/outputs/camobj_gmpo_explicitneg_bertbase_no_l2_4gpu_${STAMP}"
FILENAME="camobj_gmpo_explicitneg_bertbase_no_l2_4gpu_${STAMP}"

mkdir -p "${RUN_ROOT}/metrics" "${RUN_ROOT}/launch_logs"

for required in \
  "${PYTHON_BIN}" "${DATA_PATH}/train_data.json" "${DATA_PATH}/test_data.json" \
  "${BERT_NAME}/config.json" "${BERT_NAME}/model.safetensors" "${LLM_MODEL}/config.json" \
  "${GMPO_CKPT}" "${MMCAMOBJ_ROOT}/dataset/questions/easy_vqa.jsonl" \
  "${MMCAMOBJ_ROOT}/dataset/questions/hard_vqa.jsonl"; do
  [[ -e "${required}" ]] || { echo "missing required path: ${required}" >&2; exit 1; }
done

# Reuse the prior image-level split after verifying that it excludes Easy/Hard images.
"${PYTHON_BIN}" - "${DATA_PATH}/summary.json" "${RUN_ROOT}/provenance.json" "${GMPO_CKPT}" <<'PY'
import json
import sys
from pathlib import Path

summary_path, output_path, checkpoint = map(Path, sys.argv[1:])
summary = json.loads(summary_path.read_text())
if summary['split_unit'] != 'image_path' or summary['train_val_image_overlap'] != 0:
    raise SystemExit(f'Invalid split: {summary}')
if summary['easy_hard_benchmark_image_overlap'] != 0:
    raise SystemExit(f'Benchmark overlap: {summary["easy_hard_benchmark_image_overlap"]}')
summary.update({
    'vision_encoder_tag': 'gmpo_explicit_negative',
    'vision_encoder_checkpoint': str(checkpoint),
    'normalize_visual_features': False,
    'evaluation': 'MM-CamObj Easy/Hard MC likelihood',
})
output_path.write_text(json.dumps(summary, indent=2) + '\n')
PY

TRAIN_LOG="${RUN_ROOT}/launch_logs/gmpo_explicitneg.train.log"
PYTHON_BIN="${PYTHON_BIN}" \
FILENAME="${FILENAME}" \
DATA_PATH="${DATA_PATH}" \
BERT_NAME="${BERT_NAME}" \
LLM_MODEL="${LLM_MODEL}" \
BIOMEDCLIP_MODEL_NAME="ViT-B-32-quickgelu" \
BIOMEDCLIP_PRETRAINED="${GMPO_CKPT}" \
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
bash "${REPO_ROOT}/scripts/train_biomedclip_qwen3_baseline.sh" >"${TRAIN_LOG}" 2>&1

CHECKPOINT_DIR=$(find "${REPO_ROOT}/all_checkpoints" -maxdepth 1 -type d -name "${FILENAME}_*" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
CHECKPOINT="${CHECKPOINT_DIR}/epoch=02.ckpt"
[[ -f "${CHECKPOINT}" ]] || { echo "missing epoch-3 checkpoint: ${CHECKPOINT}" >&2; exit 1; }
printf '%s\n' "${CHECKPOINT}" >"${RUN_ROOT}/gmpo_explicitneg.checkpoint.txt"

for task in easy_vqa hard_vqa; do
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_camobj_mc_likelihood.py" \
    --checkpoint "${CHECKPOINT}" \
    --questions "${MMCAMOBJ_ROOT}/dataset/questions/${task}.jsonl" \
    --dataset-root "${MMCAMOBJ_ROOT}/dataset" \
    --output-jsonl "${RUN_ROOT}/metrics/gmpo_explicitneg_${task}.jsonl" \
    --device cuda:0 \
    >"${RUN_ROOT}/launch_logs/gmpo_explicitneg.${task}.eval.log" 2>&1
done

"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for task in ('easy_vqa', 'hard_vqa'):
    path = Path(str(root / 'metrics' / f'gmpo_explicitneg_{task}.jsonl') + '.summary.json')
    summary[task] = json.loads(path.read_text())
(root / 'metrics_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY

echo "[all:done] ${RUN_ROOT}"
