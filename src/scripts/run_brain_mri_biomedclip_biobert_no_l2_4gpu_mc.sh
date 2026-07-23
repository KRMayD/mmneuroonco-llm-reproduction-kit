#!/usr/bin/env bash
set -euo pipefail

# Train one BiomedCLIP-based MM-NeuroOnco VQA model with BioBERT Q-Former
# initialization and raw (non-L2-normalized) visual features, then run Closed-VQA.
export TRANSFORMERS_NO_SKLEARN=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

ROOT=/home/msko021220/project/finegrainedVLM-vision-encoder-llm
PY=/home/msko021220/.conda/envs/busi2/bin/python
DATA_ROOT="${ROOT}/data/mm_neuroonco_official"
BERT_ROOT=/home/msko021220/project/hf_models/dmis-lab_biobert-v1.1
LLM_ROOT=/home/msko021220/project/hf_models/Qwen_Qwen3-0.6B
OPENCLIP_SRC=/home/msko021220/finegrained-vlm-training/biomedclip_finetuning/open_clip/src
MODEL_NAME=hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
BASELINE_CKPT=/home/msko021220/MedCLIP-SAMv2-dpoloss/saliency_maps/model/openclip_model.pt

STAMP="${RUN_STAMP:-$(date +%m%d_%H%M)}"
RUN_NAME="brain_mri_biomedclip_biobert_no_l2_4gpu_${STAMP}"
RUN_ROOT="${ROOT}/outputs/${RUN_NAME}"
LOG_ROOT="${ROOT}/logs/${RUN_NAME}"
STATUS_PATH="${RUN_ROOT}/status.txt"
TRAIN_LOG="${LOG_ROOT}/train.log"
EVAL_LOG="${LOG_ROOT}/closed_mc_likelihood.log"
PREDICTIONS="${RUN_ROOT}/closed_mc_likelihood.jsonl"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

for required in \
  "${PY}" \
  "${DATA_ROOT}/train_data.json" \
  "${DATA_ROOT}/test_data.json" \
  "${DATA_ROOT}/closed_test_data.json" \
  "${BERT_ROOT}/config.json" \
  "${LLM_ROOT}/config.json" \
  "${OPENCLIP_SRC}" \
  "${BASELINE_CKPT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

cat > "${RUN_ROOT}/config.env" <<EOF
RUN_NAME=${RUN_NAME}
VISION_ENCODER=${MODEL_NAME}
VISION_CHECKPOINT=${BASELINE_CKPT}
QFORMER_BERT=${BERT_ROOT}
NORMALIZE_VISUAL_FEATURES=0
CUDA_VISIBLE_DEVICES=0,1,2,3
DEVICES=4
STRATEGY=ddp
PER_GPU_BATCH_SIZE=2
EFFECTIVE_BATCH_SIZE=8
MAX_EPOCHS=3
SEED=0
EVALUATION=MM-NeuroOnco Closed-VQA MC likelihood
EOF

printf 'TRAINING %s UTC\n' "$(date -u +%FT%TZ)" > "${STATUS_PATH}"
cd "${ROOT}"

PYTHON_BIN="${PY}" \
FILENAME="${RUN_NAME}" \
DATA_PATH="${DATA_ROOT}" \
BERT_NAME="${BERT_ROOT}" \
LLM_MODEL="${LLM_ROOT}" \
BIOMEDCLIP_MODEL_NAME="${MODEL_NAME}" \
BIOMEDCLIP_PRETRAINED="${BASELINE_CKPT}" \
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
SKIP_VALIDATION=1 \
ACCUMULATE_GRAD_BATCHES=1 \
LOG_EVERY_N_STEPS=100 \
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
  bash "${ROOT}/scripts/train_biomedclip_qwen3_baseline.sh" \
  > "${TRAIN_LOG}" 2>&1

CHECKPOINT_DIR=$(find "${ROOT}/all_checkpoints" -mindepth 1 -maxdepth 1 \
  -type d -name "${RUN_NAME}_*" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "No checkpoint directory found for ${RUN_NAME}" >&2
  exit 1
fi

CHECKPOINT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'epoch=*.ckpt' | sort | tail -n 1)
if [[ -z "${CHECKPOINT}" ]]; then
  echo "No epoch checkpoint found in ${CHECKPOINT_DIR}" >&2
  exit 1
fi

printf 'EVALUATING %s UTC\n' "$(date -u +%FT%TZ)" > "${STATUS_PATH}"
CUDA_VISIBLE_DEVICES=0 "${PY}" "${ROOT}/scripts/eval_mmneuro_mc_likelihood.py" \
  --checkpoint "${CHECKPOINT}" \
  --test-json "${DATA_ROOT}/closed_test_data.json" \
  --output-jsonl "${PREDICTIONS}" \
  --device cuda:0 \
  > "${EVAL_LOG}" 2>&1

printf 'COMPLETE %s UTC\n' "$(date -u +%FT%TZ)" > "${STATUS_PATH}"
printf '%s\n' "${CHECKPOINT}" > "${RUN_ROOT}/checkpoint_path.txt"
echo "Complete: ${RUN_ROOT}"
