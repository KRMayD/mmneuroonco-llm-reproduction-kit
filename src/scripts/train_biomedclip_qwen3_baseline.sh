#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

FILENAME="${FILENAME:-biomedclip_qwen3_baseline}"
MODE="${MODE:-train}"

DATA_PATH="${DATA_PATH:-/abs/path/to/data_root}"
BERT_NAME="${BERT_NAME:-/abs/path/to/bert-base-uncased-or-biobert}"
LLM_MODEL="${LLM_MODEL:-/abs/path/to/Qwen3-0.6B}"
CAPTION_PROMPT="${CAPTION_PROMPT:-}"

BIOMEDCLIP_MODEL_NAME="${BIOMEDCLIP_MODEL_NAME:-hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"
BIOMEDCLIP_PRETRAINED="${BIOMEDCLIP_PRETRAINED:-}"
BIOMEDCLIP_OPEN_CLIP_SRC="${BIOMEDCLIP_OPEN_CLIP_SRC:-}"
NORMALIZE_VISUAL_FEATURES="${NORMALIZE_VISUAL_FEATURES:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-bf16-mixed}"
STRATEGY_NAME="${STRATEGY_NAME:-deepspeed}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
TEXT_MAX_LEN="${TEXT_MAX_LEN:-384}"
PATH_INPUT_DIM="${PATH_INPUT_DIM:-512}"
EMBED_DIM="${EMBED_DIM:-256}"
NUM_QUERY_TOKEN="${NUM_QUERY_TOKEN:-8}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-12}"
CROSS_ATTENTION_FREQ="${CROSS_ATTENTION_FREQ:-2}"
INIT_LR="${INIT_LR:-1e-4}"
MIN_LR="${MIN_LR:-5e-6}"
WARMUP_LR="${WARMUP_LR:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
SCHEDULER="${SCHEDULER:-linear_warmup_cosine_lr}"
SEED="${SEED:-0}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
STAGE2_PATH="${STAGE2_PATH:-}"
MAX_DATASET_LENGTH="${MAX_DATASET_LENGTH:-}"

if [[ ! -f "${DATA_PATH}/train_data.json" ]]; then
  echo "Missing train file: ${DATA_PATH}/train_data.json" >&2
  exit 1
fi

if [[ ! -f "${DATA_PATH}/test_data.json" ]]; then
  echo "Missing test file: ${DATA_PATH}/test_data.json" >&2
  exit 1
fi

if [[ ! -e "${BERT_NAME}" ]]; then
  echo "Missing BERT path: ${BERT_NAME}" >&2
  exit 1
fi

if [[ ! -e "${LLM_MODEL}" ]]; then
  echo "Missing LLM path: ${LLM_MODEL}" >&2
  exit 1
fi

if [[ -n "${BIOMEDCLIP_PRETRAINED}" && ! -e "${BIOMEDCLIP_PRETRAINED}" ]]; then
  echo "Missing BiomedCLIP pretrained path: ${BIOMEDCLIP_PRETRAINED}" >&2
  exit 1
fi

if [[ -n "${BIOMEDCLIP_OPEN_CLIP_SRC}" && ! -d "${BIOMEDCLIP_OPEN_CLIP_SRC}" ]]; then
  echo "Missing open_clip source path: ${BIOMEDCLIP_OPEN_CLIP_SRC}" >&2
  exit 1
fi

if [[ "${NORMALIZE_VISUAL_FEATURES}" != "0" && "${NORMALIZE_VISUAL_FEATURES}" != "1" ]]; then
  echo "NORMALIZE_VISUAL_FEATURES must be 0 or 1, got: ${NORMALIZE_VISUAL_FEATURES}" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" train_pathflip_finetune.py
  --mode "${MODE}"
  --filename "${FILENAME}"
  --seed "${SEED}"
  --data_path "${DATA_PATH}"
  --bert_name "${BERT_NAME}"
  --llm_model "${LLM_MODEL}"
  --biomedclip_model_name "${BIOMEDCLIP_MODEL_NAME}"
  --biomedclip_open_clip_src "${BIOMEDCLIP_OPEN_CLIP_SRC}"
  --path_input_dim "${PATH_INPUT_DIM}"
  --embed_dim "${EMBED_DIM}"
  --num_query_token "${NUM_QUERY_TOKEN}"
  --num_hidden_layers "${NUM_HIDDEN_LAYERS}"
  --cross_attention_freq "${CROSS_ATTENTION_FREQ}"
  --accelerator "${ACCELERATOR}"
  --devices "${DEVICES}"
  --precision "${PRECISION}"
  --strategy_name "${STRATEGY_NAME}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_epochs "${MAX_EPOCHS}"
  --save_every_n_epochs "${SAVE_EVERY_N_EPOCHS}"
  --check_val_every_n_epoch "${CHECK_VAL_EVERY_N_EPOCH}"
  --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES}"
  --log_every_n_steps "${LOG_EVERY_N_STEPS}"
  --text_max_len "${TEXT_MAX_LEN}"
  --init_lr "${INIT_LR}"
  --min_lr "${MIN_LR}"
  --warmup_lr "${WARMUP_LR}"
  --warmup_steps "${WARMUP_STEPS}"
  --weight_decay "${WEIGHT_DECAY}"
  --scheduler "${SCHEDULER}"
)

if [[ -n "${CAPTION_PROMPT}" ]]; then
  cmd+=(--caption_prompt "${CAPTION_PROMPT}")
fi

if [[ -n "${BIOMEDCLIP_PRETRAINED}" ]]; then
  cmd+=(--biomedclip_pretrained "${BIOMEDCLIP_PRETRAINED}")
fi

if [[ "${NORMALIZE_VISUAL_FEATURES}" == "1" ]]; then
  cmd+=(--normalize_visual_features)
fi

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  cmd+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi

if [[ -n "${STAGE2_PATH}" ]]; then
  cmd+=(--stage2_path "${STAGE2_PATH}")
fi

if [[ -n "${MAX_DATASET_LENGTH}" ]]; then
  cmd+=(--max_dataset_length "${MAX_DATASET_LENGTH}")
fi

if [[ "${SKIP_VALIDATION}" == "1" ]]; then
  cmd+=(--skip_validation)
fi

printf 'Running command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
