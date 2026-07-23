#!/usr/bin/env bash
set -euo pipefail
export TRANSFORMERS_NO_SKLEARN=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/msko021220/project/finegrainedVLM-vision-encoder-llm
cd "${REPO_ROOT}"

PYTHON_BIN=/home/msko021220/.conda/envs/busi2/bin/python
DATA_PATH=/home/msko021220/project/finegrainedVLM-vision-encoder-llm/data/camobj_official
BERT_NAME=/home/msko021220/project/hf_models/dmis-lab_biobert-v1.1
LLM_MODEL=/home/msko021220/project/hf_models/Qwen_Qwen3-0.6B
OPENCLIP_SRC=/home/msko021220/finegrained-vlm-training/biomedclip_finetuning/open_clip/src
BASELINE_CKPT=/home/msko021220/dataset/clip_reference_checkpoints/openai_clip_vit_b_32_quickgelu_openclip_state_dict.pt
DPO_CKPT=/home/msko021220/finegrained-vlm-training/outputs/openclip_logs/cod10k_clip_dpo_diffusion_bg1to1_basepos_vs_bgonly_w0500_0000_1000_0250_0703_0259/checkpoints/epoch_3.pt
LOG_DIR=/home/msko021220/project/finegrainedVLM-vision-encoder-llm/logs
RUN_STAMP="$(date +%m%d_%H%M)"

wait_for_gpus() {
  local max_mem_mb="${1:-1000}"
  echo "[wait] waiting for GPUs 0,1,2,3 memory <= ${max_mem_mb} MiB"
  while true; do
    local busy
    set +e
    busy=$(/usr/bin/nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v max="${max_mem_mb}" '$1 ~ /^[[:space:]]*[0-3][[:space:]]*$/ {gsub(/ /,"",$2); if ($2+0 > max+0) print $1":"$2}')
    local status=$?
    set -e
    if [[ "${status}" -ne 0 ]]; then
      echo "[wait] nvidia-smi check failed with status ${status}; retrying"
      sleep 120
      continue
    fi
    if [[ -z "${busy}" ]]; then
      echo "[wait] GPUs are free enough"
      break
    fi
    echo "[wait] busy GPUs: ${busy//$'\n'/, }"
    sleep 120
  done
}

run_train() {
  local name="$1"
  local ckpt="$2"
  local log_path="${LOG_DIR}/train_${name}_${RUN_STAMP}.log"
  echo "[train:start] ${name}"
  echo "[train:log] ${log_path}"
  PYTHON_BIN="${PYTHON_BIN}" \
  FILENAME="${name}" \
  DATA_PATH="${DATA_PATH}" \
  BERT_NAME="${BERT_NAME}" \
  LLM_MODEL="${LLM_MODEL}" \
  BIOMEDCLIP_MODEL_NAME="ViT-B-32-quickgelu" \
  BIOMEDCLIP_PRETRAINED="${ckpt}" \
  BIOMEDCLIP_OPEN_CLIP_SRC="${OPENCLIP_SRC}" \
  CUDA_VISIBLE_DEVICES="0,1,2,3" \
  DEVICES="4" \
  STRATEGY_NAME="ddp" \
  PRECISION="bf16-mixed" \
  BATCH_SIZE="2" \
  NUM_WORKERS="8" \
  MAX_EPOCHS="3" \
  SAVE_EVERY_N_EPOCHS="1" \
  CHECK_VAL_EVERY_N_EPOCH="1" \
  ACCUMULATE_GRAD_BATCHES="1" \
  LOG_EVERY_N_STEPS="50" \
  TEXT_MAX_LEN="384" \
  PATH_INPUT_DIM="512" \
  EMBED_DIM="256" \
  NUM_QUERY_TOKEN="8" \
  NUM_HIDDEN_LAYERS="12" \
  CROSS_ATTENTION_FREQ="2" \
  INIT_LR="1e-4" \
  MIN_LR="5e-6" \
  WARMUP_LR="1e-6" \
  WARMUP_STEPS="200" \
  WEIGHT_DECAY="0.05" \
  SCHEDULER="linear_warmup_cosine_lr" \
  SEED="0" \
  bash scripts/train_biomedclip_qwen3_baseline.sh 2>&1 | tee "${log_path}"
  echo "[train:done] ${name}"
}

for required in "${DATA_PATH}/train_data.json" "${DATA_PATH}/test_data.json" "${BERT_NAME}/config.json" "${LLM_MODEL}/config.json" "${BASELINE_CKPT}" "${DPO_CKPT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required path: ${required}" >&2
    exit 1
  fi
done

wait_for_gpus 1000
run_train "camobj_clipref_4gpu_${RUN_STAMP}" "${BASELINE_CKPT}"
wait_for_gpus 1000
run_train "camobj_gmpo_sd_bgonly_4gpu_${RUN_STAMP}" "${DPO_CKPT}"

echo "[all:done] ${RUN_STAMP}"
