#!/usr/bin/env bash
set -euo pipefail

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

CLIP_CKPT=/home/msko021220/biomedclip_finetuning/checkpoints/brain_mri_clip_no_figshare_12856_ep3_openclip_0710_1004/final_model.pt
CLIPREFINE_CKPT=/home/msko021220/biomedclip_finetuning/checkpoints/brain_mri_cliprefine_no_figshare_12856_ep3_openclip_0710_1004/final_model.pt
GMPO_CKPT=/home/msko021220/finegrained-vlm-training/outputs/openclip_logs/brain_mri_dpo_sd_baseonly_no_figshare_12856_w00500511_ep3/checkpoints/epoch_3.pt

STAMP="${RUN_STAMP:-$(date +%m%d_%H%M)}"
RUN_ROOT="${ROOT}/outputs/brain_mri_llm_4vision_${STAMP}"
LOG_ROOT="${ROOT}/logs/brain_mri_llm_4vision_${STAMP}"
STATUS_ROOT="${RUN_ROOT}/status"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${STATUS_ROOT}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"

for required in \
  "${PY}" \
  "${DATA_ROOT}/train_data.json" \
  "${DATA_ROOT}/test_data.json" \
  "${DATA_ROOT}/closed_test_data.json" \
  "${BERT_ROOT}/config.json" \
  "${LLM_ROOT}/config.json" \
  "${OPENCLIP_SRC}" \
  "${CLIP_CKPT}" \
  "${CLIPREFINE_CKPT}" \
  "${GMPO_CKPT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

printf 'name\tgpu\tencoder_checkpoint\tllm_checkpoint\tresult_summary\n' > "${RUN_ROOT}/manifest.tsv"

run_one() {
  local name="$1"
  local gpu="$2"
  local encoder_checkpoint="$3"
  local run_name="brain_mri_${name}_llm_${STAMP}"
  local train_log="${LOG_ROOT}/${name}.train.log"
  local eval_log="${LOG_ROOT}/${name}.mc_likelihood.log"
  local prediction_path="${RUN_ROOT}/${name}.closed_mc_likelihood.jsonl"
  local status_path="${STATUS_ROOT}/${name}.status"

  (
    set -euo pipefail
    trap 'printf "FAILED %s UTC\n" "$(date -u +%FT%TZ)" > "'"${status_path}"'"' ERR
    printf 'TRAINING %s UTC\n' "$(date -u +%FT%TZ)" > "${status_path}"

    PYTHON_BIN="${PY}" \
    FILENAME="${run_name}" \
    DATA_PATH="${DATA_ROOT}" \
    BERT_NAME="${BERT_ROOT}" \
    LLM_MODEL="${LLM_ROOT}" \
    BIOMEDCLIP_MODEL_NAME="${MODEL_NAME}" \
    BIOMEDCLIP_PRETRAINED="${encoder_checkpoint}" \
    BIOMEDCLIP_OPEN_CLIP_SRC="${OPENCLIP_SRC}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    DEVICES=1 \
    STRATEGY_NAME=ddp \
    PRECISION=bf16-mixed \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    MAX_EPOCHS="${MAX_EPOCHS}" \
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
      > "${train_log}" 2>&1

    local checkpoint_dir
    checkpoint_dir=$(find "${ROOT}/all_checkpoints" -mindepth 1 -maxdepth 1 \
      -type d -name "${run_name}_*" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
    if [[ -z "${checkpoint_dir}" ]]; then
      echo "No LLM checkpoint directory found for ${run_name}" >&2
      exit 1
    fi

    local llm_checkpoint
    llm_checkpoint=$(find "${checkpoint_dir}" -maxdepth 1 -type f -name 'epoch=*.ckpt' | sort | tail -n 1)
    if [[ -z "${llm_checkpoint}" ]]; then
      echo "No LLM epoch checkpoint found in ${checkpoint_dir}" >&2
      exit 1
    fi

    printf 'EVALUATING %s UTC\n' "$(date -u +%FT%TZ)" > "${status_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${ROOT}/scripts/eval_mmneuro_mc_likelihood.py" \
      --checkpoint "${llm_checkpoint}" \
      --test-json "${DATA_ROOT}/closed_test_data.json" \
      --output-jsonl "${prediction_path}" \
      --device cuda:0 \
      > "${eval_log}" 2>&1

    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${name}" "${gpu}" "${encoder_checkpoint:-hf-hub baseline}" \
      "${llm_checkpoint}" "${prediction_path}.summary.json" \
      >> "${RUN_ROOT}/manifest.tsv"
    printf 'COMPLETE %s UTC\n' "$(date -u +%FT%TZ)" > "${status_path}"
  ) &
  echo $! > "${STATUS_ROOT}/${name}.pid"
  echo "Launched ${name} on physical GPU ${gpu}, pid $(cat "${STATUS_ROOT}/${name}.pid")"
}

cd "${ROOT}"
run_one baseline_biomedclip 0 ""
run_one clip_posttrained 1 "${CLIP_CKPT}"
run_one cliprefine_posttrained 2 "${CLIPREFINE_CKPT}"
run_one gmpo_sd_baseonly 3 "${GMPO_CKPT}"

wait

"${PY}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = [
    "baseline_biomedclip",
    "clip_posttrained",
    "cliprefine_posttrained",
    "gmpo_sd_baseonly",
]
comparison = {}
for model in models:
    path = root / f"{model}.closed_mc_likelihood.jsonl.summary.json"
    if path.exists():
        summary = json.loads(path.read_text())
        comparison[model] = {
            "accuracy": summary.get("accuracy"),
            "correct": summary.get("correct"),
            "total": summary.get("total"),
            "by_question_type": summary.get("by_question_type"),
            "summary_path": str(path),
        }
(root / "metrics_comparison.json").write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(comparison, ensure_ascii=False, indent=2))
PY

echo "All Brain MRI LLM training and MC-likelihood evaluations completed: ${RUN_ROOT}"
