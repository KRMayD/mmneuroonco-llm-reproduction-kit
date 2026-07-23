#!/usr/bin/env bash
set -euo pipefail

# Compare the three non-baseline vision encoders under the same downstream
# condition as the BiomedCLIP/BioBERT/no-L2 DDP run. The models are sequential
# because each training job occupies all four GPUs.
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
ENCODER_ROOT="${ENCODER_ROOT:-/home/msko021220/finegrained-vlm-training/outputs/method_faithful_brain_mri_v1}"
GMPO_NEG100_ROOT="${GMPO_NEG100_ROOT:-/home/msko021220/finegrained-vlm-training/outputs/method_faithful_brain_mri_neg100_v1}"

CLIP_CKPT="${ENCODER_ROOT}/clip_checkpoints/brain_mri_clip_method_faithful_seed0/final_model.pt"
CLIPREFINE_CKPT="${ENCODER_ROOT}/clip_checkpoints/brain_mri_cliprefine_method_faithful_seed0/final_model.pt"
GMPO_CKPT="${GMPO_NEG100_ROOT}/gmpo_openclip_logs/brain_mri_gmpo_method_faithful_seed0/checkpoints/epoch_3.pt"

STAMP="${RUN_STAMP:-$(date +%m%d_%H%M)}"
RUN_FAMILY="${RUN_FAMILY:-brain_mri_remaining3_biobert_no_l2_4gpu}"
RUN_ROOT="${ROOT}/outputs/${RUN_FAMILY}_${STAMP}"
LOG_ROOT="${ROOT}/logs/${RUN_FAMILY}_${STAMP}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${RUN_ROOT}/status"

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

cat > "${RUN_ROOT}/config.env" <<EOF
VISION_ENCODER=${MODEL_NAME}
QFORMER_BERT=${BERT_ROOT}
NORMALIZE_VISUAL_FEATURES=0
UPSTREAM_FEATURE_NORMALIZATION=${UPSTREAM_FEATURE_NORMALIZATION:-unknown}
CUDA_VISIBLE_DEVICES=0,1,2,3
DEVICES=4
STRATEGY=ddp
PER_GPU_BATCH_SIZE=2
EFFECTIVE_BATCH_SIZE=8
MAX_EPOCHS=3
SEED=0
EVALUATION=MM-NeuroOnco Closed-VQA MC likelihood
CLIP_CHECKPOINT=${CLIP_CKPT}
CLIPREFINE_CHECKPOINT=${CLIPREFINE_CKPT}
GMPO_CHECKPOINT=${GMPO_CKPT}
EOF

printf 'name\tvision_checkpoint\tllm_checkpoint\taccuracy\tcorrect\ttotal\n' > "${RUN_ROOT}/manifest.tsv"

run_one() {
  local name="$1"
  local vision_checkpoint="$2"
  local run_name="brain_mri_${name}_biobert_no_l2_4gpu_${STAMP}"
  local model_root="${RUN_ROOT}/${name}"
  local train_log="${LOG_ROOT}/${name}.train.log"
  local eval_log="${LOG_ROOT}/${name}.closed_mc_likelihood.log"
  local prediction_path="${model_root}/closed_mc_likelihood.jsonl"
  local status_path="${RUN_ROOT}/status/${name}.txt"
  mkdir -p "${model_root}"

  printf 'TRAINING %s UTC\n' "$(date -u +%FT%TZ)" > "${status_path}"
  PYTHON_BIN="${PY}" \
  FILENAME="${run_name}" \
  DATA_PATH="${DATA_ROOT}" \
  BERT_NAME="${BERT_ROOT}" \
  LLM_MODEL="${LLM_ROOT}" \
  BIOMEDCLIP_MODEL_NAME="${MODEL_NAME}" \
  BIOMEDCLIP_PRETRAINED="${vision_checkpoint}" \
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
    bash "${ROOT}/scripts/train_biomedclip_qwen3_baseline.sh" > "${train_log}" 2>&1

  local checkpoint_dir
  checkpoint_dir=$(find "${ROOT}/all_checkpoints" -mindepth 1 -maxdepth 1 \
    -type d -name "${run_name}_*" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  if [[ -z "${checkpoint_dir}" ]]; then
    echo "No LLM checkpoint directory found for ${run_name}" >&2
    return 1
  fi

  local checkpoint
  checkpoint=$(find "${checkpoint_dir}" -maxdepth 1 -type f -name 'epoch=*.ckpt' | sort | tail -n 1)
  if [[ -z "${checkpoint}" ]]; then
    echo "No epoch checkpoint found in ${checkpoint_dir}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}" > "${model_root}/checkpoint_path.txt"

  printf 'EVALUATING %s UTC\n' "$(date -u +%FT%TZ)" > "${status_path}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" "${ROOT}/scripts/eval_mmneuro_mc_likelihood.py" \
    --checkpoint "${checkpoint}" \
    --test-json "${DATA_ROOT}/closed_test_data.json" \
    --output-jsonl "${prediction_path}" \
    --device cuda:0 > "${eval_log}" 2>&1

  "${PY}" - "${prediction_path}.summary.json" "${name}" "${vision_checkpoint}" "${checkpoint}" \
    >> "${RUN_ROOT}/manifest.tsv" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print("\t".join([
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    str(summary["accuracy"]),
    str(summary["correct"]),
    str(summary["total"]),
]))
PY
  printf 'COMPLETE %s UTC\n' "$(date -u +%FT%TZ)" > "${status_path}"
}

cd "${ROOT}"
run_one clip_posttrained "${CLIP_CKPT}"
run_one cliprefine_posttrained "${CLIPREFINE_CKPT}"
run_one gmpo_sd_neg100 "${GMPO_CKPT}"

"${PY}" - "${RUN_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
comparison = {}
with (root / "manifest.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        comparison[row["name"]] = {
            "vision_checkpoint": row["vision_checkpoint"],
            "llm_checkpoint": row["llm_checkpoint"],
            "accuracy": float(row["accuracy"]),
            "correct": int(row["correct"]),
            "total": int(row["total"]),
        }
(root / "metrics_comparison.json").write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(comparison, ensure_ascii=False, indent=2))
PY

echo "Complete: ${RUN_ROOT}"
