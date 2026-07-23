#!/usr/bin/env bash
# Train and evaluate four frozen upstream vision encoders under one fixed
# MM-NeuroOnco downstream protocol. The config file supplies paths only.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_mmneuro_fixed_protocol_4encoders.sh /abs/path/mmneuro.env

Create the config from configs/mmneuro_fixed_protocol.env.example. The runner
requires pre-generated train_data.json, test_data.json, and closed_test_data.json.
EOF
}

if [[ $# -ne 1 || ! -f "$1" ]]; then
  usage >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$1"

for variable in \
  REPO_ROOT PYTHON_BIN DATA_ROOT BERT_ROOT LLM_ROOT OPENCLIP_SRC MODEL_NAME \
  BASELINE_CKPT CLIP_CKPT CLIPREFINE_CKPT GMPO_CKPT GMPO_VARIANT GPU_IDS RUN_ROOT; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required config variable: ${variable}" >&2
    exit 2
  fi
done

DOWNSTREAM_SEED="${DOWNSTREAM_SEED:-0}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${REPO_ROOT}/all_checkpoints}"

# Fixed downstream protocol. Do not expose these as config overrides: only the
# upstream checkpoint is allowed to differ between the four jobs.
export TRANSFORMERS_NO_SKLEARN=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NORMALIZE_VISUAL_FEATURES=0

readonly BATCH_SIZE=8
readonly MAX_EPOCHS=3
readonly NUM_WORKERS=8
readonly TEXT_MAX_LEN=384
readonly PATH_INPUT_DIM=512
readonly EMBED_DIM=256
readonly NUM_QUERY_TOKEN=8
readonly NUM_HIDDEN_LAYERS=12
readonly CROSS_ATTENTION_FREQ=2
readonly INIT_LR=1e-4
readonly MIN_LR=5e-6
readonly WARMUP_LR=1e-6
readonly WARMUP_STEPS=200
readonly WEIGHT_DECAY=0.05
readonly SCHEDULER=linear_warmup_cosine_lr
readonly PRECISION=bf16-mixed
readonly EXPECTED_TRAIN_ROWS=163267
readonly EXPECTED_CLOSED_ROWS=3190

for required in \
  "${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/train_biomedclip_qwen3_baseline.sh" \
  "${REPO_ROOT}/scripts/eval_mmneuro_mc_likelihood.py" \
  "${DATA_ROOT}/train_data.json" \
  "${DATA_ROOT}/test_data.json" \
  "${DATA_ROOT}/closed_test_data.json" \
  "${BERT_ROOT}/config.json" \
  "${LLM_ROOT}/config.json" \
  "${OPENCLIP_SRC}" \
  "${BASELINE_CKPT}" \
  "${CLIP_CKPT}" \
  "${CLIPREFINE_CKPT}" \
  "${GMPO_CKPT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ ${#GPU_ARRAY[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated physical GPU IDs." >&2
  exit 2
fi
if [[ "${GPU_ARRAY[0]}" == "${GPU_ARRAY[1]}" || "${GPU_ARRAY[0]}" == "${GPU_ARRAY[2]}" || \
      "${GPU_ARRAY[0]}" == "${GPU_ARRAY[3]}" || "${GPU_ARRAY[1]}" == "${GPU_ARRAY[2]}" || \
      "${GPU_ARRAY[1]}" == "${GPU_ARRAY[3]}" || "${GPU_ARRAY[2]}" == "${GPU_ARRAY[3]}" ]]; then
  echo "GPU_IDS must not contain duplicate GPU IDs." >&2
  exit 2
fi

"${PYTHON_BIN}" - "${DATA_ROOT}" "${EXPECTED_TRAIN_ROWS}" "${EXPECTED_CLOSED_ROWS}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_train = int(sys.argv[2])
expected_closed = int(sys.argv[3])

train = json.loads((root / "train_data.json").read_text(encoding="utf-8"))
closed = json.loads((root / "closed_test_data.json").read_text(encoding="utf-8"))
if len(train) != expected_train:
    raise SystemExit(f"Expected {expected_train} train rows, found {len(train)}")
if len(closed) != expected_closed:
    raise SystemExit(f"Expected {expected_closed} Closed-VQA rows, found {len(closed)}")
if any((item.get("meta") or {}).get("source") != "MM-NeuroOnco_Benchmark_Closed" for item in closed):
    raise SystemExit("closed_test_data.json contains a non-Closed-VQA sample")
for name, rows in (("train", train), ("closed", closed)):
    missing = sum(
        1
        for item in rows
        for image in (item.get("image") or [])
        if not Path(image).is_file()
    )
    if missing:
        raise SystemExit(f"{name} manifest has {missing} missing image paths")
print(f"Preflight passed: train={len(train)}, closed={len(closed)}")
PY

RUN_DIR="${RUN_ROOT}/mmneuro_fixed_protocol_seed${DOWNSTREAM_SEED}_${RUN_STAMP}"
LOG_DIR="${RUN_DIR}/logs"
STATUS_DIR="${RUN_DIR}/status"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}" "${CHECKPOINTS_ROOT}"

"${PYTHON_BIN}" - "${RUN_DIR}/protocol.json" <<PY
import json
from pathlib import Path

protocol = {
    "study": "MM-NeuroOnco fixed downstream comparison",
    "downstream_seed": int("${DOWNSTREAM_SEED}"),
    "visual_l2_normalization": False,
    "vision_encoder_frozen": True,
    "qformer_initialization": "${BERT_ROOT}",
    "llm": "${LLM_ROOT}",
    "batch_size_per_gpu": ${BATCH_SIZE},
    "epochs": ${MAX_EPOCHS},
    "precision": "${PRECISION}",
    "learning_rate": ${INIT_LR},
    "min_lr": ${MIN_LR},
    "warmup_steps": ${WARMUP_STEPS},
    "weight_decay": ${WEIGHT_DECAY},
    "scheduler": "${SCHEDULER}",
    "validation": "skipped",
    "evaluation": "Closed-VQA MC likelihood",
    "expected_train_rows": ${EXPECTED_TRAIN_ROWS},
    "expected_closed_rows": ${EXPECTED_CLOSED_ROWS},
    "encoders": {
        "baseline_biomedclip": "${BASELINE_CKPT}",
        "clip_posttrained": "${CLIP_CKPT}",
        "cliprefine_posttrained": "${CLIPREFINE_CKPT}",
        "${GMPO_VARIANT}": "${GMPO_CKPT}",
    },
}
Path("${RUN_DIR}/protocol.json").write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
)
PY

run_one() {
  local name="$1"
  local gpu="$2"
  local encoder_checkpoint="$3"
  local run_name="mmneuro_fixed_${name}_seed${DOWNSTREAM_SEED}_${RUN_STAMP}"
  local train_log="${LOG_DIR}/${name}.train.log"
  local eval_log="${LOG_DIR}/${name}.mc_likelihood.log"
  local status_path="${STATUS_DIR}/${name}.status"
  local prediction_path="${RUN_DIR}/${name}.closed_mc_likelihood.jsonl"

  (
    set -euo pipefail
    trap 'printf "FAILED %s UTC\\n" "$(date -u +%FT%TZ)" > "'"${status_path}"'"' ERR
    printf 'TRAINING %s UTC\n' "$(date -u +%FT%TZ)" > "${status_path}"

    PYTHON_BIN="${PYTHON_BIN}" \
    FILENAME="${run_name}" \
    DATA_PATH="${DATA_ROOT}" \
    BERT_NAME="${BERT_ROOT}" \
    LLM_MODEL="${LLM_ROOT}" \
    BIOMEDCLIP_MODEL_NAME="${MODEL_NAME}" \
    BIOMEDCLIP_PRETRAINED="${encoder_checkpoint}" \
    BIOMEDCLIP_OPEN_CLIP_SRC="${OPENCLIP_SRC}" \
    NORMALIZE_VISUAL_FEATURES=0 \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    DEVICES=1 \
    STRATEGY_NAME=ddp \
    PRECISION="${PRECISION}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    MAX_EPOCHS="${MAX_EPOCHS}" \
    SAVE_EVERY_N_EPOCHS=1 \
    CHECK_VAL_EVERY_N_EPOCH=1 \
    SKIP_VALIDATION=1 \
    ACCUMULATE_GRAD_BATCHES=1 \
    LOG_EVERY_N_STEPS=100 \
    TEXT_MAX_LEN="${TEXT_MAX_LEN}" \
    PATH_INPUT_DIM="${PATH_INPUT_DIM}" \
    EMBED_DIM="${EMBED_DIM}" \
    NUM_QUERY_TOKEN="${NUM_QUERY_TOKEN}" \
    NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS}" \
    CROSS_ATTENTION_FREQ="${CROSS_ATTENTION_FREQ}" \
    INIT_LR="${INIT_LR}" \
    MIN_LR="${MIN_LR}" \
    WARMUP_LR="${WARMUP_LR}" \
    WARMUP_STEPS="${WARMUP_STEPS}" \
    WEIGHT_DECAY="${WEIGHT_DECAY}" \
    SCHEDULER="${SCHEDULER}" \
    SEED="${DOWNSTREAM_SEED}" \
      bash "${REPO_ROOT}/scripts/train_biomedclip_qwen3_baseline.sh" \
      > "${train_log}" 2>&1

    local checkpoint_dir
    checkpoint_dir=$(find "${CHECKPOINTS_ROOT}" -mindepth 1 -maxdepth 1 \
      -type d -name "${run_name}_*" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
    if [[ -z "${checkpoint_dir}" ]]; then
      echo "No LLM checkpoint directory found for ${name}" >&2
      exit 1
    fi

    local llm_checkpoint
    llm_checkpoint=$(find "${checkpoint_dir}" -maxdepth 1 -type f -name 'epoch=*.ckpt' | sort | tail -n 1)
    if [[ -z "${llm_checkpoint}" ]]; then
      echo "No LLM epoch checkpoint found for ${name}" >&2
      exit 1
    fi

    printf 'EVALUATING %s UTC\n' "$(date -u +%FT%TZ)" > "${status_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      "${REPO_ROOT}/scripts/eval_mmneuro_mc_likelihood.py" \
      --checkpoint "${llm_checkpoint}" \
      --test-json "${DATA_ROOT}/closed_test_data.json" \
      --output-jsonl "${prediction_path}" \
      --device cuda:0 \
      --limit 0 \
      > "${eval_log}" 2>&1

    printf '%s\n' "${llm_checkpoint}" > "${STATUS_DIR}/${name}.checkpoint"
    printf 'COMPLETE %s UTC\n' "$(date -u +%FT%TZ)" > "${status_path}"
  ) &
  LAST_PID="$!"
}

cd "${REPO_ROOT}"
pids=()
LAST_PID=""
run_one baseline_biomedclip "${GPU_ARRAY[0]}" "${BASELINE_CKPT}"
pids+=("${LAST_PID}")
run_one clip_posttrained "${GPU_ARRAY[1]}" "${CLIP_CKPT}"
pids+=("${LAST_PID}")
run_one cliprefine_posttrained "${GPU_ARRAY[2]}" "${CLIPREFINE_CKPT}"
pids+=("${LAST_PID}")
run_one "${GMPO_VARIANT}" "${GPU_ARRAY[3]}" "${GMPO_CKPT}"
pids+=("${LAST_PID}")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

"${PYTHON_BIN}" - "${RUN_DIR}" "${GMPO_VARIANT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
gmpo_variant = sys.argv[2]
models = [
    "baseline_biomedclip",
    "clip_posttrained",
    "cliprefine_posttrained",
    gmpo_variant,
]
comparison = {}
for model in models:
    summary_path = root / f"{model}.closed_mc_likelihood.jsonl.summary.json"
    checkpoint_path = root / "status" / f"{model}.checkpoint"
    status_path = root / "status" / f"{model}.status"
    entry = {"status": status_path.read_text().strip() if status_path.exists() else "MISSING"}
    if checkpoint_path.exists():
        entry["llm_checkpoint"] = checkpoint_path.read_text().strip()
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        entry.update({
            "correct": summary.get("correct"),
            "total": summary.get("total"),
            "accuracy": summary.get("accuracy"),
            "by_question_type": summary.get("by_question_type"),
            "summary_path": str(summary_path),
        })
    comparison[model] = entry
(root / "metrics_comparison.json").write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(comparison, ensure_ascii=False, indent=2))
PY

if [[ ${failed} -ne 0 ]]; then
  echo "At least one model failed. Inspect ${STATUS_DIR} and ${LOG_DIR}." >&2
  exit 1
fi

echo "Completed fixed-protocol training and Closed-VQA evaluation: ${RUN_DIR}"
