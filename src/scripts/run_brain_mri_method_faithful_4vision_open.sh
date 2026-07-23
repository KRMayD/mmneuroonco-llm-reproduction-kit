#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

ROOT=/home/msko021220/project/finegrainedVLM-vision-encoder-llm
PY=/home/msko021220/.conda/envs/busi2/bin/python
DATA_ROOT="${ROOT}/data/mm_neuroonco_official"
BERTSCORE_MODEL=/home/msko021220/project/hf_models/dmis-lab_biobert-v1.1

# Latest method-faithful comparison: GMPO uses explicit neg100 captions.
RUN_FAMILY="${RUN_FAMILY:-brain_mri_method_faithful_neg100_4vision_open}"
STAMP="${RUN_STAMP:-$(date +%m%d_%H%M)}"
RUN_ROOT="${ROOT}/outputs/${RUN_FAMILY}_${STAMP}"
LOG_ROOT="${ROOT}/logs/${RUN_FAMILY}_${STAMP}"
STATUS_ROOT="${RUN_ROOT}/status"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${STATUS_ROOT}"

BASELINE_CKPT="${BASELINE_CKPT:-${ROOT}/all_checkpoints/brain_mri_method_faithful_baseline_biomedclip_enc0_down0_0713_0435_seed0_0713_0441/epoch=02.ckpt}"
CLIP_CKPT="${CLIP_CKPT:-${ROOT}/all_checkpoints/brain_mri_method_faithful_clip_posttrained_enc0_down0_0713_0435_seed0_0713_0441/epoch=02.ckpt}"
CLIPREFINE_CKPT="${CLIPREFINE_CKPT:-${ROOT}/all_checkpoints/brain_mri_method_faithful_cliprefine_posttrained_enc0_down0_0713_0435_seed0_0713_0441/epoch=02.ckpt}"
GMPO_CKPT="${GMPO_CKPT:-${ROOT}/all_checkpoints/brain_mri_method_faithful_gmpo_sd_neg100_enc0_down0_0713_0435_seed0_0713_0441/epoch=02.ckpt}"

GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-4}"
BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
LIMIT="${LIMIT:-0}"

for required in "$PY" "$DATA_ROOT/test_data.json" "$BERTSCORE_MODEL/config.json" \
  "$BASELINE_CKPT" "$CLIP_CKPT" "$CLIPREFINE_CKPT" "$GMPO_CKPT"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

printf 'name\tgpu\tllm_checkpoint\tresult_summary\n' > "${RUN_ROOT}/manifest.tsv"

run_one() {
  local name="$1"
  local gpu="$2"
  local checkpoint="$3"
  local output="${RUN_ROOT}/${name}.open_vqa.jsonl"
  local status="${STATUS_ROOT}/${name}.status"
  (
    trap 'printf "FAILED %s UTC\\n" "$(date -u +%FT%TZ)" > "'"${status}"'"' ERR
    printf 'RUNNING %s UTC\n' "$(date -u +%FT%TZ)" > "$status"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "${ROOT}/scripts/eval_mmneuro_open_vqa.py" \
      --checkpoint "$checkpoint" \
      --test-json "${DATA_ROOT}/test_data.json" \
      --output-jsonl "$output" \
      --device cuda:0 \
      --generation-batch-size "$GENERATION_BATCH_SIZE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --bertscore-model "$BERTSCORE_MODEL" \
      --bertscore-num-layers 12 \
      --bertscore-batch-size "$BERTSCORE_BATCH_SIZE" \
      --limit "$LIMIT" \
      > "${LOG_ROOT}/${name}.open_vqa.log" 2>&1
    printf '%s\t%s\t%s\t%s\n' "$name" "$gpu" "$checkpoint" "${output}.summary.json" >> "${RUN_ROOT}/manifest.tsv"
    printf 'COMPLETE %s UTC\n' "$(date -u +%FT%TZ)" > "$status"
  ) &
  echo "$!" > "${STATUS_ROOT}/${name}.pid"
  echo "Launched ${name} on GPU ${gpu}, pid $(cat "${STATUS_ROOT}/${name}.pid")"
}

run_one baseline_biomedclip 0 "$BASELINE_CKPT"
run_one clip_posttrained 1 "$CLIP_CKPT"
run_one cliprefine_posttrained 2 "$CLIPREFINE_CKPT"
run_one gmpo_sd_neg100 3 "$GMPO_CKPT"
wait

"$PY" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
comparison = {}
for name in ("baseline_biomedclip", "clip_posttrained", "cliprefine_posttrained", "gmpo_sd_neg100"):
    path = root / f"{name}.open_vqa.jsonl.summary.json"
    if path.exists():
        payload = json.loads(path.read_text())
        comparison[name] = {
            "overall": payload.get("overall"),
            "by_question_type": payload.get("by_question_type"),
            "summary_path": str(path),
        }
(root / "metrics_comparison.json").write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(comparison, ensure_ascii=False, indent=2))
PY

echo "Open-VQA evaluation complete: ${RUN_ROOT}"
