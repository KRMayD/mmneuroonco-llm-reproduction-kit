#!/usr/bin/env bash
# Download only public raw MM-NeuroOnco data and language-model snapshots.
# Upstream vision checkpoints remain explicit user-supplied experiment assets.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash bootstrap_mmneuro_assets.sh /abs/path/asset_root" >&2
  exit 2
fi

ASSET_ROOT="$(realpath -m "$1")"
RAW_ROOT="${ASSET_ROOT}/MM-NeuroOnco-Images"
MODEL_ROOT="${ASSET_ROOT}/models"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required. Install huggingface-hub in the project environment." >&2
  exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required to extract the official image archives." >&2
  exit 1
fi

mkdir -p "${RAW_ROOT}" "${MODEL_ROOT}"

huggingface-cli download gfnnnb/MM-NeuroOnco-Images \
  --repo-type dataset \
  --local-dir "${RAW_ROOT}"

unzip -q -n "${RAW_ROOT}/images/Dataset.zip" -d "${RAW_ROOT}/images"
unzip -q -n "${RAW_ROOT}/images/Benchmark_Images.zip" -d "${RAW_ROOT}/images"

huggingface-cli download dmis-lab/biobert-v1.1 \
  --local-dir "${MODEL_ROOT}/dmis-lab_biobert-v1.1"
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir "${MODEL_ROOT}/Qwen_Qwen3-0.6B"

for required in \
  "${RAW_ROOT}/training/train_open.jsonl" \
  "${RAW_ROOT}/training/train_no_cot_closed.jsonl" \
  "${RAW_ROOT}/Benchmark/Benchmark_VQA_Closed.json" \
  "${RAW_ROOT}/Benchmark/Benchmark_VQA_Open.json" \
  "${RAW_ROOT}/images/Dataset/Dataset" \
  "${RAW_ROOT}/images/Benchmark_Images/Benchmark_Images" \
  "${MODEL_ROOT}/dmis-lab_biobert-v1.1/config.json" \
  "${MODEL_ROOT}/Qwen_Qwen3-0.6B/config.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "Asset bootstrap incomplete; missing ${required}" >&2
    exit 1
  fi
done

cat <<EOF
Assets are ready under: ${ASSET_ROOT}

Next steps:
1. Supply the four vision checkpoints listed in artifacts/vision_checkpoints.tsv.
2. Generate local manifests with src/scripts/prepare_mmneuro_official_vqa.py.
3. Copy src/configs/mmneuro_fixed_protocol.env.example and set absolute paths.
4. Run src/scripts/run_mmneuro_fixed_protocol_4encoders.sh.
EOF
