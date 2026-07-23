#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/abs/path/to/all_checkpoints/biomedclip_qwen3_baseline_xxxx/epoch=00.ckpt}"
DEVICE="${DEVICE:-cuda:0}"

RAW_IMAGE_PATHS="${RAW_IMAGE_PATHS:-}"
INSTRUCTION="${INSTRUCTION:-}"

DO_SAMPLE="${DO_SAMPLE:-false}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-1}"
NUM_CAPTIONS="${NUM_CAPTIONS:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ -z "${RAW_IMAGE_PATHS}" ]]; then
  echo "You must set RAW_IMAGE_PATHS." >&2
  exit 1
fi

if [[ -n "${RAW_IMAGE_PATHS}" ]]; then
  IFS=',' read -r -a _raw_paths <<< "${RAW_IMAGE_PATHS}"
  for _raw_path in "${_raw_paths[@]}"; do
    _trimmed="${_raw_path#"${_raw_path%%[![:space:]]*}"}"
    _trimmed="${_trimmed%"${_trimmed##*[![:space:]]}"}"
    if [[ -z "${_trimmed}" || ! -f "${_trimmed}" ]]; then
      echo "Missing raw image file: ${_trimmed}" >&2
      exit 1
    fi
  done
fi

export CHECKPOINT_PATH
export DEVICE
export RAW_IMAGE_PATHS
export INSTRUCTION
export DO_SAMPLE
export NUM_BEAMS
export MAX_NEW_TOKENS
export MIN_NEW_TOKENS
export NUM_CAPTIONS
export OUTPUT_PATH

"${PYTHON_BIN}" - <<'PY'
import argparse
import json
import os

import torch

from model.pl_pathflip_finetune import pl_pathflip_finetune


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


checkpoint_path = os.environ["CHECKPOINT_PATH"]
device = os.environ["DEVICE"]
raw_image_paths = os.environ.get("RAW_IMAGE_PATHS", "")
instruction = os.environ.get("INSTRUCTION", "").strip()
do_sample = parse_bool(os.environ.get("DO_SAMPLE", "false"))
num_beams = int(os.environ.get("NUM_BEAMS", "1"))
max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "128"))
min_new_tokens = int(os.environ.get("MIN_NEW_TOKENS", "1"))
num_captions = int(os.environ.get("NUM_CAPTIONS", "1"))
output_path = os.environ.get("OUTPUT_PATH", "").strip()

ckpt = torch.load(checkpoint_path, map_location="cpu")
args = argparse.Namespace(**ckpt["hyper_parameters"])
model = pl_pathflip_finetune.load_from_checkpoint(
    checkpoint_path,
    strict=False,
    args=args,
).to(device).eval()
core_model = model.pathflip_finetune

image_list = [item.strip() for item in raw_image_paths.split(",") if item.strip()]
batch = {"raw_image_paths": [image_list]}

with torch.no_grad():
    if instruction:
        batch["instruction"] = instruction
        predictions = core_model.generate_with_instruction(
            batch,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            num_captions=num_captions,
        )
    else:
        predictions = core_model.generate(
            batch,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_captions=num_captions,
        )

result = {
    "checkpoint_path": checkpoint_path,
    "device": device,
    "input_mode": "raw_image",
    "raw_image_paths": [item.strip() for item in raw_image_paths.split(",") if item.strip()],
    "instruction": instruction,
    "predictions": predictions,
}

print(json.dumps(result, ensure_ascii=False, indent=2))
if output_path:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
PY
