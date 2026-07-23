# finegrainedVLM: vision-encoder-llm

This repository provides a minimal medical vision-language baseline for:

`raw image -> BiomedCLIP -> path_proj_llm -> Qwen3 + LoRA`

The goal of this repo is practical training and inference with a simple, reviewable multimodal baseline.

What this repo does:
- train from raw images
- use a frozen BiomedCLIP vision encoder
- train a small projection head
- fine-tune Qwen3 with LoRA

What this repo does not do:
- stage1 alignment training
- stage1 checkpoint loading
- global/local alignment losses
- fine-grained region-subcaption alignment

## Model Setup

This baseline uses:
- `BiomedCLIP` frozen
- `path_proj_llm` trainable
- `Qwen3 LoRA` trainable

Expected visual feature size:

`path_input_dim = 512`

`stage1_path` is unsupported.

## Quick Start

1. Prepare a data folder:

```text
your_data_root/
  train_data.json
  test_data.json
```

2. Set your external paths:
- `DATA_PATH`
- `BERT_NAME`
- `LLM_MODEL`
- optional `BIOMEDCLIP_PRETRAINED`
- optional `BIOMEDCLIP_OPEN_CLIP_SRC`

3. Run training:

```bash
cd /path/to/repo

DATA_PATH=/abs/path/to/data_root \
BERT_NAME=/abs/path/to/bert-base-uncased-or-biobert \
LLM_MODEL=/abs/path/to/Qwen3-0.6B \
BIOMEDCLIP_PRETRAINED= \
BIOMEDCLIP_OPEN_CLIP_SRC=/abs/path/to/open_clip/src \
CUDA_VISIBLE_DEVICES=0 \
DEVICES=1 \
BATCH_SIZE=1 \
MAX_EPOCHS=3 \
bash scripts/train_biomedclip_qwen3_baseline.sh
```

4. Run inference:

```bash
cd /path/to/repo

CHECKPOINT_PATH=/abs/path/to/epoch=00.ckpt \
RAW_IMAGE_PATHS=/abs/path/to/example.png \
INSTRUCTION="Describe the medical image in English." \
DEVICE=cuda:0 \
bash scripts/infer_biomedclip_qwen3_baseline.sh
```

If you installed `open_clip` as a package, `BIOMEDCLIP_OPEN_CLIP_SRC` can be left empty.

## Input Data Format

Each sample must be a JSON object with:
- `image`
- `conversations`

The training target is read from:
- `conversations[1]["from"] == "gpt"`
- `conversations[1]["value"]`

### Raw image example

```json
[
  {
    "image": ["/abs/path/to/example.png"],
    "conversations": [
      {"from": "human", "value": "Please describe the medical image in English.\n<image>"},
      {"from": "gpt", "value": "There is a low signal band across the calcaneus with surrounding edema."}
    ]
  }
]
```

If one sample should use multiple images:

```json
[
  {
    "image": [
      "/abs/path/to/view1.png",
      "/abs/path/to/view2.png"
    ],
    "conversations": [
      {"from": "human", "value": "Please describe the medical image in English.\n<image>"},
      {"from": "gpt", "value": "The image shows a heterogeneous mass with surrounding soft tissue edema."}
    ]
  }
]
```

## Input Mode

### Raw image mode

Use this when you want the model to encode images at runtime.

Behavior:
- the code opens image files with `PIL`
- BiomedCLIP extracts visual features on the fly
- multiple images in the same sample are mean-pooled
- projected visual tokens are inserted into the Qwen prompt at `<image>`

Use this mode when:
- you want to compare different BiomedCLIP weights
- you want the published interface to stay simple and image-based

## Selecting BiomedCLIP Weights

Yes, the vision encoder weights are selectable.

The relevant runtime variables are:
- `BIOMEDCLIP_PRETRAINED`
- `BIOMEDCLIP_MODEL_NAME`
- `BIOMEDCLIP_OPEN_CLIP_SRC`

Behavior:
- if `BIOMEDCLIP_PRETRAINED` is set, that local checkpoint is used
- otherwise the code falls back to `BIOMEDCLIP_MODEL_NAME`
- default model name is `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`

Example using a local checkpoint:

```bash
BIOMEDCLIP_PRETRAINED=/abs/path/to/your_biomedclip_weights.bin
```

Example using the default HF model:

```bash
BIOMEDCLIP_PRETRAINED=
```

If your environment does not have `open_clip` installed, set:

```bash
BIOMEDCLIP_OPEN_CLIP_SRC=/abs/path/to/open_clip/src
```

## Training

Main entrypoint:

```bash
scripts/train_biomedclip_qwen3_baseline.sh
```

This script checks:
- `train_data.json` exists
- `test_data.json` exists
- `BERT_NAME` exists
- `LLM_MODEL` exists
- optional BiomedCLIP paths exist if provided

Useful variables:
- `FILENAME`
- `BATCH_SIZE`
- `MAX_EPOCHS`
- `MAX_DATASET_LENGTH`
- `LOG_EVERY_N_STEPS`
- `CAPTION_PROMPT`

Example with a custom training prompt:

```bash
CAPTION_PROMPT="Instruction: Describe the medical image in English.\nInput medical image: <image>.\nResponse: "
```

Training outputs are written to:

```text
all_checkpoints/<filename>_<timestamp>/
```

Typical files:
- `epoch=00.ckpt`
- `lightning_logs/version_0/hparams.yaml`
- `lightning_logs/version_0/metrics.csv`

## Inference

Main entrypoint:

```bash
scripts/infer_biomedclip_qwen3_baseline.sh
```

### Raw image inference

```bash
cd /path/to/repo

CHECKPOINT_PATH=/abs/path/to/epoch=00.ckpt \
RAW_IMAGE_PATHS=/abs/path/to/example.png \
INSTRUCTION="Describe the medical image in English." \
DEVICE=cuda:0 \
bash scripts/infer_biomedclip_qwen3_baseline.sh
```

If one sample should use multiple images:

```bash
RAW_IMAGE_PATHS=/abs/path/to/view1.png,/abs/path/to/view2.png
```

The inference script prints JSON to stdout and can also save it with:

```bash
OUTPUT_PATH=/abs/path/to/prediction.json
```

## Important Notes

- This is a code-only repository. It does not include datasets or model weights.
- Each `image` field should contain one or more raw image paths.
- Raw image files must be readable by `PIL.Image.open(...).convert("RGB")`.
- `BERT_NAME` is still required because compatibility modules are still initialized.
- Multi-GPU and deepspeed code remains for compatibility, but the primary supported path is single-GPU training and inference.

## Optional Example

If you want a worked example for turning MedPix CSV rows into this repo's JSON format, use:

```bash
scripts/prepare_medpix_caption_baseline.py
```

This is only an example data-prep helper. The main training and inference flow of this repo is dataset-agnostic as long as your JSON follows the format above.

## Repository Layout

```text
repo_root/
  model/
  dataset/
  utils/
  scripts/
  README.md
  FILE_MAP.md
  requirements.txt
```
