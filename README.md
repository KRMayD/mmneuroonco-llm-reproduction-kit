# MM-NeuroOnco LLM Fixed-Protocol Reproduction Kit

This repository evaluates a **newly trained GMPO upstream vision encoder** on
MM-NeuroOnco Closed-VQA while keeping every downstream condition fixed. The
BiomedCLIP baseline, CLIP post-training, and CLIPrefine post-training
checkpoints are fixed controls; only the GMPO checkpoint is replaced by the new
candidate. This repository contains no medical images, MM-NeuroOnco
annotations, model weights, training outputs, or GPU runtime libraries.

## What Is Fixed

| Component | Fixed setting |
| --- | --- |
| LLM training data | MM-NeuroOnco `open_closed_nocot`, 163,267 samples |
| Closed-VQA evaluation | 3,190 questions over 1,000 images |
| Q-Former initialization | `dmis-lab/biobert-v1.1` |
| Generative model | `Qwen/Qwen3-0.6B` |
| Vision encoder behavior | frozen; only the supplied GMPO checkpoint is new |
| Visual L2 normalization | disabled |
| Batch size | 8 per independent GPU job |
| Epochs | 3 |
| Optimizer schedule | LR `1e-4`, min LR `5e-6`, warmup 200, linear warmup cosine |
| Weight decay | `0.05` |
| Validation | skipped |
| Evaluation | option-level MC likelihood |
| Seed | `0` |

The four jobs run independently on four GPUs. They are not a four-GPU DDP job
for a single model.

## Repository Layout

```text
src/                         # LLM training/evaluation source snapshot
third_party/open_clip_src/   # compatible OpenCLIP source snapshot
artifacts/vision_checkpoints.tsv
bootstrap_mmneuro_assets.sh  # downloads public raw data and base language models
CODEX_PROMPT_KO.md           # handoff prompt for Codex on a clean server
```

## What You Must Supply

The raw dataset and model weights are intentionally excluded. Download or copy
the following into paths on the target server, then place those absolute paths
in the config file.

| Artifact | Purpose |
| --- | --- |
| `gfnnnb/MM-NeuroOnco-Images` | official train JSONL, benchmark JSON, and images |
| `dmis-lab/biobert-v1.1` | Q-Former initialization |
| `Qwen/Qwen3-0.6B` | language model |
| Baseline BiomedCLIP checkpoint | frozen baseline visual encoder |
| CLIP checkpoint | frozen CLIP post-trained visual encoder |
| CLIPrefine checkpoint | frozen CLIPrefine post-trained visual encoder |
| New GMPO checkpoint | the newly trained GMPO visual encoder to evaluate |

Checkpoint filenames and SHA256 values are listed in
[`artifacts/vision_checkpoints.tsv`](artifacts/vision_checkpoints.tsv). Before
running, replace the `gmpo_candidate` row's placeholder filename and SHA256
with the new GMPO checkpoint. The historical GMPO SD neg100 row is retained
only as an optional reference, not as the default experiment target.

## Quick Start

```bash
# From this repository root on the target server.
bash bootstrap_mmneuro_assets.sh /data/mmneuro_assets

# Prepare absolute-path manifests. Do not copy manifests from another server.
PYTHONNOUSERSITE=1 /path/to/python src/scripts/prepare_mmneuro_official_vqa.py \
  --mmneuro-images-root /data/mmneuro_assets/MM-NeuroOnco-Images \
  --output-dir /data/mmneuro_assets/mm_neuroonco_official \
  --train-variant open_closed_nocot \
  --seed 42

# Fill absolute paths. BASELINE_CKPT, CLIP_CKPT, and CLIPREFINE_CKPT are fixed
# controls; GMPO_CKPT must point to the newly trained GMPO checkpoint.
cp src/configs/mmneuro_fixed_protocol.env.example /data/mmneuro.env

# Edit /data/mmneuro.env, then run all four one-GPU jobs and their evaluations.
bash src/scripts/run_mmneuro_fixed_protocol_4encoders.sh /data/mmneuro.env
```

The runner rejects an incorrect manifest count, a missing image path, a missing
checkpoint, duplicate GPU IDs, or an attempt to enable visual L2 normalization.
It writes `protocol.json`, per-model logs and predictions, and a final
`metrics_comparison.json` below the configured `RUN_ROOT`.

## Environment

Reference versions are in
[`src/requirements-mmneuro-fixed-protocol.txt`](src/requirements-mmneuro-fixed-protocol.txt).
The reference environment used Python 3.12.7 and PyTorch 2.4.1. On the target
server, install a PyTorch build compatible with its existing NVIDIA driver and
CUDA runtime. Do not modify system CUDA, cuDNN, GPU drivers, or an existing
working PyTorch CUDA installation.

## Scope

This kit runs downstream LLM adaptation and Closed-VQA evaluation only. It does
not train the upstream CLIP, CLIPrefine, or GMPO encoders. Train the new GMPO
encoder separately, verify that it is OpenCLIP/BiomedCLIP-compatible, and then
provide it through `GMPO_CKPT`. Figshare masks, SD-negative images, DPO CSVs,
SAM, and segmentation code are not required at this downstream stage.
