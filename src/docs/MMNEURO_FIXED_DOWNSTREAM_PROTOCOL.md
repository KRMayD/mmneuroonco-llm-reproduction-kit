# MM-NeuroOnco Fixed Downstream Protocol

This document evaluates a **new GMPO upstream checkpoint** in the Brain MRI
MM-NeuroOnco Closed-VQA comparison. The BiomedCLIP baseline, CLIP, and
CLIPrefine checkpoints remain fixed controls; the new GMPO checkpoint is the
only upstream input that changes. Do not use the legacy Brain MRI runner
directly: it contains machine-specific paths and its normalization default is
unsuitable for this protocol.

## Fixed components

| Component | Fixed value |
| --- | --- |
| Training data | `open_closed_nocot`, 163,267 image-instruction-answer rows |
| Evaluation | MM-NeuroOnco Closed-VQA, 3,190 questions / 1,000 images |
| Q-Former initialization | `dmis-lab/biobert-v1.1` |
| Language model | `Qwen/Qwen3-0.6B` |
| Vision encoder behavior | frozen; only the supplied GMPO candidate changes |
| Visual feature normalization | disabled (`NORMALIZE_VISUAL_FEATURES=0`) |
| Precision | BF16 mixed precision |
| Per-model batch size | 8 |
| Epochs | 3 |
| Optimizer schedule | LR 1e-4, min LR 5e-6, warmup 200, linear warmup cosine |
| Weight decay | 0.05 |
| Validation | skipped |
| Evaluation | option-level MC likelihood, not generated-answer parsing |
| Seed | 0 |

The four jobs are independent single-GPU jobs. They use GPUs 0--3 concurrently
by default; this is not four-GPU DDP for one model.

## Required artifacts

1. This exact code snapshot, including `model/`, `dataset/`, `scripts/`, and
   the compatible vendored OpenCLIP source.
2. The HF dataset `gfnnnb/MM-NeuroOnco-Images`, with both image zip archives
   extracted.
3. Local Hugging Face snapshots for `dmis-lab/biobert-v1.1` and
   `Qwen/Qwen3-0.6B`.
4. Three fixed control checkpoints: baseline BiomedCLIP, CLIP post-training,
   and CLIPrefine post-training; plus the newly trained GMPO checkpoint.

Record the filename and SHA256 of the new GMPO checkpoint in
`artifacts/vision_checkpoints.tsv` before launching the experiment. The
historical GMPO SD neg100 SHA256 is retained there as an optional reference.

## Data preparation

Download the dataset and extract the images:

```bash
export MMNEURO_RAW=/data/MM-NeuroOnco-Images
huggingface-cli download gfnnnb/MM-NeuroOnco-Images \
  --repo-type dataset --local-dir "${MMNEURO_RAW}"
unzip -q "${MMNEURO_RAW}/images/Dataset.zip" -d "${MMNEURO_RAW}/images"
unzip -q "${MMNEURO_RAW}/images/Benchmark_Images.zip" -d "${MMNEURO_RAW}/images"
```

Generate machine-local manifests rather than copying manifests from another
server, because every image path is absolute:

```bash
PYTHONNOUSERSITE=1 python scripts/prepare_mmneuro_official_vqa.py \
  --mmneuro-images-root "${MMNEURO_RAW}" \
  --output-dir /data/mm_neuroonco_official \
  --train-variant open_closed_nocot \
  --seed 42
```

Expected counts are `163267` train rows, `3190` Closed-VQA rows, and `1215`
Open-VQA rows. Open-VQA is prepared but is not part of this fixed Closed-VQA
comparison.

## Run

Copy `configs/mmneuro_fixed_protocol.env.example` outside the repository,
replace every path with an absolute path on the target server, then run:

```bash
bash scripts/run_mmneuro_fixed_protocol_4encoders.sh /data/config/mmneuro.env
```

The runner writes one training log, one MC-likelihood log, an LLM checkpoint
path, per-option predictions, per-model summaries, `protocol.json`, and
`metrics_comparison.json` under `RUN_ROOT`.

## Scope boundary

This protocol does not retrain CLIP, CLIPrefine, or GMPO upstream encoders. It
uses supplied checkpoints only. Train the new GMPO upstream model separately,
then provide it through `GMPO_CKPT`. Therefore no Figshare masks, SD-negative
images, DPO CSV, or segmentation pipeline is required for this LLM experiment.
