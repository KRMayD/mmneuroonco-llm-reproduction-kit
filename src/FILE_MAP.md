# File Map

This repository intentionally includes only the files needed for training and inference of the minimal baseline.

## Included

### User-facing entrypoints

- `scripts/train_biomedclip_qwen3_baseline.sh`
- `scripts/infer_biomedclip_qwen3_baseline.sh`

### Model code

- `model/`

### Dataset and training utilities

- `dataset/`
- `utils/`

### Optional helper

- `scripts/prepare_medpix_caption_baseline.py`

## Excluded

### Alignment-stage code not needed for this baseline

- alignment-stage training entrypoints
- alignment-stage model wrappers

### Post-processing utilities

- Excel export scripts
- translation scripts
- training report HTML helper

### Repository-only materials

- checkpoints
- datasets
- docs and assets unrelated to training/inference

## Why Some Compatibility Files Remain

Inside the internal training model, some compatibility modules are still instantiated:

- `path_encoder`
- `Qformer`
- `path_proj`
- `text_proj`
- `path_trans`

They remain to preserve compatibility with the current code and checkpoints, but the active visual baseline path uses:

`raw image -> BiomedCLIP -> path_proj_llm -> Qwen3`
