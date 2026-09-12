# SAM 2.1 Hiera-Small promptable segmentation pipeline

DIMER inference wrapper for **SAM 2.1 Hiera-Small** (`facebook/sam2.1-hiera-small`), pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline segments one object per call from point clicks and/or a box and returns boolean masks at the input resolution with the model's predicted IoU per mask. Image mode only: video tracking is not exposed.

## Upstream alignment

- Model: `facebook/sam2.1-hiera-small`
- Revision: `ee5bba1d82bb8749febdf90f45e84b687142ba03`
- Upstream weight license: Apache-2.0
- Upstream task: promptable visual segmentation in images and videos; this package exposes images only
- Repository adaptation: **none**; inference only

## Quick start

```python
import numpy as np
from PIL import Image
from sam2_segmentation_pipeline import SAM2SegmentationPipeline, mask_iou

pipe = SAM2SegmentationPipeline.from_pretrained()      # stages + verifies weights/sam2.1-hiera-small first
image = Image.open("photo.jpg")

# one foreground click -> three candidate masks, pick the one the model rates highest
result = pipe.segment(image, points=[(500, 375)], point_labels=[1])
best = result["masks"][int(np.argmax(result["iou_scores"]))]   # bool H x W

# a box (e.g. from grounding-dino-detection-pipeline) -> one mask
result = pipe.segment(image, box=[75, 275, 1725, 850], multimask=False)
mask = result["masks"][0]

# optional: score against a caller-supplied reference mask
# print(mask_iou(mask, reference_bool_mask))
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed). On a fresh clone the manifest is committed but the weights are not: `SAM2SegmentationPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/sam2.1-hiera-small/
  dimer-base-manifest.json      # modelId, revision, per-file bytes + SHA-256 (7 files)
  config.json                   # declares Sam2VideoModel; Sam2Model loads the same weights for images
  preprocessor_config.json, processor_config.json
  video_preprocessor_config.json, sam2.1_hiera_s.yaml   # present in the snapshot, unused here
  model.safetensors             # git-ignored, 184,305,280 bytes
  README.md
```

## Input ceilings

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16` points; one object per call; masks are binarised at logit `MASK_THRESHOLD = 0.0`. See `MODEL_CARD.md` for the measured CPU timings and the candidate-selection rule.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.
