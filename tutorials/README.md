# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/sam2-segmentation-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/sam2-segmentation-pipeline/blob/main/tutorials/sam2_segmentation_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fsam2.1--hiera--small-ffcc4d?style=flat)](https://huggingface.co/facebook/sam2.1-hiera-small)
[![Upstream](https://img.shields.io/badge/Upstream-facebookresearch%2Fsam2-181717?style=flat&logo=github&logoColor=white)](https://github.com/facebookresearch/sam2)
[![arXiv](https://img.shields.io/badge/arXiv-2408.00714-b31b1b.svg)](https://arxiv.org/abs/2408.00714)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `sam2_segmentation_colab.ipynb` | `TASK-INFERENCE` | promptable image segmentation (one object per call from point and/or box prompts) with `facebook/sam2.1-hiera-small`; `(K, H, W)` boolean masks plus model-predicted, uncalibrated `iou_scores`; image mode only, video tracking out of scope; `mask_iou` against the drawn rectangle as sanity evidence, no mIoU | CPU (CUDA used automatically when available) | single image file (16–4096 px per side) plus a click set through `POINT_X`/`POINT_Y`, gated off by default; no reference mask, so no IoU | **Candidate** — static checks pass; the clean-runtime execution run is pending and will be recorded in `../docs/release-verification.md`, which must be reviewed for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `SAM2SegmentationPipeline` from the repository public API rather than reimplementing model loading; model acquisition goes through the package: `stage_missing_files(WEIGHTS_DIR, allow_download=True)` fetches only the manifest entries a fresh clone lacks, at the pinned revision, `verify_snapshot` re-hashes every entry, and `from_pretrained(weights_dir=WEIGHTS_DIR)` loads the verified files (`local_files_only=True`, `trust_remote_code=False`; the expected `sam2_video → sam2` load notice is explained before the load; the notebook never calls `huggingface_hub`).
- The default sample is a synthetic 320×240 scene drawn in code (dark rectangle at [40, 60, 140, 180], red disc at [200, 80, 280, 160]) with one foreground click at (90, 120) — the same scene and click the card-pass smoke used; `mask_iou` of the best candidate against the drawn rectangle's mask and the area comparison are sanity evidence for the prompt, forward pass and up-sampling, not a segmentation metric (mIoU needs labelled masks and is not computed).
- Score semantics: `iou_scores` are the model's own uncalibrated predictions, not measured overlaps or probabilities; the conventional argmax candidate is used and stated; the pipeline ships no acceptance threshold and the caller owns any decision rule.
- Ceilings `MIN_IMAGE_SIDE` (16), `MAX_IMAGE_SIDE` (4096), `MAX_PROMPTS` (16), `NUM_MULTIMASK_OUTPUTS` (3) are printed before the model runs; one object per call is stated; video tracking, automatic segmentation, text prompts and class labels are named as out of scope.
- CPU is documented as adequate (card-measured 4.8 s load, ~0.9 s per `segment` on 320×240; encoder cost fixed by the 1024×1024 resize).
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
