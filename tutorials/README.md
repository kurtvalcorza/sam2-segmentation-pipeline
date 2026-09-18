# Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/sam2-segmentation-pipeline/blob/main/tutorials/sam2_segmentation_colab.ipynb)

Notebook specification: **DIMER Notebook Specification 2.0**. The notebook is a generated, standalone carrier: edit `tools/notebook_template.py` or the package and regenerate with `python tools/build_notebook.py`; do not hand-edit the `.ipynb`.

| Notebook | Profile | Mode | Carrier | Default E2E path | Runtime | BYOD | Release status |
|---|---|---|---|---|---|---|---|
| `sam2_segmentation_colab.ipynb` | `E2E` | `GUIDED` | standalone (generated) | 24 deterministic labelled scenes; 18/6 train/held-out split; pinned pretrained baseline; two real gradient epochs on the first mask-token hypernetwork; held-out mask IoU and prompt-box baseline; unseen-scene prediction; SafeTensors adapter export; fresh pinned-base reload equivalence | CUDA required; Colab T4 or Kaggle T4 supported | ZIP containing paired `images/` and `masks/` files, off by default | **Candidate** — fresh exact-blob clean-runtime execution is pending for carrier `a7634b491ed1`; the recorded Kaggle T4 v6 run is historical evidence for a superseded carrier |

## Conformance notes

- The carrier embeds the exact pipeline module, immutable model identity, snapshot manifest, and runtime pins. Generator parity is enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`.
- `validate_segmentation_dataset` rejects malformed, duplicate, mismatched, empty, or oversized records before training. The default sample and split are deterministic.
- `freeze_for_adaptation` freezes the pinned base except `mask_decoder.output_hypernetworks_mlps.0.*`; `finetune` records nonzero gradient-driven weight movement and validation history.
- `evaluate_adaptation` reports held-out mean mask IoU for the model and a prompt-box baseline. The synthetic result is workflow evidence, not a SAM benchmark.
- `save_artifact` writes only `adapter.safetensors` plus a closed manifest containing the pinned base identity, trainable prefixes, digest, and adaptation metadata. `from_artifact` constructs a fresh pinned base, verifies the artifact, and attaches it without pickle or remote code.
- The unseen-scene prediction must match after fresh reload. Static checks and local runs are not promotion evidence; see `../docs/release-verification.md`.

## AI Assistance Disclosure

This repository’s code and documentation were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for review, validation, and release decisions.
