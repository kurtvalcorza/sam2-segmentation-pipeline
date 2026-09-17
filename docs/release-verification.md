# Release verification

`tutorials/sam2_segmentation_colab.ipynb` is an `E2E`, standalone Candidate carrier under DIMER Notebook Specification 2.0. Static validation, unit tests, and the local full-carrier pre-flight do not satisfy the runtime gate. Promotion requires the exact committed notebook blob to execute top-to-bottom in a clean supported CUDA runtime.

## Automatic coverage

CI and the local validator must establish that:

- notebook JSON and every code cell are valid, generated content is byte-for-byte current, and no outputs or execution counts are committed;
- the embedded pipeline, immutable model identity, snapshot manifest, and dependency pins match repository source;
- the default path uses a deterministic 24-record dataset and disjoint 18/6 train/held-out split;
- the pinned pretrained baseline runs before real gradient adaptation;
- only `mask_decoder.output_hypernetworks_mlps.0.*` is trainable, its weights move, and held-out mask IoU plus the prompt-box baseline are recorded;
- an unseen-scene prediction is exported, the adapter uses SafeTensors with a closed digest-bound manifest, and a fresh pinned base reload reproduces the prediction;
- the status remains Candidate until supported-runtime evidence for the exact blob exists.

## Supported execution procedure

1. Commit the generated notebook and resolve its Git blob ID.
2. Start a new Colab T4 or Kaggle T4 runtime with no repository checkout or warm model cache.
3. Run all cells at their defaults with `USE_BYOD = False`; do not edit implementation cells.
4. Confirm the runtime reports CUDA, the immutable model revision, and the same repository revision recorded in notebook metadata.
5. Confirm dataset validation and the 18/6 split, pretrained evaluation, two training epochs, nonzero `weight_delta_l2`, held-out evaluation, unseen inference, artifact digest verification, and fresh-base reload all complete.
6. Confirm the output directory contains the dataset manifest, evaluation report, result JSON, and exactly the two-file adapter directory.
7. Record runtime versions, device, notebook commit/blob, wall time, metrics, output inventory, warnings, and clean-cache status below. Record no secrets.

A failed default path, missing gradient update, altered split, unsafe artifact, or reload mismatch blocks promotion.

## Recorded executions

### Current E2E carrier

| Date (UTC) | Commit / notebook blob | Executor | Path | Outcome |
|---|---|---|---|---|
| Pending | Not committed | Colab or Kaggle CUDA | Default generated dataset | Candidate gate open; no exact-blob supported-runtime execution yet |

### Local E2E pre-flight (not promotion evidence)

| Date | Source identity | Executor | Outcome |
|---|---|---|---|
| 2026-09-18 | Uncommitted generated carrier; metadata base revision `47b14510c514` | Windows RTX 5070 Ti Laptop GPU, Python 3.14.2, torch 2.11.0+cu128, transformers 5.8.1; pinned install skipped | **PASS** — 11/11 code cells; 18 train + 6 held-out; 36 optimizer steps; 139,808 trainable / 38,398,769 frozen; weight delta 0.108818; held-out IoU 0.998254 → 0.998457 versus 0.856681 prompt-box baseline; unseen IoU 1.0; SafeTensors artifact and fresh-base reload exact |

### Historical superseded carrier

| Date (UTC) | Commit / notebook blob | Executor | Profile | Outcome |
|---|---|---|---|---|
| 2026-09-14 | `767ac10` / `efc84c11531c` | Kaggle T4 | Earlier inference-only carrier | Passed its prior 8-cell path; it is not evidence for the current E2E notebook |

## Current status

The E2E implementation and generated carrier remain **Candidate**. The complete generated carrier passed locally on CUDA, but its interpreter differs from the declared pins and the current uncommitted notebook has no immutable blob identity. It therefore cannot receive qualifying clean-runtime evidence yet.
