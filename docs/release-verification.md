# Release verification

`tutorials/sam2_segmentation_colab.ipynb` is an `E2E`, standalone Candidate carrier under DIMER Notebook Specification 2.0. Current carrier commit `d2fc69765370077bae1a4f7663e0b2b9195e3ec3`, embedded source revision `9f3f2210e65b845f76e6a12abb176f598dc8712d`, and notebook blob `56a0f9889498b53787d2bd936d209d82c02e0944` passed top-to-bottom in a clean-cache Kaggle Tesla T4 v7 runtime. The exact-carrier execution-evidence gate is satisfied; Release-grade promotion remains a separate decision.

## Automatic coverage

CI and the local validator must establish that:

- notebook JSON and every code cell are valid, generated content is byte-for-byte current, and no outputs or execution counts are committed;
- the embedded pipeline, immutable model identity, snapshot manifest, and dependency pins match repository source;
- the default path uses a deterministic 24-record dataset and disjoint 18/6 train/held-out split;
- the pinned pretrained baseline runs before real gradient adaptation;
- only `mask_decoder.output_hypernetworks_mlps.0.*` is trainable, its weights move, and held-out mask IoU plus the prompt-box baseline are recorded;
- an unseen-scene prediction is exported, the adapter uses SafeTensors with a closed digest-bound manifest, and a fresh pinned base reload reproduces the prediction;
- the status remains Candidate until the current exact blob executes successfully, that evidence is reviewed, and an integrator promotes it.

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
| 2026-09-18 | `d2fc69765370077bae1a4f7663e0b2b9195e3ec3` / `56a0f9889498b53787d2bd936d209d82c02e0944` | Kaggle T4 (`kurtvalcorza/dimer-nb2-sam2-segmentation` v7), Python 3.12.13, torch 2.14.0+cu130 | Default generated dataset | **PASS** — clean cache; 11/11 code cells after one expected install restart; 229.4 s; 18/6 split; 36 optimizer steps; weight delta 0.108819; held-out IoU 0.998254 → 0.998457; unseen IoU 1.0; exact reload |

### Local E2E pre-flight (not promotion evidence)

| Date | Source identity | Executor | Outcome |
|---|---|---|---|
| 2026-09-18 | Uncommitted generated carrier; metadata base revision `47b14510c514` | Windows RTX 5070 Ti Laptop GPU, Python 3.14.2, torch 2.11.0+cu128, transformers 5.8.1; pinned install skipped | **PASS** — 11/11 code cells; 18 train + 6 held-out; 36 optimizer steps; 139,808 trainable / 38,398,769 frozen; weight delta 0.108818; held-out IoU 0.998254 → 0.998457 versus 0.856681 prompt-box baseline; unseen IoU 1.0; SafeTensors artifact and fresh-base reload exact |

### Historical superseded carriers

| Date (UTC) | Commit / notebook blob | Executor | Profile | Outcome |
|---|---|---|---|---|
| 2026-09-14 | `767ac10` / `efc84c11531c` | Kaggle T4 | Earlier inference-only carrier | Passed its prior 8-cell path; it is not evidence for the current E2E notebook |
| 2026-09-17 | `2bc18f1f5ab6da85a037b0b9ec7ee1764f79e748` / `d7d1cb7f803e9162335a053cba36e472a86d556f` | Kaggle T4 (`kurtvalcorza/dimer-nb2-sam2-segmentation` v2) | Earlier E2E carrier | Passed 11/11 cells, 36 optimizer steps, and fresh reload, but predates review remediation; retained as workflow history only |
| 2026-09-18 | `f17ba4834a5170f437f5015413f205dfd2f5ddd7` / `130a5d9d03f5566d080e0b82d36f1eb176056342` | Kaggle T4 (`kurtvalcorza/dimer-nb2-sam2-segmentation` v4) | Superseded E2E carrier | Passed 11/11 cells and reload, but subsequent review found transactional-state, malformed-tensor, tiny-mask, non-finite-update, fractional-box, and shape-fingerprint defects; retained as workflow history only |
| 2026-09-18 | `dc70d19e20fd0326d807159846696a273f678660` / `f9bd4220248d89f00ca63701e3f79ac0378f4f87` | Kaggle T4 (`kurtvalcorza/dimer-nb2-sam2-segmentation` v5) | Superseded E2E carrier | Passed 11/11 cells and reload, but subsequent review found a disjoint-box acceptance gap, default serving outside the adapted mask head, missing re-freeze on reload, and stale status wording; retained as workflow history only |
| 2026-09-18 | `f7a04605efd9b4a4e333e019433089d81e0feb5b` / `eab8d33961db00c33d37f1c12f96d67024fe1854` | Kaggle T4 (`kurtvalcorza/dimer-nb2-sam2-segmentation` v6) | Superseded E2E carrier | Passed 11/11 cells after one expected install restart in 239.1 s with clean cache, 36 optimizer steps, and exact reload, but predates the later artifact-lineage, BYOD, transactional-interruption, and metadata-schema repairs; retained as workflow history only |

## Current status

The current E2E implementation and generated carrier remain **Candidate**, with the exact-carrier clean-runtime gate **passed**. Kaggle T4 v7 evidence and output hashes are preserved under `verification/2026-09-18-kaggle-t4/`; the v6 evidence remains historical for superseded carrier `f7a0460`. Synthetic held-out scores are workflow evidence only and cannot establish quality improvement; Release-grade promotion remains a separate decision.
