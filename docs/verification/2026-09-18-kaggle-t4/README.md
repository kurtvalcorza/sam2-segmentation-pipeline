# SAM2 E2E Kaggle T4 verification

This directory retains the serial-suite evidence for the standalone SAM2 bounded fine-tuning E2E notebook.

## Immutable identity

- Target commit: `d2fc69765370077bae1a4f7663e0b2b9195e3ec3`
- Embedded source revision: `9f3f2210e65b845f76e6a12abb176f598dc8712d`
- Notebook: `tutorials/sam2_segmentation_colab.ipynb`
- Git blob: `56a0f9889498b53787d2bd936d209d82c02e0944`
- Kernel: `kurtvalcorza/dimer-nb2-sam2-segmentation`, version 7
- Kaggle image: `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

## Result

- Outcome: **PASS**
- Runtime: Python 3.12.13, PyTorch 2.14.0+cu130, Transformers 4.57.6, Tesla T4 15,360 MiB
- Fresh Hugging Face cache: yes
- Wall time: 229.4 seconds
- Cells: 11/11 successful after one expected restart following dependency replacement
- Model snapshot: seven manifest entries downloaded and digest-verified at revision `ee5bba1d82bb8749febdf90f45e84b687142ba03`
- Training: 18 records, two epochs, 36 optimizer steps, 139,808 trainable parameters, weight delta 0.108819
- Held-out evaluation: pretrained IoU 0.998254, adapted IoU 0.998457, prompt-box baseline 0.856681
- Unseen generated scene: IoU 1.0
- Artifact: SafeTensors adapter 559,960 bytes with a closed digest-bound manifest
- Reload: passed; mask exact and score within the declared tolerance

The metrics are synthetic sample-sanity evidence, not a segmentation benchmark.

## Preserved output hashes

| File | Bytes | SHA-256 |
|---|---:|---|
| `sam2_segmentation_dataset_manifest.json` | 460 | `7902c1973ac6f9eede78edf397da99f3debafc63f1e15a8af525ed889d7fd769` |
| `sam2_segmentation_evaluation_report.json` | 1,053 | `a1c0325b77c65a36634fc7ff5a02c271849b4307dec26368b1d6319282a12867` |
| `sam2_segmentation_result.json` | 5,085 | `9405d1caab5f44ffc7e77c2de6c52990d2bc2a5a2051dc45dad49f0b1583e5bf` |
| `sam2-mask-hypernetwork-adapter-v1/adapter.safetensors` | 559,960 | `5d5d9d628473c5d9e014daceea73964557bd55f321a1d34b654d9c94b74f9810` |
| `sam2-mask-hypernetwork-adapter-v1/manifest.json` | 3,275 | `d4eb4202a20a76fadb215a198782fe5f836b93f7a289d0b3004ab075b960e057` |

`suite/` contains the historical v2 bundle, rejected v3 and v4 bundles, superseded v5/v6 evidence, and the current v7 generated executor, kernel metadata, first-pass restart record, successful executed notebook, `run_summary.json`, and preserved outputs. `LEDGER.md` records the executor passes and qualification rejections.
