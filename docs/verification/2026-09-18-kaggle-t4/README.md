# SAM2 E2E Kaggle T4 verification

This directory retains the serial-suite evidence for the standalone SAM2 bounded fine-tuning E2E notebook.

## Immutable identity

- Target commit: `dc70d19e20fd0326d807159846696a273f678660`
- Embedded source revision: `3b292b27dc7d292b36c5512a60250cfb1ea58daa`
- Notebook: `tutorials/sam2_segmentation_colab.ipynb`
- Git blob: `f9bd4220248d89f00ca63701e3f79ac0378f4f87`
- Kernel: `kurtvalcorza/dimer-nb2-sam2-segmentation`, version 5
- Kaggle image: `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

## Result

- Outcome: **PASS**
- Runtime: Python 3.12.13, PyTorch 2.14.0+cu130, Transformers 4.57.6, Tesla T4 15,360 MiB
- Fresh Hugging Face cache: yes
- Wall time: 299.4 seconds
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
| `sam2_segmentation_result.json` | 3,918 | `82fd99065b8bff3f1c311c15e07cffd7aca4c32a597fb6bf7c326df235072df6` |
| `sam2-mask-hypernetwork-adapter-v1/adapter.safetensors` | 559,960 | `5d5d9d628473c5d9e014daceea73964557bd55f321a1d34b654d9c94b74f9810` |
| `sam2-mask-hypernetwork-adapter-v1/manifest.json` | 2,108 | `2a4a82d3a2b2dd3a5cedf0bff6cb67f5dc84be5f791ef7a06aac4d2c446d9826` |

`suite/` contains the historical v2 bundle, rejected v3 and v4 bundles, and the corrected v5 generated executor, kernel metadata, first-pass restart record, successful executed notebook, `run_summary.json`, and preserved outputs. `LEDGER.md` records the executor passes and qualification rejections.
