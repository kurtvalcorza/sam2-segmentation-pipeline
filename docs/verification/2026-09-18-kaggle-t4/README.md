# SAM2 E2E Kaggle T4 verification

This directory retains the serial-suite evidence for the standalone SAM2 bounded fine-tuning E2E notebook.

## Immutable identity

- Target commit: `f17ba4834a5170f437f5015413f205dfd2f5ddd7`
- Embedded source revision: `bf2ee2d972eb76fea165781ac1206bc65483b987`
- Notebook: `tutorials/sam2_segmentation_colab.ipynb`
- Git blob: `130a5d9d03f5566d080e0b82d36f1eb176056342`
- Kernel: `kurtvalcorza/dimer-nb2-sam2-segmentation`, version 4
- Kaggle image: `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

## Result

- Executor outcome: **PASS**
- Qualification outcome: **REJECTED** — subsequent review found six defects; this v4 bundle is retained as historical evidence and does not qualify the corrected carrier.
- Runtime: Python 3.12.13, PyTorch 2.14.0+cu130, Transformers 4.57.6, Tesla T4 15,360 MiB
- Fresh Hugging Face cache: yes
- Wall time: 226.7 seconds
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
| `sam2_segmentation_dataset_manifest.json` | 460 | `cf331436a6d3f1ddcf5591b3d336505107592d5ea2e6d4758d9524eee21133fe` |
| `sam2_segmentation_evaluation_report.json` | 1,053 | `a1c0325b77c65a36634fc7ff5a02c271849b4307dec26368b1d6319282a12867` |
| `sam2_segmentation_result.json` | 3,918 | `4411c2a22bc4aea2f2d1836da1db68adb2f07d4a74a394698d11c8693a98d66b` |
| `sam2-mask-hypernetwork-adapter-v1/adapter.safetensors` | 559,960 | `5d5d9d628473c5d9e014daceea73964557bd55f321a1d34b654d9c94b74f9810` |
| `sam2-mask-hypernetwork-adapter-v1/manifest.json` | 2,108 | `9311a21a23c29694899611b69d4cb3707b6904febd678d9b6a74b6dd51e4ed6f` |

`suite/` contains the historical v2 bundle, the rejected v3 diagnostic bundle, and the corrected v4 generated executor, kernel metadata, first-pass restart record, successful executed notebook, `run_summary.json`, and preserved outputs. `LEDGER.md` records both the executor passes and the v3 qualification rejection.
