# SAM2 E2E Kaggle T4 verification

This directory retains the serial-suite evidence for the standalone SAM2 bounded fine-tuning E2E notebook.

## Immutable identity

- Target commit: `f7a04605efd9b4a4e333e019433089d81e0feb5b`
- Embedded source revision: `1814d040490d11930fa2f8239a846f321717f2e2`
- Notebook: `tutorials/sam2_segmentation_colab.ipynb`
- Git blob: `eab8d33961db00c33d37f1c12f96d67024fe1854`
- Kernel: `kurtvalcorza/dimer-nb2-sam2-segmentation`, version 6
- Kaggle image: `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

## Result

- Outcome: **PASS**
- Runtime: Python 3.12.13, PyTorch 2.14.0+cu130, Transformers 4.57.6, Tesla T4 15,360 MiB
- Fresh Hugging Face cache: yes
- Wall time: 239.1 seconds
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
| `sam2_segmentation_result.json` | 3,918 | `29d63c116842d1bd5a4b8e52046442705b5076df47f34a22ae071d379d051efb` |
| `sam2-mask-hypernetwork-adapter-v1/adapter.safetensors` | 559,960 | `5d5d9d628473c5d9e014daceea73964557bd55f321a1d34b654d9c94b74f9810` |
| `sam2-mask-hypernetwork-adapter-v1/manifest.json` | 2,108 | `b77648d5581b93b287a779be145b9d5879f11ae21f3521be0acda213063334ab` |

`suite/` contains the historical v2 bundle, rejected v3 and v4 bundles, superseded v5 evidence, and the corrected v6 generated executor, kernel metadata, first-pass restart record, successful executed notebook, `run_summary.json`, and preserved outputs. `LEDGER.md` records the executor passes and qualification rejections.
