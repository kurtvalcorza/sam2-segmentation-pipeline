# SAM2 E2E Kaggle T4 verification

This directory retains the serial-suite evidence for the standalone SAM2 bounded fine-tuning E2E notebook.

## Immutable identity

- Target commit: `2bc18f1f5ab6da85a037b0b9ec7ee1764f79e748`
- Embedded source revision: `a062e8f91187a78563ab2971369d86874df236a2`
- Notebook: `tutorials/sam2_segmentation_colab.ipynb`
- Git blob: `d7d1cb7f803e9162335a053cba36e472a86d556f`
- Kernel: `kurtvalcorza/dimer-nb2-sam2-segmentation`, version 2
- Kaggle image: `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

## Result

- Outcome: **PASS**
- Runtime: Python 3.12.13, PyTorch 2.14.0+cu130, Transformers 4.57.6, Tesla T4 15,360 MiB
- Fresh Hugging Face cache: yes
- Wall time: 229.9 seconds
- Cells: 11/11 successful after one expected restart following dependency replacement
- Model snapshot: seven manifest entries downloaded and digest-verified at revision `ee5bba1d82bb8749febdf90f45e84b687142ba03`
- Training: 18 records, two epochs, 36 optimizer steps, 139,808 trainable parameters, weight delta 0.108795
- Held-out evaluation: pretrained IoU 0.998254, adapted IoU 0.998457, prompt-box baseline 0.856681
- Unseen generated scene: IoU 1.0
- Artifact: SafeTensors adapter 559,960 bytes with a closed digest-bound manifest
- Reload: passed; mask exact and score within the declared tolerance

The metrics are synthetic sample-sanity evidence, not a segmentation benchmark.

## Preserved output hashes

| File | Bytes | SHA-256 |
|---|---:|---|
| `sam2_segmentation_dataset_manifest.json` | 460 | `0c550c47eb2c641c8d89570fc4cf8555788d0ee1c85dcd317550ca668bba374e` |
| `sam2_segmentation_evaluation_report.json` | 854 | `83370ced505dbd41b46c8dda2b89d448b0deb547f4cdd435a1ca9aa265bda243` |
| `sam2_segmentation_result.json` | 3,711 | `e4ecce0b155279d80e402327f89c841ed12b2090252478aa54d4f2b19a0bfad6` |
| `sam2-mask-hypernetwork-adapter-v1/adapter.safetensors` | 559,960 | `ceb1fba1eab315c6455090180676ea72415b8d0ae4cd6cd3f31e347e96bf6ca6` |
| `sam2-mask-hypernetwork-adapter-v1/manifest.json` | 2,108 | `1d98ea0cefaecb9fae87797cdbabc060f12b1a32e7345c64691e0c3df05df955` |

`suite/` contains the generated executor, kernel metadata, first-pass restart record, successful executed notebook, `run_summary.json`, and preserved outputs. `LEDGER.md` is the serial-suite audit table.
