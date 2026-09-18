# Kaggle serial execution ledger

| repo | sha | notebook | blob | kernel (version) | machine | result | wall | cells | staged weights | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| sam2-segmentation-pipeline | 2bc18f1 | sam2_segmentation_colab.ipynb | d7d1cb7f803e9162335a053cba36e472a86d556f | dimer-nb2-sam2-segmentation (v2) | T4 | PASS | 229.9 s | 11/11 ok (1 restart after install cell) | 16 files, 184 MB | image torch 2.10.0+cu128; gpu Tesla T4, 15360 MiB, 580.159.04 Tesla T4, 15360 MiB, 580.159.04 |
| sam2-segmentation-pipeline | 1607bd6 | sam2_segmentation_colab.ipynb | f1ebf45f945ccd048af07c857f2d221c8487a484 | dimer-nb2-sam2-segmentation (v3) | T4 | PASS | 230.0 s | 11/11 ok (1 restart after install cell) | 16 files, 184 MB | image torch 2.10.0+cu128; gpu Tesla T4, 15360 MiB, 580.159.04 Tesla T4, 15360 MiB, 580.159.04 |
| sam2-segmentation-pipeline | 1607bd6 | sam2_segmentation_colab.ipynb | f1ebf45f945ccd048af07c857f2d221c8487a484 | dimer-nb2-sam2-segmentation (v3) | T4 | QUALIFICATION REJECTED | 230.0 s | 11/11 ok (1 restart after install cell) | 16 files, 184 MB | executor passed, but the subsequent review found split-leak, output-resolution scoring, mixed-subset baseline, BYOD decode-limit, reporting, and unseen-sample defects; corrected carrier rerun required |
