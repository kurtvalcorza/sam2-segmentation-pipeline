"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment).

The generator writes the infrastructure cells (runtime check — which stops on a CPU-only runtime, since the fine-tune
needs CUDA —, carrier, isolated install + stage runner, snapshot staging) from repository files; this template holds the
learner-facing prose, the guided layer and the learner cells. Every learner cell calls ``run_stage(...)``: the carried
``tools/tutorial_stages.py`` runs one stage per process in an isolated, hash-locked environment, so nothing is installed
into the notebook kernel and no restart is needed.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "sam2-segmentation-pipeline"

UV = {
    "version": "0.12.15",
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}

TEMPLATE = {
    "package": "sam2_segmentation_pipeline",
    "repo_name": REPO,
    "weights_key": "sam2.1-hiera-small",
    "modules": ["__init__.py", "pipeline.py"],
    "entry_module": "pipeline.py",
    "lock": "tutorials/requirements-colab.lock.txt",
    "managed_python": "3.12.12",
    "uv": UV,
    "disk_gib": {"weights": 0.2, "environment": 9.0},
    "runtime_modules": ["torch", "transformers", "safetensors"],
    "install_flags": ["--only-binary", ":all:"],
    "require_gpu": True,
    "stem": "sam2_segmentation",
    "notebook_name": "sam2_segmentation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages.py",
    "run_all": (
        "Selecting **Run all** on a fresh Linux x86_64 runtime with a CUDA GPU (Colab T4 or Kaggle GPU) builds an isolated "
        "Python environment from the carried hash-locked requirements without touching the notebook kernel's own packages, "
        "verifies the exact base snapshot, generates and validates 24 records, measures the pretrained held-out baseline under "
        "two prompt conditions, freezes the base and performs the bounded gradient update from the pinned base, exports the "
        "adapter, attaches it to a fresh base to score the held-out split per record, predicts an unseen scene with a reload "
        "check against the adapting process, and writes provenance — without a clone, credential, upload, configuration edit "
        "or runtime restart. On a CPU-only runtime Section 1 stops at once with the runtime-change instruction."
    ),
    "byod": (
        "Set `USE_BYOD = True` and either set `BYOD_ZIP_PATH` to a ZIP in this runtime (any runtime; an attached Kaggle "
        "input is found automatically when it is the only ZIP under `/kaggle/input/`) or leave it empty in Colab to open the "
        "upload dialog. The ZIP holds `images/<stem>.png|jpg` and `masks/<stem>.png` (white = foreground), at the top level "
        "or inside one wrapping folder; at least 8 pairs, at most 64. The held-out split is a seeded shuffle; set "
        "`GROUP_SEPARATOR` (for example `_`) so frames or crops of one scene (`scene1_a`, `scene1_b`) stay on one side. Your "
        "records pass through the same validation, baseline, adaptation, evaluation, export and reload path, and the unseen "
        "inference and reload check run on one of your held-out records. Every refusal names the archive and the rule."
    ),
    "title": "SAM 2.1 Hiera-Small — DIMER end-to-end promptable segmentation fine-tuning",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            f"https://github.com/kurtvalcorza/{REPO}",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/sam2_segmentation_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fsam2.1--hiera--small-ffcc4d?style=flat",
            "https://huggingface.co/facebook/sam2.1-hiera-small",
        ),
    ],
    "capability": "pinned SAM 2.1 inference plus bounded mask-hypernetwork gradient adaptation, held-out mask-IoU evaluation, safe adapter export, and fresh reload",
    "intro": (
        "This notebook performs a real local gradient update. It freezes the Hiera encoder, prompt encoder, and the rest of "
        "the mask decoder, then trains only the first mask-token output hypernetwork (139,808 parameters) with BCE plus "
        "soft-Dice loss. The default dataset is 24 deterministic generated shape scenes split 18/6 before model execution. A "
        "SafeTensors adapter and closed SHA-256 manifest are exported and attached to a fresh pinned base model. These are "
        "tutorial sample-sanity results, not a segmentation benchmark.\n\n"
        "**Read this before Section 8: the default task leaves almost nothing to adapt.** Each generated scene is one bright "
        "shape on a flat dark background, and the recorded condition gives the model the shape's exact box and an interior "
        "point. The pretrained model already scored held-out mask IoU 0.998254 in the recorded run, and the adapted model "
        "0.998457 — a change of +0.0002 on six images, which is a mechanics check, not an improvement. The notebook also "
        "scores a second held-out condition, **point-only prompts** (no box), generated from the same records, and prints "
        "every held-out record before and after. One interior point is enough on a single high-contrast shape, so point-only "
        "is at ceiling too (about 0.998 mean IoU before and after adaptation in a local check of the pinned checkpoint). "
        "Neither condition can show an adaptation effect: the lesson here is the workflow — validate, measure, adapt, export, "
        "reload, and read the result honestly — not a gain."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Validate** paired image / mask / prompt records and **explain** why the held-out split is fixed before any model runs (Sections 4, 5).\n"
        "2. **Measure** a pretrained held-out baseline under two prompt conditions and **say** how much headroom it leaves (Section 6).\n"
        "3. **Describe** what the bounded fine-tune changes — one mask hypernetwork — and why the weights moving is not evidence of improvement (Section 7).\n"
        "4. **Judge** an adaptation result from per-record held-out IoU, and **distinguish** a mechanics check from evidence of improvement (Section 8).\n"
        "5. **Verify** that the exported adapter on a fresh base reproduces the adapting process (Section 9).\n"
        "6. **Predict**, run and **explain** the effect of a larger learning rate in an optional activity (Section 11)."
    ),
    "exclusions": (
        "full-model training, video tracking, automatic segment-everything, semantic class learning, production-scale "
        "training, or benchmark claims. The adapter changes one image-mode mask hypernetwork only."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime **with a CUDA GPU** — Google Colab with a T4 GPU, or a Kaggle GPU kernel. The fine-tune needs CUDA, so Section 1 stops at once on a CPU-only runtime. The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (`torch` 2.14.0 with its CUDA libraries, `transformers` 4.57.6, `safetensors` 0.8.0, `numpy` 2.5.3, `pillow` 11.3.0). About 0.2 GB of disk is needed for the snapshot and about 9 GB for the isolated environment.",
        "- **Knowledge:** Python, binary masks and IoU, and why training and held-out data must be kept apart. The glossary below explains the SAM-specific terms.",
        "- **Data:** the default path generates 24 shape scenes. Optional BYOD accepts one ZIP with `images/` and `masks/` paired by stem (masks as PNG, white = foreground), at least 8 and at most 64 pairs, each image 16..4096 px per side and matching its mask, 32 megapixels in all. Do not upload confidential or restricted data unless you are authorized to use it in the hosted runtime.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook and read short Python, who know what a "
                "binary mask and IoU are, and who want to see a promptable segmentation model adapted end to end: a fixed held-out "
                "split, a pretrained baseline, a bounded fine-tune, an honest per-record comparison, and an adapter that reloads "
                "with identical masks. No experience with SAM is assumed; the glossary below explains every term.\n\n"
                "**Running it.** Choose a GPU runtime (*Runtime → Change runtime type → T4 GPU* in Colab), then *Runtime → Run all*. "
                "The default path needs no edit, no upload, no account, no token and no runtime restart. Section 2 builds an "
                "isolated environment, which takes the longest.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell calls "
                "`run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files; the adapted model reaches Sections 8–10 only through "
                "the exported adapter.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–11) are the workflow. *Infrastructure cells* (Sections 1–3) are "
                "collapsed and titled **Infrastructure**; you may run them without studying their implementation.\n\n"
                "**Form controls.** `USE_BYOD`, `BYOD_ZIP_PATH`, `GROUP_SEPARATOR` and `SEED` (Section 4); `EPOCHS` and "
                "`LEARNING_RATE` (Section 7); `RUN_ACTIVITY` and `ACTIVITY_LEARNING_RATE` (Section 11). Leave them at their defaults "
                "for the first run.\n\n"
                "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is "
                "produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before Sections 6, 7 and 8 a **Predict before running** prompt asks you to commit to an "
                "expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint. The sample answers quote the recorded Kaggle T4 run for the point+box condition."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Validate and split** | 24 images, each with a mask, an interior point and a box | `validate_segmentation_dataset`; fixed 18 / 6 split | a dataset manifest with content fingerprints |\n"
                "| **Baseline** | the 6 held-out records, two prompt conditions | pretrained SAM 2.1 Hiera-Small (image mode) | per-record mask IoU |\n"
                "| **Adapt** | the 18 training records | AdamW on one mask hypernetwork (BCE + soft-Dice), base frozen | an adapter (`adapter.safetensors` + manifest) |\n"
                "| **Evaluate** | the held-out records again | the adapter on a fresh base, in a fresh process | per-record change, records improved, an interpretation |\n"
                "| **Reuse** | an unseen scene | the adapter on a fresh base | a mask; reload parity with the adapting process |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1–3 | [Engineering] | GPU check, carried code, isolated environment, verified snapshot | versions, digests |\n"
                "| 4. Data | [Evaluation practice] | generated or BYOD records, the split | counts, split rule |\n"
                "| 5. Validation | [Evaluation practice] | dataset contract and leakage check | the dataset manifest |\n"
                "| 6. Baseline | [Evaluation practice] | pretrained IoU, two prompt conditions | the headroom |\n"
                "| 7. Fine-tune | [Concept] | bounded update from the pinned base; adapter exported | history, weight delta |\n"
                "| 8. Evaluation | [Evaluation practice] | per-record before/after, both conditions | records improved, interpretation |\n"
                "| 9. Unseen scene and reload | [Engineering] | inference and reload parity across processes | IoU, `verification` |\n"
                "| 10. Provenance | [Engineering] | the result record | `result.json` |\n"
                "| 11. Optional activity | [Concept] | a larger learning rate, printed beside the default | a comparison table |\n"
                "| Troubleshooting | [Engineering] | every failure and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | what was and was not shown | your conclusion |"
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Promptable segmentation** | The model returns a mask for the object a prompt points at, rather than labelling classes. |\n"
                "| **Prompt (point / box)** | A clicked point inside the object and/or a box around it; this notebook compares point+box with point-only. |\n"
                "| **Mask decoder / mask token** | The small network that turns image features and prompts into masks; one learned token per output mask. |\n"
                "| **Hypernetwork** | A small MLP that produces the weights used to read out one mask token; the only part trained here. |\n"
                "| **Mask IoU** | Intersection over union of predicted and true masks: 1.0 is a perfect match. |\n"
                "| **Soft-Dice loss** | A differentiable overlap loss used with binary cross-entropy (BCE) during training. |\n"
                "| **Headroom** | How far the pretrained score is from perfect; with no headroom, adaptation cannot show an effect. |\n"
                "| **Adapter vs base** | The adapter holds only the trained hypernetwork; it is attached to the pinned, unchanged base model. |\n"
                "| **Reload parity** | A fresh base plus the adapter gives the same masks and scores as the process that trained it. |\n"
                "| **Hash-locked environment / stage** | The isolated Python environment every stage runs in; one workflow step run as its own process. |\n\n"
                "</details>"
            ),
        ],
    },
    "cells": [
        {
            "md": (
                "## 4. Generate or supply a paired segmentation dataset · [Evaluation practice]\n\n"
                "The default 24 deterministic 256×256 scenes contain one rounded rectangle or ellipse plus a distractor rectangle on "
                "a flat background. Each record has an RGB image, a boolean mask, a foreground point and a bounding box. Records "
                "0–17 train; 18–23 are held out — fixed before any model runs, so nothing the model does can influence the split.\n\n"
                "**Bring your own data.** Set `USE_BYOD = True` and `BYOD_ZIP_PATH` (or, in Colab, leave it empty for the upload "
                "dialog; on Kaggle a single attached ZIP is found automatically). The split is a seeded shuffle of 25 % of the "
                "records; set `GROUP_SEPARATOR` so records whose stems share a prefix (`scene1_a`, `scene1_b` with `_`) stay "
                "together, otherwise every image is treated as independent. A wrapping folder, unmatched stems (named), JPEG "
                "masks, unsafe paths, oversized archives and fewer than 8 pairs each stop this cell with their own message."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "GROUP_SEPARATOR = ''  # @param {{type:\"string\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "def upload_one(what, field):\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'{{field}} is empty, and the upload dialog exists only in Google Colab: set {{field}} to {{what}} in this runtime.') from None\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise RuntimeError(f'The upload was cancelled or empty: no file was received. Run this cell again and choose {{what}}, or set {{field}}.')\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError(f'Upload exactly one file ({{what}}); got {{sorted(uploaded)}}.')\n"
                "    upload_name, payload = next(iter(uploaded.items()))\n"
                "    path = ROOT / 'inputs' / Path(upload_name).name\n"
                "    path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    path.write_bytes(payload)\n"
                "    return str(path)\n\n"
                "byod_zip = ''\n"
                "if USE_BYOD:\n"
                "    byod_zip = BYOD_ZIP_PATH or ('' if Path('/kaggle/input').is_dir() else upload_one('one BYOD ZIP with images/ and masks/', 'BYOD_ZIP_PATH'))\n"
                "run_stage('data', use_byod=USE_BYOD, byod_zip_path=byod_zip, group_separator=GROUP_SEPARATOR, seed=SEED)"
            ),
        },
        {
            "md": "**What to notice:** `dataset_kind: 'generated'`, 24 records split 18 / 6, the split rule, and the unseen scene (`generated-24`) outside both.",
        },
        {
            "md": (
                "## 5. Validate the dataset and split · [Evaluation practice]\n\n"
                "Both splits must have unique ids, exact image/mask alignment, boolean masks with foreground and background, and "
                "valid prompts (the point inside the mask, the box overlapping it); each record gets a content fingerprint. Ids on "
                "both sides of the split stop the stage. The manifest is written to `outputs/{stem}_dataset_manifest.json`."
            ),
            "code": "run_stage('validate')",
        },
        {
            "md": "**What to notice:** `overlap_ids: []`, 18 training and 6 held-out records, and the split rule repeated in the manifest.",
        },
        {
            "md": (
                "## 6. Measure the pretrained held-out baseline · [Evaluation practice]\n\n"
                "Before any update the pretrained model scores the six held-out records under two prompt conditions: **point+box** "
                "(the exact box and an interior point — the recorded condition) and **point-only** (the same point, no box). Each "
                "record's IoU is printed, with the IoU of simply filling the prompt box. Loading the model prints *You are using a "
                "model of type sam2_video to instantiate a model of type sam2*: the published `config.json` declares the video model, "
                "and image-mode `Sam2Model` loading the same weights is the intended use (upstream README), so the warning is "
                "expected.\n\n"
                "**Predict before running:** under point+box prompts, how close to 1.0 will the pretrained model's mean IoU be on "
                "these scenes? And under point-only?"
            ),
            "code": "run_stage('baseline')",
        },
        {
            "md": (
                "**What to notice:** point+box mean IoU 0.998254 in the recorded run (every record near 1.0), the prompt-box fill "
                "baseline 0.856681, and the point-only row — also near 1.0.\n\n"
                "**Checkpoint:** what is the most the fine-tune could gain under point+box prompts here?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "At most 1 − 0.998254 ≈ 0.0017 mean IoU — the pretrained model is at ceiling, because one bright shape on a flat "
                "background with its exact box is easy for a promptable segmenter. Any change Section 8 shows under that condition "
                "is too small to read on six records. Point-only prompts remove the box, so the model must find the extent itself "
                "— but one interior point is enough on a single high-contrast shape, and that baseline sits near 0.998 too, so "
                "neither condition leaves room to show an effect here. The box-fill "
                "baseline (0.857) measures what prompting adds, not what adaptation adds.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7. Freeze the base and run bounded fine-tuning · [Concept]\n\n"
                "The stage loads a fresh pinned base (so re-running this cell always starts from the base, never from an earlier "
                "adapter), freezes everything except the first mask-token output hypernetwork (139,808 parameters), and runs AdamW "
                "for `EPOCHS` epochs at `LEARNING_RATE` with batch size one on the 18 training records, with point+box prompts. "
                "Held-out IoU is computed after each epoch for **monitoring only**; the final weights are kept. A non-zero L2 weight "
                "delta is required — it shows the parameters moved, not that anything improved. The stage then exports the "
                "adapter to `outputs/sam2-mask-hypernetwork-adapter-v1/` and records its own prediction on the unseen scene, which "
                "Section 9's fresh reload must reproduce.\n\n"
                "**Predict before running:** will the training loss fall? Will the per-epoch held-out IoU move much?"
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 2e-5  # @param {{type:\"number\"}}\n"
                "run_stage('adapt', epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=SEED)"
            ),
        },
        {
            "md": (
                "**What to notice:** 139,808 trainable parameters, 36 optimizer steps, training loss 0.0071 → 0.0054 and held-out "
                "IoU 0.998383 → 0.998457 across the two epochs in the recorded run, and a weight delta around 0.109.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "The loss falls — optimisation works — but the held-out IoU barely moves because it started at ceiling. A falling "
                "loss and a non-zero weight delta are optimisation evidence; held-out quality is measured in Section 8.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Evaluate the adapted model, record by record · [Evaluation practice]\n\n"
                "A fresh process attaches the exported adapter to a fresh pinned base and scores the six held-out records under both "
                "prompt conditions. For each condition the stage prints the pretrained and adapted mean, the change, how many "
                "records improved, every record before and after, and an **interpretation** line: when the pretrained mean is 0.99 "
                "or above it says *no headroom* and calls the comparison a mechanics check. The report, with every per-record value, "
                "is written to `outputs/{stem}_evaluation_report.json`. These are sample-sanity results on six generated records: "
                "one holdout, no dispersion estimate.\n\n"
                "**Predict before running:** under point+box, will the adapted mean differ from 0.998254 by more than 0.001? How many "
                "of the six records will improve?"
            ),
            "code": "run_stage('evaluate')",
        },
        {
            "md": (
                "**What to notice:** point+box 0.998254 → 0.998457 in the recorded run with the *no headroom* interpretation, the "
                "per-record rows, and the point-only rows, which carry the same *no headroom* interpretation.\n\n"
                "**Checkpoint:** a classmate says \"fine-tuning improved SAM from 0.99825 to 0.99846\". What would you reply?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "That the difference is +0.0002 mean IoU on six records that were already at ceiling: it is within what any small "
                "perturbation would produce, and the weights moving (Section 7) does not show quality changed. The run shows the "
                "adaptation machinery works. To claim improvement you would need data with headroom — textured "
                "backgrounds, low contrast, occlusion or loose boxes; point-only prompts do not supply it on these scenes — a change that holds across most records, and more than one holdout.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 9. Infer on an unseen scene and verify a fresh reload · [Engineering]\n\n"
                "A fresh process attaches the adapter to a fresh pinned base and segments the unseen scene — `generated-24`, outside "
                "the train and held-out range, or one of your held-out records with BYOD. It then compares the mask and scores with "
                "the prediction the adapting process recorded in Section 7: the mask must match exactly and the scores within "
                "`rtol=1e-5`, `atol=1e-6`, or the stage stops with *Do not ship this artifact*. The stage only reads the adapter, so "
                "re-running it alone is safe."
            ),
            "code": "run_stage('unseen')",
        },
        {
            "md": (
                "**What to notice:** the unseen-scene IoU (1.0 in the recorded run, model score 0.992), the adapter's files, and "
                "`verification: 'PASSED'` with `mask_exact: True`.\n\n"
                "**Checkpoint:** why compare against the adapting process instead of against the ground-truth mask?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The ground truth tests quality; this check tests packaging. A user receives the adapter and the pinned base, not "
                "the trained object, so the question is whether those files rebuild exactly the model that was evaluated. An exact "
                "mask and matching scores answer it; a mismatch would mean the evaluation describes a different model from the one "
                "shipped.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 10. Export provenance and terminal summary · [Engineering]\n\n"
                "`outputs/{stem}_result.json` binds the dataset manifest, the adaptation configuration and lineage, the evaluation "
                "report, the unseen-scene result, the reload evidence, the notebook source, the base model identity and licence, "
                "and the isolated environment's versions and device — without retaining any image."
            ),
            "code": "run_stage('result')",
        },
        {
            "md": (
                "## 11. Optional activity: does a larger learning rate help? · [Concept]\n\n"
                "**Predict → Change → Run → Observe → Explain.** **Predict:** with `ACTIVITY_LEARNING_RATE = 2e-4` (10× the default), "
                "will the held-out IoU rise, fall or stay the same under each prompt condition? **Change:** tick `RUN_ACTIVITY` and "
                "set the rate. **Run** this cell: it fine-tunes a fresh copy of the base. **Observe** the table — pretrained, "
                "default, activity — for both conditions and the records improved. **Explain** what you see in terms of headroom. "
                "The activity writes only to `outputs/activity/`; the canonical outputs and the adapter are untouched."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_LEARNING_RATE = 2e-4  # @param {{type:\"number\"}}\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', learning_rate=ACTIVITY_LEARNING_RATE)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** whether the larger rate moves either condition, and in which direction.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "No hosted run of the activity is recorded yet, so compare your rows. Under point+box there is almost no room to "
                "gain, and point-only sits at ceiling too, so a larger rate can mostly only hurt either condition; read the direction "
                "per record, not the size. A rate that improves the training loss but lowers held-out IoU is overfitting 18 easy scenes.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| `No CUDA GPU detected, and this notebook's fine-tuning needs one` | a CPU-only runtime | *Runtime → Change runtime type → T4 GPU* (Colab) or turn on a GPU accelerator (Kaggle), then *Run all*. |\n"
        "| `This notebook needs a Linux x86_64 runtime` | macOS, Windows or ARM kernel | Use Colab, Kaggle or a Linux x86_64 Jupyter kernel. |\n"
        "| `Not enough free disk` | a used runtime | Start a fresh runtime. |\n"
        "| `uv … wheel size/hash mismatch` or a `--require-hashes` error | a corrupted or substituted download | Re-run Section 2; never remove a pin or a hash. |\n"
        "| a size or SHA-256 mismatch in Section 3 | a file that is not the pinned one | Delete `weights/sam2.1-hiera-small/` and re-run Section 3; never edit the manifest. |\n"
        "| `You are using a model of type sam2_video to instantiate a model of type sam2` | the published config names the video model | Expected: image-mode loading of the same weights is the intended use. |\n"
        "| `Stage '…' failed …: … is missing: run the stage that writes it` | a cell run out of order | Run the notebook in order from Section 4 (or *Run all*). |\n"
        "| `BYOD_ZIP_PATH is empty, and the upload dialog exists only in Google Colab` | BYOD outside Colab with no path | Set `BYOD_ZIP_PATH`. |\n"
        "| `/kaggle/input holds N ZIP files, not exactly one` | several attached ZIPs | Set `BYOD_ZIP_PATH` to the one you mean. |\n"
        "| `The upload was cancelled or empty` / `Upload exactly one file` | a cancelled or multi-file upload | Run the cell again and choose one ZIP. |\n"
        "| `<zip>: expected images/ and masks/ at the top level or inside one wrapping folder` | another layout | Put `images/` and `masks/` at the top level, or in one folder. |\n"
        "| `<zip>: masks must be PNG` | JPEG masks (lossy, so edges blur) | Save masks as PNG, white = foreground. |\n"
        "| `<zip>: every image needs a mask with the same stem; images without a mask: […]` | unmatched names | Rename so each `images/<stem>` has `masks/<stem>.png`. |\n"
        "| `<zip>: unsafe member paths` / `… exceeds the 256 MiB BYOD ceiling` / `… 130-member` | an unsafe or oversized archive | Rebuild the ZIP with relative paths and at most 64 pairs. |\n"
        "| `<zip>: found N image/mask pair(s); the E2E path needs at least 8` | too few pairs | Add pairs. |\n"
        "| `the groups cannot be split into at least 2 training and 2 held-out records` | `GROUP_SEPARATOR` merges everything into one or two groups | Use a separator that matches your scene prefixes, or add scenes. |\n"
        "| `The adapter attached to a fresh base does not reproduce the adapting process` | a broken export or a different base | Do not ship the adapter; re-run from Section 7. |\n\n"
        "## Interpretation and limits\n\n"
        "This run proves the exact notebook can validate paired masks, update the declared adapter surface from the pinned base, "
        "score a fixed held-out split per record under two prompt conditions, serialize only the adapter, attach it to the exact "
        "pinned base in a fresh process, and reproduce inference after reload. On the default point+box condition the pretrained "
        "model is already at ceiling, so that comparison is a mechanics check: it cannot show that adaptation helps, and the "
        "notebook says so. The point-only condition is at ceiling on these scenes as well, so it cannot show it either. "
        "Nothing here shows improvement on photographs or any deployment domain. Real use requires rights-cleared representative "
        "images, human-reviewed masks, leakage-safe (grouped) splits, and boundary-error analysis.\n\n"
        "Successful execution proves that the recorded repository revision can complete this bounded tutorial without the "
        "repository being reachable at runtime. It does **not** establish benchmark superiority or production fitness.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three sentences: how much headroom the pretrained model left under each prompt condition; what the per-record "
        "comparison does and does not show about the fine-tune; and what you would need before claiming that adaptation improves "
        "segmentation on your own images.\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "Under point+box prompts the pretrained model scored 0.998254 mean IoU on six held-out scenes, leaving almost no "
        "headroom, so the adapted 0.998457 (+0.0002) is a mechanics check: the hypernetwork trained, the adapter exported, and a "
        "fresh base plus the adapter reproduced the masks exactly. Point-only prompts were near ceiling too, so neither "
        "condition shows whether adaptation helps. Before claiming improvement on my own images I would need "
        "representative, human-reviewed masks, a grouped split, several seeds or folds for a dispersion estimate, and a "
        "condition where the pretrained model is clearly imperfect.\n\n"
        "</details>\n\n"
        "**Next experiments:** run the activity at 2e-4 and at 2e-6; bring your own ZIP with `GROUP_SEPARATOR` set and compare "
        "the two prompt conditions; change `EPOCHS` and watch the per-epoch held-out IoU.\n\n"
        "## References\n\n"
        "- Repository: https://github.com/kurtvalcorza/sam2-segmentation-pipeline\n"
        "- Repository model card: https://github.com/kurtvalcorza/sam2-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- SAM 2 paper: https://arxiv.org/abs/2408.00714"
    ),
}
