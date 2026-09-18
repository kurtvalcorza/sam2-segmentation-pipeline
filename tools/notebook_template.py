"""E2E notebook template for SAM 2.1 promptable-segmentation adaptation."""

# ruff: noqa: E501

TEMPLATE = {
    "package": "sam2_segmentation_pipeline",
    "repo_name": "sam2-segmentation-pipeline",
    "stem": "sam2_segmentation",
    "notebook_name": "sam2_segmentation_colab.ipynb",
    "profile": "E2E",
    "run_all": (
        "Selecting **Run all** on a fresh CUDA runtime installs the pinned dependencies, verifies the exact base "
        "snapshot, generates and validates 24 records, freezes the base, performs the bounded gradient update, scores "
        "the held-out split, predicts an unseen scene, exports the adapter, reloads it over a fresh base, verifies "
        "numeric equivalence, and writes provenance without a clone, credential, upload, or configuration edit."
    ),
    "byod": (
        "Set `USE_BYOD = True` and either provide `BYOD_ZIP_PATH` (including an attached Kaggle input path) or use "
        "the Colab upload fallback. The records pass through the same validation, split, adaptation, evaluation, "
        "export, and reload path as the generated dataset."
    ),
    "pipeline_class": "SAM2SegmentationPipeline",
    "weights_key": "sam2.1-hiera-small",
    "runtime_imports": ["torch", "transformers", "safetensors"],
    "title": "SAM 2.1 Hiera-Small — DIMER end-to-end promptable segmentation fine-tuning",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/sam2-segmentation-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/sam2-segmentation-pipeline/blob/main/tutorials/sam2_segmentation_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fsam2.1--hiera--small-ffcc4d?style=flat",
            "https://huggingface.co/facebook/sam2.1-hiera-small",
        ),
    ],
    "capability": "pinned SAM 2.1 inference plus bounded mask-hypernetwork gradient adaptation, held-out mask-IoU evaluation, safe adapter export, and fresh reload",
    "intro": (
        "This notebook performs a real local gradient update. It freezes the Hiera encoder, prompt encoder, and the rest "
        "of the mask decoder, then trains only the first mask-token output hypernetwork (139,808 parameters) with BCE "
        "plus soft-Dice loss. The default dataset is 24 deterministic generated shape scenes split 18/6 before model "
        "execution. A SafeTensors adapter and closed SHA-256 manifest are exported and attached to a fresh pinned base "
        "model. These are tutorial sample-sanity results, not a segmentation benchmark."
    ),
    "learning_objectives": (
        "validate paired image/mask/prompt data, preserve a held-out split, measure the pretrained baseline, run bounded "
        "AdamW adaptation, evaluate mask IoU against a prompt-box baseline, infer on unseen data, export a base-bound "
        "SafeTensors adapter, and verify it after a fresh-model reload."
    ),
    "exclusions": (
        "full-model training, video tracking, automatic segment-everything, semantic class learning, production-scale "
        "training, or benchmark claims. The adapter changes one image-mode mask hypernetwork only."
    ),
    "prerequisites": [
        "- **Runtime:** fresh Python 3.12 with an NVIDIA T4-class GPU or better. CUDA is required for the default fine-tuning path. The pinned 184 MB checkpoint is acquired automatically.",
        "- **Knowledge:** Python, binary masks, IoU, train/validation separation, and adapter-versus-base semantics.",
        "- **Data:** the default path generates 24 shape scenes. Optional BYOD accepts a ZIP with `images/` and `masks/` files paired by stem; set `BYOD_ZIP_PATH` to an attached file (for Kaggle, normally under `/kaggle/input/`) or use the Colab upload fallback. It is off by default and follows the same E2E path. Do not upload confidential or restricted data unless you are authorized to use it in the hosted runtime.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Generate or upload a paired segmentation dataset\n\n"
                "The default 24 deterministic 256×256 scenes contain varied rounded rectangles and ellipses plus "
                "distractors. Each record has an RGB image, boolean mask, foreground point, and bounding box. Records "
                "0–17 train; 18–23 are held out. BYOD reads an explicit ZIP path, auto-discovers one attached Kaggle "
                "ZIP, or falls back to the Colab upload dialog. It uses matching `images/<id>` and `masks/<id>.png` "
                "members and rejects unsafe archive paths before decoding."
            ),
            "code": (
                "import io\n"
                "import zipfile\n\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_ZIP_PATH = ''  # @param {{type:\"string\"}}; Kaggle inputs are normally under /kaggle/input/\n\n"
                "BYOD_ZIP_MAX_BYTES = 256 * 1024 * 1024\n\n"
                "def record_from_pair(record_id, image, mask_image):\n"
                "    image = image.convert('RGB')\n"
                "    mask = np.asarray(mask_image.convert('L')) > 127\n"
                "    ys, xs = np.where(mask)\n"
                "    if len(xs) == 0:\n"
                "        raise ValueError(f'{{record_id}} has an empty mask')\n"
                "    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1\n"
                "    point = [float(np.median(xs)), float(np.median(ys))]\n"
                "    if not mask[int(point[1]), int(point[0])]:\n"
                "        point = [float(xs[0]), float(ys[0])]\n"
                "    return {{'id': record_id, 'image': image, 'mask': mask, 'points': [point], 'point_labels': [1], 'box': [float(x0), float(y0), float(x1), float(y1)]}}\n\n"
                "def generated_records(start=0, count=24):\n"
                "    records = []\n"
                "    for index in range(start, start + count):\n"
                "        rng = np.random.default_rng(8100 + index)\n"
                "        image = Image.new('RGB', (256, 256), tuple(int(v) for v in rng.integers(35, 95, 3)))\n"
                "        draw = ImageDraw.Draw(image)\n"
                "        mask_image = Image.new('L', image.size, 0)\n"
                "        mask_draw = ImageDraw.Draw(mask_image)\n"
                "        x0, y0 = int(rng.integers(28, 105)), int(rng.integers(28, 105))\n"
                "        width, height = int(rng.integers(70, 125)), int(rng.integers(65, 120))\n"
                "        box = [x0, y0, min(244, x0 + width), min(244, y0 + height)]\n"
                "        colour = tuple(int(v) for v in rng.integers(145, 245, 3))\n"
                "        if index % 2:\n"
                "            draw.ellipse(box, fill=colour); mask_draw.ellipse(box, fill=255)\n"
                "        else:\n"
                "            radius = 18 + index % 12\n"
                "            draw.rounded_rectangle(box, radius=radius, fill=colour)\n"
                "            mask_draw.rounded_rectangle(box, radius=radius, fill=255)\n"
                "        distractor = [int(rng.integers(5, 55)), int(rng.integers(175, 215)), int(rng.integers(70, 120)), int(rng.integers(225, 250))]\n"
                "        draw.rectangle(distractor, fill=tuple(int(v) for v in rng.integers(90, 180, 3)))\n"
                "        records.append(record_from_pair(f'generated-{{index:02d}}', image, mask_image))\n"
                "    return records\n\n"
                "def records_from_zip(blob):\n"
                "    if len(blob) > BYOD_ZIP_MAX_BYTES:\n"
                "        raise ValueError('BYOD ZIP exceeds the 256 MiB upload ceiling')\n"
                "    with zipfile.ZipFile(io.BytesIO(blob)) as archive:\n"
                "        infos = archive.infolist()\n"
                "        names = archive.namelist()\n"
                "        normalized = [name.replace('\\\\', '/') for name in names]\n"
                "        if len(names) > 130 or sum(info.file_size for info in infos) > 512 * 1024 * 1024:\n"
                "            raise ValueError('unsafe or oversized BYOD archive')\n"
                "        if any(info.flag_bits & 1 for info in infos) or any(name.startswith('/') or '..' in name.split('/') for name in normalized):\n"
                "            raise ValueError('unsafe or oversized BYOD archive')\n"
                "        image_entries = [(name.split('/')[-1].rsplit('.', 1)[0], original) for name, original in zip(normalized, names) if name.startswith('images/') and name.lower().endswith(('.png', '.jpg', '.jpeg'))]\n"
                "        mask_entries = [(name.split('/')[-1].rsplit('.', 1)[0], original) for name, original in zip(normalized, names) if name.startswith('masks/') and name.lower().endswith('.png')]\n"
                "        images, masks = dict(image_entries), dict(mask_entries)\n"
                "        if len(images) != len(image_entries) or len(masks) != len(mask_entries):\n"
                "            raise ValueError('BYOD archive contains duplicate image or mask stems')\n"
                "        if set(images) != set(masks):\n"
                "            raise ValueError('BYOD image and mask stems must match exactly')\n"
                "        records = []\n"
                "        for key in sorted(images):\n"
                "            image = Image.open(io.BytesIO(archive.read(images[key])))\n"
                "            mask = Image.open(io.BytesIO(archive.read(masks[key])))\n"
                "            if min(image.size) < MIN_IMAGE_SIDE or max(image.size) > MAX_IMAGE_SIDE or min(mask.size) < MIN_IMAGE_SIDE or max(mask.size) > MAX_IMAGE_SIDE or image.size != mask.size:\n"
                "                raise ValueError(f'{{key}} image/mask dimensions are mismatched or outside {{MIN_IMAGE_SIDE}}..{{MAX_IMAGE_SIDE}} px')\n"
                "            image.load(); mask.load()\n"
                "            records.append(record_from_pair(key, image, mask))\n"
                "        return records\n\n"
                "def read_bounded_byod_zip(path):\n"
                "    size = path.stat().st_size\n"
                "    if size > BYOD_ZIP_MAX_BYTES:\n"
                "        raise ValueError(f'BYOD ZIP exceeds the 256 MiB upload ceiling: {{size}} bytes')\n"
                "    return path.read_bytes()\n\n"
                "def load_byod_zip():\n"
                "    if BYOD_ZIP_PATH.strip():\n"
                "        path = Path(BYOD_ZIP_PATH).expanduser()\n"
                "        if not path.is_file():\n"
                "            raise FileNotFoundError(f'BYOD ZIP not found: {{path}}')\n"
                "        return read_bounded_byod_zip(path)\n"
                "    kaggle_input = Path('/kaggle/input')\n"
                "    if kaggle_input.is_dir():\n"
                "        candidates = sorted(kaggle_input.rglob('*.zip'))\n"
                "        if len(candidates) != 1:\n"
                "            raise ValueError(f'expected exactly one attached Kaggle ZIP, found {{len(candidates)}}; set BYOD_ZIP_PATH explicitly')\n"
                "        print({{'byod_zip': str(candidates[0])}})\n"
                "        return read_bounded_byod_zip(candidates[0])\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError as exc:\n"
                "        raise RuntimeError('set BYOD_ZIP_PATH to a readable ZIP outside Colab') from exc\n"
                "    uploaded = files.upload()\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError(f'expected exactly one BYOD ZIP upload, found {{len(uploaded)}}')\n"
                "    return next(iter(uploaded.values()))\n\n"
                "if USE_BYOD:\n"
                "    dataset_records = records_from_zip(load_byod_zip())\n"
                "    dataset_kind = 'BYOD'\n"
                "else:\n"
                "    dataset_records = generated_records()\n"
                "    dataset_kind = 'generated'\n"
                "if len(dataset_records) < 8:\n"
                "    raise ValueError('E2E adaptation requires at least 8 records')\n"
                "split_at = max(2, int(len(dataset_records) * 0.75))\n"
                "train_records, val_records = dataset_records[:split_at], dataset_records[split_at:]\n"
                "print({{'dataset_kind': dataset_kind, 'records': len(dataset_records), 'train': len(train_records), 'held_out': len(val_records)}})"
            ),
        },
        {
            "md": "## 5. Validate the dataset and split\n\nBoth splits must have unique IDs, exact image/mask alignment, boolean non-empty masks, valid prompts, and deterministic content fingerprints. Cross-split IDs are rejected.",
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "train_manifest = validate_segmentation_dataset(train_records)\n"
                "val_manifest = validate_segmentation_dataset(val_records)\n"
                "overlap = set(r['id'] for r in train_records) & set(r['id'] for r in val_records)\n"
                "if overlap:\n"
                "    raise RuntimeError(f'train/validation leakage: {{sorted(overlap)}}')\n"
                "dataset_manifest = {{'kind': dataset_kind, 'train': train_manifest, 'validation': val_manifest, 'overlap_ids': []}}\n"
                "with open('outputs/{stem}_dataset_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(dataset_manifest, handle, indent=2)\n"
                "print(json.dumps(dataset_manifest, indent=2))"
            ),
        },
        {
            "md": "## 6. Measure the pretrained held-out baseline\n\nBefore any update, score the exact holdout and compare with the prompt-box baseline.",
            "code": (
                "base_eval = pipe.evaluate_adaptation(val_records)\n"
                "print({{key: round(value, 6) if isinstance(value, float) else value for key, value in base_eval.items() if key != 'per_record_mask_iou'}})"
            ),
        },
        {
            "md": "## 7. Freeze the base and run bounded fine-tuning\n\nOnly the first mask-token output hypernetwork is trainable. AdamW runs two epochs with batch size one. A non-zero L2 weight delta is mandatory.",
            "code": (
                "if not torch.cuda.is_available():\n"
                "    raise RuntimeError('The canonical SAM2 E2E path requires a CUDA GPU')\n"
                "parameter_counts = pipe.freeze_for_adaptation()\n"
                "history = pipe.finetune(train_records, val_records, epochs=2, learning_rate=2e-5, seed=42)\n"
                "print({{'parameters': parameter_counts, 'history': history, 'weight_delta_l2': pipe.adaptation_config['weight_delta_l2']}})"
            ),
        },
        {
            "md": "## 8. Evaluate the adapted model\n\nScore the untouched holdout after training. The report is sample-sanity, not benchmark evidence.",
            "code": (
                "adapted_eval = pipe.evaluate_adaptation(val_records)\n"
                "split_estimation = 'fixed generated held-out split' if dataset_kind == 'generated' else 'caller-provided BYOD held-out split'\n"
                "evaluation_report_e2e = {{'task': 'promptable-image-segmentation-adaptation', 'verdict': 'sample-sanity', 'dataset_kind': dataset_kind, 'estimation': split_estimation, 'baseline_pretrained': base_eval, 'adapted': adapted_eval, 'weight_delta_l2': pipe.adaptation_config['weight_delta_l2']}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(evaluation_report_e2e, handle, indent=2)\n"
                "print(json.dumps({{'baseline_mean_iou': base_eval['mean_mask_iou'], 'adapted_mean_iou': adapted_eval['mean_mask_iou'], 'box_baseline_mean_iou': adapted_eval['box_baseline_mean_iou']}}, indent=2))"
            ),
        },
        {
            "md": "## 9. Infer on an unseen scene\n\nA separately seeded generated record outside the train/validation index range exercises adapted serving.",
            "code": (
                "unseen = generated_records(start=24, count=1)[0]\n"
                "unseen_result = pipe.segment(unseen['image'], points=unseen['points'], point_labels=unseen['point_labels'], box=unseen['box'], multimask=False)\n"
                "unseen_iou = mask_iou(unseen_result['masks'][0], unseen['mask'])\n"
                "print({{'id': unseen['id'], 'mask_iou_sample_sanity': unseen_iou, 'model_iou_score': unseen_result['iou_scores'][0]}})"
            ),
        },
        {
            "md": "## 10. Export and verify a fresh reload\n\nExport SafeTensors plus a closed manifest. A fresh pinned base verifies the artifact; masks must match exactly and scores within the stated tolerance.",
            "code": (
                "artifact_dir = pipe.save_artifact('outputs/sam2-mask-hypernetwork-adapter-v1', producer_revision=NOTEBOOK_SOURCE['repository_revision'])\n"
                "reloaded_pipe = SAM2SegmentationPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR)\n"
                "reloaded_result = reloaded_pipe.segment(unseen['image'], points=unseen['points'], point_labels=unseen['point_labels'], box=unseen['box'], multimask=False)\n"
                "if not np.array_equal(unseen_result['masks'], reloaded_result['masks']):\n"
                "    raise RuntimeError('reloaded adapter changed the mask')\n"
                "if not np.allclose(unseen_result['iou_scores'], reloaded_result['iou_scores'], rtol=1e-5, atol=1e-6):\n"
                "    raise RuntimeError('reloaded adapter changed the score beyond tolerance')\n"
                "reload_summary = {{'verification': 'PASSED', 'mask_exact': True, 'score_rtol': 1e-5, 'score_atol': 1e-6, 'artifact_dir': str(artifact_dir)}}\n"
                "print(reload_summary)"
            ),
        },
        {
            "md": "## 11. Export provenance and terminal summary\n\nBind dataset fingerprints, hyperparameters, metrics, weight activity, base identity, runtime, and reload evidence without retaining records.",
            "code": (
                "payload = {{'dataset': dataset_manifest, 'adaptation': pipe.adaptation_config, 'evaluation': evaluation_report_e2e, 'unseen': {{'id': unseen['id'], 'mask_iou': unseen_iou}}, 'artifact': reload_summary, 'notebook_source': NOTEBOOK_SOURCE, 'repository_revision': NOTEBOOK_SOURCE['repository_revision'], 'base_model': {{'id': MODEL_ID, 'revision': MODEL_REVISION}}, 'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device}}}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2)\n"
                "print({{'status': 'E2E COMPLETE', 'train_records': len(train_records), 'held_out_records': len(val_records), 'optimizer_steps': sum(item['optimizer_steps'] for item in history), 'weight_delta_l2': pipe.adaptation_config['weight_delta_l2'], 'reload': reload_summary['verification'], 'outputs': sorted(os.listdir('outputs'))}})"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "This run proves the exact notebook can validate paired masks, update the declared adapter surface, score a held-out "
        "generated split, serialize only the adapter, attach it to the exact pinned base, and reproduce inference after "
        "reload. It does not prove improvement on photographs or any deployment domain. Real use requires rights-cleared "
        "representative images, human-reviewed masks, leakage-safe splits, and boundary-error analysis.\n\n"
        "Successful execution proves that the recorded repository revision can complete this bounded tutorial without the repository being reachable at runtime. It does **not** establish benchmark superiority or production fitness.\n\n"
        "## References\n\n"
        "- Repository: https://github.com/kurtvalcorza/sam2-segmentation-pipeline\n"
        "- Repository model card: https://github.com/kurtvalcorza/sam2-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- SAM 2 paper: https://arxiv.org/abs/2408.00714"
    ),
}
