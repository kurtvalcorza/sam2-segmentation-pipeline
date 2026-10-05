"""Stage runner for the standalone SAM 2.1 segmentation E2E tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --outputs OUTPUTS --weights WEIGHTS --stage data --options '{...}'

Nothing is installed into the notebook kernel. Each stage is a separate process and starts from files only: the
verified snapshot under ``--weights``, the records of earlier stages under ``RUN_DIR/state`` (each record as an RGB PNG,
a mask PNG and its prompts in JSON), and the learner-facing exports under ``--outputs``. The fine-tune always starts
from the pinned base, and the adapted model reaches later stages only through the exported adapter, which the
``evaluate`` and ``unseen`` stages attach to a fresh base in their own processes. On failure a stage writes
``RUN_DIR/state/<stage>.error.json``, which the notebook re-raises in the kernel.

Stages: weights → data → validate → baseline → adapt → evaluate → unseen → result, plus the optional ``activity``. The
``data`` and ``validate`` stages import no model library, so CI exercises them directly. No stage asserts a quality
level; only a broken reload contract (a fresh base plus the adapter not reproducing the adapting process) stops the run.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import importlib
import io
import json
import math
import random
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any

STEM = "sam2_segmentation"
PACKAGE = "sam2_segmentation_pipeline"
SEED = 42
GENERATED_RECORDS = 24
GENERATED_TRAIN = 18  # generated-00..17 train, generated-18..23 held out (the recorded split)
HELD_OUT_FRACTION = 0.25
MIN_RECORDS = 8
ARTIFACT_DIRNAME = "sam2-mask-hypernetwork-adapter-v1"
RELOAD_RTOL, RELOAD_ATOL = 1e-5, 1e-6
SATURATED_IOU = 0.99  # a held-out mean IoU at or above this leaves no room to show an adaptation effect
BYOD_ZIP_MAX_BYTES = 256 * 1024 * 1024
BYOD_MAX_MEMBERS = 130
BYOD_MAX_UNCOMPRESSED = 512 * 1024 * 1024
BYOD_MAX_TOTAL_PIXELS = 32 * 1024 * 1024
CONDITIONS = {
    "point+box": "the shape's exact bounding box and one interior point (the recorded condition)",
    "point-only": "one interior point, no box (harder: the model must find the extent itself)",
}


class Run:
    """Paths of one run: carried sources and state under ``root``; learner-facing files under ``outputs``."""

    def __init__(self, root: Path, outputs: Path, weights: Path, options: dict[str, Any]) -> None:
        self.root = root
        self.out = outputs
        self.weights = weights
        self.options = options
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook in order from Section 4)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path


def package(root: Path):
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(f"{PACKAGE}.pipeline")  # the module: it also defines MANIFEST_NAME and ARTIFACT_MANIFEST_NAME


def snapshot_dir(run: Run, P) -> Path:
    return run.weights / P.MODEL_KEY


def rounded(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


def source_record(run: Run) -> dict[str, Any]:
    path = run.root / "source.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


# --------------------------------------------------------------------------------------------------
# records (data and validate import no model library)
# --------------------------------------------------------------------------------------------------


def record_from_pair(record_id: str, image, mask_image) -> dict[str, Any]:
    import numpy as np

    image = image.convert("RGB")
    mask = np.asarray(mask_image.convert("L")) > 127
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError(f"{record_id}: the mask is empty (no pixel above 127); every mask needs foreground")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    point = [float(np.median(xs)), float(np.median(ys))]
    if not mask[int(point[1]), int(point[0])]:
        point = [float(xs[0]), float(ys[0])]
    return {"id": record_id, "image": image, "mask": mask, "points": [point], "point_labels": [1], "box": [float(x0), float(y0), float(x1), float(y1)]}


def generated_records(start: int = 0, count: int = GENERATED_RECORDS) -> list[dict[str, Any]]:
    """The deterministic tutorial scenes: one bright rounded rectangle or ellipse and one distractor on a flat dark
    background (high contrast on purpose: see Section 6 for what that means for the comparison)."""
    import numpy as np
    from PIL import Image, ImageDraw

    records = []
    for index in range(start, start + count):
        rng = np.random.default_rng(8100 + index)
        image = Image.new("RGB", (256, 256), tuple(int(v) for v in rng.integers(35, 95, 3)))
        draw = ImageDraw.Draw(image)
        mask_image = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask_image)
        x0, y0 = int(rng.integers(28, 105)), int(rng.integers(28, 105))
        width, height = int(rng.integers(70, 125)), int(rng.integers(65, 120))
        box = [x0, y0, min(244, x0 + width), min(244, y0 + height)]
        colour = tuple(int(v) for v in rng.integers(145, 245, 3))
        if index % 2:
            draw.ellipse(box, fill=colour)
            mask_draw.ellipse(box, fill=255)
        else:
            radius = 18 + index % 12
            draw.rounded_rectangle(box, radius=radius, fill=colour)
            mask_draw.rounded_rectangle(box, radius=radius, fill=255)
        distractor = [int(rng.integers(5, 55)), int(rng.integers(175, 215)), int(rng.integers(70, 120)), int(rng.integers(225, 250))]
        draw.rectangle(distractor, fill=tuple(int(v) for v in rng.integers(90, 180, 3)))
        records.append(record_from_pair(f"generated-{index:02d}", image, mask_image))
    return records


def records_from_zip(P, blob: bytes, name: str) -> list[dict[str, Any]]:
    """Read paired ``images/<stem>.(png|jpg|jpeg)`` and ``masks/<stem>.png`` members (optionally under one wrapping
    folder) without extracting; every refusal names the archive and the rule (SAM-m1)."""
    from PIL import Image

    if len(blob) > BYOD_ZIP_MAX_BYTES:
        raise ValueError(f"{name}: {len(blob)} bytes exceeds the 256 MiB BYOD ceiling")
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        raise ValueError(f"{name}: not a ZIP archive") from None
    with archive:
        infos = archive.infolist()
        if len(infos) > BYOD_MAX_MEMBERS:
            raise ValueError(f"{name}: {len(infos)} members exceeds the {BYOD_MAX_MEMBERS}-member BYOD ceiling (at most 64 image/mask pairs)")
        if sum(i.file_size for i in infos) > BYOD_MAX_UNCOMPRESSED:
            raise ValueError(f"{name}: the members expand to more than 512 MiB; refusing to read it")
        if any(i.flag_bits & 1 for i in infos):
            raise ValueError(f"{name}: encrypted members are not supported")
        names = [i.filename.replace("\\", "/") for i in infos if not i.is_dir()]
        unsafe = [n for n in names if n.startswith("/") or ".." in n.split("/")]
        if unsafe:
            raise ValueError(f"{name}: unsafe member paths {unsafe[:3]} (absolute or containing '..'); refusing the archive")
        prefix = ""
        if not any(n.startswith(("images/", "masks/")) for n in names):
            tops = sorted({n.split("/")[0] for n in names if "/" in n})
            wrapped = [t for t in tops if any(n.startswith(f"{t}/images/") for n in names) and any(n.startswith(f"{t}/masks/") for n in names)]
            if len(wrapped) == 1:
                prefix = wrapped[0] + "/"
                print({"byod_wrapping_folder": wrapped[0], "note": "pairs read from inside this single wrapping folder"})
            else:
                raise ValueError(f"{name}: expected images/ and masks/ at the top level or inside one wrapping folder; top-level folders found: {tops or 'none'}")
        originals = {i.filename.replace("\\", "/"): i.filename for i in infos}
        images: dict[str, str] = {}
        masks: dict[str, str] = {}
        non_png_masks = []
        for n in names:
            if not n.startswith(prefix):
                continue
            rel = n[len(prefix):]
            stem, _, ext = rel.split("/")[-1].rpartition(".")
            if rel.startswith("images/") and ext.lower() in ("png", "jpg", "jpeg"):
                if stem in images:
                    raise ValueError(f"{name}: duplicate image stem {stem!r}")
                images[stem] = originals[n]
            elif rel.startswith("masks/"):
                if ext.lower() != "png":
                    non_png_masks.append(rel)
                    continue
                if stem in masks:
                    raise ValueError(f"{name}: duplicate mask stem {stem!r}")
                masks[stem] = originals[n]
        if non_png_masks:
            raise ValueError(f"{name}: masks must be PNG (`masks/<stem>.png`, lossless, white = foreground); found {non_png_masks[:3]}")
        if set(images) != set(masks):
            raise ValueError(f"{name}: every image needs a mask with the same stem; images without a mask: {sorted(set(images) - set(masks))[:5]}, masks without an image: {sorted(set(masks) - set(images))[:5]}")
        if len(images) < MIN_RECORDS:
            raise ValueError(f"{name}: found {len(images)} image/mask pair(s); the E2E path needs at least {MIN_RECORDS} (6 to train, 2 to hold out)")
        records, total = [], 0
        for stem in sorted(images):
            image = Image.open(io.BytesIO(archive.read(images[stem])))
            mask = Image.open(io.BytesIO(archive.read(masks[stem])))
            if min(image.size) < P.MIN_IMAGE_SIDE or max(image.size) > P.MAX_IMAGE_SIDE or image.size != mask.size:
                raise ValueError(f"{name}: {stem}: image {image.size} and mask {mask.size} must match and lie within {P.MIN_IMAGE_SIDE}..{P.MAX_IMAGE_SIDE} px")
            total += image.width * image.height
            if total > BYOD_MAX_TOTAL_PIXELS:
                raise ValueError(f"{name}: the images exceed the 32-megapixel total decoded-pixel ceiling")
            image.load()
            mask.load()
            records.append(record_from_pair(stem, image, mask))
        return records


def group_split(records: list[dict[str, Any]], seed: int, separator: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Seeded, group-aware held-out split for BYOD (SAM-m1): records sharing a group (the stem up to ``separator``) stay
    on one side; without a separator every record is its own group (independent images assumed)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        key = r["id"].split(separator)[0] if separator else r["id"]
        groups.setdefault(key, []).append(r)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    target = max(2, round(len(records) * HELD_OUT_FRACTION))
    held: list[dict[str, Any]] = []
    for key in order:
        if len(held) >= target:
            break
        if len(records) - len(held) - len(groups[key]) < 2:
            continue
        held.extend(groups[key])
    held_ids = {r["id"] for r in held}
    train = [r for r in records if r["id"] not in held_ids]
    if len(held) < 2 or len(train) < 2:
        raise ValueError(f"the groups cannot be split into at least 2 training and 2 held-out records ({len(groups)} groups); add images or check GROUP_SEPARATOR={separator!r}")
    note = f"seeded (seed {seed}) shuffle of {len(groups)} group(s) by the stem prefix before {separator!r}; a group never straddles the split" if separator else f"seeded (seed {seed}) shuffle of {len(records)} records, each its own group (independent images assumed: set GROUP_SEPARATOR if frames or crops share a scene)"
    return train, held, note


def save_records(run: Run, records: list[dict[str, Any]], split: str, rows: list[dict[str, Any]]) -> None:
    from PIL import Image

    folder = run.state / "records"
    folder.mkdir(exist_ok=True)
    for r in records:
        index = len(rows)
        r["image"].save(folder / f"{index:03d}-image.png")
        Image.fromarray(r["mask"].astype("uint8") * 255).save(folder / f"{index:03d}-mask.png")
        rows.append({"id": r["id"], "index": index, "split": split, "points": r["points"], "point_labels": r["point_labels"], "box": r["box"]})


def load_records(run: Run, split: str, needed_by: str) -> list[dict[str, Any]]:
    import numpy as np
    from PIL import Image

    rows = run.read_state("records.json", needed_by)
    out = []
    for row in rows:
        if row["split"] != split:
            continue
        image = Image.open(run.state / "records" / f"{row['index']:03d}-image.png").convert("RGB")
        mask = np.asarray(Image.open(run.state / "records" / f"{row['index']:03d}-mask.png").convert("L")) > 127
        out.append({"id": row["id"], "image": image, "mask": mask, "points": row["points"], "point_labels": row["point_labels"], "box": row["box"]})
    return out


def condition(records: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return records if name == "point+box" else [{**r, "box": None} for r in records]


# --------------------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    P = package(run.root)
    snapshot = snapshot_dir(run, P)
    snapshot.mkdir(parents=True, exist_ok=True)
    carried = run.root / "weights" / P.MODEL_KEY / P.MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (P.MODEL_ID, P.MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity carried by the package; regenerate the notebook")
    shutil.copyfile(carried, snapshot / P.MANIFEST_NAME)
    print({"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION, "license": P.MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]})
    fetched = P.stage_missing_files(snapshot, allow_download=True)
    print({"weights_dir": str(snapshot), "fetched": fetched})
    verified = P.verify_snapshot(snapshot)
    print({"verified_files": verified["files"], "revision": verified["revision"], "total_bytes": verified.get("total_bytes")})
    run.write_state("weights.json", {"snapshot": str(snapshot), "fetched": fetched})


# --------------------------------------------------------------------------------------------------
# data and validation (SAM-m1)
# --------------------------------------------------------------------------------------------------


def find_byod_zip(opts: dict[str, Any]) -> Path:
    raw = str(opts.get("byod_zip_path") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"BYOD_ZIP_PATH {raw!r} is not a file in this runtime: upload the ZIP or correct the path.")
        return path
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        candidates = sorted(kaggle.rglob("*.zip"))
        if len(candidates) != 1:
            raise ValueError(f"BYOD_ZIP_PATH is empty and /kaggle/input holds {len(candidates)} ZIP files, not exactly one: set BYOD_ZIP_PATH explicitly.")
        print({"byod_zip": str(candidates[0]), "note": "the one ZIP attached under /kaggle/input"})
        return candidates[0]
    raise FileNotFoundError("BYOD_ZIP_PATH is empty: set it to your ZIP (or, in Colab, leave it empty to open the upload dialog).")


def stage_data(run: Run) -> None:
    P = package(run.root)
    opts = run.options
    use_byod = bool(opts.get("use_byod", False))
    seed = int(opts.get("seed", SEED))
    separator = str(opts.get("group_separator") or "")
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
    for name in ("records.json", "data.json", "baseline.json", "adapt.json", "evaluate.json", "unseen.json"):
        (run.state / name).unlink(missing_ok=True)
    shutil.rmtree(run.state / "records", ignore_errors=True)
    if use_byod:
        path = find_byod_zip(opts)
        if path.stat().st_size > BYOD_ZIP_MAX_BYTES:
            raise ValueError(f"{path.name}: {path.stat().st_size} bytes exceeds the 256 MiB BYOD ceiling")
        records = records_from_zip(P, path.read_bytes(), path.name)
        train, held, split_note = group_split(records, seed, separator)
        unseen = [held[0]]
        unseen_source = f"held-out BYOD record {held[0]['id']!r} (also scored in Section 8)"
        kind, origin = "BYOD", path.name
    else:
        records = generated_records()
        train, held = records[:GENERATED_TRAIN], records[GENERATED_TRAIN:]
        split_note = "fixed generated split: generated-00..17 train, generated-18..23 held out (the recorded split)"
        unseen = generated_records(start=GENERATED_RECORDS, count=1)
        unseen_source = f"a separately generated scene ({unseen[0]['id']}) outside the train and held-out index range"
        kind, origin = "generated", "deterministic generated shape scenes (seeds 8100..8123)"
    rows: list[dict[str, Any]] = []
    save_records(run, train, "train", rows)
    save_records(run, held, "held_out", rows)
    save_records(run, unseen, "unseen", rows)
    run.write_state("records.json", rows)
    run.write_state("data.json", {"kind": kind, "origin": origin, "seed": seed, "split": split_note, "unseen_source": unseen_source, "unseen_id": unseen[0]["id"]})
    print({"dataset_kind": kind, "origin": origin, "records": len(records), "train": len(train), "held_out": len(held)})
    print({"split": split_note})
    print({"unseen": unseen_source})


def stage_validate(run: Run) -> None:
    P = package(run.root)
    data = run.read_state("data.json", "validate")
    train = load_records(run, "train", "validate")
    held = load_records(run, "held_out", "validate")
    train_manifest = P.validate_segmentation_dataset(train)
    held_manifest = P.validate_segmentation_dataset(held)
    overlap = sorted({r["id"] for r in train} & {r["id"] for r in held})
    if overlap:
        raise ValueError(f"train/held-out leakage: ids {overlap} are on both sides")
    manifest = {"kind": data["kind"], "split": data["split"], "train": train_manifest, "validation": held_manifest, "overlap_ids": overlap}
    run.write_output(f"{STEM}_dataset_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("train", "validation")}, indent=2))
    print({"train_records": train_manifest.get("records"), "held_out_records": held_manifest.get("records"), "written": f"outputs/{STEM}_dataset_manifest.json"})


# --------------------------------------------------------------------------------------------------
# model stages
# --------------------------------------------------------------------------------------------------


def base_pipeline(run: Run, P):
    return P.SAM2SegmentationPipeline.from_pretrained(weights_dir=snapshot_dir(run, P), allow_download=False)


def adapted_pipeline(run: Run, P):
    adapt = run.read_state("adapt.json", "adapter reload")
    return P.SAM2SegmentationPipeline.from_artifact(Path(adapt["artifact_dir"]), weights_dir=snapshot_dir(run, P)), adapt


def evaluate_conditions(pipe, held: list[dict[str, Any]]) -> dict[str, Any]:
    return {name: pipe.evaluate_adaptation(condition(held, name)) for name in CONDITIONS}


def stage_baseline(run: Run) -> None:
    P = package(run.root)
    held = load_records(run, "held_out", "baseline")
    pipe = base_pipeline(run, P)
    results = evaluate_conditions(pipe, held)
    for name, result in results.items():
        print({"condition": name, "mean_mask_iou": round(result["mean_mask_iou"], 6), "per_record": [round(v, 4) for v in result["per_record_mask_iou"]], "box_baseline_mean_iou": rounded(result["box_baseline_mean_iou"])})
    run.write_state("baseline.json", {"ids": [r["id"] for r in held], "results": results, "device": pipe.device})


def stage_adapt(run: Run) -> None:
    import time

    import numpy as np

    P = package(run.root)
    epochs = int(run.options.get("epochs", 2))
    learning_rate = float(run.options.get("learning_rate", 2e-5))
    seed = int(run.options.get("seed", SEED))
    train = load_records(run, "train", "adapt")
    held = load_records(run, "held_out", "adapt")
    unseen = load_records(run, "unseen", "adapt")[0]
    pipe = base_pipeline(run, P)  # always from the pinned base, so a re-run compares cleanly (SAM-S3)
    started = time.perf_counter()
    counts = pipe.freeze_for_adaptation()
    history = pipe.finetune(train, held, epochs=epochs, learning_rate=learning_rate, seed=seed)
    seconds = round(time.perf_counter() - started, 2)
    print({"parameters": counts, "device": pipe.device, "seconds": seconds, "weight_delta_l2": pipe.adaptation_config["weight_delta_l2"]})
    for entry in history:
        print(rounded(entry))
    # The adapting process's own prediction on the unseen record is the reference Section 9's fresh reload must reproduce.
    reference = pipe.segment(unseen["image"], points=unseen["points"], point_labels=unseen["point_labels"], box=unseen["box"], multimask=False)
    np.save(run.state / "unseen_reference_mask.npy", reference["masks"])
    artifact = run.out / ARTIFACT_DIRNAME
    if artifact.exists():
        if not artifact.is_dir():
            raise RuntimeError(f"notebook-owned artifact path is not a directory: {artifact}")
        shutil.rmtree(artifact)
    revision = source_record(run).get("revision") or ""
    if len(revision) != 40:
        raise RuntimeError("source.json does not record the 40-hex repository revision the adapter must name as its producer; regenerate the notebook from the repository")
    artifact = pipe.save_artifact(artifact, producer_revision=revision)
    run.write_state("adapt.json", {"settings": {"epochs": epochs, "learning_rate": learning_rate, "seed": seed}, "parameters": counts, "history": history, "seconds": seconds, "weight_delta_l2": pipe.adaptation_config["weight_delta_l2"], "adaptation_config": pipe.adaptation_config, "artifact_dir": str(artifact), "unseen_reference_scores": reference["iou_scores"], "device": pipe.device})
    print({"adapter": f"outputs/{ARTIFACT_DIRNAME}", "note": "Sections 8-10 attach this adapter to a fresh pinned base in their own processes"})


def interpretation(base: float, adapted: float, improved: int, n: int, name: str) -> str:
    if base >= SATURATED_IOU:
        return f"{name}: no headroom — the pretrained model already scores {base:.4f} mean IoU, so the adapted {adapted:.4f} ({adapted - base:+.4f}, {improved}/{n} records improved) is a mechanics check, not evidence that adaptation helped"
    return f"{name}: pretrained {base:.4f} → adapted {adapted:.4f} ({adapted - base:+.4f}); {improved} of {n} held-out records improved. One holdout of {n} records with no dispersion estimate: read the direction and the per-record pattern, not the size"


def stage_evaluate(run: Run) -> None:
    P = package(run.root)
    data = run.read_state("data.json", "evaluate")
    baseline = run.read_state("baseline.json", "evaluate")
    held = load_records(run, "held_out", "evaluate")
    pipe, adapt = adapted_pipeline(run, P)
    adapted = evaluate_conditions(pipe, held)
    summary, readings = {}, []
    for name in CONDITIONS:
        before = baseline["results"][name]["per_record_mask_iou"]
        after = adapted[name]["per_record_mask_iou"]
        rows = [{"id": i, "pretrained": round(b, 6), "adapted": round(a, 6), "change": round(a - b, 6)} for i, b, a in zip(baseline["ids"], before, after, strict=True)]
        improved = sum(a > b for b, a in zip(before, after, strict=True))
        base_mean, adapted_mean = baseline["results"][name]["mean_mask_iou"], adapted[name]["mean_mask_iou"]
        reading = interpretation(base_mean, adapted_mean, improved, len(rows), name)
        readings.append(reading)
        summary[name] = {"prompts": CONDITIONS[name], "pretrained_mean_iou": base_mean, "adapted_mean_iou": adapted_mean, "change": adapted_mean - base_mean, "records_improved": improved, "records": len(rows), "per_record": rows, "headroom_before": 1.0 - base_mean}
        print({"condition": name, "pretrained_mean_iou": round(base_mean, 6), "adapted_mean_iou": round(adapted_mean, 6), "change": round(adapted_mean - base_mean, 6), "records_improved": f"{improved}/{len(rows)}"})
        for row in rows:
            print(row)
    box_baseline = adapted["point+box"]["box_baseline_mean_iou"]
    print({"prompt_box_fill_baseline_mean_iou": rounded(box_baseline), "note": "filling the prompt box shows what prompting adds; it is not the adaptation comparison"})
    for reading in readings:
        print({"interpretation": reading})
    report = {
        "task": "promptable-image-segmentation-adaptation",
        "verdict": "sample-sanity",
        "dataset_kind": data["kind"],
        "estimation": data["split"],
        "conditions": summary,
        "baseline_pretrained": baseline["results"]["point+box"],
        "adapted": adapted["point+box"],
        "prompt_box_fill_baseline_mean_iou": box_baseline,
        "interpretation": readings,
        "weights_moved_vs_quality_changed": "weight_delta_l2 > 0 shows the parameters moved; only the held-out per-record change can show whether quality changed",
        "weight_delta_l2": adapt["weight_delta_l2"],
        "computed_by": "a fresh process that attached the exported adapter to a fresh pinned base",
    }
    run.write_output(f"{STEM}_evaluation_report.json", report)
    run.write_state("evaluate.json", {name: {k: summary[name][k] for k in ("pretrained_mean_iou", "adapted_mean_iou", "records_improved", "records")} for name in summary})
    print({"report": f"outputs/{STEM}_evaluation_report.json"})


def stage_unseen(run: Run) -> None:
    import numpy as np

    P = package(run.root)
    data = run.read_state("data.json", "unseen")
    unseen = load_records(run, "unseen", "unseen")[0]
    pipe, adapt = adapted_pipeline(run, P)
    result = pipe.segment(unseen["image"], points=unseen["points"], point_labels=unseen["point_labels"], box=unseen["box"], multimask=False)
    iou = P.mask_iou(result["masks"][0], unseen["mask"])
    print({"record": data["unseen_source"], "mask_iou_sample_sanity": round(iou, 6), "model_iou_score": round(result["iou_scores"][0], 6)})
    reference_mask = np.load(run.state / "unseen_reference_mask.npy")
    mask_exact = bool(np.array_equal(reference_mask, result["masks"]))
    scores_close = bool(np.allclose(adapt["unseen_reference_scores"], result["iou_scores"], rtol=RELOAD_RTOL, atol=RELOAD_ATOL))
    summary = {"verification": "PASSED" if mask_exact and scores_close else "FAILED", "process": "fresh (separate from the adapting stage)", "mask_exact": mask_exact, "scores_within_tolerance": scores_close, "score_rtol": RELOAD_RTOL, "score_atol": RELOAD_ATOL, "artifact_dir": adapt["artifact_dir"]}
    run.write_state("unseen.json", {"id": unseen["id"], "source": data["unseen_source"], "mask_iou": iou, "model_iou_score": result["iou_scores"][0], "reload": summary})
    if not (mask_exact and scores_close):
        raise RuntimeError(f"The adapter attached to a fresh base does not reproduce the adapting process (mask exact: {mask_exact}, scores within rtol {RELOAD_RTOL}: {scores_close}). Do not ship this artifact.")
    manifest = json.loads((Path(adapt["artifact_dir"]) / P.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    print({"artifact_format": manifest.get("format"), "files": sorted(p.name for p in Path(adapt["artifact_dir"]).iterdir())})
    print(summary)


def stage_result(run: Run) -> None:
    import importlib.metadata
    import platform

    P = package(run.root)
    data = run.read_state("data.json", "result")
    adapt = run.read_state("adapt.json", "result")
    unseen = run.read_state("unseen.json", "result")
    source = source_record(run)
    payload = {
        "dataset": json.loads((run.out / f"{STEM}_dataset_manifest.json").read_text(encoding="utf-8")),
        "adaptation": adapt["adaptation_config"],
        "evaluation": json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8")),
        "unseen": {"id": unseen["id"], "source": unseen["source"], "mask_iou": unseen["mask_iou"]},
        "artifact": unseen["reload"],
        "notebook_source": source,
        "repository_revision": source.get("revision"),
        "base_model": {"id": P.MODEL_ID, "revision": P.MODEL_REVISION, "license": P.MODEL_LICENSE},
        "runtime": {"python": platform.python_version(), **{d: importlib.metadata.version(d) for d in ("torch", "transformers", "safetensors")}, "device": adapt["device"]},
    }
    run.write_output(f"{STEM}_result.json", payload)
    print({"status": "E2E COMPLETE", "dataset_kind": data["kind"], "optimizer_steps": sum(item.get("optimizer_steps", 0) for item in adapt["history"]), "weight_delta_l2": round(adapt["weight_delta_l2"], 6), "reload": unseen["reload"]["verification"], "runtime": payload["runtime"]})
    print("outputs/:")
    for path in sorted(run.out.rglob("*")):
        if path.is_file():
            print(f"  - {path.relative_to(run.out.parent).as_posix()} ({path.stat().st_size / 1024:.1f} KB)")


def stage_activity(run: Run) -> None:
    """SAM-M3: re-run the fine-tune from the pinned base with a different learning rate and print both prompt conditions
    beside the pretrained and default results; writes only to outputs/activity/."""
    P = package(run.root)
    adapt = run.read_state("adapt.json", "activity")
    baseline = run.read_state("baseline.json", "activity")
    default = run.read_state("evaluate.json", "activity")
    train = load_records(run, "train", "activity")
    held = load_records(run, "held_out", "activity")
    learning_rate = float(run.options.get("learning_rate", 2e-4))
    before = sorted(p.name for p in run.out.glob(f"{STEM}_*"))
    pipe = base_pipeline(run, P)
    pipe.freeze_for_adaptation()
    history = pipe.finetune(train, held, epochs=adapt["settings"]["epochs"], learning_rate=learning_rate, seed=adapt["settings"]["seed"])
    results = evaluate_conditions(pipe, held)
    rows = []
    for name in CONDITIONS:
        rows.append({"condition": name, "pretrained": round(baseline["results"][name]["mean_mask_iou"], 6), f"default (lr {adapt['settings']['learning_rate']:g})": round(default[name]["adapted_mean_iou"], 6), f"activity (lr {learning_rate:g})": round(results[name]["mean_mask_iou"], 6), "activity_records_improved": f"{sum(a > b for a, b in zip(results[name]['per_record_mask_iou'], baseline['results'][name]['per_record_mask_iou'], strict=True))}/{len(held)}"})
    for row in rows:
        print(row)
    print({"activity_history": rounded(history), "weight_delta_l2": round(pipe.adaptation_config["weight_delta_l2"], 6)})
    run.write_output(f"activity/{STEM}_activity_lr_{learning_rate:g}.json", {"learning_rate": learning_rate, "rows": rows, "history": history, "weight_delta_l2": pipe.adaptation_config["weight_delta_l2"]})
    if sorted(p.name for p in run.out.glob(f"{STEM}_*")) != before:
        raise RuntimeError("the activity changed the canonical outputs; it must write only to outputs/activity/")
    print(f"wrote outputs/activity/{STEM}_activity_lr_{learning_rate:g}.json (the canonical outputs and the adapter are unchanged)")


STAGES = {
    "weights": stage_weights,
    "data": stage_data,
    "validate": stage_validate,
    "baseline": stage_baseline,
    "adapt": stage_adapt,
    "evaluate": stage_evaluate,
    "unseen": stage_unseen,
    "result": stage_result,
    "activity": stage_activity,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    run = Run(args.root.resolve(), args.outputs.resolve(), args.weights.resolve(), json.loads(args.options))
    error_file = run.state / f"{args.stage}.error.json"
    error_file.unlink(missing_ok=True)
    try:
        STAGES[args.stage](run)
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        traceback.print_exc()
        error_file.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
