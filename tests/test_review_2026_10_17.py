"""Regression tests for the 2026-10-17 review (SAM2-M1, SAM2-m4, SAM2-m5, SAM2-m6, NOTEBOOK_SPEC 2.5)."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re

import numpy as np

from test_notebook_review_fixes import NB, ROOT, STAGES, _nb, _pairs, _quiet, _run, _src, _zip

STEM = STAGES.STEM
REFUSED = "not run: the Section 4 data stage refused its input"


def _main(run, stage: str, options: dict) -> tuple[int, str]:
    argv = ["--root", str(run.root), "--outputs", str(run.out), "--weights", str(run.weights)]
    argv += ["--stage", stage, "--options", json.dumps(options)]
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = STAGES.main(argv)
    return code, out.getvalue()


# ---------------------------------------------------------------- SAM2-M1: visible-region ground truth


def test_SAM2_M1_every_generated_mask_pixel_shows_the_target_colour() -> None:
    for record in STAGES.generated_records(0, STAGES.GENERATED_RECORDS + 1):
        pixels = np.asarray(record["image"]).reshape(-1, 3)[record["mask"].reshape(-1)]
        assert len(np.unique(pixels, axis=0)) == 1, record["id"]
        x, y = record["points"][0]
        assert record["mask"][int(y), int(x)], record["id"]
        ys, xs = np.where(record["mask"])
        assert record["box"] == [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def test_SAM2_M1_images_are_unchanged_and_only_the_two_occluded_masks_shrink() -> None:
    records = {r["id"]: r for r in STAGES.generated_records(0, STAGES.GENERATED_RECORDS + 1)}
    # The distractor covers 13 and 48 target pixels of these two scenes; those pixels are no longer target.
    assert int(records["generated-15"]["mask"].sum()) == 8661
    assert int(records["generated-22"]["mask"].sum()) == 6956
    # The scenes are the pre-fix scenes byte for byte (raw RGB SHA-256 prefixes recorded at 0b0c7ba).
    ids = ("generated-15", "generated-22")
    digests = {k: hashlib.sha256(np.asarray(records[k]["image"]).tobytes()).hexdigest()[:16] for k in ids}
    assert digests == {"generated-15": "b52331564f92021b", "generated-22": "0e45d528f7474fc3"}


# ---------------------------------------------------------------- SAM2-m4: one message; earlier work is kept


def test_SAM2_m4_refusal_is_one_message_keeps_prior_outputs_and_blocks_later_stages(tmp_path) -> None:
    run = _run(tmp_path)
    _quiet(STAGES.stage_data, run)  # an earlier, valid default run
    adapter = run.out / STAGES.ARTIFACT_DIRNAME
    adapter.mkdir()
    (adapter / "adapter.safetensors").write_bytes(b"earlier run")
    (run.out / f"{STEM}_evaluation_report.json").write_text("{}")
    bad = tmp_path / "mine.zip"
    extra = _pairs(["x"])["images/x.png"]
    bad.write_bytes(_zip({**_pairs([f"s{i}" for i in range(8)]), "images/extra.png": extra}))

    code, out = _main(run, "data", {"use_byod": True, "byod_zip_path": str(bad)})
    assert code == 2 and "Traceback" not in out
    assert "mine.zip" in out and "images without a mask: ['extra']" in out
    error = json.loads((run.state / "data.error.json").read_text())
    assert error["refusal"] is True and error["type"] == "Refusal"
    # Earlier work is not destroyed, but the run is no longer data-ready and the marker lists what is stale.
    assert (adapter / "adapter.safetensors").read_bytes() == b"earlier run"
    assert (run.out / f"{STEM}_evaluation_report.json").is_file()
    assert not (run.state / "data.json").exists() and not (run.state / "records.json").exists()
    marker = json.loads((run.state / STAGES.REFUSED_STATE).read_text())
    assert f"{STAGES.ARTIFACT_DIRNAME}/" in marker["stale_outputs"]

    for stage in ("validate", "baseline", "adapt", "evaluate", "unseen", "result", "activity"):
        code, out = _main(run, stage, {})
        assert code == 2 and "Traceback" not in out and REFUSED in out, stage

    _quiet(STAGES.stage_data, run)  # a corrected data stage clears the refusal
    assert not (run.state / STAGES.REFUSED_STATE).exists() and (run.state / "data.json").is_file()


def test_SAM2_m4_a_genuine_failure_keeps_its_traceback(tmp_path, monkeypatch) -> None:
    run = _run(tmp_path)

    def broken(_run):
        raise KeyError("a bug, not a refusal")

    monkeypatch.setitem(STAGES.STAGES, "data", broken)
    code, out = _main(run, "data", {})
    assert code == 1 and "Traceback" in out
    assert "refusal" not in json.loads((run.state / "data.error.json").read_text())


def test_SAM2_m4_kernel_reports_a_refusal_without_raising() -> None:
    cells = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "code"]
    install = next(c for c in cells if "def run_stage(" in c)
    assert "error.get('refusal')" in install and "return False" in install
    assert "Fix the named file" in install


# ---------------------------------------------------------------- SAM2-m5, SAM2-m6


def test_SAM2_m5_section_1_does_not_call_the_gpu_optional() -> None:
    assert not [c for c in _nb()["cells"] if "A GPU is recommended, not required" in _src(c)]


def test_SAM2_m6_finetune_target_is_not_dilated() -> None:
    source = (ROOT / "src" / "sam2_segmentation_pipeline" / "pipeline.py").read_text(encoding="utf-8")
    assert not re.search(r'mode="area"\s*\)\s*\[0,\s*0\]\s*>\s*0\b', source)
    assert "target_low = fraction >= 0.5" in source and "if not bool(target_low.any()):" in source


# ---------------------------------------------------------------- NOTEBOOK_SPEC 2.5 (KND1, DOC13-DOC17)


def test_spec_2_5_kind_notice_and_numbered_references() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    before = [_src(c) for c in nb["cells"][:first_code] if c["cell_type"] == "markdown"]
    assert any("> **⚠️ Notice.**" in c for c in before)
    md = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    body, refs = md.split("## References", 1)
    entries = set(re.findall(r"^\[(\d+)\] ", refs, re.M))
    assert entries and entries == set(re.findall(r"\[(\d+)\]", body))
    assert "SA-1B" in md and "arxiv.org/abs/2408.00714" in refs
