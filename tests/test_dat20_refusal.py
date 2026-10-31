"""DAT19/DAT20: a refused BYOD input shows exactly one `Refused: <file>: <reason>` line and stops Run all.

Runs the committed kernel `run_stage` (from the notebook's install cell) against the carried stage runner in a
subprocess, as the notebook does. Covers the scripted hosted refusal input (`BYOD_ZIP_PATH='/etc/hostname'`: a
file that is not a ZIP), a directory, a cancelled upload, the later-stage stop, and a genuine failure.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_notebook_review_fixes import NB, ROOT, _run

REFUSAL_CELL = "sam2_segmentation-14"


def _cells() -> list[dict]:
    return [c for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


@pytest.fixture
def kernel(tmp_path):
    run = _run(tmp_path)
    shutil.copy(ROOT / "tools" / "tutorial_stages.py", run.root / "tutorial_stages.py")
    install = next("".join(c["source"]) for c in _cells() if "def run_stage(" in "".join(c["source"]))
    block = install[install.index("class StageRefusal(") : install.index("def load_record(")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    ns = {"ROOT": run.root, "OUTPUTS": run.out, "WEIGHTS": run.weights, "PYTHON": Path(sys.executable)}
    ns.update(ENV=env, json=json, subprocess=subprocess)
    exec(compile(block, "<install>", "exec"), ns)
    return ns


def _refused(kernel, capsys, **options) -> str:
    with pytest.raises(kernel["StageRefusal"]) as caught:
        kernel["run_stage"]("data", use_byod=True, **options)
    assert caught.value.dimer_refusal is True
    rendered = caught.value._render_traceback_()
    printed = capsys.readouterr().out
    assert "Traceback" not in printed and "Refused" not in printed, printed  # the stage does not echo it
    assert len(rendered) == 1 and "\n" not in rendered[0], rendered
    assert rendered[0].startswith("Refused: ") and "Traceback" not in rendered[0]
    return rendered[0]


def test_scripted_hosted_input_a_file_that_is_not_a_zip(kernel, capsys, tmp_path):
    hostname = tmp_path / "hostname"
    hostname.write_text("colab-vm\n")
    line = _refused(kernel, capsys, byod_zip_path=str(hostname))
    assert line.startswith("Refused: hostname: not a ZIP archive. Fix the named file"), line


def test_a_directory_and_a_cancelled_upload(kernel, capsys, tmp_path):
    line = _refused(kernel, capsys, byod_zip_path=str(tmp_path))
    assert line.startswith(f"Refused: {tmp_path}: BYOD_ZIP_PATH is not a file in this runtime"), line
    cancelled = "The upload was cancelled or empty: no file was received."
    line = _refused(kernel, capsys, acquisition_error=cancelled, acquisition_file="(no file)")
    assert line.startswith("Refused: (no file): The upload was cancelled or empty"), line
    line = _refused(
        kernel, capsys, acquisition_error="Upload exactly one file; got 2.", acquisition_file="a.zip, b.zip"
    )
    assert line.startswith("Refused: a.zip, b.zip: Upload exactly one file"), line


def test_later_stages_stop_cleanly_after_a_refusal(kernel, capsys, tmp_path):
    _refused(kernel, capsys, byod_zip_path=str(tmp_path / "missing.zip"))
    with pytest.raises(kernel["StageNotRun"]) as caught:
        kernel["run_stage"]("validate")
    rendered = caught.value._render_traceback_()
    assert len(rendered) == 1 and rendered[0].startswith(
        "not run: the Section 4 data stage refused its input"
    )


def test_genuine_failure_keeps_runtime_error(kernel):
    with pytest.raises(RuntimeError, match="failed"):
        kernel["run_stage"]("no-such-stage")


def test_refusal_cell_id():
    ids = [c["id"] for c in _cells() if "USE_BYOD = False  # @param" in "".join(c["source"])]
    assert ids == [REFUSAL_CELL]
