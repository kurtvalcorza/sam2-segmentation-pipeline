"""Regression tests for the 2026-10-05 notebook review (SAM-M1..M3, SAM-m1..m2, SAM-S3).

They need numpy and pillow (CI's dependencies) but no model: they exec the notebook's own kernel cells with stand-ins for
`run_stage`, `google.colab` and `nvidia-smi`, run the stage runner's model-free stages (`data`, `validate`) and BYOD reader
in-process, and check the generated notebook statically. Each test names its finding.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
NB = ROOT / "tutorials" / "sam2_segmentation_colab.ipynb"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("sam_tutorial_stages", TOOLS / "tutorial_stages.py")
sys.path.insert(0, str(ROOT / "src"))
import sam2_segmentation_pipeline.pipeline as P  # noqa: E402


def _nb() -> dict:
    return json.loads(NB.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _cell_with(needle: str) -> str:
    found = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "code" and needle in _src(c)]
    assert len(found) == 1, needle
    return found[0]


def _set(src: str, name: str, value) -> str:
    out = []
    for line in src.split("\n"):
        if line.startswith(f"{name} = "):
            line = f"{name} = {value!r}" + (line[line.index("  #"):] if "  #" in line else "")
        out.append(line)
    return "\n".join(out)


@contextlib.contextmanager
def _colab(queue):
    files = types.SimpleNamespace(calls=0)

    def upload():
        files.calls += 1
        return queue.pop(0)

    files.upload = upload
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield files
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _kernel(tmp_path: Path) -> tuple[dict, list]:
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "Path": Path, "shutil": shutil, "run_stage": lambda stage, **o: calls.append((stage, o))}
    return ns, calls


def _run(tmp_path: Path, options: dict | None = None):
    root = tmp_path / "run"
    if not (root / "src").exists():
        root.mkdir(parents=True, exist_ok=True)
        (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    return STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", options or {})


def _quiet(fn, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*args)
    return out.getvalue()


def _png(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _pair(i: int) -> tuple[bytes, bytes]:
    img = np.full((32, 32, 3), 40, np.uint8)
    mask = np.zeros((32, 32), np.uint8)
    mask[8 + i % 4 : 20 + i % 4, 6:22] = 255
    img[mask > 0] = 220
    return _png(img), _png(mask)


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def _pairs(stems: list[str], prefix: str = "") -> dict[str, bytes]:
    out = {}
    for i, stem in enumerate(stems):
        image, mask = _pair(i)
        out[f"{prefix}images/{stem}.png"] = image
        out[f"{prefix}masks/{stem}.png"] = mask
    return out


@pytest.fixture
def fake_gpu(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = bin_dir / "nvidia-smi"
    tool.write_text("#!/bin/sh\necho 'Tesla T4, 15360 MiB'\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")


# ---------------------------------------------------------------- SAM-M1: isolated runtime


def test_M1_no_kernel_install_and_no_restart_instruction() -> None:
    text = NB.read_text(encoding="utf-8")
    assert "Restart the runtime" not in text and "restart the runtime" not in text
    own = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")]
    for src in own:
        assert not re.search(r"['\"]-m['\"]\s*,\s*['\"]pip['\"]|^\s*[%!]\s*pip\b|['\"]pip install", src, re.M), src[:80]
        assert "import torch" not in src and "from sam2_segmentation_pipeline" not in src
    install = _cell_with("# @title Infrastructure: build (or reuse) the isolated")
    assert "'--require-hashes'" in install and "--managed-python" in install
    assert "MPLBACKEND='Agg'" in install and "'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'" in install


def _exec_check_cell(ns: dict, **overrides) -> None:
    src = _cell_with("# @title Infrastructure: check the runtime")
    src = re.sub(r"'environment': [0-9.]+}", "'environment': 0.0}", src, count=1)
    src = re.sub(r"\{'weights': max\(0\.0, [0-9.]+", "{'weights': max(0.0, 0.0", src, count=1)
    for name, value in overrides.items():
        src = _set(src, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<check>", "exec"), ns)


def test_M1_section1_is_idempotent(tmp_path, monkeypatch, fake_gpu) -> None:
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns)
    first = ns["ROOT"]
    (first / "tutorial_stages.py").write_text("# carried")
    _exec_check_cell(ns)
    assert ns["ROOT"] == first and (first / "tutorial_stages.py").is_file()
    _exec_check_cell(ns, NEW_RUN_DIRECTORY=True)
    assert ns["ROOT"] != first


def test_M1_second_run_all_reuses_the_matching_environment(tmp_path, monkeypatch, fake_gpu) -> None:
    import subprocess as real_subprocess

    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns)
    ns["ENV_ROOT"] = tmp_path / "uvroot"
    ns["NOTEBOOK_SOURCE"] = {"revision": "test"}
    src = _cell_with("# @title Infrastructure: build (or reuse) the isolated")
    version = re.search(r"UV = ENV_ROOT / 'uv-([0-9.]+)'", src).group(1)
    ns["ENV_ROOT"].mkdir()
    (ns["ENV_ROOT"] / f"uv-{version}").write_bytes(b"uv stand-in")
    (ns["ENV_ROOT"] / f"uv-{version}.sha256").write_text(hashlib.sha256(b"uv stand-in").hexdigest())
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append([str(c) for c in cmd])
        if cmd[1] == "venv":
            python = Path(cmd[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("")
        out = json.dumps({"python": "3.12.12", "torch": "x", "transformers": "x", "safetensors": "x", "cuda": True})
        return types.SimpleNamespace(stdout=out + "\n", returncode=0)

    fake = types.SimpleNamespace(run=fake_run, Popen=real_subprocess.Popen, PIPE=real_subprocess.PIPE, STDOUT=real_subprocess.STDOUT)
    for attempt in range(2):
        ns["subprocess"] = fake
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src.replace("import zipfile\n", "import zipfile\nsubprocess = globals()['subprocess']\n", 1), "<install>", "exec"), ns)
        ns["subprocess"] = fake
        assert ns["environment_reused"] is (attempt == 1)
    assert len([c for c in commands if c[1] in ("venv", "pip")]) == 2


# ---------------------------------------------------------------- SAM-m2: GPU checked up front; warning explained


def test_m2_cpu_runtime_stops_in_section_1(tmp_path, monkeypatch) -> None:
    """SAM-m2: with no nvidia-smi the first code cell stops with the runtime-change instruction, before any download."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    with pytest.raises(RuntimeError, match=r"No CUDA GPU detected.*Change runtime type > T4 GPU.*Nothing has been downloaded yet"):
        _exec_check_cell({})


def test_m2_sam2_video_warning_is_explained() -> None:
    md = "\n".join(_src(c) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    assert "sam2_video" in md and "intended use" in md


# ---------------------------------------------------------------- SAM-M2: saturation disclosed; harder condition; per-record


def test_M2_saturation_is_disclosed_and_a_harder_condition_is_scored() -> None:
    md = "\n".join(_src(c) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    assert "the default task leaves almost nothing to adapt" in md and "0.998254" in md and "mechanics check" in md
    assert "point-only" in md and "No hosted run of the point-only condition is recorded yet" in md
    assert set(STAGES.CONDITIONS) == {"point+box", "point-only"}
    records = [{"id": "a", "box": [1.0, 1.0, 5.0, 5.0], "points": [[2.0, 2.0]], "point_labels": [1]}]
    assert STAGES.condition(records, "point-only")[0]["box"] is None and STAGES.condition(records, "point+box")[0]["box"] == [1.0, 1.0, 5.0, 5.0]


def test_M2_interpretation_says_no_headroom_at_ceiling_and_reads_per_record_otherwise() -> None:
    at_ceiling = STAGES.interpretation(0.998254, 0.998457, 4, 6, "point+box")
    assert "no headroom" in at_ceiling and "mechanics check" in at_ceiling and "+0.0002" in at_ceiling
    room = STAGES.interpretation(0.82, 0.86, 5, 6, "point-only")
    assert "no headroom" not in room and "5 of 6 held-out records improved" in room and "no dispersion estimate" in room


def test_M2_evaluate_reports_every_record_and_the_records_improved() -> None:
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_evaluate"):source.index("def stage_unseen")]
    assert "pipe, adapt = adapted_pipeline(run, P)" in body and '"per_record": rows' in body and '"records_improved": improved' in body
    assert "weights_moved_vs_quality_changed" in body


# ---------------------------------------------------------------- SAM-M3: guided layer, activity, verdicts

GUIDED = ("## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model → Output", "## Roadmap", "<summary><strong>Glossary</strong>", "Predict before running", "**What to notice:**", "Check your reasoning", "## Troubleshooting", "## Conclusion", "Sample conclusion")


def test_M3_guided_layer_and_collapsed_infrastructure() -> None:
    nb = _nb()
    md = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    for heading in GUIDED:
        assert heading in md, heading
    assert md.count("Predict before running") >= 3 and "{{" not in md
    infra = [c for c in nb["cells"] if c["cell_type"] == "code" and _src(c).startswith("# @title Infrastructure:")]
    assert len(infra) == 4 and all(c["metadata"].get("cellView") == "form" for c in infra)


def test_M3_learning_rate_activity_from_the_base() -> None:
    cell = _cell_with("RUN_ACTIVITY = False  # @param")
    assert "ACTIVITY_LEARNING_RATE = 2e-4  # @param" in cell and "run_stage('activity', learning_rate=ACTIVITY_LEARNING_RATE)" in cell
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_activity"):source.index("STAGES = {")]
    assert "pipe = base_pipeline(run, P)" in body and 'write_output(f"activity/' in body and "!= before" in body


def test_no_quality_assert_in_the_stage_runner() -> None:
    assert not [n for n in ast.walk(ast.parse((TOOLS / "tutorial_stages.py").read_text(encoding="utf-8"))) if isinstance(n, ast.Assert)]


def test_S3_finetune_always_starts_from_the_pinned_base() -> None:
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_adapt"):source.index("def interpretation")]
    assert "pipe = base_pipeline(run, P)" in body and "from_artifact" not in body


# ---------------------------------------------------------------- SAM-m1: BYOD


def test_m1_wrapping_folder_is_accepted(capsys) -> None:
    records = STAGES.records_from_zip(P, _zip(_pairs([f"s{i:02d}" for i in range(8)], prefix="mydata/")), "wrapped.zip")
    assert [r["id"] for r in records] == [f"s{i:02d}" for i in range(8)]
    assert "byod_wrapping_folder" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ({**_pairs([f"s{i}" for i in range(8)]), "images/extra.png": _pair(0)[0]}, r"images without a mask: \['extra'\]"),
        ({**{k: v for k, v in _pairs([f"s{i}" for i in range(8)]).items() if k != "masks/s3.png"}, "masks/s3.jpg": b"x"}, r"masks must be PNG"),
        ({**_pairs([f"s{i}" for i in range(8)]), "../evil.png": b"x"}, r"unsafe member paths \['\.\./evil\.png'\]"),
        (_pairs([f"s{i}" for i in range(7)]), r"found 7 image/mask pair\(s\); the E2E path needs at least 8"),
        ({"a/images/x.png": b"1", "a/masks/x.png": b"1", "b/images/y.png": b"1", "b/masks/y.png": b"1"}, r"top-level folders found: \['a', 'b'\]"),
    ],
    ids=["unmatched-stem", "jpeg-mask", "traversal", "too-few", "two-wrappers"],
)
def test_m1_refusals_name_the_archive_and_the_rule(members, message) -> None:
    with pytest.raises(ValueError, match=r"mine\.zip: .*" + message):
        STAGES.records_from_zip(P, _zip(members), "mine.zip")


def test_m1_size_and_traversal_messages_differ() -> None:
    with pytest.raises(ValueError, match=r"exceeds the 256 MiB BYOD ceiling"):
        STAGES.records_from_zip(P, b"0" * (STAGES.BYOD_ZIP_MAX_BYTES + 1), "big.zip")


def test_m1_split_is_seeded_and_group_aware() -> None:
    records = [{"id": f"scene{g}_{k}"} for g in range(6) for k in range(3)]
    train, held, note = STAGES.group_split(records, 42, "_")
    groups = lambda rows: {r["id"].split("_")[0] for r in rows}  # noqa: E731
    assert not (groups(train) & groups(held)) and len(held) >= 3 and "a group never straddles the split" in note
    assert STAGES.group_split(records, 42, "_")[1] == held and STAGES.group_split(records, 7, "_")[1] != held
    _train, held_independent, note2 = STAGES.group_split(records, 42, "")
    assert groups(held_independent) & groups(_train) and "independent images assumed" in note2


def test_m1_byod_unseen_record_is_a_held_out_byod_record(tmp_path) -> None:
    path = tmp_path / "mine.zip"
    path.write_bytes(_zip(_pairs([f"s{i:02d}" for i in range(12)])))
    run = _run(tmp_path, {"use_byod": True, "byod_zip_path": str(path), "seed": 42})
    _quiet(STAGES.stage_data, run)
    rows = json.loads((run.state / "records.json").read_text())
    held = [r["id"] for r in rows if r["split"] == "held_out"]
    unseen = [r["id"] for r in rows if r["split"] == "unseen"]
    assert len(held) == 3 and unseen == [held[0]]
    assert "held-out BYOD record" in json.loads((run.state / "data.json").read_text())["unseen_source"]


def test_m1_byod_path_and_upload_branches(tmp_path, monkeypatch) -> None:
    cell = _cell_with("USE_BYOD = False  # @param")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns, calls = _kernel(tmp_path)
    exec(compile(_set(_set(cell, "USE_BYOD", True), "BYOD_ZIP_PATH", "/d/mine.zip"), "<s4>", "exec"), ns)
    assert calls[0] == ("data", {"use_byod": True, "byod_zip_path": "/d/mine.zip", "group_separator": "", "seed": 42})
    if not Path("/kaggle/input").is_dir():
        ns, calls = _kernel(tmp_path)
        with pytest.raises(RuntimeError, match="BYOD_ZIP_PATH is empty, and the upload dialog exists only in Google Colab"):
            exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
        monkeypatch.undo()
        with _colab([{}]) as files:
            ns, calls = _kernel(tmp_path)
            with pytest.raises(RuntimeError, match="cancelled or empty"):
                exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
            assert files.calls == 1 and calls == []


# ---------------------------------------------------------------- default path, model-free stages


def test_default_data_and_validate_stages_keep_the_recorded_split(tmp_path) -> None:
    run = _run(tmp_path, {"use_byod": False})
    _quiet(STAGES.stage_data, run)
    rows = json.loads((run.state / "records.json").read_text())
    by = {s: [r["id"] for r in rows if r["split"] == s] for s in ("train", "held_out", "unseen")}
    assert by["train"] == [f"generated-{i:02d}" for i in range(18)] and by["held_out"] == [f"generated-{i:02d}" for i in range(18, 24)] and by["unseen"] == ["generated-24"]
    _quiet(STAGES.stage_validate, run)
    manifest = json.loads((run.out / "sam2_segmentation_dataset_manifest.json").read_text())
    assert manifest["overlap_ids"] == [] and manifest["kind"] == "generated"
    held = STAGES.load_records(run, "held_out", "t")
    fresh = STAGES.generated_records()[18:]
    assert all(np.array_equal(a["mask"], b["mask"]) and a["box"] == b["box"] for a, b in zip(held, fresh, strict=True))


def test_reload_is_a_contract_check_against_the_adapting_process() -> None:
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_unseen"):source.index("def stage_result")]
    assert "pipe, adapt = adapted_pipeline(run, P)" in body and "Do not ship this artifact" in body
    assert (STAGES.RELOAD_RTOL, STAGES.RELOAD_ATOL) == (1e-5, 1e-6)
