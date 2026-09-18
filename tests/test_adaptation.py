from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from sam2_segmentation_pipeline.pipeline import (
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_WEIGHTS_NAME,
    SAM2SegmentationPipeline,
    _segmentation_record_content_sha256,
    validate_segmentation_dataset,
)


class _Batch(dict):
    def to(self, device):
        return _Batch({key: value.to(device) for key, value in self.items()})


class _Processor:
    def __call__(self, *, images, return_tensors, **kwargs):
        assert return_tensors == "pt"
        values = torch.ones(1, 1, 4, 4)
        batch = _Batch(pixel_values=values)
        if "input_points" in kwargs:
            batch["input_points"] = torch.tensor(kwargs["input_points"], dtype=torch.float32)
            batch["input_labels"] = torch.tensor(kwargs["input_labels"], dtype=torch.long)
        if "input_boxes" in kwargs:
            batch["input_boxes"] = torch.tensor(kwargs["input_boxes"], dtype=torch.float32)
        return batch

    def post_process_masks(self, pred_masks, original_sizes, *, mask_threshold):
        height, width = (int(value) for value in original_sizes[0])
        resized = torch.nn.functional.interpolate(
            pred_masks[:, 0], size=(height, width), mode="bilinear", align_corners=False
        )
        return [(resized > mask_threshold)]


class _FakeSAM2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)
        self.mask_decoder = torch.nn.Module()
        self.mask_decoder.output_hypernetworks_mlps = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, 1, 1)]
        )

    def get_image_embeddings(self, pixel_values):
        return [pixel_values]

    def forward(self, *, image_embeddings, multimask_output, **kwargs):
        assert multimask_output is False
        logits = self.mask_decoder.output_hypernetworks_mlps[0](image_embeddings[-1])
        return SimpleNamespace(pred_masks=logits.unsqueeze(1))


def _records() -> list[dict]:
    records = []
    for index in range(4):
        image = Image.new("RGB", (16, 16), (index * 20, 40, 80))
        mask = np.zeros((16, 16), dtype=np.bool_)
        mask[3:13, 4:12] = True
        records.append(
            {
                "id": f"shape-{index}",
                "image": image,
                "mask": mask,
                "points": [[8.0, 8.0]],
                "point_labels": [1],
                "box": [4.0, 3.0, 12.0, 13.0],
            }
        )
    return records


def _pipeline(model=None) -> SAM2SegmentationPipeline:
    model = model or _FakeSAM2()

    def runner(*args):
        return np.zeros((1, 16, 16), dtype=np.bool_), [0.0]

    return SAM2SegmentationPipeline(
        runner,
        "cpu",
        model=model,
        processor=_Processor(),
    )


def test_dataset_validation_rejects_duplicate_ids_and_non_boolean_masks():
    records = _records()
    assert validate_segmentation_dataset(records)["records"] == 4
    records[1]["id"] = records[0]["id"]
    with pytest.raises(ValueError, match="duplicate record id"):
        validate_segmentation_dataset(records)
    records = _records()
    records[0]["mask"] = records[0]["mask"].astype(np.uint8)
    with pytest.raises(TypeError, match="mask must be boolean"):
        validate_segmentation_dataset(records)

    records = _records()
    records[0]["points"] = [[0.0, 0.0]]
    with pytest.raises(ValueError, match="contradicts target mask"):
        validate_segmentation_dataset(records)


def test_dataset_fingerprint_includes_prompts():
    records = _records()
    original = validate_segmentation_dataset(records)["dataset_sha256"]
    records[0]["box"] = [3.0, 2.0, 13.0, 14.0]
    changed = validate_segmentation_dataset(records)["dataset_sha256"]
    assert changed != original


def test_dataset_validation_rejects_box_that_misses_target_foreground():
    records = _records()
    records[0]["box"] = [0.0, 0.0, 2.0, 2.0]
    with pytest.raises(ValueError, match="box does not overlap target mask foreground"):
        validate_segmentation_dataset(records)


def test_content_fingerprint_includes_image_and_mask_dimensions():
    pixels = np.arange(16 * 32 * 3, dtype=np.uint8)
    mask_values = np.arange(16 * 32) % 2 == 0
    wide = {
        "id": "wide",
        "image": Image.fromarray(pixels.reshape(16, 32, 3)),
        "mask": mask_values.reshape(16, 32),
        "points": [[0.0, 0.0]],
        "point_labels": [1],
    }
    tall = {
        "id": "tall",
        "image": Image.fromarray(pixels.reshape(32, 16, 3)),
        "mask": mask_values.reshape(32, 16),
        "points": [[0.0, 0.0]],
        "point_labels": [1],
    }
    assert _segmentation_record_content_sha256(wide) != _segmentation_record_content_sha256(
        tall
    )
    assert validate_segmentation_dataset([wide, dict(wide, id="wide-2")])[
        "dataset_sha256"
    ] != (
        validate_segmentation_dataset([tall, dict(tall, id="tall-2")])["dataset_sha256"]
    )


def test_finetune_updates_only_declared_adapter_surface():
    pipeline = _pipeline()
    counts = pipeline.freeze_for_adaptation()
    assert counts["trainable_parameters"] > 0
    assert pipeline.model.backbone.weight.requires_grad is False
    before = pipeline.model.mask_decoder.output_hypernetworks_mlps[0].weight.detach().clone()
    history = pipeline.finetune(_records()[:2], _records()[2:], epochs=1, learning_rate=1e-2)
    after = pipeline.model.mask_decoder.output_hypernetworks_mlps[0].weight.detach()
    assert history[0]["optimizer_steps"] == 2
    assert not torch.equal(before, after)
    assert pipeline.adaptation_config["weight_delta_l2"] > 0


def test_finetune_preserves_a_single_pixel_foreground_when_downsampling(monkeypatch):
    mask = np.zeros((16, 16), dtype=np.bool_)
    mask[1, 1] = True
    record = {
        "id": "single-pixel",
        "image": Image.new("RGB", (16, 16), (20, 40, 80)),
        "mask": mask,
        "points": [[1.0, 1.0]],
        "point_labels": [1],
        "box": [1.0, 1.0, 2.0, 2.0],
    }
    observed_targets = []
    original_bce = torch.nn.functional.binary_cross_entropy_with_logits

    def capture_target(logits, target, *args, **kwargs):
        observed_targets.append(target.detach().clone())
        return original_bce(logits, target, *args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "binary_cross_entropy_with_logits", capture_target
    )
    pipeline = _pipeline()
    pipeline.freeze_for_adaptation()
    pipeline.finetune([record, dict(record, id="single-pixel-2")], epochs=1, learning_rate=1e-2)
    assert observed_targets
    assert all(float(target.sum()) >= 1.0 for target in observed_targets)


def test_failed_retraining_restores_weights_and_completion_metadata(monkeypatch):
    pipeline = _pipeline()
    pipeline.freeze_for_adaptation()
    pipeline.adaptation_config.update({"weight_delta_l2": 1.0, "history": [{"epoch": 1}]})
    previous_config = dict(pipeline.adaptation_config)
    before = {
        name: parameter.detach().clone()
        for name, parameter in pipeline.model.named_parameters()
        if parameter.requires_grad
    }
    original_forward = pipeline.model.forward

    def non_finite_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        return SimpleNamespace(pred_masks=outputs.pred_masks * float("nan"))

    monkeypatch.setattr(pipeline.model, "forward", non_finite_forward)
    with pytest.raises(RuntimeError, match="non-finite loss"):
        pipeline.finetune(_records()[:2], epochs=1, learning_rate=1e-2)
    assert pipeline.adaptation_config == previous_config
    for name, parameter in pipeline.model.named_parameters():
        if name in before:
            assert torch.equal(parameter, before[name])


def test_finetune_rejects_train_validation_overlap_by_id_or_content():
    records = _records()
    with pytest.raises(ValueError, match="overlap by id"):
        _pipeline().finetune(records[:2], records[1:3], epochs=1)

    renamed = dict(records[0], id="renamed-shape")
    with pytest.raises(ValueError, match="overlap by image/mask content"):
        _pipeline().finetune(records[:2], [renamed, records[2]], epochs=1)

    reprompted = dict(
        records[0],
        id="reprompted-shape",
        points=[[6.0, 6.0]],
        box=[3.0, 2.0, 13.0, 14.0],
    )
    with pytest.raises(ValueError, match="overlap by image/mask content"):
        _pipeline().finetune(records[:2], [reprompted, records[2]], epochs=1)


def test_point_only_evaluation_omits_box_baseline():
    records = [dict(record, box=None) for record in _records()[:2]]
    report = _pipeline().evaluate_adaptation(records)
    assert report["box_baseline_records"] == 0
    assert report["box_prompt_model_mean_iou"] is None
    assert report["box_baseline_mean_iou"] is None
    assert report["delta_over_box_baseline"] is None


def test_mixed_prompt_baseline_uses_only_box_records_for_delta():
    records = _records()[:2]
    records[1] = dict(records[1], box=None)

    def runner(_image, _points, _labels, box, _multimask):
        mask = records[0]["mask"] if box is not None else np.zeros((16, 16), dtype=np.bool_)
        return mask[None], [0.0]

    pipeline = _pipeline()
    pipeline._runner = runner
    report = pipeline.evaluate_adaptation(records)
    assert report["box_baseline_records"] == 1
    assert report["box_prompt_model_mean_iou"] == 1.0
    assert report["mean_mask_iou"] == 0.5
    assert report["delta_over_box_baseline"] == pytest.approx(
        1.0 - report["box_baseline_mean_iou"]
    )


def test_fractional_box_baseline_covers_a_nonempty_pixel_region():
    record = _records()[0]
    mask = np.zeros((16, 16), dtype=np.bool_)
    mask[0, 0] = True
    record = dict(
        record,
        mask=mask,
        points=[[0.0, 0.0]],
        box=[0.1, 0.1, 0.4, 0.4],
    )
    pipeline = _pipeline()
    pipeline._runner = lambda *_args: (mask[None], [0.0])
    report = pipeline.evaluate_adaptation([record, dict(record, id="shape-fractional-2")])
    assert report["box_baseline_mean_iou"] == 1.0
    assert report["delta_over_box_baseline"] == 0.0


def test_artifact_round_trip_and_integrity_rejection(tmp_path):
    source = _pipeline()
    source.freeze_for_adaptation()
    source.adaptation_config.update({"weight_delta_l2": 1.0, "history": [{"epoch": 1}]})
    artifact = source.save_artifact(tmp_path / "artifact", producer_revision="a" * 40)

    fresh = _pipeline()
    fresh.load_artifact(artifact)
    expected = source.model.mask_decoder.output_hypernetworks_mlps[0].weight
    actual = fresh.model.mask_decoder.output_hypernetworks_mlps[0].weight
    assert torch.equal(expected, actual)

    unexpected = artifact / "unexpected.txt"
    unexpected.write_text("not declared", encoding="utf-8")
    with pytest.raises(ValueError, match="contain exactly"):
        _pipeline().load_artifact(artifact)
    unexpected.unlink()

    unexpected_dir = artifact / "retained-data"
    unexpected_dir.mkdir()
    with pytest.raises(ValueError, match="contain exactly"):
        _pipeline().load_artifact(artifact)
    unexpected_dir.rmdir()

    manifest_path = artifact / ARTIFACT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adaptation"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rejected = _pipeline()
    before = rejected.model.mask_decoder.output_hypernetworks_mlps[0].weight.detach().clone()
    with pytest.raises(ValueError, match="adaptation metadata"):
        rejected.load_artifact(artifact)
    after = rejected.model.mask_decoder.output_hypernetworks_mlps[0].weight.detach()
    assert torch.equal(before, after)

    manifest["adaptation"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rejected_empty = _pipeline()
    before_empty = (
        rejected_empty.model.mask_decoder.output_hypernetworks_mlps[0].weight.detach().clone()
    )
    with pytest.raises(ValueError, match="adaptation metadata"):
        rejected_empty.load_artifact(artifact)
    after_empty = rejected_empty.model.mask_decoder.output_hypernetworks_mlps[0].weight.detach()
    assert torch.equal(before_empty, after_empty)

    manifest["adaptation"] = source.adaptation_config
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        _pipeline().load_artifact(artifact)
    assert (artifact / ARTIFACT_WEIGHTS_NAME).is_file()

    malformed = source.save_artifact(tmp_path / "malformed", producer_revision="b" * 40)
    malformed_weights = malformed / ARTIFACT_WEIGHTS_NAME
    malformed_state = load_file(str(malformed_weights))
    tensor_name = next(name for name, tensor in malformed_state.items() if tensor.ndim > 1)
    malformed_state[tensor_name] = malformed_state[tensor_name].reshape(-1)
    save_file(malformed_state, str(malformed_weights))
    malformed_manifest_path = malformed / ARTIFACT_MANIFEST_NAME
    malformed_manifest = json.loads(malformed_manifest_path.read_text(encoding="utf-8"))
    malformed_manifest["files"][0]["bytes"] = malformed_weights.stat().st_size
    malformed_manifest["files"][0]["sha256"] = hashlib.sha256(
        malformed_weights.read_bytes()
    ).hexdigest()
    malformed_manifest_path.write_text(json.dumps(malformed_manifest), encoding="utf-8")
    rejected_shape = _pipeline()
    before_shape = {
        name: tensor.detach().clone() for name, tensor in rejected_shape.model.state_dict().items()
    }
    with pytest.raises(ValueError, match="shape/dtype"):
        rejected_shape.load_artifact(malformed)
    for name, tensor in rejected_shape.model.state_dict().items():
        assert torch.equal(tensor, before_shape[name])


def test_loaded_artifact_refreezes_base_and_supports_continued_finetuning(tmp_path):
    source = _pipeline()
    source.freeze_for_adaptation()
    source.adaptation_config.update({"weight_delta_l2": 1.0, "history": [{"epoch": 1}]})
    artifact = source.save_artifact(tmp_path / "artifact", producer_revision="c" * 40)

    loaded = _pipeline()
    loaded.load_artifact(artifact)
    assert loaded.model.backbone.weight.requires_grad is False
    assert loaded.model.mask_decoder.output_hypernetworks_mlps[0].weight.requires_grad is True

    history = loaded.finetune(_records()[:2], _records()[2:], epochs=1, learning_rate=1e-2)
    assert history[0]["optimizer_steps"] == 2
    assert len(loaded.adaptation_config["training_runs"]) == 2
    assert loaded.adaptation_config["training_runs"][0]["weight_delta_l2"] == 1.0
    assert loaded.adaptation_config["training_runs"][1]["weight_delta_l2"] > 0
    assert loaded.adaptation_config["artifact_lineage"][0]["producer"]["revision"] == "c" * 40

    continued = loaded.save_artifact(tmp_path / "continued", producer_revision="d" * 40)
    continued_manifest = json.loads(
        (continued / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert len(continued_manifest["adaptation"]["training_runs"]) == 2
    assert (
        continued_manifest["adaptation"]["artifact_lineage"][0]["producer"]["revision"]
        == "c" * 40
    )


def test_adapted_segment_defaults_to_single_mask_and_rejects_multimask():
    calls = []

    def runner(image, _points, _labels, _box, multimask):
        calls.append(multimask)
        count = 3 if multimask else 1
        return np.zeros((count, image.height, image.width), dtype=np.bool_), [0.0] * count

    pipeline = SAM2SegmentationPipeline(
        runner,
        "cpu",
        adaptation_config={"method": "frozen-backbone-mask-hypernetwork-gradient-adaptation"},
    )
    result = pipeline.segment(Image.new("RGB", (16, 16)), box=[1.0, 1.0, 8.0, 8.0])
    assert result["masks"].shape == (1, 16, 16)
    assert result["multimask"] is False
    assert calls == [False]

    with pytest.raises(ValueError, match="adapted pipelines require multimask=False"):
        pipeline.segment(
            Image.new("RGB", (16, 16)), box=[1.0, 1.0, 8.0, 8.0], multimask=True
        )


def test_artifact_export_rejects_non_finite_completion_metadata(tmp_path):
    pipeline = _pipeline()
    pipeline.freeze_for_adaptation()
    pipeline.adaptation_config.update(
        {"weight_delta_l2": float("nan"), "history": [{"epoch": 1}]}
    )
    with pytest.raises(RuntimeError, match="completed fine-tuning"):
        pipeline.save_artifact(tmp_path / "artifact", producer_revision="a" * 40)
