from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sam2_segmentation_pipeline.pipeline import (
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_WEIGHTS_NAME,
    SAM2SegmentationPipeline,
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

    manifest["adaptation"] = source.adaptation_config
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        _pipeline().load_artifact(artifact)
    assert (artifact / ARTIFACT_WEIGHTS_NAME).is_file()
