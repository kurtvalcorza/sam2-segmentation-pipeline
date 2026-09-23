---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: mask-generation
task: "Others - Promptable Image Segmentation"
base_model: facebook/sam2.1-hiera-small
date_published: "2024-09-24"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/facebook/sam2.1-hiera-small)"
---

# SAM 2.1 Hiera-Small — Promptable Image Segmentation (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fsam2.1--hiera--small-ffcc4d?style=flat)](https://huggingface.co/facebook/sam2.1-hiera-small)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-facebookresearch%2Fsam2-181717?style=flat&logo=github&logoColor=white)](https://github.com/facebookresearch/sam2)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2408.00714-b31b1b.svg)](https://arxiv.org/abs/2408.00714)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **End-to-End Fine-Tuning Tutorial**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/sam2-segmentation-pipeline/blob/main/tutorials/sam2_segmentation_colab.ipynb) [`sam2_segmentation_colab.ipynb`](https://github.com/kurtvalcorza/sam2-segmentation-pipeline/blob/main/tutorials/sam2_segmentation_colab.ipynb)  
  *A standalone E2E path over deterministic labelled scenes: pinned pretrained baseline, bounded mask-hypernetwork gradient adaptation, held-out mask IoU against a prompt-box baseline, SafeTensors export, unseen inference, and fresh-model reload.*

---

#### Description

`facebook/sam2.1-hiera-small` is the Transformers-format release of the SAM 2.1 Hiera-Small checkpoint from Meta FAIR's *SAM 2: Segment Anything in Images and Videos* (Ravi et al., arXiv:2408.00714), pinned here to revision `ee5bba1d82bb8749febdf90f45e84b687142ba03`. The snapshot `config.json` declares the `Sam2VideoModel` architecture: a Hiera hierarchical vision backbone with an FPN neck (`fpn_hidden_size` 256, three feature levels at 256/128/64 px on a 1024x1024 input), a prompt encoder for points and boxes (`hidden_size` 256, 4 point embeddings), a two-layer mask decoder with an IoU-prediction head and `num_multimask_outputs` 3, and a memory attention/encoder stack used only for video. At inference in image mode the model embeds the image once, encodes the caller's point clicks and/or box, and decodes one or three candidate masks with a predicted IoU each. This repository adds verified packaging and an optional bounded adaptation route: `freeze_for_adaptation` leaves only the first mask-token output hypernetwork trainable; `finetune` uses caller-owned labelled image/mask/prompt records; `evaluate_adaptation` reports held-out mask IoU and a prompt-box baseline; and `save_artifact`/`from_artifact` export and reload only that surface through a digest-bound SafeTensors artifact.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is promptable single-object image segmentation: input one RGB still (`PIL.Image.Image`, any mode, converted to RGB) plus, for one object, up to 16 point clicks labelled 1 (foreground) or 0 (background) and/or one xyxy box; output `masks`, a boolean array of shape `(K, H, W)` at the input resolution with `K = 3` candidate masks when `multimask=True` or `K = 1` when `multimask=False`, and `iou_scores`, the model's own predicted IoU per mask. Base pipelines default to the three-candidate path; adapted pipelines default to the single-mask path trained by this adapter and reject `multimask=True`. The optional adaptation task accepts 2–128 uniquely identified records with an exact-size non-empty boolean mask and valid prompts, trains only the declared hypernetwork surface, and requires a separate held-out split for evaluation. Within DIMER the pipeline is an inference component and bounded adaptation baseline, not an automatic "segment everything" service and not a video tracker.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision researchers, and application developers integrating promptable segmentation into annotation tooling, research prototypes or in-house systems. A user is expected to understand that the model segments whatever visual region the prompt points at, without knowing what the object is; that `iou_scores` is the model's estimate of its own mask quality, uncalibrated and unrelated to semantic correctness; that a single click is ambiguous (part, object, or group) and the three candidates exist for that reason; that mask boundaries at the input resolution are up-sampled from a 256x256 logit grid; and that mean IoU can only be measured against labelled masks they supply. Users who need semantic labels, video tracking, or automatic whole-image segmentation are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** video segmentation and tracking are not exposed — `Sam2VideoModel`, `Sam2VideoProcessor`, memory attention, and the snapshot's `video_preprocessor_config.json` and native `sam2.1_hiera_s.yaml` are unused; no automatic mask generation (the `mask-generation` pipeline is not wrapped); no class labels, no text prompts (use `grounding-dino-detection-pipeline` to obtain boxes from text), one object per call.
2. **Input boundary:** `segment` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, a call with neither points nor box, more than `MAX_PROMPTS = 16` points, points without labels or with labels other than 0/1, points or boxes outside the image, empty or inverted boxes, and a non-bool `multimask` (`ValueError`/`TypeError`). Every image is resized to 1024x1024, so thin structures below roughly 1/1024 of the longer side are lost.
3. **Input boundary:** non-photographic imagery (medical volumes, microscopy, radar, line art) falls outside the SA-1B/SA-V distribution; masks on it are undefined and the pipeline does not detect it.
4. **Decision boundary:** not for autonomous measurement or triage with medical, legal, safety, or financial consequence (lesion area, property boundaries, defect acceptance) without a human reviewing each mask and a locally measured IoU on labelled data.

#### Factors

###### Groups

This pipeline is not human-centric by design: it delineates the region a prompt points at and neither classifies nor identifies people. It will, however, segment a person, face, or body part as readily as any object when prompted there, and the upstream training data (SA-1B images and SA-V videos, per the SAM and SAM 2 papers; the snapshot README lists no training data) are large licensed photo and video collections whose geographic and demographic composition the SAM 2 paper describes only in aggregate and which this repository has not audited for skin tone, age, gender, disability, or dress. Any difference in mask quality across such groups is therefore unknown, not known to be absent. A downstream operator who segments images of people is responsible for a fairness audit on their own data: stratify a labelled sample by the relevant groups and compare `mask_iou` per stratum before relying on the output.

###### Instrumentation

The upstream training masks were produced by a model-in-the-loop annotation engine over licensed photographs (SA-1B) and videos (SA-V) of unspecified camera provenance; the masks are therefore themselves partly model-generated and human-corrected rather than sensor ground truth. Inference images arrive from whatever camera the operator uses; sensor resolution, lens distortion, compression, motion blur, and low light change the edges the model sees, and the 1024x1024 resize (`preprocessor_config.json`, bilinear, ImageNet mean/std) plus the 256x256 mask grid (`mask_size`) bound the boundary precision regardless of source resolution. Prompts are a second instrument: a click a few pixels off the object, or a box that clips it, changes the mask; the pipeline validates that prompts lie inside the image but cannot tell whether they lie on the intended object.

###### Environment

The declared notebook environment is Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, and the exact pins in `pyproject.toml`. The earlier E2E blob completed clean-room execution on Kaggle Python 3.12.13 with PyTorch 2.14.0+cu130 and a Tesla T4, but the current review-remediation revision requires fresh exact-blob execution before promotion. Model cost is dominated by the fixed 1024x1024 working resolution, while the caller's resolution mostly sets the size of up-sampled masks. The model assumes an ordinary photograph with a visible prompted boundary; low contrast, transparency, thin structures, and heavy occlusion can still produce masks that bleed or fragment.

#### Metrics

###### Performance Measures

Each inference mask carries an `iou_scores` self-estimate, not a measurement against ground truth. `mask_iou(a, b)` computes measured overlap when a reference exists, and `evaluate_adaptation(records)` averages the selected-mask IoU over validated labelled records while also scoring the prompt box as a transparent baseline. The E2E tutorial reports the pinned pretrained and adapted results on the same held-out split with the verdict `sample-sanity`. Synthetic results are workflow evidence, not a SAM benchmark; deployment claims require a representative caller-owned test set that was not used for optimization.

###### Decision thresholds

One threshold is applied: the up-sampled mask logits are binarised at `MASK_THRESHOLD = 0.0` (`Sam2Processor.post_process_masks` default), so a pixel is foreground when its logit is positive; this is the implicit decision rule behind every `True` in `masks` and is not tuned per domain. No threshold is applied to `iou_scores`: all `K` candidates are returned with their scores and the caller picks (the smoke run used `argmax`, itself a threshold-free rule that the card names here). The choice between `multimask=True` and `False` is a second decision the caller owns. A deployment that wants an acceptance rule — for example rejecting masks whose predicted IoU is below some value, or shifting the logit threshold to favour tighter or looser masks — must set it against its own labelled data, weighing the cost of a mask that bleeds into the background against one that misses part of the object, and owns revisiting it when the image source changes.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings, mask areas, and one IoU on a synthetic scene, not accuracy. Run-to-run variability comes from floating-point kernel selection across CPU builds and accelerators and from bilinear up-sampling of the 256x256 logits to the input size; there is no sampling and no seed to set, so a fixed input and prompt on fixed hardware is repeatable but not guaranteed bitwise-identical across machines. `iou_scores` is the model's own estimate and is not calibrated: a 0.99 does not mean a 99 % chance the mask is right, and on the smoke scene the three candidates scored 0.010, 0.993 and 0.400 for masks of very different extent. A caller who needs calibrated confidence must fit a map from `iou_scores` to measured `mask_iou` on their own labelled data; a caller who needs an uncertainty estimate for a metric must compute it over many labelled images or bootstrap resamples themselves.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README discloses no training data; the SAM 2 paper reports training on SA-1B (about 11 M licensed images with about 1.1 B masks, from the SAM paper) and SA-V (about 51 k videos with about 643 k masklets), collections that Meta describes as licensed and privacy-filtered (faces and licence plates blurred in SA-1B) but whose per-item consent status this repository cannot verify, so personal data in the corpus is not ruled out. This repository distributes code, tests, and documentation; it does not distribute the 184,305,280-byte `model.safetensors`, which is staged locally under `weights/sam2.1-hiera-small/` and git-ignored, and it ships no sample images. The operator must audit the images and prompts they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will delineate whatever it is pointed at.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — outlining lesions or organs in medical images, isolating people in surveillance stills, measuring damage for insurance claims, delineating land parcels — would be admissible only with human review of every mask before action, a locally measured IoU against expert-drawn masks on the deployment's own data, a documented threshold and candidate-selection policy, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 7 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True` and always passes `trust_remote_code=False`. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused.
- **Input integrity:** the public `validate_inputs(image, *, points, point_labels, box, multimask)` stage and `segment` share the same image, prompt, and boolean-multimask checker; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; `validate_prompts` rejects a call with no prompt, more than 16 points, missing or non-0/1 labels, out-of-image points, and empty, inverted, or out-of-image boxes; dataset validation additionally requires labelled points to agree with the target and every supplied box to overlap target foreground. `segment` rejects `multimask=True` once an adapter is configured and raises on a backend result whose shape, dtype, or score count is wrong.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; every result carries `model_id`, `model_revision`, and the exact validated `points`, `point_labels`, and `box` sent to the model.
- **Refusals:** video tracking, automatic mask generation, batching, and download without the explicit flag are not exposed; the image-mode class is loaded even though the snapshot config names the video class, and the loader comment records why.
- **Bounded adaptation:** training rejects more than 128 records, freezes every parameter outside the declared first mask-token hypernetwork, records trainable/frozen counts and weight movement, and exports only SafeTensors plus a closed digest-bound manifest. Dataset balancing and deployment-specific fairness mitigation remain the operator's responsibility.

###### Risks and harms

- **Wrong extent:** a click on a part yields the part, the object, or a group; the wrong candidate is chosen by `argmax` and the harm falls on whoever acts on the area or crop; likely for ambiguous single clicks.
- **Boundary bleed:** low-contrast or transparent edges produce masks that include background or exclude the object; the operator bears the harm when the mask drives a measurement.
- **Overconfident self-score:** `iou_scores` can be high for a mask that is semantically wrong; the value is a regression output, not a bounded IoU, so it can exceed 1.0 (observed as 1.014 on the sibling `sam-vit-segmentation-pipeline` on 2026-09-12; not yet observed on this model); automation bias follows when reviewers trust the number.
- **Prompt-driven targeting:** the model will isolate any person or body part it is pointed at; the data subject bears the harm when such masks drive tracking, redaction failures, or manipulation.
- **Privacy exposure:** images of people or private spaces are processed without any content check.
- **Bias amplification:** any imbalance in SA-1B/SA-V is reproduced as uneven mask quality across appearance groups, undetected because no per-group evaluation exists.
- **Resource use:** ~1 s per prompt on the reference CPU, a 184 MB checkpoint, and up to 48 MiB of boolean masks per call at the 4096-px ceiling; a request stream can saturate a shared host.

###### Use cases

Prohibited even where the model would work: covert surveillance or tracking of individuals, biometric or demographic profiling, social scoring, and any use that discriminates unlawfully in employment, housing, credit, insurance, education, or healthcare access. Also prohibited are deceptive or non-consensual image manipulation — isolating a person to composite, undress, or misrepresent them, or fabricating "measured" evidence from a mask — and any use that violates the upstream Apache-2.0 licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the images processed. Autonomous high-consequence actions triggered by an unreviewed mask are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `facebook/sam2.1-hiera-small`
- Revision: `ee5bba1d82bb8749febdf90f45e84b687142ba03`
- Snapshot manifest: `weights/sam2.1-hiera-small/dimer-base-manifest.json`, 7 files, `totalBytes` 184336196
- `model.safetensors` SHA-256: `0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60` (184,305,280 bytes)
- `config.json` SHA-256: `97ff9f65b76d107acda4247885f0a5555d0048850ae3c5f97183df289aaecde9` (5,698 bytes)
- Weight format: SafeTensors; loader `Sam2Model.from_pretrained(<dir>, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False)` with `Sam2Processor` from the same directory. The config declares `Sam2VideoModel` (`model_type: sam2_video`); Transformers logs a type notice and loads the shared weights into the image-mode class, as the upstream README's Transformers example does.

## Input/output contract

- `SAM2SegmentationPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`.
- `segment(image, *, points=None, point_labels=None, box=None, multimask=None) -> dict` — one object per call; `points` is a sequence of `[x, y]` pixel pairs with `point_labels` of 0/1, `box` is `[x0, y0, x1, y1]` in pixels. `None` selects three candidates for an unadapted base pipeline and the trained single-mask head for an adapted pipeline. Adapted pipelines reject `True`. Returns `masks` (bool `(K, H, W)`, `K` = 3 if `multimask` else 1), `iou_scores` (list of `K` floats, model-predicted, uncalibrated), the resolved `multimask`, validated prompts, dimensions, and model identity.
- Ceilings: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16`; `NUM_MULTIMASK_OUTPUTS = 3`; `MASK_THRESHOLD = 0.0`.
- `mask_iou(a, b) -> float` on boolean arrays; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12.
- Precision: float32; preprocessing resize to 1024x1024, bilinear, ImageNet mean/std (`Sam2ImageProcessorFast` from the snapshot); masks decoded at 256x256 and up-sampled to the input size, binarised at logit 0.
- Measured 2026-09-12 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=""`, device `cpu`: `verify_snapshot` 0.12 s (7 files, 184 MB); load 4.80 s; `segment` on a synthetic 320x240 scene (grey background, dark rectangle at [40, 60, 140, 180], red disc at [200, 80, 280, 160]) with one foreground click at (90, 120) → `masks (3, 240, 320)` bool, `iou_scores` [0.010, 0.993, 0.400], areas [1767, 12220, 19768] px, `mask_iou` of the argmax candidate against the drawn rectangle 1.000, 0.962 s; box `[200, 80, 281, 161]` with `multimask=False` → one mask of 5145 px (drawn disc about 5026 px), `iou_scores` [0.987], 0.938 s; 4096x4096 uniform-noise image with a box → `(1, 4096, 4096)` in 1.19 s. Process wall 11 s.
- Tests: `pytest -q -o addopts= tests` — 11 passed, offline, no weights required; `ruff check src tests` clean.
- Not executed: bfloat16 autocast, video mode, any IoU measurement against real labelled masks, or a CUDA throughput benchmark.

- **Load-time notice (expected):** transformers logs `You are using a model of type sam2_video to instantiate a model of type sam2` on every load, because the pinned snapshot's `config.json` declares `model_type: sam2_video` while this pipeline deliberately instantiates the image-only `Sam2Model`. The message is a config-name mismatch, not an error; the owner chose on 2026-09-12 to keep `Sam2Model` and document the line rather than load the video class or filter the logger.

## References

- Ravi, Gabeur, Hu, Hu, Ryali, Ma, Khedr, Rädle, Rolland, Gustafson, Mintun, Pan, Alwala, Carion, Wu, Girshick, Dollár, Feichtenhofer. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714, 2024. https://arxiv.org/abs/2408.00714
- Kirillov et al. Segment Anything. ICCV 2023. https://arxiv.org/abs/2304.02643
- Ryali et al. Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles. ICML 2023. https://arxiv.org/abs/2306.00989
- Upstream code: https://github.com/facebookresearch/sam2
- Upstream card: https://huggingface.co/facebook/sam2.1-hiera-small
- Transformers `SAM2` documentation: https://huggingface.co/docs/transformers/model_doc/sam2
