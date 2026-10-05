# SAM 2.1 Segmentation E2E Notebook — Review

**Verdict: Needs revision** (three Majors, two Minors)  
**Review date:** 5 October 2026  
**Repository:** `kurtvalcorza/sam2-segmentation-pipeline`  
**Notebook:** `tutorials/sam2_segmentation_colab.ipynb`  
**Reviewed commit:** `df023e1` (`main`)  
**Notebook Git blob:** `56a0f9889498b53787d2bd936d209d82c02e0944`, generated from `9f3f221`; last notebook change `d2fc697` (2026-09-18). This is the blob executed in the recorded Kaggle T4 v7 run. At the reviewed commit, `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0.  
**Finding prefix:** `SAM`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2, `ml-worker` `origin/main` at `b9fdd1f`.

## Executive assessment

The engineering underneath this notebook is careful. It:

- carries the 1,264-line package verbatim, with a parity test;
- stages the `facebook/sam2.1-hiera-small` snapshot at an immutable commit and re-hashes all seven files;
- validates paired image/mask/prompt records, with a train/held-out check that compares ids and content;
- freezes everything except the first mask-token hypernetwork (139,808 parameters);
- trains with BCE plus soft-Dice, records a transactional training lineage, and does not select on the held-out split;
- exports a SafeTensors adapter bound to the base identity in a closed manifest;
- reloads that adapter over a fresh base, checking masks exactly and scores within a tolerance.

Three problems keep it from serving learners:

1. **`Run all` needs a manual restart (SAM-M1).** The recorded Kaggle T4 pass 1 stopped in the install cell with `Core dependencies changed … cuda-bindings 12.9.4 → 13.4.2; numpy 2.0.2 → 2.5.3 … Restart the runtime`. Only pass 2 completed. The release record counts the restarted run as PASS.
2. **The default task leaves nothing to adapt (SAM-M2).**
   - The pretrained model already scores held-out mask IoU **0.998254**. The adapted model scores **0.998457**, a change of +0.0002 on six images.
   - The generated scenes are single high-contrast shapes on flat backgrounds, prompted with both the exact box and an interior point.
   - The E2E's central demonstration (adapt, then compare on held-out data) therefore measures nothing, and the notebook does not tell the learner so.
3. **Declared `GUIDED`, but the guided layer is absent (SAM-M3).** There is no audience statement, how-to-use, roadmap, glossary, prediction, checkpoint, "What to notice", troubleshooting or conclusion template, and no cell is labelled or collapsed as infrastructure.

The two Minors concern BYOD guidance and splitting (SAM-m1), and a CPU runtime that fails only at Section 7 alongside an unexplained model-type warning (SAM-m2).

| Measure (Kaggle T4 v7, blob `56a0f98`, 2026-09-18) | Value |
|---|---|
| Code cells | pass 1: stopped at cell 3 with the restart `RuntimeError`; pass 2: 11/11 (229.4 s total) |
| Split | 18 train / 6 held-out (generated) |
| Held-out mask IoU | pretrained 0.998254 → adapted 0.998457; prompt-box baseline 0.856681 |
| Epoch history | train loss 0.0071 → 0.0054; held-out IoU 0.998383 → 0.998457 |
| Unseen scene | IoU 1.0 (model score 0.992) |
| Reload | masks exact, scores within rtol 1e-5 |

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites: "Python, binary masks, IoU, train/validation separation, and adapter-versus-base semantics" |
| Supported runtime | "fresh Python 3.12 with an NVIDIA T4-class GPU or better. CUDA is required for the default fine-tuning path" |
| Promised outcomes | Pinned install; verified snapshot; 24 generated records validated and split 18/6; pretrained baseline; bounded AdamW adaptation; held-out mask IoU against a prompt-box baseline; unseen-scene inference; base-bound SafeTensors export; fresh reload; provenance; BYOD "through the same validation, split, adaptation, evaluation, export, and reload path" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py` |
| Release status | `Candidate` (`STATUS.md`): Kaggle T4 v7 recorded "after one expected dependency-replacement restart" |

### Evidence actually obtained

- **Source inspection:**
  - all 25 cells (11 code; cell 5 is the carried `pipeline.py`);
  - `pipeline.py`: `finetune` (training loop, held-out scoring per epoch with no selection, lineage), `evaluate_adaptation`, `validate_segmentation_dataset`;
  - `STATUS.md` and `docs/release-verification.md`;
  - the v7 evidence bundle (`executed-pass1.ipynb`, `executed.ipynb`, `run_summary.json`).
- **Documented execution evidence:** Kaggle T4 v7, the reviewed blob, clean Hugging Face cache. Pass 1 failed at the install guard; pass 2 completed 11/11. There is no Colab run and no BYOD run.
- **Direct execution (this review), torch-free** (Python 3.12, `numpy` 2.5.3, `pillow` 11.3.0):
  - **P3.** The notebook's own cell 9 was executed with BYOD off. It builds the 24 generated records and the split. For each held-out record the probe measured:
    - the IoU of the filled prompt box (the notebook's prompt-box baseline);
    - mask size;
    - foreground/background contrast.
  - **P4.** The notebook's own `records_from_zip` and the cell's ≥ 8-record check were run on 8 constructed ZIPs.
  - Scripts and results are in `sam2_segmentation_colab_Review_Probes.zip`.
- **Not executed here:** the model. The Hugging Face Hub is unreachable from this container, and the locked torch is the multi-GB CUDA build. Model numbers come from the Kaggle record.
- **Learner observation:** none.

## 2. Separate judgments

- **Technical correctness.** The adaptation, export and reload machinery is correct. Held-out scoring is monitor-only, so there is no selection leakage. P3 reproduces the prompt-box baseline exactly (0.856683 against the recorded 0.856681). The installation pattern forces a restart (SAM-M1).
- **Promise fulfilment.**
  - Every stage runs and writes its outputs.
  - The adaptation demonstration is uninformative because the default task is saturated (SAM-M2).
  - BYOD does not reach new-data inference or reload on the user's own data (SAM-m1).
- **Learner experience.** The notebook is a terse, correct script. It has no guided layer (SAM-M3), and Sections 6–8 print numbers without saying how to read them.
- **Spec conformance.**
  - Unresolved MUSTs: RUN1, RUN10, ENV6 and REL2 (SAM-M1); EVAL15 and RUN8 (SAM-M2); SPL3 and DAT19 (SAM-m1).
  - SHOULD deviations: GDL1–GDL14, UX4, UX8 (SAM-M3); UX10 (SAM-m2).
  - The notebook declares spec 2.0.

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass Run all | cell 3: in-kernel `pip install` plus stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 11/11 | Section 1 prose presents the stop as designed | **Not met** (SAM-M1) |
| Verified pinned snapshot | cell 7 | 7/7 files verified at `ee5bba1d…` | clear | Met |
| Validated dataset and leakage-checked split | cells 9, 11 | 18/6; id and content overlap checked | — | Met (generated); BYOD split SAM-m1 |
| Pretrained held-out baseline | cell 13 | IoU 0.998254; box baseline 0.856681 | none | Met (no reading given) |
| Bounded adaptation | cell 15 | 36 steps, 139,808 trainable, Δw 0.109 | none | Met |
| "Evaluate the adapted model" against the baseline | cell 17 | 0.998254 → 0.998457 (+0.0002, 6 images) | none: three numbers printed | **Not met in substance** (SAM-M2) |
| Unseen-scene inference | cell 19 | IoU 1.0 | — | Met |
| Export and fresh reload | cell 21 | exact masks; scores within tolerance | — | Met |
| Provenance | cell 23 | `result.json` with dataset fingerprints, lineage, identity, reload | — | Met |
| BYOD through "the same … path" | cell 9 | P4: 1 compatible ZIP accepted; 7 refused; unseen inference and reload use a generated scene | contract partly stated | Partly met (SAM-m1) |
| Guided learning layer | — | absent | — | **Not met** (SAM-M3) |

The learning objectives are listed as actions the code performs. None is phrased as an observable learner outcome, and none is followed by a check of understanding (GDL5).

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 25 cells | The learner gets accurate one-line section introductions and no guidance (SAM-M3). After Section 8 they see 0.99825 → 0.99846 and are not told that the change is within noise, nor that the task left no room for improvement (SAM-M2). They also see an unexplained warning: "You are using a model of type sam2_video to instantiate a model of type sam2 … can yield errors" (SAM-m2). |
| **Clean default** | Documented (Kaggle T4 v7, reviewed blob) | Pass 1 stopped at the install guard (`cuda-bindings`, `numpy`). Pass 2 completed 11/11, with the numbers in the table above (SAM-M1). There is no Colab run, although the notebook is named and badged for Colab. |
| **Active learning** | Source | The notebook offers no experiment. Re-running Section 7 continues training from the adapted weights. The lineage records this honestly in `training_runs`, but the Section 6 baseline is not re-measured. |
| **Reuse and recovery** | Direct (P4) + source | See the BYOD results below the table. |

**BYOD results (P4).**

- **Accepted:** 10 flat pairs. The held-out split is the alphabetically last three stems (`s07`–`s09`): the split is by sorted name, with no shuffle and no grouping.
- **Refused:**
  - **A ZIP whose pairs sit under a wrapping folder** (`mydata/images/…`). The error is "E2E adaptation requires at least 8 records", because zero records were found.
  - **A `.jpg` mask, or an unmatched image.** The error is "BYOD image and mask stems must match exactly", without naming the stems.
  - **A `../` member.** The error is "unsafe or oversized BYOD archive".
  - **Mask problems.** An empty mask and a size mismatch are each named correctly.
  - **Too few records.** 7 pairs are rejected by the ≥ 8 rule.
- **Not exercised here:** the Colab upload dialog and the Kaggle auto-discovery of "exactly one attached ZIP".

## 5. Findings

### Major

#### SAM-M1 — `Run all` needs a manual restart after the install cell, and the release record counts the restarted run

- **Cell/section:** cell 3 and the Section 1 prose; generator `tools/build_notebook.py` / `tools/notebook_template.py` install block; `docs/release-verification.md` and `STATUS.md`.
- **Observed issue:**
  - The cell runs `pip install` for eight pins into the running kernel.
  - It then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` whenever a loaded distribution changed.
  - Section 1 presents this stop as designed.
  - The release record reports "11/11 code cells after one expected install restart" as PASS.
- **Consequence:** a learner who selects **Run all** on a stock Kaggle or Colab image meets an error in the first code cell and must restart and run again. RUN1, RUN10 and ENV6 forbid this.
- **Evidence:** documented. Kaggle T4 v7 `executed-pass1.ipynb`: `cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`. Pass 2 completed 11/11. The superseded v6 run shows the same pattern.
- **Recommended correction:** adopt the fleet's isolated **uv** environment pattern. Install nothing into the kernel. Use:
  - a pinned `uv` wheel checked by size and SHA-256;
  - `uv venv --managed-python --python 3.12.12`;
  - a hash-locked `requirements.txt` installed with `--require-hashes --only-binary :all:`;
  - every stage run in a subprocess.

  Reference implementations on `main` are `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb` and the four standalone notebooks built on 2026-10-04/05 (`mediapipe-face-landmarker-pipeline`, `nafnet-deblurring-pipeline`, `raft-stereo-pipeline`, `swin2sr-x4-super-resolution-pipeline`). Implement the pattern in the generator, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so that a restart-dependent run is not reported as a Run all PASS.
- **Acceptance check:**
  - A fresh Kaggle or Colab GPU runtime completes every code cell in a single **Run all**, with no restart and no error.
  - That run is recorded with the notebook blob and `restarted: false`.
  - `grep -n "Restart the runtime" tutorials/sam2_segmentation_colab.ipynb` returns nothing.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### SAM-M2 — The default task is saturated, so the adaptation and held-out comparison measure nothing, and the notebook does not say so

- **Cell/section:** cell 9 (`generated_records`: one flat background, one bright shape, one distractor rectangle); cells 13, 15 and 17 (baseline, fine-tune, adapted evaluation); the interpretation section (cell 24). Generator: `tools/notebook_template.py`.
- **Observed issue:**
  - **The scenes are trivially separable.** Each generated scene is a single rounded rectangle or ellipse in a bright colour on a flat dark background. Foreground/background contrast is 107–128 grey levels (P3).
  - **The prompts are nearly exact.** The model receives both the shape's exact bounding box and an interior point.
  - **No headroom.** A promptable segmenter is at ceiling before any update: the pretrained held-out IoU is 0.998254.
  - **No measurable effect.** After 36 steps the held-out IoU is 0.998457: +0.0002 on six images, with no per-record spread and no statement of what difference would matter.
  - **No explanation.** The notebook prints the three numbers (pretrained, adapted, box baseline) without a "What to notice". The interpretation says only that this "does not prove improvement on photographs".
  - **The comparison that is shown is not the adaptation.** The box baseline (0.857) shows what prompting adds, not what adaptation adds.
- **Consequence:**
  - The E2E profile's core demonstration, that adaptation changes held-out performance and that the learner can measure it, cannot be read from this run.
  - A learner is likely to read 0.99825 → 0.99846 as "fine-tuning improved the model", which is exactly the unsupported claim FT7 and EVAL15 guard against.
  - The weight-delta check proves only that the parameters moved, not that anything was learned.
- **Evidence:**
  - Documented: Kaggle v7 cells 13–17. Per-epoch held-out IoU was 0.998383 and then 0.998457, so the curve is flat at ceiling.
  - Direct (P3): the held-out box-fill IoU per record is 0.78–0.95, with a mean of 0.856683. This reproduces the recorded box baseline (0.856681). The P3 contrast measurements quantify how easy the scenes are.
- **Recommended correction:** make the default task leave measurable headroom and teach the learner to read the result.
  - (a) Change the generated data so that the pretrained model is clearly imperfect. Options:
    - textured or cluttered backgrounds;
    - low-contrast or partially occluded targets;
    - thin structures;
    - point-only prompts, or loose boxes.

    Check that the pretrained held-out IoU lands well below 1.0 on the documented runtime.
  - (b) Print per-record IoU for pretrained and adapted, with the number of records improved.
  - (c) Add a "What to notice" that states the headroom and says how large a change six records can show.
  - (d) In the interpretation, distinguish "the weights moved" from "held-out quality changed".

  If the default must stay synthetic and easy, say plainly that the comparison is a mechanics check with no headroom, and remove it from the learning objectives' "evaluate" claim.
- **Acceptance check:**
  - On a default hosted run, the pretrained held-out IoU is reported with per-record values.
  - Either the default data leave headroom (pretrained IoU clearly below 1.0, so that the adapted-minus-pretrained difference is interpretable), or the notebook text explicitly labels the comparison as having no headroom.
  - No prose implies an improvement that the recorded numbers cannot support.
- **Spec:** EVAL15, RUN8, FT7, UX4.

#### SAM-M3 — Declared `GUIDED`, but the guided layer is absent

- **Cell/section:** opening cells 0–1, every section boundary, the carried cell 5, end of notebook. Generator: `tools/notebook_template.py`.
- **Observed issue:**
  - **Orientation is missing:** no intended-learner statement, no **How to use this notebook**, no roadmap.
  - **No task contract:** there is no Input → Model → Output contract.
  - **No glossary**, for example of promptable segmentation, the mask decoder, the hypernetwork, mask tokens, IoU, the soft-Dice loss and adapter versus base.
  - **No active-learning prompts:** no prediction before the baseline, the fine-tune or the adapted evaluation; no checkpoint with a sample answer; no "What to notice".
  - **No support material:** no troubleshooting section (GPU absent, download failure, digest mismatch, BYOD errors) and no conclusion template.
  - **No infrastructure labelling:** the 1,264-line carried cell 5 and the install cell are not labelled **Infrastructure** or collapsed (`cellView` absent on every code cell).
- **Consequence:** a self-paced learner gets a correct script, but no help deciding what matters or how to interpret the saturated result (SAM-M2).
- **Evidence:** source inspection (no such headings or markers in any cell; no `cellView` metadata).
- **Recommended correction:** add the GDL layer in the template, following NOTEBOOK_SPEC 2.2's guided-mode requirements. The four 2026-10-04/05 notebooks named in SAM-M1 show the full set: audience, how-to-use, roadmap, task contract, glossary, predictions, collapsed checkpoints, a Predict → Change → Run → Observe → Explain activity built on a restart-from-base fine-tune, troubleshooting, a conclusion scaffold, and Infrastructure-titled collapsed cells.
- **Acceptance check:** each of GDL1–GDL14 maps to a named cell, and the install and carrier cells carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL14, UX4, UX8.

### Minor

#### SAM-m1 — BYOD: unguided refusals, a name-ordered split with no grouping, and inference and reload on a generated scene instead of the user's data

- **Cell/section:** cell 9 (`records_from_zip`, `load_byod_zip`, the split); cells 19 and 21.
- **Observed issue:**
  - **A wrapping folder is not explained.** A ZIP whose pairs sit under one wrapping folder yields zero records, and the user sees "E2E adaptation requires at least 8 records".
  - **Stems are not named.** A `.jpg` mask, or an image without a mask, both produce "BYOD image and mask stems must match exactly", without naming the stem. The rule that masks must be PNG is stated only in the code.
  - **Two causes share one message.** Path traversal and an oversized archive both produce "unsafe or oversized BYOD archive".
  - **The split ignores scene structure.** The held-out split is the alphabetically last 25 % of stems. Frames or crops of one scene, named sequentially, end up on both sides. No `group` support exists, and the notebook does not state the independence assumption.
  - **Inference and reload ignore the user's data.** In BYOD mode, the unseen-scene inference (cell 19) and the reload check (cell 21) still use a generated scene.
- **Consequence:**
  - A first BYOD attempt with a typical ZIP fails with a misleading message.
  - A BYOD held-out score can be inflated by near-duplicate scenes.
  - The user never sees the adapted model, or its reload, on their own data.
- **Evidence:** direct (P4), using the notebook's own functions; source.
- **Recommended correction:**
  - Tolerate one wrapping folder, or name it in the error.
  - Name the unmatched stems and state the PNG-mask rule.
  - Separate the traversal message from the size message.
  - Shuffle the split with the seed and support a group prefix or manifest, stating the independence assumption.
  - In BYOD mode, run inference and the reload check on a held-out BYOD record.
- **Acceptance check:**
  - A wrapped ZIP is accepted, or refused with a message naming the folder.
  - An unmatched stem is named.
  - The BYOD split is seeded and group-aware.
  - BYOD runs show inference and reload parity on a BYOD record.
- **Spec:** DAT12, DAT14, DAT19, SPL3, SPL5, UX10.

#### SAM-m2 — A CPU runtime fails only at Section 7, and a model-type warning goes unexplained

- **Cell/section:** cell 3 (no early GPU check); cell 15 (`raise RuntimeError('The canonical SAM2 E2E path requires a CUDA GPU')`); cells 7 and 21 (the transformers warning).
- **Observed issue:**
  - **CPU runtimes fail late.** On a CPU runtime the notebook installs, downloads 184 MB, loads the model and scores the baseline before it stops at Section 7. The Prerequisites say CUDA is required, but nothing checks for a GPU up front.
  - **An unexplained warning.** Each model load prints "You are using a model of type sam2_video to instantiate a model of type sam2. This is not supported for all configurations of models and can yield errors." No note says whether this is expected.
- **Consequence:**
  - CPU users lose minutes before a predictable failure.
  - Every user sees a warning that suggests the model may be wrong, without guidance.
- **Evidence:** source; documented (Kaggle v7 cells 7 and 21).
- **Recommended correction:**
  - Check for CUDA in the first code cell, with the runtime-change instruction.
  - Add one sentence, or a Troubleshooting row, explaining the `sam2_video` → `sam2` warning: the published config's `model_type` is `sam2_video`, and image-mode loading is the intended use.
- **Acceptance check:**
  - A CPU runtime stops in the first code cell with a runtime-change message.
  - The warning is explained in the notebook.
- **Spec:** UX10, GDL13.

### Suggestions

- **SAM-S1:** Declare `notebook_spec` 2.2 once the restart and guided-layer fixes land.
- **SAM-S2:** Record a Colab run. The notebook is named and badged for Colab, but every recorded execution is on Kaggle.
- **SAM-S3:** Re-measure the baseline at the start of each fine-tune, or make the fine-tune restart from the base, so that a re-run of Section 7 compares cleanly. The lineage already records cumulative training honestly.
- **SAM-S4:** Show a visual: the image, the prompt, the ground truth, and the pretrained and adapted masks for the held-out records.

## 6. Readiness

**Needs revision.**

- **Open Majors:** SAM-M1 to SAM-M3.
- **Gates remaining after the fixes:**
  - a one-pass hosted Run all of the regenerated blob (RUN1, RUN10, REL2);
  - a default run whose held-out comparison is interpretable (SAM-M2);
  - the REL12 BYOD journey recorded in `docs/release-verification.md`: one compatible ZIP and one incompatible ZIP.

## 7. Verified versus inferred

- **Verified by direct execution (torch-free):**
  - the generated split;
  - the held-out box-fill IoU, reproducing the recorded prompt-box baseline;
  - the foreground/background contrast of the held-out scenes;
  - the BYOD reader's acceptance and refusal behaviour, and its name-ordered split.
- **Verified from documented evidence:**
  - the restart on Kaggle pass 1;
  - every model number from pass 2;
  - the `sam2_video` warning.
- **Inferred from source:**
  - the CPU failure point;
  - BYOD inference and reload on a generated scene;
  - continued training when Section 7 is re-run.
- **Only Kurt can confirm:** whether the default task should be made harder (preferred) or explicitly framed as a mechanics check (SAM-M2's minimum).
- **Most likely to be wrong:** SAM-M2's severity. A maintainer could argue that the notebook never claims an improvement. It is rated Major because the E2E profile's adaptation-and-evaluation promise cannot be read from the default run, and nothing tells the learner so.
