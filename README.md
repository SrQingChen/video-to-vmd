# VTV — Video to VMD

**Author:** SrQingChen
**License:** GPL-3.0-only
**Status:** v0.5.0 research preview

VTV turns a monocular dance video into a VMD that can be loaded directly on complex,
standard-bone PMX characters. It is built for the end-to-end case that generic
video-to-VMD tools usually leave unsolved: **target-model mismatch, unstable
feedback, long clips, and reproducible export/QC**.

---

## Why this project

Most video-to-VMD pipelines stop at “keypoints → bone names → VMD”. On real
characters—especially anime/game PMX models with P/C shoulders, leg D-chains,
twist helpers, IK, non-uniform rest poses and per-model morphs—the result is often:

- legs detached from the intended chain, arms reversed, shoulders collapsing;
- joints that look plausible in isolation but are wrong relative to the torso;
- jitter, spike frames and depth-flipped limbs;
- VMDs that only work on the model used for development.

VTV treats the **target PMX model** and **temporal/structural correctness** as
first-class problems, not as a post-processing afterthought.

---

## What is different from typical video→VMD tools

*This table describes design priorities, not a measured head-to-head benchmark
against every existing tool.*

| Capability | Typical video→VMD pipeline | VTV |
|---|---|---|
| Target model | keypoint-to-bone-name mapping | per-model PMX adapter: rest pose, P/C, D-chain, twist, IK, scale, morph aliases |
| Mismatch diagnosis | usually visual | compatibility report + safe writable-bone set; failures traceable to a specific bone/rest/inherit/IK item |
| Feedback | raw keypoints or smoothing | learned structural residual network trained on paired GT; no pixel/silhouette comparison |
| Validation | visual / no GT | synthetic paired GT, held-out motion/model/view, per-bone + contact + temporal gates |
| Long clips | one clip at a time | 10 s chunks + overlap, root alignment, VMD merge, 24 seam checks + per-chunk QC |
| Output contract | body-only or unknown helper tracks | 53 tracks (23 body + 30 fingers) + morphs, 4 IK off, no D/helper tracks, zero finger translation |
| Reproducibility | GUI/cloud, not inspectable | CLI, cache/checkpoint, manifests, QC JSON/Markdown, fixed-camera evidence |

The main technical bet is **cross-identity structural feedback**: the video
character and the target PMX never share an appearance, so RGB/silhouette
comparison is invalid. VTV compares structure—2D joints, limb directions, joint
angles, contact and temporal behaviour—in the target model's space.

---

## Evidence and data

### Paired data engine

| Item | Current result |
|---|---|
| Paired clips | **77 segments** (R1/R2/R3 × 3 PMX models × fixed cameras + training-only views/augmentations) |
| Split | **58 train / 10 val / 9 test**; R3 is an entire held-out motion, never trained or tuned |
| Training frames | **18,430** (2.43× the first M3.5 dataset) |
| Hand supervision | **3,180 L0/L1 frames** (6,361 side-frames), up from 462 |
| Ground truth | three-model body + 30 finger tracks + model-specific morphs |
| Leakage | whole-motion/model/view split; audit passes with 0 problems |
| Teacher reachability | v2 teacher on R3: **2.34° → 0.06°**, 0 clipped factors |

### Learned structural feedback (M6 v2, production default)

| Metric | Baseline (cleanup+IK+M3) | VTV v2 | Change |
|---|---:|---:|---:|
| R3 held-out body-relative main | 37.55° | **24.59°** | **−34.5%** |
| chest group | 43.67° | **12.76°** | −30.91° |
| upperarm group | 30.68° | **25.74°** | −4.94° |
| forearm group | 38.30° | **35.25°** | −3.05° |
| world angle (secondary) | 68.65° | 67.79° | **−0.86°** |
| 3D position (secondary) | 8.08%H | 7.95%H | **−0.13%H** |
| contact recall / slide / penetration | — | — | unchanged |
| jerk ratio | 0.4226 | 0.4348 | +2.9% |

- **9/9 held-out clips, 3/3 models, 3/3 camera views improve.**
- Per model: si 37.72→24.94°, luoxi 35.41→22.22°, tiflos 39.52→26.59°.
- Per view: front 38.30→23.85°, three-quarter 37.50→22.59°, side 36.86→27.31°.
- Model: 2.18M-parameter BiGRU over 289 evidence features → 79 bounded residual
  factors; TCN / Transformer / MRN / Graph / per-factor heads were compared and
  BiGRU won after fallback.
- The default is one-shot residual correction. Two-pass residual refinement was
  implemented and evaluated, but had no stable gain and over-corrected side views;
  it is not the default and is not called a closed-loop controller.
- Full hand/face feedback is available as an opt-in experiment, not the default
  (hand MAE is still worse than the previous full setting; T5 finger step grows).

### Production export and long video

- Final VMD: **53 tracks** (23 body + 30 fingers), 4 IK bones off, no D/helper
  tracks, zero finger translation, model-specific morphs.
- T5 three-model 2K renders: D-chain **0/0**, verify **<2e-6**, scale warning 0.
- Long-video stress test: **80.67 s / 2420 frames**, 9 chunks × 300 frames + 30
  overlap, **24/24 seam checks** and **27/27 segment QC checks** pass, merged
  jerk +2.56–3.19%, merged VMD remains 53 tracks/IK off/no D-helper.
- One command produces VMD + 2K before/after + QC + manifest; cache/resume and
  per-chunk rerun are built in.

---

## Pipeline

```
video
 → 30 fps normalization
 → optional 10 s chunks (1 s overlap)
 → GVHMR evidence
 → per-model PMX adapter
 → selective cleanup
 → contact / leg IK
 → hand + face evidence (L0–L3)
 → M6 v2 structural feedback (default: torso .85 / arm .30; hands/face/root off)
 → VMD (53 tracks, IK off, no D/helper)
 → 2K render + QC
 → long-video merge + seam/segment QC (when chunked)
```

The feedback is applied **after** the body/hand/face assembly and before VMD
export. High-confidence observations and unmodified tracks are preserved bit-for-bit;
bad windows fall back to the pre-feedback pose and are recorded in QC.

---

## Quick start

```bash
pip install -e .

# short clip (T5-like)
python scripts/run_vtv.py \
  --video testdata/T5_高速舞蹈_001/001.mp4 \
  --out outputs/demo_t5 \
  --models si,luoxi,tiflos \
  --feedback v2 \
  --render

# long video: chunking is automatic when longer than 10 s
python scripts/run_vtv.py \
  --video <long_video.mp4> \
  --out outputs/demo_long \
  --models si,luoxi,tiflos \
  --feedback v2 \
  --render
```

Useful switches:

- `--feedback v2` — production default (recommended).
- `--feedback default` — previous M3.5 conservative model.
- `--feedback off` — pure `cleanup+ik+M3` baseline.
- `--feedback v2_full` — opt-in hand/face residual experiment, not default.
- `--no-render` — export VMD/QC only.
- `--chunk-seconds`, `--overlap-seconds`, `--force-chunk` — long-video control.

Before a full run:

```bash
python scripts/check_env.py --models si,luoxi,tiflos --long
```

See [docs/M5_使用指南.md](docs/M5_使用指南.md) for installation, input formats,
models, cache, QC and failure diagnosis.

---

## Current scope

| Area | Status |
|---|---|
| Body / PMX adaptation | stable; three models; VMD/2K/QC |
| Structural feedback v2 | production default; R3 held-out −34.5% main |
| Hands | 30 tracks emitted; visible L0/L1 is still limited on T5; fallback is smooth |
| Expression | blink + procedural fallback; full expression set not yet reliable |
| Long video | 80.7 s stitched stress test passed; real continuous long clip is M7 |
| Slow motion + chunks | not yet supported; explicit refusal |
| Real-person video | not yet validated; M7/M8 |
| Full hand/face feedback | opt-in; not default until T5 stability and hand accuracy improve |

**Known limits are intentional and reported:** T5 has no GT; the long-video test is
a stitched stress clip, not a real continuous take; hand/face coverage is
input-limited; and the absolute body-relative error (24.59°) is improved but still
not “zero manual correction”.

---

## Repository layout

```
src/vtv/
  cli.py                 command line
  pipeline.py            stage orchestration / manifests / cache / resume
  longvideo.py           chunk planning / alignment / VMD merge / seam QC
  framing.py             per-model camera fitting
  feedback_apply.py      v2 residual apply / gates / fallback
  feedback_net.py        BiGRU / TCN / Transformer / MRN / graph models
  qc.py                  certificates / stability / change audit / secondary metrics
  mmd/                   PMX / VMD read-write / Blender bake helpers
scripts/                 run_vtv, data engine, training, evaluation, QC, render
configs/                 model and pipeline configuration
tests/                   unit + regression tests
docs/                    execution reports and guides
demo/                    release demo checklist (no restricted assets)
```

---

## License

**Project code: GNU GPL v3.0 only (GPL-3.0-only).**
Copyright (C) 2026 **SrQingChen**. See [LICENSE](LICENSE).

Third-party components remain under their own licenses; they are not relicensed
by this project. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[MODEL_LICENSES.md](MODEL_LICENSES.md). In particular:

- vendored `mmd2gltf` code remains MIT;
- MediaPipe runtime/models remain Apache-2.0;
- GVHMR / SMPL-X and the project-trained feedback checkpoints carry research /
  non-commercial constraints;
- Blender, MMD Tools and FFmpeg are external tools with their own licenses.

**Not redistributed:** PMX models, SMPL/SMPL-X body models, GVHMR weights,
project feedback checkpoints, VMD/R1–R3 motions and T1–T5 videos. They must be
obtained by the user under their respective licenses.

---

## Documentation

- [M6 execution: structural feedback v2](docs/M6_执行结果_后馈网络增强.md)
- [M5 execution: long video and release candidate](docs/M5_执行结果_长视频与发布候选.md)
- [M5 usage guide](docs/M5_使用指南.md)
- [M4 execution: production pipeline](docs/M4_执行结果_产线集成与开源.md)
- [Roadmap v2](开发路线图_v2.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Model licenses](MODEL_LICENSES.md)

---

## Author

**SrQingChen**
Project: [github.com/SrQingChen/video-to-vmd](https://github.com/SrQingChen/video-to-vmd)
Maintainer/author metadata: [AUTHORS.md](AUTHORS.md)
