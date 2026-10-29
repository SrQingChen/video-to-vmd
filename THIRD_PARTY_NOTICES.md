# Third-party notices

This repository is a research tool. It does **not** redistribute model
weights, PMX models, VMD motions, SMPL/SMPL-X body models, or test videos.
Users must obtain those separately and comply with their own licenses.

## mmd2gltf-gui (MIT)

- Source: https://github.com/masaka1024/mmd2gltf-gui (see `third_party/mmd2gltf-gui/`)
- Vendored files: `src/vtv/mmd/pmx.py`, `vmd.py`, `animation.py`, `mathutil.py`
- License text: `src/vtv/mmd/LICENSE.mmd2gltf`
- Local changes: the VMD interpolation parser in `vmd.py` was corrected to read
  the four 16-byte channel blocks (control points at +0/+4/+8/+12) instead of a
  single 16-byte window; `animation.py` gained an `apply_all_translations`
  option to reproduce Blender/MMD Tools' non-movable-bone translation behaviour
  for the audit.

## MediaPipe (Apache-2.0)

- Python package: `mediapipe==1.1.1` (Apache-2.0), used by `scripts/run_hand.py`
  and `scripts/run_face.py`.
- Model bundles: `hand_landmarker.task` and `face_landmarker.task`, downloaded
  from `https://storage.googleapis.com/mediapipe-models/.../latest/`. They are
  stored under `models/mediapipe/` and **not redistributed** by this repository.
- No MediaPipe C++ source is vendored; only the public Python Tasks API is used.

## Python runtime libraries

- **PyTorch 2.3.0** (BSD-3-Clause) and **torchvision 0.18.0** (BSD-3-Clause):
  network inference (M3.5 feedback) and GVHMR.
- **NumPy** (BSD-3-Clause, pinned `<2`) and **OpenCV**
  (`opencv-python==4.11.0.86`, Apache-2.0): array/image I/O for the hand/face
  and render pipelines.
- Installed from pip; not vendored in this repository.

## Blender and MMD Tools (external tools)

- **Blender 5.2.x** (GPL-2.0-or-later) is invoked headless by
  `scripts/render_synth.py` and `scripts/mmd_blender.py` for 2K rendering and
  the D-chain/twist bake. It is not redistributed or linked into this
  repository's code; it is a separate program run as a tool.
- **MMD Tools** (GPL-3.0) is a Blender extension installed by the user; this
  repository imports it at render time but does not copy its source.
- If you redistribute a bundled Blender/MMD Tools, that distribution is under
  the GPL and is separate from the GPL-3.0-only core here.

## FFmpeg (external tool)

- `ffmpeg` (LGPL-2.1-or-later or GPL-2.0-or-later depending on the build) is
  invoked as an external encoder for the before/after MP4s. It is not
  redistributed; check your build's license if you redistribute it.


## M5 long-video chunking (original code + external ffmpeg)

- `src/vtv/longvideo.py`, `src/vtv/framing.py` and `src/vtv/cli.py` are original
  GPL-3.0-only project code. The chunk planning, root yaw/translation alignment,
  quaternion cross-fade, VMD merge and seam QC do not copy third-party source.
- Chunk cutting and the 30 fps master video are produced by invoking the user's
  **ffmpeg** as an external command (`select`/`fps`/libx264); ffmpeg is not
  redistributed. See the FFmpeg section above.
- The auto-frame camera fit uses Blender's documented sensor/FOV model and is
  invoked through the existing `scripts/render_synth.py`; Blender/MMD Tools stay
  external GPL tools (above).

## M3.5 feedback checkpoint (project-trained, research-only)

- `outputs/m35/net/m35_best.pt` is trained by this project from synthetic
  renders of the user's PMX models, with teachers derived from GVHMR/SMPL-X
  and the deterministic M2/M3 pipeline. It is **not redistributed**
  (git-ignored) and inherits the research/non-commercial constraints of
  GVHMR and SMPL-X.
- The architecture (`src/vtv/feedback_net.py`), training code and application
  code are original GPL-3.0-only code; only the weights carry that constraint.

## M6 v2 feedback checkpoint (project-trained, research-only)

- `outputs/m6/net/m6_v2_hf_ft_best.pt` is trained by this project from the same
  synthetic renders plus synthetic hand/face L0 evidence, with teachers derived
  from GVHMR/SMPL-X and the deterministic M2/M3 pipeline. It is **not
  redistributed** (git-ignored) and inherits the research/non-commercial
  constraints of GVHMR and SMPL-X.
- The v2 architectures (`src/vtv/feedback_net.py`), two-pass application
  (`src/vtv/feedback_apply_v2.py`), training/data scripts and tests are
  original GPL-3.0-only code; only the weights carry that constraint.
- M6.0 benchmarked GVHMR flip-TTA and UnderPressure. No third-party code or
  weights from those benchmarks are redistributed; UnderPressure remains
  vendored under its own license in `third_party/UnderPressure/`.
- The 2024–2026 candidates reviewed but not run (MotionRefineNet, HTD-Refine,
  FootMR/ExMR, SynSP, SMPLest-X, PromptHMR, CycleAdapt, STRIDE, GMR) are cited
  in `docs/M6.0_研究调研与benchmark.md`; no code from them was copied.

## GVHMR (research / non-commercial)

- Source: https://github.com/zju3dv/GVHMR (see `third_party/GVHMR/`)
- `scripts/run_gvhmr.py` is an adaptation of `tools/demo/demo.py`: the
  pytorch3d renderer import and the `render_incam` / `render_global` calls were
  removed, preprocessing and prediction were kept, and result saving was added.
- GVHMR's license permits educational, research and non-profit use only and
  requires derivative source to remain open-source and non-commercial. The
  GVHMR-derived script and the GVHMR checkout remain under that license; the
  rest of this repository's original code is GPL-3.0-only.
- Cite the GVHMR paper when using it.

## reze-mipo (GPL)

- Source: https://github.com/AmyangXYZ/reze-mipo (see `third_party/reze-mipo/`)
- Only used as a reference for MMD bone names / VMD writing behaviour.
- **No GPL code is copied into `src/`.**

## VPoser v2.0 (MPG, non-commercial)

- Weights: user-downloaded from the official MPI SMPL-X site
  (https://smpl-x.is.tue.mpg.de/), placed under `models/vposer/` and **not
  redistributed** by this repository.
- Official model code: `models/vposer/code/vposer_model.py` and
  `model_components.py`, fetched from
  https://github.com/nghorbani/human_body_prior (main branch,
  `human_body_prior/models/`); MPG license, research/non-commercial use.
  This directory is git-ignored; it is not part of the GPL-3.0-only core.
- `scripts/check_vposer.py` is original project code (GPL-3.0-only); it loads the
  optional local VPoser v2 code/weights and performs a smoke test.

## SMPL-X / SMPL body models

- Downloaded separately into `body_models/`; not redistributed.
- SMPL-X is non-commercial research use; see the official license.
- GVHMR checkpoints are also not redistributed.

## AIblender scenes and PMX models

- The three PMX characters and the three-person `.blend` scenes are user-owned
  assets outside this repository. Do not commit them.
