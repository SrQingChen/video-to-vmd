# Model and dependency licenses

This repository's original code is GPL-3.0-only (see `LICENSE`). Third-party
software, model weights, body models, PMX assets and test media keep their own
licenses. **No model weights, body models, PMX models, VMD motions or test
videos are redistributed here.**

| Component | Role | License / status |
|---|---|---|
| M3.5 feedback network (`m35_best.pt`) | M4/M6 `--feedback default` torso/arm residual network | **Project-trained** (2.10M BiGRU) from synthetic renders + GVHMR/SMPL-X-derived teachers; research/non-commercial because of those teachers; checkpoint **not redistributed** (`outputs/` git-ignored) |
| M6 v2 feedback network (`m6_v2_hf_ft_best.pt`) | Default M6 `--feedback v2` torso/arm residual network | **Project-trained** (2.18M BiGRU) from synthetic renders, synthetic hand/face L0 evidence, and GVHMR/SMPL-X-derived teachers; research/non-commercial for the same reason; checkpoint **not redistributed** (`outputs/` git-ignored). `v2_full` reuses the same checkpoint with hand/face heads. |
| GVHMR | Feed-forward body/global HMR (`scripts/run_gvhmr.py`, optional backend) | Research / non-commercial; weights not bundled; cite the paper |
| WHAM | Optional body backend | MIT code; SMPL body model license applies |
| OSX | Optional hand backend | MIT code; SMPL-X dependency |
| SMPL-X / SMPL | Body models | Non-commercial research; must be downloaded from MPI; not redistributable |
| DWPose / RTMW | 2D whole-body keypoints | Apache ecosystem; verify weight terms |
| SAM 2 | Person/part segmentation (optional) | Apache; verify weight terms |
| Depth Anything v2 / DA3 | Relative depth / camera (optional) | Apache code; verify weight terms |
| MediaPipe | Hand/face fallback (M3, used) | Apache-2.0; `hand_landmarker.task` / `face_landmarker.task` downloaded from the official `storage.googleapis.com/mediapipe-models` bucket, **not redistributed** (git-ignored `models/mediapipe/`) |
| HaMeR / WiLoR / SMPLer-X | Optional hand experts (M3) | Verify individual model licenses before use |
| VPoser v2.0 | Optional SMPL-X pose prior (M2.4d) | Non-commercial research (MPG); weights and official code are **not** redistributed; local `models/vposer/` (git-ignored); downloaded from https://smpl-x.is.tue.mpg.de/ |
| Multi-HMR v1 | Off-the-shelf 3D body benchmark (M2.4e); **not integrated** | NAVER Multi-HMR non-commercial license; code/weights not redistributed; `third_party/multi-hmr/` git-ignored; official checkpoint from download.europe.naverlabs.com / HuggingFace `naver/multiHMR_896_L` |
| SAM 3D Body | Feasibility only (M2.4e); **not downloaded** | SAM License (custom; commercial use permitted under its terms); HuggingFace checkpoints are gated (`facebook/sam-3d-body-dinov3`, `facebook/sam-3d-body-vith`); output MHR, no official SMPL-X converter in repo |
| SMPLest-X | Feasibility only (M2.4e); **not downloaded** | S-Lab License 1.0 (non-commercial); only Huge 8.2 GB checkpoint; Linux/conda install; not usable on the 8 GB target |
| WHAC | Feasibility only (M2.4e); **not downloaded** | S-Lab License 1.0 (non-commercial); depends on SMPLest-X Huge + DPVO; not used |
| nvdiffrast | Research-only differentiable rasterizer | NVIDIA research terms; **not** a production dependency in v2 |
| mmd2gltf-gui | PMX/VMD parser and evaluator reference | MIT; vendored files retain `src/vtv/mmd/LICENSE.mmd2gltf` |
| reze-mipo | Bone-name/VMD-writing reference | GPL-3.0; **no GPL code is copied into `src/`** |
| reze-rig / vmd_sizing | Retargeting/sizing references | Check upstream licenses before copying code |
| PMX-VMD-Scripting-Tools | Format/reference tools | Check upstream license before copying code |
| Blender + MMD Tools | External renderer / MMD importer (M4 2K, D-chain bake) | Blender GPL-2.0-or-later, MMD Tools GPL-3.0; installed by the user, **not redistributed**; invoked as separate tools |
| FFmpeg | External MP4 encoder + M5 chunk cutting / 30 fps master video | LGPL/GPL depending on build; not redistributed |
| AIblender PMX models, R1-R3 VMDs, T1-T5 videos | User-owned test assets | Not included; do not redistribute |

## M5 original code

`src/vtv/longvideo.py` (chunk plan / alignment / VMD merge / seam QC),
`src/vtv/framing.py` (auto-frame), `src/vtv/cli.py` and `demo/` are original
GPL-3.0-only project code. They call the same external tools (GVHMR, ffmpeg,
Blender/MMD Tools) under those tools' own licenses; no new third-party code,
weights or assets are added.

## Default profile

The default research profile uses GVHMR + SMPL-X and the project-trained M6 v2
feedback network (M3.5 remains available via `--feedback default`), and is
therefore **non-commercial research use**. The networks were trained on
synthetic renders generated from the user's PMX assets, with
GVHMR/SMPL-X-derived teachers; their weights are not distributed here. Commercial
use requires replacing the non-commercial backends and obtaining the relevant
licenses. The README does not grant commercial rights to third-party
components.
