# Model and dependency licenses

This repository's original code is Apache-2.0 (see `LICENSE`). Third-party
software, model weights, body models, PMX assets and test media keep their own
licenses. **No model weights, body models, PMX models, VMD motions or test
videos are redistributed here.**

| Component | Role | License / status |
|---|---|---|
| GVHMR | Feed-forward body/global HMR (`scripts/run_gvhmr.py`, optional backend) | Research / non-commercial; weights not bundled; cite the paper |
| WHAM | Optional body backend | MIT code; SMPL body model license applies |
| OSX | Optional hand backend | MIT code; SMPL-X dependency |
| SMPL-X / SMPL | Body models | Non-commercial research; must be downloaded from MPI; not redistributable |
| DWPose / RTMW | 2D whole-body keypoints | Apache ecosystem; verify weight terms |
| SAM 2 | Person/part segmentation (optional) | Apache; verify weight terms |
| Depth Anything v2 / DA3 | Relative depth / camera (optional) | Apache code; verify weight terms |
| MediaPipe | Hand/face fallback (optional) | Apache |
| HaMeR / WiLoR / SMPLer-X | Optional hand experts (M3) | Verify individual model licenses before use |
| nvdiffrast | Research-only differentiable rasterizer | NVIDIA research terms; **not** a production dependency in v2 |
| mmd2gltf-gui | PMX/VMD parser and evaluator reference | MIT; vendored files retain `src/vtv/mmd/LICENSE.mmd2gltf` |
| reze-mipo | Bone-name/VMD-writing reference | GPL-3.0; **no GPL code is copied into `src/`** |
| reze-rig / vmd_sizing | Retargeting/sizing references | Check upstream licenses before copying code |
| PMX-VMD-Scripting-Tools | Format/reference tools | Check upstream license before copying code |
| AIblender PMX models, R1-R3 VMDs, T1-T5 videos | User-owned test assets | Not included; do not redistribute |

## Default profile

The default research profile uses GVHMR + SMPL-X and is therefore
**non-commercial research use**. Commercial use requires replacing the
non-commercial backends and obtaining the relevant licenses. The README does
not grant commercial rights to third-party components.
