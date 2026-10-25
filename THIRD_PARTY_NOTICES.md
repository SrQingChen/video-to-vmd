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

## GVHMR (research / non-commercial)

- Source: https://github.com/zju3dv/GVHMR (see `third_party/GVHMR/`)
- `scripts/run_gvhmr.py` is an adaptation of `tools/demo/demo.py`: the
  pytorch3d renderer import and the `render_incam` / `render_global` calls were
  removed, preprocessing and prediction were kept, and result saving was added.
- GVHMR's license permits educational, research and non-profit use only and
  requires derivative source to remain open-source and non-commercial. The
  GVHMR-derived script and the GVHMR checkout remain under that license; the
  rest of this repository's original code is Apache-2.0.
- Cite the GVHMR paper when using it.

## reze-mipo (GPL)

- Source: https://github.com/AmyangXYZ/reze-mipo (see `third_party/reze-mipo/`)
- Only used as a reference for MMD bone names / VMD writing behaviour.
- **No GPL code is copied into `src/`.**

## SMPL-X / SMPL body models

- Downloaded separately into `body_models/`; not redistributed.
- SMPL-X is non-commercial research use; see the official license.
- GVHMR checkpoints are also not redistributed.

## AIblender scenes and PMX models

- The three PMX characters and the three-person `.blend` scenes are user-owned
  assets outside this repository. Do not commit them.
