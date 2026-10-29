"""M6.0 benchmark: UnderPressure foot-contact detection on VTV clips.

UnderPressure (Mourot et al., CGF/SCA 2022) predicts per-foot contact labels
from a 23-joint skeletal motion at 100 fps.  It is the only 2024-2026 contact
candidate with weights already on disk (``third_party/UnderPressure``); its
footskate cleanup is not used here.  The 23-joint topology differs from SMPL-X,
so joints are mapped approximately (spine/chest duplicated) and the result is
reported as a component-level benchmark, not as a body-pose candidate.

Baseline contact is the same height threshold used by ``build_features``
(toe height <= ground + 2% of body height).  GT contact is the same threshold
on the GT VMD.  Metrics: precision / recall / F1 per side and averaged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "shim"))
sys.path.insert(0, str(ROOT / "third_party" / "GVHMR"))
sys.path.insert(0, str(ROOT / "third_party" / "UnderPressure"))

from vtv.metrics.core import pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def smplx_joints(npz_path: Path, chunk: int = 64) -> np.ndarray:
    """(N,55,3) SMPL-X joints in metres from a GVHMR npz."""
    from hmr4d.utils.smplx_utils import make_smplx

    d = np.load(npz_path)
    bp = torch.tensor(d["smpl_params_global/body_pose"], dtype=torch.float32)
    go = torch.tensor(d["smpl_params_global/global_orient"], dtype=torch.float32)
    tr = torch.tensor(d["smpl_params_global/transl"], dtype=torch.float32)
    be = torch.tensor(d["smpl_params_global/betas"], dtype=torch.float32)
    model = make_smplx("supermotion", num_betas=10)
    outs = []
    with torch.no_grad():
        for i in range(0, len(bp), chunk):
            out = model(betas=be[i:i + chunk], body_pose=bp[i:i + chunk],
                        global_orient=go[i:i + chunk], transl=tr[i:i + chunk])
            outs.append(out.joints.cpu().numpy())
    return np.concatenate(outs, 0)


# SMPL-X body joint index -> UnderPressure 23-joint topology.  Their skeleton
# has four spine joints; SMPL-X has three, so spine_4 reuses the chest joint.
UP_MAP = {
    "pelvis": 0,
    "spine_1": 3, "spine_2": 6, "spine_3": 9, "spine_4": 9,
    "neck": 12, "head": 15,
    "right_clavicle": 14, "right_shoulder": 17, "right_elbow": 19,
    "right_wrist": 21,
    "left_clavicle": 13, "left_shoulder": 16, "left_elbow": 18,
    "left_wrist": 20,
    "right_hip": 2, "right_knee": 5, "right_ankle": 8, "right_foot": 11,
    "left_hip": 1, "left_knee": 4, "left_ankle": 7, "left_foot": 10,
}
UP_ORDER = list(UP_MAP)


def to_up_topology(joints: np.ndarray) -> np.ndarray:
    """(N,55,3) Y-up SMPL-X -> (N,23,3) Z-up UnderPressure order."""
    p = joints[:, [UP_MAP[k] for k in UP_ORDER]]
    # Y-up -> Z-up; the horizontal sign is irrelevant for contact timing.
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def resample(a: np.ndarray, n_out: int) -> np.ndarray:
    n = len(a)
    x = np.linspace(0, n - 1, n_out)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (x - i0)[:, None, None]
    return a[i0] * (1 - w) + a[i1] * w


def contact_metrics(pred: np.ndarray, gt: np.ndarray):
    out = {}
    for s, name in enumerate(("l", "r")):
        p = pred[:, s].astype(bool)
        g = gt[:, s].astype(bool)
        tp = int((p & g).sum())
        fp = int((p & ~g).sum())
        fn = int((~p & g).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[name] = {"precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4), "pred_frames": int(p.sum()),
                     "gt_frames": int(g.sum())}
    out["mean_f1"] = round(float(np.mean([out["l"]["f1"], out["r"]["f1"]])), 4)
    out["mean_recall"] = round(float(np.mean([out["l"]["recall"],
                                              out["r"]["recall"]])), 4)
    out["mean_precision"] = round(float(np.mean([out["l"]["precision"],
                                                 out["r"]["precision"]])), 4)
    return out


def gt_contacts(gt_vmd: Path, model, profile, n: int, height: float):
    gt = parse_vmd(str(gt_vmd))
    frames = list(range(n))
    tr = pose_track(model, gt, profile, frames=frames)
    ys = []
    for side in ("l", "r"):
        toe = np.asarray(tr["world"][f"toe_{side}"], float)
        ys.append(toe[:, 1] / height)
    y = np.stack(ys, axis=1)
    ground = float(np.min(y))
    return (y <= ground + 0.02).astype(np.int8)


def _rest(npz_path, betas):
    from vtv.canonical_body import SMPLX_INDEX
    from hmr4d.utils.smplx_utils import make_smplx
    m = make_smplx("supermotion", num_betas=10)
    with torch.no_grad():
        out = m(betas=torch.tensor(np.asarray(betas), dtype=torch.float32),
                body_pose=torch.zeros(1, 63), global_orient=torch.zeros(1, 3),
                transl=torch.zeros(1, 3))
    J = out.joints[0, :22].numpy()
    return {j: [float(c) for c in J[i]] for j, i in SMPLX_INDEX.items()}, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="R3_si_front")
    ap.add_argument("--out", default=str(ROOT / "outputs/m60/underpressure"))
    args = ap.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    from data import TOPOLOGY  # noqa: E402
    import models  # noqa: E402

    model = models.DeepNetwork(
        state_dict=torch.load(ROOT / "third_party/UnderPressure/pretrained.tar",
                              map_location="cpu")["model"]).eval()
    records = []
    for cid in args.clips.split(","):
        cid = cid.strip()
        parts = cid.split("_")
        motion, mkey = parts[0], parts[1]
        npz = ROOT / "outputs/m35/gvhmr" / cid / "gvhmr_result.npz"
        gt_vmd = ROOT / f"outputs/m35/gt/{motion}/{mkey}.vmd"
        cfg = load_json(ROOT / "configs/models.json")["models"][mkey]
        profile = load_json(ROOT / f"profiles/{mkey}.json")
        pmx = parse_pmx(cfg["pmx"])
        height = float(profile["height"])
        j = to_up_topology(smplx_joints(npz))
        n = len(j)
        # UnderPressure was trained at 100 fps.
        j100 = resample(j, int(round(n / 30.0 * 100.0)))
        with torch.no_grad():
            contacts = model.contacts(torch.tensor(j100, dtype=torch.float32)[None])
        c = contacts[0]  # (F, LR, ...)
        while c.dim() > 2:
            c = c.any(dim=-1)
        c = c.numpy().astype(np.int8)
        # Back to 30 fps by majority over the 100/30 window.
        idx = np.clip(np.round(np.arange(n) / 30.0 * 100.0).astype(int), 0, len(c) - 1)
        pred = c[idx]
        gt = gt_contacts(gt_vmd, pmx, profile, n, height)
        # Baseline: predicted toe height from the same GVHMR npz.
        base = pred_toe_contacts_baseline(npz, pmx, profile, n, height)
        rec = {"clip": cid, "frames": n,
               "underpressure": contact_metrics(pred, gt),
               "height_baseline": contact_metrics(base, gt),
               "up_frames": int(pred.sum()), "baseline_frames": int(base.sum()),
               "gt_frames": int(gt.sum())}
        records.append(rec)
        print(cid, json.dumps(rec, ensure_ascii=False), flush=True)
    (out_root / "report.json").write_text(
        json.dumps({"clips": records}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("WROTE", out_root / "report.json")


def pred_toe_contacts_baseline(npz_path, model, profile, n, height):
    """Contact labels from the predicted VMD's toe height (build_features)."""
    from vtv.adapter import adapt, canonical_from_smplx
    from vtv.metrics.core import pose_track
    from vtv.mmd.writer import write_vmd
    import tempfile

    d = np.load(npz_path)
    params = {k: d[f"smpl_params_global/{k}"] for k in
              ("body_pose", "global_orient", "transl")}
    rest, _ = _rest(npz_path, d["smpl_params_global/betas"][:1])
    motion = canonical_from_smplx(params, rest, slow_factor=1.0)
    res = adapt(motion, model, profile, ground_calibrate=True,
                stage_yaw_deg=180.0)
    tmp = Path(tempfile.gettempdir()) / "m60_up_base.vmd"
    write_vmd(str(tmp), model.get("name", ""), res["bone_frames"], {},
              res.get("ik_frames", []))
    tr = pose_track(model, parse_vmd(str(tmp)), profile, frames=list(range(n)))
    y = np.stack([np.asarray(tr["world"][f"toe_{s}"], float)[:, 1] / height
                  for s in ("l", "r")], axis=1)
    ground = float(np.min(y))
    return (y <= ground + 0.02).astype(np.int8)


if __name__ == "__main__":
    main()
