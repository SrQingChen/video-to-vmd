"""M6.0: does flip-TTA improve the *current* M3.5 default when used upstream?

For a held-out R3 clip:
  body = flip-averaged GVHMR -> adapter -> cleanup -> contact IK
  evidence = the original observation (kp2d, hand/face) at the same frames
  feature = rebuild_features(body, evidence)
  residual = M3.5 net -> conservative policy (torso .85 / arm .2, hands/face off)
  score = the same body-relative metric vector as M3.5

Baseline is the M3.5 default on the unflipped body.  This tells us whether the
flip average is a useful input ensemble for v2, even though it is not a
replacement for the learned residual.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "shim"))
sys.path.insert(0, str(ROOT / "third_party" / "GVHMR"))

from vtv import feedback_data as FD  # noqa: E402
from vtv import feedback_apply as FA  # noqa: E402
from vtv.eval_helpers import predict  # noqa: E402
from vtv.metrics.core import compute_metrics, pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402

METRIC_KEYS = (
    "angle_mean_deg", "world_angle_mean_deg", "position_pct_height",
    "position_p99_pct_height", "contact_recall", "contact_precision",
    "contact_slide_delta", "contact_penetration_delta",
    "contact_height_mae_pct_height", "jerk_rms_ratio",
)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def evidence_for(clip: str, n: int):
    g = np.load(ROOT / "outputs/m35/gvhmr" / clip / "gvhmr_result.npz")
    hand = np.load(ROOT / "outputs/m35/pred" / clip / "hand/hand.npz")
    face = np.load(ROOT / "outputs/m35/pred" / clip / "face/face.npz")
    meta = load_json(ROOT / "outputs/m35/gvhmr" / clip / "gvhmr_meta.json")
    adapter_meta = load_json(ROOT / "outputs/m35/pred" / clip / "adapter.vmd.json")
    fit = lambda a: np.asarray(a, np.float32)[:n]
    return {
        "kp2d": fit(g["kp2d"]), "bbx_xys": fit(g["bbx_xys"]),
        "img_wh": (int(meta["width"]), int(meta["height"])),
        "adapter_scale": float(adapter_meta.get("scale", 1.0)),
        "hand": {"angles": fit(hand["angles"]), "level": fit(hand["level"]),
                 "score": fit(hand["score"]), "detected": fit(hand["detected"])},
        "face": {"weights": fit(face["weights"]), "level": fit(face["level"]),
                 "detected": fit(face["detected"])},
    }


def score(gt_vmd, vmd, model, profile):
    gt = parse_vmd(str(gt_vmd))
    n = min(int(gt["max_frame"]), int(vmd["max_frame"])) + 1
    frames = list(range(n))
    gt_t = pose_track(model, gt, profile, frames=frames)
    pr_t = pose_track(model, vmd, profile, frames=frames)
    m = compute_metrics(gt_t, pr_t, profile)
    rel = FD.body_relative_errors(gt_t, pr_t)
    out = {k: m["vector"][k] for k in METRIC_KEYS}
    out.update({k: v for k, v in rel.items() if k.startswith("group_")
                or k == "body_relative_main_deg"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="R3_si_front")
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/m35/net/m35_best.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs/m60/flip_m35"))
    args = ap.parse_args()
    fb = FA.load_feedback_model(args.ckpt)
    cfg = FA.resolve_config("default")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    for cid in args.clips.split(","):
        cid = cid.strip()
        if not cid:
            continue
        parts = cid.split("_")
        motion, mkey = parts[0], parts[1]
        cfg_m = load_json(ROOT / "configs/models.json")["models"][mkey]
        profile = load_json(ROOT / f"profiles/{mkey}.json")
        model = parse_pmx(cfg_m["pmx"])
        gt_vmd = ROOT / f"outputs/m35/gt/{motion}/{mkey}.vmd"
        base_body = ROOT / "outputs/m35/pred" / cid / "ik.vmd"
        flip_body = ROOT / "outputs/m60/flip_tta" / cid / "flipavg/ik.vmd"
        if not flip_body.exists():
            print(f"SKIP {cid}: missing flipavg body", flush=True)
            continue
        body = parse_vmd(str(flip_body))
        n = int(body["max_frame"]) + 1
        ev = evidence_for(cid, n)
        res = FA.apply_feedback(body, model, profile, fb, ev, cfg)
        out = out_root / f"{cid}_flip_m35.vmd"
        from vtv.mmd.writer import write_vmd
        write_vmd(str(out), res["vmd"].get("model_name", ""), res["vmd"]["bones"],
                  res["vmd"]["morphs"], res["vmd"]["ik_frames"])
        # Baseline: M3.5 default on the unflipped body, same code path.
        base_body_vmd = parse_vmd(str(base_body))
        ev_b = evidence_for(cid, int(base_body_vmd["max_frame"]) + 1)
        res_b = FA.apply_feedback(base_body_vmd, model, profile, fb, ev_b, cfg)
        b = score(gt_vmd, res_b["vmd"], model, profile)
        c = score(gt_vmd, res["vmd"], model, profile)
        d = {k: round(c[k] - b[k], 5) for k in b
             if b[k] is not None and c[k] is not None}
        records.append({"clip": cid, "baseline": b, "flip_m35": c, "delta": d,
                        "flip_residual_mean_abs": res["residual_mean_abs"],
                        "base_residual_mean_abs": res_b["residual_mean_abs"]})
        print(f"{cid}: main {b['body_relative_main_deg']:.2f}->"
              f"{c['body_relative_main_deg']:.2f} "
              f"chest {b['group_chest_deg']:.2f}->{c['group_chest_deg']:.2f} "
              f"pos {b['position_pct_height']:.3f}->{c['position_pct_height']:.3f} "
              f"jerk {b['jerk_rms_ratio']:.3f}->{c['jerk_rms_ratio']:.3f}",
              flush=True)
    (out_root / "report.json").write_text(
        json.dumps({"clips": records}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("WROTE", out_root / "report.json")


if __name__ == "__main__":
    main()
