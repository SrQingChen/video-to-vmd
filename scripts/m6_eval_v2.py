"""Evaluate M6 v2 (one-pass or two-pass) on held-out clips.

Uses the same metric vector and gates as ``eval_feedback.py`` but applies the
residual through ``feedback_apply_v2`` so the second pass sees the corrected
pose and the corrected hand/face baseline.  Baseline is always the exact
``cleanup+ik`` body + M3 hand/face, so the comparison is unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import feedback_apply as FA  # noqa: E402
from vtv import feedback_apply_v2 as FAV2  # noqa: E402
from vtv import feedback_data as FD  # noqa: E402
from vtv.metrics.core import compute_metrics, pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402
from eval_feedback import aggregate, gates  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fit(a, n):
    out = np.zeros((n,) + np.asarray(a).shape[1:], np.asarray(a).dtype)
    m = min(n, len(a))
    out[:m] = np.asarray(a)[:m]
    return out


def evidence(meta, n):
    g = np.load(ROOT / meta["paths"]["gvhmr"])
    gm = load_json((ROOT / meta["paths"]["gvhmr"]).parent / "gvhmr_meta.json")
    hand = np.load(ROOT / meta["paths"]["hand"])
    face = np.load(ROOT / meta["paths"]["face"])
    am = load_json((ROOT / meta["paths"]["body"]).parent / "adapter.vmd.json")
    return {
        "kp2d": fit(g["kp2d"], n), "bbx_xys": fit(g["bbx_xys"], n),
        "img_wh": (int(gm["width"]), int(gm["height"])),
        "adapter_scale": float(am.get("scale", 1.0)),
        "hand": {"angles": fit(hand["angles"], n), "level": fit(hand["level"], n),
                 "score": fit(hand["score"], n),
                 "detected": fit(hand["detected"], n)},
        "face": {"weights": fit(face["weights"], n), "level": fit(face["level"], n),
                 "detected": fit(face["detected"], n)},
    }


def hand_error(gt, pred, level):
    m = np.asarray(level) <= 1
    if not m.any():
        return None, 0
    e = np.abs(np.asarray(gt, float) - np.asarray(pred, float))[m]
    return float(np.nanmean(e)), int(m.sum())


def face_error(gt, pred, level, channels):
    m = (np.asarray(level) <= 1)[:, None] & (np.asarray(channels) > 0)[None, :]
    if not m.any():
        return None, 0
    e = np.abs(np.asarray(gt, float) - np.asarray(pred, float))[m]
    return float(np.nanmean(e)), int(m.sum())


def score(meta, vmd, pair, gt_vmd, model, profile, hand_out=None, face_out=None):
    n = min(int(vmd["max_frame"]), int(gt_vmd["max_frame"])) + 1
    frames = list(range(n))
    gt_t = pose_track(model, gt_vmd, profile, frames=frames)
    corr_t = pose_track(model, vmd, profile, frames=frames)
    body = parse_vmd(str(ROOT / meta["paths"]["body"]))
    base_t = pose_track(model, body, profile, frames=frames)
    mb = compute_metrics(gt_t, base_t, profile)
    mc = compute_metrics(gt_t, corr_t, profile)
    rb = FD.body_relative_errors(gt_t, base_t)
    rc = FD.body_relative_errors(gt_t, corr_t)
    gt_hand = fit(pair["gt_hand"], n)
    gt_face = fit(pair["gt_face"], n)
    hl = np.asarray(pair["hand_level"])[:n]
    fl = np.asarray(pair["face_level"])[:n]
    hb, _ = hand_error(gt_hand, pair["pred_hand"], hl)
    hc, _ = hand_error(gt_hand, hand_out if hand_out is not None
                        else pair["pred_hand"], hl)
    fb, _ = face_error(gt_face, pair["pred_face"], fl, pair["face_mask"])
    fc, _ = face_error(gt_face, face_out if face_out is not None
                        else pair["pred_face"], fl, pair["face_mask"])
    keys = ("angle_mean_deg", "world_angle_mean_deg", "position_pct_height",
            "position_p99_pct_height", "root_traj_pct_height",
            "root_drift_pct_height", "contact_recall", "contact_precision",
            "contact_slide_delta", "contact_penetration_delta",
            "contact_height_mae_pct_height", "jerk_rms_ratio", "spike_delta")
    b = {k: mb["vector"][k] for k in keys}
    c = {k: mc["vector"][k] for k in keys}
    b.update({k: v for k, v in rb.items() if k.endswith("_deg")})
    c.update({k: v for k, v in rc.items() if k.endswith("_deg")})
    return {"baseline": b, "corrected": c,
            "hand_mae_deg": {"before": hb, "after": hc},
            "face_mae": {"before": fb, "after": fc},
            "delta": {k: round(c[k] - b[k], 5) for k in b
                      if b[k] is not None and c[k] is not None}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt2", default=None)
    ap.add_argument("--index", default=str(ROOT / "outputs/m6/data/index.json"))
    ap.add_argument("--pair-root", default=str(ROOT / "outputs/m6/data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out", default=str(ROOT / "outputs/m6/eval"))
    ap.add_argument("--mode", default="full")
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--clips", default=None)
    ap.add_argument("--no-arm-torso-check", action="store_true",
                    help="fallback ablation: keep jerk/slide/penetration, drop "
                         "the arm-torso capsule proxy trigger")
    ap.add_argument("--jerk-tol", type=float, default=None,
                    help="fallback jerk tolerance (default qc.JERK_TOL)")
    args = ap.parse_args()
    fb1 = FA.load_feedback_model(args.ckpt)
    fb2 = FA.load_feedback_model(args.ckpt2) if args.ckpt2 else None
    cfg = FA.resolve_config(args.mode, overrides=(
        json.loads(args.overrides) if args.overrides else None))
    clips = load_json(args.index)["clips"]
    keep = set(args.split.split(","))
    if args.clips:
        wanted = set(args.clips.split(","))
        clips = [c for c in clips if c["clip"] in wanted]
    else:
        clips = [c for c in clips if c["split"] in keep]
    out_dir = (Path(args.out) / args.tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fallback = dict(FA.DEFAULT_FALLBACK)
    if args.no_arm_torso_check:
        fallback["arm_torso_check"] = False
    if args.jerk_tol is not None:
        fallback["jerk_tol"] = float(args.jerk_tol)
    recs = []
    for i, meta in enumerate(clips, 1):
        cid = meta["clip"]
        cfg_m = load_json(ROOT / "configs/models.json")["models"][meta["model"]]
        profile = load_json(ROOT / f"profiles/{meta['model']}.json")
        model = parse_pmx(cfg_m["pmx"])
        body = parse_vmd(str(ROOT / meta["paths"]["body"]))
        gt = parse_vmd(str(ROOT / meta["paths"]["gt"]))
        n = min(int(body["max_frame"]), int(gt["max_frame"])) + 1
        ev = evidence(meta, n)
        pair = np.load(Path(args.pair_root) / cid / "pair.npz")
        res = FAV2.apply_feedback_v2(body, model, profile, fb1, fb2, ev, cfg,
                                      fallback)
        vmd = res["vmd"]
        p = out_dir / f"{cid}_{args.tag}.vmd"
        write_vmd(str(p), vmd.get("model_name", ""), vmd["bones"],
                  vmd["morphs"], vmd["ik_frames"])
        r = score(meta, parse_vmd(str(p)), pair, gt, model, profile,
                  hand_out=res.get("hand_out"), face_out=res.get("face_out"))
        r.update({"clip": cid, "motion": meta["motion"], "model": meta["model"],
                  "view": meta["view"], "split": meta["split"], "frames": n,
                  "after_vmd": str(p.relative_to(ROOT)),
                  "segments_fallback": res["segments_fallback"],
                  "whole_clip_fallback": res["whole_clip_fallback"],
                  "handface_fallback": res["handface_fallback"]})
        recs.append(r)
        d = r["delta"]
        print(f"[{i}/{len(clips)}] {cid} main "
              f"{r['baseline']['body_relative_main_deg']:.2f}->"
              f"{r['corrected']['body_relative_main_deg']:.2f} "
              f"pos {d['position_pct_height']:+.3f} "
              f"slide {d['contact_slide_delta']:+.4f} "
              f"jerk {d['jerk_rms_ratio']:+.3f}", flush=True)
    agg = aggregate(recs)
    gate = gates(agg, recs)
    report = {"ckpt": args.ckpt, "ckpt2": args.ckpt2, "tag": args.tag,
              "mode": args.mode, "split": args.split, "clips": recs,
              "aggregate": agg, "gates": gate}
    (Path(args.out) / f"report_{args.tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("AGG", {k: (round(v["corrected"]["body_relative_main_deg"], 2)
                     if "corrected" in v else None) for k, v in agg.items()})
    print("GATES", json.dumps(gate, ensure_ascii=False))
    print("WROTE", Path(args.out) / f"report_{args.tag}.json")


if __name__ == "__main__":
    main()
