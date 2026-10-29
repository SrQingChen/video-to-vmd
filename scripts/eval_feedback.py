"""Evaluate the M3.5 residual network on held-out segments.

For every clip: features -> network -> bounded residual -> corrected VMD
(body residual + L0/L1-gated hand/face residual) -> score against GT with the
same metric vector used by M0-M3, plus the body-relative arm/chest metrics.

Baseline is exactly ``cleanup+ik`` (``pred/ik.vmd`` + M3 hands/face), i.e. the
current default pipeline.  Gates are reported, not softened.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.eval_helpers import (  # noqa: E402
    gate_residual, load_net, predict, smooth_residual,
)
from vtv.metrics.core import compute_metrics, pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


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


def score_clip(meta, residual, out_dir, tag="after", write=True, ck=None,
               pair_root=None, root_contact_joint=False):
    clip = meta["clip"]
    model_key = meta["model"]
    cfg = load_json(ROOT / "configs/models.json")["models"][model_key]
    profile = load_json(ROOT / f"profiles/{model_key}.json")
    model = parse_pmx(cfg["pmx"])
    gt_vmd = parse_vmd(str(ROOT / meta["paths"]["gt"]))
    body = parse_vmd(str(ROOT / meta["paths"]["body"]))
    pr = Path(pair_root) if pair_root else ROOT / "outputs/m35/data"
    pair = np.load(pr / clip / "pair.npz")
    hand = np.load(ROOT / meta["paths"]["hand"])
    face = np.load(ROOT / meta["paths"]["face"])
    n = int(body["max_frame"]) + 1
    frames = list(range(n))
    gt_track = pose_track(model, gt_vmd, profile, frames=frames)
    gt_strip = pose_track(model, FD.strip_twist_vmd(gt_vmd, profile), profile,
                          frames=frames)
    base_track = pose_track(model, body, profile, frames=frames)

    # ---- apply -----------------------------------------------------------
    res = np.array(residual, np.float32, copy=True)
    if meta["variant"] != "base":
        res[:, FD.HAND_SLICE] = 0.0
    hl = pair["hand_level"]
    fl = pair["face_level"]
    ck = ck or {}
    res = gate_residual(res, hl, fl, FD.face_target_names(profile),
                        ck.get("hand_supported"), ck.get("face_supported"))

    corr_body = FD.apply_body_residual(body, model, profile, res)
    if root_contact_joint and np.any(res[:, FD.ROOT_SLICE]):
        from vtv.contact_ik import solve_contact_ik
        corr_body = solve_contact_ik(corr_body, model, profile,
                                     kp2d=pair["kp2d"])["vmd"]
    ha = FD.apply_hand_residual(pair["pred_hand"], res[:, FD.HAND_SLICE])
    fw = FD.apply_face_residual(pair["pred_face"], res[:, FD.FACE_SLICE])
    vmd = FD.assemble_vmd(corr_body, profile, ha, fw, n_frames=n)
    out_p = Path(out_dir) / f"{clip}_{tag}.vmd"
    if write:
        write_vmd(str(out_p), vmd.get("model_name", ""), vmd["bones"],
                  vmd["morphs"], vmd["ik_frames"])
    corr_track = pose_track(model, vmd, profile, frames=frames)

    # ---- score -----------------------------------------------------------
    mb = compute_metrics(gt_track, base_track, profile)
    mc = compute_metrics(gt_track, corr_track, profile)
    rb = FD.body_relative_errors(gt_track, base_track)
    rc = FD.body_relative_errors(gt_track, corr_track)
    # twist-normalised variant: what the VMD representation can actually match
    rb_s = FD.body_relative_errors(gt_strip, base_track)
    rc_s = FD.body_relative_errors(gt_strip, corr_track)
    gt_hand = _fit(pair["gt_hand"], n)
    gt_face = _fit(pair["gt_face"], n)
    hb, hnb = hand_error(gt_hand, pair["pred_hand"], hl)
    hc, hnc = hand_error(gt_hand, ha, hl)
    fb, fnb = face_error(gt_face, pair["pred_face"], fl, pair["face_mask"])
    fc, fnc = face_error(gt_face, fw, fl, pair["face_mask"])

    def vec(m):
        keys = ("angle_mean_deg", "world_angle_mean_deg", "position_pct_height",
                "position_p99_pct_height", "root_traj_pct_height",
                "root_drift_pct_height", "contact_recall", "contact_precision",
                "contact_slide_delta", "contact_penetration_delta",
                "contact_height_mae_pct_height", "jerk_rms_ratio", "spike_delta")
        return {k: m["vector"][k] for k in keys}

    def diag(group_slice):
        msk = np.asarray(pair["mask"][:, group_slice]) > 0
        if not msk.any():
            return {"masked": 0}
        a = np.asarray(pair["target"][:, group_slice])[msk]
        b = np.asarray(res[:, group_slice])[msk]
        lim_all = np.asarray((ck or {}).get("limits", FD.residual_limits()))
        sub = lim_all[group_slice]
        sub_cols = np.tile(sub, (msk.shape[0], 1))[msk]
        ea = a / sub_cols
        eb = b / sub_cols
        corr = (float(np.corrcoef(a, b)[0, 1])
                if a.std() > 1e-9 and b.std() > 1e-9 else None)
        return {"masked": int(msk.sum()),
                "target_abs": round(float(np.abs(a).mean()), 4),
                "pred_abs": round(float(np.abs(b).mean()), 4),
                "corr": None if corr is None else round(corr, 4),
                "rmse_norm": round(float(np.sqrt(np.mean((ea - eb) ** 2))), 4)}

    rec = {
        "clip": clip, "motion": meta["motion"], "model": model_key,
        "view": meta["view"], "variant": meta["variant"], "split": meta["split"],
        "frames": n,
        "residual_diag": {k: diag(sl) for k, sl in FD.residual_groups().items()},
        "baseline": {**vec(mb), **rb, "body_relative_strip_deg": rb_s["body_relative_main_deg"]},
        "corrected": {**vec(mc), **rc, "body_relative_strip_deg": rc_s["body_relative_main_deg"]},
        "hand_mae_deg": {"before": hb, "after": hc, "frames": hnb},
        "face_mae": {"before": fb, "after": fc, "frames": fnb},
        "residual_mean_abs": {k: float(np.abs(res[:, sl]).mean())
                              for k, sl in FD.residual_groups().items()},
        "after_vmd": str(out_p.relative_to(ROOT)) if write else None,
    }
    keys = [k for k in rec["baseline"] if isinstance(rec["baseline"][k], (int, float))]
    rec["delta"] = {k: round(rec["corrected"][k] - rec["baseline"][k], 5)
                    for k in keys if rec["corrected"].get(k) is not None}
    return rec


def _fit(a, n):
    out = np.zeros((n,) + a.shape[1:], np.float32)
    m = min(n, len(a))
    out[:m] = a[:m]
    return out


def aggregate(recs):
    out = {}
    for split in ("train", "val", "test"):
        rs = [r for r in recs if r["split"] == split]
        if not rs:
            continue
        keys = [k for k, v in rs[0]["baseline"].items() if isinstance(v, (int, float))]
        agg = {"clips": len(rs), "frames": int(sum(r["frames"] for r in rs))}
        for side in ("baseline", "corrected"):
            agg[side] = {k: float(np.mean([r[side][k] for r in rs]))
                         for k in keys if all(r[side].get(k) is not None for r in rs)}
        agg["delta"] = {k: round(agg["corrected"][k] - agg["baseline"][k], 5)
                        for k in agg["baseline"]}
        for grp in ("model", "view"):
            agg[f"by_{grp}"] = {}
            for val in sorted({r[grp] for r in rs}):
                sub = [r for r in rs if r[grp] == val]
                agg[f"by_{grp}"][val] = {
                    "clips": len(sub),
                    "main_before": float(np.mean([r["baseline"]["body_relative_main_deg"] for r in sub])),
                    "main_after": float(np.mean([r["corrected"]["body_relative_main_deg"] for r in sub])),
                }
        out[split] = agg
    return out


def gates(agg, recs):
    t = agg.get("test")
    if not t:
        return {"ok": False, "reason": "no test clips"}
    b, c = t["baseline"], t["corrected"]
    checks = {}
    # 1) body-relative main metric improves and does so on multiple models/views
    checks["body_relative_main_improves"] = c["body_relative_main_deg"] < b["body_relative_main_deg"]
    for grp in ("by_model", "by_view"):
        vals = t.get(grp, {})
        good = sum(1 for v in vals.values() if v["main_after"] < v["main_before"])
        checks[f"{grp}_majority_improves"] = good >= max(2, len(vals) - 1)
    # 2) no overall angle/position regression
    secondary = {
        "angle_not_worse": c["angle_mean_deg"] <= b["angle_mean_deg"] * 1.01 + 0.05,
        "world_angle_not_worse": c["world_angle_mean_deg"] <= b["world_angle_mean_deg"] * 1.01 + 0.05,
        "position_not_worse": c["position_pct_height"] <= b["position_pct_height"] * 1.01 + 0.02,
    }
    # 3) contact / penetration / timing
    checks["contact_recall_not_worse"] = (c["contact_recall"] is not None
                                          and b["contact_recall"] is not None
                                          and c["contact_recall"] >= b["contact_recall"] - 0.02)
    checks["contact_slide_not_worse"] = c["contact_slide_delta"] <= b["contact_slide_delta"] * 1.05 + 0.01
    checks["penetration_not_worse"] = c["contact_penetration_delta"] <= b["contact_penetration_delta"] + 0.05
    checks["jerk_not_worse"] = c["jerk_rms_ratio"] <= b["jerk_rms_ratio"] * 1.10
    # 4) hands / face not worse on the supervised frames
    hs = [r for r in recs if r["split"] == "test" and r["hand_mae_deg"]["before"] is not None]
    if hs:
        hb = float(np.mean([r["hand_mae_deg"]["before"] for r in hs]))
        ha = float(np.mean([r["hand_mae_deg"]["after"] for r in hs]))
        checks["hands_not_worse"] = ha <= hb * 1.02 + 1e-3
    else:
        hb = ha = None
        checks["hands_not_worse"] = True
    fs = [r for r in recs if r["split"] == "test" and r["face_mae"]["before"] is not None]
    if fs:
        fb = float(np.mean([r["face_mae"]["before"] for r in fs]))
        fa = float(np.mean([r["face_mae"]["after"] for r in fs]))
        checks["face_not_worse"] = fa <= fb + 0.02
    else:
        fb = fa = None
        checks["face_not_worse"] = True
    return {"ok": all(checks.values()) and all(secondary.values()),
            "checks": checks, "secondary": secondary,
            "task_gate": bool(all(checks.values())),
            "note": "gate on the test split (R3, held-out motion); val is not a gate. "
                    "task_gate = the M3.5 stated criteria (main + by model/view + "
                    "contact/penetration/timing + hands/face); secondary = angle/"
                    "world-angle/position, reported but not part of the stated gate.",
            "test_hand_mae_deg": {"before": hb, "after": ha},
            "test_face_mae": {"before": fb, "after": fa}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/m35/net/m35_best.pt"))
    ap.add_argument("--index", default=str(ROOT / "outputs/m35/data/index.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/m35/eval"))
    ap.add_argument("--split", default="val,test")
    ap.add_argument("--tag", default="after")
    ap.add_argument("--ablate", default="none",
                    choices=["none", "no-body", "no-root", "no-hands", "no-face",
                             "no-arms", "no-chest"])
    ap.add_argument("--gain", type=float, default=1.0,
                    help="scale the torso residual (trade-off ablation)")
    ap.add_argument("--arm-gain", type=float, default=None,
                    help="separate scale for clavicle/upperarm/forearm")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--apply-root", action="store_true",
                    help="apply the learned root XZ residual (off by default: it "
                         "moves planted feet and fails the contact-slide gate)")
    ap.add_argument("--smooth", type=int, default=3,
                    help="temporal smoothing of the predicted correction (1=off)")
    ap.add_argument("--clips", default=None)
    ap.add_argument("--pair-root", default=None,
                    help="pair.npz root (default outputs/m35/data)")
    ap.add_argument("--root-contact-joint", action="store_true",
                    help="after applying the root residual, re-solve contact IK")
    args = ap.parse_args()

    net, ck = load_net(args.ckpt)
    clips = load_json(args.index)["clips"]
    keep = set(args.split.split(","))
    if args.clips:
        keep_c = set(args.clips.split(","))
        clips = [c for c in clips if c["clip"] in keep_c]
    else:
        clips = [c for c in clips if c["split"] in keep]
    out_dir = (Path(args.out) / args.tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, meta in enumerate(clips, 1):
        pr = Path(args.pair_root) if args.pair_root else ROOT / "outputs/m35/data"
        pair = np.load(pr / meta["clip"] / "pair.npz")
        res = predict(net, ck, pair["features"])
        if not args.apply_root:
            res[:, FD.ROOT_SLICE] = 0.0
        for i, bone in enumerate(FD.BODY_FACTOR_NAMES):
            is_arm = bone.startswith(("clavicle", "upperarm", "forearm"))
            g = (args.arm_gain if (is_arm and args.arm_gain is not None)
                 else args.gain)
            res[:, 2 + 3 * i:5 + 3 * i] *= float(g)
        if args.no_hands:
            res[:, FD.HAND_SLICE] = 0.0
        if args.no_face:
            res[:, FD.FACE_SLICE] = 0.0
        res = smooth_residual(res, pair["hand_level"], pair["face_level"],
                              win=args.smooth)
        if args.ablate == "no-body":
            res[:, FD.ROT_SLICE] = 0.0
        elif args.ablate == "no-root":
            res[:, FD.ROOT_SLICE] = 0.0
        elif args.ablate == "no-hands":
            res[:, FD.HAND_SLICE] = 0.0
        elif args.ablate == "no-face":
            res[:, FD.FACE_SLICE] = 0.0
        elif args.ablate in ("no-arms", "no-chest"):
            for i, bone in enumerate(FD.BODY_FACTOR_NAMES):
                is_arm = bone.startswith(("clavicle", "upperarm", "forearm"))
                drop = is_arm if args.ablate == "no-arms" else not is_arm
                if drop:
                    res[:, 2 + 3 * i:5 + 3 * i] = 0.0
        rec = score_clip(meta, res, out_dir, tag=args.tag, ck=ck,
                         pair_root=args.pair_root,
                         root_contact_joint=args.root_contact_joint)
        recs.append(rec)
        d = rec["delta"]
        print(f"[{i}/{len(clips)}] {meta['clip']:24s} "
              f"main {rec['baseline']['body_relative_main_deg']:.2f}->"
              f"{rec['corrected']['body_relative_main_deg']:.2f} "
              f"posΔ {d['position_pct_height']:+.3f} "
              f"recallΔ {(rec['corrected']['contact_recall'] or 0)-(rec['baseline']['contact_recall'] or 0):+.3f} "
              f"jerkΔ {d['jerk_rms_ratio']:+.2f}", flush=True)
    agg = aggregate(recs)
    gate = gates(agg, recs)
    report = {"ckpt": str(args.ckpt), "tag": args.tag, "ablate": args.ablate,
              "split": args.split, "clips": recs, "aggregate": agg, "gates": gate,
              "params": net.n_params(), "cfg": ck["cfg"], "epoch": ck.get("epoch")}
    (Path(args.out) / f"report_{args.tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nAGGREGATE")
    for split, a in agg.items():
        print(f"  {split:5s} n={a['clips']:2d} main "
              f"{a['baseline']['body_relative_main_deg']:.2f}->"
              f"{a['corrected']['body_relative_main_deg']:.2f}  "
              f"angle {a['baseline']['angle_mean_deg']:.2f}->{a['corrected']['angle_mean_deg']:.2f}  "
              f"pos {a['baseline']['position_pct_height']:.2f}->{a['corrected']['position_pct_height']:.2f}")
    print("GATES", json.dumps(gate, ensure_ascii=False))
    print("WROTE", Path(args.out) / f"report_{args.tag}.json")


if __name__ == "__main__":
    main()
