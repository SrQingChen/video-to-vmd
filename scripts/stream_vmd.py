"""Streaming / gap completion for a VMD (M2.5 v0, non-learned).

Examples:
  # offline refinement (final delivery): gaps re-planned from both ends
  python scripts/stream_vmd.py --vmd outputs/m2/vmd/front_clean_ik.vmd --model si \
      --confidence outputs/m25/front_conf.npy --mode offline \
      --out outputs/m25/front_stream.vmd

  # streaming preview: 0.4 s lookahead
  python scripts/stream_vmd.py --vmd ... --confidence ... --mode stream \
      --lookahead 0.4 --out ..._stream.vmd

The confidence file is either an ``.npy`` array ``(F,)`` / ``(F, J)`` or a
JSON list / dict.  Frame ``t`` is the VMD frame (0-based).  ``--heuristic``
derives a weak confidence from single-frame observation outliers when no
external evidence exists; it is not an occlusion detector.

The output is a normal dense IK-off VMD.  Frames with confidence >= ``--high``
are copied from the input unchanged, so the module never degrades the parts of
the clip that were actually observed.  ``--gt`` is only used to score the run.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.metrics.core import compute_metrics, pose_track
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd
from vtv.mmd.writer import write_vmd
from vtv.streaming import (StreamingConfig, fill_vmd, heuristic_confidence,
                           normalize_confidence, summarize)


def load_confidence(path):
    if path is None:
        return None
    if path.endswith(".npy"):
        return np.load(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "confidence" in data:
        data = data["confidence"]
    if isinstance(data, dict):
        return {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
    return np.asarray(data, dtype=np.float64)


def score(model, profile, gt_vmd, pred_vmd, contact_margin, frames):
    gt = pose_track(model, gt_vmd, profile, frames=frames)
    pr = pose_track(model, pred_vmd, profile, frames=frames)
    return compute_metrics(gt, pr, profile, contact_margin=contact_margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--vmd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--confidence", default=None,
                    help=".npy (F,) or (F,J), or JSON list/dict")
    ap.add_argument("--heuristic", action="store_true",
                    help="derive weak confidence from observation outliers")
    ap.add_argument("--mode", choices=("stream", "offline"), default="offline")
    ap.add_argument("--lookahead", type=float, default=0.4,
                    help="streaming lookahead seconds (0.3-0.5)")
    ap.add_argument("--max-rollout", type=float, default=1.5)
    ap.add_argument("--contact-hold", type=float, default=1.0)
    ap.add_argument("--high", type=float, default=0.75)
    ap.add_argument("--low", type=float, default=0.35)
    ap.add_argument("--no-contact", action="store_true",
                    help="disable stance continuation/damping")
    ap.add_argument("--no-smooth", action="store_true",
                    help="disable offline residual smoothing")
    ap.add_argument("--anchor-stance", action="store_true",
                    help="M2.5 v0.1: anchor observed stance feet inside long holes")
    ap.add_argument("--stance-release", type=float, default=0.4,
                    help="anchor release window at the end of a hole (s)")
    ap.add_argument("--stance-release-mode", choices=("budget", "full"),
                    default="budget",
                    help="'budget' keeps the toe within --stance-slide-budget "
                         "of the anchor at the last gap frame; 'full' releases "
                         "to the v0 bridge (ablation)")
    ap.add_argument("--stance-slide-budget", type=float, default=0.045,
                    help="allowed toe XZ net movement at the last gap frame")
    ap.add_argument("--stance-orientation", choices=("v0", "anchor"),
                    default="v0",
                    help="'v0' anchors the toe XZ and follows the v0 foot "
                         "orientation; 'anchor' also freezes the pre-gap foot "
                         "world orientation (literal spec, ablation)")
    ap.add_argument("--gt", default=None, help="score only; never read by the filler")
    ap.add_argument("--contact-margin", type=float, default=0.02)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cfg_model = json.load(open(args.config, encoding="utf-8"))["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg_model["pmx"])
    vmd = parse_vmd(args.vmd)
    cfg = StreamingConfig(
        high=args.high, low=args.low,
        lookahead_s=args.lookahead, max_rollout_s=args.max_rollout,
        contact_hold_s=args.contact_hold,
        smooth_gaps=not args.no_smooth,
        anchor_stance=args.anchor_stance)
    stance = None
    if args.anchor_stance:
        from vtv.stance_ik import StanceConfig
        stance = StanceConfig(
            high=args.high, low=args.low, fps=cfg.fps,
            release_s=args.stance_release,
            release_mode=args.stance_release_mode,
            slide_budget=args.stance_slide_budget,
            anchor_orientation=args.stance_orientation)

    conf = load_confidence(args.confidence)
    if conf is None and args.heuristic:
        from vtv.cleanup import dense_tracks
        _, tracks = dense_tracks(vmd)
        conf = heuristic_confidence(tracks)
        print("heuristic confidence: min/mean/max",
              float(np.min(conf)), float(np.mean(conf)), float(np.max(conf)))
    if conf is None:
        print("no confidence supplied: every frame is treated as observed")

    res = fill_vmd(
        vmd, conf, model=None if args.no_contact else model,
        profile=None if args.no_contact else profile,
        cfg=cfg, mode=args.mode,
        anchor_stance=args.anchor_stance, stance=stance)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_vmd(out, res["vmd"].get("model_name", ""), res["vmd"]["bones"],
              res["vmd"].get("morphs", {}), res["vmd"].get("ik_frames", []))
    report = {
        "in": os.path.abspath(args.vmd),
        "out": out,
        "model": args.model,
        "confidence": os.path.abspath(args.confidence) if args.confidence else None,
        "summary": summarize(res["report"]),
        "streaming": res["report"],
    }
    if args.gt:
        frames = list(range(vmd.get("max_frame", 0) + 1))
        before = score(model, profile, parse_vmd(args.gt), vmd,
                       args.contact_margin, frames)
        after = score(model, profile, parse_vmd(args.gt), res["vmd"],
                      args.contact_margin, frames)
        report["metrics"] = {
            "gt": os.path.abspath(args.gt),
            "before": {"vector": before["vector"], "groups": before["groups"]},
            "after": {"vector": after["vector"], "groups": after["groups"]},
            "delta": {k: round(after["vector"][k] - before["vector"][k], 4)
                      for k in before["vector"]
                      if isinstance(before["vector"][k], (int, float))
                      and isinstance(after["vector"].get(k), (int, float))},
        }
        keys = ("angle_mean_deg", "angle_p99_deg", "position_pct_height",
                "contact_recall", "contact_precision", "contact_slide_delta",
                "contact_penetration_delta", "jerk_rms_ratio")
        print("before:", {k: before["vector"][k] for k in keys})
        print("after :", {k: after["vector"][k] for k in keys})
    rp = args.report or (out + ".streaming.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("summary:", json.dumps(report["summary"], ensure_ascii=False))
    print("WROTE", out)
    print("WROTE", rp)


if __name__ == "__main__":
    main()
