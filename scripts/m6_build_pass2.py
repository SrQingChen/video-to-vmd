"""M6 closed-loop pass 2: apply pass 1, recompute evidence, rebuild targets.

For every training clip:
  1. pass-1 features -> pass-1 net -> bounded residual -> policy (gains,
     L0/L1 gating, smoothing);
  2. apply the residual to the body VMD and the hand/face tracks;
  3. rebuild features from the *corrected* pose (the 2D/contact/hand/face
     evidence is unchanged; the pose and the baseline hand/face are new);
  4. target = GT - corrected, with the same v2 bounds and masks.

The saved pairs train a second residual net.  At inference the same two steps
are applied in order; this is the "apply -> recompute evidence -> predict
again" loop, not a recurrent controller.
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
from vtv import feedback_data as FD  # noqa: E402
from vtv.eval_helpers import load_net, predict  # noqa: E402
from vtv.metrics.core import pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fit(a, n):
    out = np.zeros((n,) + np.asarray(a).shape[1:], np.asarray(a).dtype)
    m = min(n, len(a))
    out[:m] = np.asarray(a)[:m]
    return out


def evidence_from_meta(meta, n):
    gv = np.load(ROOT / meta["paths"]["gvhmr"])
    gmeta = load_json((ROOT / meta["paths"]["gvhmr"]).parent / "gvhmr_meta.json")
    hand = np.load(ROOT / meta["paths"]["hand"])
    face = np.load(ROOT / meta["paths"]["face"])
    body_p = ROOT / meta["paths"]["body"]
    ameta = load_json(body_p.parent / "adapter.vmd.json")
    return {
        "kp2d": fit(gv["kp2d"], n), "bbx_xys": fit(gv["bbx_xys"], n),
        "img_wh": (int(gmeta["width"]), int(gmeta["height"])),
        "adapter_scale": float(ameta.get("scale", 1.0)),
        "hand": {"angles": fit(hand["angles"], n), "level": fit(hand["level"], n),
                 "score": fit(hand["score"], n),
                 "detected": fit(hand["detected"], n)},
        "face": {"weights": fit(face["weights"], n), "level": fit(face["level"], n),
                 "detected": fit(face["detected"], n)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / "outputs/m6/data/index.json"))
    ap.add_argument("--pair-root", default=str(ROOT / "outputs/m6/data"))
    ap.add_argument("--out", default=str(ROOT / "outputs/m6/data_pass2"))
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/m6/net/m6_v2_best.pt"))
    ap.add_argument("--mode", default="default")
    ap.add_argument("--overrides", default=None,
                    help="JSON dict of policy overrides (e.g. hands/face on)")
    ap.add_argument("--splits", default="train")
    ap.add_argument("--clips", default=None)
    args = ap.parse_args()
    net, ck = load_net(args.ckpt)
    cfg = FA.resolve_config(args.mode, overrides=(
        json.loads(args.overrides) if args.overrides else None))
    clips = load_json(args.index)["clips"]
    keep = set(args.splits.split(","))
    if args.clips:
        wanted = set(args.clips.split(","))
        clips = [c for c in clips if c["clip"] in wanted]
    else:
        clips = [c for c in clips if c["split"] in keep]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, meta in enumerate(clips, 1):
        cid = meta["clip"]
        pair = np.load(Path(args.pair_root) / cid / "pair.npz")
        n = len(pair["features"])
        cfg_m = load_json(ROOT / "configs/models.json")["models"][meta["model"]]
        profile = load_json(ROOT / f"profiles/{meta['model']}.json")
        model = parse_pmx(cfg_m["pmx"])
        body = parse_vmd(str(ROOT / meta["paths"]["body"]))
        gt = parse_vmd(str(ROOT / meta["paths"]["gt"]))
        n = min(n, int(body["max_frame"]) + 1, int(gt["max_frame"]) + 1)
        frames = list(range(n))
        ev = evidence_from_meta(meta, n)
        tr = pose_track(model, body, profile, frames=frames)
        feats1 = FD.build_features(
            tr, float(meta["height"]), kp2d=ev["kp2d"], bbx_xys=ev["bbx_xys"],
            hand_angles=ev["hand"]["angles"], hand_level=ev["hand"]["level"],
            hand_score=ev["hand"]["score"], hand_detect=ev["hand"]["detected"],
            face_weights=ev["face"]["weights"], face_level=ev["face"]["level"],
            face_detect=ev["face"]["detected"], profile=profile,
            adapter_scale=ev["adapter_scale"], img_wh=ev["img_wh"],
            base_hand_angles=ev["hand"]["angles"],
            base_face_weights=ev["face"]["weights"],
            feature_set=ck.get("feature_set", "v1"))
        raw1 = predict(net, ck, feats1)
        res1 = FA.shape_residual(raw1, ev["hand"]["level"],
                                 ev["face"]["level"], profile, ck, cfg)
        body1 = FD.apply_body_residual(body, model, profile, res1)
        ha1 = (FD.apply_hand_residual(ev["hand"]["angles"], res1[:, FD.HAND_SLICE])
               if cfg.get("apply_hands") else ev["hand"]["angles"])
        fw1 = (FD.apply_face_residual(ev["face"]["weights"], res1[:, FD.FACE_SLICE])
               if cfg.get("apply_face") else ev["face"]["weights"])
        tr1 = pose_track(model, body1, profile, frames=frames)
        gt_t = pose_track(model, gt, profile, frames=frames)
        gt_npz = np.load(ROOT / f"outputs/m35/gt/{meta['motion']}/{meta['model']}.gt.npz")
        gt_hand = fit(gt_npz["hand_angles"], n)
        gt_face = fit(gt_npz["face_weights"], n)
        gt_chan = np.asarray(gt_npz["face_mask"], np.float32)
        feats2 = FD.build_features(
            tr1, float(meta["height"]), kp2d=ev["kp2d"], bbx_xys=ev["bbx_xys"],
            hand_angles=ev["hand"]["angles"], hand_level=ev["hand"]["level"],
            hand_score=ev["hand"]["score"], hand_detect=ev["hand"]["detected"],
            face_weights=ev["face"]["weights"], face_level=ev["face"]["level"],
            face_detect=ev["face"]["detected"], profile=profile,
            adapter_scale=ev["adapter_scale"], img_wh=ev["img_wh"],
            base_hand_angles=ha1, base_face_weights=fw1,
            feature_set=ck.get("feature_set", "v1"))
        body_res = FD.extract_body_residual(
            gt_t, tr1, bounds=FD.ROT_BOUND_V2 if ck.get("bounds") == "v2" else None)
        hand_res = FD.extract_hand_residual(gt_hand, ha1, ev["hand"]["level"])
        face_res = FD.extract_face_residual(gt_face, fw1, ev["face"]["level"],
                                            gt_channels=gt_chan)
        # Same masking rules as pass 1.
        if meta["variant"] in ("occ", "mix", "lowres", "blur"):
            hand_res["mask"][:] = 0.0
            face_res["mask"][:] = 0.0
        elif meta["variant"] == "handgt":
            face_res["mask"][:] = 0.0
        elif meta["variant"] == "facegt":
            hand_res["mask"][:] = 0.0
        target = FD.combine_residuals(body_res["target"], hand_res["target"],
                                      face_res["target"])
        mask = FD.combine_residuals(body_res["mask"], hand_res["mask"],
                                    face_res["mask"])
        out = out_root / cid
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "pair.npz", features=feats2.astype(np.float32),
            target=target.astype(np.float32), mask=mask.astype(np.float32),
            frames=np.arange(n, dtype=np.int32),
            kp2d=ev["kp2d"][:n].astype(np.float32),
            bbx_xys=ev["bbx_xys"][:n].astype(np.float32),
            pred_hand=ha1.astype(np.float32),
            hand_level=ev["hand"]["level"][:n].astype(np.int8),
            gt_hand=gt_hand.astype(np.float32),
            pred_face=fw1.astype(np.float32),
            face_level=ev["face"]["level"][:n].astype(np.int8),
            gt_face=gt_face.astype(np.float32),
            face_mask=gt_chan.astype(np.float32),
        )
        m = dict(meta)
        m.update({"pass": 2, "frames": n, "feature_dim": int(feats2.shape[1]),
                  "residual_dim": int(target.shape[1]),
                  "residual_mean_abs_pass1": {
                      k: float(np.abs(res1[:, sl]).mean())
                      for k, sl in FD.residual_groups().items()},
                  "body_target_mean_abs": {
                      k: float(np.abs(target[:, sl]).mean())
                      for k, sl in FD.residual_groups().items()}})
        (out / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        recs.append(m)
        print(f"[{i}/{len(clips)}] {cid} n={n} "
              f"pass1 body {m['residual_mean_abs_pass1']['body']:.2f} "
              f"pass2 body {m['body_target_mean_abs']['body']:.2f}", flush=True)
    (out_root / "index.json").write_text(json.dumps(
        {"clips": recs, "leakage": FD.leakage_report(recs)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("WROTE", out_root / "index.json")


if __name__ == "__main__":
    main()
