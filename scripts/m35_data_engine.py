"""M3.5 data engine: rendered clip -> GVHMR -> VMD pipeline -> paired data.

Stages (all resumable, select with --stage):

  gvhmr     run GVHMR on every rendered (or augmented) clip
  pipeline  adapter -> clean_vmd -> contact IK -> run_hand/run_face -> assemble
  pairs     evidence features + bounded residual teachers -> pair.npz

Inference evidence (features) never reads GT.  GT is read only in ``pairs`` to
build the residual targets and masks.

Clip ids: ``<motion>_<model>_<view>`` for base clips and
``<motion>_<model>_<view>__<variant>`` for augmented ones.  Augmented samples
inherit the base split, supervise body/root only (hand/face masks are zeroed),
and reuse the base clip's hand/face evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.cleanup import clean_vmd  # noqa: E402
from vtv.contact_ik import solve_contact_ik  # noqa: E402
from vtv.metrics.core import pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402

PY = sys.executable
MOTIONS = ("R1", "R2", "R3")
MODELS = ("si", "luoxi", "tiflos")
VIEWS = ("front", "three_quarter", "side")
AUG_VARIANTS = ("occ", "mix")
# M6: observation-only degradations, a training-only back view, and two
# evidence variants that keep the base body but replace hand/face evidence
# (synthetic GT hands, medium-shot face).  They are train-only.
M6_AUG_VARIANTS = ("lowres", "blur")
M6_SPECIAL_VARIANTS = ("handgt", "facegt")
M6_EXTRA_VIEWS = ("front_left",)
BOUNDS = "v1"
FEATURE_SET = "v1"
PAIR_ROOT = None


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def all_clips():
    out = []
    for m in MOTIONS:
        for mo in MODELS:
            for v in VIEWS:
                out.append(f"{m}_{mo}_{v}")
    for m in FD.TRAIN_MOTIONS:
        for mo in FD.TRAIN_MODELS:
            for v in M6_EXTRA_VIEWS:
                out.append(f"{m}_{mo}_{v}")
            for v in FD.TRAIN_VIEWS:
                for a in AUG_VARIANTS + M6_AUG_VARIANTS + M6_SPECIAL_VARIANTS:
                    out.append(f"{m}_{mo}_{v}__{a}")
    return out


def parse_clip(cid):
    if "__" in cid:
        base, variant = cid.split("__", 1)
    else:
        base, variant = cid, "base"
    # view names can contain an underscore ("three_quarter"), so split only
    # the motion and model and keep the rest as the view
    parts = base.split("_")
    motion, model = parts[0], parts[1]
    view = "_".join(parts[2:])
    split = FD.assign_split(motion, model, view)
    if split is None:
        raise SystemExit(f"clip {cid} has no split")
    if variant != "base":
        if split != "train":
            raise SystemExit(f"augmented clip {cid} is not train")
        split = "train"
    return {"clip": cid, "base": base, "motion": motion, "model": model,
            "view": view, "variant": variant, "split": split}


def paths(cid, kind):
    base = cid.split("__")[0]
    variant = cid.split("__", 1)[1] if "__" in cid else "base"
    if kind == "render":
        if variant == "facecam":
            return ROOT / "outputs/m35/facecam" / base / f"{base}.mp4"
        if "__" in cid:
            return ROOT / "outputs/m35/aug" / f"{cid}.mp4"
        return ROOT / "outputs/m35/render" / base / f"{base}.mp4"
    if kind in ("gvhmr", "pred"):
        # handgt/facecam only replace evidence; the body observation is the
        # base clip (same frames, same kp2d).
        target = base if variant in M6_SPECIAL_VARIANTS else cid
        return ROOT / f"outputs/m35/{kind}" / target
    if kind == "pair":
        return Path(PAIR_ROOT) / cid if PAIR_ROOT else (
            ROOT / "outputs/m35/data" / cid)
    raise KeyError(kind)


def video_size(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return w, h, n


def load_evidence(cid):
    g = paths(cid, "gvhmr")
    npz_p = g / "gvhmr_result.npz"
    if not npz_p.exists():
        raise SystemExit(f"missing {npz_p}")
    d = np.load(npz_p)
    if "kp2d" in d:
        kp = d["kp2d"]
    else:
        import torch
        kp = torch.load(g / "preprocess/vitpose.pt", map_location="cpu")
        kp = kp.cpu().numpy() if hasattr(kp, "cpu") else np.asarray(kp)
    bbx = d["bbx_xys"]
    meta = load_json(g / "gvhmr_meta.json")
    return np.asarray(kp, np.float32), np.asarray(bbx, np.float32), (
        int(meta["width"]), int(meta["height"]))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_gvhmr(cid):
    info = parse_clip(cid)
    if info["variant"] in M6_SPECIAL_VARIANTS:
        return "skipped"
    out = paths(cid, "gvhmr")
    if (out / "gvhmr_result.npz").exists():
        return "cached"
    video = paths(cid, "render")
    if not video.exists():
        raise SystemExit(f"missing video {video}")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(ROOT / "scripts/run_gvhmr.py"), "--video", str(video),
           "--static_cam", "--output", str(out)]
    log = out / "gvhmr.log"
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    if r.returncode != 0 or not (out / "gvhmr_result.npz").exists():
        raise SystemExit(f"GVHMR failed for {cid}: see {log}")
    return "ok"


def _write_body(vmd, p):
    write_vmd(str(p), vmd.get("model_name", ""), vmd["bones"],
              vmd.get("morphs", {}), vmd.get("ik_frames", []))


def stage_pipeline(cid):
    info = parse_clip(cid)
    if info["variant"] in M6_SPECIAL_VARIANTS:
        return "skipped"
    cfg = load_json(ROOT / "configs/models.json")["models"][info["model"]]
    profile = load_json(ROOT / f"profiles/{info['model']}.json")
    model = parse_pmx(cfg["pmx"])
    g = paths(cid, "gvhmr")
    pred = paths(cid, "pred")
    pred.mkdir(parents=True, exist_ok=True)
    adapter_p = pred / "adapter.vmd"
    clean_p = pred / "clean.vmd"
    ik_p = pred / "ik.vmd"
    done_p = pred / "pipeline.json"
    if done_p.exists() and ik_p.exists():
        return "cached"

    if not adapter_p.exists():
        cmd = [PY, str(ROOT / "scripts/run_adapter.py"),
               "--npz", str(g / "gvhmr_result.npz"), "--model", info["model"],
               "--out", str(adapter_p)]
        log = pred / "adapter.log"
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        if r.returncode != 0 or not adapter_p.exists():
            raise SystemExit(f"adapter failed for {cid}: {log}")
    adapter_meta = load_json(pred / "adapter.vmd.json")
    scale = float(adapter_meta.get("scale", 1.0))

    if not clean_p.exists():
        res = clean_vmd(parse_vmd(str(adapter_p)), model, profile)
        _write_body(res["vmd"], clean_p)
        (pred / "clean.json").write_text(json.dumps(res["report"], indent=1),
                                         encoding="utf-8")
    if not ik_p.exists():
        kp, _, _ = load_evidence(cid)
        res = solve_contact_ik(parse_vmd(str(clean_p)), model, profile, kp2d=kp)
        _write_body(res["vmd"], ik_p)
        (pred / "ik.json").write_text(json.dumps(res["report"], indent=1,
                                                 default=float), encoding="utf-8")

    n_frames = int(parse_vmd(str(ik_p))["max_frame"]) + 1
    base = info["base"]
    hand_dir = paths(base, "pred") / "hand"
    face_dir = paths(base, "pred") / "face"
    if not (hand_dir / "hand.npz").exists():
        cmd = [PY, str(ROOT / "scripts/run_hand.py"),
               "--video", str(paths(base, "render")),
               "--kp", str(paths(base, "gvhmr") / "preprocess/vitpose.pt"),
               "--out", str(hand_dir), "--overlays", "0"]
        log = pred / "hand.log"
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        if r.returncode != 0:
            raise SystemExit(f"run_hand failed for {base}: {log}")
    if not (face_dir / "face.npz").exists():
        cmd = [PY, str(ROOT / "scripts/run_face.py"),
               "--video", str(paths(base, "render")),
               "--kp", str(paths(base, "gvhmr") / "preprocess/vitpose.pt"),
               "--out", str(face_dir)]
        log = pred / "face.log"
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        if r.returncode != 0:
            raise SystemExit(f"run_face failed for {base}: {log}")

    hand = np.load(hand_dir / "hand.npz")
    face = np.load(face_dir / "face.npz")
    ha = _fit_len(hand["angles"], n_frames)
    fw = _fit_len(face["weights"], n_frames)
    body = parse_vmd(str(ik_p))
    vmd = FD.assemble_vmd(body, profile, ha, fw, n_frames=n_frames)
    write_vmd(str(pred / "pred_before.vmd"), vmd.get("model_name", ""),
              vmd["bones"], vmd["morphs"], vmd["ik_frames"])

    out = parse_vmd(str(pred / "pred_before.vmd"))
    cert = {
        "clip": cid, "base": base, "model": info["model"], "frames": n_frames,
        "bone_tracks": len(out["bones"]),
        "finger_tracks": sum(1 for k in profile["finger_map"].values()
                             if k in out["bones"]),
        "morph_tracks": sorted(out["morphs"]),
        "ik_off": bool(out["ik_frames"]) and all(
            not any(fr["ik"].values()) for fr in out["ik_frames"]),
        "quat_norm_max_err": max(abs(math.sqrt(sum(c * c for c in k["rot"])) - 1.0)
                                 for ks in out["bones"].values() for k in ks),
        "adapter_scale": scale,
        "hand_levels": {s: int((hand["level"][:n_frames, i] <= 1).sum())
                        for i, s in enumerate(("l", "r"))},
        "face_L01": int((face["level"][:n_frames] <= 1).sum()),
        "morphs_in_profile": all(nm in {m["name"] for m in profile["morphs"]}
                                 for nm in out["morphs"]),
    }
    cert["ok"] = (cert["bone_tracks"] == 53 and cert["finger_tracks"] == 30
                  and cert["ik_off"] and cert["quat_norm_max_err"] < 1e-6
                  and cert["morphs_in_profile"])
    (pred / "pipeline.json").write_text(json.dumps(cert, indent=1), encoding="utf-8")
    if not cert["ok"]:
        raise SystemExit(f"pipeline certificate failed for {cid}: {cert}")
    return "ok"


def _fit_len(a, n):
    out = np.zeros((n,) + a.shape[1:], np.float32)
    m = min(n, len(a))
    out[:m] = a[:m]
    return out


def stage_pairs(cid):
    info = parse_clip(cid)
    pred = paths(cid, "pred")
    pair_dir = paths(cid, "pair")
    pair_dir.mkdir(parents=True, exist_ok=True)
    npz_p = pair_dir / "pair.npz"
    meta_p = pair_dir / "meta.json"
    if npz_p.exists() and meta_p.exists():
        return "cached"
    cfg = load_json(ROOT / "configs/models.json")["models"][info["model"]]
    profile = load_json(ROOT / f"profiles/{info['model']}.json")
    model = parse_pmx(cfg["pmx"])
    height = float(profile["height"])
    gt = parse_vmd(str(ROOT / f"outputs/m35/gt/{info['motion']}/{info['model']}.vmd"))
    gt_npz = np.load(ROOT / f"outputs/m35/gt/{info['motion']}/{info['model']}.gt.npz")
    body = parse_vmd(str(pred / "ik.vmd"))
    n = int(body["max_frame"]) + 1
    frames = list(range(n))
    # Local track values are unaffected by the parent twist helper, so the
    # teacher is evaluated on the full GT; the evaluation additionally reports
    # a twist-normalised diagnostic (body_relative_strip_deg).
    gt_track = pose_track(model, gt, profile, frames=frames)
    pred_track = pose_track(model, body, profile, frames=frames)

    base = info["base"]
    base_pred = paths(base, "pred")
    hand_p = base_pred / "hand/hand.npz"
    face_p = base_pred / "face/face.npz"
    if info["variant"] == "handgt":
        hand_p = ROOT / "outputs/m6/handgt" / base / "hand.npz"
    elif info["variant"] == "facegt":
        face_p = ROOT / "outputs/m6/facegt" / base / "face.npz"
    if not hand_p.exists() or not face_p.exists():
        raise SystemExit(f"missing evidence for {cid}: {hand_p} / {face_p}")
    hand = np.load(hand_p)
    face = np.load(face_p)
    # The evidence the network *sees* can be synthetic (handgt/facegt); the
    # residual target must still be GT minus the real baseline pose that the
    # production pipeline starts from.  Otherwise the hand head is trained to
    # predict ~0 and does nothing at inference.
    base_hand = np.load(base_pred / "hand/hand.npz")
    base_face = np.load(base_pred / "face/face.npz")
    ha = _fit_len(hand["angles"], n)
    hl = _fit_len(hand["level"], n)
    hs = _fit_len(hand["score"], n)
    hd = _fit_len(hand["detected"], n)
    fw = _fit_len(face["weights"], n)
    fl = _fit_len(face["level"], n)
    fd = _fit_len(face["detected"], n)
    ha_base = _fit_len(base_hand["angles"], n)
    fw_base = _fit_len(base_face["weights"], n)

    bounds = FD.ROT_BOUND_V2 if BOUNDS == "v2" else None
    body_res = FD.extract_body_residual(gt_track, pred_track, bounds=bounds)
    gt_hand = _fit_len(gt_npz["hand_angles"], n)
    gt_face = _fit_len(gt_npz["face_weights"], n)
    gt_chan = np.asarray(gt_npz["face_mask"], np.float32)
    hand_base_ref = ha_base if info["variant"] == "handgt" else ha
    face_base_ref = fw_base if info["variant"] == "facegt" else fw
    hand_res = FD.extract_hand_residual(gt_hand, hand_base_ref, hl)
    face_res = FD.extract_face_residual(gt_face, face_base_ref, fl,
                                        gt_channels=gt_chan)
    # Observation-degradation variants keep the base hand/face evidence and
    # supervise the body only.  handgt/facecam deliberately replace one
    # evidence stream, so only the *other* stream is masked.
    if info["variant"] in AUG_VARIANTS + M6_AUG_VARIANTS:
        hand_res["mask"][:] = 0.0
        face_res["mask"][:] = 0.0
    elif info["variant"] == "handgt":
        face_res["mask"][:] = 0.0
    elif info["variant"] == "facegt":
        hand_res["mask"][:] = 0.0
    target = FD.combine_residuals(body_res["target"], hand_res["target"],
                                  face_res["target"])
    mask = FD.combine_residuals(body_res["mask"], hand_res["mask"],
                                face_res["mask"])

    kp, bbx, wh = load_evidence(cid)
    adapter_meta = load_json(pred / "adapter.vmd.json")
    feats = FD.build_features(
        pred_track, height, kp2d=kp[:n], bbx_xys=bbx[:n],
        hand_angles=ha, hand_level=hl, hand_score=hs, hand_detect=hd,
        face_weights=fw, face_level=fl, face_detect=fd,
        profile=profile, adapter_scale=float(adapter_meta.get("scale", 1.0)),
        img_wh=wh, base_hand_angles=ha_base, base_face_weights=fw_base,
        feature_set=FEATURE_SET)
    names = FD.feature_names(profile, FEATURE_SET)
    if feats.shape[1] != len(names):
        raise SystemExit(f"feature dim {feats.shape[1]} != names {len(names)}")

    np.savez_compressed(
        npz_p,
        features=feats.astype(np.float32),
        target=target.astype(np.float32),
        mask=mask.astype(np.float32),
        frames=np.asarray(frames, np.int32),
        kp2d=kp[:n].astype(np.float32),
        bbx_xys=bbx[:n].astype(np.float32),
        pred_hand=ha_base.astype(np.float32),
        hand_level=hl.astype(np.int8),
        gt_hand=gt_hand.astype(np.float32),
        pred_face=fw_base.astype(np.float32),
        face_level=fl.astype(np.int8),
        gt_face=gt_face.astype(np.float32),
        face_mask=gt_chan.astype(np.float32),
    )
    meta = dict(info)
    meta.update({
        "frames": n, "height": height,
        "feature_dim": int(feats.shape[1]),
        "residual_dim": int(target.shape[1]),
        "feature_names": names,
        "body_target_mean_abs": {k: float(np.abs(target[:, sl]).mean())
                                 for k, sl in FD.residual_groups().items()},
        "mask_fraction": {k: float(mask[:, sl].mean())
                          for k, sl in FD.residual_groups().items()},
        "paths": {"gt": f"outputs/m35/gt/{info['motion']}/{info['model']}.vmd",
                  "body": str((pred / "ik.vmd").relative_to(ROOT)),
                  "pred_before": str((pred / "pred_before.vmd").relative_to(ROOT)),
                  "hand": str(hand_p.relative_to(ROOT)),
                  "face": str(face_p.relative_to(ROOT)),
                  "gvhmr": str((paths(cid, "gvhmr") / "gvhmr_result.npz").relative_to(ROOT))},
        "split": info["split"],
    })
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    return "ok"


def main():
    global BOUNDS, FEATURE_SET, PAIR_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["gvhmr", "pipeline", "pairs", "all"])
    ap.add_argument("--clips", default=None)
    ap.add_argument("--bounds", default="v1", choices=["v1", "v2"],
                    help="v2 widens the shoulder-chain teacher bounds (M6)")
    ap.add_argument("--feature-set", default="v1", choices=["v1", "v2"],
                    help="v2 appends the baseline hand/face pose to the features")
    ap.add_argument("--pair-root", default=None,
                    help="write pairs here (default outputs/m35/data); use "
                         "outputs/m6/data to keep M3.5 pairs untouched")
    ap.add_argument("--index-out", default=str(ROOT / "outputs/m35/data/index.json"))
    ap.add_argument("--merge-index", default=None,
                    help="existing index.json whose clips are merged into --index-out")
    args = ap.parse_args()
    BOUNDS = args.bounds
    FEATURE_SET = args.feature_set
    PAIR_ROOT = Path(args.pair_root) if args.pair_root else None
    clips = args.clips.split(",") if args.clips else all_clips()
    records = []
    for i, cid in enumerate(clips, 1):
        info = parse_clip(cid)
        records.append(info)
        print(f"[{i}/{len(clips)}] {cid} ({info['split']})", flush=True)
        special = info["variant"] in M6_SPECIAL_VARIANTS
        if args.stage in ("gvhmr", "all") and not special:
            print("  gvhmr:", stage_gvhmr(cid), flush=True)
        if args.stage in ("pipeline", "all") and not special:
            print("  pipeline:", stage_pipeline(cid), flush=True)
        if args.stage in ("pairs", "all"):
            print("  pairs:", stage_pairs(cid), flush=True)
    if args.stage in ("pairs", "all"):
        out = Path(args.index_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        by_clip = {}
        if args.merge_index and Path(args.merge_index).exists():
            for m in load_json(args.merge_index).get("clips", []):
                by_clip[m["clip"]] = m
        for r in records:
            mp = paths(r["clip"], "pair") / "meta.json"
            if mp.exists():
                by_clip[r["clip"]] = load_json(mp)
        idx = list(by_clip.values())
        leak = FD.leakage_report(idx)
        out.write_text(json.dumps(
            {"clips": idx, "leakage": leak}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print("LEAKAGE", json.dumps(leak, ensure_ascii=False))
        if not leak["ok"]:
            raise SystemExit("split leakage audit failed")


if __name__ == "__main__":
    main()
