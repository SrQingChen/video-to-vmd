"""Build M3.5 GT VMDs: body + fingers + face morphs for every motion x model.

Sources
  * si body GT: the safe baked VMDs (R3/R2 exist; R1 is baked here on first use)
  * luoxi/tiflos body GT: the M1 adapter from the si safe GT (VMD -> VMD)
  * fingers: remapped from the si GT in hand-anatomical coordinates
  * face: raw motion VMD morphs -> 12 semantic channels -> the model's morph name

Outputs ``outputs/m35/gt/<motion>/<model>.vmd`` plus ``.gt.npz`` (hand angles and
semantic face weights used only as teachers) and a manifest with certificates.

Usage:
  python scripts/m35_build_gt.py --motions R1,R2,R3 --models si,luoxi,tiflos
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.adapter import adapt, canonical_from_pmx  # noqa: E402
from vtv.face import compress_morph_keys  # noqa: E402
from vtv.hands import HandRig  # noqa: E402
from vtv.metrics.core import pose_track  # noqa: E402
from vtv.mmd.bake import bake_evaluated_vmd  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402

RAW = {
    "R1": "testdata/R1_妄想天使/动作.vmd",
    "R2": "testdata/R2_王牌竞速/动作＋表情.vmd",
    "R3": "testdata/R3_花火主播/火花带表情.vmd",
}
SAFE = {
    "R1": "outputs/gt/si_R1_safe.vmd",
    "R2": "outputs/gt/si_R2_safe.vmd",
    "R3": "outputs/gt/si_R3_safe.vmd",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_safe(model, profile, motion, raw_path):
    out = ROOT / SAFE[motion]
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    vmd = parse_vmd(str(ROOT / raw_path))
    res = bake_evaluated_vmd(model, vmd, profile, str(out), solve_ik=False,
                             apply_all_translations=False)
    print(f"BAKED {motion} -> {out}  tracks={res['bones']} frames={res['frames']}")
    return out


def model_body_tracks(model_key, motion, src_vmd, src_model, src_profile,
                      tgt_model, tgt_profile):
    if model_key == "si":
        return {k: [dict(x) for x in v] for k, v in src_vmd["bones"].items()}, {
            "source": "si safe GT (all tracks copied)", "scale": 1.0,
            "ik_frames": [dict(x) for x in src_vmd.get("ik_frames", [])]}
    motion_c = canonical_from_pmx(src_model, src_profile, src_vmd)
    res = adapt(motion_c, tgt_model, tgt_profile, ground_calibrate=True,
                stage_yaw_deg=0.0)
    return res["bone_frames"], {
        "source": "adapter from si safe GT", "scale": float(res["scale"]),
        "ik_frames": res["ik_frames"],
        "ground_shift_units": float(res["ground_shift_units"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", default="R1,R2,R3")
    ap.add_argument("--models", default="si,luoxi,tiflos")
    ap.add_argument("--out", default=str(ROOT / "outputs/m35/gt"))
    args = ap.parse_args()

    cfg = load_json(ROOT / "configs/models.json")["models"]
    profiles = {m: load_json(ROOT / f"profiles/{m}.json")
                for m in ("si", "luoxi", "tiflos")}
    models = {m: parse_pmx(cfg[m]["pmx"]) for m in profiles}
    src_model = models["si"]
    src_profile = profiles["si"]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"motions": {}, "models": args.models.split(","),
                "layout": {"body": "source/adapter", "fingers": "anatomical remap",
                           "face": "raw VMD morphs -> semantics -> model morph"}}
    prev_p = out_root / "manifest.json"
    if prev_p.exists():
        prev = load_json(prev_p)
        if prev.get("motions"):
            manifest["motions"] = prev["motions"]
            manifest["models"] = sorted(set(prev.get("models", []))
                                        | set(manifest["models"]))

    for motion in args.motions.split(","):
        raw_path = RAW[motion]
        safe_path = ensure_safe(src_model, src_profile, motion, raw_path)
        src_vmd = parse_vmd(str(safe_path))
        raw_vmd = parse_vmd(str(ROOT / raw_path))
        n = int(src_vmd["max_frame"]) + 1
        if int(raw_vmd["max_frame"]) + 1 < n:
            n = int(raw_vmd["max_frame"]) + 1
        frames = list(range(n))
        mdir = out_root / motion
        mdir.mkdir(parents=True, exist_ok=True)
        manifest["motions"][motion] = {"frames": n, "raw": raw_path,
                                       "safe": str(safe_path.relative_to(ROOT)),
                                       "models": {}}
        src_rigs = {s: HandRig.from_profile(src_profile, s) for s in ("l", "r")}

        for model_key in args.models.split(","):
            profile = profiles[model_key]
            model = models[model_key]
            body, body_meta = model_body_tracks(
                model_key, motion, src_vmd, src_model, src_profile,
                model, profile)
            fingers = FD.remap_fingers(
                src_model, src_profile, src_vmd, model, profile, frames=frames,
                src_rigs=src_rigs)
            # ---- face ---------------------------------------------------
            target_names = FD.face_target_names(profile)
            sem, sources = FD.gt_semantic_weights(raw_vmd, n, target_names)
            # write one morph track per target morph name
            morph_tracks = {}
            by_name = {}
            for si, s in enumerate(FD.SEMANTIC):
                if s not in target_names:
                    continue
                name = target_names[s]
                by_name.setdefault(name, np.zeros(n))
                by_name[name] = np.maximum(by_name[name], sem[:, si])
            for name, w in by_name.items():
                if float(w.max()) <= 1e-4:
                    continue
                keys = compress_morph_keys(w)
                if keys:
                    morph_tracks[name] = keys

            tracks = dict(body)
            tracks.update(fingers)
            dst = mdir / f"{model_key}.vmd"
            write_vmd(str(dst), src_vmd.get("model_name", ""), tracks,
                      morph_tracks, body_meta["ik_frames"])
            written = parse_vmd(str(dst))

            # ---- teachers ----------------------------------------------
            gt_track = pose_track(model, written, profile, frames=frames)
            hand_gt = FD.gt_hand_angles(model, profile, written, frames)
            face_mask = np.zeros(len(FD.SEMANTIC), np.float32)
            for si, s in enumerate(FD.SEMANTIC):
                if s in sources:
                    face_mask[si] = 1.0
            np.savez_compressed(
                mdir / f"{model_key}.gt.npz",
                hand_angles=hand_gt.astype(np.float32),
                face_weights=sem.astype(np.float32),
                face_mask=face_mask,
                frames=np.asarray(frames, np.int32),
                motion=motion, model=model_key)

            # ---- certificates ------------------------------------------
            finger_names = sorted({k for s in ("l", "r")
                                   for k in profile["finger_map"].values()
                                   if k in written["bones"]})
            # anatomical consistency: target GT hand angles vs source GT angles
            src_hand = FD.gt_hand_angles(src_model, src_profile, src_vmd, frames)
            ang_err = np.abs(src_hand - hand_gt)
            if model_key == "si":
                equal = True
                for name, keys in src_vmd["bones"].items():
                    if name not in written["bones"]:
                        equal = False
                        break
                    if len(keys) != len(written["bones"][name]):
                        equal = False
                        break
                body_meta["tracks_equal_to_source"] = bool(equal)
            rec = {
                "vmd": str(dst.relative_to(ROOT)),
                "frames": n,
                "tracks": len(written["bones"]),
                "finger_tracks": len(finger_names),
                "morph_tracks": sorted(written["morphs"]),
                "morph_sources": sources,
                "face_mask_channels": [FD.SEMANTIC[i] for i in range(len(FD.SEMANTIC))
                                       if face_mask[i] > 0],
                "hand_angle_vs_si_max_deg": round(float(ang_err.max()), 4),
                "hand_angle_vs_si_mean_deg": round(float(ang_err.mean()), 4),
                "body": body_meta,
                "height": float(profile.get("height", 0)),
            }
            manifest["motions"][motion]["models"][model_key] = rec
            print(f"{motion}/{model_key}: {rec['tracks']} tracks, "
                  f"{rec['finger_tracks']} fingers, morphs={rec['morph_tracks']}, "
                  f"hand_vs_si={rec['hand_angle_vs_si_max_deg']}deg")

    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("WROTE", out_root / "manifest.json")


if __name__ == "__main__":
    main()
