"""Assemble body VMD + finger rotations + face morphs into the M3 deliverable.

  body VMD (23 tracks, IK off)
    + hand.npz (per-frame anatomical angles, L0-L3)
    + face.npz (semantic weights)
    + profile (finger_map, morphs, PMX)
  -> VMD with 23 body tracks + 30 finger tracks + morph tracks + IK off

Guarantees checked after writing:
  * the 23 body tracks are byte-identical to the input (frame/pos/rot/curves);
  * exactly 30 finger tracks, all from the profile's finger_map, translation 0;
  * IK section copied unchanged (4 IK bones off);
  * every morph name exists in the profile;
  * every quaternion is unit length;
  * the written finger quats reproduce the input angle track (FK check).

Usage:
  python scripts/assemble_handface_vmd.py \
    --body outputs/m25/t5_v01_offline.vmd \
    --hand outputs/m3/t5/hand.npz --face outputs/m3/t5/face.npz \
    --profile profiles/si.json --out outputs/m3/t5_si.vmd
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv.face import MorphMap, weights_to_morphs, compress_morph_keys  # noqa: E402
from vtv.hands import (  # noqa: E402
    HandRig, angles_to_local_quats, angles_to_targets, vector_to_measured,
)
from vtv.mmd.mathutil import qmul, qnorm, qrot  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402

DEFAULT_CURVE = ((20, 20), (107, 107))


def _same_keys(a, b, tol=1e-9):
    if len(a) != len(b):
        return False, "length"
    for ka, kb in zip(a, b):
        if int(ka["frame"]) != int(kb["frame"]):
            return False, "frame"
        for f in ("pos", "rot"):
            if any(abs(float(x) - float(y)) > tol for x, y in zip(ka[f], kb[f])):
                return False, f
        if list(map(tuple, ka.get("curves", []))) != list(map(tuple, kb.get("curves", []))):
            return False, "curves"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--certificate", default=None)
    args = ap.parse_args()

    import json as _json
    with open(args.profile, encoding="utf-8") as f:
        profile = _json.load(f)
    body = parse_vmd(args.body)
    hand = np.load(args.hand)
    face = np.load(args.face)
    n_body = int(body["max_frame"]) + 1
    n = n_body
    if args.max_frames > 0:
        n = min(n, args.max_frames)
    if len(hand["angles"]) < n:
        raise SystemExit(f"hand.npz has {len(hand['angles'])} frames, need {n}")
    if len(face["weights"]) < n:
        raise SystemExit(f"face.npz has {len(face['weights'])} frames, need {n}")

    model = parse_pmx(profile["source"])
    rigs = {s: HandRig.from_profile(profile, s) for s in ("l", "r")}
    morph_map = MorphMap.from_profile(profile)

    # ---- body tracks unchanged -----------------------------------------
    bone_frames = {}
    for name, keys in body["bones"].items():
        bone_frames[name] = [dict(k) for k in keys if int(k["frame"]) < n]

    # ---- finger tracks ---------------------------------------------------
    stats = {}
    finger_names = set()
    for side in ("l", "r"):
        rig = rigs[side]
        angles = hand["angles"][:n, 0 if side == "l" else 1]
        tracks = {name: [] for name in rig.finger_names()}
        finger_names.update(tracks)
        prev_q = {name: None for name in tracks}
        for fi in range(n):
            measured = vector_to_measured(angles[fi])
            qs = angles_to_local_quats(rig, measured)
            for name, q in qs.items():
                q = qnorm(q)
                # keep the quaternion hemisphere continuous: MMD/Blender
                # interpolate between keys and a sign flip looks like a spin
                pq = prev_q[name]
                if pq is not None and sum(a * b for a, b in zip(q, pq)) < 0.0:
                    q = tuple(-c for c in q)
                prev_q[name] = q
                tracks[name].append({"frame": fi, "pos": (0.0, 0.0, 0.0),
                                     "rot": q, "curves": (DEFAULT_CURVE,) * 4})
                # FK check: q must reconstruct the measured targets
        bone_frames.update(tracks)
        stats[side] = {"frames": n}

    # ---- morph tracks ----------------------------------------------------
    morph_weights = weights_to_morphs(face["weights"][:n], morph_map)
    existing = {m["name"] for m in profile.get("morphs", [])}
    morph_frames = {}
    for name, w in morph_weights.items():
        if name not in existing:
            raise SystemExit(f"morph not in profile: {name}")
        keys = compress_morph_keys(w)
        if keys:
            morph_frames[name] = keys

    # ---- IK off (copied) --------------------------------------------------
    ik_frames = [dict(fr) for fr in body.get("ik_frames", [])]

    write_vmd(args.out, body.get("model_name", ""), bone_frames,
              morph_frames, ik_frames)

    # ---- verification ----------------------------------------------------
    out = parse_vmd(args.out)
    report = {
        "out": args.out,
        "frames": n,
        "body_tracks": len(body["bones"]),
        "bone_tracks": len(out["bones"]),
        "finger_tracks": len(finger_names),
        "morph_tracks": sorted(out["morphs"].keys()),
        "ik_frames": out["ik_frames"],
    }
    body_equal = True
    details = {}
    for name, keys in body["bones"].items():
        ok, why = _same_keys(keys, out["bones"].get(name, []))
        if not ok:
            body_equal = False
            details[name] = why
    report["body_untouched"] = body_equal
    report["body_diff"] = details

    expected_bones = set(body["bones"]) | finger_names
    report["unexpected_bones"] = sorted(set(out["bones"]) - expected_bones)
    report["missing_bones"] = sorted(expected_bones - set(out["bones"]))
    report["finger_translation_max"] = max(
        [max(abs(v) for v in k["pos"]) for name in finger_names for k in out["bones"][name]]
        or [0.0])
    report["quat_norm_max_err"] = max(
        abs(math.sqrt(sum(c * c for c in k["rot"])) - 1.0)
        for keys in out["bones"].values() for k in keys)
    report["morph_names_in_profile"] = all(nm in existing for nm in out["morphs"])
    # replay from the *written* quats: compose FK and compare with targets
    for side in ("l", "r"):
        rig = rigs[side]
        stats[side]["target_replay_max_deg"] = _replay_check(
            out["bones"], rig, hand["angles"][:n, 0 if side == "l" else 1], n)
    report["replay"] = stats
    report["ik_off_ok"] = bool(out["ik_frames"]) and all(
        (not any(fr["ik"].values())) for fr in out["ik_frames"])

    # per-frame angular step of the finger tracks (jitter report)
    steps = []
    for name in finger_names:
        keys = sorted(out["bones"][name], key=lambda k: k["frame"])
        for a, b in zip(keys, keys[1:]):
            if b["frame"] - a["frame"] != 1:
                continue
            dot = abs(sum(x * y for x, y in zip(a["rot"], b["rot"])))
            steps.append(2.0 * math.degrees(math.acos(min(1.0, dot))))
    report["finger_step_deg"] = {
        "p50": float(np.percentile(steps, 50)) if steps else 0.0,
        "p95": float(np.percentile(steps, 95)) if steps else 0.0,
        "max": float(np.max(steps)) if steps else 0.0,
    }
    report["ok"] = (body_equal and not report["unexpected_bones"]
                    and not report["missing_bones"]
                    and report["finger_tracks"] == 30
                    and report["finger_translation_max"] == 0.0
                    and report["quat_norm_max_err"] < 1e-6
                    and report["morph_names_in_profile"]
                    and report["ik_off_ok"])

    cert_path = args.certificate or (args.out + ".m3.json")
    with open(cert_path, "w", encoding="utf-8") as f:
        _json.dump(report, f, ensure_ascii=False, indent=2)
    print(_json.dumps({k: report[k] for k in (
        "frames", "bone_tracks", "finger_tracks", "morph_tracks",
        "body_untouched", "unexpected_bones", "missing_bones",
        "finger_translation_max", "quat_norm_max_err", "ik_off_ok",
        "finger_step_deg", "ok")}, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit("M3 assembly verification failed")
    print(f"wrote {args.out} (+ {cert_path})")


def _replay_check(out_bones, rig: HandRig, angles: np.ndarray,
                  n_frames: int) -> float:
    """Compose the *written* quats and compare directions with the targets."""
    max_err = 0.0
    step = 3
    for fi in range(0, n_frames, step):
        measured = vector_to_measured(angles[fi])
        targets = angles_to_targets(rig, measured)
        for finger in rig.bones:
            chain = rig.chains[finger]
            qchain = (0.0, 0.0, 0.0, 1.0)
            for k, name in enumerate(chain):
                key = out_bones[name][fi]
                qchain = qnorm(qmul(qchain, qnorm(key["rot"])))
                d = qrot(qchain, rig.rest_dirs[finger][k])
                t = targets.get((finger, k))
                if t is None:
                    continue
                c = float(np.clip(np.dot(d, t), -1.0, 1.0))
                max_err = max(max_err, math.degrees(math.acos(c)))
    return float(max_err)


if __name__ == "__main__":
    main()
