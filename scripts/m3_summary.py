"""M3 summary: hands/face coverage, VMD certificate, render bake certificate.

Usage:
  python scripts/m3_summary.py \
    --body outputs/m25/t5_v01_offline.vmd \
    --vmd outputs/m3/t5_si.vmd \
    --hand outputs/m3/t5/hand.npz --face outputs/m3/t5/face.npz \
    --profile profiles/si.json \
    --render outputs/m3/render/t5_si_2k \
    --out outputs/m3/t5_si.summary.json
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

from vtv.face import SEMANTIC, MorphMap  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402

D_HELPERS = ("捩", "D", "助", "helper", "h1", "h2")


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--vmd", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--cert", default=None, help="assemble certificate (default <vmd>.m3.json)")
    ap.add_argument("--render", default=None, help="render output dir with meta.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    profile = _load_json(args.profile)
    body = parse_vmd(args.body)
    vmd = parse_vmd(args.vmd)
    hand = np.load(args.hand)
    face = np.load(args.face)
    cert = _load_json(args.cert or (args.vmd + ".m3.json"))

    finger_names = set(profile["finger_map"].values())
    profile_morphs = {m["name"] for m in profile.get("morphs", [])}
    body_names = set(body["bones"])
    out_names = set(vmd["bones"])

    levels_h = hand["level"]  # (N,2)
    n = levels_h.shape[0]
    side_names = ("l", "r")
    hand_levels = {}
    for si, s in enumerate(side_names):
        counts = {int(k): int((levels_h[:, si] == k).sum()) for k in (0, 1, 2, 3)}
        hand_levels[s] = {
            "L0": counts[0], "L1": counts[1], "L2": counts[2], "L3": counts[3],
            "L0_pct": 100 * counts[0] / n, "L1_pct": 100 * counts[1] / n,
            "L2_pct": 100 * counts[2] / n, "L3_pct": 100 * counts[3] / n,
            "L0L1_pct": 100 * (counts[0] + counts[1]) / n,
        }

    hand_json = {}
    hj_path = os.path.join(os.path.dirname(args.hand), "hand.json")
    if os.path.exists(hj_path):
        hand_json = _load_json(hj_path)
    face_json = {}
    fj_path = os.path.join(os.path.dirname(args.face), "face.json")
    if os.path.exists(fj_path):
        face_json = _load_json(fj_path)

    levels_f = face["level"]
    face_counts = {int(k): int((levels_f == k).sum()) for k in (0, 1, 2, 3)}
    used_morphs = sorted(vmd["morphs"].keys())
    morph_map = MorphMap.from_profile(profile)
    semantic_used = [s for s in SEMANTIC if morph_map.available.get(s) in used_morphs]

    blink = None
    if "まばたき" in vmd["morphs"]:
        keys = vmd["morphs"]["まばたき"]
        blink = {
            "keys": len(keys),
            "max": max((w for _, w in keys), default=0.0),
            "n_frames_gt_0.2": sum(1 for f, w in keys if w > 0.2),
            "max_step": max((abs(keys[i][1] - keys[i - 1][1])
                             for i in range(1, len(keys))), default=0.0),
        }

    # body invariance
    body_ok = True
    for name, keys in body["bones"].items():
        vkeys = vmd["bones"].get(name, [])
        if len(keys) != len(vkeys):
            body_ok = False
            break
        for a, b in zip(keys, vkeys):
            if int(a["frame"]) != int(b["frame"]):
                body_ok = False
                break
            if any(abs(x - y) > 1e-9 for x, y in zip(a["pos"], b["pos"])):
                body_ok = False
                break
            if any(abs(x - y) > 1e-9 for x, y in zip(a["rot"], b["rot"])):
                body_ok = False
                break
            if list(map(tuple, a["curves"])) != list(map(tuple, b["curves"])):
                body_ok = False
                break
        if not body_ok:
            break

    forbidden = sorted(nm for nm in out_names
                       if any(tok in nm for tok in ("捩", "D", "助", "helper"))
                       and nm not in finger_names and nm not in body_names)
    ik_off = bool(vmd["ik_frames"]) and all(not any(fr["ik"].values())
                                            for fr in vmd["ik_frames"])

    render = None
    if args.render:
        meta_path = os.path.join(args.render, "meta.json")
        if os.path.exists(meta_path):
            meta = _load_json(meta_path)
            bake = meta.get("inherit_bake", {})
            render = {
                "meta": meta_path,
                "resolution": meta.get("resolution"),
                "samples": meta.get("samples"),
                "d_chain": bake.get("d_chain_vs_fk"),
                "verify": bake.get("verify"),
                "scale_warning_count": bake.get("scale_warning_count"),
                "scale_warnings": bake.get("scale_warnings"),
            }

    gates = {
        "body_untouched": body_ok and cert.get("body_untouched", False),
        "bone_tracks_53": len(vmd["bones"]) == 53,
        "finger_tracks_30": len(finger_names & out_names) == 30,
        "ik_off_4": ik_off and len(vmd["ik_frames"][0]["ik"]) == 4,
        "morph_names_exist": all(nm in profile_morphs for nm in used_morphs),
        "no_d_helper": not forbidden,
        "finger_translation_zero": cert.get("finger_translation_max", 1.0) == 0.0,
        "quat_unit": cert.get("quat_norm_max_err", 1.0) < 1e-6,
        "replay_ok": all(stats.get("target_replay_max_deg", 1.0) < 1e-3
                         for stats in cert.get("replay", {}).values()),
        "finger_step_smooth": cert.get("finger_step_deg", {}).get("p95", 99) <= 10.0,
        "blink_step_ok": blink is None or blink["max_step"] <= 0.6,
    }
    if render:
        gates["d_chain_zero"] = bool(render["d_chain"] and
                                     render["d_chain"].get("max_pos_err") == 0.0 and
                                     render["d_chain"].get("max_rot_err_deg") == 0.0)
        gates["verify_lt_2e6"] = bool(render["verify"] and
                                      render["verify"].get("max_head_error", 1.0) < 2e-6)
        gates["scale_warning_zero"] = render.get("scale_warning_count") == 0

    report = {
        "profile": args.profile,
        "model_key": profile.get("model_key"),
        "frames": n,
        "hands": {
            "levels": hand_levels,
            "detect": {s: (hand_json.get("sides", {}).get(s, {}).get("detect_rate")) for s in side_names},
            "valid": {s: (hand_json.get("sides", {}).get(s, {}).get("valid_rate")) for s in side_names},
            "l1_gain": {s: (hand_json.get("sides", {}).get(s, {}).get("fit_l1_gain")) for s in side_names},
            "segments": {s: [g for g in hand_json.get("sides", {}).get(s, {}).get("segments", [])
                             if g.get("level") in ("core", "weak")] for s in side_names},
            "runtime": hand_json.get("runtime"),
        },
        "face": {
            "levels": {f"L{k}": face_counts[k] for k in (0, 1, 2, 3)},
            "L0_pct": 100 * face_counts[0] / n,
            "L1_pct": 100 * face_counts[1] / n,
            "detect_rate": face_json.get("detect_rate"),
            "patch_ok_rate_detected": face_json.get("patch_ok_rate_detected"),
            "procedural_blink_frames": face_json.get("procedural_blink_frames"),
            "unreliable": face_json.get("unreliable"),
            "morph_tracks": used_morphs,
            "semantics_used": semantic_used,
            "semantic_coverage": f"{len(semantic_used)}/{len(SEMANTIC)}",
            "blink": blink,
            "runtime": face_json.get("runtime"),
        },
        "vmd": {
            "bone_tracks": len(vmd["bones"]),
            "finger_tracks": len(finger_names & out_names),
            "morph_tracks": used_morphs,
            "ik_bones_off": sorted(vmd["ik_frames"][0]["ik"]) if vmd["ik_frames"] else [],
            "forbidden_bones": forbidden,
            "finger_step_deg": cert.get("finger_step_deg"),
            "quat_norm_max_err": cert.get("quat_norm_max_err"),
        },
        "render": render,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "manual_review": {
            "required": True,
            "note": "hands/face visible naturalness is a visual check; see render stills",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    print(json.dumps({
        "hands": {s: {k: round(v, 1) if isinstance(v, float) else v
                      for k, v in hand_levels[s].items()} for s in side_names},
        "face": report["face"]["levels"],
        "vmd": {"tracks": report["vmd"]["bone_tracks"],
                "fingers": report["vmd"]["finger_tracks"],
                "morphs": used_morphs,
                "finger_step": report["vmd"]["finger_step_deg"]},
        "gates": gates,
        "all_gates_pass": report["all_gates_pass"],
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
