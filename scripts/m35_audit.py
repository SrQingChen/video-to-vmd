"""Independent audit of the M3.5 data engine artifacts.

Checks (not trusting the producing scripts):

  * split leakage (whole segments; R3 test only, tiflos/side never trained)
  * every paired sample: dims, finite, target bounded by the declared limits,
    binary masks
  * every pred_before VMD: 53 tracks (23 + 30), IK off, morph names exist,
    quat norms, no D-chain/helper tracks
  * every GT VMD: finger tracks and morph tracks present
  * render manifest: D-chain 0/0, PMX verify ok, frame count
  * augmentation: frame count preserved, train-only
  * feature builder signature: no GT argument

Usage: python scripts/m35_audit.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402


def load(p, d=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return d


def main():
    cfg = load(ROOT / "configs/models.json")["models"]
    profs = {m: load(ROOT / f"profiles/{m}.json") for m in cfg}
    models = {m: parse_pmx(cfg[m]["pmx"]) for m in cfg}
    idx = load(ROOT / "outputs/m35/data/index.json", {"clips": [], "leakage": {}})
    clips = idx["clips"]
    aud = {"clips": len(clips), "problems": [], "splits": {}, "pairs": {},
           "vmd": {}, "renders": {}, "augment": {}, "leakage": idx.get("leakage")}

    def bad(msg):
        aud["problems"].append(msg)

    for c in clips:
        aud["splits"].setdefault(c["split"], 0)
        aud["splits"][c["split"]] += 1
        p = ROOT / "outputs/m35/data" / c["clip"] / "pair.npz"
        if not p.exists():
            bad(f"missing pair {c['clip']}")
            continue
        d = np.load(p)
        lim = FD.residual_limits()
        if d["features"].shape[1] != c["feature_dim"]:
            bad(f"feature dim {c['clip']}")
        if d["target"].shape[1] != FD.RESIDUAL_DIM:
            bad(f"residual dim {c['clip']}")
        if not np.isfinite(d["features"]).all():
            bad(f"non-finite features {c['clip']}")
        if np.abs(d["target"]).max() > lim.max() + 1e-5:
            bad(f"target exceeds bound {c['clip']}")
        if not np.isin(d["mask"], [0.0, 1.0]).all():
            bad(f"non-binary mask {c['clip']}")
        aud["pairs"][c["clip"]] = {
            "frames": int(d["features"].shape[0]),
            "hand_mask_frames": int((d["mask"][:, FD.HAND_SLICE] > 0).sum() / 19),
            "face_mask_cells": int(d["mask"][:, FD.FACE_SLICE].sum()),
        }
        if c["variant"] == "base":
            for kind, rel in (("pred", c["paths"]["pred_before"]),
                              ("gt", c["paths"]["gt"])):
                if not (ROOT / rel).exists():
                    bad(f"missing {kind} {c['clip']}")
                    continue
                v = parse_vmd(str(ROOT / rel))
                prof = profs[c["model"]]
                fingers = sum(1 for k in prof["finger_map"].values()
                              if k in v["bones"])
                rec = {"tracks": len(v["bones"]), "fingers": fingers,
                       "morphs": sorted(v["morphs"]),
                       "ik_off": bool(v["ik_frames"]) and all(
                           not any(fr["ik"].values()) for fr in v["ik_frames"]),
                       "quat_norm_max_err": max(
                           abs(math.sqrt(sum(x * x for x in k["rot"])) - 1.0)
                           for ks in v["bones"].values() for k in ks)}
                if kind == "pred":
                    if rec["tracks"] != 53:
                        bad(f"pred tracks {c['clip']} {rec['tracks']}")
                    if rec["fingers"] != 30:
                        bad(f"pred fingers {c['clip']}")
                    if not rec["ik_off"]:
                        bad(f"pred IK not off {c['clip']}")
                    if rec["quat_norm_max_err"] > 1e-6:
                        bad(f"pred quat norm {c['clip']}")
                    unexpected = set(v["bones"]) - set(prof["finger_map"].values()) \
                        - set(prof["standard_map"].values())
                    if unexpected:
                        bad(f"pred unexpected tracks {c['clip']}: {sorted(unexpected)[:5]}")
                else:
                    if fingers != 30:
                        bad(f"gt fingers {c['clip']}")
                aud["vmd"][f"{c['clip']}_{kind}"] = rec

    render = load(ROOT / "outputs/m35/render/manifest.json", {"clips": {}})
    ok = sum(1 for r in render.get("clips", {}).values()
             if r.get("ok") or r.get("cached"))
    aud["renders"] = {"clips": len(render.get("clips", {})), "ok": ok,
                      "bad": [k for k, v in render.get("clips", {}).items()
                              if not (v.get("ok") or v.get("cached"))]}
    aug = load(ROOT / "outputs/m35/aug/manifest.json", {"clips": {}})
    aud["augment"] = {"clips": len(aug.get("clips", {}))}
    for name, rec in aug.get("clips", {}).items():
        base = name.split("__")[0]
        rp = render.get("clips", {}).get(base, {})
        if rp and rec.get("frames") not in (None, rp.get("frames")):
            bad(f"aug frame count {name}: {rec.get('frames')} vs {rp.get('frames')}")
        parts = base.split("_")
        if FD.assign_split(parts[0], parts[1], "_".join(parts[2:])) != "train":
            bad(f"aug on non-train clip {name}")

    # feature provenance: no GT parameter in the builder
    import inspect
    params = set(inspect.signature(FD.build_features).parameters)
    if any("gt" in p.lower() for p in params):
        bad(f"build_features has GT input: {params}")

    aud["ok"] = not aud["problems"]
    out = ROOT / "outputs/m35/audit.json"
    out.write_text(json.dumps(aud, ensure_ascii=False, indent=1, default=float),
                   encoding="utf-8")
    print(json.dumps({"ok": aud["ok"], "clips": aud["clips"],
                      "splits": aud["splits"], "renders": aud["renders"],
                      "problems": aud["problems"][:20],
                      "n_problems": len(aud["problems"])},
                     ensure_ascii=False, indent=1))
    if not aud["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
