# -*- coding: utf-8 -*-
"""Bake an evaluated VMD into a portable, safe VMD.

The bake writes only source bones (no inheritance-driven helpers, no IK
targets) and forces IK off, so the result replays identically in MMD and
Blender.  This is how synthetic GT is frozen after applying the MMD
semantics we intend the pipeline to reproduce.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from .evaluate import VmdEvaluator
from .writer import write_vmd


def select_bake_bones(model: Dict, profile: Dict) -> List[str]:
    """Source bones worth keying; inheritance-driven bones are excluded."""
    names = set(profile["writable_bones"].values())
    names.update(profile.get("root_chain") or [])
    for side in ("l", "r"):
        p = profile.get("pc", {}).get(side, {}).get("p")
        if p:
            names.add(p)
    for rec in (profile.get("twist") or {}).values():
        if rec.get("main"):
            names.add(rec["main"])
    out = []
    for b in model["bones"]:
        if b["name"] not in names:
            continue
        flags = b["flags"]
        if flags & (0x0100 | 0x0200):   # inherit rotation/translation
            continue
        if flags & 0x0020:              # IK target/solver bone
            continue
        out.append(b["name"])
    return out


def ik_names(model: Dict) -> List[str]:
    return [b["name"] for b in model["bones"]
            if b["flags"] & 0x0020 and "ik" in b]


def bake_evaluated_vmd(model: Dict, vmd: Dict, profile: Dict, out_path: str,
                       frames: Optional[Iterable[int]] = None,
                       solve_ik: bool = False,
                       apply_all_translations: bool = False,
                       bones: Optional[Sequence[str]] = None) -> Dict:
    """Evaluate `vmd` (default: MMD semantics, IK off) and write a safe VMD."""
    ev = VmdEvaluator(model, vmd, solve_ik=solve_ik,
                      force_ik_off=not solve_ik,
                      apply_all_translations=apply_all_translations)
    names = list(bones or select_bake_bones(model, profile))
    name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
    if frames is None:
        frames = range(vmd["max_frame"] + 1)
    frames = list(frames)
    tracks = {name: [] for name in names}
    for f in frames:
        rec = ev.frame(f)
        for name in names:
            i = name2i[name]
            tracks[name].append({
                "frame": f,
                "pos": rec["local_t"][i],
                "rot": rec["local_q"][i],
            })
    ik_off = [{"frame": 0, "show": True,
               "ik": {name: False for name in ik_names(model)}}]
    # Helper/root-chain bones that never move add 250 useless keys per bone
    # (M0's GT had all-zero 左肩P/右肩P tracks).  Dropping them changes nothing
    # in the evaluated pose: an absent track is the bind pose.
    writable = set((profile.get("writable_bones") or {}).values())
    dropped = []
    for name in list(tracks):
        if name in writable:
            continue
        keys = tracks[name]
        still = all(abs(k["pos"][0]) + abs(k["pos"][1]) + abs(k["pos"][2]) < 1e-9
                    and abs(k["rot"][3]) > 1.0 - 1e-9
                    and abs(k["rot"][0]) + abs(k["rot"][1]) + abs(k["rot"][2]) < 1e-9
                    for k in keys)
        if still:
            tracks.pop(name)
            dropped.append(name)
    write_vmd(out_path, vmd.get("model_name", ""), tracks, {}, ik_off)
    return {"bones": [n for n in names if n in tracks], "frames": len(frames),
            "path": out_path, "dropped_static_tracks": dropped}
