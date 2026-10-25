# -*- coding: utf-8 -*-
"""Evaluate a VMD on a PMX skeleton (MMD semantics) and expose world joints.

Uses the vendored mmd2gltf deform pipeline (inherit rotation/translation +
CCD IK) so that "what MMD would do" is the reference for the audit and the
synthetic closed loop.  IK can be disabled globally or per VMD IK frames.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .animation import FPS, Skeleton, Track
from .pmx import parse_pmx
from .vmd import parse_vmd


def _name_key(name: str, size: int) -> bytes:
    return name.encode("shift-jis", errors="replace")[:size]


def build_name_map(model: Dict) -> Dict[bytes, int]:
    out = {}
    for i, b in enumerate(model["bones"]):
        out.setdefault(_name_key(b["name"], 15), i)
    return out


def ik_bones(model: Dict) -> List[int]:
    return [i for i, b in enumerate(model["bones"])
            if b["flags"] & 0x0020 and "ik" in b]


class VmdEvaluator:
    def __init__(self, model: Dict, vmd: Dict, solve_ik: bool = True,
                 force_ik_off: bool = False, apply_all_translations: bool = False):
        self.model = model
        self.vmd = vmd
        self.solve_ik = solve_ik and not force_ik_off
        self.n = len(model["bones"])
        self.name_map = build_name_map(model)
        self.tracks: Dict[int, Track] = {}
        self.unmatched: List[str] = []
        for name, keys in vmd.get("bones", {}).items():
            i = self.name_map.get(_name_key(name, 15))
            if i is None:
                self.unmatched.append(name)
            else:
                self.tracks[i] = Track(keys)
        self.skeleton = Skeleton(model, apply_all_translations=apply_all_translations)
        self.all_ik = ik_bones(model)
        self.base_enabled: Set[int] = set(self.all_ik) if self.solve_ik else set()
        self.ik_key_frames = self._build_ik_frames()
        # cached per frame
        self._cache: Dict[int, Dict] = {}

    def _build_ik_frames(self):
        out = []
        ik_map = {}
        for i in self.all_ik:
            ik_map.setdefault(_name_key(self.model["bones"][i]["name"], 20), i)
        for fr in self.vmd.get("ik_frames", []):
            states = {}
            for name, on in fr.get("ik", {}).items():
                i = ik_map.get(_name_key(name, 20))
                if i is not None:
                    states[i] = bool(on)
            out.append((int(fr["frame"]), states))
        out.sort(key=lambda x: x[0])
        return out

    def ik_enabled_at(self, frame: int) -> Set[int]:
        if not self.ik_key_frames:
            return set(self.base_enabled)
        state: Dict[int, bool] = {}
        for f0, m in self.ik_key_frames:
            if f0 > frame:
                break
            state.update(m)
        return {i for i in self.base_enabled if state.get(i, True)}

    def frame(self, frame: int) -> Dict:
        """Evaluate one VMD frame; returns local/world data for every bone."""
        cached = self._cache.get(frame)
        if cached is not None:
            return cached
        sampled = {i: tr.sample(frame) for i, tr in self.tracks.items()}
        self.skeleton.pose(sampled, ik_enabled=self.ik_enabled_at(frame))
        sk = self.skeleton
        n = self.n
        local_q = list(sk.local_q)
        local_t = list(sk.local_t)
        world_q = [None] * n
        world_pos = [None] * n
        for i in range(n):
            q, p = sk.world(i)
            world_q[i] = q
            world_pos[i] = p
        rec = {
            "frame": frame,
            "local_q": local_q,
            "local_t": local_t,
            "world_q": world_q,
            "world_pos": world_pos,
        }
        self._cache[frame] = rec
        return rec

    def world_positions(self, frames: Iterable[int],
                        bones: Optional[Sequence[int]] = None) -> Dict[str, List]:
        """Return {bone_name: [(f, x,y,z), ...]} for selected bones."""
        if bones is None:
            bones = list(range(self.n))
        names = [self.model["bones"][i]["name"] for i in bones]
        out = {name: [] for name in names}
        for f in frames:
            rec = self.frame(f)
            for i, name in zip(bones, names):
                p = rec["world_pos"][i]
                out[name].append((f, p[0], p[1], p[2]))
        return out


def load_and_evaluate(model_path: str, vmd_path: str, solve_ik: bool = True,
                      force_ik_off: bool = False):
    model = parse_pmx(model_path)
    vmd = parse_vmd(vmd_path)
    return model, vmd, VmdEvaluator(model, vmd, solve_ik=solve_ik,
                                    force_ik_off=force_ik_off)
