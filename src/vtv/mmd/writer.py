# -*- coding: utf-8 -*-
"""VMD (Vocaloid Motion Data 0002) writer.

Interoperability target: MMD Tools' VMD exporter/importer and MMD itself.
Layout follows MMD Tools ``core/vmd/exporter.py``:

  64-byte interpolation = 4 channel blocks (X, Y, Z, R), 16 bytes each;
  control points x1,y1,x2,y2 sit at +0,+4,+8,+12 inside a block.

All coordinates stay in MMD's native left-handed space (PMX units).
Writer only emits bone / morph / IK records; camera, light and self-shadow
sections are empty on purpose (synthetic GT uses a fixed camera).
"""
from __future__ import annotations

import struct
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

VMD_SIGN = b"Vocaloid Motion Data 0002"
DEFAULT_CURVE = ((20, 20), (107, 107))


def _encode_name(name: str, size: int) -> bytes:
    """Encode a VMD name field (cp932/shift-jis), truncating on a boundary."""
    try:
        raw = name.encode("cp932")
    except UnicodeEncodeError:
        raw = name.encode("cp932", errors="replace")
    if len(raw) > size:
        raw = raw[:size]
        # do not leave a half shift-jis character at the tail
        while raw and (raw[-1] & 0xC0) == 0x80:
            raw = raw[:-1]
    return raw.ljust(size, b"\x00")


def _curve_pairs(curve) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Accept (x1,y1,x2,y2) or ((x1,y1),(x2,y2))."""
    if len(curve) == 4 and all(isinstance(v, int) for v in curve):
        return ((curve[0], curve[1]), (curve[2], curve[3]))
    return (tuple(curve[0]), tuple(curve[1]))


def _interp_bytes(curves: Sequence) -> bytes:
    """Build the 64-byte interpolation block the way MMD Tools does."""
    if len(curves) != 4:
        raise ValueError("expected 4 curves (X,Y,Z,R)")
    curves = [_curve_pairs(c) for c in curves]
    x, y, z, r = curves
    x_x1, x_y1 = x[0]
    x_x2, x_y2 = x[1]
    y_x1, y_y1 = y[0]
    y_x2, y_y2 = y[1]
    z_x1, z_y1 = z[0]
    z_x2, z_y2 = z[1]
    r_x1, r_y1 = r[0]
    r_x2, r_y2 = r[1]
    data = [
        x_x1, y_x1, 0, 0, x_y1, y_y1, z_y1, r_y1,
        x_x2, y_x2, z_x2, r_x2, x_y2, y_y2, z_y2, r_y2,
        y_x1, z_x1, r_x1, x_y1, y_y1, z_y1, r_y1, x_x2,
        y_x2, z_x2, r_x2, x_y2, y_y2, z_y2, r_y2, 0,
        z_x1, r_x1, x_y1, y_y1, z_y1, r_y1, x_x2, y_x2,
        z_x2, r_x2, x_y2, y_y2, z_y2, r_y2, 0, 0,
        r_x1, x_y1, y_y1, z_y1, r_y1, x_x2, y_x2, z_x2,
        r_x2, x_y2, y_y2, z_y2, r_y2, 0, 0, 0,
    ]
    for v in data:
        if not 0 <= v <= 255:
            raise ValueError("interpolation control point out of range: %r" % (v,))
    return bytes(data)


def write_vmd(
    path: str,
    model_name: str = "",
    bone_frames: Optional[Mapping[str, Iterable[Mapping]]] = None,
    morph_frames: Optional[Mapping[str, Iterable[Tuple[int, float]]]] = None,
    ik_frames: Optional[Iterable[Mapping]] = None,
) -> None:
    """Write a VMD 0002 file.

    bone_frames:
      {bone_name: [{frame, pos:(x,y,z), rot:(x,y,z,w), curves?}, ...]}
      curves optional; defaults to (20,20),(107,107) for X/Y/Z/R.
    morph_frames:
      {morph_name: [(frame, weight), ...]}
    ik_frames:
      [{frame, show, ik: {bone_name: bool}}, ...]
    """
    bone_frames = bone_frames or {}
    morph_frames = morph_frames or {}
    ik_frames = list(ik_frames or [])

    with open(path, "wb") as f:
        f.write(VMD_SIGN.ljust(30, b"\x00"))
        f.write(_encode_name(model_name, 20))

        keys = [(name, k) for name, frames in bone_frames.items() for k in frames]
        keys.sort(key=lambda item: (item[0], item[1]["frame"]))
        f.write(struct.pack("<I", len(keys)))
        for name, k in keys:
            curves = k.get("curves") or (DEFAULT_CURVE,) * 4
            f.write(struct.pack(
                "<15sI3f4f64B",
                _encode_name(name, 15),
                int(k["frame"]),
                *[float(v) for v in k["pos"]],
                *[float(v) for v in k["rot"]],
                *_interp_bytes(curves),
            ))

        counts = [(name, sorted(frames, key=lambda x: x[0]))
                  for name, frames in morph_frames.items()]
        f.write(struct.pack("<I", sum(len(v) for _, v in counts)))
        for name, frames in counts:
            for frame, weight in frames:
                f.write(struct.pack("<15sIf", _encode_name(name, 15),
                                    int(frame), float(weight)))

        f.write(struct.pack("<I", 0))  # camera
        f.write(struct.pack("<I", 0))  # light
        f.write(struct.pack("<I", 0))  # self shadow

        f.write(struct.pack("<I", len(ik_frames)))
        for fr in sorted(ik_frames, key=lambda x: int(x["frame"])):
            states = fr.get("ik", {})
            f.write(struct.pack("<IBI", int(fr["frame"]),
                                1 if fr.get("show", True) else 0, len(states)))
            for name, on in states.items():
                f.write(struct.pack("<20sB", _encode_name(name, 20), 1 if on else 0))


def identity_vmd(
    bone_names: Sequence[str],
    frame: int = 0,
    ik_names: Sequence[str] = (),
    include_ik_off: bool = True,
    model_name: str = "",
) -> Dict:
    """Build a key set that keeps the bind pose (and disables leg IK)."""
    bones = {name: [{"frame": frame, "pos": (0.0, 0.0, 0.0),
                     "rot": (0.0, 0.0, 0.0, 1.0)}] for name in bone_names}
    ik_frames = []
    if include_ik_off and ik_names:
        ik_frames = [{"frame": frame, "show": True,
                      "ik": {name: False for name in ik_names}}]
    return {"model_name": model_name, "bone_frames": bones,
            "morph_frames": {}, "ik_frames": ik_frames}


def write_identity(path: str, bone_names: Sequence[str], ik_names: Sequence[str] = (),
                   model_name: str = "") -> None:
    data = identity_vmd(bone_names, ik_names=ik_names, model_name=model_name)
    write_vmd(path, data["model_name"], data["bone_frames"],
              data["morph_frames"], data["ik_frames"])
