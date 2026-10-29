# -*- coding: utf-8 -*-
"""Per-model auto-framing for the Blender evidence renders (M5).

M4 used a fixed formula that only fitted the *vertical* field of view and then
added an ad-hoc horizontal term (``width * margin * 1.2``).  In a portrait 2K
frame (1440x2560) the horizontal FOV is much narrower than the vertical one, so
wide poses (tiflos' arms, si's sleeves) were rendered outside the image even
though the clip-level bbox was known.

This module is the pure-math replacement used by ``scripts/render_synth.py``:

* sensor fit / aspect -> true horizontal and vertical FOV (Blender AUTO fit uses
  the 36 mm sensor on the *larger* image dimension);
* the camera basis matches ``look.to_track_quat("-Z", "Y")``;
* ``fit_distance`` binary-searches the smallest distance at which every bbox
  corner stays inside ``1 / margin`` of the frame on both axes;
* ``framing_report`` projects the per-frame bboxes with the chosen camera so the
  render can be gated on "no part of the character left the frame".

No bpy here: the functions take plain numbers/arrays, so they are unit-testable
outside Blender.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_SENSOR_WIDTH = 36.0
DEFAULT_MARGIN = 1.15


def sensor_fovs(res: Sequence[int], lens: float,
                sensor_width: float = DEFAULT_SENSOR_WIDTH,
                sensor_fit: str = "AUTO") -> Tuple[float, float]:
    """(hfov, vfov) in radians for Blender's camera model.

    ``sensor_fit='AUTO'`` puts ``sensor_width`` on the larger image dimension
    (vertical for a portrait render), which is why a 1440x2560 frame has a much
    narrower horizontal FOV than a square one.
    """
    rx, ry = int(res[0]), int(res[1])
    if rx <= 0 or ry <= 0:
        raise ValueError(f"bad resolution {res!r}")
    if lens <= 0:
        raise ValueError(f"bad lens {lens!r}")
    aspect = rx / float(ry)
    fit = str(sensor_fit or "AUTO").upper()
    if fit == "HORIZONTAL":
        hfov = 2.0 * math.atan(sensor_width / (2.0 * lens))
        vfov = 2.0 * math.atan(math.tan(hfov * 0.5) / aspect)
    elif fit == "VERTICAL":
        vfov = 2.0 * math.atan(sensor_width / (2.0 * lens))
        hfov = 2.0 * math.atan(math.tan(vfov * 0.5) * aspect)
    else:  # AUTO
        if aspect >= 1.0:  # landscape: sensor width is the horizontal side
            hfov = 2.0 * math.atan(sensor_width / (2.0 * lens))
            vfov = 2.0 * math.atan(math.tan(hfov * 0.5) / aspect)
        else:              # portrait: sensor width is the vertical side
            vfov = 2.0 * math.atan(sensor_width / (2.0 * lens))
            hfov = 2.0 * math.atan(math.tan(vfov * 0.5) * aspect)
    return hfov, vfov


def camera_basis(direction: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, forward) world axes of a camera looking at ``-direction``.

    ``direction`` is the unit vector from the target towards the camera (the
    same convention as the azimuth in ``render_synth.py``).  The basis matches
    Blender's ``look.to_track_quat("-Z", "Y")``: forward = -Z_cam,
    up = world +Z projected perpendicular to forward, right = up x (-forward).
    """
    d = np.asarray(direction, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        raise ValueError("degenerate camera direction")
    d = d / n
    forward = -d  # camera looks from cam position towards the target
    world_up = np.array([0.0, 0.0, 1.0])
    up = world_up - float(np.dot(world_up, forward)) * forward
    nu = float(np.linalg.norm(up))
    if nu < 1e-9:  # looking straight down/up: pick a stable screen-up
        up = np.array([0.0, 1.0, 0.0])
        up = up - float(np.dot(up, forward)) * forward
        nu = float(np.linalg.norm(up))
    up = up / nu
    right = np.cross(up, -forward)  # = up x Z_cam
    right = right / max(1e-12, float(np.linalg.norm(right)))
    return right, up, forward


def bbox_corners(lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return np.asarray([[x, y, z] for x in (lo[0], hi[0])
                       for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                      dtype=np.float64)


def project_points(points, cam_pos, right, up, forward,
                   hfov: float, vfov: float) -> np.ndarray:
    """Normalised device coordinates (x right, y up) for world ``points``.

    Points behind the camera get ``inf`` so a caller cannot mistake them for
    in-frame.  NDC is +-1 at the image border (no margin applied here).
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    d = p - np.asarray(cam_pos, dtype=np.float64)
    depth = d @ np.asarray(forward, dtype=np.float64)
    tx = math.tan(hfov * 0.5)
    ty = math.tan(vfov * 0.5)
    out = np.full((len(p), 2), np.inf, dtype=np.float64)
    front = depth > 1e-6
    out[front, 0] = (d[front] @ np.asarray(right)) / (depth[front] * tx)
    out[front, 1] = (d[front] @ np.asarray(up)) / (depth[front] * ty)
    return out


def max_abs_ndc(corners, cam_pos, direction, res, lens,
                sensor_width: float = DEFAULT_SENSOR_WIDTH,
                sensor_fit: str = "AUTO") -> float:
    right, up, forward = camera_basis(direction)
    hfov, vfov = sensor_fovs(res, lens, sensor_width, sensor_fit)
    ndc = project_points(corners, cam_pos, right, up, forward, hfov, vfov)
    if not np.isfinite(ndc).all():
        return float("inf")
    return float(np.abs(ndc).max())


def fit_distance(corners, center, direction, res, lens,
                 sensor_width: float = DEFAULT_SENSOR_WIDTH,
                 sensor_fit: str = "AUTO",
                 margin: float = DEFAULT_MARGIN,
                 lo: float = 0.05, hi: float = 200.0,
                 iters: int = 64) -> dict:
    """Smallest target-to-camera distance that keeps every corner in frame.

    ``margin`` is a safety factor: the fit targets ``max|ndc| <= 1/margin``,
    i.e. 1.15 leaves 15% padding.  Returns a report with the FOVs and the
    achieved extents so the render can record *why* the camera is where it is.
    """
    if margin < 1.0:
        raise ValueError("margin must be >= 1")
    center = np.asarray(center, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = d / float(np.linalg.norm(d))
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 3)
    target = 1.0 / float(margin)
    hfov, vfov = sensor_fovs(res, lens, sensor_width, sensor_fit)

    def extent(dist):
        cam = center + d * dist
        return max_abs_ndc(corners, cam, d, res, lens, sensor_width, sensor_fit)

    e_hi = extent(hi)
    if not math.isfinite(e_hi) or e_hi > target:
        raise ValueError(
            f"camera cannot frame the character even at {hi:.1f} units "
            f"(max |ndc| {e_hi:.3f} > {target:.3f})")
    a, b = lo, hi
    for _ in range(int(iters)):
        mid = 0.5 * (a + b)
        if extent(mid) <= target:
            b = mid
        else:
            a = mid
    dist = b * 1.001  # a hair outside the bisection boundary
    cam = center + d * dist
    ndc = project_points(corners, cam, *camera_basis(d), hfov, vfov)
    return {
        "distance": float(dist),
        "camera": [float(c) for c in cam],
        "hfov_deg": math.degrees(hfov),
        "vfov_deg": math.degrees(vfov),
        "margin": float(margin),
        "target_abs_ndc": float(target),
        "bbox_abs_ndc": [float(np.abs(ndc[:, 0]).max()),
                         float(np.abs(ndc[:, 1]).max())],
        "ok": bool(np.abs(ndc).max() <= 1.0 + 1e-9),
        "space": "target_abs_ndc is the fit target; the image border is 1.0",
    }


def framing_report(per_frame_bboxes: Iterable[Mapping],
                   cam_pos, direction, res, lens,
                   sensor_width: float = DEFAULT_SENSOR_WIDTH,
                   sensor_fit: str = "AUTO",
                   margin: float = DEFAULT_MARGIN) -> dict:
    """Project every per-frame bbox with an auto-framed camera."""
    right, up, forward = camera_basis(direction)
    hfov, vfov = sensor_fovs(res, lens, sensor_width, sensor_fit)
    return framing_report_basis(per_frame_bboxes, cam_pos, right, up, forward,
                                hfov, vfov, margin)


def framing_report_basis(per_frame_bboxes: Iterable[Mapping],
                         cam_pos, right, up, forward,
                         hfov: float, vfov: float,
                         margin: float = DEFAULT_MARGIN) -> dict:
    """Project every per-frame bbox with an explicit camera basis.

    Use this for a reused camera (``--camera-from``): its rotation need not look
    exactly at the new bbox centre, so the basis must come from its quaternion.
    ``ok`` means every sampled frame stays inside the image (NDC <= 1);
    ``within_margin`` additionally checks the 1/margin safety area.
    """
    worst = None
    n = 0
    max_abs = 0.0
    for rec in per_frame_bboxes:
        corners = bbox_corners(rec["min"], rec["max"])
        ndc = project_points(corners, cam_pos, right, up, forward, hfov, vfov)
        if not np.isfinite(ndc).all():
            return {"ok": False, "reason": "a bbox corner is behind the camera",
                    "frames": n, "max_abs_ndc": None, "worst_frame": rec.get("frame")}
        val = float(np.abs(ndc).max())
        if val > max_abs:
            max_abs = val
            worst = int(rec.get("frame", -1))
        n += 1
    if n == 0:
        return {"ok": False, "reason": "no frames sampled", "frames": 0,
                "max_abs_ndc": None, "worst_frame": None}
    return {
        "ok": bool(max_abs <= 1.0),
        "within_margin": bool(max_abs <= 1.0 / float(margin)),
        "frames": n,
        "max_abs_ndc": float(max_abs),
        "worst_frame": worst,
        "margin": float(margin),
        "hfov_deg": math.degrees(hfov),
        "vfov_deg": math.degrees(vfov),
        "note": "NDC 1.0 = image border; within_margin uses 1/margin",
    }
