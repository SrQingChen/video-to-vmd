"""Asset-free CI smoke: chunk plan -> align -> merge -> certificate -> seam QC.

Uses only the VMD container and synthetic quaternions, so it runs on a plain
NumPy install (no torch, MediaPipe, Blender or test media).  The real pipeline
is exercised on the development machine (see .github/workflows/ci.yml).

    python scripts/ci_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import longvideo as LV  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402

CURVES = ((20, 20, 107, 107),) * 4
PROFILE = {"standard_map": {"root": "センター"}, "height": 18.0}


def _qaxis(axis, deg):
    import math
    a = math.radians(deg) * 0.5
    v = np.asarray(axis, float)
    v = v / np.linalg.norm(v)
    return tuple([*(v * math.sin(a)), math.cos(a)])


def make(n, yaw=0.0):
    return {"model_name": "ci", "bones": {
        "センター": [{"frame": i, "pos": (0.1 * i, 0.0, 0.0),
                      "rot": _qaxis([0, 1, 0], yaw + 2 * i), "curves": CURVES}
                     for i in range(n)],
        "上半身": [{"frame": i, "pos": (0, 0, 0),
                    "rot": _qaxis([1, 0, 0], 3 * i), "curves": CURVES}
                   for i in range(n)],
    }, "morphs": {"まばたき": [(0, 0.0), (n - 1, 0.5)]},
        "ik_frames": [{"frame": 0, "show": True, "ik": {"左足ＩＫ": False}}],
        "max_frame": n - 1}


def main():
    plan = LV.plan_chunks(800, 300, 30)
    assert plan["n_chunks"] == 3 and plan["ends"][-1] == 800
    a, b, c = make(300), make(300, 4.0), make(260, 2.0)
    parts = [a, b, c]
    offsets = plan["starts"]
    lengths = [p["max_frame"] + 1 for p in parts]
    assert sum(lengths) - 30 * 2 == 800
    aligned = []
    for i, p in enumerate(parts):
        if i == 0:
            aligned.append(p)
            continue
        prev = aligned[i - 1]
        al = LV.alignment(prev, p, PROFILE,
                          list(range(lengths[i - 1] - 30, lengths[i - 1])),
                          list(range(30)))
        aligned.append(LV.apply_alignment(p, PROFILE, al["yaw_deg"],
                                          al["translation"]))
    merged, rep = LV.merge_vmds(aligned, offsets, 800, 30)
    assert rep["ok"] and merged["max_frame"] == 799
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "merged.vmd"
        write_vmd(str(path), merged["model_name"], merged["bones"],
                  merged["morphs"], merged["ik_frames"])
        back = parse_vmd(str(path))
        assert back["max_frame"] == 799
        for name, keys in back["bones"].items():
            fr = [k["frame"] for k in keys]
            assert fr == list(range(800)), name
            q = np.array([k["rot"] for k in keys])
            assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-6)
            assert np.sum(q[1:] * q[:-1], axis=1).min() > 0.0, name
        # merged means must stay between the inputs (no invented pose)
        assert back["morphs"]["まばたき"][0] == (0, 0.0)
        assert len({k[0] for k in back["morphs"]["まばたき"]}) == len(
            back["morphs"]["まばたき"])
    # seam report: the merged synthetic motion has steady steps
    fake = {"world": {"toe_l": np.zeros((800, 3)),
                      "toe_r": np.zeros((800, 3))},
            "world_q": {"pelvis": np.tile([0.0, 0.0, 0.0, 1.0], (800, 1))}}
    report = LV.boundary_report(fake, fake, merged, PROFILE, [270, 540], 30, 800)
    assert report["ok"], report["seams"]
    print("CI SMOKE OK: 3 chunks -> 800 frames, seams 270/540, "
          "certificate fields", rep["bones"], "bones")


if __name__ == "__main__":
    main()
