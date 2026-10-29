"""M5 long-video tests: chunk planning, alignment, VMD merge, seam report.

Run:  python tests/test_longvideo.py
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv import longvideo as LV  # noqa: E402

CURVES = ((20, 20, 107, 107),) * 4
PROFILE = {"standard_map": {"root": "センター"}, "height": 18.0}


def q_axis(axis, deg):
    a = math.radians(deg) * 0.5
    v = np.asarray(axis, float)
    v = v / np.linalg.norm(v)
    return tuple([*(v * math.sin(a)), math.cos(a)])


def make_vmd(n, yaw0=0.0, pitch=0.0, x0=0.0, fps=1.0):
    bones = {
        "センター": [{"frame": i, "pos": (x0 + 0.1 * i, 0.0, 0.0),
                      "rot": q_axis([0, 1, 0], yaw0 + 2.0 * i),
                      "curves": CURVES} for i in range(n)],
        "上半身": [{"frame": i, "pos": (0.0, 0.0, 0.0),
                    "rot": q_axis([1, 0, 0], pitch + 3.0 * i),
                    "curves": CURVES} for i in range(n)],
    }
    return {"model_name": "m", "bones": bones,
            "morphs": {"まばたき": [(0, 0.0), (max(1, n - 1), 0.5)]},
            "ik_frames": [{"frame": 0, "show": True,
                           "ik": {"左足ＩＫ": False, "右足ＩＫ": False}}],
            "max_frame": n - 1}


def slice_vmd(vmd, start, end):
    """Frames [start, end) re-based to 0 (same content, new local frames)."""
    out = {"model_name": vmd.get("model_name", ""), "bones": {},
           "morphs": {}, "ik_frames": [dict(x) for x in vmd["ik_frames"]],
           "max_frame": end - start - 1}
    for name, keys in vmd["bones"].items():
        out["bones"][name] = [{"frame": k["frame"] - start,
                               "pos": k["pos"], "rot": k["rot"],
                               "curves": k["curves"]}
                              for k in keys[start:end]]
    return out


def close_tuple(a, b, tol=1e-9):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


# ---------------------------------------------------------------------------
def test_plan_invariants():
    for n in (1, 2, 299, 300, 301, 330, 480, 541, 571, 600, 1630, 1810, 2178,
              3600):
        p = LV.plan_chunks(n, 300, 30, 30)
        starts, ends = p["starts"], p["ends"]
        assert starts[0] == 0 and ends[-1] == n
        assert len(starts) == len(ends) == p["n_chunks"]
        assert p["lengths"] == [e - s for s, e in zip(starts, ends)]
        seen = np.zeros(n, np.int32)
        for s, e in zip(starts, ends):
            assert 0 <= s < e <= n
            seen[s:e] += 1
        assert seen.min() >= 1 and seen.max() <= 2
        for i in range(1, len(starts)):
            assert ends[i - 1] - starts[i] == 30
        assert p["lengths"][-1] >= 30 or len(starts) == 1
    assert LV.plan_chunks(300, 300, 30)["n_chunks"] == 1
    for args in ((0, 300, 30), (10, 0, 0), (10, 300, 300)):
        try:
            LV.plan_chunks(*args)
            raise AssertionError(f"plan_chunks{args} should fail")
        except ValueError:
            pass


def test_merge_weights():
    W = LV.merge_weights([0, 270], [300, 300], 570, 30)
    assert np.allclose(W.sum(1), 1.0)
    assert W[0, 0] == 1.0 and W[-1, 1] == 1.0
    assert W[270, 0] > 0.99 and W[270, 1] < 0.01
    assert W[299, 1] > 0.99 and W[299, 0] < 0.01
    assert np.all(np.diff(W[270:300, 1]) >= 0)
    try:
        LV.merge_weights([0, 270], [300, 300], 570, 29)
        raise AssertionError("wrong overlap must fail")
    except ValueError:
        pass


def test_alignment_recovers_rigid_transform():
    a = make_vmd(20)
    b = LV.apply_alignment(a, PROFILE, 12.0, [1.0, 2.0, 3.0])
    al = LV.alignment(a, b, PROFILE, list(range(5, 15)), list(range(5, 15)))
    assert abs(al["yaw_deg"] + 12.0) < 1e-6, al
    back = LV.apply_alignment(b, PROFILE, al["yaw_deg"], al["translation"])
    pa = a["bones"]["センター"]
    pb = back["bones"]["センター"]
    for ka, kb in zip(pa, pb):
        assert close_tuple(ka["pos"], kb["pos"], 1e-8)
        assert close_tuple(ka["rot"], kb["rot"], 1e-8)
    assert al["residual_root_max_units"] < 1e-8
    assert al["root_rotation_residual_after_deg"]["max"] < 1e-6


def test_alignment_identity():
    a = make_vmd(10)
    al = LV.alignment(a, a, PROFILE, [0, 1, 2], [0, 1, 2])
    assert abs(al["yaw_deg"]) < 1e-9 and al["residual_root_max_units"] < 1e-9


def test_merge_frame_count_and_density():
    a = make_vmd(170)
    b = make_vmd(170, yaw0=5.0, x0=1.0)
    b = LV.apply_alignment(b, PROFILE, 4.0, [0.5, 0.0, -0.2])
    M, rep = LV.merge_vmds([a, b], [0, 140], 310, 30)
    assert M["max_frame"] == 309
    for name, keys in M["bones"].items():
        frames = [k["frame"] for k in keys]
        assert frames == list(range(310)), name
        q = np.array([k["rot"] for k in keys])
        dot = np.sum(q[1:] * q[:-1], axis=1)
        assert dot.min() > 0.0, f"{name} hemisphere discontinuity"
        assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-9)
    assert rep["ok"] and rep["frames"] == 310 and rep["bones"] == 2
    assert M["ik_frames"][0]["ik"] == {"左足ＩＫ": False, "右足ＩＫ": False}
    keys = M["morphs"]["まばたき"]
    fr = [k[0] for k in keys]
    assert len(fr) == len(set(fr)) and keys[0] == (0, 0.0)
    try:
        LV.merge_vmds([a], [0], 400, 30)
        raise AssertionError("missing frames must fail")
    except ValueError:
        pass


def test_merge_preserves_acceptable_track_equality():
    """A track identical in each chunk's before/final stays identical."""
    a = make_vmd(120)
    b = make_vmd(120, yaw0=3.0)
    before = [a, b]
    final = []
    for part in before:
        p = {"model_name": part["model_name"], "bones": dict(part["bones"]),
             "morphs": dict(part["morphs"]),
             "ik_frames": part["ik_frames"], "max_frame": part["max_frame"]}
        # feedback may only change 上半身; root stays identical
        p["bones"] = dict(part["bones"])
        p["bones"]["上半身"] = [dict(k, rot=q_axis([1, 0, 0], 30.0))
                               for k in part["bones"]["上半身"]]
        final.append(p)
    Mb, _ = LV.merge_vmds(before, [0, 90], 210, 30)
    Mf, _ = LV.merge_vmds(final, [0, 90], 210, 30)
    for name in ("センター",):
        for kb, kf in zip(Mb["bones"][name], Mf["bones"][name]):
            assert close_tuple(kb["rot"], kf["rot"], 1e-12)
            assert close_tuple(kb["pos"], kf["pos"], 1e-12)
    assert any(not close_tuple(kb["rot"], kf["rot"], 1e-9)
               for kb, kf in zip(Mb["bones"]["上半身"], Mf["bones"]["上半身"]))


def test_merge_seam_is_smooth_after_alignment():
    full = make_vmd(460)
    a = slice_vmd(full, 0, 400)
    tail = slice_vmd(full, 340, 460)
    tail_shift = LV.apply_alignment(tail, PROFILE, 7.0, [0.3, 2.0, -1.0])
    al = LV.alignment(a, tail_shift, PROFILE, list(range(340, 400)),
                      list(range(0, 60)))
    tail_aligned = LV.apply_alignment(tail_shift, PROFILE, al["yaw_deg"],
                                      al["translation"])
    M, _ = LV.merge_vmds([a, tail_aligned], [0, 340], 460, 60)
    step = LV.local_step_series(M, 460)
    interior = step[10:330]
    seam = step[340:400]
    assert seam.max() <= max(np.percentile(interior, 95) * 1.5, 2.0), (
        seam.max(), np.percentile(interior, 95))
    # and the root is continuous through the seam
    root = LV.root_step_series(M, PROFILE, 460)
    assert root[340:400].max() <= max(np.percentile(root[10:330], 95) * 1.5,
                                      0.05)


def test_merge_feedback_offsets():
    r1 = {"mode": "default", "applied": True, "ckpt": "c", "ckpt_sha256": "s",
          "segments_fallback": [{"start": 10, "end": 20, "reasons": ["jerk"]}],
          "whole_clip_fallback": False, "handface_fallback": False,
          "residual_mean_abs": {"root": 0.0, "body": 1.0, "hands": 0.0, "face": 0.0},
          "raw_residual_mean_abs": {"root": 0.0, "body": 2.0, "hands": 0.0, "face": 0.0}}
    r2 = dict(r1, segments_fallback=[{"start": 0, "end": 5, "reasons": ["slide"]}],
              residual_mean_abs={"root": 0.0, "body": 3.0, "hands": 0.0, "face": 0.0})
    fb = LV.merge_feedback([r1, r2], [0, 270], [300, 300], "default")
    assert [w["start"] for w in fb["segments_fallback"]] == [10, 270]
    assert abs(fb["residual_mean_abs"]["body"] - 2.0) < 1e-9
    assert fb["ckpt_mismatch"] is False
    bad = dict(r2, ckpt_sha256="other")
    m = LV.merge_feedback([r1, bad], [0, 270], [300, 300], "default")
    assert m["ckpt_mismatch"] is True and m["ckpt_sha256"] is None


def _fake_track(n):
    world = {"toe_l": np.zeros((n, 3)), "toe_r": np.zeros((n, 3)),
             "pelvis": np.zeros((n, 3))}
    world["toe_l"][:, 0] = 0.01 * np.arange(n)
    world["toe_r"][:, 0] = 0.01 * np.arange(n)
    return {"world": world, "world_q": {"pelvis": np.tile(
        [0.0, 0.0, 0.0, 1.0], (n, 1))}}


def test_boundary_report_smooth_and_jump():
    n, overlap = 240, 30
    vmd = make_vmd(n)
    rep = LV.boundary_report(_fake_track(n), _fake_track(n), vmd, PROFILE,
                             [120], overlap, n)
    assert rep["enabled"] and rep["seams"]
    assert rep["seams"][0]["ok"], rep["seams"][0]
    vmd2 = make_vmd(n)
    for i in range(120, n):
        vmd2["bones"]["上半身"][i]["rot"] = q_axis([0, 0, 1], 90.0)
    rep2 = LV.boundary_report(_fake_track(n), _fake_track(n), vmd2, PROFILE,
                              [120], overlap, n)
    assert not rep2["seams"][0]["checks"]["jump"], rep2["seams"][0]
    assert not rep2["ok"]
    rep3 = LV.boundary_report(_fake_track(n), _fake_track(n), vmd2, PROFILE,
                              [120], overlap, n, source_max_step_deg=[95.0])
    assert rep3["seams"][0]["checks"]["jump"], rep3["seams"][0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print(f"{len(tests)} long-video tests passed")
