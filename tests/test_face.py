"""Unit tests for src/vtv/face.py (M3).

Run:  python tests/test_face.py
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.face import (  # noqa: E402
    LEVEL_L0, LEVEL_L1, LEVEL_L2, LEVEL_L3, SEMANTIC, SEM_INDEX,
    FaceTrackConfig, MorphMap, blendshape_weights, build_face_track,
    compress_morph_keys, geometry_weights, procedural_blinks, weights_to_morphs,
)

PROFILES = {k: json.load(open(os.path.join(ROOT, "profiles", f"{k}.json"),
                               encoding="utf-8"))
            for k in ("si", "luoxi", "tiflos")}


def test_morph_map_covers_all_profiles():
    for key, prof in PROFILES.items():
        mm = MorphMap.from_profile(prof)
        assert not mm.missing, (key, mm.missing)
        existing = {m["name"] for m in prof["morphs"]}
        for sem, name in mm.available.items():
            assert name in existing, (key, sem, name)
        assert mm.available["blink"] == "まばたき"
        assert mm.available["mouth_a"] == "あ"
        # every model has at least one eye-closure and one mouth shape
        assert mm.available["wink_l"] != mm.available["wink_r"]


def test_blendshape_mapping():
    bs = {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}
    w = blendshape_weights(bs)
    assert w[SEM_INDEX["blink"]] > 0.8
    assert w[SEM_INDEX["wink_l"]] == 0 and w[SEM_INDEX["wink_r"]] == 0
    bs = {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.0}
    w = blendshape_weights(bs)
    assert w[SEM_INDEX["wink_l"]] > 0.8
    assert w[SEM_INDEX["blink"]] < 0.2
    bs = {"jawOpen": 0.8}
    w = blendshape_weights(bs)
    assert 0.6 < w[SEM_INDEX["mouth_a"]] < 0.75
    bs = {"mouthSmileLeft": 0.6, "mouthSmileRight": 0.6}
    w = blendshape_weights(bs)
    assert w[SEM_INDEX["smile"]] > 0.7
    bs = {"browDownLeft": 0.7, "browDownRight": 0.7}
    w = blendshape_weights(bs)
    assert w[SEM_INDEX["angry"]] > 0.5
    # out-of-range inputs are clamped
    w = blendshape_weights({"jawOpen": 5.0, "eyeBlinkLeft": -3.0})
    assert 0 <= w.min() and w.max() <= 1.0


def test_geometry_weights():
    base = {"ear_l": 0.30, "ear_r": 0.30}
    w = geometry_weights({"ear_l": 0.10, "ear_r": 0.10, "mar": 0.0}, base)
    assert w[SEM_INDEX["blink"]] >= 0.79
    # one-sided closure: default = soft bilateral blink (no visible wink)
    w = geometry_weights({"ear_l": 0.10, "ear_r": 0.30, "mar": 0.0}, base)
    assert w[SEM_INDEX["wink_l"]] == 0
    assert w[SEM_INDEX["blink"]] > 0.4
    # explicit wink mode keeps the wink channel for L0-style input
    w = geometry_weights({"ear_l": 0.10, "ear_r": 0.30, "mar": 0.0}, base,
                         allow_wink=True)
    assert w[SEM_INDEX["wink_l"]] > 0.5
    w = geometry_weights({"ear_l": 0.30, "ear_r": 0.30, "mar": 0.18}, base, mar_open=0.18)
    assert w[SEM_INDEX["mouth_a"]] > 0.9
    w = geometry_weights({"ear_l": 0.30, "ear_r": 0.30, "mar": 0.0}, base)
    assert w.max() < 1e-9


def test_procedural_blinks_deterministic_and_sparse():
    cfg = FaceTrackConfig()
    a = procedural_blinks(242, cfg)
    b = procedural_blinks(242, cfg)
    assert np.array_equal(a, b)
    assert a.max() <= 1.0 and a.min() >= 0.0
    n_blinks = int((a > 0.5).sum())
    assert n_blinks >= 1, a.sum()
    occupied = np.zeros(242, bool)
    occupied[40:60] = True
    c = procedural_blinks(242, cfg, occupied)
    assert c[40:60].max() < 1e-9


def _landmarks(n=478, shift=0.0):
    p = np.zeros((n, 2), float)
    # eye corners/eyelids and mouth points used by geometry/step checks
    p[33] = [100, 100]
    p[133] = [120, 100]
    p[159] = [110, 96]
    p[145] = [110, 104]
    p[362] = [200, 100]
    p[263] = [220, 100]
    p[386] = [210, 96]
    p[374] = [210, 104]
    p[61] = [140, 150]
    p[291] = [180, 150]
    p[13] = [160, 146]
    p[14] = [160, 154]
    p += shift
    return p


def _rec(blink=0.0, jaw=0.0, patch_ok=True, bbox_ok=True, shift=0.0):
    return {
        "lm": _landmarks(shift=shift),
        "blendshapes": {"eyeBlinkLeft": blink, "eyeBlinkRight": blink,
                        "jawOpen": jaw},
        "geometry": {"ear_l": 0.30, "ear_r": 0.30, "mar": 0.02},
        "patch_ok": patch_ok, "bbox_ok": bbox_ok,
    }


def test_face_track_levels_and_smoothing():
    cfg = FaceTrackConfig()
    n = 20
    recs = [None] * n
    for i in range(5, 11):
        recs[i] = _rec()
    w, levels, diag = build_face_track(recs, n, cfg)
    assert w.shape == (n, len(SEMANTIC))
    assert levels[5] in (LEVEL_L0, LEVEL_L1) and levels[10] in (LEVEL_L0, LEVEL_L1)
    assert levels[0] == LEVEL_L3
    assert np.isfinite(w).all()
    assert w.max() <= 1.0
    for i in range(1, n):
        d = np.abs(w[i] - w[i - 1])
        assert d.max() <= cfg.blink_max_step + 1e-6, (i, d.max())
    # no detection at all + a procedural blink somewhere
    w2, levels2, diag2 = build_face_track([None] * n, n, cfg)
    assert (levels2 == LEVEL_L3).all()
    assert diag2["level_counts"][3] == n
    assert w2.max() == 0.0 or w2[:, SEM_INDEX["blink"]].max() > 0


def test_face_rejects_unstable_segment():
    cfg = FaceTrackConfig()
    n = 10
    recs = []
    for i in range(n):
        recs.append(_rec(shift=(i % 2) * 40.0))  # 40 px jumps
    w, levels, diag = build_face_track(recs, n, cfg)
    assert all(l >= LEVEL_L2 for l in levels), levels
    assert diag["segments"][0]["level"] == "rejected"


def test_blendshape_hold_is_replaced_by_procedural():
    cfg = FaceTrackConfig()
    n = 40
    recs = [_rec(blink=1.0) for _ in range(n)]
    w, levels, diag = build_face_track(recs, n, cfg)
    assert "blink_hold" in diag["unreliable"], diag["unreliable"]
    # after the replacement the blink track is no longer stuck at 1.0
    assert w[:, SEM_INDEX["blink"]].max() <= 1.0
    assert int((w[:, SEM_INDEX["blink"]] > 0.9).sum()) < 20


def test_compress_morph_keys():
    w = np.zeros(10)
    w[3:6] = [0.2, 0.8, 0.4]
    keys = compress_morph_keys(w)
    assert keys[0][1] > 0 and keys[-1][1] == 0.0
    assert all(0 <= f < 10 for f, _ in keys)
    assert all(0.0 <= x <= 1.0 for _, x in keys)
    assert keys == sorted(keys)


def test_weights_to_morphs_caps():
    prof = PROFILES["tiflos"]
    mm = MorphMap.from_profile(prof)
    n = 5
    w = np.zeros((n, len(SEMANTIC)))
    for i in range(len(SEMANTIC)):
        w[:, i] = 1.0
    out = weights_to_morphs(w, mm)
    existing = {m["name"] for m in prof["morphs"]}
    assert set(out) <= existing
    for name, arr in out.items():
        assert len(arr) == n
        assert arr.min() >= 0.0 and arr.max() <= 1.0
    # at most max_active_morphs non-zero at each frame
    for i in range(n):
        active = sum(1 for arr in out.values() if arr[i] > 0)
        assert active <= FaceTrackConfig().max_active_morphs, (i, active)
    # all morphs that cannot be mapped are absent, not invented
    for name in mm.missing:
        assert name not in out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"test_face: OK ({len(tests)} tests)")
