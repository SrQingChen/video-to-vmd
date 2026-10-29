"""Unit tests for the M4 pipeline cache, manifest, QC summary and helpers.

Run:  python tests/test_pipeline.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv import pipeline as P  # noqa: E402

CFG = json.load(open(os.path.join(ROOT, "configs", "pipeline.json"),
                     encoding="utf-8"))
def _tmp():
    return Path(tempfile.mkdtemp(prefix="vtv_m4_test_"))


def _pipeline(tmp: Path, **kw):
    video = tmp / "v.mp4"
    video.write_bytes(b"not a real video")
    return P.Pipeline(video=video, out_dir=tmp / "run", models=["si"],
                      config=CFG, feedback_mode=kw.pop("feedback_mode", "off"),
                      render=False, resume=kw.pop("resume", True), **kw)


def test_hash_and_fingerprint():
    tmp = _tmp()
    try:
        a = tmp / "a.txt"
        a.write_text("one")
        f1 = P.fingerprint(a)
        f2 = P.fingerprint(a)
        assert f1 == f2 and len(f1["sha256"]) == 64
        a.write_text("two")
        assert P.fingerprint(a)["sha256"] != f1["sha256"]
        missing = tmp / "nope"
        try:
            P.fingerprint(missing)
            raise AssertionError("fingerprint should fail on missing input")
        except FileNotFoundError:
            pass
        assert len(P.code_fingerprint()) == 16
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_manifest_cache_and_invalidation():
    tmp = _tmp()
    try:
        inp = tmp / "in.txt"
        inp.write_text("x")
        out = tmp / "out.txt"
        m = P.Manifest(tmp / "manifest.json", resume=True)
        ins = {"input": P.fingerprint(inp)}
        conf = {"a": 1}
        assert not m.is_valid("s", ins, conf, [out])[0]
        out.write_text("ok")
        m.write_ok("s", ins, conf, [out], 1.25)
        m2 = P.Manifest(tmp / "manifest.json", resume=True)
        ok, why = m2.is_valid("s", ins, conf, [out])
        assert ok, why
        # input change invalidates
        inp.write_text("y")
        ins2 = {"input": P.fingerprint(inp)}
        assert not m2.is_valid("s", ins2, conf, [out])[0]
        # config change invalidates
        assert not m2.is_valid("s", ins, {"a": 2}, [out])[0]
        # missing output invalidates
        out.unlink()
        assert not m2.is_valid("s", ins, conf, [out])[0]
        out.write_text("ok")
        # resume off always re-runs
        m3 = P.Manifest(tmp / "manifest.json", resume=False)
        assert not m3.is_valid("s", ins, conf, [out])[0]
        # record stores seconds and output hashes
        rec = m3.get("s")
        assert rec["seconds"] == 1.25 and rec["outputs"][0]["sha256"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stage_caching_and_failure_record():
    tmp = _tmp()
    try:
        pipe = _pipeline(tmp)
        pipe._prepare_dirs()
        calls = {"n": 0}
        out = pipe.run_dir / "artifact.txt"

        def fn():
            calls["n"] += 1
            out.write_text("done")

        # first run executes, second is cached
        assert pipe._run_stage("demo", fn, {}, {"x": 1}, [out]) == "ok"
        assert calls["n"] == 1
        assert pipe._run_stage("demo", fn, {}, {"x": 1}, [out]) == "cached"
        assert calls["n"] == 1
        # forced stage re-runs
        pipe.force = {"demo"}
        assert pipe._run_stage("demo", fn, {}, {"x": 1}, [out]) == "ok"
        assert calls["n"] == 2
        # failure is recorded, not silently skipped
        pipe.force = set()

        def boom():
            raise RuntimeError("simulated failure")

        try:
            pipe._run_stage("bad", boom, {}, {}, [pipe.run_dir / "never.txt"])
            raise AssertionError("stage failure must raise")
        except P.StageError:
            pass
        rec = pipe.manifest.get("bad")
        assert rec["status"] == "failed" and "simulated failure" in rec["error"]
        # a failed stage is not cached
        assert not pipe.manifest.is_valid(
            "bad", {}, {}, [pipe.run_dir / "never.txt"])[0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stage_filter_and_force_names():
    tmp = _tmp()
    try:
        pipe = _pipeline(tmp, stages=["feedback"])
        assert pipe._selected("si:feedback")
        assert pipe._selected("feedback")
        assert not pipe._selected("si:adapter")
        pipe.force = {"feedback"}
        assert pipe._forced("si:feedback")
        assert not pipe._forced("si:adapter")
        pipe.force = {"all"}
        assert pipe._forced("anything")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_render_frames_clamped():
    tmp = _tmp()
    try:
        pipe = _pipeline(tmp)
        assert pipe._render_frames(242) == (70, 230)
        # a short clip cannot render past its last frame
        a, b = pipe._render_frames(100)
        assert a == 70 and b == 100
        a, b = pipe._render_frames(40)
        assert b <= 40 and a >= 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _fake_qc(model, ok=True, fallback=0, warnings=None):
    return {
        "model": model, "frames": 100, "ok": ok,
        "certificate": {"ok": True}, "change_audit": {"ok": True},
        "feedback": {"mode": "default", "applied": True,
                     "segments_fallback": [{"start": 0, "end": 29}] * fallback},
        "stability": {
            "before": {"jerk_rms": 100.0, "foot_slide": 0.1,
                       "foot_penetration": 0.0, "finger_step_p95": 2.0,
                       "morph_step_max": 0.4},
            "after": {"jerk_rms": 104.0, "foot_slide": 0.1,
                      "foot_penetration": 0.0, "finger_step_p95": 2.0,
                      "morph_step_max": 0.4},
        },
        "secondary": {"available": False, "heldout_reference": {
            "path": "outputs/m35/eval/report_default.json", "task_gate": True,
            "gates_ok": False, "secondary": {"world_angle_not_worse": False,
                                             "position_not_worse": False},
            "clips": 9, "frames": 4860,
            "body_relative_main_before": 37.55,
            "body_relative_main_after": 27.38,
            "world_angle_delta_deg": 1.65,
            "position_delta_pct_height": 0.141}},
        "render": {"ok": True},
        "warnings": warnings or [],
    }


def test_summary_builder():
    tmp = _tmp()
    try:
        run = tmp / "run"
        for name, ok, fb in (("si", True, 0), ("luoxi", True, 2),
                             ("tiflos", False, 0)):
            d = run / "models" / name
            d.mkdir(parents=True)
            P.write_json(d / "qc.json", _fake_qc(name, ok=ok, fallback=fb,
                                                 warnings=[] if ok else ["x"]))
        (run / "qc").mkdir()
        summary = P.build_run_summary(run)
        assert summary["gates"]["task_gate"] is False
        assert summary["gates"]["render"] is True
        # held-out secondary is false and stays visible, not an overall pass
        assert summary["gates"]["secondary_ok"] is False
        assert "luoxi" in summary["fallback_segments"]
        assert (run / "qc" / "summary.json").exists()
        md = (run / "qc" / "summary.md").read_text(encoding="utf-8")
        assert "Task gate: FAIL" in md and "luoxi" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_seed_config_uses_hand_section():
    tmp = _tmp()
    try:
        cfg = json.loads(json.dumps(CFG))
        cfg["hand"]["conf"] = 0.33
        pipe = _pipeline(tmp, feedback_mode="off")
        pipe.cfg = cfg
        assert pipe._seed_config("hands")["conf"] == 0.33
        assert "conf" in pipe._seed_config("face")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_meta_and_manifest_json():
    tmp = _tmp()
    try:
        pipe = _pipeline(tmp, feedback_mode="off")
        P.write_json(pipe.run_dir / "manifest.json",
                     {"version": 1, "run": {}, "stages": {}})
        m = P.Manifest(pipe.run_dir / "manifest.json")
        m.data["run"] = {"video": "x", "models": ["si"]}
        m.save()
        assert P.load_json(pipe.run_dir / "manifest.json")["run"]["video"] == "x"
        # JSON helper must survive numpy scalars/arrays
        P.write_json(pipe.run_dir / "n.json",
                     {"a": np.float32(1.5), "b": np.array([1, 2])})
        d = P.load_json(pipe.run_dir / "n.json")
        assert d["a"] == 1.5 and d["b"] == [1, 2]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print(f"{len(tests)} pipeline tests passed")
