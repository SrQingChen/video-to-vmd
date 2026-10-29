# -*- coding: utf-8 -*-
"""M4 one-command pipeline: staged execution, caching, resume, QC.

Stage graph (a run directory holds every artifact and its provenance):

    video
      gvhmr                     scripts/run_gvhmr.py (30 fps, static camera default)
      hands / face              MediaPipe evidence (shared by all models)
      <model>:adapter           run_adapter.py (GVHMR -> target PMX)
      <model>:cleanup           selective cleanup (M2)
      <model>:contact_ik        contact-gated leg IK (M2.2)
      <model>:assemble          body + M3 fingers + morphs -> before.vmd
      <model>:feedback          M3.5 conservative residual -> after/final.vmd
      <model>:render_before     fixed-camera 2K render of before.vmd
      <model>:render_after      same camera, final.vmd
      <model>:qc                certificate + stability + audit + secondary

Every stage writes a manifest record with input hashes, config, output hashes
and timing.  ``--resume`` (default) skips a stage only when its inputs and
config are unchanged and all outputs still exist; ``--force`` re-runs named
stages and lets their changed outputs invalidate downstream stages naturally.

The default feedback is the M3.5 conservative profile (torso .85 / arm .2 /
smooth 3; hands, face and root off).  It never touches legs, feet, root, hands
or morphs, and the final VMD must keep 53 tracks + IK off.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np

from . import feedback_data as FD
from . import feedback_apply as FA
from . import longvideo as LV
from . import qc
from .cleanup import clean_vmd
from .contact_ik import solve_contact_ik
from .metrics.core import pose_track
from .mmd.pmx import parse_pmx
from .mmd.vmd import parse_vmd
from .mmd.writer import write_vmd

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

# Code fingerprint: a cache entry must be invalidated when the stage
# implementation changes, not only when its data inputs do.
_CODE_FILES = (
    "src/vtv/pipeline.py", "src/vtv/feedback_apply.py", "src/vtv/qc.py",
    "src/vtv/feedback_data.py", "src/vtv/eval_helpers.py",
    "src/vtv/cleanup.py", "src/vtv/contact_ik.py", "src/vtv/adapter.py",
    "src/vtv/canonical_body.py", "src/vtv/hands.py", "src/vtv/face.py",
    "src/vtv/longvideo.py", "src/vtv/framing.py", "src/vtv/cli.py",
    "src/vtv/mmd/pmx.py", "src/vtv/mmd/vmd.py", "src/vtv/mmd/writer.py",
    "src/vtv/mmd/animation.py", "src/vtv/mmd/evaluate.py",
    "scripts/run_gvhmr.py", "scripts/run_adapter.py", "scripts/run_hand.py",
    "scripts/run_face.py", "scripts/render_synth.py", "scripts/run_vtv.py",
)


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for rel in _CODE_FILES:
        p = ROOT / rel
        h.update(rel.encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode() if p.exists() else b"missing")
    return h.hexdigest()[:16]


class StageError(RuntimeError):
    def __init__(self, key: str, message: str, log: Optional[Path] = None):
        super().__init__(f"[{key}] {message}" + (f" (log: {log})" if log else ""))
        self.key = key
        self.log = str(log) if log else None


# ---------------------------------------------------------------------------
# small io helpers
# ---------------------------------------------------------------------------
def load_json(path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _norm(o):
    """Round-trip through JSON so tuples/numpy types compare stably."""
    return json.loads(json.dumps(o, default=_json_default, ensure_ascii=False))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_norm(obj), f, ensure_ascii=False, indent=1)
    tmp.replace(path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


CODE_SHA = code_fingerprint()


def fingerprint(path) -> Dict:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"missing input: {p}")
    st = p.stat()
    # Large media (PMX/blend/PMX videos) are fingerprinted by size+mtime: the
    # pipeline invalidates on edit without hashing tens of megabytes each run.
    if st.st_size <= 16 * 1024 * 1024:
        return {"path": str(p), "sha256": sha256_file(p), "size": st.st_size}
    return {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _fmt_secs(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m{s % 60:04.1f}s"


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
class Manifest:
    """Per-run stage journal used for caching, resume and reporting."""

    def __init__(self, path, resume: bool = True):
        self.path = Path(path)
        self.resume = bool(resume)
        if self.path.exists():
            try:
                self.data = load_json(self.path)
            except Exception:
                self.data = {"version": 1, "stages": {}, "run": {}}
        else:
            self.data = {"version": 1, "stages": {}, "run": {}}
        self.data.setdefault("stages", {})
        self.data.setdefault("run", {})

    def save(self):
        write_json(self.path, self.data)

    def get(self, key):
        return self.data["stages"].get(key)

    def is_valid(self, key: str, inputs: Mapping, config: Mapping,
                 outputs: Iterable) -> tuple:
        if not self.resume:
            return False, "resume disabled"
        rec = self.get(key)
        if not rec:
            return False, "no record"
        if rec.get("status") != "ok":
            return False, f"last status={rec.get('status')}"
        if rec.get("config") != _norm(config):
            return False, "config changed"
        if rec.get("inputs") != _norm(inputs):
            return False, "inputs changed"
        for o in outputs:
            if not Path(o).exists():
                return False, f"missing output {o}"
        return True, ""

    def write_ok(self, key: str, inputs: Mapping, config: Mapping,
                 outputs: Iterable, seconds: float, extra: Optional[Dict] = None):
        rec = {
            "status": "ok",
            "inputs": _norm(inputs),
            "config": _norm(config),
            "outputs": [fingerprint(o) for o in outputs],
            "seconds": round(float(seconds), 2),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if extra:
            rec.update(_norm(extra))
        self.data["stages"][key] = rec
        self.save()


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, video, out_dir, models: List[str], config: Mapping,
                 feedback_mode: str = "default",
                 ckpt: Optional[str] = None,
                 slow_factor: float = 1.0,
                 render: bool = True,
                 stages: Optional[List[str]] = None,
                 force: Optional[Iterable[str]] = None,
                 resume: bool = True,
                 seeds: Optional[Mapping[str, str]] = None,
                 gt: Optional[Mapping[str, str]] = None,
                 keep_going: bool = False,
                 chunk_frames: Optional[int] = None,
                 overlap_frames: int = LV.DEFAULT_OVERLAP_FRAMES,
                 force_chunks: Optional[Iterable[int]] = None):
        self.video = Path(video).resolve()
        if not self.video.exists():
            raise FileNotFoundError(f"video not found: {self.video}")
        self.run_dir = Path(out_dir).resolve()
        self.models = [m.strip() for m in models if m.strip()]
        self.cfg = config
        self.feedback_mode = feedback_mode
        fb = config["feedback"]
        if ckpt:
            self.ckpt = ckpt
        elif feedback_mode in ("v2", "v2_full"):
            self.ckpt = fb.get("ckpt_v2") or fb["ckpt"]
        else:
            self.ckpt = fb["ckpt"]
        self.slow_factor = float(slow_factor)
        self.do_render = bool(render)
        self.stages = set(stages or [])
        self.force = set(force or [])
        self.keep_going = bool(keep_going)
        self.gt = {k: str((ROOT / v).resolve() if not Path(v).is_absolute() else Path(v))
                   for k, v in (gt or {}).items()}
        self.seeds = dict(seeds or {})
        self.manifest = Manifest(self.run_dir / "manifest.json", resume=resume)
        self.errors: List[Dict] = []
        self._fb_cache = None
        self.summary: Dict = {}
        self.t0 = time.time()
        self.chunk_frames = int(chunk_frames) if chunk_frames else 0
        self.overlap_frames = int(overlap_frames)
        self.force_chunks = {int(x) for x in (force_chunks or [])}
        self.chunk_plan: Optional[Dict] = None
        self.boundary_frames: List[int] = []

        if self.chunk_frames:
            if self.seeds:
                raise ValueError("--seed is not supported with chunked runs "
                                 "(chunks have their own GVHMR caches)")
            if self.gt:
                raise ValueError("--gt is not supported with chunked runs")
            if self.overlap_frames < 0 or self.overlap_frames >= self.chunk_frames:
                raise ValueError("overlap must be < chunk")

        self.models_cfg = load_json(ROOT / "configs/models.json")["models"]
        for m in self.models:
            if m not in self.models_cfg:
                raise ValueError(f"unknown model {m!r}; have {sorted(self.models_cfg)}")
        self.heldout = self._load_heldout()

    # -- layout ------------------------------------------------------------
    def _prepare_dirs(self):
        self.gvhmr_dir = self.run_dir / "video" / "gvhmr"
        self.hand_dir = self.run_dir / "evidence" / "hand"
        self.face_dir = self.run_dir / "evidence" / "face"
        self.qc_dir = self.run_dir / "qc"
        self.log_dir = self.run_dir / "logs"
        for d in (self.gvhmr_dir, self.hand_dir, self.face_dir, self.qc_dir,
                  self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        for m in self.models:
            self.model_dir(m).mkdir(parents=True, exist_ok=True)

    def model_dir(self, model: str) -> Path:
        return self.run_dir / "models" / model

    def profile(self, model: str) -> Dict:
        p = ROOT / "profiles" / f"{model}.json"
        if not p.exists():
            raise StageError(
                "profile", f"missing {p}; generate it with scripts/make_profiles.py "
                "after placing the PMX models")
        return load_json(p)

    def model_pmx(self, model: str) -> Dict:
        return parse_pmx(self.models_cfg[model]["pmx"])

    # -- logging / subprocess ---------------------------------------------
    def log(self, msg: str):
        print(msg, flush=True)

    def _run_cmd(self, key: str, cmd: List[str], retries: int = 0):
        log = self.log_dir / (key.replace(":", "_") + ".log")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        last = None
        for attempt in range(retries + 1):
            with open(log, "w", encoding="utf-8") as f:
                f.write("CMD " + " ".join(cmd) + "\n\n")
                f.flush()
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                   cwd=str(ROOT), env=env)
            if r.returncode == 0:
                return
            last = f"exit {r.returncode}"
            if attempt < retries:
                self.log(f"[{key}] {last}, retry {attempt + 1}/{retries}")
                time.sleep(2)
        raise StageError(key, f"command failed ({last})", log)

    # -- stage runner ------------------------------------------------------
    def _forced(self, key: str) -> bool:
        cand = {key, key.split(":")[-1]}
        if key.startswith("chunk:"):
            cand.add("chunks")
        if key.endswith(":merge"):
            cand.add("merge")
        return bool("all" in self.force or (cand & self.force))

    def _selected(self, key: str) -> bool:
        if not self.stages:
            return True
        cand = {key, key.split(":")[-1]}
        if key.startswith("chunk:"):
            cand.add("chunks")
        if key.endswith(":merge"):
            cand.add("merge")
        return bool(cand & self.stages)

    def _hash_inputs(self, inputs: Mapping) -> Dict:
        return {k: fingerprint(v) for k, v in inputs.items() if v is not None}

    def _run_stage(self, key: str, fn: Callable, inputs: Mapping,
                   config: Mapping, outputs: Iterable, retries: int = 0,
                   force_override: bool = False):
        outputs = [Path(o) for o in outputs]
        selected = self._selected(key)
        ins = self._hash_inputs(inputs)
        conf = _norm(dict(config, pipeline_code=CODE_SHA))
        valid, why = self.manifest.is_valid(key, ins, conf, outputs)
        if valid and not self._forced(key) and not force_override:
            self.log(f"[{key}] cached ({_fmt_secs(self.manifest.get(key)['seconds'])})")
            return "cached"
        if not selected:
            self.log(f"[{key}] skipped (stage filter); needs re-run: {why}")
            return "skipped"
        self.log(f"[{key}] running" + (f" ({why})" if why else ""))
        t0 = time.time()
        for attempt in range(retries + 1):
            try:
                fn()
                missing = [str(o) for o in outputs if not o.exists()]
                if missing:
                    raise StageError(key, f"stage finished but outputs are missing: {missing}")
                break
            except Exception as e:  # noqa: BLE001 - recorded for QC/diagnosis
                if attempt >= retries:
                    self.manifest.data["stages"][key] = {
                        "status": "failed", "error": str(e),
                        "traceback": traceback.format_exc(),
                        "inputs": ins, "config": conf,
                        "seconds": round(time.time() - t0, 2),
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    self.manifest.save()
                    raise StageError(key, str(e)) from e
                self.log(f"[{key}] failed ({e}); retry {attempt + 1}/{retries}")
                time.sleep(2)
        sec = time.time() - t0
        self.manifest.write_ok(key, ins, conf, outputs, sec)
        self.log(f"[{key}] ok ({_fmt_secs(sec)})")
        return "ok"

    # -- held-out reference -------------------------------------------------
    def _load_heldout(self) -> Optional[Dict]:
        qc_cfg = self.cfg.get("qc") or {}
        ref = (qc_cfg.get("heldout_reference_v2")
               if self.feedback_mode in ("v2", "v2_full")
               else qc_cfg.get("heldout_reference"))
        if not ref:
            return None
        p = ROOT / ref
        if not p.exists():
            return None
        try:
            d = load_json(p)
            g = d.get("gates", {})
            t = d.get("aggregate", {}).get("test") or {}
            if not t:
                return None
            return {
                "path": str(ref),
                "ckpt": d.get("ckpt"),
                "tag": d.get("tag"),
                "task_gate": bool(g.get("task_gate")),
                "gates_ok": bool(g.get("ok")),
                "checks": g.get("checks"),
                "secondary": g.get("secondary"),
                "clips": t.get("clips"),
                "frames": t.get("frames"),
                "body_relative_main_before": t["baseline"]["body_relative_main_deg"],
                "body_relative_main_after": t["corrected"]["body_relative_main_deg"],
                "world_angle_delta_deg": t["delta"]["world_angle_mean_deg"],
                "position_delta_pct_height": t["delta"]["position_pct_height"],
                "jerk_delta": t["delta"]["jerk_rms_ratio"],
            }
        except Exception:
            return None

    # -- seeds -------------------------------------------------------------
    def _apply_seeds(self):
        """Copy a previous stage's output dir into the cache with provenance."""
        if not self.seeds:
            return
        table = {
            "gvhmr": (self.gvhmr_dir,
                      [self.gvhmr_dir / "gvhmr_result.npz",
                       self.gvhmr_dir / "gvhmr_meta.json",
                       self.gvhmr_dir / "0_input_video.mp4",
                       self.gvhmr_dir / "preprocess/vitpose.pt"]),
            "hands": (self.hand_dir,
                      [self.hand_dir / "hand.npz", self.hand_dir / "hand.json"]),
            "face": (self.face_dir,
                     [self.face_dir / "face.npz", self.face_dir / "face.json"]),
        }
        for name in [n for n in ("gvhmr", "hands", "face") if n in self.seeds]:
            src = self.seeds[name]
            if name not in table:
                raise ValueError(f"--seed {name}: expected one of {sorted(table)}")
            dst, outputs = table[name]
            src = Path(src).resolve()
            if not src.exists():
                raise FileNotFoundError(f"seed dir not found: {src}")
            dst_r = dst.resolve()
            if src == dst_r or src in dst_r.parents or dst_r in src.parents:
                raise StageError(
                    f"seed:{name}",
                    f"seed source {src} overlaps the run cache {dst_r}; "
                    "point --seed at a different run directory")
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            self.log(f"[seed {name}] copied {src} -> {dst}")
            # A seed is only trusted for the *same* input video.  Provenance is
            # written now; the normal stage validation compares hashes later.
            if name == "gvhmr":
                meta = load_json(dst / "gvhmr_meta.json")
                src_video = Path(meta.get("source_video", "")).resolve()
                if src_video != self.video:
                    raise StageError(
                        "seed:gvhmr",
                        f"seeded GVHMR was run on {src_video}, not {self.video}")
            if not all(o.exists() for o in outputs):
                raise StageError(f"seed:{name}", "seed is missing expected outputs")
            self.manifest.write_ok(
                name, self._hash_inputs(self._seed_inputs(name)),
                self._seed_config(name), outputs, 0.0,
                {"seeded_from": str(src)})

    def _seed_inputs(self, name: str) -> Dict:
        if name == "gvhmr":
            return {"video": self.video}
        if name == "hands":
            return {"npz": self.gvhmr_dir / "gvhmr_result.npz",
                    "video": self.gvhmr_dir / "0_input_video.mp4",
                    "kp": self.gvhmr_dir / "preprocess/vitpose.pt"}
        return {"npz": self.gvhmr_dir / "gvhmr_result.npz",
                "video": self.gvhmr_dir / "0_input_video.mp4",
                "kp": self.gvhmr_dir / "preprocess/vitpose.pt"}

    def _seed_config(self, name: str) -> Dict:
        if name == "gvhmr":
            base = {"static_cam": bool((self.cfg.get("gvhmr") or {}).get(
                "static_cam", True))}
        else:
            base = dict(self.cfg.get("hand" if name == "hands" else "face") or {})
        return dict(base, pipeline_code=CODE_SHA)

    # -- tools -------------------------------------------------------------
    def _render_cfg(self) -> Dict:
        return dict(self.cfg.get("render") or {})

    def _blender(self) -> str:
        return os.environ.get("VTV_BLENDER") or str(
            self._render_cfg().get("blender", "blender"))

    def _ffmpeg(self) -> str:
        return os.environ.get("VTV_FFMPEG") or str(
            self._render_cfg().get("ffmpeg", "ffmpeg"))

    def _blend_path(self, model: str) -> Path:
        env = os.environ.get("VTV_BLEND")
        if env:
            return Path(env)
        return self._render_blend(model)

    # -- long video (M5) ----------------------------------------------------
    def chunk_dir(self, index: int) -> Path:
        return self.run_dir / "chunks" / f"seg_{int(index):03d}"

    def _source_frames_30fps(self) -> int:
        """Cheap CFR estimate used only to decide whether to chunk."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(self.video))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps <= 0 or n <= 0:
                return 0
            return int(round(n * 30.0 / fps))
        except Exception:
            return 0

    def _use_chunking(self) -> bool:
        if not self.chunk_frames:
            return False
        n = self._source_frames_30fps()
        # if the estimate is unavailable, let the video30 stage decide
        return n == 0 or n > self.chunk_frames

    def _stage_video30(self):
        out = self.run_dir / "video" / "30fps.mp4"
        r = LV.make_30fps_video(self.video, out, ffmpeg=self._ffmpeg(), fps=30,
                                log=self.log_dir / "video30.log")
        r["source"] = str(self.video)
        r["source_sha256"] = sha256_file(self.video) if (
            Path(self.video).stat().st_size <= 16 * 1024 * 1024) else None
        write_json(self.run_dir / "video" / "30fps.json", r)

    def _prepare_chunk_inputs(self, plan: Mapping, video30: Path) -> Dict:
        fp = fingerprint(video30)
        plan_path = self.run_dir / "chunk_plan.json"
        prev = load_json(plan_path) if plan_path.exists() else {}
        recut = (prev.get("video30_fingerprint") != _norm(fp)
                 or prev.get("starts") != plan["starts"]
                 or prev.get("ends") != plan["ends"])
        cuts = []
        for i, (s, e) in enumerate(zip(plan["starts"], plan["ends"])):
            cut = self.chunk_dir(i) / "input.mp4"
            want = int(e) - int(s)
            need = recut or not cut.exists()
            if not need:
                try:
                    need = LV.video_frames(cut) != want
                except Exception:
                    need = True
            if need:
                self.log(f"[chunk {i:03d}] cut {s}:{e} -> {cut}")
                LV.cut_frames(video30, cut, int(s), want,
                              ffmpeg=self._ffmpeg())
            cuts.append({"index": i, "start": int(s), "end": int(e),
                         "frames": want, "path": str(cut),
                         "sha256": sha256_file(cut)})
        out = dict(plan)
        out["video30"] = str(video30)
        out["video30_fingerprint"] = _norm(fp)
        out["cuts"] = cuts
        write_json(plan_path, out)
        self.chunk_plan = plan
        self.boundary_frames = [int(x) for x in plan["starts"][1:]]
        return out

    def _run_chunk(self, index: int, cut: Path) -> Dict:
        cfg = json.loads(json.dumps(self.cfg))
        cfg.setdefault("render", {})["enabled"] = False
        chunk_dir = self.chunk_dir(index)
        pipe = Pipeline(video=cut, out_dir=chunk_dir, models=self.models,
                        config=cfg, feedback_mode=self.feedback_mode,
                        ckpt=self.ckpt, slow_factor=self.slow_factor,
                        render=False, stages=None, force=self.force,
                        resume=self.manifest.resume,
                        keep_going=self.keep_going)
        return pipe.run()

    def _chunked_stages(self):
        if self.slow_factor != 1.0:
            raise StageError(
                "chunking",
                "chunked runs require --slow-factor 1.0 (per-chunk output "
                "lengths would no longer tile the timeline exactly); run the "
                "whole clip with --chunk-seconds 0 for slow-motion sources")
        v30 = self.run_dir / "video" / "30fps.mp4"
        v30_info = self.run_dir / "video" / "30fps.json"
        self._run_stage("video30", self._stage_video30, {"video": self.video},
                        {"fps": 30, "crf": LV.DEFAULT_CRF},
                        [v30, v30_info])
        if not v30_info.exists():
            raise StageError("chunking", "video30 was skipped and no 30fps "
                             "video exists; run without --stage filters first")
        info = load_json(v30_info)
        n30 = int(info["frames"])
        chunk_cfg = self.cfg.get("chunking") or {}
        min_tail = int(round(float(chunk_cfg.get("min_tail_seconds", 1.0)) * 30))
        plan = LV.plan_chunks(n30, self.chunk_frames, self.overlap_frames,
                              max(0, min_tail))
        if plan["n_chunks"] <= 1:
            self.log(f"[chunking] {n30} frames <= {self.chunk_frames}: "
                     "single clip path")
            self._global_stages()
            for model in self.models:
                self._run_model(model)
            return
        cuts = self._prepare_chunk_inputs(plan, v30)
        self.log(f"[chunking] {n30} frames -> {plan['n_chunks']} chunks "
                 f"(chunk {plan['chunk_frames']}, overlap "
                 f"{plan['overlap_frames']})")
        for i, c in enumerate(cuts["cuts"]):
            key = f"chunk:{i:03d}"
            outputs = [self.chunk_dir(i) / "models" / m / n
                       for m in self.models
                       for n in ("before.vmd", "final.vmd", "feedback.json")]
            self._run_stage(
                key, lambda i=i, cut=Path(c["path"]): self._run_chunk(i, cut),
                {"input": Path(c["path"]),
                 "plan": self.run_dir / "chunk_plan.json"},
                {"chunk": i, "start": c["start"], "end": c["end"],
                 "models": self.models, "feedback": self.feedback_mode,
                 "slow_factor": self.slow_factor,
                 "chunk_frames": plan["chunk_frames"],
                 "overlap_frames": plan["overlap_frames"]},
                outputs, force_override=(i in self.force_chunks))
        for model in self.models:
            try:
                self._merge_model_stage(model)
            except StageError as e:
                self.errors.append({"model": model, "error": str(e)})
                self.log(f"[{model}] merge FAILED: {e}")
                if not self.keep_going:
                    raise
        for model in self.models:
            try:
                self._model_quality_stages(model)
            except StageError as e:
                self.errors.append({"model": model, "error": str(e)})
                self.log(f"[{model}] FAILED: {e}")
                if not self.keep_going:
                    raise

    def _merge_model_stage(self, model: str):
        mdir = self.model_dir(model)
        plan_path = self.run_dir / "chunk_plan.json"
        plan = load_json(plan_path)
        n = int(plan["n_chunks"])
        O = int(plan["overlap_frames"])
        inputs = {"plan": plan_path, "profile": ROOT / "profiles" / f"{model}.json",
                  "pmx": Path(self.models_cfg[model]["pmx"])}
        for i in range(n):
            for name in ("before.vmd", "final.vmd", "feedback.json"):
                inputs[f"chunk{i}:{name}"] = self.chunk_dir(i) / "models" / model / name
        outputs = [mdir / "before.vmd", mdir / "final.vmd",
                   mdir / "feedback.json", mdir / "merge.json",
                   mdir / "before.cert.json"]
        self._run_stage(
            f"{model}:merge", lambda: self._stage_merge(model), inputs,
            {"model": model, "n_chunks": n, "overlap_frames": O,
             "chunk_frames": int(plan["chunk_frames"]),
             "frame_count": int(plan["n_frames"])},
            outputs)

    def _stage_merge(self, model: str):
        profile = self.profile(model)
        plan = load_json(self.run_dir / "chunk_plan.json")
        starts = [int(x) for x in plan["starts"]]
        ends = [int(x) for x in plan["ends"]]
        O = int(plan["overlap_frames"])
        mdir = self.model_dir(model)
        parts_b, parts_f, shifts, lengths = [], [], [], []
        for i, (s, e) in enumerate(zip(starts, ends)):
            cb = parse_vmd(str(self.chunk_dir(i) / "models" / model / "before.vmd"))
            cf = parse_vmd(str(self.chunk_dir(i) / "models" / model / "final.vmd"))
            L = int(cb["max_frame"]) + 1
            if int(cf["max_frame"]) + 1 != L:
                raise StageError(f"{model}:merge",
                                 f"chunk {i}: before/final frame counts differ")
            if L != e - s:
                raise StageError(f"{model}:merge",
                                 f"chunk {i}: {L} frames, expected {e - s}")
            if i == 0:
                sh = {"yaw_deg": 0.0, "translation": [0.0, 0.0, 0.0],
                      "note": "reference chunk (no shift)"}
            else:
                prev_L = lengths[i - 1]
                sh = LV.alignment(
                    parts_b[i - 1], cb, profile,
                    prev_frames=list(range(prev_L - O, prev_L)),
                    cur_frames=list(range(O)))
            aligned_b = LV.apply_alignment(cb, profile, sh["yaw_deg"],
                                           sh["translation"])
            aligned_f = LV.apply_alignment(cf, profile, sh["yaw_deg"],
                                           sh["translation"])
            parts_b.append(aligned_b)
            parts_f.append(aligned_f)
            shifts.append(sh)
            lengths.append(L)
        n_out = sum(lengths) - O * (len(lengths) - 1)
        if n_out != int(plan["n_frames"]):
            raise StageError(f"{model}:merge",
                             f"merged frames {n_out} != planned {plan['n_frames']}")
        merged_b, rep_b = LV.merge_vmds(parts_b, starts, n_out, O)
        merged_f, rep_f = LV.merge_vmds(parts_f, starts, n_out, O)
        for name, vmd in (("before.vmd", merged_b), ("final.vmd", merged_f)):
            write_vmd(str(mdir / name), vmd.get("model_name", ""), vmd["bones"],
                      vmd["morphs"], vmd["ik_frames"])
        reports = [load_json(self.chunk_dir(i) / "models" / model / "feedback.json")
                   for i in range(len(parts_b))]
        fb = LV.merge_feedback(reports, starts, lengths, self.feedback_mode)
        write_json(mdir / "feedback.json", fb)
        cert = qc.vmd_certificate(parse_vmd(str(mdir / "before.vmd")), profile)
        cert["hand_levels_l01"] = {"l": 0, "r": 0}
        cert["face_l01"] = 0
        for i in range(len(parts_b)):
            c = load_json(self.chunk_dir(i) / "models" / model / "before.cert.json")
            for side in ("l", "r"):
                cert["hand_levels_l01"][side] += int(
                    (c.get("hand_levels_l01") or {}).get(side, 0))
            cert["face_l01"] += int(c.get("face_l01") or 0)
        cert["source_frames"] = int(plan["n_frames"])
        cert["chunked"] = True
        write_json(mdir / "before.cert.json", cert)
        if not cert["ok"]:
            raise StageError(f"{model}:merge",
                             f"merged VMD certificate failed: {cert}")
        # unmerged source step at each seam (a hard content cut is not a merge bug)
        src_steps = []
        for i in range(1, len(parts_b)):
            prev = parts_b[i - 1]
            cur = parts_b[i]
            sp = LV.local_step_series(prev, lengths[i - 1])
            sc = LV.local_step_series(cur, lengths[i])
            src_steps.append(round(float(max(
                sp[max(0, lengths[i - 1] - O - 1):].max(initial=0.0),
                sc[:O + 1].max(initial=0.0))), 4))
        report = {
            "model": model,
            "enabled": True,
            "plan": plan,
            "frames_expected": int(plan["n_frames"]),
            "frames_written": n_out,
            "chunks": [{"index": i, "start": starts[i], "end": ends[i],
                        "length": lengths[i], "offset": starts[i],
                        "shift": shifts[i]} for i in range(len(parts_b))],
            "seams": [{"frame": starts[i], "source_max_step_deg": src_steps[i - 1]}
                      for i in range(1, len(parts_b))],
            "merge_before": rep_b,
            "merge_final": rep_f,
            "certificate": cert,
            "feedback": {
                "mode": fb.get("mode"),
                "ckpt": fb.get("ckpt"),
                "ckpt_sha256": fb.get("ckpt_sha256"),
                "ckpt_mismatch": fb.get("ckpt_mismatch"),
                "mode_mismatch": fb.get("mode_mismatch"),
                "segments_fallback": len(fb.get("segments_fallback") or []),
                "whole_clip_fallback": fb.get("whole_clip_fallback"),
                "handface_fallback": fb.get("handface_fallback"),
            },
            "ok": bool(n_out == int(plan["n_frames"]) and cert["ok"]),
            "note": ("chunks aligned by root yaw+translation in the overlap and "
                     "cross-faded with smoothstep weights; before/final use the "
                     "same transforms"),
        }
        write_json(mdir / "merge.json", report)
        self.log(f"[{model}:merge] {n_out} frames, {len(parts_b)} chunks, "
                 f"seams {len(src_steps)}, cert "
                 f"{'OK' if cert['ok'] else 'FAIL'}")

    # -- global stages ------------------------------------------------------
    def _stage_gvhmr(self):
        # A changed input video must not reuse the previous npz/preprocess: the
        # stage was invalidated because its inputs changed, so start clean.
        if self.gvhmr_dir.exists():
            shutil.rmtree(self.gvhmr_dir, ignore_errors=True)
        self.gvhmr_dir.mkdir(parents=True, exist_ok=True)
        g = self.cfg.get("gvhmr") or {}
        cmd = [PY, str(ROOT / "scripts/run_gvhmr.py"),
               "--video", str(self.video), "--output", str(self.gvhmr_dir)]
        if g.get("static_cam", True):
            cmd.append("--static_cam")
        self._run_cmd("gvhmr", cmd)
        meta = load_json(self.gvhmr_dir / "gvhmr_meta.json")
        if int(meta.get("frames", 0)) < 2:
            raise StageError("gvhmr", "GVHMR returned fewer than 2 frames")

    def _stage_hands(self):
        h = self.cfg.get("hand") or {}
        cmd = [PY, str(ROOT / "scripts/run_hand.py"),
               "--video", str(self.gvhmr_dir / "0_input_video.mp4"),
               "--kp", str(self.gvhmr_dir / "preprocess/vitpose.pt"),
               "--out", str(self.hand_dir),
               "--input-size", str(int(h.get("input_size", 640))),
               "--conf", str(float(h.get("conf", 0.12))),
               "--overlays", str(int(h.get("overlays", 0)))]
        self._run_cmd("hands", cmd)

    def _stage_face(self):
        f = self.cfg.get("face") or {}
        cmd = [PY, str(ROOT / "scripts/run_face.py"),
               "--video", str(self.gvhmr_dir / "0_input_video.mp4"),
               "--kp", str(self.gvhmr_dir / "preprocess/vitpose.pt"),
               "--out", str(self.face_dir),
               "--input-size", str(int(f.get("input_size", 512))),
               "--conf", str(float(f.get("conf", 0.2))),
               "--overlays", str(int(f.get("overlays", 0)))]
        self._run_cmd("face", cmd)

    def _global_stages(self):
        gvhmr_out = [self.gvhmr_dir / "gvhmr_result.npz",
                     self.gvhmr_dir / "gvhmr_meta.json",
                     self.gvhmr_dir / "0_input_video.mp4",
                     self.gvhmr_dir / "preprocess/vitpose.pt"]
        self._run_stage("gvhmr", self._stage_gvhmr,
                        {"video": self.video},
                        {"static_cam": bool((self.cfg.get("gvhmr") or {}).get(
                            "static_cam", True))},
                        gvhmr_out)
        ev_in = {"npz": self.gvhmr_dir / "gvhmr_result.npz",
                 "video": self.gvhmr_dir / "0_input_video.mp4",
                 "kp": self.gvhmr_dir / "preprocess/vitpose.pt"}
        self._run_stage("hands", self._stage_hands, ev_in,
                        dict(self.cfg.get("hand") or {}),
                        [self.hand_dir / "hand.npz", self.hand_dir / "hand.json"])
        self._run_stage("face", self._stage_face, ev_in,
                        dict(self.cfg.get("face") or {}),
                        [self.face_dir / "face.npz", self.face_dir / "face.json"])

    # -- per-model stages ---------------------------------------------------
    def _evidence(self) -> Dict:
        """Raw evidence at source (GVHMR 30 fps) rate."""
        g = self.gvhmr_dir
        d = np.load(g / "gvhmr_result.npz")
        if "kp2d" in d:
            kp = np.asarray(d["kp2d"], np.float32)
        else:
            import torch
            kp = torch.load(g / "preprocess/vitpose.pt", map_location="cpu")
            kp = kp.cpu().numpy() if hasattr(kp, "cpu") else np.asarray(kp)
        bbx = np.asarray(d["bbx_xys"], np.float32)
        meta = load_json(g / "gvhmr_meta.json")
        hand = np.load(self.hand_dir / "hand.npz")
        face = np.load(self.face_dir / "face.npz")
        return {
            "kp2d": kp, "bbx_xys": bbx,
            "img_wh": (int(meta["width"]), int(meta["height"])),
            "n_src": int(meta["frames"]),
            "hand": {k: hand[k] for k in ("angles", "level", "score", "detected")},
            "face": {k: face[k] for k in ("weights", "level", "detected")},
        }

    def _evidence_at_rate(self, n: int, adapter_scale: float) -> Dict:
        ev = self._evidence()
        out = FA.resample_evidence(ev, n, self.slow_factor)
        out["adapter_scale"] = float(adapter_scale)
        return out

    def _kp_at_rate(self, n: int) -> np.ndarray:
        """ViTPose 2D evidence resampled to the VMD frame rate (no hands/face)."""
        g = self.gvhmr_dir
        d = np.load(g / "gvhmr_result.npz")
        if "kp2d" in d:
            kp = np.asarray(d["kp2d"], np.float32)
        else:
            import torch
            kp = torch.load(g / "preprocess/vitpose.pt", map_location="cpu")
            kp = kp.cpu().numpy() if hasattr(kp, "cpu") else np.asarray(kp)
        idx = FA.vmd_source_indices(n, self.slow_factor, len(kp))
        return np.asarray(kp)[idx]

    def _stage_adapter(self, model: str):
        out = self.model_dir(model) / "adapter.vmd"
        cmd = [PY, str(ROOT / "scripts/run_adapter.py"),
               "--npz", str(self.gvhmr_dir / "gvhmr_result.npz"),
               "--model", model, "--slow-factor", str(self.slow_factor),
               "--out", str(out)]
        self._run_cmd(f"{model}:adapter", cmd)

    def _stage_cleanup(self, model: str):
        profile = self.profile(model)
        pmx = self.model_pmx(model)
        adapter = parse_vmd(str(self.model_dir(model) / "adapter.vmd"))
        res = clean_vmd(adapter, pmx, profile)
        write_vmd(str(self.model_dir(model) / "clean.vmd"),
                  res["vmd"].get("model_name", ""), res["vmd"]["bones"],
                  res["vmd"].get("morphs", {}), res["vmd"].get("ik_frames", []))
        write_json(self.model_dir(model) / "clean.json", res["report"])

    def _stage_contact_ik(self, model: str):
        profile = self.profile(model)
        pmx = self.model_pmx(model)
        clean = parse_vmd(str(self.model_dir(model) / "clean.vmd"))
        n = int(clean["max_frame"]) + 1
        res = solve_contact_ik(clean, pmx, profile, kp2d=self._kp_at_rate(n))
        write_vmd(str(self.model_dir(model) / "ik.vmd"),
                  res["vmd"].get("model_name", ""), res["vmd"]["bones"],
                  res["vmd"].get("morphs", {}), res["vmd"].get("ik_frames", []))
        write_json(self.model_dir(model) / "ik.json", res["report"])

    def _stage_assemble(self, model: str):
        profile = self.profile(model)
        body = parse_vmd(str(self.model_dir(model) / "ik.vmd"))
        n = int(body["max_frame"]) + 1
        scale = float(load_json(self.model_dir(model) / "adapter.vmd.json").get(
            "scale", 1.0))
        ev = self._evidence_at_rate(n, scale)
        hand = ev["hand"]
        face = ev["face"]
        vmd = FD.assemble_vmd(body, profile, hand["angles"], face["weights"],
                              n_frames=n)
        out = self.model_dir(model) / "before.vmd"
        write_vmd(str(out), vmd.get("model_name", ""), vmd["bones"],
                  vmd["morphs"], vmd["ik_frames"])
        parsed = parse_vmd(str(out))
        cert = qc.vmd_certificate(parsed, profile)
        cert["hand_levels_l01"] = {
            s: int((np.asarray(hand["level"][:, i]) <= 1).sum())
            for i, s in enumerate(("l", "r"))}
        cert["face_l01"] = int((np.asarray(face["level"]) <= 1).sum())
        cert["source_frames"] = int(ev.get("n_src") or 0)
        write_json(self.model_dir(model) / "before.cert.json", cert)
        if not cert["ok"]:
            raise StageError(f"{model}:assemble",
                             f"53-track/IK-off certificate failed: {cert}")

    def _feedback_model(self) -> Dict:
        if self._fb_cache is None:
            self._fb_cache = FA.load_feedback_model(str(ROOT / self.ckpt))
        return self._fb_cache

    def _stage_feedback(self, model: str):
        profile = self.profile(model)
        pmx = self.model_pmx(model)
        before = parse_vmd(str(self.model_dir(model) / "before.vmd"))
        n = int(before["max_frame"]) + 1
        fb_cfg = self.cfg["feedback"]
        mode_cfg = FA.resolve_config(
            self.feedback_mode, fb_cfg.get("modes"),
            (self.cfg.get("feedback") or {}).get("overrides"))
        fb_key = ("fallback_v2" if self.feedback_mode in ("v2", "v2_full")
                  else "fallback")
        fallback = dict(FA.DEFAULT_FALLBACK, **(fb_cfg.get(fb_key) or {}))
        mdir = self.model_dir(model)
        if self.feedback_mode == "off":
            for name in ("after.vmd", "final.vmd"):
                shutil.copyfile(mdir / "before.vmd", mdir / name)
            payload = {
                "mode": "off", "config": mode_cfg, "applied": False,
                "ckpt": None, "ckpt_sha256": None,
                "segments_fallback": [], "whole_clip_fallback": False,
                "handface_fallback": False,
                "residual_mean_abs": {"root": 0.0, "body": 0.0,
                                      "hands": 0.0, "face": 0.0},
                "note": "feedback disabled: final = M3 output",
            }
            write_json(mdir / "feedback.json", payload)
            return

        fb = self._feedback_model()
        scale = float(load_json(mdir / "adapter.vmd.json").get("scale", 1.0))
        ev = self._evidence_at_rate(n, scale)
        res = FA.apply_feedback(before, pmx, profile, fb, ev, mode_cfg, fallback)
        write_vmd(str(mdir / "after.vmd"), res["vmd"].get("model_name", ""),
                  res["vmd"]["bones"], res["vmd"]["morphs"],
                  res["vmd"]["ik_frames"])
        if res["whole_clip_fallback"]:
            shutil.copyfile(mdir / "before.vmd", mdir / "final.vmd")
        else:
            shutil.copyfile(mdir / "after.vmd", mdir / "final.vmd")
        payload = {
            "mode": self.feedback_mode,
            "config": mode_cfg,
            "applied": bool(res["applied"]),
            "ckpt": fb["path"],
            "ckpt_sha256": fb["sha256"],
            "ckpt_params": fb["params"],
            "ckpt_epoch": fb["epoch"],
            "segments_fallback": res["segments_fallback"],
            "whole_clip_fallback": res["whole_clip_fallback"],
            "handface_fallback": res["handface_fallback"],
            "residual_mean_abs": res["residual_mean_abs"],
            "raw_residual_mean_abs": res["raw_residual_mean_abs"],
            "features_shape": res["features_shape"],
            "final_source": "before" if res["whole_clip_fallback"] else "after",
        }
        write_json(mdir / "feedback.json", payload)

    # -- render -------------------------------------------------------------
    def _render_frames(self, n: int) -> tuple:
        r = self.cfg.get("render") or {}
        b_idx = r.get("boundary")
        if self.boundary_frames and b_idx is not None:
            seams = self.boundary_frames
            try:
                idx = int(b_idx)
            except (TypeError, ValueError):
                idx = 0
            if idx < 0:
                idx += len(seams)
            idx = max(0, min(len(seams) - 1, idx))
            seam = int(seams[idx]) + 1  # Blender frame = VMD frame + 1
            half = max(5, int(r.get("boundary_half", 30)))
            a = max(1, seam - half)
            b = min(n, seam + half)
            if b - a >= 2:
                return a, b
        a, b = [int(x) for x in str(r.get("frames", "70:150")).split(":")]
        a = max(1, a)
        b = min(b, n)  # Blender frame = VMD frame + 1
        if b - a < 2:
            a, b = 1, max(2, min(n, 3))
        return a, b

    def _render_stills(self, a: int, b: int) -> List[int]:
        r = self.cfg.get("render") or {}
        if self.boundary_frames and r.get("boundary") is not None:
            seams = self.boundary_frames
            idx = max(0, min(len(seams) - 1, int(r.get("boundary") or 0)))
            seam = int(seams[idx]) + 1
            want = [seam - 2, seam - 1, seam, seam + 1, seam + 2]
        else:
            want = [int(x) for x in (r.get("stills") or [])]
        return sorted({f for f in want if a <= f <= b})

    def _bbox_scope(self) -> str:
        r = self.cfg.get("render") or {}
        if r.get("bbox_scope"):
            return str(r["bbox_scope"])
        return "range" if self.chunk_plan else "clip"

    def _render_blend(self, model: str) -> Path:
        """Scene used for before/after evidence.

        The proven three-character scene is used for all models (it contains
        CHR_祀 / CHR_洛茜 / CHR_提弗洛斯); a per-model ``blend`` in
        configs/models.json is the fallback for other setups.
        """
        r = self.cfg.get("render") or {}
        return Path(r.get("blend") or self.models_cfg[model]["blend"])

    def _stage_render(self, model: str, tag: str, vmd_path: Path,
                      camera_from: Optional[Path]):
        r = self.cfg.get("render") or {}
        mdir = self.model_dir(model)
        out_dir = mdir / "render" / tag
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = int(parse_vmd(str(vmd_path))["max_frame"]) + 1
        a, b = self._render_frames(n)
        stills_frames = self._render_stills(a, b)
        cmd = [str(self._blender()), "--background",
               str(self._blend_path(model)), "--python",
               str(ROOT / "scripts/render_synth.py"), "--",
               "--config", str(ROOT / "configs/models.json"),
               "--model", model, "--vmd", str(vmd_path),
               "--view", str(r.get("view", "front")), "--out", str(out_dir),
               "--frames", f"{a}:{b}", "--res", str(r.get("res", "1440x2560")),
               "--samples", str(int(r.get("samples", 12))),
               "--fps", str(int(r.get("fps", 30))),
               "--bbox-scope", self._bbox_scope(),
               "--bbox-frames", str(int(r.get("bbox_frames", 0)))]
        if camera_from is not None:
            cmd += ["--camera-from", str(camera_from)]
        self._run_cmd(f"{model}:render_{tag}", cmd,
                      retries=int(r.get("retries", 2)))
        frames = [out_dir / "frames" / f"frame_{f:04d}.png" for f in range(a, b + 1)]
        missing = [f.name for f in frames if not f.exists()]
        if missing:
            raise StageError(f"{model}:render_{tag}",
                             f"{len(missing)} frames missing (first {missing[:3]})")
        mp4 = out_dir / f"{model}_{tag}_{a}-{b}.mp4"
        self._encode(out_dir, a, b, mp4, int(r.get("fps", 30)))
        stills_dir = out_dir / "stills"
        stills_dir.mkdir(exist_ok=True)
        stills = []
        for f in stills_frames:
            shutil.copyfile(out_dir / "frames" / f"frame_{f:04d}.png",
                            stills_dir / f"frame_{f:04d}.png")
            stills.append(f)
        meta = load_json(out_dir / "meta.json")
        bake = meta.get("inherit_bake") or {}
        d = bake.get("d_chain_vs_fk") or {}
        v = bake.get("verify") or {}
        fr = meta.get("framing") or {}
        report = {
            "model": model, "tag": tag, "vmd": str(vmd_path),
            "vmd_sha256": sha256_file(vmd_path),
            "frames": [a, b], "frames_rendered": b - a + 1,
            "resolution": meta.get("resolution"),
            "samples": meta.get("samples"),
            "camera": meta.get("camera"),
            "camera_reused": bool((meta.get("camera") or {}).get("reused")),
            "bbox_scope": meta.get("bbox", {}).get("scope"),
            "framing": fr,
            "framing_ok": bool(fr.get("ok", True)),
            "d_chain": {"ok": bool(d.get("ok")),
                        "max_pos_err": d.get("max_pos_err"),
                        "max_rot_err_deg": d.get("max_rot_err_deg")},
            "verify": {"ok": bool(v.get("ok")),
                       "max_head_error": v.get("max_head_error")},
            "scale_warning_count": int(bake.get("scale_warning_count", 0)),
            "stills": stills,
            "mp4": str(mp4.relative_to(ROOT)) if mp4.is_relative_to(ROOT) else str(mp4),
        }
        report["ok"] = bool(
            report["d_chain"]["ok"]
            and abs(float(report["d_chain"]["max_pos_err"] or 0.0)) == 0.0
            and abs(float(report["d_chain"]["max_rot_err_deg"] or 0.0)) == 0.0
            and report["verify"]["ok"]
            and float(report["verify"]["max_head_error"] or 1.0) < 2e-6
            and report["scale_warning_count"] == 0
            and report["framing_ok"])
        write_json(out_dir / "render.json", report)
        if not report["ok"]:
            raise StageError(f"{model}:render_{tag}",
                             f"render certificate failed: {report}")

    def _encode(self, out_dir: Path, a: int, b: int, mp4: Path, fps: int):
        ff = self._ffmpeg()
        cmd = [str(ff), "-y", "-loglevel", "error", "-framerate", str(fps),
               "-start_number", str(a),
               "-i", str(out_dir / "frames" / "frame_%04d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
               str(mp4)]
        self._run_cmd(f"encode_{out_dir.parent.name}_{out_dir.name}", cmd)

    # -- qc -----------------------------------------------------------------
    def _source_seam_steps(self, model: str, profile: Mapping, pmx: Mapping,
                           plan: Mapping):
        """Unmerged root/toe steps at each seam (a content cut is not a merge bug)."""
        starts = [int(x) for x in plan["starts"]]
        O = int(plan["overlap_frames"])
        out_root, out_toe = [], []
        for i in range(1, len(starts)):
            prev = parse_vmd(str(self.chunk_dir(i - 1) / "models" / model / "before.vmd"))
            cur = parse_vmd(str(self.chunk_dir(i) / "models" / model / "before.vmd"))
            Lp = int(prev["max_frame"]) + 1
            Lc = int(cur["max_frame"]) + 1
            pf = list(range(max(0, Lp - O - 2), Lp))
            cf = list(range(0, min(Lc, O + 2)))

            def steps(vmd, frames):
                tr = pose_track(pmx, vmd, profile, frames=frames)
                p = np.asarray(tr["world"]["pelvis"])
                r = float(np.linalg.norm(np.diff(p, axis=0),
                                         axis=1).max(initial=0.0))
                toe = 0.0
                for side in ("l", "r"):
                    k = f"toe_{side}"
                    if k in tr["world"]:
                        v = np.asarray(tr["world"][k])
                        toe = max(toe, float(np.linalg.norm(
                            np.diff(v, axis=0), axis=1).max(initial=0.0)))
                return r, toe

            pr, pt = steps(prev, pf)
            cr, ct = steps(cur, cf)
            out_root.append(max(pr, cr))
            out_toe.append(max(pt, ct))
        return out_root, out_toe

    def _stage_qc(self, model: str):
        profile = self.profile(model)
        pmx = self.model_pmx(model)
        mdir = self.model_dir(model)
        before = parse_vmd(str(mdir / "before.vmd"))
        final = parse_vmd(str(mdir / "final.vmd"))
        n = int(final["max_frame"]) + 1
        frames = list(range(n))
        bt = pose_track(pmx, before, profile, frames=frames)
        ft = pose_track(pmx, final, profile, frames=frames)
        gt_track = None
        if model in self.gt:
            gt_track = pose_track(pmx, parse_vmd(self.gt[model]), profile,
                                  frames=frames)
        feedback = load_json(mdir / "feedback.json")
        render = None
        rp = mdir / "render" / "after" / "render.json"
        if rp.exists():
            render = load_json(rp)
        chunking = None
        mp = mdir / "merge.json"
        if self.chunk_plan is not None and mp.exists():
            merge = load_json(mp)
            plan = merge.get("plan") or {}
            boundaries = [int(s["frame"]) for s in (merge.get("seams") or [])]
            src = [s.get("source_max_step_deg") for s in (merge.get("seams") or [])]
            src_root, src_toe = self._source_seam_steps(model, profile, pmx, plan)
            chunking = LV.boundary_report(
                bt, ft, final, profile, boundaries,
                int(plan.get("overlap_frames") or 0), n,
                source_max_step_deg=[None if v is None else float(v) for v in src],
                source_max_root_step=src_root,
                source_max_toe_step=src_toe)
            chunking["merge"] = {
                "frames_expected": merge.get("frames_expected"),
                "frames_written": merge.get("frames_written"),
                "n_chunks": len(merge.get("chunks") or []),
                "chunk_frames": plan.get("chunk_frames"),
                "overlap_frames": plan.get("overlap_frames"),
                "certificate_ok": bool((merge.get("certificate") or {}).get("ok")),
                "source": str(mp.relative_to(ROOT)) if mp.is_relative_to(ROOT) else str(mp),
            }
        rec = qc.evaluate_model(model, profile, before, final, bt, ft,
                                feedback=feedback, render=render,
                                gt_track=gt_track, heldout=self.heldout, n=n,
                                model=pmx, chunking=chunking)
        rec["profile"] = f"profiles/{model}.json"
        rec["pmx"] = str(self.models_cfg[model]["pmx"])
        write_json(mdir / "qc.json", rec)
        ok = "PASS" if rec["ok"] else "FAIL"
        self.log(f"[{model}:qc] {ok} warnings={len(rec['warnings'])}")

    # -- model driver -------------------------------------------------------
    def _model_body_stages(self, model: str):
        mdir = self.model_dir(model)
        profile_path = ROOT / "profiles" / f"{model}.json"
        pmx_path = Path(self.models_cfg[model]["pmx"])
        g_npz = self.gvhmr_dir / "gvhmr_result.npz"
        g_meta = self.gvhmr_dir / "gvhmr_meta.json"
        hand_npz = self.hand_dir / "hand.npz"
        face_npz = self.face_dir / "face.npz"

        self._run_stage(
            f"{model}:adapter", lambda: self._stage_adapter(model),
            {"npz": g_npz, "profile": profile_path, "pmx": pmx_path,
             "models_json": ROOT / "configs/models.json"},
            {"model": model, "slow_factor": self.slow_factor,
             "stage_yaw_deg": 180.0, "ground": True},
            [mdir / "adapter.vmd", mdir / "adapter.vmd.json"])

        self._run_stage(
            f"{model}:cleanup", lambda: self._stage_cleanup(model),
            {"adapter": mdir / "adapter.vmd", "profile": profile_path,
             "pmx": pmx_path},
            {"model": model},
            [mdir / "clean.vmd", mdir / "clean.json"])

        self._run_stage(
            f"{model}:contact_ik", lambda: self._stage_contact_ik(model),
            {"clean": mdir / "clean.vmd", "kp": g_npz,
             "profile": profile_path, "pmx": pmx_path},
            {"model": model, "slow_factor": self.slow_factor},
            [mdir / "ik.vmd", mdir / "ik.json"])

        self._run_stage(
            f"{model}:assemble", lambda: self._stage_assemble(model),
            {"ik": mdir / "ik.vmd", "hand": hand_npz, "face": face_npz,
             "profile": profile_path, "pmx": pmx_path},
            {"model": model, "slow_factor": self.slow_factor},
            [mdir / "before.vmd", mdir / "before.cert.json"])

        fb_cfg = self.cfg["feedback"]
        fb_config = {
            "mode": self.feedback_mode,
            "mode_config": FA.resolve_config(
                self.feedback_mode, fb_cfg.get("modes"),
                (self.cfg.get("feedback") or {}).get("overrides")),
            "fallback": dict(FA.DEFAULT_FALLBACK, **(fb_cfg.get("fallback") or {})),
            "slow_factor": self.slow_factor,
            "ckpt": None if self.feedback_mode == "off" else str(ROOT / self.ckpt),
        }
        fb_inputs = {"before": mdir / "before.vmd", "hand": hand_npz,
                     "face": face_npz, "npz": g_npz, "meta": g_meta,
                     "profile": profile_path, "pmx": pmx_path}
        if self.feedback_mode != "off":
            fb_inputs["ckpt"] = ROOT / self.ckpt
        self._run_stage(
            f"{model}:feedback", lambda: self._stage_feedback(model),
            fb_inputs, fb_config,
            [mdir / "feedback.json", mdir / "after.vmd", mdir / "final.vmd"])

    def _model_quality_stages(self, model: str):
        mdir = self.model_dir(model)
        profile_path = ROOT / "profiles" / f"{model}.json"
        pmx_path = Path(self.models_cfg[model]["pmx"])
        if self.do_render:
            r = self.cfg.get("render") or {}
            n_b = int(parse_vmd(str(mdir / "before.vmd"))["max_frame"]) + 1
            a_b, b_b = self._render_frames(n_b)
            n_a = int(parse_vmd(str(mdir / "final.vmd"))["max_frame"]) + 1
            a_a, b_a = self._render_frames(n_a)
            render_config = {
                "frames": r.get("frames"), "res": r.get("res"),
                "samples": int(r.get("samples", 12)), "view": r.get("view"),
                "stills": list(r.get("stills") or []), "fps": int(r.get("fps", 30)),
                "camera_from": None, "tag": "before",
                "boundary": r.get("boundary"),
                "boundary_half": int(r.get("boundary_half", 30)),
                "bbox_scope": self._bbox_scope(),
                "bbox_frames": int(r.get("bbox_frames", 0)),
                "render_frames": [a_b, b_b],
            }
            self._run_stage(
                f"{model}:render_before",
                lambda: self._stage_render(model, "before", mdir / "before.vmd", None),
                {"vmd": mdir / "before.vmd", "blend": self._blend_path(model),
                 "configs": ROOT / "configs/models.json"},
                render_config,
                [mdir / "render" / "before" / "render.json",
                 mdir / "render" / "before" / "meta.json",
                 mdir / "render" / "before" / f"{model}_before_{a_b}-{b_b}.mp4"],
                retries=0)
            before_meta = mdir / "render" / "before" / "meta.json"
            render_config2 = dict(render_config, tag="after",
                                  camera_from=str(before_meta),
                                  render_frames=[a_a, b_a])
            self._run_stage(
                f"{model}:render_after",
                lambda: self._stage_render(model, "after", mdir / "final.vmd", before_meta),
                {"vmd": mdir / "final.vmd", "camera": before_meta,
                 "blend": self._blend_path(model),
                 "configs": ROOT / "configs/models.json"},
                render_config2,
                [mdir / "render" / "after" / "render.json",
                 mdir / "render" / "after" / "meta.json",
                 mdir / "render" / "after" / f"{model}_after_{a_a}-{b_a}.mp4"],
                retries=0)

        qc_inputs = {"before": mdir / "before.vmd", "final": mdir / "final.vmd",
                     "feedback": mdir / "feedback.json", "profile": profile_path,
                     "pmx": pmx_path}
        if (mdir / "render" / "after" / "render.json").exists():
            qc_inputs["render"] = mdir / "render" / "after" / "render.json"
        if model in self.gt:
            qc_inputs["gt"] = self.gt[model]
        if self.chunk_plan is not None and (mdir / "merge.json").exists():
            qc_inputs["merge"] = mdir / "merge.json"
            plan_path = self.run_dir / "chunk_plan.json"
            if plan_path.exists():
                qc_inputs["chunk_plan"] = plan_path
            plan = load_json(plan_path) if plan_path.exists() else {}
            for i in range(int(plan.get("n_chunks") or 0)):
                qc_inputs[f"chunk{i}:before"] = (
                    self.chunk_dir(i) / "models" / model / "before.vmd")
        self._run_stage(
            f"{model}:qc", lambda: self._stage_qc(model), qc_inputs,
            {"mode": self.feedback_mode,
             "heldout": (self.heldout or {}).get("path"),
             "chunked": bool(self.chunk_plan),
             "thresholds": {"jerk": qc.JERK_TOL, "slide": qc.SLIDE_TOL,
                            "penetration": qc.PENETRATION_ABS,
                            "finger": qc.FINGER_TOL, "morph": qc.MORPH_ABS}},
            [mdir / "qc.json"])

    def _run_model(self, model: str):
        try:
            self._model_body_stages(model)
            self._model_quality_stages(model)
        except StageError as e:
            self.errors.append({"model": model, "error": str(e)})
            self.log(f"[{model}] FAILED: {e}")
            if not self.keep_going:
                raise

    # -- entry --------------------------------------------------------------
    def _write_run_meta(self):
        try:
            head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  cwd=str(ROOT), capture_output=True, text=True)
            dirty = subprocess.run(["git", "status", "--porcelain"],
                                   cwd=str(ROOT), capture_output=True, text=True)
            git = {"head": head.stdout.strip() if head.returncode == 0 else None,
                   "dirty_files": len([x for x in dirty.stdout.splitlines() if x])
                   if dirty.returncode == 0 else None}
        except Exception:
            git = {"head": None, "dirty_files": None}
        import platform
        try:
            import torch
            torch_v = torch.__version__
        except Exception:
            torch_v = None
        self.manifest.data["run"] = _norm({
            "video": str(self.video),
            "video_sha256": sha256_file(self.video),
            "models": self.models,
            "feedback_mode": self.feedback_mode,
            "feedback_config": FA.resolve_config(
                self.feedback_mode, (self.cfg.get("feedback") or {}).get("modes"),
                (self.cfg.get("feedback") or {}).get("overrides")),
            "ckpt": None if self.feedback_mode == "off" else str(ROOT / self.ckpt),
            "ckpt_sha256": (self._feedback_model()["sha256"]
                            if self.feedback_mode != "off" else None),
            "slow_factor": self.slow_factor,
            "render": bool(self.do_render),
            "python": platform.python_version(),
            "torch": torch_v,
            "git": git,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self.manifest.save()

    def run(self) -> Dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_dirs()
        self._write_run_meta()
        self._apply_seeds()
        self.log(f"run: {self.run_dir}")
        self.log(f"video: {self.video}  models: {', '.join(self.models)}  "
                 f"feedback: {self.feedback_mode}")
        try:
            if self._use_chunking():
                self.log(f"[chunking] long video: {self.chunk_frames}-frame "
                         f"chunks, {self.overlap_frames}-frame overlap")
                self._chunked_stages()
            else:
                self._global_stages()
                for model in self.models:
                    self._run_model(model)
        except StageError as e:
            self.errors.append({"model": None, "error": str(e)})
            self._finish()
            raise
        return self._finish()

    def _finish(self) -> Dict:
        summary = build_run_summary(self.run_dir)
        summary["errors"] = self.errors
        summary["elapsed_seconds"] = round(time.time() - self.t0, 1)
        write_json(self.qc_dir / "summary.json", summary)
        write_json(self.run_dir / "summary.json", summary)
        self.summary = summary
        self.log("")
        self.log(f"TASK GATE: {'PASS' if summary['gates']['task_gate'] else 'FAIL'}"
                 f"   render: {summary['gates'].get('render')}"
                 f"   secondary: {summary['gates'].get('secondary_ok')}"
                 f"   boundary: {summary['gates'].get('boundary_continuity')}")
        for line in summary.get("warnings", []):
            self.log("  WARN " + line)
        self.log(f"summary: {self.qc_dir / 'summary.json'}")
        return summary


# ---------------------------------------------------------------------------
# run summary (also used by scripts/qc_report.py)
# ---------------------------------------------------------------------------
def build_run_summary(run_dir) -> Dict:
    run_dir = Path(run_dir)
    models: Dict[str, Dict] = {}
    warnings: List[str] = []
    fallback: Dict[str, List] = {}
    mroot = run_dir / "models"
    if mroot.exists():
        for d in sorted(p for p in mroot.iterdir() if p.is_dir()):
            qp = d / "qc.json"
            if not qp.exists():
                continue
            rec = load_json(qp)
            models[d.name] = rec
            if rec.get("warnings"):
                warnings += [f"{d.name}: {w}" for w in rec["warnings"]]
            fb = (rec.get("feedback") or {}).get("segments_fallback") or []
            if fb:
                fallback[d.name] = fb
    task_gate = bool(models) and all(bool(v.get("ok")) for v in models.values())
    render_checks = [v.get("render") for v in models.values()]
    render_ok = (all(bool(r.get("ok")) for r in render_checks)
                 if render_checks and all(r is not None for r in render_checks)
                 else None)
    chunk_models = {name: rec.get("chunking") for name, rec in models.items()
                    if (rec.get("chunking") or {}).get("enabled")}
    boundary_ok = (all(bool(c.get("ok")) for c in chunk_models.values())
                   if chunk_models else None)
    segment_qc_ok = None
    for name, c in chunk_models.items():
        for seam in (c.get("seams") or []):
            if not seam.get("ok"):
                warnings.append(f"{name}: boundary {seam.get('frame')} failed "
                                f"{[k for k, v in (seam.get('checks') or {}).items() if not v]}")
    chunking_summary = None
    cp = run_dir / "chunk_plan.json"
    if cp.exists():
        plan = load_json(cp)
        chunking_summary = {
            "chunk_frames": plan.get("chunk_frames"),
            "overlap_frames": plan.get("overlap_frames"),
            "n_frames": plan.get("n_frames"),
            "n_chunks": plan.get("n_chunks"),
            "starts": plan.get("starts"),
            "ends": plan.get("ends"),
            "lengths": plan.get("lengths"),
            "models": {name: {
                "ok": bool(c.get("ok")),
                "seams": [{"frame": s.get("frame"),
                           "max_step_deg": s.get("max_step_deg"),
                           "before_world_step_max_deg": s.get("before_world_step_max_deg"),
                           "source_max_step_deg": s.get("source_max_step_deg"),
                           "source_max_root_step": s.get("source_max_root_step"),
                           "source_max_toe_step": s.get("source_max_toe_step"),
                           "freeze_run": s.get("freeze_run"),
                           "slide_rate": s.get("slide_rate"),
                           "checks": s.get("checks"),
                           "ok": s.get("ok")} for s in (c.get("seams") or [])],
                "reference": c.get("reference"),
                "merge": c.get("merge")} for name, c in chunk_models.items()},
        }
    heldouts = [v["secondary"].get("heldout_reference")
                for v in models.values()
                if (v.get("secondary") or {}).get("heldout_reference")]
    heldout = heldouts[0] if heldouts else None
    gt_secondary = [v["secondary"].get("checks") for v in models.values()
                    if (v.get("secondary") or {}).get("available")]
    if gt_secondary:
        secondary_ok = all(all(bool(x) for x in c.values()) for c in gt_secondary)
        secondary_note = "GT secondary deltas measured on this run"
    elif heldout is not None:
        secondary_ok = bool(all(bool(x) for x in (heldout.get("secondary") or {}).values()))
        secondary_note = ("no GT in this run; secondary cost is the held-out "
                          f"reference ({heldout.get('path')})")
    else:
        secondary_ok = None
        secondary_note = "no GT and no held-out reference"

    manifest_path = run_dir / "manifest.json"
    stages = []
    run_meta = {}
    if manifest_path.exists():
        m = load_json(manifest_path)
        run_meta = m.get("run") or {}
        for key, rec in m.get("stages", {}).items():
            stages.append({"stage": key, "status": rec.get("status"),
                           "seconds": rec.get("seconds"),
                           "error": rec.get("error")})
    # per-segment QC (the guide requires segment level, not only the merged clip)
    if chunking_summary is not None:
        stage_sec = {st["stage"]: st.get("seconds") for st in stages}
        plan = load_json(run_dir / "chunk_plan.json")
        segs = []
        for i in range(int(plan.get("n_chunks") or 0)):
            cdir = run_dir / "chunks" / f"seg_{i:03d}"
            per = {}
            mroot = cdir / "models"
            if mroot.exists():
                for d in sorted(p for p in mroot.iterdir() if p.is_dir()):
                    qp = d / "qc.json"
                    if not qp.exists():
                        continue
                    q = load_json(qp)
                    st = q.get("stability") or {}
                    before = float((st.get("before") or {}).get("jerk_rms", 0.0))
                    after = float((st.get("after") or {}).get("jerk_rms", 0.0))
                    per[d.name] = {
                        "ok": bool(q.get("ok")),
                        "task_gate": bool(q.get("task_gate")),
                        "warnings": list(q.get("warnings") or []),
                        "fallback_windows": len(((q.get("feedback") or {})
                                                 .get("segments_fallback") or [])),
                        "jerk_ratio": round(after / max(1e-9, before), 5),
                    }
            segs.append({"index": i,
                         "start": plan["starts"][i], "end": plan["ends"][i],
                         "frames": plan["lengths"][i],
                         "seconds": stage_sec.get(f"chunk:{i:03d}"),
                         "qc": per})
        chunking_summary["segments"] = segs
        chunking_summary["segments_all_ok"] = all(
            all(bool(v.get("ok")) for v in seg["qc"].values())
            for seg in segs if seg["qc"])
        segment_qc_ok = bool(chunking_summary["segments_all_ok"])
        if not segment_qc_ok:
            warnings.append("one or more chunk-level QC records failed "
                            "(see Segment QC)")
        task_gate = task_gate and segment_qc_ok
    summary = {
        "run": str(run_dir),
        "video": run_meta.get("video"),
        "video_sha256": run_meta.get("video_sha256"),
        "feedback_mode": run_meta.get("feedback_mode"),
        "feedback_config": run_meta.get("feedback_config"),
        "ckpt": run_meta.get("ckpt"),
        "ckpt_sha256": run_meta.get("ckpt_sha256"),
        "models": models,
        "gates": {
            "task_gate": task_gate,
            "render": render_ok,
            "secondary_ok": secondary_ok,
            "boundary_continuity": boundary_ok,
            "segment_qc": segment_qc_ok,
            "heldout_task_gate": (heldout or {}).get("task_gate"),
            "heldout_gates_ok": (heldout or {}).get("gates_ok"),
            "note": ("task_gate covers the VMD certificate, GT-free stability, "
                     "the root/legs/hands/face change audit, the chunk-boundary "
                     "continuity and per-chunk QC (long videos) and (when GT is "
                     "supplied) the "
                     "body-relative + contact/timing checks. "
                     "Secondary world-angle/position cost is reported separately "
                     "and is never silently hidden."),
        },
        "secondary": {"heldout": heldout, "note": secondary_note},
        "chunking": chunking_summary,
        "fallback_segments": fallback,
        "warnings": warnings,
        "stages": stages,
    }
    (run_dir / "qc").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "qc" / "summary.json", summary)
    (run_dir / "qc" / "summary.md").write_text(render_summary_md(summary),
                                               encoding="utf-8")
    return summary


def render_summary_md(s: Mapping) -> str:
    lines = ["# VTV pipeline QC summary", ""]
    lines.append(f"- run: `{s.get('run')}`")
    lines.append(f"- video: `{s.get('video')}` (sha256 `{(s.get('video_sha256') or '')[:12]}`)")
    lines.append(f"- feedback: **{s.get('feedback_mode')}** "
                 f"`{_norm(s.get('feedback_config'))}`")
    lines.append(f"- checkpoint: `{s.get('ckpt')}` "
                 f"(sha256 `{(s.get('ckpt_sha256') or '')[:12]}`)")
    g = s["gates"]
    lines.append("")
    lines.append(f"**Task gate: {'PASS' if g['task_gate'] else 'FAIL'}** · "
                 f"render: {g.get('render')} · secondary: {g.get('secondary_ok')} · "
                 f"boundary: {g.get('boundary_continuity')} · segment QC: {g.get('segment_qc')} · "
                 f"held-out task gate: {g.get('heldout_task_gate')}")
    lines.append("")
    ch = s.get("chunking")
    if ch:
        lines.append(f"- chunking: **{ch.get('n_chunks')} chunks** "
                     f"({ch.get('chunk_frames')} frames, overlap "
                     f"{ch.get('overlap_frames')}), {ch.get('n_frames')} frames")
    lines.append("")
    lines.append("| model | cert | jerk before→after | slide | penetration | "
                 "finger p95 | morph | feedback | fallback windows | task |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in s.get("models", {}).items():
        st = r.get("stability", {})
        b, a = st.get("before", {}), st.get("after", {})
        cert = (r.get("certificate") or {}).get("ok")
        fb = r.get("feedback") or {}
        lines.append(
            f"| {name} | {'OK' if cert else 'FAIL'} | "
            f"{(b.get('jerk_rms') or 0):.0f}→{(a.get('jerk_rms') or 0):.0f} | "
            f"{(b.get('foot_slide') or 0):.4f}→{(a.get('foot_slide') or 0):.4f} | "
            f"{(b.get('foot_penetration') or 0):.4f}→{(a.get('foot_penetration') or 0):.4f} | "
            f"{(b.get('finger_step_p95') or 0):.2f}→{(a.get('finger_step_p95') or 0):.2f} | "
            f"{(b.get('morph_step_max') or 0):.2f}→{(a.get('morph_step_max') or 0):.2f} | "
            f"{fb.get('mode')} | {len(fb.get('segments_fallback') or [])} | "
            f"{'PASS' if r.get('ok') else 'FAIL'} |")
    lines.append("")
    if ch:
        lines.append("## Chunk boundaries")
        lines.append("")
        lines.append("| model | seam | max step° | before world° | source° | "
                     "src root | src toe | freeze | slide/frame | jump | root | toe | freeze | slide | ok |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, rec in (ch.get("models") or {}).items():
            for seam in (rec.get("seams") or []):
                ck = seam.get("checks") or {}
                lines.append(
                    f"| {name} | {seam.get('frame')} | "
                    f"{seam.get('max_step_deg')} | "
                    f"{seam.get('before_world_step_max_deg')} | "
                    f"{seam.get('source_max_step_deg')} | "
                    f"{seam.get('source_max_root_step')} | "
                    f"{seam.get('source_max_toe_step')} | "
                    f"{seam.get('freeze_run')} | {seam.get('slide_rate')} | "
                    f"{'ok' if ck.get('jump') else 'FAIL'} | "
                    f"{'ok' if ck.get('root_step') else 'FAIL'} | "
                    f"{'ok' if ck.get('toe_step') else 'FAIL'} | "
                    f"{'ok' if ck.get('freeze') else 'FAIL'} | "
                    f"{'ok' if ck.get('slide') else 'FAIL'} | "
                    f"{'PASS' if seam.get('ok') else 'FAIL'} |")
        lines.append("")
        lines.append("Seam windows are compared with the interior of the same clip "
                     "(p95 ×1.5 + absolute floors); `source°` is the unmerged "
                     "chunk step at the seam, so a genuine content cut is not "
                     "blamed on the merge.")
        lines.append("")
        segs = ch.get("segments") or []
        if segs:
            names = sorted({m for seg in segs for m in (seg.get("qc") or {})})
            lines.append("## Segment QC (per chunk, before merge)")
            lines.append("")
            lines.append("| chunk | frames | seconds | " + " | ".join(names) + " |")
            lines.append("|---|---|---|" + "---|" * len(names))
            for seg in segs:
                cells = []
                for n in names:
                    q = (seg.get("qc") or {}).get(n) or {}
                    cells.append(
                        f"{'PASS' if q.get('ok') else 'FAIL'} "
                        f"(jerk ×{q.get('jerk_ratio')}, "
                        f"fallback {q.get('fallback_windows')})" if q else "-")
                lines.append(f"| {seg.get('index')} | {seg.get('frames')} | "
                             f"{seg.get('seconds')} | " + " | ".join(cells) + " |")
            lines.append("")
    if s.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for w in s["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    h = (s.get("secondary") or {}).get("heldout")
    if h:
        lines.append("## Secondary cost (not a task gate)")
        lines.append("")
        lines.append(f"- held-out reference `{h.get('path')}` "
                     f"({h.get('clips')} clips / {h.get('frames')} frames): "
                     f"body-relative main {h.get('body_relative_main_before'):.2f}° → "
                     f"{h.get('body_relative_main_after'):.2f}°, "
                     f"world {h.get('world_angle_delta_deg'):+.2f}°, "
                     f"position {h.get('position_delta_pct_height'):+.3f}%H.")
        lines.append(f"- {s['secondary'].get('note')}")
        lines.append("")
    if s.get("stages"):
        lines.append("## Stages")
        lines.append("")
        lines.append("| stage | status | seconds |")
        lines.append("|---|---|---|")
        for st in s["stages"]:
            sec = st.get("seconds")
            lines.append(f"| {st['stage']} | {st['status']} | "
                         f"{'' if sec is None else f'{sec:.1f}'} |")
    return "\n".join(lines) + "\n"
