"""Preflight check: can this machine run the VTV pipeline (M4/M5)?

  python scripts/check_env.py
  python scripts/check_env.py --models si --require-feedback
  python scripts/check_env.py --long            # chunked long-video path
  python scripts/check_env.py --out outputs/m5/preflight
  python scripts/check_env.py --json

Checks the Python stack, the (non-redistributed) weights/assets, the PMX
profiles, Blender/ffmpeg, the M3.5 checkpoint and the long-video prerequisites.
Exit code 1 when a required item is missing; optional items are reported WARN.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _run(cmd, timeout=30):
    try:
        r = subprocess.run([str(c) for c in cmd], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return -1, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma separated model keys")
    ap.add_argument("--require-feedback", action="store_true",
                    help="treat the M3.5 checkpoint as required (default profile)")
    ap.add_argument("--config", default=str(ROOT / "configs/pipeline.json"))
    ap.add_argument("--long", action="store_true",
                    help="also check the M5 chunked long-video path")
    ap.add_argument("--out", default=None,
                    help="check that this run directory is writable")
    ap.add_argument("--json", action="store_true",
                    help="machine readable rows (still exits 1 on missing required)")
    args = ap.parse_args()

    rows = []

    def add(name, ok, detail, required=True):
        rows.append({"name": name, "ok": bool(ok), "required": bool(required),
                     "detail": str(detail)})

    # ---- python stack ----------------------------------------------------
    ok = sys.version_info >= (3, 10)
    add("python>=3.10", ok, sys.version.split()[0], True)
    try:
        import numpy as np
        add("numpy<2", int(np.__version__.split(".")[0]) < 2, np.__version__, True)
    except Exception as e:  # noqa: BLE001
        add("numpy", False, repr(e), True)
    for mod, label in (("torch", "torch"), ("cv2", "opencv"),
                       ("mediapipe", "mediapipe")):
        try:
            m = __import__(mod)
            add(label, True, getattr(m, "__version__", "?"), True)
        except Exception as e:  # noqa: BLE001
            add(label, False, repr(e), True)
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if cuda else "no CUDA device"
        add("GPU/CUDA", cuda, f"{name} (torch {torch.__version__})", True)
    except Exception as e:  # noqa: BLE001
        add("GPU/CUDA", False, repr(e), True)

    # ---- external tools --------------------------------------------------
    cfg = load(args.config)
    render = cfg.get("render") or {}
    blender = Path(os.environ.get("VTV_BLENDER") or render.get(
        "blender", "blender"))
    ffmpeg = Path(os.environ.get("VTV_FFMPEG") or render.get(
        "ffmpeg", "ffmpeg"))
    render_required = bool(render.get("enabled", True))
    add("blender", blender.exists(), str(blender), render_required)
    add("ffmpeg", ffmpeg.exists() or shutil.which(str(ffmpeg)) is not None,
        str(ffmpeg), True)
    code, out = _run([ffmpeg, "-version"])
    add("ffmpeg runs", code == 0, out.splitlines()[0] if out else f"exit {code}",
        True)
    add("ffmpeg libx264", "libx264" in out, "encoder for before/after MP4s",
        render_required)
    blend_path = os.environ.get("VTV_BLEND") or render.get("blend")
    if blend_path:
        add("render scene (.blend)", Path(blend_path).exists(), blend_path,
            render_required)
    if blender.exists():
        code, bout = _run([blender, "--version"], timeout=30)
        add("blender runs", code == 0,
            bout.splitlines()[0] if bout else f"exit {code}", render_required)

    # ---- research assets (not redistributed) ------------------------------
    gvhmr = ROOT / "third_party/GVHMR"
    add("GVHMR checkout", gvhmr.exists(), str(gvhmr), True)
    add("GVHMR checkpoint",
        (gvhmr / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt").exists(),
        "third_party/GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
        True)
    smplx = list((ROOT / "body_models/smplx").glob("SMPLX_*.npz"))
    add("SMPL-X body models", len(smplx) >= 1,
        f"{len(smplx)} npz in body_models/smplx", True)
    add("MediaPipe hand model",
        (ROOT / "models/mediapipe/hand_landmarker.task").exists(),
        "models/mediapipe/hand_landmarker.task", True)
    add("MediaPipe face model",
        (ROOT / "models/mediapipe/face_landmarker.task").exists(),
        "models/mediapipe/face_landmarker.task", True)
    ckpt = ROOT / cfg["feedback"]["ckpt"]
    add("M3.5 feedback checkpoint", ckpt.exists(), str(ckpt),
        args.require_feedback or (cfg.get("feedback") or {}).get("mode") == "default")
    v2 = (cfg.get("feedback") or {}).get("ckpt_v2")
    if v2:
        p = ROOT / v2
        add("M6 v2 feedback checkpoint", p.exists(), str(p),
            args.require_feedback
            or (cfg.get("feedback") or {}).get("mode") in ("v2", "v2_full"))

    # ---- per-model assets -------------------------------------------------
    mc = load(ROOT / "configs/models.json")["models"]
    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else list(cfg.get("models") or mc.keys()))
    for m in models:
        if m not in mc:
            add(f"model {m}", False, "not in configs/models.json", True)
            continue
        pmx = Path(mc[m]["pmx"])
        blend = Path(mc[m]["blend"])
        prof = ROOT / "profiles" / f"{m}.json"
        add(f"{m}: PMX", pmx.exists(), str(pmx), True)
        add(f"{m}: blend", blend.exists(), str(blend), False)
        add(f"{m}: profile", prof.exists(), str(prof), True)

    # ---- long-video (M5) --------------------------------------------------
    chunk = cfg.get("chunking") or {}
    add("pipeline.json chunking block", bool(chunk), json.dumps(chunk), args.long)
    add("scripts/make_long_video.py",
        (ROOT / "scripts/make_long_video.py").exists(),
        "asset-free long-video builder for demos", args.long)
    add("src/vtv/longvideo.py", (ROOT / "src/vtv/longvideo.py").exists(),
        "chunk plan / align / merge / seam QC", True)
    add("src/vtv/framing.py", (ROOT / "src/vtv/framing.py").exists(),
        "per-model auto-frame + framing report", True)
    add("demo manifest", (ROOT / "demo/manifest.json").exists(),
        "release demo checklist (assets not included)", False)
    if args.long:
        # ffmpeg needs the `select` filter for frame-exact cuts; it is built in
        # everywhere, but report it explicitly.
        code, fout = _run([ffmpeg, "-filters"])
        add("ffmpeg select filter", "select" in fout,
            "frame-exact chunk cuts", True)

    # ---- output dir -------------------------------------------------------
    if args.out:
        out = Path(args.out)
        try:
            out.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(out), delete=True) as f:
                f.write(b"vtv")
            add("output dir writable", True, str(out), True)
        except Exception as e:  # noqa: BLE001
            add("output dir writable", False, f"{out}: {e}", True)

    width = max(len(r["name"]) for r in rows)
    failed = []
    for r in rows:
        status = "OK  " if r["ok"] else ("FAIL" if r["required"] else "WARN")
        if not args.json:
            print(f"[{status}] {r['name']:<{width}}  {r['detail']}")
        if not r["ok"] and r["required"]:
            failed.append(r["name"])
    if args.json:
        print(json.dumps({"ok": not failed, "rows": rows, "models": models,
                          "failed": failed}, ensure_ascii=False, indent=1))
    else:
        print()
        if failed:
            print(f"{len(failed)} required item(s) missing: {', '.join(failed)}")
            print("See docs/M5_使用指南.md (安装) and docs/开发前准备清单.md.")
        else:
            print(f"environment OK for models: {', '.join(models)}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
