# -*- coding: utf-8 -*-
"""Command line entry point for the VTV pipeline.

``scripts/run_vtv.py`` is a thin shim around this module; the same function is
exposed as the ``vtv`` console script in ``pyproject.toml``.

Examples
--------
    # T5, three models, default conservative feedback, 2K before/after
    python scripts/run_vtv.py --video testdata/T5_高速舞蹈_001/001.mp4 \
        --out outputs/m4/t5 --models si,luoxi,tiflos --render

    # long video: 10 s chunks with 1 s overlap, per-model merge + seam QC
    python scripts/run_vtv.py --video demo/long.mp4 --out outputs/m5/long \
        --models si,luoxi,tiflos --render --render-boundary 0

    # no render, feedback off (pure M3 baseline)
    python scripts/run_vtv.py --video ... --out ... --feedback off --no-render

    # warm start the expensive upstream stages, then run only the model stages
    python scripts/run_vtv.py --video ... --out outputs/m4/t5 \
        --seed gvhmr=outputs/gvhmr/T5 --seed hands=outputs/m3/t5 --seed face=outputs/m3/t5

    # re-run one stage (downstream is invalidated by changed inputs)
    python scripts/run_vtv.py --video ... --out outputs/m4/t5 --force feedback

    # installation check only
    python scripts/check_env.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from .pipeline import Pipeline, StageError, load_json  # noqa: E402

FPS = 30


def parse_pairs(values):
    out = {}
    for v in values or []:
        if "=" not in v:
            raise argparse.ArgumentTypeError(f"expected name=path, got {v!r}")
        k, p = v.split("=", 1)
        out[k.strip()] = p.strip()
    return out


def parse_ints(values):
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [int(x) for x in values]
    return [int(x) for x in str(values).split(",") if str(x).strip()]


def build_parser(cfg_path: Path) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="VTV one-command pipeline (M5): video -> VMD -> 2K -> QC, "
                    "with automatic chunking for long videos")
    ap.add_argument("--video", required=True, help="input video (monocular)")
    ap.add_argument("--out", default=None,
                    help="run/output directory (default outputs/m5/<video stem>)")
    ap.add_argument("--models", default=None,
                    help="comma separated model keys (default from config)")
    ap.add_argument("--config", default=str(cfg_path))
    ap.add_argument("--feedback", default=None,
                    choices=["off", "default", "full", "v2", "v2_full"],
                    help="feedback profile (config default is v2; M3.5 is "
                         "'default'/'full')")
    ap.add_argument("--ckpt", default=None,
                    help="M3.5 checkpoint (default: config feedback.ckpt)")
    ap.add_argument("--slow-factor", type=float, default=None,
                    help="source slow-motion factor: 1.0 normal, 0.5 T2, 0.25 T1")
    ap.add_argument("--chunk-seconds", type=float, default=None,
                    help="long-video chunk length in seconds (0 disables chunking; "
                         "default: config chunking.chunk_seconds)")
    ap.add_argument("--overlap-seconds", type=float, default=None,
                    help="chunk overlap in seconds (default: config)")
    ap.add_argument("--no-chunk", action="store_true",
                    help="force a single clip even for long videos")
    ap.add_argument("--force-chunk", default=None,
                    help="comma separated chunk indices to bypass the top-level "
                         "chunk cache (the chunk's own stage cache still applies)")
    ap.add_argument("--render", dest="render", action="store_true", default=None,
                    help="render 2K before/after (default: config)")
    ap.add_argument("--no-render", dest="render", action="store_false")
    ap.add_argument("--res", default=None, help="render resolution WxH")
    ap.add_argument("--frames", default=None, help="render frame range a:b")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--view", default=None, help="front|three_quarter|side")
    ap.add_argument("--stills", default=None,
                    help="comma separated still frames inside --frames")
    ap.add_argument("--render-boundary", default=None, metavar="N|none",
                    help="chunked runs: render a window around seam N instead "
                         "of --frames (default: config render.boundary)")
    ap.add_argument("--boundary-half", type=int, default=None,
                    help="frames on each side of the seam window")
    ap.add_argument("--bbox-scope", default=None, choices=["clip", "range"],
                    help="auto-frame bbox scope (default: clip; chunked runs: range)")
    ap.add_argument("--stage", default=None,
                    help="comma separated stages to run (video30,chunks,gvhmr,"
                         "hands,face,adapter,cleanup,contact_ik,assemble,"
                         "feedback,merge,render_before,render_after,qc); "
                         "default: all")
    ap.add_argument("--force", default=None,
                    help="comma separated stages to re-run (or 'all')")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the manifest and re-run every selected stage")
    ap.add_argument("--seed", action="append", default=None, metavar="NAME=DIR",
                    help="warm-start gvhmr/hands/face from a previous run")
    ap.add_argument("--gt", action="append", default=None, metavar="MODEL=VMD",
                    help="ground-truth VMD for synthetic clips (enables GT QC)")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue other models after a model failure")
    return ap


def main(argv=None):
    ap = build_parser(ROOT / "configs/pipeline.json")
    args = ap.parse_args(argv)

    cfg = copy.deepcopy(load_json(args.config))
    if args.models:
        cfg["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.feedback:
        cfg["feedback"]["mode"] = args.feedback
    if args.ckpt:
        key = ("ckpt_v2" if cfg["feedback"].get("mode") in ("v2", "v2_full")
               else "ckpt")
        cfg["feedback"][key] = args.ckpt
    if args.slow_factor is not None:
        cfg["slow_factor"] = float(args.slow_factor)
    if args.render is not None:
        cfg["render"]["enabled"] = bool(args.render)
    if args.res:
        cfg["render"]["res"] = args.res
    if args.frames:
        cfg["render"]["frames"] = args.frames
    if args.samples is not None:
        cfg["render"]["samples"] = int(args.samples)
    if args.view:
        cfg["render"]["view"] = args.view
    if args.stills:
        cfg["render"]["stills"] = [int(x) for x in args.stills.split(",") if x.strip()]
    if args.bbox_scope:
        cfg["render"]["bbox_scope"] = args.bbox_scope
    if args.boundary_half is not None:
        cfg["render"]["boundary_half"] = int(args.boundary_half)
    if args.render_boundary is not None:
        cfg["render"]["boundary"] = (None if str(args.render_boundary).lower()
                                     in ("none", "off", "-1")
                                     else int(args.render_boundary))

    chunk = dict(cfg.get("chunking") or {})
    chunk.setdefault("enabled", True)
    chunk.setdefault("chunk_seconds", 10.0)
    chunk.setdefault("overlap_seconds", 1.0)
    if args.no_chunk:
        chunk["enabled"] = False
    if args.chunk_seconds is not None:
        chunk["chunk_seconds"] = float(args.chunk_seconds)
        chunk["enabled"] = float(args.chunk_seconds) > 0
    if args.overlap_seconds is not None:
        chunk["overlap_seconds"] = float(args.overlap_seconds)
    if not chunk["enabled"]:
        chunk_frames, overlap_frames = 0, 0
    else:
        chunk_frames = int(round(float(chunk["chunk_seconds"]) * FPS))
        overlap_frames = int(round(float(chunk["overlap_seconds"]) * FPS))
        if chunk_frames <= 0:
            chunk_frames, overlap_frames = 0, 0
        elif overlap_frames >= chunk_frames:
            raise SystemExit("configuration error: overlap must be < chunk")
    cfg["chunking"] = chunk
    if args.render_boundary is None and chunk_frames and cfg["render"].get(
            "boundary") is None and cfg["render"].get("enabled", True):
        # a long run with rendering asks for seam evidence unless told otherwise
        cfg["render"]["boundary"] = 0

    out = args.out or str(ROOT / "outputs" / "m5" / Path(args.video).stem)
    stages = [s.strip() for s in args.stage.split(",")] if args.stage else None
    force = [s.strip() for s in args.force.split(",")] if args.force else None
    try:
        fb_mode = cfg["feedback"]["mode"]
        fb_ckpt = (cfg["feedback"].get("ckpt_v2")
                   if fb_mode in ("v2", "v2_full")
                   else cfg["feedback"]["ckpt"])
        pipe = Pipeline(
            video=args.video, out_dir=out, models=cfg["models"], config=cfg,
            feedback_mode=fb_mode,
            ckpt=fb_ckpt,
            slow_factor=float(cfg.get("slow_factor", 1.0)),
            render=bool(cfg["render"].get("enabled", True)),
            stages=stages, force=force, resume=not args.no_resume,
            seeds=parse_pairs(args.seed), gt=parse_pairs(args.gt),
            keep_going=args.keep_going,
            chunk_frames=chunk_frames, overlap_frames=overlap_frames,
            force_chunks=parse_ints(args.force_chunk))
    except Exception as e:  # noqa: BLE001 - configuration error, show it plainly
        raise SystemExit(f"configuration error: {e}")
    try:
        summary = pipe.run()
    except StageError as e:
        raise SystemExit(f"pipeline failed: {e}")

    g = summary["gates"]
    print()
    print(f"run dir: {summary['run']}")
    print(f"task gate: {'PASS' if g['task_gate'] else 'FAIL'}"
          f" | render: {g.get('render')}"
          f" | secondary: {g.get('secondary_ok')}"
          f" | boundary: {g.get('boundary_continuity')}"
          f" | segment QC: {g.get('segment_qc')}"
          f" | held-out task gate: {g.get('heldout_task_gate')}")
    ch = summary.get("chunking")
    if ch:
        print(f"chunks: {ch.get('n_chunks')} x {ch.get('chunk_frames')} frames, "
              f"overlap {ch.get('overlap_frames')}, "
              f"{ch.get('n_frames')} frames merged")
    print(f"summary: {Path(summary['run']) / 'qc' / 'summary.md'}")
    ok = bool(g["task_gate"]) and (g.get("render") is not False)
    raise SystemExit(0 if ok and not summary.get("errors") else 1)


if __name__ == "__main__":
    main()
