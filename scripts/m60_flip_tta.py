"""M6.0 benchmark: GVHMR horizontal-flip test-time augmentation (TTA).

The current HMR (GVHMR) already implements a flip test in its training/test
pipeline (``hmr4d/model/gvhmr/gvhmr_pl.py``): run the model on the mirrored
video, flip the SMPL-X parameters back with ``flip_smplx_params`` and average
the in-camera body pose with ``avg_smplx_aa``.  The released demo path used by
this repository does not expose it, so this script reproduces the same average
outside the model and scores the result with the M3.5 metric vector.

Protocol (small-sample, M6.0 gate):
  * three held-out R3 clips (si front / three_quarter / side);
  * baseline = existing GVHMR npz -> adapter -> cleanup -> contact IK;
  * candidate = flipped-video GVHMR -> flip back -> average body pose/betas ->
    the same adapter/cleanup/contact IK;
  * compare body-relative main, chest/upperarm/forearm, world angle, position,
    contact and jerk.  No GT is used to build the candidate.

This is a benchmark, not a production path.  Outputs live under
``outputs/m60/flip_tta`` and the weights are not redistributed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "shim"))
sys.path.insert(0, str(ROOT / "third_party" / "GVHMR"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.cleanup import clean_vmd  # noqa: E402
from vtv.contact_ik import solve_contact_ik  # noqa: E402
from vtv.metrics.core import compute_metrics, pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402

PY = sys.executable
METRIC_KEYS = (
    "angle_mean_deg", "world_angle_mean_deg", "position_pct_height",
    "position_p99_pct_height", "root_traj_pct_height", "root_drift_pct_height",
    "contact_recall", "contact_precision", "contact_slide_delta",
    "contact_penetration_delta", "contact_height_mae_pct_height",
    "jerk_rms_ratio", "spike_delta",
)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def model_profile(model_key):
    cfg = load_json(ROOT / "configs/models.json")["models"][model_key]
    profile = load_json(ROOT / f"profiles/{model_key}.json")
    return parse_pmx(cfg["pmx"]), profile


def flip_video(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or "D:/ffmpeg/bin/ffmpeg.exe"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
           "-vf", "hflip", "-c:v", "libx264", "-crf", "18",
           "-preset", "veryfast", "-an", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg hflip failed for {src}: {r.stderr[-1000:]}")


def run_gvhmr(video: Path, out: Path):
    npz = out / "gvhmr_result.npz"
    if npz.exists():
        return npz
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(ROOT / "scripts/run_gvhmr.py"), "--video", str(video),
           "--static_cam", "--output", str(out)]
    log = out / "gvhmr.log"
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           cwd=str(ROOT))
    if r.returncode != 0 or not npz.exists():
        raise SystemExit(f"GVHMR flip run failed: see {log}")
    return npz


def averaged_npz(orig_path: Path, flip_path: Path, out_path: Path) -> Path:
    """Reproduce GVHMR's flip test: flip back and average incam body pose."""
    if out_path.exists():
        return out_path
    import torch
    from hmr4d.utils.geo.flip_utils import avg_smplx_aa, flip_smplx_params

    o = np.load(orig_path)
    f = np.load(flip_path)
    n = min(len(o["smpl_params_global/body_pose"]),
            len(f["smpl_params_incam/body_pose"]))
    keys = ("body_pose", "global_orient", "betas", "transl")
    orig = {k: torch.tensor(o[f"smpl_params_global/{k}"][:n], dtype=torch.float32)
            for k in keys}
    flic = {k: torch.tensor(f[f"smpl_params_incam/{k}"][:n], dtype=torch.float32)
            for k in keys}
    back = flip_smplx_params(flic)
    orig_incam = {k: torch.tensor(o[f"smpl_params_incam/{k}"][:n],
                                  dtype=torch.float32) for k in keys}
    body_avg = avg_smplx_aa(orig_incam["body_pose"], back["body_pose"])
    orient_incam_avg = avg_smplx_aa(orig_incam["global_orient"],
                                    back["global_orient"])
    betas_avg = (orig_incam["betas"] + back["betas"]) / 2.0
    # Official GVHMR flip test: average the *incam* pose, copy body_pose and
    # betas into the global prediction, but keep the global root orientation
    # from the unflipped run (averaging global and incam orientation mixes two
    # different frames and produces a ~90 deg root yaw error).
    out = {k: o[k] for k in o.files}
    for k in list(out):
        if k.startswith("smpl_params_"):
            del out[k]
    out["smpl_params_global/body_pose"] = body_avg.cpu().numpy()
    out["smpl_params_global/global_orient"] = o["smpl_params_global/global_orient"][:n]
    out["smpl_params_global/betas"] = betas_avg.cpu().numpy()
    out["smpl_params_global/transl"] = o["smpl_params_global/transl"][:n]
    out["smpl_params_incam/body_pose"] = body_avg.cpu().numpy()
    out["smpl_params_incam/global_orient"] = orient_incam_avg.cpu().numpy()
    out["smpl_params_incam/betas"] = betas_avg.cpu().numpy()
    out["smpl_params_incam/transl"] = o["smpl_params_incam/transl"][:n]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    return out_path


def run_pipeline(npz: Path, model_key: str, out_dir: Path, kp: np.ndarray):
    model, profile = model_profile(model_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_p = out_dir / "adapter.vmd"
    cmd = [PY, str(ROOT / "scripts/run_adapter.py"), "--npz", str(npz),
           "--model", model_key, "--out", str(adapter_p)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0 or not adapter_p.exists():
        raise SystemExit(f"adapter failed: {r.stderr[-1000:]}")
    clean = clean_vmd(parse_vmd(str(adapter_p)), model, profile)
    ik = solve_contact_ik(clean["vmd"], model, profile, kp2d=kp)
    write_vmd(str(out_dir / "ik.vmd"), ik["vmd"].get("model_name", ""),
              ik["vmd"]["bones"], {}, ik["vmd"].get("ik_frames", []))
    return out_dir / "ik.vmd", model, profile


def score(gt_vmd: Path, pred_vmd: Path, model, profile):
    gt = parse_vmd(str(gt_vmd))
    pred = parse_vmd(str(pred_vmd))
    n = min(int(gt["max_frame"]), int(pred["max_frame"])) + 1
    frames = list(range(n))
    gt_t = pose_track(model, gt, profile, frames=frames)
    pr_t = pose_track(model, pred, profile, frames=frames)
    m = compute_metrics(gt_t, pr_t, profile)
    rel = FD.body_relative_errors(gt_t, pr_t)
    out = {k: m["vector"][k] for k in METRIC_KEYS}
    out.update({k: v for k, v in rel.items() if k.endswith("_deg")})
    out["frames"] = n
    return out


def t5_entry():
    return {
        "clip": "T5_si",
        "video": ROOT / "testdata/T5_高速舞蹈_001/001.mp4",
        "npz": ROOT / "outputs/m4/t5/video/gvhmr/gvhmr_result.npz",
        "model": "si",
    }


def stability_of(vmd_path: Path, model, profile):
    from vtv.eval_helpers import stability
    return stability(model, profile, parse_vmd(str(vmd_path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="R3_si_front,R3_si_three_quarter,R3_si_side")
    ap.add_argument("--root", default=str(ROOT / "outputs/m60/flip_tta"))
    ap.add_argument("--t5", action="store_true",
                    help="no-GT T5 stability check (si) instead of R3 scoring")
    args = ap.parse_args()
    out_root = Path(args.root)
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    if args.t5:
        e = t5_entry()
        work = out_root / e["clip"]
        flip_video(e["video"], work / "flip.mp4")
        flip_out = run_gvhmr(work / "flip.mp4", work / "gvhmr_flip")
        avg = averaged_npz(e["npz"], flip_out, work / "gvhmr_avg.npz")
        kp = np.load(e["npz"])["kp2d"]
        base_vmd, model, profile = run_pipeline(e["npz"], e["model"],
                                                work / "base", kp)
        cand_vmd, _, _ = run_pipeline(avg, e["model"], work / "flipavg", kp)
        b = stability_of(base_vmd, model, profile)
        c = stability_of(cand_vmd, model, profile)
        rec = {"clip": e["clip"], "baseline": b, "flip_tta": c,
               "delta": {k: round(c[k] - b[k], 5) for k in b
                         if isinstance(b[k], (int, float)) and k != "frames"}}
        records.append(rec)
        print("T5 stability", json.dumps(rec, ensure_ascii=False), flush=True)
        report = {"clips": records, "protocol": "T5 no-GT stability; si"}
        (out_root / "report_t5.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print("WROTE", out_root / "report_t5.json")
        return
    for cid in args.clips.split(","):
        cid = cid.strip()
        if not cid:
            continue
        parts = cid.split("_")
        motion, model_key = parts[0], parts[1]
        video = ROOT / "outputs/m35/render" / cid / f"{cid}.mp4"
        orig_npz = ROOT / "outputs/m35/gvhmr" / cid / "gvhmr_result.npz"
        gt_vmd = ROOT / f"outputs/m35/gt/{motion}/{model_key}.vmd"
        if not video.exists() or not orig_npz.exists():
            print(f"SKIP {cid}: missing render or GVHMR npz", flush=True)
            continue
        work = out_root / cid
        flip_video(video, work / "flip.mp4")
        flip_out = run_gvhmr(work / "flip.mp4", work / "gvhmr_flip")
        avg = averaged_npz(orig_npz, flip_out, work / "gvhmr_avg.npz")
        kp = np.load(orig_npz)["kp2d"]
        base_vmd, model, profile = run_pipeline(orig_npz, model_key,
                                                work / "base", kp)
        cand_vmd, _, _ = run_pipeline(avg, model_key, work / "flipavg", kp)
        b = score(gt_vmd, base_vmd, model, profile)
        c = score(gt_vmd, cand_vmd, model, profile)
        d = {k: round(c[k] - b[k], 5) for k in b
             if k != "frames" and b[k] is not None and c[k] is not None}
        rec = {"clip": cid, "baseline": b, "flip_tta": c, "delta": d}
        records.append(rec)
        print(f"{cid}: main {b['body_relative_main_deg']:.2f}->"
              f"{c['body_relative_main_deg']:.2f} "
              f"chest {b['group_chest_deg']:.2f}->{c['group_chest_deg']:.2f} "
              f"pos {b['position_pct_height']:.3f}->{c['position_pct_height']:.3f} "
              f"jerk {b['jerk_rms_ratio']:.3f}->{c['jerk_rms_ratio']:.3f}",
              flush=True)
    agg = {}
    for k in (records[0]["baseline"] if records else []):
        if k == "frames":
            continue
        vals_b = [r["baseline"][k] for r in records if r["baseline"][k] is not None]
        vals_c = [r["flip_tta"][k] for r in records if r["flip_tta"][k] is not None]
        if not vals_b or not vals_c:
            agg[k] = {"baseline": None, "flip_tta": None, "delta": None}
            continue
        agg[k] = {
            "baseline": float(np.mean(vals_b)),
            "flip_tta": float(np.mean(vals_c)),
        }
        agg[k]["delta"] = round(agg[k]["flip_tta"] - agg[k]["baseline"], 5)
    report = {"clips": records, "aggregate": agg,
              "protocol": "GVHMR flip test, 3 held-out R3 clips; no GT in candidate"}
    (out_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("WROTE", out_root / "report.json")


if __name__ == "__main__":
    main()
