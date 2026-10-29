"""Run Multi-HMR on a folder of frames and save GVHMR-compatible SMPL-X params.

This is a benchmark adapter, not a pipeline stage.  The output npz uses the
same keys as ``outputs/gvhmr/*/gvhmr_result.npz`` so ``scripts/run_adapter.py``
can retarget it unchanged:

    smpl_params_global/{body_pose,global_orient,transl,betas}

Multi-HMR predicts one SMPL-X body pose per image in camera space.  The
``body_pose`` is parent-relative and therefore independent of the root; the
global orientation and translation are written only so the adapter can place
the character.  For the body-relative arm gate the root choice cancels.

Example:
    python scripts/mh_infer.py \
        --frames-dir outputs/bench/external/R2/front/frames \
        --out outputs/bench/external/mh/R2_front.npz --fov 60
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MH = os.path.join(ROOT, "third_party", "multi-hmr")
SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mh-dir", default=DEFAULT_MH)
    ap.add_argument("--model-name", default="multiHMR_896_L")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fov", type=float, default=60.0,
                    help="horizontal FOV of the square model input (deg)")
    ap.add_argument("--fx", type=float, default=None,
                    help="focal in model-input pixels; overrides --fov")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--det-thresh", type=float, default=0.3)
    ap.add_argument("--nms-kernel", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    frames_dir = os.path.abspath(args.frames_dir)
    out_path = os.path.abspath(args.out)
    mh_dir = os.path.abspath(args.mh_dir)
    sys.path.insert(0, mh_dir)
    os.chdir(mh_dir)

    import torch
    # Dinov2Backbone hardcodes torch.hub.load('facebookresearch/dinov2', ...),
    # which does a GitHub round-trip on every load.  Use the local hub cache so
    # inference never depends on the network (the R2 run already populated it).
    _hub = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub",
                        "facebookresearch_dinov2_main")
    if os.path.isdir(_hub):
        _orig_hub_load = torch.hub.load

        def _local_hub_load(repo_or_dir, *a, **kw):
            if repo_or_dir == "facebookresearch/dinov2":
                repo_or_dir = _hub
                kw["source"] = "local"
            return _orig_hub_load(repo_or_dir, *a, **kw)

        torch.hub.load = _local_hub_load
    from model import Model
    from utils import normalize_rgb, get_focalLength_from_fieldOfView

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(mh_dir, "models", "multiHMR",
                             args.model_name + ".pt")
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = {}
    for k, v in vars(ckpt["args"]).items():
        kwargs[k] = v
    kwargs["type"] = ckpt["args"].train_return_type
    kwargs["img_size"] = ckpt["args"].img_size[0]
    model = Model(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"loaded {args.model_name}: img_size={model.img_size} "
          f"backbone={kwargs.get('backbone')} device={device}")

    files = sorted(f for f in os.listdir(frames_dir)
                   if f.lower().endswith(SUFFIXES))
    end = len(files) if args.end is None else min(args.end, len(files))
    files = files[args.start:end]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("no frames found")

    # Camera: the model input is a square, aspect-preserving, zero-padded
    # image.  For a 1080x1920 frame the 1920 side maps to img_size, so the
    # focal scales by img_size / max(W, H); the principal point is the centre
    # of the padded square.
    img_size = model.img_size
    first = Image.open(os.path.join(frames_dir, files[0])).convert("RGB")
    scale = img_size / float(max(first.size))
    if args.fx is not None:
        f = float(args.fx)
    else:
        f = get_focalLength_from_fieldOfView(fov=args.fov, img_size=img_size)
    K = torch.tensor([[[f, 0.0, img_size / 2.0],
                       [0.0, f, img_size / 2.0],
                       [0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)

    n = len(files)
    body_pose = np.full((n, 63), np.nan, dtype=np.float32)
    global_orient = np.full((n, 3), np.nan, dtype=np.float32)
    transl = np.full((n, 3), np.nan, dtype=np.float32)
    betas = np.full((n, 10), np.nan, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    num_people = np.zeros(n, dtype=np.int32)
    infer_ms = np.zeros(n, dtype=np.float32)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(frames_dir, fn)).convert("RGB")
        img = ImageOps.contain(img, (img_size, img_size))
        img = ImageOps.pad(img, size=(img_size, img_size))
        x = torch.from_numpy(normalize_rgb(np.asarray(img))).unsqueeze(0).to(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                humans = model(x, is_training=False,
                               nms_kernel_size=int(args.nms_kernel),
                               det_thresh=float(args.det_thresh), K=K)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        infer_ms[i] = dt
        times.append(dt)
        num_people[i] = len(humans)
        if humans:
            # Highest detector score; rows with a larger person are a fallback.
            best = max(humans, key=lambda h: float(h["scores"]))
            rotvec = best["rotvec"].detach().float().cpu().numpy()
            assert rotvec.shape[0] >= 22, rotvec.shape
            global_orient[i] = rotvec[0]
            body_pose[i] = rotvec[1:22].reshape(-1)
            betas[i] = best["shape"].detach().float().cpu().numpy()[:10]
            transl[i] = best["transl_pelvis"].detach().float().cpu().numpy()
            scores[i] = float(best["scores"])
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}  last {dt:.0f} ms  "
                  f"median {np.median(times):.0f} ms  people {num_people[i]}",
                  flush=True)

    out = out_path
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        **{"smpl_params_global/body_pose": body_pose,
           "smpl_params_global/global_orient": global_orient,
           "smpl_params_global/transl": transl,
           "smpl_params_global/betas": betas,
           "smpl_params_incam/body_pose": body_pose,
           "smpl_params_incam/global_orient": global_orient,
           "smpl_params_incam/transl": transl,
           "smpl_params_incam/betas": betas,
           "scores": scores,
           "num_people": num_people,
           "infer_ms": infer_ms,
           "K": np.asarray(K.detach().cpu().numpy()[0]),
           "frames": np.asarray(files, dtype=object)},
    )
    valid = np.isfinite(body_pose).all(1)
    if device.type == "cuda":
        print(f"peak_vram_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")
    print(f"WROTE {out}")
    print(f"frames={n} valid={int(valid.sum())} "
          f"median_ms={np.median(times):.1f} mean_ms={np.mean(times):.1f} "
          f"people>1={int((num_people > 1).sum())} no_person={int((num_people == 0).sum())}")


if __name__ == "__main__":
    main()
