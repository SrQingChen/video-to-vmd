"""Run GVHMR inference without the pytorch3d renderer.

Adapted from third_party/GVHMR/tools/demo/demo.py (MIT).  The demo imports
``hmr4d.utils.vis.renderer`` (pytorch3d.renderer) and unconditionally renders
in-camera/global videos.  This script keeps only preprocessing + prediction
and saves SMPL-X parameters for the VTV pipeline.

Usage (from the repo root, E: venv):
  python scripts/run_gvhmr.py --video testdata/T5_高速舞蹈_001/001.mp4 \
      --static_cam --output outputs/gvhmr/T5
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GVHMR_ROOT = ROOT / "third_party" / "GVHMR"
SHIM = ROOT / "shim"
# The shim must win over any installed pytorch3d.
sys.path.insert(0, str(SHIM))
sys.path.insert(0, str(GVHMR_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_module  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from hmr4d.configs import register_store_gvhmr  # noqa: E402
# Importing DemoPL registers the 'model/gvhmr/gvhmr_pl_demo' config before
# compose(); the demo config references it by name.
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: E402
from hmr4d.utils.geo.hmr_cam import (  # noqa: E402
    convert_K_to_K4, create_camera_sensor, estimate_K, get_bbx_xys_from_xyxy,
)
from hmr4d.utils.geo_transform import compute_cam_angvel  # noqa: E402
from hmr4d.utils.net_utils import detach_to_cpu  # noqa: E402
from hmr4d.utils.pylogger import Log  # noqa: E402
from hmr4d.utils.preproc import Extractor, SimpleVO, Tracker, VitPoseExtractor  # noqa: E402
from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer  # noqa: E402
from pytorch3d.transforms import quaternion_to_matrix  # noqa: E402
from tqdm import tqdm  # noqa: E402

CRF = 23


def parse_args_to_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default=None,
                        help="output directory (default outputs/gvhmr/<stem>)")
    parser.add_argument("-s", "--static_cam", action="store_true",
                        help="fixed camera: skip VO (recommended for T1/T2/T5)")
    parser.add_argument("--use_dpvo", action="store_true")
    parser.add_argument("--f_mm", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    assert video_path.exists(), f"video not found: {video_path}"
    length, width, height = get_video_lwh(video_path)
    out_dir = Path(args.output).resolve() if args.output else (
        ROOT / "outputs" / "gvhmr" / video_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    # GVHMR's config uses relative checkpoint paths.
    os.chdir(GVHMR_ROOT)
    # Hydra's override grammar cannot carry non-ASCII or dotted chars, and the
    # test clips are named like "10月24日0.5.mp4"; the output dir is set
    # explicitly below, so a plain safe name is enough here.
    safe_stem = "".join(c if (c.isascii() and (c.isalnum() or c in "-_"))
                        else "_" for c in video_path.stem)
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=[
            f"video_name={safe_stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"use_dpvo={args.use_dpvo}",
        ] + ([f"f_mm={args.f_mm}"] if args.f_mm is not None else []))
    # Windows paths must not go through Hydra's override grammar.
    OmegaConf.set_struct(cfg, False)
    cfg.output_dir = str(out_dir)
    cfg.preprocess_dir = str(out_dir / "preprocess")
    cfg.source_video = str(video_path)
    cfg.video_path = str(out_dir / "0_input_video.mp4")
    cfg.paths.bbx = str(out_dir / "preprocess" / "bbx.pt")
    cfg.paths.vitpose = str(out_dir / "preprocess" / "vitpose.pt")
    cfg.paths.vit_features = str(out_dir / "preprocess" / "vit_features.pt")
    cfg.paths.slam = str(out_dir / "preprocess" / "slam_results.pt")
    cfg.paths.hmr4d_results = str(out_dir / "hmr4d_results.pt")
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    Log.info(f"[Input] {video_path}  (L,W,H)=({length},{width},{height})")
    Log.info(f"[Output] {out_dir}")
    return cfg, args


def prepare_30fps_video(src, dst, target_fps=30):
    """Downsample to VMD-native 30fps while preserving duration."""
    import cv2
    src, dst = str(src), str(dst)
    cap = cv2.VideoCapture(src)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    cap.release()
    step = max(1, int(round(src_fps / target_fps)))
    if Path(dst).exists():
        return step
    reader = get_video_reader(src)
    writer = get_writer(dst, fps=target_fps, crf=CRF)
    n = get_video_lwh(src)[0]
    for i, img in enumerate(tqdm(reader, total=n, desc="Downsample 30fps")):
        if i % step == 0:
            writer.write_frame(img)
    writer.close()
    reader.close()
    return step


@torch.no_grad()
def run_preprocess(cfg):
    Log.info("[Preprocess] start")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths

    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        Log.info(f"[Preprocess] bbx <- {paths.bbx}")
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]

    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        Log.info(f"[Preprocess] vitpose <- {paths.vitpose}")

    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features <- {paths.vit_features}")

    if not cfg.static_cam:
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8,
                                     method="sift", f_mm=cfg.f_mm)
                torch.save(simple_vo.compute(), paths.slam)
            else:
                from hmr4d.utils.preproc.slam import SLAMModel
                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                slam = SLAMModel(video_path, width, height,
                                 convert_K_to_K4(K_fullimg), buffer=4000, resize=0.5)
                bar = tqdm(total=length, desc="DPVO")
                while slam.track():
                    bar.update()
                torch.save(slam.process(), paths.slam)
        else:
            Log.info(f"[Preprocess] slam <- {paths.slam}")
    Log.info(f"[Preprocess] done in {Log.time() - tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(paths.slam)
        if cfg.use_dpvo:
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        else:
            R_w2c = torch.from_numpy(traj[:, :3, :3])
    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    return {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }


def save_results(cfg, pred, data):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(pred, cfg.paths.hmr4d_results)

    length, width, height = get_video_lwh(cfg.video_path)
    arrays = {
        "bbx_xys": data["bbx_xys"].cpu().numpy(),
        "bbx_xyxy": torch.load(cfg.paths.bbx)["bbx_xyxy"].cpu().numpy(),
        "K_fullimg": data["K_fullimg"].cpu().numpy(),
        "cam_angvel": data["cam_angvel"].cpu().numpy(),
        # Keeping the ViTPose COCO-17 evidence in the npz makes the 2D
        # reprojection stage self-contained (no re-run of preprocessing).
        "kp2d": data["kp2d"].cpu().numpy(),
    }
    for key in ("smpl_params_global", "smpl_params_incam"):
        val = pred.get(key)
        if val is None:
            continue
        for sub, tensor in val.items():
            if torch.is_tensor(tensor):
                arrays[f"{key}/{sub}"] = tensor.cpu().numpy()
            else:
                arrays[f"{key}/{sub}"] = np.asarray(tensor)
    npz = out / "gvhmr_result.npz"
    np.savez_compressed(npz, **arrays)
    meta = {
        "video": cfg.video_path,
        "source_video": getattr(cfg, "source_video", cfg.video_path),
        "source_fps_step": int(getattr(cfg, "video_step", 1)),
        "frames": int(length),
        "width": int(width),
        "height": int(height),
        "fps": 30,
        "static_cam": bool(cfg.static_cam),
        "f_mm": cfg.f_mm,
        "npz": str(npz),
        "pt": str(cfg.paths.hmr4d_results),
        "keys": sorted(arrays),
    }
    with open(out / "gvhmr_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    Log.info(f"[Save] {npz}")
    Log.info(f"[Save] {cfg.paths.hmr4d_results}")
    return npz


if __name__ == "__main__":
    cfg, args = parse_args_to_cfg()
    Log.info(f"[GPU] {torch.cuda.get_device_name()}")
    step = prepare_30fps_video(cfg.source_video, cfg.video_path, target_fps=30)
    Log.info(f"[Video] 30fps step={step} -> {cfg.video_path}")
    cfg.video_step = step
    run_preprocess(cfg)
    data = load_data_dict(cfg)

    if Path(cfg.paths.hmr4d_results).exists():
        Log.info(f"[HMR4D] cached <- {cfg.paths.hmr4d_results}")
        pred = torch.load(cfg.paths.hmr4d_results)
    else:
        from hydra.utils import instantiate
        Log.info("[HMR4D] predicting")
        model: DemoPL = instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        Log.info(f"[HMR4D] done in {Log.sync_time() - tic:.2f}s")
    save_results(cfg, pred, data)
