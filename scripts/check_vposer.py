"""Load and smoke-test the local VPoser v2 checkpoint.

Usage:
  python scripts/check_vposer.py
  python scripts/check_vposer.py --ckpt models/vposer/V02_05/snapshots/V02_05_epoch=13_val_loss=0.03.ckpt

The VPoser v2 checkpoint is a Lightning checkpoint whose state_dict keys are
prefixed with ``vp_model.``; ``bm_train`` buffers are not part of the pose
prior and are ignored. The official VPoser v2 model definition is kept under
``models/vposer/code/`` because it is MPG-licensed and is not redistributed
with this repository.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CKPT = os.path.join(
    ROOT, "models", "vposer", "V02_05", "snapshots",
    "V02_05_epoch=13_val_loss=0.03.ckpt",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--npz", default=None, help="optional GVHMR npz for a real body_pose smoke test")
    ap.add_argument("--neurons", type=int, default=512)
    ap.add_argument("--latent", type=int, default=32)
    args = ap.parse_args()

    code_dir = os.path.join(ROOT, "models", "vposer", "code")
    sys.path.insert(0, code_dir)
    from vposer_model import VPoser  # noqa: E402

    obj = torch.load(args.ckpt, map_location="cpu")
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    state = {k[len("vp_model."):]: v for k, v in state.items() if k.startswith("vp_model.")}
    if not state:
        raise SystemExit("no vp_model.* weights found in checkpoint")

    model_ps = SimpleNamespace(
        model_params=SimpleNamespace(num_neurons=args.neurons, latentD=args.latent)
    )
    model = VPoser(model_ps)
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        x = torch.zeros(2, 63)
        q = model.encode(x)
        out = model.decode(q.mean)
    print("VPoser v2 loaded")
    print("  checkpoint:", args.ckpt)
    print("  encoder mean/std:", tuple(q.mean.shape), tuple(q.scale.shape))
    print("  decoded pose_body:", tuple(out["pose_body"].shape))
    print("  decoded matrot:", tuple(out["pose_body_matrot"].shape))
    print("  missing:", list(missing), "unexpected:", list(unexpected))

    if args.npz:
        import numpy as np
        d = np.load(args.npz)
        x = torch.tensor(d["smpl_params_global/body_pose"], dtype=torch.float32)
        with torch.no_grad():
            q = model.encode(x)
            out = model.decode(q.mean)
            kl = 0.5 * (q.mean ** 2 + q.scale ** 2 - 1 - 2 * torch.log(q.scale)).sum(-1)
        print("  npz:", args.npz)
        print("  real body_pose:", tuple(x.shape))
        print("  decoded:", tuple(out["pose_body"].shape), "finite:", bool(torch.isfinite(out["pose_body"]).all()))
        print("  latent KL mean/median:", float(kl.mean()), float(kl.median()))


if __name__ == "__main__":
    main()
