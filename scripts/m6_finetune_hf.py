"""Fine-tune only the hand/face output rows of a v2 checkpoint on real L0/L1.

The expanded-data body model is trained mostly on synthetic hand/face
evidence; its hand head underperforms on the real MediaPipe L0/L1 test frames.
This script freezes the encoder and every output row except hands/face, and
fine-tunes those rows on base clips (real evidence only) with the same masked
Huber.  It cannot change the body/root correction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.eval_helpers import load_net  # noqa: E402
from vtv.feedback_net import make_windows  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", default=str(ROOT / "outputs/m6/data/index.json"))
    ap.add_argument("--pair-root", default=str(ROOT / "outputs/m6/data"))
    ap.add_argument("--out", default=str(ROOT / "outputs/m6/net"))
    ap.add_argument("--tag", default="m6_v2_hf_ft")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--window", type=int, default=31)
    args = ap.parse_args()
    net, ck = load_net(args.ckpt)
    cfg = net.cfg
    if getattr(cfg, "heads", "shared") != "shared":
        raise SystemExit("fine-tune expects a shared-head checkpoint")
    # freeze everything, then re-enable only the last linear's hand/face rows
    for p in net.parameters():
        p.requires_grad_(False)
    last = net.head[-1]
    last.weight.requires_grad_(True)
    last.bias.requires_grad_(True)
    n_out = last.weight.shape[0]
    keep = np.zeros(n_out, bool)
    keep[FD.HAND_SLICE] = True
    keep[FD.FACE_SLICE] = True
    keep_t = torch.tensor(keep)
    clips = load_json(args.index)["clips"]
    train = [c for c in clips if c["split"] == "train" and c["variant"] == "base"]
    val = [c for c in clips if c["split"] == "val" and c["variant"] == "base"]
    if not train or not val:
        raise SystemExit(f"need base train+val, got {len(train)}/{len(val)}")

    def pack(cs):
        xs, ys, ms = [], [], []
        for c in cs:
            d = np.load(Path(args.pair_root) / c["clip"] / "pair.npz")
            x = (d["features"] - ck["feat_mean"]) / ck["feat_std"]
            xs.append(np.clip(x, -6, 6).astype(np.float32))
            ys.append(d["target"].astype(np.float32))
            ms.append(d["mask"].astype(np.float32))
        return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ms))

    X, Y, M = pack(train)
    VX, VY, VM = pack(val)
    W, TW, MW = make_windows(X, Y, M, args.window)
    limits = torch.tensor(ck["limits"], dtype=torch.float32)
    opt = torch.optim.AdamW([last.weight, last.bias], lr=args.lr,
                            weight_decay=0.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    limits = limits.to(device)
    keep_t = keep_t.to(device)
    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = np.random.permutation(len(W))
        tot, nb = 0.0, 0
        for i in range(0, len(W), args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.tensor(W[idx], device=device)
            yb = torch.tensor(TW[idx], device=device)
            mb = torch.tensor(MW[idx], device=device)
            pred = net(xb)[:, args.window // 2]
            err = (pred - yb) / limits
            hub = torch.where(err.abs() < 1, 0.5 * err ** 2, err.abs() - 0.5)
            # only hand/face channels carry the loss
            sl = keep_t
            loss = ((hub * mb)[:, sl].sum()
                    / mb[:, sl].sum().clamp_min(1.0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # keep body/root rows untouched even under weight decay
            with torch.no_grad():
                last.weight.grad[~keep_t] = 0.0
                last.bias.grad[~keep_t] = 0.0
            opt.step()
            tot += float(loss)
            nb += 1
        net.eval()
        with torch.no_grad():
            nv = len(VX)
            vl = 0.0
            for i in range(0, nv, 256):
                xb = torch.tensor(VX[i:i + 256], device=device)
                yb = torch.tensor(VY[i:i + 256], device=device)
                mb = torch.tensor(VM[i:i + 256], device=device)
                pred = net(xb)
                err = (pred - yb) / limits
                hub = torch.where(err.abs() < 1, 0.5 * err ** 2,
                                  err.abs() - 0.5)
                vl += float(((hub * mb)[:, keep_t].sum()
                             / mb[:, keep_t].sum().clamp_min(1.0))) * len(xb)
            vl /= max(nv, 1)
        print(f"epoch {epoch:3d} train {tot/max(nb,1):.4f} val_hf {vl:.4f}",
              flush=True)
        if vl < best - 1e-6:
            best = vl
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck2 = dict(ck)
    ck2.update({"model": net.state_dict(), "finetune": "handface_real",
                "finetune_val_hf": best, "epoch": args.epochs})
    torch.save(ck2, out / f"{args.tag}_best.pt")
    print("WROTE", out / f"{args.tag}_best.pt", "best_hf", best)


if __name__ == "__main__":
    main()
