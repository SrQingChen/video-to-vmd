"""Train the M3.5 bounded residual network on the paired data engine output.

Split (whole segments, from vtv.feedback_data):
  train  R1/R2 x {si,luoxi} x {front,three_quarter} (+ observation-only aug)
  val    R1/R2 x held-out model (tiflos) and/or held-out camera (side)
  test   R3 x all models x all cameras (never touched here)

Usage:
  python scripts/train_feedback.py --epochs 200 --arch gru --bidirectional
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.feedback_net import (  # noqa: E402
    FeedbackNet, NetConfig, group_losses, make_windows, masked_rmse, select_device,
)


def load_index(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["clips"]


def load_pair(meta, pair_root=None):
    root = Path(pair_root) if pair_root else ROOT / "outputs/m35/data"
    p = root / meta["clip"] / "pair.npz"
    d = np.load(p)
    return (d["features"].astype(np.float32), d["target"].astype(np.float32),
            d["mask"].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / "outputs/m35/data/index.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/m35/net"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--window", type=int, default=31)
    ap.add_argument("--arch", default="gru",
                    choices=["gru", "mlp", "tcn", "transformer", "mrn", "graph"])
    ap.add_argument("--heads", default="shared", choices=["shared", "factor"])
    ap.add_argument("--causal", action="store_true",
                    help="uni-directional/causal encoder (gru/tcn/transformer)")
    ap.add_argument("--bounds", default="v1", choices=["v1", "v2"])
    ap.add_argument("--feature-set", default="v1", choices=["v1", "v2"])
    ap.add_argument("--pair-root", default=None,
                    help="pair.npz root (default outputs/m35/data)")
    ap.add_argument("--hand-weight", type=float, default=0.5)
    ap.add_argument("--face-weight", type=float, default=0.5)
    ap.add_argument("--select", default="total", choices=["total", "body"],
                    help="validation loss used for early stopping")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="m35")
    ap.add_argument("--train-clips", default=None,
                    help="comma separated clip ids (ablation); default all train")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    clips = load_index(args.index)
    train = [c for c in clips if c["split"] == "train"]
    val = [c for c in clips if c["split"] == "val"]
    if args.train_clips:
        keep = set(args.train_clips.split(","))
        train = [c for c in train if c["clip"] in keep]
    if not train or not val:
        raise SystemExit(f"need train+val pairs, got {len(train)}/{len(val)}")

    xtr, ytr, mtr = [], [], []
    for c in train:
        x, y, m = load_pair(c, args.pair_root)
        xtr.append(x); ytr.append(y); mtr.append(m)
    X = np.concatenate(xtr); Y = np.concatenate(ytr); M = np.concatenate(mtr)
    # Channels with no training supervision must stay at baseline: an
    # untrained L0 face channel (e.g. surprise) would otherwise emit an
    # arbitrary morph on the test motion.
    face_supported = (M[:, FD.FACE_SLICE] > 0).any(axis=0).astype(np.float32)
    hand_supported = np.asarray(
        [(M[:, FD.HAND_SLICE][:, s * 19:(s + 1) * 19] > 0).any()
         for s in range(2)], np.float32)
    print("face_supported", face_supported, "hand_supported", hand_supported)
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-3)
    Xn = np.clip((X - mean) / std, -6.0, 6.0)

    val_set = [(c, *load_pair(c, args.pair_root)) for c in val]
    for c, x, y, m in val_set:
        assert len(x) == len(y) == len(m)
    feat_dim = X.shape[1]
    limits = FD.residual_limits(args.bounds)
    cfg = NetConfig(feat_dim=feat_dim, out_dim=len(limits), hidden=args.hidden,
                    layers=args.layers, dropout=args.dropout, arch=args.arch,
                    bidirectional=not args.causal, window=args.window,
                    heads=args.heads, causal=args.causal)
    device = "cpu" if args.cpu else select_device()
    net = FeedbackNet(cfg, limits).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    W, TW, MW = make_windows(Xn, Y, M, args.window)
    n = len(W)
    print(f"train windows {n}, clips {len(train)} (aug included), "
          f"val clips {len(val)}, dims {feat_dim}->{len(limits)}, "
          f"params {net.n_params()/1e6:.2f}M, device {device}")

    group_slices = FD.residual_groups()
    # Body corrections are small in radians (0.5 deg typical) while the root
    # target is large and is disabled at application by the contact gate.  The
    # body group is up-weighted so its Huber (~1e-3) is comparable to the
    # other groups; otherwise the main gate would get almost no gradient.
    gw = {"root": 0.05, "body": 10.0,
          "hands": float(args.hand_weight), "face": float(args.face_weight)}
    lim_t = torch.tensor(limits, device=device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_p = out_dir / f"{args.tag}_best.pt"
    best = float("inf")
    bad = 0
    history = []
    t0 = time.time()

    def eval_val():
        net.eval()
        tot, parts, nf = 0.0, {}, 0
        with torch.no_grad():
            for c, x, y, m in val_set:
                xn = np.clip((x - mean) / std, -6.0, 6.0)
                tx = torch.tensor(xn[None], device=device)
                ty = torch.tensor(y[None], device=device)
                tm = torch.tensor(m[None], device=device)
                pred = net(tx)
                loss, p = group_losses(pred, ty, tm, lim_t, group_slices, gw)
                tot += float(loss) * len(x)
                nf += len(x)
                for k, v in p.items():
                    parts[k] = parts.get(k, 0.0) + float(v) * len(x)
        return tot / max(nf, 1), {k: v / max(nf, 1) for k, v in parts.items()}

    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = np.random.permutation(n)
        run, nb = 0.0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.tensor(W[idx], device=device)
            yb = torch.tensor(TW[idx], device=device)
            mb = torch.tensor(MW[idx], device=device)
            full = net(xb)
            pred = full[:, args.window // 2]       # target is the centre frame
            loss, _ = group_losses(pred, yb, mb, lim_t, group_slices, gw,
                                   smooth_weight=0.0, mag_weight=0.0)
            # sequence regularisers on the whole window prediction
            loss = loss + 0.05 * ((full[:, 1:] - full[:, :-1]) / lim_t).pow(2).mean()
            loss = loss + 0.01 * (full / lim_t).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += float(loss)
            nb += 1
        sched.step()
        vl, vparts = eval_val()
        vl_sel = vparts.get("body", vl) if args.select == "body" else vl
        rec = {"epoch": epoch, "train": run / max(nb, 1), "val": vl,
               "val_select": vl_sel,
               "val_parts": vparts, "lr": sched.get_last_lr()[0],
               "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"epoch {epoch:3d} train {run/max(nb,1):.4f} val {vl:.4f} "
              f"(sel {vl_sel:.4f} body {vparts.get('body', float('nan')):.4f} "
              f"root {vparts.get('root', float('nan')):.4f} "
              f"hands {vparts.get('hands', float('nan')):.4f})", flush=True)
        if vl_sel < best - 1e-6:
            best = vl_sel
            bad = 0
            torch.save({
                "model": net.state_dict(), "cfg": cfg.to_dict(),
                "feat_mean": mean, "feat_std": std, "limits": limits,
                "feature_names": FD.feature_names(None, args.feature_set),
                "feature_set": args.feature_set,
                "bounds": args.bounds,
                "face_supported": face_supported,
                "hand_supported": hand_supported,
                "epoch": epoch, "val_loss": vl, "val_select": vl_sel,
                "val_parts": vparts,
                "train_clips": [c["clip"] for c in train],
                "val_clips": [c["clip"] for c in val],
            }, ckpt_p)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch} (best {best:.4f})")
                break

    with open(out_dir / f"{args.tag}_history.json", "w", encoding="utf-8") as f:
        json.dump({"cfg": cfg.to_dict(), "best_val": best, "history": history,
                   "params": net.n_params(), "device": device}, f, indent=1)
    net.load_state_dict(torch.load(ckpt_p, map_location="cpu")["model"])
    net.eval()
    # final val RMSE in physical units
    preds, ys, ms = [], [], []
    with torch.no_grad():
        for c, x, y, m in val_set:
            xn = np.clip((x - mean) / std, -6.0, 6.0)
            preds.append(net(torch.tensor(xn[None], device=device))[0].cpu().numpy())
            ys.append(y); ms.append(m)
    rmse = masked_rmse(np.concatenate(preds), np.concatenate(ys),
                       np.concatenate(ms), limits)
    print("val masked RMSE (normalised by limits):", rmse)
    (out_dir / f"{args.tag}_val_rmse.json").write_text(
        json.dumps(rmse, indent=1), encoding="utf-8")
    print("WROTE", ckpt_p)


if __name__ == "__main__":
    main()
