"""Canonical 2D reprojection stage (M2, non-learned).

  python scripts/refine_2d.py --npz outputs/gvhmr/R2fix_front/gvhmr_result.npz \
      --out outputs/m2/reproj/R2fix_front.npz --joints upper

The output npz keeps every original array and replaces both body_pose slices.
Run ``scripts/run_adapter.py --npz <out>`` next, then ``scripts/clean_vmd.py``.
The trust region may reject the correction (``accepted: false``); the npz is
then a byte-equivalent copy and the adapter output is the unchanged baseline.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "shim"))
sys.path.insert(0, str(ROOT / "third_party" / "GVHMR"))

from vtv.reproj2d import fit, write_npz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kp", default=None,
                    help="ViTPose (17,3) .pt; default npz kp2d or sibling preprocess/vitpose.pt")
    ap.add_argument("--joints", default="upper", choices=["upper", "fore", "both"])
    ap.add_argument("--align", default="raw", choices=["raw", "trans", "sim"])
    ap.add_argument("--mode", default="const", choices=["const", "perframe"])
    ap.add_argument("--lam3d", type=float, default=0.1)
    ap.add_argument("--lampose", type=float, default=0.01)
    ap.add_argument("--lamtemp", type=float, default=0.2)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--min-improve", type=float, default=0.05)
    ap.add_argument("--max-delta-deg", type=float, default=15.0)
    ap.add_argument("--max-disp-pct", type=float, default=2.5)
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    res = fit(args.npz, kp_path=args.kp, joints=args.joints, align=args.align,
              mode=args.mode, lam3d=args.lam3d, lampose=args.lampose,
              lamtemp=args.lamtemp, max_iter=args.max_iter,
              min_improve=args.min_improve,
              max_delta_deg=args.max_delta_deg,
              max_disp_pct=args.max_disp_pct)
    report = res["report"]
    report["out"] = os.path.abspath(args.out) if not args.dry_run else None
    if not args.dry_run:
        write_npz(args.npz, args.out, res["body_pose"], res["delta"],
                  report["accepted"], report, res["err_before"], res["err_after"])
        print("WROTE", args.out)
    rp = args.report or (args.out + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(rp)), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("WROTE", rp)
    print(json.dumps({k: report[k] for k in
                      ("accepted", "before_2d_median_px", "after_2d_median_px",
                       "improve_median", "delta_max_deg",
                       "disp3d_mean_pct_height", "disp3d_max_pct_height")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
