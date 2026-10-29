"""Multi-view arm upper bound (M2.3 experiment).

Examples:
  python scripts/refine_multiview.py --model si \
      --npz front=outputs/gvhmr/R2fix_front/gvhmr_result.npz \
             three_quarter=outputs/gvhmr/R2fix_three_quarter/gvhmr_result.npz \
             side=outputs/gvhmr/R2fix_side/gvhmr_result.npz \
      --joints upper --out outputs/m2/multiview/all

Writes one corrected npz per view (shared arm body_pose) plus report.json.
This is an upper-bound experiment, not a pipeline stage.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "shim"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "GVHMR"))

from vtv.multiview import fit, write_view_npz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True,
                    help="view=path pairs")
    ap.add_argument("--kp", nargs="*", default=None, help="view=path overrides")
    ap.add_argument("--out", required=True)
    ap.add_argument("--joints", default="upper", choices=("upper", "both", "fore"))
    ap.add_argument("--mode", default="perframe", choices=("perframe", "const"),
                    help="perframe: shared absolute arm pose (upper bound); "
                         "const: shared constant correction on each view's base")
    ap.add_argument("--lam3d", type=float, default=0.1)
    ap.add_argument("--lampose", type=float, default=0.01)
    ap.add_argument("--lamtemp", type=float, default=0.0)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--min-improve", type=float, default=0.03)
    ap.add_argument("--max-delta-deg", type=float, default=15.0)
    ap.add_argument("--max-disp-pct", type=float, default=2.5)
    args = ap.parse_args()

    def pairs(xs):
        out = {}
        for x in xs or []:
            k, v = x.split("=", 1)
            out[k] = v
        return out

    views = pairs(args.npz)
    kp = pairs(args.kp) if args.kp else None
    res = fit(views, kp=kp, joints=args.joints, mode=args.mode,
              lam3d=args.lam3d, lampose=args.lampose, lamtemp=args.lamtemp,
              max_iter=args.max_iter,
              min_improve=args.min_improve, max_delta_deg=args.max_delta_deg,
              max_disp_pct=args.max_disp_pct)
    rep = res["report"]
    print(json.dumps({k: v for k, v in rep.items() if k != "param"},
                     ensure_ascii=False, indent=1))
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    for n in views:
        dst = os.path.join(outdir, f"{n}.npz")
        write_view_npz(views[n], dst, res["param"], res["joints"], rep,
                       mode=res["mode"])
        print("WROTE", dst)
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("WROTE", os.path.join(outdir, "report.json"))
    if not rep["accepted"]:
        print("NOTE: trust region rejected at least one view", file=sys.stderr)


if __name__ == "__main__":
    main()
