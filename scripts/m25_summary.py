"""Summarize M2.5 dropout-bench JSONs with the final gate semantics.

  python scripts/m25_summary.py --bench outputs/m25/r2_bench_v2.json \
      --out outputs/m25/summary_r2.json

Recomputes the gate from the stored per-scenario aggregates (so a gate change
does not require rerunning the expensive PMX evaluation) and prints one row per
view / scenario / pipeline.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from m25_dropout_bench import gate_for  # noqa: E402


def _f(x, fmt="{:.2f}"):
    return "-" if x is None else fmt.format(x)


def summarize(paths):
    out = {"benches": [], "rows": []}
    for path in paths:
        d = json.load(open(path, encoding="utf-8"))
        out["benches"].append(os.path.abspath(path))
        for view, rec in d.get("views", {}).items():
            for sc, srec in rec.get("scenarios", {}).items():
                agg = srec.get("aggregate") or {}
                hold = agg.get("hold") or {}
                for pipe in ("hold", "stream", "offline"):
                    a = agg.get(pipe)
                    if not a:
                        continue
                    gate = None if pipe == "hold" else gate_for(a, hold)
                    row = {
                        "bench": os.path.basename(path),
                        "view": view,
                        "scenario": sc,
                        "pipeline": pipe,
                        "gap_fraction": srec.get("mean_gap_fraction"),
                        "long_gap_frames": a.get("long_gap_frames"),
                        "long_abs_deg": a.get("long_gap_angle_mean_deg"),
                        "long_abs_delta_vs_hold": (
                            None if (a.get("long_gap_angle_mean_deg") is None
                                     or hold.get("long_gap_angle_mean_deg") is None)
                            else round(a["long_gap_angle_mean_deg"]
                                       - hold["long_gap_angle_mean_deg"], 4)),
                        "long_rel_deg": a.get("long_rel_angle_mean_deg"),
                        "freeze_ratio_median": a.get("freeze_ratio_median"),
                        "frozen_frac": a.get("frozen_frac"),
                        "jump_rate": a.get("jump_rate"),
                        "contact_recall": a.get("contact_recall"),
                        "contact_precision": a.get("contact_precision"),
                        "contact_slide_units": a.get("contact_slide_mean_units"),
                        "contact_gt_slide_units": a.get("contact_gt_slide_mean_units"),
                        "contact_penetration_units": a.get("contact_penetration_units"),
                        "gate": gate,
                        "pass": None if gate is None else gate.get("pass"),
                    }
                    out["rows"].append(row)
    return out


def print_table(res):
    print(f"{'view':14s} {'scenario':12s} {'pipe':7s} {'longAbs':>8s} "
          f"{'dAbs':>7s} {'longRel':>8s} {'freeze':>7s} {'frozen':>7s} "
          f"{'jump':>6s} {'cRec':>6s} {'cPrec':>6s} {'slide':>7s} {'pass':>5s}")
    for r in res["rows"]:
        print(f"{r['view'][:14]:14s} {r['scenario']:12s} {r['pipeline']:7s} "
              f"{_f(r['long_abs_deg']):>8s} {_f(r['long_abs_delta_vs_hold'], '{:+.2f}'):>7s} "
              f"{_f(r['long_rel_deg']):>8s} {_f(r['freeze_ratio_median']):>7s} "
              f"{_f(r['frozen_frac']):>7s} {_f(r['jump_rate']):>6s} "
              f"{_f(r['contact_recall']):>6s} {_f(r['contact_precision']):>6s} "
              f"{_f(r['contact_slide_units'], '{:.3f}'):>7s} "
              f"{'-' if r['pass'] is None else ('PASS' if r['pass'] else 'fail'):>5s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "m25",
                                                  "summary.json"))
    args = ap.parse_args()
    res = summarize(args.bench)
    print_table(res)
    n = sum(1 for r in res["rows"] if r["pipeline"] != "hold" and r["pass"])
    m = sum(1 for r in res["rows"] if r["pipeline"] != "hold" and r["pass"] is not None)
    print(f"\npass: {n}/{m}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("WROTE", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
