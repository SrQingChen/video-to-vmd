"""Collect M3.5 artifacts into one machine-readable summary + markdown tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/m35/summary.json"))
    args = ap.parse_args()
    out = {"data": {}, "renders": {}, "gt": {}, "train": {}, "eval": {},
           "ablations": {}, "t5": {}}

    idx = load(ROOT / "outputs/m35/data/index.json", {"clips": [], "leakage": {}})
    out["data"]["leakage"] = idx.get("leakage")
    clips = idx.get("clips", [])
    out["data"]["clips"] = len(clips)
    out["data"]["by_split"] = {
        s: {"clips": sum(1 for c in clips if c["split"] == s),
            "frames": int(sum(c["frames"] for c in clips if c["split"] == s))}
        for s in ("train", "val", "test")}
    out["data"]["feature_dim"] = sorted({c.get("feature_dim") for c in clips})
    out["data"]["residual_dim"] = sorted({c.get("residual_dim") for c in clips})
    out["data"]["mask_fraction"] = {
        s: {k: float(np.mean([c["mask_fraction"][k] for c in clips
                              if c["split"] == s]))
            for k in ("root", "body", "hands", "face")}
        for s in ("train", "val", "test")}
    hand_counts = {"train": 0, "val": 0, "test": 0}
    for c in clips:
        cert = load(ROOT / "outputs/m35/pred" / c["clip"] / "pipeline.json", {})
        hl = cert.get("hand_levels", {})
        hand_counts[c["split"]] += int(hl.get("l", 0)) + int(hl.get("r", 0))
    out["data"]["hand_L01_frames"] = hand_counts

    out["renders"] = load(ROOT / "outputs/m35/render/manifest.json", {})
    out["gt"] = load(ROOT / "outputs/m35/gt/manifest.json", {})

    hist = load(ROOT / "outputs/m35/net/m35_history.json", {})
    out["train"] = {
        "cfg": (hist or {}).get("cfg"),
        "best_val": (hist or {}).get("best_val"),
        "params": (hist or {}).get("params"),
        "epochs": len((hist or {}).get("history", [])),
        "best_epoch": max(((h["epoch"], h["val"]) for h in
                           (hist or {}).get("history", [])),
                          key=lambda x: -x[1])[0] if (hist or {}).get("history") else None,
    }

    ev = load(ROOT / "outputs/m35/eval/report_default.json")
    if ev is None:
        ev = load(ROOT / "outputs/m35/eval/report_after.json")
    if ev:
        out["eval"] = {"aggregate": ev["aggregate"], "gates": ev["gates"],
                       "clips": [{"clip": r["clip"], "split": r["split"],
                                  "main_before": r["baseline"]["body_relative_main_deg"],
                                  "main_after": r["corrected"]["body_relative_main_deg"],
                                  "pos_before": r["baseline"]["position_pct_height"],
                                  "pos_after": r["corrected"]["position_pct_height"],
                                  "recall_before": r["baseline"]["contact_recall"],
                                  "recall_after": r["corrected"]["contact_recall"],
                                  "slide_before": r["baseline"]["contact_slide_delta"],
                                  "slide_after": r["corrected"]["contact_slide_delta"],
                                  "jerk_before": r["baseline"]["jerk_rms_ratio"],
                                  "jerk_after": r["corrected"]["jerk_rms_ratio"]}
                                 for r in ev["clips"]]}
    for p in sorted((ROOT / "outputs/m35/eval").glob("report_*.json")):
        tag = p.stem.replace("report_", "")
        if tag in ("default", "after"):
            continue
        d = load(p)
        if d:
            out["ablations"][tag] = {"aggregate": d["aggregate"].get("test"),
                                     "gates": d["gates"]}
    out["t5"] = load(ROOT / "outputs/m35/t5_t085a02/t5_report.json")
    if out["t5"] is None:
        out["t5"] = load(ROOT / "outputs/m35/t5_final2/t5_report.json")
    out["t5_full"] = load(ROOT / "outputs/m35/t5/t5_report.json")
    out["t5_render"] = "outputs/m35/t5_final2/render"

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                         default=float), encoding="utf-8")
    print("WROTE", args.out)
    # ---- markdown ---------------------------------------------------------
    md = []
    md.append("# M3.5 machine summary\n")
    md.append("## Data\n")
    md.append(f"- clips {out['data']['clips']}, dims {out['data']['feature_dim']}"
              f" -> {out['data']['residual_dim']}")
    md.append(f"- splits {out['data']['by_split']}")
    md.append(f"- hand L0/L1 frames {out['data']['hand_L01_frames']}")
    md.append(f"- leakage ok: {(out['data']['leakage'] or {}).get('ok')}\n")
    if ev:
        md.append("## Test aggregate (R3)\n")
        t = ev["aggregate"].get("test", {})
        for k in ("body_relative_main_deg", "position_pct_height",
                  "contact_recall", "contact_slide_delta",
                  "contact_penetration_delta", "jerk_rms_ratio"):
            if k in t.get("baseline", {}):
                md.append(f"- {k}: {t['baseline'][k]} -> {t['corrected'][k]}")
        md.append(f"- gates: {json.dumps(ev['gates']['checks'])}\n")
    if out["t5"]:
        md.append("## T5 (no GT)\n")
        for m, r in out["t5"].get("models", {}).items():
            md.append(f"- {m}: ok={r['ok']} checks={r['checks']}")
    p = Path(args.out).with_suffix(".md")
    p.write_text("\n".join(md), encoding="utf-8")
    print("WROTE", p)


if __name__ == "__main__":
    main()
