"""Rebuild / print the QC summary of a finished run without re-running stages.

  python scripts/qc_report.py --run outputs/m4/t5
  python scripts/qc_report.py --run outputs/m4/t5 --json

It reads ``models/*/qc.json``, each model's ``feedback.json`` and render
certificate, and rewrites ``qc/summary.json`` + ``qc/summary.md``.  Useful after
re-running one stage with ``run_vtv.py --stage ... --force ...``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv.pipeline import build_run_summary, render_summary_md  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--json", action="store_true", help="print the JSON summary")
    ap.add_argument("--no-write", action="store_true",
                    help="print only; do not rewrite qc/summary.{json,md}")
    ap.add_argument("--models", default=None,
                    help="comma separated model filter for the printed report")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    if not run.exists():
        raise SystemExit(f"run directory not found: {run}")
    summary = build_run_summary(run)
    shown = summary
    if args.models:
        keep = [m.strip() for m in args.models.split(",") if m.strip()]
        shown = json.loads(json.dumps(summary))
        shown["models"] = {k: v for k, v in summary["models"].items() if k in keep}
        shown["gates"]["task_gate"] = bool(shown["models"]) and all(
            v.get("ok") for v in shown["models"].values())
    if args.json:
        print(json.dumps(shown, ensure_ascii=False, indent=1))
    else:
        print(render_summary_md(shown))
    if not args.no_write:
        from vtv.pipeline import write_json
        (run / "qc").mkdir(parents=True, exist_ok=True)
        write_json(run / "qc" / "summary.json", summary)
        (run / "qc" / "summary.md").write_text(render_summary_md(summary),
                                              encoding="utf-8")
    raise SystemExit(0 if shown["gates"]["task_gate"] else 1)


if __name__ == "__main__":
    main()
