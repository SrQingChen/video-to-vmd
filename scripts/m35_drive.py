"""Overlap GVHMR (GPU subprocess) with pipeline + paired-data (CPU).

After the render grid exists, this driver:

  thread 1: run GVHMR for every clip, base clips first, waiting for augmented
            videos to appear
  main:     for each clip whose GVHMR result is ready, run
            adapter -> clean -> contact IK -> hands/face -> assemble -> pairs

Both halves are resumable (each stage skips existing artifacts), so an
interrupted run can simply be restarted.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import m35_data_engine as DE  # noqa: E402


def main():
    clips = DE.all_clips()
    status = {}
    lock = threading.Lock()

    def gvhmr_worker():
        for cid in clips:
            try:
                if "__" in cid:
                    v = DE.paths(cid, "render")
                    waited = 0
                    while not v.exists():
                        time.sleep(5)
                        waited += 5
                        if waited > 4 * 3600:
                            raise TimeoutError(f"augmented video missing: {v}")
                r = DE.stage_gvhmr(cid)
                with lock:
                    status[cid] = ("gvhmr_ok", r)
                print(f"GVHMR {cid}: {r}", flush=True)
            except Exception as e:  # noqa: BLE001
                with lock:
                    status[cid] = ("gvhmr_fail", str(e))
                print(f"GVHMR FAIL {cid}: {e}", flush=True)
                traceback.print_exc()

    th = threading.Thread(target=gvhmr_worker, daemon=True)
    th.start()

    for i, cid in enumerate(clips, 1):
        while True:
            with lock:
                st = status.get(cid)
            if st is not None:
                break
            time.sleep(2)
        if st[0] != "gvhmr_ok":
            print(f"[{i}/{len(clips)}] SKIP {cid}: {st[1]}", flush=True)
            continue
        print(f"[{i}/{len(clips)}] {cid}", flush=True)
        try:
            print("  pipeline:", DE.stage_pipeline(cid), flush=True)
            print("  pairs:", DE.stage_pairs(cid), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {cid}: {e}", flush=True)
            traceback.print_exc()

    th.join(timeout=60)
    idx = []
    for cid in clips:
        mp = DE.paths(cid, "pair") / "meta.json"
        if mp.exists():
            with open(mp, encoding="utf-8") as f:
                idx.append(json.load(f))
    out = ROOT / "outputs/m35/data"
    out.mkdir(parents=True, exist_ok=True)
    leak = DE.FD.leakage_report(idx)
    (out / "index.json").write_text(
        json.dumps({"clips": idx, "leakage": leak}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("LEAKAGE", json.dumps(leak, ensure_ascii=False))
    print("DRIVE_DONE", len(idx), "/", len(clips))


if __name__ == "__main__":
    main()
