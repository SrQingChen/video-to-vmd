"""M0 self-test: profiles, VMD round-trip, identity/single-bone, GT replay.

Run with the system Python (pure-Python parts):
  python scripts/m0_selftest.py
"""
import json
import math
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.audit.checks import identity_check, single_bone_check
from vtv.mmd.evaluate import VmdEvaluator
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd
from vtv.mmd.writer import write_vmd

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + str(detail)) if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    with open(os.path.join(ROOT, "configs", "models.json"), encoding="utf-8") as f:
        cfg = json.load(f)["models"]

    # ---- VMD round-trip --------------------------------------------------
    src = parse_vmd(os.path.join(ROOT, "testdata", "R2_王牌竞速", "动作＋表情.vmd"))
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "rt.vmd")
        write_vmd(out, src["model_name"], src["bones"], src["morphs"], src["ik_frames"])
        dst = parse_vmd(out)
    check("vmd_roundtrip_bones", set(src["bones"]) == set(dst["bones"]),
          (len(src["bones"]), len(dst["bones"])))
    check("vmd_roundtrip_ik", src["ik_frames"] == dst["ik_frames"])
    check("vmd_roundtrip_morphs", src["morphs"] == dst["morphs"])

    # ---- profiles + identity / single bone ------------------------------
    for key in cfg:
        ppath = os.path.join(ROOT, "profiles", f"{key}.json")
        check(f"{key}_profile_exists", os.path.exists(ppath))
        if not os.path.exists(ppath):
            continue
        profile = json.load(open(ppath, encoding="utf-8"))
        model = parse_pmx(cfg[key]["pmx"])
        check(f"{key}_standard_joints", not profile["standard_missing"],
              profile["standard_missing"])
        check(f"{key}_fingers", not profile["finger_missing"],
              len(profile["finger_missing"]))
        idc = identity_check(model, profile)
        check(f"{key}_identity", idc["max_err"] < 1e-5, idc)
        sbc = single_bone_check(model, profile)
        check(f"{key}_single_bone", not sbc["failed"], len(sbc["failed"]))

    # ---- safe GT replay --------------------------------------------------
    gt_path = os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd")
    if os.path.exists(gt_path):
        model = parse_pmx(cfg["si"]["pmx"])
        profile = json.load(open(os.path.join(ROOT, "profiles", "si.json"),
                                 encoding="utf-8"))
        gtv = parse_vmd(gt_path)
        ev = VmdEvaluator(model, gtv, force_ik_off=True,
                          apply_all_translations=False)
        name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
        errs = []
        for j, bn in profile["standard_map"].items():
            if bn in name2i:
                i = name2i[bn]
                for f in range(gtv["max_frame"] + 1):
                    p = ev.frame(f)["world_pos"][i]
                    errs.append(math.dist(p, model["bones"][i]["pos"]))
        # not rest comparison; just ensure no NaN and sane magnitude
        check("gt_vmd_finite", all(math.isfinite(x) for x in errs))
    else:
        print("SKIP gt replay (run scripts/bake_gt_vmd.py first)")

    # ---- GVHMR outputs (optional) ---------------------------------------
    for clip in ("T5", "T1"):
        npz = os.path.join(ROOT, "outputs", "gvhmr", clip, "gvhmr_result.npz")
        if not os.path.exists(npz):
            print(f"SKIP gvhmr {clip} (not run)")
            continue
        import numpy as np
        d = np.load(npz)
        ok = True
        n = 0
        for prefix in ("smpl_params_global", "smpl_params_incam"):
            if f"{prefix}/transl" not in d.files:
                ok = False
                continue
            n = d[f"{prefix}/transl"].shape[0]
            ok = ok and bool(np.isfinite(d[f"{prefix}/transl"]).all())
            ok = ok and bool(np.isfinite(d[f"{prefix}/body_pose"]).all())
        check(f"{clip}_gvhmr_params", ok and n > 0, {"frames": n})

    # ---- closed loop copy sanity ----------------------------------------
    base = os.path.join(ROOT, "outputs", "closed_loop", "m0", "baselines.json")
    if os.path.exists(base):
        b = json.load(open(base, encoding="utf-8"))
        copy_v = b["results"]["copy"]["vector"]
        check("closed_loop_copy_zero",
              all(abs(copy_v[k]) < 1e-6 for k in (
                  "angle_mean_deg", "world_angle_mean_deg",
                  "position_pct_height", "contact_slide_delta",
                  "contact_penetration_delta")),
              {k: copy_v[k] for k in ("angle_mean_deg", "position_pct_height")})
        check("closed_loop_no_gate_scalar",
              b["results"]["copy"]["summary_S"].get("is_gate") is False,
              b["results"]["copy"]["summary_S"])
        check("closed_loop_calibration", b.get("checks_ok") is not None
              and b["checks_ok"].split("/")[0] == b["checks_ok"].split("/")[1],
              b.get("checks_ok"))
    else:
        print("SKIP closed loop (run scripts/closed_loop.py first)")

    print()
    if FAILS:
        print("M0 SELFTEST FAILED:", FAILS)
        return 1
    print("M0 SELFTEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
