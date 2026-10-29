"""D-chain / inherit regression test.

Two layers:
  * pure Python (always): MMD assets must carry NO inherit-helper tracks
    (D chain, 捩1, 肩C/P) and every model must declare the D chain;
  * Blender (``--blender``): bake a 60-degree left-knee VMD and assert that
    (a) Blender's deformation matches the PMX evaluator, (b) the D chain --
    which owns the leg mesh weights -- actually bends.

Run:
  python tests/test_inherit_bake.py
  python tests/test_inherit_bake.py --blender
  VTV_RUN_BLENDER_TESTS=1 python scripts/m0_selftest.py
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402

BLENDER = os.environ.get("VTV_BLENDER", r"D:\blender\blender.exe")
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + str(detail)) if detail else ""))
    if not cond:
        FAILS.append(name)


def profiles():
    out = {}
    for name in ("si", "tiflos", "luoxi"):
        p = os.path.join(ROOT, "profiles", f"{name}.json")
        if os.path.exists(p):
            out[name] = json.load(open(p, encoding="utf-8"))
    return out


def forbidden_names(profile):
    bad = set()
    for side in ("l", "r"):
        for rec in profile.get("d_chain", {}).get(side, []):
            bad.add(rec["name"])
        tw = profile.get("twist", {})
        for key in (f"forearm_{side}", f"wrist_{side}"):
            if key in tw and tw[key].get("helper"):
                bad.add(tw[key]["helper"])
        pc = profile.get("pc", {}).get(side, {})
        for k in ("p", "c"):
            if pc.get(k):
                bad.add(pc[k])
    return bad


def pure_checks():
    profs = profiles()
    check("profiles_loaded", len(profs) == 3, sorted(profs))
    for key, prof in profs.items():
        dchain = prof.get("d_chain", {})
        ok = all(len(dchain.get(s, [])) == 3 for s in ("l", "r"))
        check(f"{key}_d_chain_declared", ok,
              {s: [b["name"] for b in dchain.get(s, [])] for s in ("l", "r")})
        for s in ("l", "r"):
            for rec in dchain.get(s, []):
                inh = rec.get("inherit") or {}
                check(f"{key}_d_inherit_{rec['name']}",
                      abs(float(inh.get("ratio", 0)) - 1.0) < 1e-9,
                      inh)
        wl = set((prof.get("writable_bones") or {}).values())
        bad = forbidden_names(prof) & wl
        check(f"{key}_helpers_not_writable", not bad, sorted(bad))

    assets = {
        "si": [
            ("outputs/gt/si_R2_safe.vmd", "si"),
            ("outputs/t5_m1/t5_242.vmd", "si"),
            ("outputs/t1_m1/t1_242.vmd", "si"),
            ("outputs/t2_m1/t2_242.vmd", "si"),
            ("outputs/adapter/si_R2_to_tiflos.vmd", "tiflos"),
            ("outputs/adapter/si_R2_to_luoxi.vmd", "luoxi"),
        ],
    }
    for key, items in assets.items():
        for rel, model in items:
            path = os.path.join(ROOT, rel)
            if not os.path.exists(path):
                print("SKIP missing asset", rel)
                continue
            vmd = parse_vmd(path)
            prof = profs.get(model) or {}
            bad = sorted(set(vmd["bones"]) & forbidden_names(prof))
            check(f"no_helper_tracks_{os.path.basename(rel)}", not bad, bad)


def blender_check():
    if not os.path.exists(BLENDER):
        check("blender_available", False, BLENDER)
        return
    cfg = json.load(open(os.path.join(ROOT, "configs", "models.json"),
                        encoding="utf-8"))["models"]["si"]
    with tempfile.TemporaryDirectory() as td:
        vmd = os.path.join(td, "knee60_l.vmd")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_pose_test_vmd.py"),
                        "--bone", "左ひざ", "--degrees", "60", "--out", vmd],
                       check=True, cwd=ROOT)
        report = os.path.join(td, "bake.json")
        cmd = [BLENDER, "--background", cfg["blend"],
               "--python", os.path.join(ROOT, "scripts", "bake_inherit_blender.py"),
               "--", "--model", "si", "--vmd", vmd, "--frames", "1:61",
               "--report", report]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=600)
        check("blender_bake_exit", proc.returncode == 0, proc.returncode)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-2000:])
            return
        rep = json.load(open(report, encoding="utf-8"))
        ver = rep.get("verify", {})
        check("blender_vs_pmx_position", ver.get("ok") is True, ver.get("max_head_error"))
        check("blender_vs_pmx_position_lt_1e-3",
              ver.get("max_head_error", 9) < 1e-3, ver.get("max_head_error"))
        check("blender_vs_pmx_rotation",
              ver.get("max_rot_error_deg", 9) < 1e-3, ver.get("max_rot_error_deg"))
        check("d_chain_equals_fk", rep.get("d_chain_vs_fk", {}).get("ok") is True,
              rep.get("d_chain_vs_fk", {}).get("max_pos_err"))

        before = rep.get("measure_before", [{}])[0]
        after = rep.get("measure_after", [{}])[0]
        check("d_straight_before_bake", before.get("l_d_flex", 99) < 5.0,
              before.get("l_d_flex"))
        check("d_bends_after_bake", 50.0 <= after.get("l_d_flex", -1) <= 65.0,
              after.get("l_d_flex"))
        check("d_follows_fk_after_bake",
              abs(after.get("l_d_flex", -99) - after.get("l_fk_flex", 0)) < 0.01,
              (after.get("l_d_flex"), after.get("l_fk_flex")))
        check("other_leg_untouched", after.get("r_d_flex", 99) < 5.0,
              after.get("r_d_flex"))
        check("ankle_rises", after.get("l_ankle_z", 0) > before.get("l_ankle_z", 9) + 0.1,
              (before.get("l_ankle_z"), after.get("l_ankle_z")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", action="store_true",
                    help="also run the headless Blender bake")
    args = ap.parse_args()
    pure_checks()
    if args.blender or os.environ.get("VTV_RUN_BLENDER_TESTS") == "1":
        blender_check()
    else:
        print("SKIP blender bake (pass --blender or set VTV_RUN_BLENDER_TESTS=1)")
    print()
    if FAILS:
        print("INHERIT TEST FAILED:", FAILS)
        return 1
    print("INHERIT TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
