"""Generate profiles/<model>.json for every configured PMX.

Usage:
  python scripts/make_profiles.py [--config configs/models.json] [--out profiles]
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.mmd.profile import build_profile, save_profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "profiles"))
    ap.add_argument("--only", default=None, help="comma separated model keys")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    only = set(args.only.split(",")) if args.only else None
    for key, m in cfg["models"].items():
        if only and key not in only:
            continue
        pmx = m["pmx"]
        print(f"[profile] {key}: {pmx}")
        profile = build_profile(pmx, key)
        profile["display"] = m.get("display", key)
        out = os.path.join(args.out, f"{key}.json")
        save_profile(profile, out)
        print(f"  -> {out}  bones={profile['pmx']['bone_count']} "
              f"morphs={profile['pmx']['morph_count']} height={profile['height']} "
              f"missing={profile['standard_missing']}")


if __name__ == "__main__":
    main()
