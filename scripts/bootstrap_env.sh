#!/usr/bin/env bash
# Day 0-1: create E: venv, install torch 2.3.0+cu121 and GVHMR deps.
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMP="$PWD/.tmp"
export TEMP="$PWD/.tmp"
export HF_HOME="$PWD/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$TMP" "$HF_HOME"

if [ ! -x .venv/Scripts/python.exe ]; then
  python -m venv .venv
fi
PY=.venv/Scripts/python.exe
"$PY" -m pip install -U pip wheel setuptools

# torch cu121 for py310 (do not use system torch 2.7.1)
"$PY" -m pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# GVHMR requirements: drop the Linux pytorch3d wheel, pin pycolmap, and drop
# packages that do not build on Windows (Wis3D/cython_bbox are not on the
# inference import path).
grep -v -E '^(pytorch3d|cython_bbox|wis3d)([[:space:]]|$)' \
  third_party/GVHMR/requirements.txt > scripts/requirements-gvhmr-win.txt
echo 'pycolmap==0.6.1' > scripts/constraints-gvhmr.txt

# chumpy's setup.py imports pip; it only builds without build isolation.
"$PY" -m pip install chumpy --no-build-isolation
"$PY" -m pip install -r scripts/requirements-gvhmr-win.txt \
  -c scripts/constraints-gvhmr.txt --no-build-isolation

# lightning 2.3 imports pkg_resources (removed in setuptools>=81); HMR2 needs yacs.
"$PY" -m pip install "setuptools<81" yacs
"$PY" -m pip install -e third_party/GVHMR --no-deps
"$PY" - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "py", sys.version)
import numpy
print("numpy", numpy.__version__)
PY
echo "BOOTSTRAP OK"
