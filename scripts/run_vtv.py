"""Thin shim for the VTV CLI (see ``src/vtv/cli.py``).

Kept so every existing command line keeps working unchanged:

    python scripts/run_vtv.py --video ... --out ... --models si,luoxi,tiflos
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
