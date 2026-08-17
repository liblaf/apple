from __future__ import annotations

import sys
from pathlib import Path

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
REFERENCE_GROUP = REPO_ROOT / "exp/2026/06/17/human-face-smile-prestrain-v2"
REFERENCE_SRC = REFERENCE_GROUP / "src"
PREPARED_MESH = REFERENCE_GROUP / "data/10-human-face-prepared.vtu"


def enable_reference_modules() -> None:
    if not REFERENCE_SRC.is_dir():
        msg = f"missing reference experiment source directory: {REFERENCE_SRC}"
        raise FileNotFoundError(msg)
    reference = str(REFERENCE_SRC)
    if reference not in sys.path:
        # Keep this experiment's compatibility shim ahead of the reference modules.
        sys.path.append(reference)
