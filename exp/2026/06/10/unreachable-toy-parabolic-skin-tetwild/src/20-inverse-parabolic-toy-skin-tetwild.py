from __future__ import annotations

import sys
from pathlib import Path

from liblaf import cherries

HELPER_DIR = (
    Path(__file__).resolve().parents[2] / "unreachable-toy-skin-tetwild" / "src"
)
sys.path.insert(0, str(HELPER_DIR))

from _toy_skin_tetwild import run_inverse  # noqa: E402

if __name__ == "__main__":
    cherries.main(run_inverse)
