"""Separate Cherries entrypoint for an independently pinned contact benchmark."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("10-benchmark.py")), run_name="__main__"
    )
