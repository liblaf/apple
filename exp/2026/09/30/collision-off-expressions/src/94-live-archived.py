"""Mark the collision-off runtime preview as archived without altering its evidence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GROUP = Path(__file__).resolve().parent.parent
BASE = GROUP / "src/92-live-regularized-deadline.py"
OUTPUT = GROUP / "data/preview/index.html"
MARKER = '<section id="collision-off-archived"'


def main() -> None:
    subprocess.run([sys.executable, str(BASE)], check=True)
    page = OUTPUT.read_text()
    assert MARKER not in page
    assert "<h1>Collision-off corrected-neutral expressions</h1>" in page
    banner = (
        '<section id="collision-off-archived" style="border-color:#8c6d1f;background:#fff7d6">'
        "<h2>Archived and superseded</h2>"
        "<p>Collision-off computation is stopped. This page preserves historical receipts only. "
        "The active work is the collision-on MouthOpen and Smile pair; no collision-off card, "
        "reservation, or historic RUNNING label indicates a live numerical process.</p>"
        "</section>"
    )
    anchor = "<h1>Collision-off corrected-neutral expressions</h1>"
    OUTPUT.write_text(page.replace(anchor, anchor + banner, 1))
    rendered = OUTPUT.read_text()
    assert rendered.count(MARKER) == 1
    assert "Collision-off computation is stopped." in rendered


if __name__ == "__main__":
    main()
