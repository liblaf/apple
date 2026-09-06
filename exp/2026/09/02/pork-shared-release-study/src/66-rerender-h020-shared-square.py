"""DEBUG-only wrapper that refreshes exactly one static h=.20 shared square."""

from __future__ import annotations

import subprocess
from pathlib import Path

from liblaf import cherries


def main() -> None:
    group = Path.cwd()
    command = [
        "/usr/bin/pvpython",
        str(group / "src/63-render-focused-h010-materials.py"),
        "--h010-root",
        str(
            (
                group / "../../../08/31/unreachable-pork-factor-study/data/10-pork-2d"
            ).resolve()
        ),
        "--h020-canonical-root",
        str((group / "data/20-canonical-h020").resolve()),
        "--loss-root",
        str(
            (
                group / "../../../08/31/unreachable-pork-factor-study/data/10-pork-2d"
            ).resolve()
        ),
        "--output-root",
        str((group / "data/70-focused-h010-existing-results").resolve()),
        "--square-only",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    cherries.main(main)
