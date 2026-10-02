"""Render saved FiberRegion and successful fixed-control replay states only."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from liblaf import cherries

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    manifest: Path = ROOT / "docs" / "54-fixed-control-replays-preview.json"
    output: Path = ROOT / "data" / "72-fixed-control-replays-smooth"


def main(cfg: Config) -> None:
    command = [
        "python",
        "src/50-render-face-comparisons.py",
        str(cfg.manifest),
        "--output",
        str(cfg.output),
    ]
    logger.info("Rendering saved equilibria only with %s", command)
    subprocess.run(command, check=True)
    cherries.log_output(cfg.output)
    cherries.log_metrics(
        {"render/cases": 5, "render/saved_states": 11, "render/solver_runs": 0}
    )


if __name__ == "__main__":
    cherries.main(main)
