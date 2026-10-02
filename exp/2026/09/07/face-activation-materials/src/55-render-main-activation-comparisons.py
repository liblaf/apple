"""Render exact saved activation comparison states without executing solvers."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from liblaf import cherries

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    manifest: Path = ROOT / "docs" / "55-main-activation-comparisons.json"
    output: Path = ROOT / "data" / "75-final-face-comparisons-caption-legend"


def main(cfg: Config) -> None:
    command = [
        "python",
        "src/50-render-face-comparisons.py",
        str(cfg.manifest),
        "--output",
        str(cfg.output),
    ]
    logger.info("Rendering exact saved states only with %s", command)
    subprocess.run(command, check=True)
    cherries.log_output(cfg.output)
    cherries.log_metrics({"render/cases": 10, "render/solver_runs": 0})


if __name__ == "__main__":
    cherries.main(main)
