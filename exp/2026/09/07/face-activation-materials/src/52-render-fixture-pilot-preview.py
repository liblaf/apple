"""Render fixture/pilot saved geometry through Cherries without running physics."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from liblaf import cherries

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    manifest: Path = ROOT / "docs" / "52-fixture-pilot-preview.json"
    output: Path = ROOT / "data" / "62-fixture-pilot-preview-lazy-caption"


def main(cfg: Config) -> None:
    command = [
        "python",
        "src/50-render-face-comparisons.py",
        str(cfg.manifest),
        "--output",
        str(cfg.output),
    ]
    logger.info("Rendering fixture and saved pilot endpoint with %s", command)
    subprocess.run(command, check=True)
    cherries.log_output(cfg.output)
    cherries.log_metrics({"render/cases": 1, "render/solver_runs": 0})


if __name__ == "__main__":
    cherries.main(main)
