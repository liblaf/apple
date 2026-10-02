"""Continue inverse Adam, resetting forward displacement after each failed solve."""

from __future__ import annotations

import importlib
from pathlib import Path

from liblaf import cherries

continued = importlib.import_module("120-continue-inexact")


class Config(continued.Config):
    output: Path = Path("140-reset-continuation")
    reset_forward_on_failure: bool = True


def main(cfg: Config) -> None:
    continued.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=continued.runner.ProfileActivationStudy)
