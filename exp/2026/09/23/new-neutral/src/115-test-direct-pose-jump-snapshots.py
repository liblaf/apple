"""Persist a bounded no-contact pose-jump relaxation snapshot series."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from liblaf import cherries

HERE = Path(__file__).resolve().parent
GROUP = HERE.parent
spec = importlib.util.spec_from_file_location(
    "pose_jump_runner", HERE / "112-test-pose-jump.py"
)
assert spec is not None
assert spec.loader is not None
runner: Any = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class Config(cherries.BaseConfig):
    source_run: Path = GROUP / "data/inverse-mouthopen-003"
    output_dir: Path = GROUP / "data/pose-jump-004"
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    forward_atol: float = 1e-8
    max_newton_steps: int = 3000
    off_wall_seconds: float = 120.0
    ipc_threads: int = 4


def main(cfg: Config) -> None:
    """Run the unchanged proposal initializer with accepted-state snapshots."""
    run_cfg = runner.Config.model_construct(
        **cfg.model_dump(), mode="proposal", entrypoint_source=Path(__file__)
    )
    runner.main(run_cfg)


if __name__ == "__main__":
    cherries.main(main, profile=runner.ProfileJoint)
