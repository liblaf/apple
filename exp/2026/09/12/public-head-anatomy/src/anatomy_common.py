# Copyright 2026 liblaf
"""Shared paths and evidence helpers for the public-anatomy comparison."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pyvista as pv
from liblaf.cherries import core, plugins, profiles

ROOT = Path(__file__).resolve().parents[6]
BASELINE = ROOT / "exp/2026/09/07/face-activation-materials/data/10-fixture"
BASELINE_SOURCE = ROOT / "exp/2026/09/07/face-activation-materials/src"


class ProfileCometNoCommit(profiles.Profile):
    """Record normal experiment evidence without committing unrelated work."""

    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def camera(plotter: pv.Plotter, center: Sequence[float], scale: float = 0.5) -> None:
    """The current fixture is X lateral, Y superior, Z anterior, in meters."""
    plotter.camera_position = [
        [center[0], center[1], center[2] + scale],
        list(center),
        [0, 1, 0],
    ]
    plotter.enable_parallel_projection()
