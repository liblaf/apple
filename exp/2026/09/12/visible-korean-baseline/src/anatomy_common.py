# Copyright 2026 liblaf
"""Run configuration and evidence helpers for the Visible Korean baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from liblaf.cherries import core, plugins, profiles


class ProfileCometNoCommit(profiles.Profile):
    """Keep normal Cherries and Comet evidence while preserving the worktree."""

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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
