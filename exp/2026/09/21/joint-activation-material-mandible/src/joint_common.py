"""Evidence and runtime helpers for the joint inverse experiment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from liblaf.cherries import core, plugins, profiles

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
HISTORICAL = ROOT / "exp/2026/09/07/tensor-active-stress/src"


class ProfileJoint(profiles.Profile):
    def init(self) -> core.Run:
        # Per-run source archives already capture the exact dirty implementation.
        # Avoid Comet scanning unrelated high-cardinality experiment outputs.
        os.environ.setdefault("COMET_AUTO_LOG_GIT_METADATA", "false")
        os.environ.setdefault("COMET_AUTO_LOG_GIT_PATCH", "false")
        run = core.run
        run.plugins.register(
            plugins.Comet(run=run, disabled=os.environ.get("DEBUG") == "1")
        )
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Logging(run=run))
        # Logging resets root handlers; attach Local's snapshot handler afterward.
        run.plugins.register(plugins.Local(run=run))
        return run


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_sources(output: Path) -> dict:
    target = output / "sources"
    assert not target.exists(), target
    for source, name in (
        (GROUP / "src", "experiment"),
        (ROOT / "src/liblaf/apple", "apple"),
        (HISTORICAL, "tensor-reference"),
    ):
        shutil.copytree(
            source, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    receipt = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "python": sys.version,
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "commit_enabled": False,
        "sources": {
            str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob("*.py"))
        },
    }
    write_json(output / "provenance.json", receipt)
    return receipt
