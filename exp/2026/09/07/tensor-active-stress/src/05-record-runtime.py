"""Record the runtime and Python dependencies used during the face batch."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pydantic_settings as ps
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[4]
COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("05-runtime", mkdir=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


def main(cfg: Config) -> None:
    global COMPLETED  # noqa: PLW0603
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir())
    versions = {
        package: importlib.metadata.version(package)
        for package in (
            "numpy",
            "scipy",
            "torch",
            "warp-lang",
            "pyvista",
            "vtk",
            "liblaf-apple",
            "liblaf-peach",
            "liblaf-cherries",
            "cupy-cuda13x",
        )
    }
    sources = {}
    for module_name in ("liblaf.apple", "liblaf.peach", "liblaf.cherries"):
        module = importlib.import_module(module_name)
        assert module.__file__ is not None
        origin = Path(module.__file__).resolve().parent
        target = out / "sources" / module_name.replace(".", "/")
        hashes = {}
        for path in sorted(origin.rglob("*.py")):
            relative = path.relative_to(origin)
            before = sha256(path)
            copy = target / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
            assert sha256(copy) == sha256(path) == before
            hashes[str(relative)] = before
        assert hashes
        sources[module_name] = {"origin": str(origin), "source_sha256": hashes}
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(REPO / name, out / name)
    result = {
        "status": "recorded_during_face_batch",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "gpu": command(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        "git_head": command("git", "rev-parse", "HEAD"),
        "tracked_worktree_diff": command("git", "diff", "--stat"),
        "tracked_index_diff": command("git", "diff", "--cached", "--stat"),
        "sources": sources,
        "dependency_files": {
            name: sha256(out / name) for name in ("pyproject.toml", "uv.lock")
        },
        "shared_hardware": "The face runs share one GPU. Wall times are not a single-process benchmark.",
        "source_note": "Each numerical run also archives its own experiment and apple source files before solving.",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    cherries.log_output(out / "summary.json")
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
