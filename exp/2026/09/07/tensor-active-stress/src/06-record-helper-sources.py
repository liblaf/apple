# Copyright (c) 2026 liblaf
"""Archive the exact external helper sources used by post-processing."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pydantic_settings as ps
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
DATE_ROOT = EXPERIMENT.parent
COMPLETED = False
HELPERS = (
    "face-actuation-diagnosis/src/41-surface-roughness.py",
    "face-activation-materials/src/40-audit-face-results.py",
    "face-actuation-diagnosis/src/50-render-diagnosis.py",
    "face-actuation-diagnosis/src/60-publish-diagnosis.py",
    "face-actuation-diagnosis/src/61-validate-diagnosis-artifacts.py",
)
IMPORT_DISTRIBUTIONS = {
    "ipctk": "ipctk",
    "liblaf.cherries": "liblaf-cherries",
    "markdown_it": "markdown-it-py",
    "numpy": "numpy",
    "PIL": "Pillow",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "pyvista": "pyvista",
    "scipy": "scipy",
    "vtk": "vtk",
}
PACKAGE_MODULES = ("liblaf.apple", "liblaf.peach", "liblaf.cherries")


class Config(cherries.BaseConfig):
    """Cherries-managed immutable helper-source archive."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("06-helper-sources", mkdir=True)


def sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Return a path, byte count, and digest receipt."""
    receipt: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if relative_to is not None:
        receipt["relative_path"] = (
            path.resolve().relative_to(relative_to.resolve()).as_posix()
        )
    return receipt


def copy_with_receipt(
    source: Path,
    destination: Path,
    *,
    source_root: Path,
    output: Path,
) -> dict[str, dict[str, Any]]:
    """Copy one file and require its byte count and digest to stay exact."""
    source_record = record(source, relative_to=source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    archived_record = record(destination, relative_to=output)
    if (
        archived_record["bytes"],
        archived_record["sha256"],
    ) != (source_record["bytes"], source_record["sha256"]):
        message = f"archived source identity changed: {source}"
        raise OSError(message)
    return {"source": source_record, "archived": archived_record}


def main(cfg: Config) -> None:
    """Copy explicit helper sources and record their installed dependencies."""
    global COMPLETED  # noqa: PLW0603
    output = cfg.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        message = f"choose an empty output directory: {output}"
        raise FileExistsError(message)

    helpers: dict[str, dict[str, Any]] = {}
    for raw in HELPERS:
        relative = Path(raw)
        source = (DATE_ROOT / relative).resolve()
        source.relative_to(DATE_ROOT.resolve())
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / relative
        helpers[relative.as_posix()] = copy_with_receipt(
            source,
            destination,
            source_root=DATE_ROOT,
            output=output,
        )

    typed_package_sources: dict[str, dict[str, Any]] = {}
    for module_name in PACKAGE_MODULES:
        module = importlib.import_module(module_name)
        if module.__file__ is None:
            raise ImportError(module_name)
        origin = Path(module.__file__).resolve().parent
        candidates = sorted(
            {
                *origin.rglob("*.pyi"),
                *origin.rglob("py.typed"),
            }
        )
        if not candidates:
            message = f"typed package source inventory is empty: {module_name}"
            raise FileNotFoundError(message)
        module_archive = output / "package-sources" / Path(*module_name.split("."))
        typed_package_sources[module_name] = {
            "origin": str(origin),
            "files": {
                path.relative_to(origin).as_posix(): copy_with_receipt(
                    path,
                    module_archive / path.relative_to(origin),
                    source_root=origin,
                    output=output,
                )
                for path in candidates
            },
        }

    recorder_source = Path(__file__).resolve()
    recorder_copy = output / recorder_source.name
    shutil.copy2(recorder_source, recorder_copy)
    result = {
        "schema_version": 1,
        "status": "completed",
        "scope": (
            "exact source-only archive for external helpers loaded by tensor "
            "surface measurement, rendering, and publication; no helper imported "
            "and no solver or renderer executed"
        ),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "helpers": helpers,
        "typed_package_sources": typed_package_sources,
        "packages": {
            import_name: {
                "distribution": distribution,
                "version": importlib.metadata.version(distribution),
            }
            for import_name, distribution in IMPORT_DISTRIBUTIONS.items()
        },
        "recorder": {
            "source": record(recorder_source),
            "archived": record(recorder_copy, relative_to=output),
        },
        "relationship_to_runtime_record": (
            "data/05-runtime separately records core apple, peach, and cherries "
            "sources plus pyproject.toml and uv.lock"
        ),
    }
    summary = output / "summary.json"
    summary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for path in sorted(output.rglob("*")):
        if path.is_file():
            cherries.log_output(path)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
