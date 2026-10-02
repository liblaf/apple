"""Record the exact runtime and compact Python source provenance."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = EXPERIMENT_ROOT / "data"
ARCHIVE_ROOT = DATA_ROOT / "runtime-sources"
ENVIRONMENT_PATH = DATA_ROOT / "environment.json"
PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "torch",
    "warp-lang",
    "pyvista",
    "vtk",
    "ipctk",
    "liblaf-apple",
    "liblaf-peach",
    "liblaf-cherries",
)
SUPPORT_SOURCE_NAMES = (
    "40-audit-face-results.py",
    "face_physics.py",
    "activation_models.py",
)


class ProvenanceError(RuntimeError):
    """Raised when an exact provenance contract cannot be established."""


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path) -> str:
    """Run a read-only command and return stripped standard output."""
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository_record(repository_root: Path) -> dict[str, Any]:
    """Record the repository identity without retaining the status paths."""
    status = run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=repository_root,
    )
    return {
        "root": str(repository_root),
        "head": run(["git", "rev-parse", "HEAD"], cwd=repository_root),
        "dirty": bool(status),
    }


def python_sources(root: Path) -> list[Path]:
    """List regular Python sources below a package root."""
    sources = [
        path
        for path in root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    if not sources:
        message = f"no Python sources found below {root}"
        raise ProvenanceError(message)
    return sorted(sources)


def copy_verified(source: Path, destination: Path) -> str:
    """Copy one file and prove that its content did not change in transit."""
    before = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    after = sha256(source)
    copied = sha256(destination)
    if not before == after == copied:
        message = f"source changed or copy verification failed: {source}"
        raise ProvenanceError(message)
    return before


def copy_tree_sources(
    source_root: Path,
    destination_root: Path,
) -> dict[str, str]:
    """Copy and hash every Python source below one root."""
    hashes: dict[str, str] = {}
    for source in python_sources(source_root):
        relative = source.relative_to(source_root)
        hashes[relative.as_posix()] = copy_verified(
            source,
            destination_root / relative,
        )
    return hashes


def module_root(module_name: str) -> Path:
    """Resolve the loaded source root for a regular Python package."""
    module = importlib.import_module(module_name)
    if module.__file__ is None:
        message = f"{module_name} has no source file"
        raise ProvenanceError(message)
    root = Path(module.__file__).resolve().parent
    if not root.is_dir():
        message = f"{module_name} source root is not a directory: {root}"
        raise ProvenanceError(message)
    return root


def module_record(
    *,
    name: str,
    module_name: str,
    stage_root: Path,
    repository_root: Path,
    repository_head: str,
) -> dict[str, Any]:
    """Snapshot one imported package and record its exact source identity."""
    source_root = module_root(module_name)
    tracked_here = False
    if source_root.is_relative_to(repository_root):
        repository_relative = source_root.relative_to(repository_root)
        tracked_here = bool(
            run(
                ["git", "ls-files", "--", repository_relative.as_posix()],
                cwd=repository_root,
            )
        )
    hashes = copy_tree_sources(source_root, stage_root / name)
    return {
        "module_path": str(source_root),
        "snapshot_path": f"runtime-sources/{name}",
        "tracked_in_working_repository": tracked_here,
        "repository_commit": repository_head if tracked_here else None,
        "source_file_count": len(hashes),
        "source_sha256": hashes,
    }


def reference_record(
    *,
    source_root: Path,
    sources: list[Path],
    archive_name: str,
    stage_root: Path,
) -> dict[str, Any]:
    """Snapshot explicitly selected experiment sources with source paths."""
    if not sources:
        message = f"no reference sources selected below {source_root}"
        raise ProvenanceError(message)
    files: dict[str, dict[str, str]] = {}
    for source in sorted(sources):
        relative = source.relative_to(source_root)
        archive_relative = Path(archive_name) / relative
        digest = copy_verified(source, stage_root / archive_relative)
        files[relative.as_posix()] = {
            "source_path": str(source),
            "snapshot_path": f"runtime-sources/{archive_relative.as_posix()}",
            "sha256": digest,
        }
    return {
        "source_root": str(source_root),
        "snapshot_path": f"runtime-sources/{archive_name}",
        "source_file_count": len(files),
        "files": files,
    }


def nvidia_record(repository_root: Path) -> dict[str, Any]:
    """Query NVIDIA device metadata without starting a GPU workload."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"available": False, "query": command, "reason": "nvidia-smi not found"}

    if completed.returncode != 0:
        return {
            "available": False,
            "query": command,
            "returncode": completed.returncode,
            "reason": "nvidia-smi query failed",
        }

    devices = []
    for line in completed.stdout.splitlines():
        index, name, driver, memory_mib = (field.strip() for field in line.split(","))
        devices.append(
            {
                "index": int(index),
                "name": name,
                "driver_version": driver,
                "memory_mib": int(memory_mib),
            }
        )
    return {"available": True, "query": command, "devices": devices}


def verify_archive(root: Path, expected: dict[str, str]) -> None:
    """Verify the complete archive file set and every recorded digest."""
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_paths != expected.keys():
        missing = sorted(expected.keys() - actual_paths)
        unexpected = sorted(actual_paths - expected.keys())
        message = (
            f"archive file set mismatch; missing={missing}, unexpected={unexpected}"
        )
        raise ProvenanceError(message)
    for relative, digest in expected.items():
        if sha256(root / relative) != digest:
            message = f"archive digest mismatch: {relative}"
            raise ProvenanceError(message)


def main() -> None:
    """Write a fresh environment receipt and verified source archive."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    repository_root = Path(
        run(["git", "rev-parse", "--show-toplevel"], cwd=EXPERIMENT_ROOT)
    ).resolve()
    repository = repository_record(repository_root)

    support_root = (
        repository_root / "exp/2026/09/07/face-activation-materials/src"
    ).resolve()
    support_sources = [support_root / name for name in SUPPORT_SOURCE_NAMES]
    for source in support_sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    june_root = (
        repository_root / "exp/2026/06/17/human-face-smile-prestrain-v2/src"
    ).resolve()
    june_sources = sorted(june_root.glob("*.py"))

    with tempfile.TemporaryDirectory(
        prefix=".runtime-sources-",
        dir=DATA_ROOT,
    ) as temporary:
        stage_root = Path(temporary) / "archive"
        stage_root.mkdir()

        modules = {
            name: module_record(
                name=name,
                module_name=f"liblaf.{name}",
                stage_root=stage_root,
                repository_root=repository_root,
                repository_head=repository["head"],
            )
            for name in ("apple", "peach")
        }
        references = {
            "face_activation_materials_support": reference_record(
                source_root=support_root,
                sources=support_sources,
                archive_name="reference-face-activation-materials",
                stage_root=stage_root,
            ),
            "june_human_face_smile_prestrain_v2": reference_record(
                source_root=june_root,
                sources=june_sources,
                archive_name="reference-june-human-face-smile-prestrain-v2",
                stage_root=stage_root,
            ),
        }

        expected_hashes: dict[str, str] = {}
        for name, record in modules.items():
            expected_hashes.update(
                {
                    f"{name}/{relative}": digest
                    for relative, digest in record["source_sha256"].items()
                }
            )
        for record in references.values():
            expected_hashes.update(
                {
                    str(
                        Path(record["snapshot_path"]).relative_to("runtime-sources")
                        / relative
                    ): metadata["sha256"]
                    for relative, metadata in record["files"].items()
                }
            )
        verify_archive(stage_root, expected_hashes)

        if ARCHIVE_ROOT.exists():
            shutil.rmtree(ARCHIVE_ROOT)
        stage_root.rename(ARCHIVE_ROOT)

    verify_archive(ARCHIVE_ROOT, expected_hashes)

    import torch

    receipt = {
        "recorded_unix_time": time.time(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGE_NAMES},
        "repository": repository,
        "cuda": {
            "torch_build": torch.version.cuda,
            "torch_available": torch.cuda.is_available(),
            "nvidia_smi": nvidia_record(repository_root),
        },
        "module_identity": modules,
        "reference_sources": references,
        "archive_verification": {
            "verified": True,
            "file_count": len(expected_hashes),
            "sha256_algorithm": "SHA-256",
        },
        "scope": (
            "Read-only runtime identity and verified Python source archive; "
            "installed Peach is not attributed to the enclosing Apple Git commit. "
            "Environment variables and secrets are excluded."
        ),
    }
    temporary_environment = ENVIRONMENT_PATH.with_suffix(".json.tmp")
    temporary_environment.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary_environment.replace(ENVIRONMENT_PATH)
    print(ENVIRONMENT_PATH)
    print(ARCHIVE_ROOT)
    print(f"verified {len(expected_hashes)} archived Python source files")


if __name__ == "__main__":
    main()
