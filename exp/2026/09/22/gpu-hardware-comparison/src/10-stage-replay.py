# Copyright (c) 2026 liblaf
"""Stage the archived RTX 4090 Hessian benchmark for a CUDA 12 V100 replay.

The staged tree is deliberately self-contained and preserves the archived
physical/model sources byte-for-byte. A small compatibility package maps the
retired Peach import paths to Apple's extracted solver implementation without
changing any hash-bound historical source.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "exp/2026/09/22/solver-performance"
ARCHIVE = SOLVER / "data/hessian-representations-001/sources"
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
TENSOR = ROOT / "exp/2026/09/07/tensor-active-stress"
DEFAULT_TARGET = GROUP / "tmp/replay/apple"
PRETEND_VERSION = "27.dev197+gd56fa1b55.d20260922"

LEGACY_SOURCE_ROOT = ROOT
LEGACY_EXTERNAL_TARGETS = {
    Path(os.environ["APPLE_MELON_HEAD"]) / "11-mandible.landmarks.json": Path(
        "remote-external/melon-11-mandible.landmarks.json"
    ),
    Path(os.environ["APPLE_MELON_HEAD"]) / "13-cranium.ply": Path(
        "remote-external/melon-13-cranium.ply"
    ),
    Path(os.environ["APPLE_MELON_HEAD"]) / "13-mandible.ply": Path(
        "remote-external/melon-13-mandible.ply"
    ),
    Path(os.environ["APPLE_MELON_HEAD"]) / "20-eye.ply": Path(
        "remote-external/melon-20-eye.ply"
    ),
    Path(os.environ["APPLE_MELON_HEAD"]) / "22-skin.landmarks.json": Path(
        "remote-external/melon-22-skin.landmarks.json"
    ),
    Path(os.environ["APPLE_MELON_HEAD"]) / "22-skin.vtp": Path(
        "remote-external/melon-22-skin.vtp"
    ),
    Path(os.environ["APPLE_MELON_HEAD"]) / "62-tetmesh-3191k.vtu": Path(
        "remote-external/melon-62-tetmesh-3191k.vtu"
    ),
}
HISTORICAL_UV_LOCK = (
    ROOT / "exp/2026/09/07/tensor-active-stress/data/05-runtime/uv.lock"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, *, base: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(base)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def copy_file(source: Path, destination: Path) -> None:
    assert source.is_file(), source
    assert not destination.exists(), destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    assert sha256(source) == sha256(destination)


def copy_tree(source: Path, destination: Path) -> None:
    assert source.is_dir(), source
    assert not destination.exists(), destination
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def absolute_json_references(value: object, *, key: str = "") -> list[Path]:
    references: list[Path] = []
    if isinstance(value, dict):
        for child_key, item in value.items():
            references.extend(absolute_json_references(item, key=child_key))
    elif isinstance(value, list):
        for item in value:
            references.extend(absolute_json_references(item, key=key))
    elif (
        isinstance(value, str)
        and Path(value).is_absolute()
        and (
            key in {"path", "directory"}
            or key.endswith(("_path", "_npz", "_manifest", "_directory"))
        )
    ):
        references.append(Path(value))
    return references


def staging_source(path: Path) -> Path:
    if path in LEGACY_EXTERNAL_TARGETS:
        return path
    if path.is_absolute():
        try:
            return ROOT / path.relative_to(LEGACY_SOURCE_ROOT)
        except ValueError:
            path.relative_to(ROOT)
    return path


def staging_target(path: Path) -> Path:
    legacy = path
    if path in LEGACY_EXTERNAL_TARGETS:
        return LEGACY_EXTERNAL_TARGETS[path]
    if path.is_absolute():
        with contextlib.suppress(ValueError):
            legacy = LEGACY_SOURCE_ROOT / path.relative_to(ROOT)
    return legacy.relative_to(LEGACY_SOURCE_ROOT)


def collect_runtime_closure(roots: list[Path]) -> list[tuple[Path, Path]]:
    pending = [staging_source(path) for path in roots]
    files: dict[Path, Path] = {}
    while pending:
        source = pending.pop().resolve()
        if source in files:
            continue
        if source.is_dir():
            manifest = source / "manifest.json"
            assert manifest.is_file(), f"staging directory has no manifest: {source}"
            pending.append(manifest)
            continue
        assert source.is_file(), f"missing staging dependency: {source}"
        external = next(
            (path for path in LEGACY_EXTERNAL_TARGETS if path.resolve() == source),
            None,
        )
        legacy = (
            external
            if external is not None
            else LEGACY_SOURCE_ROOT / source.relative_to(ROOT)
        )
        files[source] = staging_target(legacy)
        if source.suffix == ".json":
            pending.extend(
                staging_source(reference)
                for reference in absolute_json_references(
                    json.loads(source.read_text())
                )
            )
    return sorted(files.items(), key=lambda item: str(item[1]))


def install_peach_compatibility(target: Path) -> dict[str, Any]:
    path = target / "src/liblaf/peach/__init__.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        '''"""Compatibility aliases for hash-bound experiments recorded with Peach."""

from __future__ import annotations

import importlib
import sys

linalg = importlib.import_module("liblaf.apple.solvers.linalg")
optim = importlib.import_module("liblaf.apple.solvers.optim")

for alias, module in {
    "liblaf.peach.linalg": "liblaf.apple.solvers.linalg",
    "liblaf.peach.linalg.base": "liblaf.apple.solvers.linalg.base",
    "liblaf.peach.linalg.cupy": "liblaf.apple.solvers.linalg.cupy",
    "liblaf.peach.optim": "liblaf.apple.solvers.optim",
    "liblaf.peach.optim.base": "liblaf.apple.solvers.optim.base",
    "liblaf.peach.optim.pncg": "liblaf.apple.solvers.optim.pncg",
    "liblaf.peach.optim.pncg._direction": "liblaf.apple.solvers.optim.pncg._direction",
}.items():
    sys.modules[alias] = importlib.import_module(module)

__all__ = ["linalg", "optim"]
'''
    )
    return {
        "scope": "Import-only compatibility aliases; archived sources remain byte-exact",
        "file": record(path, base=target),
    }


def patch_benchmark(path: Path, *, base: Path) -> dict[str, Any]:
    source = path.read_text()
    before_sha256 = sha256(path)

    old_config = """    output_dir: Path = cold.EXPERIMENT / "data/hessian-representations-001"
    baseline: Path = cold.EXPERIMENT / "data/cold-forward-comparison-001/summary.json"
    hvp_repeats: int = 30
"""
    new_config = """    output_dir: Path = cold.EXPERIMENT / "data/hessian-representations-comp03-v100-001"
    baseline: Path = cold.EXPERIMENT / "data/cold-forward-comparison-001/summary.json"
    reference_summary: Path = cold.EXPERIMENT / "data/reference-hessian-representations-4090/summary.json"
    replay_manifest: Path = cold.EXPERIMENT.parents[4] / "replay-manifest.json"
    hvp_repeats: int = 30
"""
    assert source.count(old_config) == 1
    source = source.replace(old_config, new_config, 1)

    old_setup = """    baseline = json.loads(cfg.baseline.read_text())
    assert benchmark.sha256(cfg.checkpoint) == baseline["protocol"]["checkpoint"]["sha256"]
    assert benchmark.sha256(cfg.neutral_checkpoint) == baseline["protocol"]["neutral_checkpoint"]["sha256"]
    for name, digest in baseline["protocol"]["inputs"].items():
        assert benchmark.sha256(cfg.inputs_dir / name) == digest
    cfg.output_dir.mkdir(parents=True)
    provenance = benchmark.archive_benchmark_sources(cfg)
    old_sources = json.loads((cfg.baseline.parent / "provenance.json").read_text())["sources"]
    changes = [name for name, digest in old_sources.items() if name.startswith(("apple/", "joint-experiment/", "tensor-reference/")) and provenance["sources"][name] != digest]
    assert not changes, changes
"""
    new_setup = """    baseline = json.loads(cfg.baseline.read_text())
    reference_summary = json.loads(cfg.reference_summary.read_text())
    reference_by_state = {row["state"]: row for row in reference_summary["results"]}
    assert set(reference_by_state) == {"cold_loaded_neutral", "saved_smile"}
    fixed_reference_shifts = {
        name: float(row["shift"]) for name, row in reference_by_state.items()
    }
    assert benchmark.sha256(cfg.checkpoint) == baseline["protocol"]["checkpoint"]["sha256"]
    assert benchmark.sha256(cfg.neutral_checkpoint) == baseline["protocol"]["neutral_checkpoint"]["sha256"]
    for name, digest in baseline["protocol"]["inputs"].items():
        assert benchmark.sha256(cfg.inputs_dir / name) == digest
    replay_manifest = json.loads(cfg.replay_manifest.read_text())
    assert replay_manifest["physical_sources"]["normalized_equivalent"] is True
    cfg.output_dir.mkdir(parents=True)
    provenance = benchmark.archive_benchmark_sources(cfg)
    changes = replay_manifest["physical_sources"]["adapted_files"]
"""
    assert source.count(old_setup) == 1
    source = source.replace(old_setup, new_setup, 1)

    old_protocol = """        "physical_source_changes": changes,
        "checkpoint_sha256": benchmark.sha256(cfg.checkpoint),
        "neutral_checkpoint_sha256": benchmark.sha256(cfg.neutral_checkpoint),
        "seed": 20260922,
"""
    new_protocol = """        "physical_source_changes": changes,
        "physical_source_normalized_equivalence": True,
        "replay_manifest": {"path": str(cfg.replay_manifest), "sha256": benchmark.sha256(cfg.replay_manifest)},
        "checkpoint_sha256": benchmark.sha256(cfg.checkpoint),
        "neutral_checkpoint_sha256": benchmark.sha256(cfg.neutral_checkpoint),
        "seed": 20260922,
        "fixed_reference_shifts": fixed_reference_shifts,
        "shift_policy": "Use the exact RTX 4090 reference-selected shift for each state; fail visibly if that fixed linear system is rejected on this GPU.",
"""
    assert source.count(old_protocol) == 1
    source = source.replace(old_protocol, new_protocol, 1)

    old_shift = """                shift = 0.0
                calibration = []
                for _ in range(10):
                    started = time.perf_counter()
                    try:
                        reference_solution, cg = pcg(lambda p: control(p) + shift * p, lambda r: r / (diagonal + shift), rhs, rtol=cfg.linear_rtol)
                        torch.cuda.synchronize()
                        calibration.append({"shift": shift, "seconds": time.perf_counter() - started, "success": True, **cg})
                        break
                    except LinearRejection as error:
                        torch.cuda.synchronize()
                        calibration.append({"shift": shift, "seconds": time.perf_counter() - started, "success": False, "error": str(error)})
                        shift = float(diagonal.mean()) * 1e-6 if shift == 0 else 10 * shift
                else:
                    raise RuntimeError(f"reference shift selection failed: {calibration}")
                row["shift"] = shift
                row["shift_calibration"] = calibration
"""
    new_shift = """                shift = fixed_reference_shifts[state_name]
                started = time.perf_counter()
                reference_solution, cg = pcg(
                    lambda p: control(p) + shift * p,
                    lambda r: r / (diagonal + shift),
                    rhs,
                    rtol=cfg.linear_rtol,
                )
                torch.cuda.synchronize()
                row["shift"] = shift
                row["shift_calibration"] = [{
                    "shift": shift,
                    "seconds": time.perf_counter() - started,
                    "success": True,
                    "source": "frozen RTX 4090 reference summary",
                    **cg,
                }]
"""
    assert source.count(old_shift) == 1
    source = source.replace(old_shift, new_shift, 1)

    path.write_text(source)
    return {
        "path": str(path.relative_to(base)),
        "archived_namespace_migrated_sha256": before_sha256,
        "staged_sha256": sha256(path),
        "changes": [
            "output defaults point to the COMP03 V100 result directory",
            "physical-source guard reads the explicit normalized-equivalence replay manifest",
            "PCG shifts are frozen to the RTX 4090 reference-selected values",
        ],
    }


def stage(target: Path, *, replace: bool) -> dict[str, Any]:  # noqa: C901
    target = target.resolve()
    assert target.name == "apple", target
    if target.exists():
        assert replace, f"target exists; pass --replace: {target}"
        shutil.rmtree(target)
    target.mkdir(parents=True)

    source_map = {
        ARCHIVE / "apple": target / "src/liblaf/apple",
        ARCHIVE / "solver-performance": target
        / "exp/2026/09/22/solver-performance/src",
        ARCHIVE / "joint-experiment": target
        / "exp/2026/09/21/joint-activation-material-mandible/src",
        ARCHIVE / "tensor-reference": target
        / "exp/2026/09/07/tensor-active-stress/src",
    }
    for source, destination in source_map.items():
        copy_tree(source, destination)

    archived_files = {
        str(path.relative_to(ARCHIVE)): path for path in sorted(ARCHIVE.rglob("*.py"))
    }
    for relative, archived in archived_files.items():
        staged = {
            "apple": target / "src/liblaf/apple",
            "solver-performance": target / "exp/2026/09/22/solver-performance/src",
            "joint-experiment": target
            / "exp/2026/09/21/joint-activation-material-mandible/src",
            "tensor-reference": target / "exp/2026/09/07/tensor-active-stress/src",
        }[relative.split("/", maxsplit=1)[0]] / relative.split("/", maxsplit=1)[1]
        assert staged.read_bytes() == archived.read_bytes(), staged

    solvers_target = target / "src/liblaf/apple/solvers"
    copy_tree(ROOT / "src/liblaf/apple/solvers", solvers_target)
    peach_compatibility = install_peach_compatibility(target)

    runtime_closure = collect_runtime_closure([JOINT / "data/expression-inputs-002"])
    closure_records = []
    for source, relative in runtime_closure:
        destination = target / relative
        if destination.exists():
            # The archived Python snapshot is authoritative over today's
            # working-tree sources. Hash-bound runtime checks validate it.
            closure_records.append(
                {
                    "path": str(relative),
                    "source": str(source),
                    "staged_sha256": sha256(destination),
                    "already_staged": True,
                }
            )
            continue
        actual_source = HISTORICAL_UV_LOCK if relative == Path("uv.lock") else source
        copy_file(actual_source, destination)
        closure_records.append(
            {
                "path": str(relative),
                "source": str(actual_source),
                "staged_sha256": sha256(destination),
                "already_staged": False,
            }
        )

    inputs_target = (
        target
        / "exp/2026/09/21/joint-activation-material-mandible/data/expression-inputs-002"
    )
    copy_file(
        JOINT / "data/expression-inputs-002/provenance.json",
        inputs_target / "provenance.json",
    )

    copied_inputs = [
        (
            SOLVER
            / "data/inverse-duration-hard-001/zero_smoothing/historical/expressions/Smile/latest.pt",
            target
            / "exp/2026/09/22/solver-performance/data/inverse-duration-hard-001/zero_smoothing/historical/expressions/Smile/latest.pt",
        ),
        (
            SOLVER
            / "data/smile-fit-adam03-no-smoothness-005/arms/hybrid_diag/expressions/Smile/initial.pt",
            target
            / "exp/2026/09/22/solver-performance/data/smile-fit-adam03-no-smoothness-005/arms/hybrid_diag/expressions/Smile/initial.pt",
        ),
        (
            SOLVER / "data/cold-forward-comparison-001/summary.json",
            target
            / "exp/2026/09/22/solver-performance/data/cold-forward-comparison-001/summary.json",
        ),
        (
            SOLVER / "data/cold-forward-comparison-001/provenance.json",
            target
            / "exp/2026/09/22/solver-performance/data/cold-forward-comparison-001/provenance.json",
        ),
        (
            SOLVER / "data/hessian-representations-001/summary.json",
            target
            / "exp/2026/09/22/solver-performance/data/reference-hessian-representations-4090/summary.json",
        ),
        (
            SOLVER / "data/hessian-representations-001/protocol.json",
            target
            / "exp/2026/09/22/solver-performance/data/reference-hessian-representations-4090/protocol.json",
        ),
        (
            SOLVER / "data/hessian-representations-001/provenance.json",
            target
            / "exp/2026/09/22/solver-performance/data/reference-hessian-representations-4090/provenance.json",
        ),
    ]
    for source, destination in copied_inputs:
        copy_file(source, destination)

    for name in (
        "README.md",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "verify.py",
    ):
        copy_file(
            ROOT / "environments/cuda12" / name,
            target / "environments/cuda12" / name,
        )
    for name in ("pyproject.toml", "README.md"):
        copy_file(ROOT / name, target / name)

    benchmark_path = (
        target
        / "exp/2026/09/22/solver-performance/src/49-benchmark-hessian-representations.py"
    )
    benchmark_patch = patch_benchmark(benchmark_path, base=target)

    exact_source_files = []
    for relative, archived in archived_files.items():
        staged = {
            "apple": target / "src/liblaf/apple",
            "solver-performance": target / "exp/2026/09/22/solver-performance/src",
            "joint-experiment": target
            / "exp/2026/09/21/joint-activation-material-mandible/src",
            "tensor-reference": target / "exp/2026/09/07/tensor-active-stress/src",
        }[relative.split("/", maxsplit=1)[0]] / relative.split("/", maxsplit=1)[1]
        if staged == benchmark_path:
            continue
        assert staged.read_bytes() == archived.read_bytes(), staged
        exact_source_files.append(record(staged, base=target))
    manifest = {
        "schema": "gpu-hardware-comparison-replay-v1",
        "source": {
            "reference_run": "hessian-representations-001",
            "reference_gpu": "NVIDIA GeForce RTX 4090",
            "reference_summary": "exp/2026/09/22/solver-performance/data/reference-hessian-representations-4090/summary.json",
        },
        "physical_sources": {
            "normalized_equivalent": True,
            "equivalence": "byte-exact archived physical and experiment Python sources; benchmark driver patch recorded separately",
            "adapted_files": [peach_compatibility],
            "archived_python_file_count": len(archived_files),
            "byte_exact_files": exact_source_files,
        },
        "solver_namespace": {
            "source": "src/liblaf/apple/solvers",
            "notice": record(solvers_target / "NOTICE.md", base=target),
            "python_files": [
                record(path, base=target)
                for path in sorted(solvers_target.rglob("*.py"))
            ],
        },
        "benchmark_patch": benchmark_patch,
        "runtime_input_closure": {
            "root": "exp/2026/09/21/joint-activation-material-mandible/data/expression-inputs-002",
            "files": closure_records,
        },
        "fixed_reference_shifts": {
            "cold_loaded_neutral": 0.0,
            "saved_smile": 2.8459193252829196e-7,
        },
        "cuda12_environment": {
            "setuptools_scm_pretend_version": PRETEND_VERSION,
            "metadata_files": [
                record(target / "environments/cuda12" / name, base=target)
                for name in (
                    "README.md",
                    "pyproject.toml",
                    "uv.lock",
                    ".python-version",
                    "verify.py",
                )
            ],
        },
        "payload": {
            "files": [
                record(path, base=target)
                for path in sorted(target.rglob("*"))
                if path.is_file() and path.name != "replay-manifest.json"
            ]
        },
    }
    manifest["payload"]["bytes"] = sum(
        item["bytes"] for item in manifest["payload"]["files"]
    )
    manifest_path = target / "replay-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "target": str(target),
                "manifest": str(manifest_path),
                "payload_bytes": manifest["payload"]["bytes"],
                "compatibility_adapters": 1,
                "normalized_equivalent": True,
            },
            indent=2,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    stage(args.target, replace=args.replace)


if __name__ == "__main__":
    main()
