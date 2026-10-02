"""Explicit current-runtime binding of immutable historical neutral inputs."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joint_common import sha256, write_json
from joint_data import array_sha256
from joint_frozen_neutral import FrozenNeutral
from joint_materials import StableNeoHookeanStress

from liblaf.apple.collision._collision import Collision
from liblaf.apple.warp.fem import WarpPotentialFem

IMPORT_MIGRATIONS = {
    "apple/forward/_forward.py",
    "apple/forward/_problem.py",
    "apple/inverse/_diff_forward.py",
    "experiment/joint_equilibrium.py",
    "experiment/joint_newton.py",
    "tensor-reference/face_physics.py",
}


def _code_without_imports(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            node.module = node.module.replace("liblaf.peach", "liblaf.apple.solvers")
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    return ast.dump(tree, include_attributes=False)


def _imports(source: str) -> list[str]:
    return sorted(
        ast.dump(node, include_attributes=False).replace(
            "liblaf.peach", "liblaf.apple.solvers"
        )
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )


def load_current_binding(neutral_dir: Path, output_dir: Path) -> FrozenNeutral:
    """Validate original data and bind only independently audited runtime changes.

    The historical manifest and selected-neutral pointer are never modified.
    """
    manifest_path = neutral_dir / "manifest.json"
    original = json.loads(manifest_path.read_text())
    assert original["schema"] == "joint-frozen-neutral-v1"
    assert original["success"] is True
    for item in [*original["sources"].values(), *original["artifacts"].values()]:
        assert sha256(Path(item["path"])) == item["sha256"], item["path"]
    with np.load(neutral_dir / "state.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    assert set(arrays) == set(original["arrays"])
    for name, value in arrays.items():
        assert array_sha256(value) == original["arrays"][name]["sha256"], name

    rebound = copy.deepcopy(original)
    changes = {}
    for name, item in original["runtime_sources"].items():
        current_path = Path(item["path"])
        current_hash = sha256(current_path)
        if current_hash == item["sha256"]:
            continue
        assert name in IMPORT_MIGRATIONS | {"apple/_version.py", "uv.lock"}, name
        if name == "uv.lock":
            old_bytes = subprocess.check_output(
                ["git", "show", "HEAD:uv.lock"], cwd=current_path.parent
            )
            assert hashlib.sha256(old_bytes).hexdigest() == item["sha256"]
            old_packages = {
                row["name"]: row.get("version")
                for row in tomllib.loads(old_bytes.decode())["package"]
            }
            current_packages = {
                row["name"]: row.get("version")
                for row in tomllib.loads(current_path.read_text())["package"]
            }
            assert old_packages.keys() - current_packages.keys() == {"liblaf-peach"}
            assert current_packages.keys() - old_packages.keys() == {"cupy-cuda12x"}
            assert all(
                old_packages[key] == current_packages[key]
                for key in old_packages.keys() & current_packages.keys()
            )
            (output_dir / "historical-uv.lock").write_bytes(old_bytes)
            reason = "solver package migration and CUDA extra resolution; all shared package versions unchanged"
            archived_source = "git HEAD:uv.lock, exact historical SHA-256 verified"
        else:
            archived = neutral_dir / "sources" / name
            assert sha256(archived) == item["sha256"], archived
            old_text, current_text = archived.read_text(), current_path.read_text()
            if name in IMPORT_MIGRATIONS:
                assert _code_without_imports(old_text) == _code_without_imports(
                    current_text
                ), name
                assert _imports(old_text) == _imports(current_text), name
                reason = "imports only: liblaf.peach solver interfaces moved to liblaf.apple.solvers"
            else:

                def without_version(source: str) -> list[str]:
                    return [
                        line
                        for line in source.splitlines()
                        if not line.startswith(("__version__ =", "__version_tuple__ ="))
                    ]

                assert without_version(old_text) == without_version(current_text)
                reason = "generated version metadata only"
            archived_source = str(archived.resolve())
        new_item = {
            **item,
            "sha256": current_hash,
            "bytes": current_path.stat().st_size,
        }
        rebound["runtime_sources"][name] = new_item
        changes[name] = {
            "historical": item,
            "current": new_item,
            "verified_historical_source": archived_source,
            "reason": reason,
        }
    write_json(
        output_dir / "model-binding.json",
        {
            "schema": "neutral-current-runtime-binding-v1",
            "historical_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256(manifest_path),
            },
            "historical_data_and_arrays_verified": True,
            "historical_manifest_modified": False,
            "physical_material_reference_and_contact_source_hashes_unchanged": True,
            "runtime_changes": changes,
        },
    )
    return FrozenNeutral(arrays=arrays, manifest=rebound, directory=neutral_dir)


def _exact_collision_diagonal(
    self: Collision, state: Any, u: torch.Tensor, output: torch.Tensor
) -> None:
    if state.hess is None:
        positions = (self.vertices + u[self.indices]).numpy(force=True)
        state.hess = self.potential.hessian(
            collisions=state.collisions, mesh=self.collision_mesh, X=positions
        )
    diagonal = torch.as_tensor(
        state.hess.diagonal(), device=output.device, dtype=output.dtype
    )
    output.index_add_(0, self.indices, diagonal.reshape(self.vertices.shape))


def install_exact_bulk_diagonal() -> dict:
    """Install exact bulk and IPC diagonals in this process for literal Jacobi.

    Membrane diagonal is already analytic and unclamped. Physical energies,
    gradients, and Hessian-vector products are unchanged.
    """
    StableNeoHookeanStress.hess_diag_kernel = WarpPotentialFem.make_hess_diag_kernel(
        StableNeoHookeanStress.hess_diag_func,
        clamp_hess_diag=False,
    )
    Collision.hess_diag = _exact_collision_diagonal
    return {
        "scope": "process-local kernel/method binding; source files unchanged",
        "bulk": "analytic unprojected diagonal; per-cell negative entries retained before assembly",
        "skin": "existing analytic exact plane-stress membrane diagonal; no clamping",
        "contact": "diagonal of the same exact cached IPC Hessian used by Hessian-vector products",
        "energy_gradient_and_hessian_vector_products_changed": False,
    }
