"""Validate local active-strain FEM sparse assembly against GPU contact HVPs."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import pyvista as pv
import torch

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem
from liblaf.apple.forward.hessian._problem import HessianProblem

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
PARENT = ROOT / "exp/2026/09/29/mouthopen-activation"
sys.path.insert(0, str(PARENT / "src"))
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from active_assembled_fem import ActiveSparseHessianProblem  # noqa: E402
from experiment import Profile  # noqa: E402
from stress_physics import (  # noqa: E402
    FacePhysics,
    configure,
    strain_to_activation_inv,
)
from transition_contact import build_self_contact  # noqa: E402


class Config(cherries.BaseConfig):
    fixture: Path = PARENT / "data/30-pruned-fixture"
    saved_state: Path = PARENT / "data/70-mouthopen-four-stage/rankone_learned/last.npz"
    output: Path = cherries.output(
        "12-active-sparse-contact-check/summary.json", mkdir=True
    )
    directions: int = 3
    estimate_only: bool = False
    seed: int = 20260930
    stiffness_mpa: float = 0.0012
    minimum_distance_m: float = 1e-8


def _elapsed(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
    }


def _install_active_strain(model: Any, physics: FacePhysics, s: torch.Tensor) -> None:
    assert s.shape == (len(physics.ids), 3, 3)
    assert bool(torch.allclose(s, s.mT, rtol=0, atol=1e-12))
    physics.materials["muscle"]["activation_inv"] = torch.zeros(
        (physics.mesh.n_cells, 6), device="cuda"
    ).index_copy(0, physics.id_t, strain_to_activation_inv(s))
    model.set_materials(physics.materials)


def _contact_state(contact: Any, u: torch.Tensor) -> Any:
    state = contact.state_at(u)
    positions = (contact.vertices + u[contact.indices]).numpy(force=True)
    state.hess = contact.potential.hessian(
        collisions=state.collisions,
        mesh=contact.collision_mesh,
        X=positions,
        project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP,
    )
    return state


def _compare_state(
    *,
    name: str,
    model: Any,
    u: torch.Tensor,
    directions: int,
    seed: int,
) -> dict[str, Any]:
    state = model.State(u=u, collision=_contact_state(model.collision, u))
    problem = ForwardProblem(model=model)
    gpu_contact = HessianProblem(problem, "gpu_contact")
    gpu_sparse = ActiveSparseHessianProblem(problem)
    diagonal_contact, diagonal_contact_seconds = _elapsed(
        lambda: gpu_contact.hess_diag(state)
    )
    diagonal_sparse, diagonal_sparse_seconds = _elapsed(
        lambda: gpu_sparse.hess_diag(state)
    )
    diagonal_relative_error = float(
        torch.linalg.vector_norm(diagonal_contact - diagonal_sparse)
        / torch.linalg.vector_norm(diagonal_contact)
    )
    assert diagonal_relative_error <= 1e-12
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = []
    for index in range(directions):
        direction = torch.randn(problem.grad(state).shape, generator=generator)
        direction /= torch.linalg.vector_norm(direction)
        contact_hvp, contact_seconds = _elapsed(
            lambda direction=direction: gpu_contact.hess_prod(state, direction)
        )
        sparse_hvp, sparse_seconds = _elapsed(
            lambda direction=direction: gpu_sparse.hess_prod(state, direction)
        )
        relative_error = float(
            torch.linalg.vector_norm(contact_hvp - sparse_hvp)
            / torch.linalg.vector_norm(contact_hvp)
        )
        assert relative_error <= 1e-9
        rows.append(
            {
                "index": index,
                "relative_hvp_error": relative_error,
                "gpu_contact_seconds": contact_seconds,
                "active_sparse_seconds": sparse_seconds,
            }
        )
    return {
        "name": name,
        "active_contacts": len(state.collision.collisions),
        "diagonal_relative_error": diagonal_relative_error,
        "diagonal_seconds": {
            "gpu_contact": diagonal_contact_seconds,
            "active_sparse": diagonal_sparse_seconds,
        },
        "directions": rows,
        "maximum_relative_hvp_error": max(row["relative_hvp_error"] for row in rows),
        "gpu_contact_report": gpu_contact.report(),
        "active_sparse_report": gpu_sparse.report(),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


@torch.no_grad()
def main(cfg: Config) -> None:
    assert cfg.directions == 3
    if cfg.estimate_only:
        mesh = pv.read(cfg.fixture / "volume.vtu")
        cells = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
        fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
        raw_blocks = len(cells) * 16
        raw_scalars = raw_blocks * 9
        bsr_bytes_upper = raw_blocks * (9 * 8 + 8) + (mesh.n_points + 1) * 8
        csr_bytes_upper = raw_scalars * (8 + 8) + (3 * (~fixed).sum() + 1) * 8
        summary = {
            "schema": "experiment-local-active-strain-sparse-contact-check-v1",
            "status": "deferred_for_gpu_memory",
            "reason": "full-mesh GPU benchmark deferred while main23 owns the GPU",
            "fixture": _record(cfg.fixture / "volume.vtu"),
            "mesh": {"points": mesh.n_points, "tetrahedra": len(cells)},
            "structural_upper_bound": {
                "fem_bsr_blocks": raw_blocks,
                "fem_bsr_storage_bytes": int(bsr_bytes_upper),
                "free_scalar_entries": raw_scalars,
                "free_csr_values_and_columns_bytes": int(csr_bytes_upper),
            },
            "interpretation": "the temporary GPU union-plan tensors add to this upper bound; a full run is not safe with approximately 5 GiB free",
        }
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        cfg.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        cherries.log_metrics({"comparison/full_mesh_gpu_benchmark_deferred": 1})
        return
    configure()
    physics = FacePhysics(cfg.fixture, activation_model="strain")
    model = physics.forward.model
    contact, contact_receipt = build_self_contact(
        physics.mesh, cfg.stiffness_mpa, cfg.minimum_distance_m
    )
    contact.collision_set_type = ipctk.NormalCollisions.CollisionSetType.IPC
    fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=np.int32)
    contact.collision_mesh.can_collide = ipctk.make_vertex_patches_filter(
        fixed[contact.indices.detach().cpu().numpy()]
    )
    model.collision = contact
    with np.load(cfg.saved_state) as saved:
        saved_s = torch.as_tensor(saved["S"].copy(), device="cuda")
        saved_u = torch.as_tensor(saved["u"].copy(), device="cuda")
    assert saved_u.shape == tuple(physics.points.shape)
    zero_s = torch.zeros_like(saved_s)
    results = []
    for index, (name, s, u) in enumerate(
        (
            ("rest_zero_S", zero_s, torch.zeros_like(saved_u)),
            ("deformed_saved_S", saved_s, saved_u),
        )
    ):
        _install_active_strain(model, physics, s)
        torch.cuda.reset_peak_memory_stats()
        results.append(
            _compare_state(
                name=name,
                model=model,
                u=u,
                directions=cfg.directions,
                seed=cfg.seed + index,
            )
        )
    summary = {
        "schema": "experiment-local-active-strain-sparse-contact-check-v1",
        "status": "passed",
        "inputs": {
            "fixture": _record(cfg.fixture / "volume.vtu"),
            "saved_state": _record(cfg.saved_state),
        },
        "contact": {
            **contact_receipt,
            "collision_set_type": "IPC",
            "pair_filter": "vertex patch labels: prescribed fixed versus free; cross-patch pairs only",
        },
        "contact_hessian": "precomputed IPC physical barrier Hessian with PSDProjectionMethod.CLAMP",
        "integration": "ActiveSparseHessianProblem overrides HessianProblem._prepare to install ActiveAssembledFemHvp before the existing GpuFreeSparseHessian merge",
        "states": results,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    cherries.log_metrics(
        {
            f"{row['name']}/maximum_relative_hvp_error": row[
                "maximum_relative_hvp_error"
            ]
            for row in results
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
