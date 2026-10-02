# ruff: noqa: EM101, TRY003
"""Bounded unshifted-PCG probe for the rest contact Newton system."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import ipctk
import numpy as np
import torch

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem
from liblaf.apple.forward.hessian._problem import HessianProblem

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
PARENT = ROOT / "exp/2026/09/29/mouthopen-activation"
sys.path.insert(0, str(PARENT / "src"))
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
sys.path.insert(0, str(ROOT / "exp/2026/09/22/solver-performance/src"))
from accelerated_solvers import LinearRejection, pcg  # noqa: E402
from experiment import Profile  # noqa: E402
from stress_physics import (  # noqa: E402
    FacePhysics,
    configure,
    strain_to_activation_inv,
)
from transition_contact import build_self_contact  # noqa: E402


class Config(cherries.BaseConfig):
    fixture: Path = PARENT / "data/30-pruned-fixture"
    output: Path = cherries.output("14-unshifted-pcg-probe/summary.json", mkdir=True)
    linear_rtol: float = 1e-3
    max_steps: int = 20_000
    wall_seconds: float = 120.0
    memory_fraction: float = 0.07
    minimum_free_bytes: int = 2_500_000_000
    stiffness_mpa: float = 0.0012
    minimum_distance_m: float = 1e-8


@torch.no_grad()
def main(cfg: Config) -> None:
    assert 0 < cfg.linear_rtol < 1
    assert cfg.max_steps == 20_000
    assert cfg.wall_seconds > 0
    configure()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < cfg.minimum_free_bytes:
        message = f"need {cfg.minimum_free_bytes} free GPU bytes, found {free_bytes}"
        raise RuntimeError(message)
    torch.cuda.set_per_process_memory_fraction(cfg.memory_fraction)
    physics = FacePhysics(cfg.fixture, activation_model="strain")
    model = physics.forward.model
    contact, contact_receipt = build_self_contact(
        physics.mesh, cfg.stiffness_mpa, cfg.minimum_distance_m
    )
    contact.collision_set_type = ipctk.NormalCollisions.CollisionSetType.IPC
    fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    patches = np.where(fixed, 0, np.arange(physics.mesh.n_points) + 1).astype(np.int32)
    contact.collision_mesh.can_collide = ipctk.make_vertex_patches_filter(
        patches[contact.indices.detach().cpu().numpy()]
    )
    model.collision = contact
    zero_s = torch.zeros((len(physics.ids), 3, 3), device="cuda")
    physics.materials["muscle"]["activation_inv"] = torch.zeros(
        (physics.mesh.n_cells, 6), device="cuda"
    ).index_copy(0, physics.id_t, strain_to_activation_inv(zero_s))
    model.set_materials(physics.materials)
    u = torch.zeros_like(torch.as_tensor(physics.points))
    collision = contact.state_at(u)
    positions = (contact.vertices + u[contact.indices]).numpy(force=True)
    collision.hess = contact.potential.hessian(
        collisions=collision.collisions,
        mesh=contact.collision_mesh,
        X=positions,
        project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP,
    )
    state = model.State(u=u, collision=collision)
    problem = ForwardProblem(model=model)
    hessian = HessianProblem(problem, "gpu_contact")
    force = problem.grad(state)
    rhs = -force
    diagonal = hessian.hess_diag(state)
    diagonal_positive = bool(torch.all(diagonal > 0))
    started = time.perf_counter()
    products = 0

    def matvec(direction: torch.Tensor) -> torch.Tensor:
        nonlocal products
        if time.perf_counter() - started > cfg.wall_seconds:
            raise TimeoutError("unshifted PCG wall budget exhausted")
        products += 1
        return hessian.hess_prod(state, direction)

    def precondition(residual: torch.Tensor) -> torch.Tensor:
        return residual / diagonal.abs()

    result: dict[str, object]
    try:
        _, receipt = pcg(
            matvec,
            precondition,
            rhs,
            rtol=cfg.linear_rtol,
            max_steps=cfg.max_steps,
        )
        result = {"status": "converged", **receipt}
    except (LinearRejection, TimeoutError) as error:
        result = {"status": "rejected_or_budgeted", "reason": str(error)}
    elapsed = time.perf_counter() - started
    summary = {
        "schema": "unshifted-rest-pcg-contact-probe-v1",
        "force_l2": float(torch.linalg.vector_norm(force)),
        "active_contacts": len(collision.collisions),
        "linear_rtol": cfg.linear_rtol,
        "max_steps": cfg.max_steps,
        "wall_seconds": cfg.wall_seconds,
        "elapsed_seconds": elapsed,
        "hessian_products": products,
        "diagonal_positive": diagonal_positive,
        "diagonal_minimum": float(diagonal.min()),
        "preconditioner": "main safeguarded_newton_step diagonal.abs() at zero shift",
        "result": result,
        "gpu_memory_before_bytes": {"free": free_bytes, "total": total_bytes},
        "contact": {
            **contact_receipt,
            "collision_set_type": "IPC",
            "pair_filter": "all fixed vertices share patch 0; every free vertex has a distinct patch, excluding fixed/fixed only",
            "hessian_projection": "PSDProjectionMethod.CLAMP",
        },
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    cherries.log_metrics(
        {
            "pcg/force_l2": summary["force_l2"],
            "pcg/hessian_products": products,
            "pcg/elapsed_seconds": elapsed,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
