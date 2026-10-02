# ruff: noqa: EM102, TRY003
"""Validate the low-memory active-strain bulk BSR plus IPC HVP backend."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
from active_bulk_contact import ActiveBulkContactHessian  # noqa: E402
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
        "13-active-bulk-contact-check/summary.json", mkdir=True
    )
    directions: int = 3
    seed: int = 20260930
    memory_fraction: float = 0.08
    minimum_free_bytes: int = 2_500_000_000
    stiffness_mpa: float = 0.0012
    minimum_distance_m: float = 1e-8


def _elapsed(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


def _install(model: Any, physics: FacePhysics, s: torch.Tensor) -> None:
    assert s.shape == (len(physics.ids), 3, 3)
    assert bool(torch.allclose(s, s.mT, rtol=0, atol=1e-12))
    physics.materials["muscle"]["activation_inv"] = torch.zeros(
        (physics.mesh.n_cells, 6), device="cuda"
    ).index_copy(0, physics.id_t, strain_to_activation_inv(s))
    model.set_materials(physics.materials)


def _state(contact: Any, model: Any, u: torch.Tensor) -> Any:
    collision = contact.state_at(u)
    positions = (contact.vertices + u[contact.indices]).numpy(force=True)
    collision.hess = contact.potential.hessian(
        collisions=collision.collisions,
        mesh=contact.collision_mesh,
        X=positions,
        project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP,
    )
    return model.State(u=u, collision=collision)


def _check(
    *, name: str, model: Any, state: Any, directions: int, seed: int
) -> dict[str, Any]:
    problem = ForwardProblem(model=model)
    baseline = HessianProblem(problem, "gpu_contact")
    low_memory = ActiveBulkContactHessian(problem)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = []
    for index in range(directions):
        direction = torch.randn(problem.grad(state).shape, generator=generator)
        direction /= torch.linalg.vector_norm(direction)
        reference, reference_seconds = _elapsed(
            lambda direction=direction: baseline.hess_prod(state, direction)
        )
        candidate, candidate_seconds = _elapsed(
            lambda direction=direction: low_memory.hess_prod(state, direction)
        )
        relative_error = float(
            torch.linalg.vector_norm(reference - candidate)
            / torch.linalg.vector_norm(reference)
        )
        assert relative_error <= 1e-9
        rows.append(
            {
                "index": index,
                "relative_hvp_error": relative_error,
                "gpu_contact_seconds": reference_seconds,
                "active_bulk_contact_seconds": candidate_seconds,
            }
        )
    return {
        "name": name,
        "active_contacts": len(state.collision.collisions),
        "directions": rows,
        "maximum_relative_hvp_error": max(row["relative_hvp_error"] for row in rows),
        "gpu_contact_report": baseline.report(),
        "active_bulk_contact_report": low_memory.report(),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


@torch.no_grad()
def main(cfg: Config) -> None:
    assert cfg.directions == 3
    assert 0 < cfg.memory_fraction <= 0.1
    configure()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < cfg.minimum_free_bytes:
        raise RuntimeError(
            f"need {cfg.minimum_free_bytes} free GPU bytes, found {free_bytes}"
        )
    torch.cuda.set_per_process_memory_fraction(cfg.memory_fraction)
    torch.cuda.reset_peak_memory_stats()
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
    with np.load(cfg.saved_state) as saved:
        saved_s = torch.as_tensor(saved["S"].copy(), device="cuda")
        saved_u = torch.as_tensor(saved["u"].copy(), device="cuda")
    assert saved_u.shape == tuple(physics.points.shape)
    states = []
    for index, (name, s, u) in enumerate(
        (
            ("rest_zero_S", torch.zeros_like(saved_s), torch.zeros_like(saved_u)),
            ("deformed_saved_S", saved_s, saved_u),
        )
    ):
        _install(model, physics, s)
        torch.cuda.reset_peak_memory_stats()
        states.append(
            _check(
                name=name,
                model=model,
                state=_state(contact, model, u),
                directions=cfg.directions,
                seed=cfg.seed + index,
            )
        )
    summary = {
        "schema": "active-bulk-bsr-plus-ipc-product-check-v1",
        "status": "passed",
        "saved_state_sha256": hashlib.sha256(cfg.saved_state.read_bytes()).hexdigest(),
        "gpu_memory_before_bytes": {"free": free_bytes, "total": total_bytes},
        "memory_fraction": cfg.memory_fraction,
        "contact": {
            **contact_receipt,
            "collision_set_type": "IPC",
            "pair_filter": "all fixed vertices share patch 0; every free vertex has a distinct patch, excluding fixed/fixed only",
        },
        "contact_hessian": "PSDProjectionMethod.CLAMP before each backend product",
        "reuse": "one bulk BSR matrix per model topology, refreshed in place as displacement/materials change; IPC CSR upload reused per collision state",
        "states": states,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    cherries.log_metrics(
        {
            f"{row['name']}/maximum_relative_hvp_error": row[
                "maximum_relative_hvp_error"
            ]
            for row in states
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
