# Copyright (c) 2026 liblaf
"""Bounded GPU wiring check for the repaired fixed-reference contact model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem
from liblaf.apple.forward.hessian._problem import HessianProblem

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
PARENT = ROOT / "exp/2026/09/29/mouthopen-activation"
sys.path[:0] = [
    str(GROUP / "src"),
    str(PARENT / "src"),
    str(ROOT / "exp/2026/09/21/stress-activation-loss/src"),
]
from experiment import Profile  # noqa: E402
from fixed_reference_contact import (  # noqa: E402
    attach_fixed_reference_contact,
    audit_fixed_reference_contact,
    build_fixed_reference_contact,
)
from stress_physics import FacePhysics, configure  # noqa: E402


class Config(cherries.BaseConfig):
    output: Path = Path("53-fixed-reference-gpu-check")
    fixture: Path = GROUP / "data/50-fixed-reference/fixture"
    pose: Path = PARENT / "data/10-mandible/prepared.npz"
    stiffness_mpa: float = 1.3544
    dhat_m: float = 1e-4
    minimum_distance_m: float = 1e-8
    pose_fraction: float = 0.002
    memory_fraction: float = 0.1


@torch.no_grad()
def main(cfg: Config) -> None:
    assert 0 < cfg.pose_fraction < 1
    assert 0 < cfg.memory_fraction <= 0.1
    out = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not out.exists()
    configure()
    free_before, total = torch.cuda.mem_get_info()
    assert free_before > 3_000_000_000
    torch.cuda.set_per_process_memory_fraction(cfg.memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    physics = FacePhysics(cfg.fixture, activation_model="strain")
    reference = build_fixed_reference_contact(
        physics.mesh,
        stiffness_mpa=cfg.stiffness_mpa,
        dhat_m=cfg.dhat_m,
        minimum_distance_m=cfg.minimum_distance_m,
    )
    points = torch.as_tensor(physics.points)
    with np.load(cfg.pose, allow_pickle=False) as archive:
        full_pose = torch.as_tensor(archive["pose"].copy())
        pivot = torch.as_tensor(archive["pivot"].copy())
    zero_pose = torch.zeros_like(full_pose)
    zero_values = reference.boundary_values(points, zero_pose, pivot)
    model = attach_fixed_reference_contact(
        physics.forward.model, reference, zero_values
    )
    state = model.init()
    assert state.u.shape == (reference.full_point_count, 3)
    assert state.u.is_cuda
    problem = ForwardProblem(model)
    force = problem.grad(state)
    force_l2 = float(torch.linalg.vector_norm(force))
    assert force_l2 <= 1e-18
    rest = audit_fixed_reference_contact(reference, state.u)
    assert rest["contact_numerically_valid"]
    assert rest["clearance_at_least_dhat"]
    small_pose = cfg.pose_fraction * full_pose
    small_values = reference.boundary_values(points, small_pose, pivot)
    fraction = float(
        reference.contact.max_step_size(
            state.collision, state.u, small_values - state.u
        )
    )
    assert fraction == 1.0
    posed_model = attach_fixed_reference_contact(
        physics.forward.model, reference, small_values
    )
    posed_state = posed_model.init()
    assert torch.equal(posed_state.u, small_values)
    posed = audit_fixed_reference_contact(reference, posed_state.u)
    assert posed["contact_numerically_valid"]
    generator = torch.Generator(device="cuda").manual_seed(20260930)
    direction = torch.randn(posed_model.n_free, generator=generator, device="cuda")
    direction /= torch.linalg.vector_norm(direction)
    posed_problem = ForwardProblem(posed_model)
    direct = posed_problem.hess_prod(posed_state, direction)
    gpu = HessianProblem(posed_problem, "gpu_contact").hess_prod(posed_state, direction)
    relative = float(
        torch.linalg.vector_norm(direct - gpu) / torch.linalg.vector_norm(direct)
    )
    assert relative <= 1e-11
    free_after, _ = torch.cuda.mem_get_info()
    result = {
        "schema": "fixed-reference-gpu-wiring-check-v1",
        "fixture": str(cfg.fixture.resolve()),
        "model": {
            "physical_points": physics.mesh.n_points,
            "full_points": model.n_points,
            "physical_free_dofs": physics.forward.model.n_free,
            "extended_free_dofs": model.n_free,
            "fixed_dofs": model.n_fixed,
            "materials": sorted(model.get_materials()),
        },
        "rest": {"free_force_l2_mpa_m2": force_l2, **rest},
        "small_pose": {
            "fraction": cfg.pose_fraction,
            "ccd_fraction": fraction,
            **posed,
        },
        "hvp": {
            "backend": "core gpu_contact",
            "reference": "core matrix_free",
            "relative_error": relative,
        },
        "gpu_memory": {
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "total_bytes": total,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
