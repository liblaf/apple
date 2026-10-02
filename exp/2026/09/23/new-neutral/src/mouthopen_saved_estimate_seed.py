# ruff: noqa: EM101, PLR0915, SLF001, TRY003, TRY300, TRY301
"""Repair an explicitly saved, possibly nonconverged collision-off estimate.

This diagnostic preserves the original strict initializer's failure. A saved
Newton state may be an unaccepted trial after a deadline; its force and energy
are recomputed here without claiming that the solver accepted it. The caller
still owns the unchanged seed/endpoint geometry caps and collision-on solve.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joint_common import sha256, write_json
from joint_equilibrium import ForwardConvergenceError
from mouthopen_collision_off_seed import _check_deadline, _contact, _save
from mouthopen_signed_pushout import signed_repair as _repair
from mouthopen_tet_policy import geometry_metrics

from liblaf.apple.forward._problem import ForwardProblem


def _identical(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Require identical shape, dtype, and bytes, including signed zero."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    actual_bytes = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    expected_bytes = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


def _materials_identical(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for name, fields in expected.items():
        assert actual[name].keys() == fields.keys()
        for field, value in fields.items():
            _identical(actual[name][field], value)


@torch.no_grad()
def prepare_saved_estimate_seed(
    physics: Any,
    materials: Any,
    old_q: torch.Tensor,
    new_q: torch.Tensor,
    old_pose_rad_m: torch.Tensor,
    new_pose_rad_m: torch.Tensor,
    seed: torch.Tensor,
    output_dir: Path,
    *,
    estimate_path: Path,
    deadline: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Verify a saved raw estimate, repair it, and restore all mutable model state.

    The source checkpoint must contain ``u_full``, ``q``, and ``pose_rad_m``;
    its sibling ``target-materials.pt`` binds the complete material state.
    There is no whole collision-off solve and no relaxed physical force gate.
    """
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime = physics.runtime
    model = runtime.forward.model
    collision = model.collision
    assert collision is not None
    original_fixed = model.dof_map.fixed_values
    original_state = runtime.forward.state
    original_u = original_state.u
    original_contact_state = original_state.collision
    original_materials = {
        name: {field: value.detach().clone() for field, value in fields.items()}
        for name, fields in model.get_materials().items()
    }
    candidate = seed.detach().clone()
    receipt: dict[str, Any] = {
        "method": "saved-collision-off-estimate-then-volume-pushout",
        "success": False,
        "stage": "validation",
        "equilibrium_claimed": False,
        "final_strict_equilibrium_required": True,
        "caller_geometry_caps_required": True,
        "accepted_newton_state_verified": False,
        "saved_state_status": "raw diagnostic state; may be an unaccepted Newton trial",
        "original_strict_initializer_result_changed": False,
        "whole_collision_off_solve_performed": False,
        "old_pose_rad_m": old_pose_rad_m.detach().cpu().tolist(),
        "new_pose_rad_m": new_pose_rad_m.detach().cpu().tolist(),
    }
    try:
        _check_deadline(deadline)
        assert old_pose_rad_m.shape == new_pose_rad_m.shape == (6,)
        assert old_q.shape == new_q.shape
        assert seed.shape == original_u.shape
        assert bool(torch.isfinite(seed).all())
        source = estimate_path.resolve(strict=True)
        source_materials = source.with_name("target-materials.pt").resolve(strict=True)
        source_hash = sha256(source)
        material_hash = sha256(source_materials)
        receipt["source_estimate"] = {"path": str(source), "sha256": source_hash}
        receipt["source_materials"] = {
            "path": str(source_materials),
            "sha256": material_hash,
        }
        saved = torch.load(source, map_location="cpu", weights_only=False)
        saved_materials = torch.load(
            source_materials, map_location="cpu", weights_only=False
        )
        assert sha256(source) == source_hash
        assert sha256(source_materials) == material_hash
        _identical(saved["q"], new_q)
        _identical(saved["pose_rad_m"], new_pose_rad_m)
        assert saved["u_full"].shape == seed.shape
        assert saved["u_full"].dtype == seed.dtype
        assert bool(torch.isfinite(saved["u_full"]).all())
        candidate = saved["u_full"].to(device=seed.device).detach().clone()
        base = physics.base if hasattr(physics, "base") else physics
        fixed_mask = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
        expected_fixed = np.r_[
            np.repeat(fixed_mask, 3),
            np.ones(model.dof_map.n_full - 3 * len(fixed_mask), dtype=bool),
        ]
        np.testing.assert_array_equal(
            model.dof_map.fixed_indices.cpu().numpy(), np.flatnonzero(expected_fixed)
        )
        np.testing.assert_array_equal(
            model.dof_map.free_indices.cpu().numpy(), np.flatnonzero(~expected_fixed)
        )
        np.testing.assert_array_equal(
            base._mouthopen_retained_tetrahedron_ids,
            np.flatnonzero(~fixed_mask[np.asarray(physics.tets)].all(axis=1)),
        )
        assert not fixed_mask[
            np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)
        ].any()
        old_fixed = physics.boundary(old_pose_rad_m).detach().clone()
        fixed = physics.boundary(new_pose_rad_m).detach().clone()
        _identical(seed.flatten()[model.dof_map.fixed_indices], old_fixed)
        _identical(candidate.flatten()[model.dof_map.fixed_indices], fixed)
        receipt["fixed_values_exact"] = True
        receipt["source_contact"] = _contact(collision, seed)
        if not receipt["source_contact"]["admitted"]:
            raise ForwardConvergenceError(
                "old source seed must be contact admissible for side hints"
            )
        target_materials = materials(new_q)
        _materials_identical(target_materials, saved_materials)
        assert "activation_inv" in target_materials["skin"]
        assert "activation_inv" in target_materials["muscle"]
        for fields in target_materials.values():
            for value in fields.values():
                assert bool(torch.isfinite(value).all())
        model.set_materials(target_materials)
        _materials_identical(model.get_materials(), target_materials)
        model.dof_map.fixed_values = fixed
        torch.save(saved_materials, output_dir / "target-materials.pt")
        _save(output_dir / "saved-estimate.pt", candidate, new_pose_rad_m, new_q)
        receipt["complete_target_materials_match_source"] = True
        receipt["stage"] = "fresh_collision_off_evaluation"
        _check_deadline(deadline)
        model.collision = None
        try:
            state = model.State(u=candidate.detach().clone(), collision=None)
            problem = ForwardProblem(model=model)
            energy = problem.fun(state)
            gradient = problem.grad(state)
            assert bool(torch.isfinite(energy))
            assert bool(torch.isfinite(gradient).all())
            force = float(torch.linalg.vector_norm(gradient))
            assert np.isfinite(force)
            atol = float(runtime.tolerances["atol"])
            assert 0 < atol <= 1e-8
            receipt["estimate_evaluation"] = {
                "energy": float(energy),
                "true_free_force_norm": force,
                "force_threshold": atol,
                "force_converged": force <= atol,
                "approximate_estimate": force > atol,
                "collision_enabled": False,
                "fresh_state_and_gradient": True,
                "accepted_newton_state_verified": False,
            }
        finally:
            model.collision = collision
        receipt["estimate_retained_geometry"] = geometry_metrics(physics, candidate)
        receipt["estimate_contact"] = _contact(collision, candidate)
        write_json(output_dir / "summary.json", receipt)
        _check_deadline(deadline)
        receipt["stage"] = "pushout"
        candidate = _repair(
            physics,
            seed,
            candidate,
            fixed,
            new_pose_rad_m,
            new_q,
            output_dir,
            receipt,
            iterations=8,
            deadline=deadline,
        )
        _identical(candidate.flatten()[model.dof_map.fixed_indices], fixed)
        receipt["retained_geometry"] = geometry_metrics(physics, candidate)
        receipt["contact"] = _contact(collision, candidate)
        assert receipt["contact"]["admitted"]
        _save(output_dir / "repaired-seed.pt", candidate, new_pose_rad_m, new_q)
        receipt["success"] = True
        receipt["stage"] = "awaiting_caller_geometry_gate_and_collision_on_solve"
        return candidate, receipt
    except Exception as error:
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        if isinstance(error, ForwardConvergenceError):
            receipt["failure"]["receipt"] = error.receipt
        _save(output_dir / "failed-stage.pt", candidate, new_pose_rad_m, new_q)
        raise
    finally:
        model.collision = collision
        model.dof_map.fixed_values = original_fixed
        model.set_materials(original_materials)
        _materials_identical(model.get_materials(), original_materials)
        assert physics.runtime is runtime
        assert runtime.forward.state is original_state
        assert original_state.u is original_u
        assert original_state.collision is original_contact_state
        receipt["model_state_restored"] = True
        receipt["seconds"] = time.perf_counter() - started
        write_json(output_dir / "summary.json", receipt)


__all__ = ["prepare_saved_estimate_seed"]
