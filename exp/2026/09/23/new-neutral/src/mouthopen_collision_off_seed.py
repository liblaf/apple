# ruff: noqa: C901, EM101, PLR0912, PLR0915, SLF001, TRY003, TRY300, TRY301
"""Fixed-pose collision-off estimate followed by contact seed repair.

The returned displacement is only an initializer. The caller must apply its
unchanged retained-tetrahedron allowance and run the strict collision-on solve.
The default repair uses verified closed rigid geometry, ray containment, and
closest-feature distances. The earlier local-plane routine remains below for
diagnostic provenance; it is no longer selected by this initializer.
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path
from typing import Any, override

import ipctk
import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from accelerated_solvers import CachedProblem, safeguarded_newton
from hybrid_first_solver import SparseNewtonProblem
from joint_common import write_json
from joint_data import _collision_geometry
from joint_equilibrium import ForwardConvergenceError
from mouthopen_tet_policy import geometry_metrics
from pncg_first import run_pncg_phase
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

from liblaf.apple.forward._problem import ForwardProblem


class _CollisionOffProblem(ForwardProblem):
    @override
    def max_step_size(self, state: Any, direction: torch.Tensor) -> torch.Tensor:
        assert self.model.collision is None
        assert state.collision is None
        return torch.ones((), device=direction.device, dtype=direction.dtype)


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise ForwardConvergenceError("declared seed wall budget exhausted")


def _save(path: Path, u: torch.Tensor, pose: torch.Tensor, q: torch.Tensor) -> None:
    """Keep the latest numeric state even when a subsequent stage rejects it."""
    torch.save(
        {
            "u_full": u.detach().cpu(),
            "pose_rad_m": pose.detach().cpu(),
            "q": q.detach().cpu(),
        },
        path,
    )


def _poly(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    return pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)))


def _normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1)
    assert np.all(lengths > 0)
    return normals / lengths[:, None]


def _contact(collision: Any, full: torch.Tensor) -> dict[str, Any]:
    state = collision.state_at(full)
    receipt = collision.diagnostics(state, full)
    # An actual penetration can yield an infinite IPC barrier. Preserve the
    # invalid gate and label its value without making the JSON evidence fail.
    nonfinite = {
        name: str(value)
        for name, value in receipt.items()
        if isinstance(value, float) and not math.isfinite(value)
    }
    for name in nonfinite:
        receipt[name] = None
    if nonfinite:
        receipt["nonfinite_fields"] = nonfinite
        receipt["contact_numerically_valid"] = False
    points = (collision.vertices + full[collision.indices]).numpy(force=True)
    intersects = bool(
        ipctk.has_intersections(collision.collision_mesh, points, ipctk.LBVH())
    )
    gap = receipt["minimum_active_distance_m"]
    return {
        **receipt,
        "no_intersections": not intersects,
        "admitted": bool(
            receipt["contact_numerically_valid"]
            and not intersects
            and (gap is None or gap >= collision.min_distance)
        ),
    }


class _VolumeExtension:
    """Positive volume-weighted graph Laplacian on the reference tetrahedra."""

    def __init__(
        self,
        points: np.ndarray,
        tets: np.ndarray,
        surface_ids: np.ndarray,
        fixed_ids: np.ndarray,
    ) -> None:
        self.surface_ids = surface_ids
        self.fixed_ids = fixed_ids
        self.boundary = np.union1d(surface_ids, fixed_ids)
        self.interior = np.setdiff1d(np.arange(len(points)), self.boundary)
        self.shape = points.shape
        edges = np.concatenate(
            [tets[:, pair] for pair in itertools.combinations(range(4), 2)]
        )
        rest_det = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]])
        assert np.all(rest_det > 0)
        lengths2 = np.sum((points[edges[:, 0]] - points[edges[:, 1]]) ** 2, axis=1)
        assert np.all(lengths2 > 0)
        weights = np.tile(rest_det / 6, 6) / lengths2
        adjacency = sp.coo_matrix(
            (
                np.r_[weights, weights],
                (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
            ),
            shape=(len(points), len(points)),
        ).tocsr()
        laplacian = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
        self.matrix = laplacian[self.interior][:, self.interior].tocsr()
        self.coupling = laplacian[self.interior][:, self.boundary].tocsr()
        diagonal = self.matrix.diagonal()
        assert np.all(diagonal > 0)
        self.preconditioner = sp.diags(1 / diagonal)

    def extend(
        self, surface_increment: np.ndarray, deadline: float | None
    ) -> tuple[np.ndarray, list[float]]:
        increment = np.zeros(self.shape, dtype=np.float64)
        increment[self.surface_ids] = surface_increment
        increment[self.fixed_ids] = 0
        residuals = []
        for axis in range(3):
            _check_deadline(deadline)
            rhs = -(self.coupling @ increment[self.boundary, axis])
            if not len(self.interior):
                residuals.append(0.0)
                continue
            value, info = spla.cg(
                self.matrix,
                rhs,
                M=self.preconditioner,
                rtol=1e-9,
                atol=0,
                maxiter=4000,
                callback=lambda _: _check_deadline(deadline),
            )
            residual = np.linalg.norm(self.matrix @ value - rhs)
            allowed = 1e-9 * np.linalg.norm(rhs)
            if info != 0 or not np.isfinite(value).all() or residual > allowed:
                raise ForwardConvergenceError(
                    "volume extension failed its linear residual gate",
                    receipt={
                        "axis": axis,
                        "info": int(info),
                        "residual": float(residual),
                        "allowed": float(allowed),
                    },
                )
            increment[self.interior, axis] = value
            residuals.append(float(residual))
        return increment, residuals


def _repair(
    physics: Any,
    old: torch.Tensor,
    candidate: torch.Tensor,
    fixed: torch.Tensor,
    pose: torch.Tensor,
    q: torch.Tensor,
    output: Path,
    receipt: dict,
    *,
    iterations: int,
    deadline: float | None,
) -> torch.Tensor:
    model = physics.runtime.forward.model
    collision = model.collision
    geometry = physics.full_skull.geometry
    reference = np.asarray(physics.points, dtype=np.float64)
    n_fem = len(reference)
    soft_ids = np.asarray(geometry.soft_global_ids, dtype=np.int64)
    soft_faces = np.asarray(geometry.soft_faces, dtype=np.int64)
    fixed_mask = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    fixed_ids = np.flatnonzero(fixed_mask)
    free_surface = ~fixed_mask[soft_ids]
    assert free_surface.any()
    # Use exactly the existing collision mesh's vertex map and geometry.
    counts = [
        len(soft_ids),
        len(geometry.cranium_points_m),
        len(geometry.mandible_points_m),
        len(physics.eyes.points_m),
    ]
    offsets = np.r_[0, np.cumsum(counts)]
    assert offsets[-1] == len(collision.vertices)
    np.testing.assert_array_equal(
        collision.indices[: len(soft_ids)].cpu().numpy(), soft_ids
    )
    expected_rest = np.concatenate(
        (
            reference[soft_ids],
            geometry.cranium_points_m,
            geometry.mandible_points_m,
            physics.eyes.points_m,
        )
    )
    np.testing.assert_allclose(
        collision.vertices.cpu().numpy(), expected_rest, rtol=0, atol=1e-14
    )
    old_points = (collision.vertices + old[collision.indices]).numpy(force=True)
    target_points = (collision.vertices + candidate[collision.indices]).numpy(
        force=True
    )
    obstacles = []
    for index, (name, face_data) in enumerate(
        (
            ("cranium", geometry.cranium_faces),
            ("mandible", geometry.mandible_faces),
            ("eyes", physics.eyes.triangles),
        ),
        start=1,
    ):
        faces = np.asarray(face_data, dtype=np.int64)
        points = target_points[offsets[index] : offsets[index + 1]].copy()
        previous = old_points[offsets[index] : offsets[index + 1]]
        obstacle = _poly(points, faces)
        signed = None
        if name == "eyes":
            # The saved eyes duplicate seam vertices. Exact welding closes
            # those seams without moving any point or changing the collision
            # mesh. Use the welded copy only for the containment query.
            closed_eye = obstacle.clean(tolerance=0.0, absolute=True)
            assert closed_eye.n_open_edges == 0
            closed_eye = closed_eye.compute_normals(
                cell_normals=True,
                point_normals=False,
                consistent_normals=True,
                auto_orient_normals=True,
                split_vertices=False,
            )
            signed = vtkImplicitPolyDataDistance()
            signed.SetInput(closed_eye)
            receipt["eye_containment_topology"] = {
                "collision_mesh_unchanged": True,
                "original_vertices": obstacle.n_points,
                "original_open_edges": obstacle.n_open_edges,
                "exact_welded_vertices": closed_eye.n_points,
                "exact_welded_open_edges": closed_eye.n_open_edges,
                "query_only_exact_duplicate_welding": True,
            }
        obstacles.append(
            (
                name,
                points,
                faces,
                obstacle,
                _normals(points, faces),
                previous,
                _normals(previous, faces),
                signed,
            )
        )
    extension = _VolumeExtension(
        reference, np.asarray(physics.tets, dtype=np.int64), soft_ids, fixed_ids
    )
    clearance = 1.001 * float(collision.potential.dhat)
    receipt["projection"] = {
        "method": "intersecting-triangle-plane-separation-and-volume-graph-harmonic-extension",
        "open_bone_side_rule": "old admissible soft-face centroid side of corresponding old rigid triangle plane",
        "open_bone_global_inside_outside_claimed": False,
        "requested_clearance_m": clearance,
        "requested_clearance_applies_only_to_repaired_features": True,
        "contact_gate_clearance_m": float(collision.min_distance),
        "free_surface_vertices": int(free_surface.sum()),
        "fixed_surface_vertices": int((~free_surface).sum()),
        "trace": [],
    }
    for iteration in range(iterations + 1):
        _check_deadline(deadline)
        contact = _contact(collision, candidate)
        retained = geometry_metrics(physics, candidate)
        row = {
            "iteration": iteration,
            "contact": contact,
            "retained_geometry": retained,
        }
        receipt["projection"]["trace"].append(row)
        _save(output / "projection-latest.pt", candidate, pose, q)
        write_json(output / "summary.json", receipt)
        # Closed eyes also need a containment check: disjoint surfaces can be nested.
        surface = reference[soft_ids] + candidate[:n_fem].cpu().numpy()[soft_ids]
        eye_signed = np.asarray(
            [obstacles[-1][-1].EvaluateFunction(point) for point in surface]
        )
        row["inside_eye_vertices"] = int(np.count_nonzero(eye_signed < 0))
        if contact["admitted"] and np.all(eye_signed >= float(collision.min_distance)):
            return candidate
        if iteration == iterations:
            break
        projected = surface.copy()
        measures = {}
        for (
            name,
            points,
            faces,
            obstacle,
            normals,
            previous,
            old_normals,
            signed,
        ) in obstacles:
            _check_deadline(deadline)
            pairs, *_ = _collision_geometry(_poly(projected, soft_faces), obstacle)
            for soft_face, rigid_face in pairs:
                ids = soft_faces[soft_face]
                moving = ids[free_surface[ids]]
                if not len(moving):
                    raise ForwardConvergenceError(
                        "intersecting soft face has no free vertices",
                        receipt={"obstacle": name, "soft_face": int(soft_face)},
                    )
                normal = normals[rigid_face]
                base = points[faces[rigid_face, 0]]
                if signed is None:
                    hint = float(
                        (old_points[ids].mean(axis=0) - previous[faces[rigid_face, 0]])
                        @ old_normals[rigid_face]
                    )
                    if abs(hint) <= 1e-14:
                        raise ForwardConvergenceError(
                            "open-surface contact-plane side is ambiguous",
                            receipt={"obstacle": name, "rigid_face": int(rigid_face)},
                        )
                    normal = normal * np.sign(hint)
                else:
                    normal = np.asarray(obstacle.cell_normals)[rigid_face]
                amount = np.maximum(clearance - (projected[moving] - base) @ normal, 0)
                projected[moving] += amount[:, None] * normal
            cells, closest = obstacle.find_closest_cell(
                projected, return_closest_point=True
            )
            direction = projected - np.asarray(closest)
            distance = np.linalg.norm(direction, axis=1)
            near = free_surface & (distance < clearance)
            gap = distance.copy()
            if signed is not None:
                gap = np.asarray(
                    [signed.EvaluateFunction(point) for point in projected]
                )
                near = free_surface & (gap < clearance)
                direction[gap < 0] *= -1
            nonzero = distance > 1e-14
            direction[nonzero] /= distance[nonzero, None]
            # Exact coincidence has no distance direction; use the old side hint.
            for vertex in np.flatnonzero(near & ~nonzero):
                cell = int(cells[vertex])
                hint = float(
                    (old_points[vertex] - previous[faces[cell, 0]]) @ old_normals[cell]
                )
                if abs(hint) <= 1e-14:
                    raise ForwardConvergenceError(
                        "closest-feature pushout direction is ambiguous"
                    )
                direction[vertex] = normals[cell] * np.sign(hint)
            projected[near] += (clearance - gap[near])[:, None] * direction[near]
            measures[name] = {
                "intersection_pairs": len(pairs),
                "clearance_updates": int(near.sum()),
                "minimum_sample_gap_m": float(gap.min()),
                "sample_gap_is_signed": signed is not None,
            }
        np.testing.assert_array_equal(projected[~free_surface], surface[~free_surface])
        increment, residuals = extension.extend(projected - surface, deadline)
        row.update(
            {
                "obstacles": measures,
                "maximum_surface_increment_m": float(
                    np.linalg.norm(projected - surface, axis=1).max()
                ),
                "extension_absolute_residuals": residuals,
            }
        )
        candidate = candidate.detach().clone()
        candidate[:n_fem] += torch.as_tensor(
            increment, device=candidate.device, dtype=candidate.dtype
        )
        candidate.flatten()[model.dof_map.fixed_indices] = fixed
        torch.testing.assert_close(
            candidate.flatten()[model.dof_map.fixed_indices], fixed, rtol=0, atol=0
        )
    raise ForwardConvergenceError(
        "collision-off seed pushout exhausted its contact repair budget",
        receipt=receipt["projection"],
    )


@torch.no_grad()
def prepare_collision_off_seed(
    physics: Any,
    materials: Any,
    old_q: torch.Tensor,
    new_q: torch.Tensor,
    old_pose_rad_m: torch.Tensor,
    new_pose_rad_m: torch.Tensor,
    seed: torch.Tensor,
    output_dir: Path,
    *,
    forward_atol: float | None = None,
    max_newton_steps: int | None = None,
    off_wall_seconds: float = 300.0,
    no_contact_linear_max_steps: int = 1000,
    deadline: float | None = None,
    pushout_iterations: int = 8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate at target materials/6DOF pose, repair contact, restore model state.

    ``materials(new_q)`` must return every material field, including skin
    prestrain. Already-installed corrected IsFixed and all-fixed-tet exclusion
    are asserted. No implicit backward, collision-on solve, or geometry cap is
    performed here. The caller owns the unchanged geometry-cap checks.
    """
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime = physics.runtime
    model = runtime.forward.model
    collision = model.collision
    assert collision is not None
    assert old_pose_rad_m.shape == new_pose_rad_m.shape == (6,)
    assert old_q.shape == new_q.shape
    assert seed.shape == runtime.forward.state.u.shape
    assert off_wall_seconds > 0
    assert pushout_iterations > 0
    assert no_contact_linear_max_steps > 0
    original_materials = {
        name: {field: value.detach().clone() for field, value in fields.items()}
        for name, fields in model.get_materials().items()
    }
    original_fixed = model.dof_map.fixed_values
    original_state = runtime.forward.state
    old_runtime_u = original_state.u
    old_runtime_contact = original_state.collision
    local_state = model.State(u=seed.detach().clone(), collision=None)
    receipt: dict[str, Any] = {
        "method": "fixed-target-pose-collision-off-hybrid-then-volume-pushout",
        "success": False,
        "equilibrium_claimed": False,
        "final_strict_equilibrium_required": True,
        "caller_geometry_caps_required": True,
        "collision_disabled_during_estimate": True,
        "ccd_disabled_during_estimate": True,
        "old_pose_rad_m": old_pose_rad_m.cpu().tolist(),
        "new_pose_rad_m": new_pose_rad_m.cpu().tolist(),
        "stage": "validation",
    }
    candidate = seed.detach().clone()
    try:
        _check_deadline(deadline)
        base = physics.base if hasattr(physics, "base") else physics
        retained_ids = np.asarray(base._mouthopen_retained_tetrahedron_ids)
        fixed_mask = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
        expected_fixed = np.repeat(fixed_mask, 3)
        actual = model.dof_map.fixed_indices.cpu().numpy()
        np.testing.assert_array_equal(
            actual[actual < len(expected_fixed)], np.flatnonzero(expected_fixed)
        )
        np.testing.assert_array_equal(
            retained_ids,
            np.flatnonzero(~fixed_mask[np.asarray(physics.tets)].all(axis=1)),
        )
        assert "activation_inv" in original_materials["skin"]
        assert "activation_inv" in original_materials["muscle"]
        old_fixed = physics.boundary(old_pose_rad_m)
        torch.testing.assert_close(
            seed.flatten()[model.dof_map.fixed_indices], old_fixed, rtol=0, atol=1e-13
        )
        receipt["source_contact"] = _contact(collision, seed)
        if not receipt["source_contact"]["admitted"]:
            raise ForwardConvergenceError(
                "source seed must be contact admissible for side hints"
            )
        target_materials = materials(new_q)
        assert target_materials.keys() == original_materials.keys()
        for name, fields in target_materials.items():
            assert fields.keys() == original_materials[name].keys()
            for field, value in fields.items():
                assert value.shape == original_materials[name][field].shape
                assert bool(torch.isfinite(value).all())
        torch.save(
            {
                name: {field: value.detach().cpu() for field, value in fields.items()}
                for name, fields in target_materials.items()
            },
            output_dir / "target-materials.pt",
        )
        model.set_materials(target_materials)
        fixed = physics.boundary(new_pose_rad_m).detach().clone()
        model.dof_map.fixed_values = fixed
        local_state.u = (
            model.dof_map.to_full(model.dof_map.to_free(seed)).detach().clone()
        )
        _save(output_dir / "target-boundary.pt", local_state.u, new_pose_rad_m, new_q)
        model.collision = None
        receipt["stage"] = "collision_off"
        atol = runtime.tolerances["atol"] if forward_atol is None else forward_atol
        assert 0 < atol <= runtime.tolerances["atol"]
        steps = (
            runtime.newton_max_steps if max_newton_steps is None else max_newton_steps
        )
        assert steps > 0
        off_deadline = time.perf_counter() + off_wall_seconds
        if deadline is not None:
            off_deadline = min(off_deadline, deadline)
        _check_deadline(off_deadline)
        problem = CachedProblem(
            _CollisionOffProblem(model=model),
            exact_curvature=False,
            wall_seconds=off_deadline - time.perf_counter(),
        )
        off = receipt["collision_off"] = {
            "force_threshold": atol,
            "success": False,
            "checkpoint_is_diagnostic_raw_state_unless_force_gate_passes": True,
        }
        try:
            local_state, pncg = run_pncg_phase(
                problem, local_state, atol=atol, max_step_norm=runtime.max_step_norm_m
            )
            off["pncg"] = pncg
            if pncg["reason"] != "converged":
                sparse = SparseNewtonProblem(problem)
                local_state, newton = safeguarded_newton(
                    sparse,
                    local_state,
                    atol=atol,
                    linear_rtol=runtime.newton_linear_rtol,
                    linear_max_steps=no_contact_linear_max_steps,
                    max_steps=steps,
                    max_step_norm=runtime.max_step_norm_m,
                    preconditioner="diag",
                    shift_policy="reuse",
                    reuse_shift_force_ratio=0.0,
                    shift_scale_policy="signed_mean",
                )
                off["newton"] = newton
            force = float(torch.linalg.vector_norm(problem.grad(local_state)))
            off.update({"grad_norm": force, "success": force <= atol})
            if not off["success"]:
                raise ForwardConvergenceError(
                    "collision-off estimate failed its force gate", receipt=off
                )
        finally:
            candidate = local_state.u.detach().clone()
            _save(output_dir / "collision-off.pt", candidate, new_pose_rad_m, new_q)
            model.collision = collision
            off["seconds"] = time.perf_counter() - started
            off["retained_geometry"] = geometry_metrics(physics, candidate)
            off["contact_after_restoration"] = _contact(collision, candidate)
            write_json(output_dir / "summary.json", receipt)
        receipt["stage"] = "pushout"
        from mouthopen_signed_pushout import signed_repair

        candidate = signed_repair(
            physics,
            seed,
            candidate,
            fixed,
            new_pose_rad_m,
            new_q,
            output_dir,
            receipt,
            iterations=pushout_iterations,
            deadline=deadline,
        )
        receipt["retained_geometry"] = geometry_metrics(physics, candidate)
        receipt["contact"] = _contact(collision, candidate)
        receipt["success"] = True
        receipt["stage"] = "awaiting_caller_geometry_gate_and_collision_on_solve"
        _save(output_dir / "repaired-seed.pt", candidate, new_pose_rad_m, new_q)
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
        assert runtime.forward.state is original_state
        assert original_state.u is old_runtime_u
        assert original_state.collision is old_runtime_contact
        restored = model.get_materials()
        for name, fields in original_materials.items():
            for field, value in fields.items():
                torch.testing.assert_close(restored[name][field], value, rtol=0, atol=0)
        receipt["model_state_restored"] = True
        receipt["seconds"] = time.perf_counter() - started
        write_json(output_dir / "summary.json", receipt)


__all__ = ["prepare_collision_off_seed"]
