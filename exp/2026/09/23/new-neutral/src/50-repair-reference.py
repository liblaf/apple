# ruff: noqa: PLR0915
"""Create a clearance-repaired constitutive FEM reference without a solve."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import torch

from liblaf import cherries

HERE = Path(__file__).resolve().parent
GROUP = HERE.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
NEUTRAL = ROOT / "exp/2026/09/22/neutral-newton"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(SOLVERS), str(JOINT / "src"), str(NEUTRAL / "src")]

from joint_common import ProfileJoint, sha256, write_json  # noqa: E402
from joint_data import _collision_geometry  # noqa: E402
from joint_equilibrium import configure_cuda  # noqa: E402
from joint_rigid_eye_contact import build_eye_collision_physics  # noqa: E402
from profile_input_binding import bind_frozen_neutral_load  # noqa: E402
from reference_seed import (  # noqa: E402
    _harmonic_extension,
    _separate_contact_planes,
    repair,
)
from smile_collision import (  # noqa: E402
    audit_collision_state,
    audit_required_collision,
)


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/reference-clearance-001"
    neutral_dir: Path = JOINT / "data/frozen-neutral-004"
    eyes_dir: Path = JOINT / "data/rigid-eyes-001"
    dhat_m: float = 1.0e-4
    margin: float = 1.001
    screen_factor: float = 2.0
    screen_stiffness_mpa: float = 0.01
    max_iterations: int = 64
    stencil_max_iterations: int = 64


def _record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _tet_metrics(
    reference: np.ndarray, candidate: np.ndarray, tets: np.ndarray
) -> dict[str, Any]:
    dm = np.transpose(reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1))
    ds = np.transpose(candidate[tets[:, 1:]] - candidate[tets[:, :1]], (0, 2, 1))
    det_f = np.linalg.det(ds) / np.linalg.det(dm)
    assert np.isfinite(det_f).all()
    return {
        "detF_min": float(det_f.min()),
        "detF_p001": float(np.quantile(det_f, 0.001)),
        "detF_max": float(det_f.max()),
        "inverted_tetrahedra": int(np.count_nonzero(det_f <= 0.0)),
    }


def _strict_ipc_audit(
    physics: Any,
    displacement: np.ndarray,
    target: float,
    screen: float,
    stiffness: float,
) -> dict[str, Any]:
    """Use the bound IPC mesh at a larger radius to certify the target gap."""
    model = physics.runtime.forward.model
    collision = model.collision
    assert collision is not None
    original = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(original.barrier)(),
        screen,
        stiffness,
        collision.use_physical_barrier,
    )
    old_minimum = collision.min_distance
    old_inflation = collision.inflation_radius
    collision.min_distance = target
    collision.inflation_radius = screen
    try:
        pose = torch.zeros(
            6,
            device=model.dof_map.fixed_values.device,
            dtype=model.dof_map.fixed_values.dtype,
        )
        value = torch.as_tensor(displacement, device=pose.device, dtype=pose.dtype)
        audit = audit_collision_state(physics, value, pose)
        diagnostic = collision.diagnostics(
            collision.state_at(physics.full_skull.extend_seed(value, pose)),
            physics.full_skull.extend_seed(value, pose),
        )
    finally:
        collision.potential = original
        collision.min_distance = old_minimum
        collision.inflation_radius = old_inflation
    active = diagnostic["minimum_active_distance_m"]
    # Null means no permitted IPC pair lies within the deliberately larger
    # screen radius, which proves the requested lower bound.
    observed_lower_bound = screen if active is None else float(active)
    return {
        "method": "bound IPC collision mesh with dhat temporarily set to screen radius",
        "target_dhat_m": target,
        "screen_radius_m": screen,
        "active_minimum_distance_m": active,
        "minimum_distance_lower_bound_m": observed_lower_bound,
        "active_contact_count": diagnostic["active_contact_count"],
        "soft_rigid_intersection_free": audit["soft_rigid_intersection_free"],
        "contact_numerically_valid": diagnostic["contact_numerically_valid"],
        "meets_target": bool(
            audit["soft_rigid_intersection_free"] and observed_lower_bound >= target
        ),
    }


def _write_meshes(
    physics: Any, points: np.ndarray, output: Path
) -> dict[str, dict[str, str]]:
    volume = physics.mesh.copy(deep=True)
    volume.points = points
    skin = physics.skin.copy(deep=True)
    skin.points = points[np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)]
    volume_path = output / "repaired-reference-volume.vtu"
    skin_path = output / "repaired-reference-skin.vtp"
    volume.save(volume_path)
    skin.save(skin_path)
    return {"volume": _record(volume_path), "skin": _record(skin_path)}


def _surface(points: np.ndarray, faces: np.ndarray) -> Any:
    return repair.poly(points, faces).compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
    )


def _project_complete_obstacles(
    physics: Any, clearance: float, max_iterations: int, trace_path: Path
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Move the pure-soft surface away from every complete rigid obstacle.

    Each iteration uses closest points for every soft vertex and every soft-edge
    midpoint.  The latter catches edge--edge approaches that vertex-only
    projection can miss.  The returned geometry is still independently gated
    by IPC's exact permitted vertex-face and edge-edge distances.
    """
    geometry = physics.full_skull.geometry
    reference = np.asarray(physics.points, dtype=np.float64)
    soft_ids = np.asarray(geometry.soft_global_ids, dtype=np.int64)
    fixed = np.asarray(geometry.fixed_global_ids, dtype=np.int64)
    faces = np.asarray(geometry.soft_faces, dtype=np.int64)
    assert not np.intersect1d(soft_ids, fixed).size
    edges = np.unique(
        np.sort(
            np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)])),
            axis=0,
        ),
        axis=0,
    )
    obstacles = {
        "cranium": _surface(geometry.cranium_points_m, geometry.cranium_faces),
        "mandible": _surface(geometry.mandible_points_m, geometry.mandible_faces),
        "eyes": _surface(physics.eyes.points_m, physics.eyes.triangles),
    }
    surface = reference[soft_ids].copy()
    trace: list[dict[str, Any]] = []
    for iteration in range(max_iterations):
        correction = np.zeros_like(surface)
        measurements: dict[str, float] = {}
        for name, obstacle in obstacles.items():
            samples = np.concatenate((surface, surface[edges].mean(axis=1)))
            cell_ids, closest = obstacle.find_closest_cell(
                samples, return_closest_point=True
            )
            closest = np.asarray(closest, dtype=np.float64)
            direction = samples - closest
            distance = np.linalg.norm(direction, axis=1)
            normals = np.asarray(obstacle.cell_normals)[np.asarray(cell_ids)]
            nonzero = distance > 1e-12
            direction[nonzero] /= distance[nonzero, None]
            direction[~nonzero] = normals[~nonzero]
            # Source skull components do not provide a reliable global normal
            # sign. Their supplied geometry is already intersection-free, so
            # the closest-point ray selects the exterior side. Eyes are closed
            # and use signed distance to expel contained soft vertices.
            if name == "eyes":
                signed = repair.signed_distance(samples, obstacle)
                direction[signed < 0.0] *= -1.0
                gap = signed
            else:
                gap = distance
            amount = np.maximum(clearance - gap, 0.0)
            correction += amount[: len(surface), None] * direction[: len(surface)]
            edge_amount = amount[len(surface) :]
            edge_direction = direction[len(surface) :]
            np.add.at(
                correction, edges[:, 0], 0.5 * edge_amount[:, None] * edge_direction
            )
            np.add.at(
                correction, edges[:, 1], 0.5 * edge_amount[:, None] * edge_direction
            )
            crossed, *_ = _collision_geometry(_surface(surface, faces), obstacle)
            if len(crossed):
                _separate_contact_planes(surface, faces, crossed, obstacle, clearance)
            measurements[f"{name}_minimum_sample_gap_m"] = float(gap.min())
            measurements[f"{name}_intersections"] = len(crossed)
        correction_norm = np.linalg.norm(correction, axis=1)
        correction[correction_norm > clearance] *= (
            clearance / correction_norm[correction_norm > clearance, None]
        )
        surface += correction
        row = {
            "iteration": iteration + 1,
            "max_surface_correction_m": float(np.linalg.norm(correction, axis=1).max()),
            **measurements,
        }
        trace.append(row)
        write_json(trace_path, trace)
        if all(
            value >= clearance
            for key, value in measurements.items()
            if key.endswith("gap_m")
        ) and all(
            value == 0
            for key, value in measurements.items()
            if key.endswith("intersections")
        ):
            displacement = _harmonic_extension(
                reference, physics.tets, soft_ids, fixed, surface
            )
            return displacement, trace
    msg = "All-obstacle closest-feature projection did not converge"
    raise RuntimeError(msg)


def _project_ipc_stencils(
    physics: Any,
    displacement: np.ndarray,
    target: float,
    screen: float,
    stiffness: float,
    max_iterations: int,
    trace_path: Path,
) -> np.ndarray:
    """Project every active IPC distance constraint through its exact stencil."""
    model = physics.runtime.forward.model
    collision = model.collision
    assert collision is not None
    geometry = physics.full_skull.geometry
    fixed = np.asarray(geometry.fixed_global_ids, dtype=np.int64)
    free_soft = np.zeros(len(displacement), dtype=bool)
    free_soft[np.asarray(geometry.soft_global_ids, dtype=np.int64)] = True
    free_soft[fixed] = False
    original = collision.potential
    old_inflation = collision.inflation_radius
    collision.potential = ipctk.BarrierPotential(
        type(original.barrier)(), screen, stiffness, collision.use_physical_barrier
    )
    collision.inflation_radius = screen
    pose = torch.zeros(
        6,
        device=model.dof_map.fixed_values.device,
        dtype=model.dof_map.fixed_values.dtype,
    )
    local_to_global = collision.indices.numpy(force=True)
    trace: list[dict[str, Any]] = []
    try:
        for iteration in range(max_iterations):
            value = torch.as_tensor(displacement, device=pose.device, dtype=pose.dtype)
            full = physics.full_skull.extend_seed(value, pose)
            state = collision.state_at(full)
            positions = (collision.vertices + full[collision.indices]).numpy(force=True)
            update = np.zeros_like(displacement)
            residual: list[dict[str, Any]] = []
            for family in (
                "vv_collisions",
                "ev_collisions",
                "ee_collisions",
                "fv_collisions",
            ):
                for stencil in getattr(state.collisions, family):
                    local_ids = np.asarray(
                        stencil.vertex_ids(
                            collision.collision_mesh.edges,
                            collision.collision_mesh.faces,
                        ),
                        dtype=np.int64,
                    )
                    valid = local_ids >= 0
                    local_ids = local_ids[valid]
                    dof = np.asarray(
                        stencil.dof(
                            positions,
                            collision.collision_mesh.edges,
                            collision.collision_mesh.faces,
                        ),
                        dtype=np.float64,
                    )
                    distance_sq = float(stencil.compute_distance(dof))
                    assert distance_sq >= 0.0
                    distance = float(np.sqrt(distance_sq))
                    if distance >= target:
                        continue
                    global_ids = local_to_global[local_ids]
                    movable = (global_ids < len(displacement)) & free_soft[
                        global_ids.clip(max=len(displacement) - 1)
                    ]
                    entry = {
                        "family": family,
                        "local_vertex_ids": local_ids.tolist(),
                        "global_vertex_ids": global_ids.tolist(),
                        "distance_m": distance,
                        "movable_soft_vertices": int(movable.sum()),
                    }
                    if distance <= 1e-12 or not movable.any():
                        residual.append(entry)
                        continue
                    gradient_sq = np.asarray(
                        stencil.compute_distance_gradient(dof), dtype=np.float64
                    ).reshape(-1, 3)
                    assert len(gradient_sq) == len(local_ids)
                    gradient = gradient_sq / (2.0 * distance)
                    denominator = float(np.sum(gradient[movable] ** 2))
                    if denominator <= 0.0:
                        residual.append(entry)
                        continue
                    step = (target - distance) / denominator
                    np.add.at(update, global_ids[movable], step * gradient[movable])
            row = {
                "iteration": iteration + 1,
                "active_stencils": len(state.collisions),
                "violating_stencils": len(residual),
                "minimum_active_distance_m": (
                    float(
                        np.sqrt(
                            state.collisions.compute_minimum_distance(
                                collision.collision_mesh, positions
                            )
                        )
                    )
                    if len(state.collisions)
                    else None
                ),
                "maximum_update_m": float(np.linalg.norm(update, axis=1).max()),
                "unresolved": residual,
            }
            trace.append(row)
            write_json(trace_path, trace)
            if (
                not residual
                and row["minimum_active_distance_m"] is not None
                and row["minimum_active_distance_m"] >= target
            ):
                return displacement
            if residual:
                msg = (
                    "IPC stencil projection has an immovable or zero-distance residual"
                )
                raise RuntimeError(msg)
            displacement = displacement + update
    finally:
        collision.potential = original
        collision.inflation_radius = old_inflation
    msg = "IPC stencil projection did not reach the requested clearance"
    raise RuntimeError(msg)


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.dhat_m > 0
    assert cfg.margin > 1.0
    assert cfg.screen_factor > 1.0
    assert cfg.screen_stiffness_mpa > 0
    cfg.output_dir.mkdir(parents=True)
    configure_cuda()
    with bind_frozen_neutral_load(
        cfg.neutral_dir,
        cfg.output_dir,
        allow_pncg_curvature_clamps=True,
        allow_isfixed_boundary=True,
        unused_inverse_sha256="0334053c9c21b7b5e7a8d3e084091c68946f1c9eb76dc41c0089530ce5d24ba4",
    ) as neutral:
        physics, baseline = build_eye_collision_physics(neutral, cfg.eyes_dir)
    del baseline
    coverage = audit_required_collision(physics)
    target = cfg.dhat_m * cfg.margin
    screen = target * cfg.screen_factor
    reference = np.asarray(physics.points, dtype=np.float64).copy()
    tets = np.asarray(physics.tets, dtype=np.int64)
    fixed = np.asarray(physics.full_skull.geometry.fixed_global_ids, dtype=np.int64)
    initial = _strict_ipc_audit(
        physics, np.zeros_like(reference), target, screen, cfg.screen_stiffness_mpa
    )
    attempts: list[dict[str, Any]] = []
    candidate: np.ndarray | None = None
    final: dict[str, Any] | None = None
    # Existing projection includes every pure-soft source face and all complete
    # obstacles.  Retry only with increasing geometric clearance; no solve,
    # reference rebase, or rigid-coordinate motion enters this procedure.
    for index, scale in enumerate((1.0, 1.25, 1.5, 2.0), start=1):
        attempt = cfg.output_dir / f"attempt-{index:02d}"
        attempt.mkdir()
        displacement, _trace = _project_complete_obstacles(
            physics,
            target * scale,
            cfg.max_iterations,
            attempt / "projection-trace.json",
        )
        displacement = _project_ipc_stencils(
            physics,
            displacement,
            target,
            screen,
            cfg.screen_stiffness_mpa,
            cfg.stencil_max_iterations,
            attempt / "ipc-stencil-trace.json",
        )
        points = reference + displacement
        projected = attempt / "surface-projected.npz"
        np.savez_compressed(
            projected,
            reference_points_m=reference,
            repaired_points_m=points,
            displacement_m=displacement,
        )
        tet = _tet_metrics(reference, points, tets)
        ipc = _strict_ipc_audit(
            physics, displacement, target, screen, cfg.screen_stiffness_mpa
        )
        item = {
            "attempt": index,
            "projection_clearance_m": target * scale,
            "projection_trace": _record(attempt / "projection-trace.json"),
            "ipc_stencil_trace": _record(attempt / "ipc-stencil-trace.json"),
            "surface_projected": _record(projected),
            "tetrahedra": tet,
            "ipc": ipc,
            "maximum_displacement_mm": float(
                np.linalg.norm(displacement, axis=1).max() * 1e3
            ),
        }
        attempts.append(item)
        if tet["inverted_tetrahedra"] == 0 and ipc["meets_target"]:
            candidate, final = points, item
            break
    if candidate is None or final is None:
        receipt = {
            "schema": "reference-clearance-repair-v1",
            "success": False,
            "reason": "No geometric projection attempt satisfied both positive tetrahedra and strict IPC clearance.",
            "target_dhat_m": cfg.dhat_m,
            "target_with_margin_m": target,
            "initial_ipc": initial,
            "attempts": attempts,
        }
        write_json(cfg.output_dir / "receipt.json", receipt)
        raise RuntimeError(receipt["reason"])
    displacement = candidate - reference
    assert np.array_equal(displacement[fixed], np.zeros((len(fixed), 3)))
    npz = cfg.output_dir / "reference-clearance.npz"
    np.savez_compressed(
        npz,
        reference_points_m=reference,
        repaired_points_m=candidate,
        displacement_m=displacement,
    )
    meshes = _write_meshes(physics, candidate, cfg.output_dir)
    receipt = {
        "schema": "reference-clearance-repair-v1",
        "success": True,
        "scope": "geometric reference repair only; no forward equilibrium solve",
        "target_dhat_m": cfg.dhat_m,
        "target_with_margin_m": target,
        "rigid_coordinates_changed": False,
        "fixed_coordinate_displacement_max_m": float(np.abs(displacement[fixed]).max()),
        "coverage": coverage,
        "inputs": {
            "neutral_manifest": _record(cfg.neutral_dir / "manifest.json"),
            "eyes_manifest": _record(cfg.eyes_dir / "manifest.json"),
        },
        "initial_ipc": initial,
        "selected_attempt": final,
        "attempts": attempts,
        "archive": _record(npz),
        "meshes": meshes,
        "arrays": {
            "reference_points_m": list(reference.shape),
            "repaired_points_m": list(candidate.shape),
            "displacement_m": list(displacement.shape),
        },
        "source": _record(Path(__file__)),
    }
    write_json(cfg.output_dir / "receipt.json", receipt)
    cherries.log_output(npz)
    cherries.log_output(cfg.output_dir / "receipt.json")
    cherries.log_metrics(
        {
            "reference/initial_minimum_lower_bound_m": initial[
                "minimum_distance_lower_bound_m"
            ],
            "reference/final_minimum_lower_bound_m": final["ipc"][
                "minimum_distance_lower_bound_m"
            ],
            "reference/detF_min": final["tetrahedra"]["detF_min"],
            "reference/maximum_displacement_mm": final["maximum_displacement_mm"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
