"""Independently audit a saved coupled-expression endpoint with retained tet policy."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import ipctk
import numpy as np
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT)]

from joint_common import ProfileJoint, sha256, write_json  # noqa: E402
from joint_equilibrium import configure_cuda  # noqa: E402
from joint_expression_equilibrium import FeasibleExpressionProblem  # noqa: E402
from mouthopen_tet_policy import (  # noqa: E402
    exclude_fully_fixed_tetrahedra,
    geometry_metrics,
)
from neutral_active_strain import install_active_strain  # noqa: E402
from reference_rebase import build_rebased_physics  # noqa: E402


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/inverse-expression-coupled-001"
    output_name: str = "independent-audit.json"
    ipc_threads: int = 4
    binding_mirror_root: Path | None = None


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def mirror_suffix(original: Path) -> Path:
    """Return the preserved path below the experiment-root directory."""
    anchors = [index for index, part in enumerate(original.parts) if part == GROUP.name]
    assert len(anchors) == 1, original
    suffix = Path(*original.parts[anchors[0] + 1 :])
    assert suffix.parts, original
    return suffix


def bound(item: dict[str, str], mirror_root: Path | None) -> Path:
    """Resolve a saved binding directly or through an explicit local mirror."""
    original = Path(item["path"])
    candidates = [original]
    if mirror_root is not None:
        suffix = mirror_suffix(original)
        candidate = mirror_root / suffix
        assert candidate.relative_to(mirror_root) == suffix
        candidates.append(candidate)
    for path in candidates:
        if path.is_file() and sha256(path) == item["sha256"]:
            return path
    raise AssertionError(
        {"binding": item, "candidates": [str(path) for path in candidates]}
    )


def main(cfg: Config) -> None:  # noqa: PLR0915
    run = cfg.run_dir.resolve()
    mirror_root = (
        None if cfg.binding_mirror_root is None else cfg.binding_mirror_root.resolve()
    )
    output = run / cfg.output_name
    assert not output.exists(), output
    protocol = json.loads((run / "protocol.json").read_text())
    assert protocol["schema"] == "new-neutral-expression-rigid6-inverse-v1"
    expression_name = str(protocol["expression_name"])
    expression_index = int(protocol["expression_index"])
    assert expression_name
    endpoint = run / "endpoint.npz"
    summary = run / "summary.json"
    saved_summary = json.loads(summary.read_text())
    assert bound(saved_summary["endpoint"], mirror_root) == endpoint
    target_path = bound(protocol["sources"]["blendshapes"], mirror_root)
    neutral_endpoint = bound(protocol["sources"]["neutral_endpoint"], mirror_root)
    neutral_summary = bound(protocol["sources"]["neutral_summary"], mirror_root)
    reference = bound(protocol["sources"]["reference_repair"], mirror_root)
    with np.load(endpoint, allow_pickle=False) as archive:
        u_np = np.asarray(archive["displacement_m"], dtype=np.float64)
        q_np = np.asarray(archive["activation_inv"], dtype=np.float64)
        active_ids_np = np.asarray(archive["active_cell_ids"], dtype=np.int64)
        pose_np = np.asarray(archive["pose_rad_m"], dtype=np.float64)
    assert pose_np.shape == (6,)
    assert np.isfinite(pose_np).all()
    assert np.isfinite(u_np).all()
    assert np.isfinite(q_np).all()

    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    physics, _ = build_rebased_physics(reference.parent, inverse=True)
    model = physics.runtime.forward.model
    isfixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    expected_fixed = np.concatenate(
        (
            np.repeat(isfixed, 3),
            np.ones((model.dof_map.n_full - 3 * len(isfixed)), dtype=bool),
        )
    )
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.cpu().numpy(), np.flatnonzero(expected_fixed)
    )
    np.testing.assert_array_equal(
        model.dof_map.free_indices.cpu().numpy(), np.flatnonzero(~expected_fixed)
    )
    lip = np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)
    assert not np.any(isfixed[lip])
    baseline, strain_receipt, _ = install_active_strain(model)
    exclusion = exclude_fully_fixed_tetrahedra(physics)
    # Filtering reconstructs the bulk potentials and their material arrays.
    # Capture this filtered baseline before injecting the saved active strain.
    baseline = model.get_materials()
    neutral_dir = neutral_endpoint.parent
    assert neutral_summary.parent == neutral_dir
    with np.load(neutral_dir / "active-strain-fields.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            baseline["skin"]["activation_inv"].cpu().numpy(),
            saved["skin_activation_inverse"],
        )
        np.testing.assert_array_equal(
            baseline["skin"]["mu"].cpu().numpy(), saved["skin_mu_mpa"]
        )
        np.testing.assert_array_equal(
            baseline["skin"]["thickness"].cpu().numpy(),
            saved["skin_thickness_m"],
        )
    active_ids = physics.base.active_t
    active_source_ids = np.asarray(physics.base.retained_active_cell_ids)
    np.testing.assert_array_equal(active_ids_np, active_source_ids)
    assert q_np.shape == (len(active_ids), 6)
    materials = {name: dict(fields) for name, fields in baseline.items()}
    activation = baseline["muscle"]["activation_inv"]
    materials["muscle"]["activation_inv"] = activation.index_copy(
        0,
        active_ids,
        torch.as_tensor(q_np, device=activation.device, dtype=activation.dtype),
    )
    model.set_materials(materials)
    kappa = float(protocol["ipc_stiffness_mpa"])
    collision = model.collision
    assert collision is not None
    potential = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(), potential.dhat, kappa, collision.use_physical_barrier
    )
    pose = torch.as_tensor(pose_np, device=activation.device, dtype=activation.dtype)
    fixed = physics.boundary(pose)
    model.dof_map.fixed_values = fixed.detach().clone()
    u = torch.as_tensor(u_np, device=activation.device, dtype=activation.dtype)
    assert u.shape == physics.full_skull.full_reference_points_m.shape
    endpoint_fixed_error = float(
        np.abs(
            u.flatten()[model.dof_map.fixed_indices].cpu().numpy() - fixed.cpu().numpy()
        ).max()
    )
    assert endpoint_fixed_error <= 5e-16
    state = model.State(u=u.detach().clone())
    state.collision = collision.state_at(state.u)
    problem = FeasibleExpressionProblem(model=model, collision_step_safety=0.9)
    free_force = problem.grad(state)
    raw_free_force = float(torch.linalg.vector_norm(free_force))
    contact = collision.diagnostics(state.collision, state.u)
    positions = (collision.vertices + state.u[collision.indices]).numpy(force=True)
    intersects = bool(
        ipctk.has_intersections(collision.collision_mesh, positions, ipctk.LBVH())
    )
    gap = contact["minimum_active_distance_m"]
    contact_ok = (
        bool(contact["contact_numerically_valid"])
        and not intersects
        and (gap is None or gap >= collision.min_distance)
    )

    rest = np.asarray(physics.points, dtype=np.float64)
    deformed = rest + u_np[: len(rest)]
    metrics = geometry_metrics(physics, u)
    original_tets = np.asarray(physics.tets, dtype=np.int64)
    retained_ids = np.flatnonzero(~isfixed[original_tets].all(axis=1))
    retained_tets = original_tets[retained_ids]
    rest_determinants = np.linalg.det(
        rest[retained_tets[:, 1:]] - rest[retained_tets[:, :1]]
    )
    determinants = (
        np.linalg.det(deformed[retained_tets[:, 1:]] - deformed[retained_tets[:, :1]])
        / rest_determinants
    )
    inverted_local = np.flatnonzero(determinants <= 0)
    assert len(inverted_local) == metrics["inverted_tetrahedra"]
    inverted_cells = [
        {
            "original_tetrahedron_id": int(retained_ids[index]),
            "detF": float(determinants[index]),
            "rest_volume_m3": float(rest_determinants[index] / 6),
            "deformed_centroid_m": deformed[retained_tets[index]].mean(axis=0).tolist(),
        }
        for index in inverted_local
    ]

    with np.load(target_path, allow_pickle=False) as archive:
        names = list(np.asarray(archive["expression_names"], dtype=str))
        index = names.index(expression_name)
        assert index == expression_index
        skin_ids = np.asarray(archive["skin_global_ids"], dtype=np.int64)
        triangles = np.asarray(archive["skin_triangles"], dtype=np.int64)
        neutral_skin = np.asarray(archive["new_neutral_points_m"], dtype=np.float64)
        target_skin = np.asarray(archive["target_points_m"][index], dtype=np.float64)
    with np.load(neutral_endpoint, allow_pickle=False) as archive:
        neutral_u = np.asarray(archive["displacement_m"], dtype=np.float64)
    np.testing.assert_array_equal(rest[skin_ids] + neutral_u[skin_ids], neutral_skin)
    triangles_xyz = neutral_skin[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(
            triangles_xyz[:, 1] - triangles_xyz[:, 0],
            triangles_xyz[:, 2] - triangles_xyz[:, 0],
        ),
        axis=1,
    )
    weights = np.zeros(len(skin_ids), dtype=np.float64)
    np.add.at(weights, triangles.ravel(), np.repeat(area / 3, 3))
    weights /= weights.sum()
    residual = deformed[skin_ids] - target_skin
    fit_rms = float(math.sqrt(np.sum(weights * np.sum(residual**2, axis=1))))
    threshold = float(protocol["force_contract"]["atol"])
    saved_final = saved_summary["final"]
    np.testing.assert_allclose(
        raw_free_force * 1e6, saved_final["force_norm_n"], rtol=1e-8, atol=1e-12
    )
    saved_positional_rms = float(
        saved_final.get("positional_fit_rms_mm", saved_final["fit_rms_mm"])
    )
    np.testing.assert_allclose(
        1000 * fit_rms, saved_positional_rms, rtol=1e-10, atol=1e-12
    )
    objective_terms = protocol.get("objective_terms")
    if objective_terms is not None:
        assert isinstance(objective_terms, dict)
        components = saved_final["loss_components"]
        assert float(components["loss"]) == float(saved_final["loss"])
        component_rms = (
            math.sqrt(
                float(components["position_loss"]) * float(objective_terms["scale2_m2"])
            )
            * 1000
        )
        np.testing.assert_allclose(
            component_rms, saved_positional_rms, rtol=1e-10, atol=1e-12
        )
    for key, value in metrics.items():
        saved_value = saved_final["geometry"][key]
        if isinstance(value, int):
            assert value == saved_value
        else:
            np.testing.assert_allclose(value, saved_value, rtol=1e-10, atol=1e-12)
    result = {
        "schema": "expression-coupled-independent-audit-v1",
        "expression_name": expression_name,
        "scope": "Fresh rebuilt inverse physics and saved endpoint evaluation; no forward or inverse solve.",
        "inputs": {
            "protocol": record(run / "protocol.json"),
            "summary": record(summary),
            "endpoint": record(endpoint),
            "blendshapes": record(target_path),
            "neutral_endpoint": record(neutral_endpoint),
            "neutral_summary": record(neutral_summary),
            "reference": record(reference),
        },
        "source_bindings": {
            "blendshapes": protocol["sources"]["blendshapes"],
            "neutral_endpoint": protocol["sources"]["neutral_endpoint"],
            "neutral_summary": protocol["sources"]["neutral_summary"],
            "reference_repair": protocol["sources"]["reference_repair"],
        },
        "isfixed": {
            "fem_vertices": len(isfixed),
            "fixed_vertices": int(isfixed.sum()),
            "fixed_dofs": int(expected_fixed.sum()),
            "free_dofs": int((~expected_fixed).sum()),
            "runtime_dof_map_exact": True,
            "lip_vertices": int(lip.sum()),
            "fixed_lip_vertices": int(isfixed[lip].sum()),
            "all_lip_vertices_free": True,
            "endpoint_fixed_max_abs_error_m": endpoint_fixed_error,
            "endpoint_fixed_roundoff_tolerance_m": 5e-16,
        },
        "materials": {
            "formulation": strain_receipt["formulation"],
            "skin_prestretch_arrays_match_neutral": True,
            "active_cell_ids_match_endpoint": True,
            "active_cell_ids_semantics": "original retained tetrahedron IDs; active-strain rows use retained-local active_t",
            "endpoint_raw6_shape": list(q_np.shape),
            "ipc_stiffness_mpa": kappa,
        },
        "force": {
            "raw_free_force_mpa_m2": raw_free_force,
            "newtons": raw_free_force * 1e6,
            "threshold_mpa_m2": threshold,
            "threshold_newtons": threshold * 1e6,
            "converged": raw_free_force <= threshold,
        },
        "collision": {
            "contact": contact,
            "intersections": intersects,
            "minimum_gap_gate": gap is None or gap >= collision.min_distance,
            "feasible": contact_ok,
        },
        "geometry": metrics,
        "inverted_retained_cells": inverted_cells,
        "fit": {"weighted_skin_rms_m": fit_rms, "weighted_skin_rms_mm": 1000 * fit_rms},
        "objective": {
            "mode": None if objective_terms is None else objective_terms["mode"],
            "positional_metric_matches_saved": True,
            "total_loss_is_not_used_as_rms": True,
            "normal_and_smoothness": (
                "Saved-component consistency only; this physical endpoint audit does not "
                "independently recompute normal or smoothness terms."
            ),
        },
        "saved_final_match": {
            "force_n": True,
            "positional_weighted_skin_rms_mm": True,
            "retained_geometry_metrics": True,
        },
    }
    policy = protocol["inversion_policy"]
    allowed = (
        result["geometry"]["inverted_tetrahedra"]
        <= policy["maximum_inverted_tetrahedra"]
        and result["geometry"]["inverted_rest_volume_fraction"]
        <= policy["maximum_inverted_rest_volume_fraction"]
    )
    saved_exclusion = dict(protocol["tetrahedron_policy"])
    neutral_free_equation_proof = saved_exclusion.pop("neutral_free_equation_proof")
    assert saved_exclusion == exclusion
    result["tetrahedron_policy"] = {
        "protocol": protocol["tetrahedron_policy"],
        "rebuilt_exclusion": exclusion,
        "rebuild_matches_protocol_stable_fields": True,
        "neutral_free_equation_proof": neutral_free_equation_proof,
    }
    result["inversion_policy"] = policy
    result["inversion_free"] = result["geometry"]["inverted_tetrahedra"] == 0
    result["valid_forward"] = (
        result["force"]["converged"] and result["collision"]["feasible"] and allowed
    )
    write_json(output, result)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "audit/force_n": result["force"]["newtons"],
            "audit/detF_min": result["geometry"]["detF_min"],
            "audit/inverted": result["geometry"]["inverted_tetrahedra"],
            "audit/fit_rms_mm": result["fit"]["weighted_skin_rms_mm"],
            "audit/valid_forward": float(result["valid_forward"]),
        }
    )
    assert result["valid_forward"], result


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
