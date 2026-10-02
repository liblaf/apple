"""Independently rebuild and audit a collision-off expression endpoint."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

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
from mouthopen_tet_policy import (  # noqa: E402
    exclude_fully_fixed_tetrahedra,
    geometry_metrics,
)
from neutral_active_strain import install_active_strain  # noqa: E402
from reference_rebase import build_rebased_physics  # noqa: E402


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/inverse-mouthopen-collision-off"
    output_name: str = "independent-audit.json"
    ipc_threads: int = 4


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def bound(item: dict[str, str]) -> Path:
    path = Path(item["path"])
    candidates = [path]
    if "data" in path.parts:
        candidates.append(
            GROUP / "data" / Path(*path.parts[path.parts.index("data") + 1 :])
        )
    for candidate in candidates:
        if candidate.is_file() and sha256(candidate) == item["sha256"]:
            return candidate
    raise AssertionError(item)


def expression_target(
    path: Path, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = [str(value) for value in archive["expression_names"]]
        index = names.index(name)
        return (
            np.asarray(archive["skin_global_ids"], dtype=np.int64),
            np.asarray(archive["skin_triangles"], dtype=np.int64),
            np.asarray(archive["new_neutral_points_m"], dtype=np.float64),
            np.asarray(archive["target_points_m"][index], dtype=np.float64),
        )


def main(cfg: Config) -> None:  # noqa: PLR0915
    run = cfg.run_dir.resolve()
    output = run / cfg.output_name
    assert not output.exists(), output
    protocol_path, summary_path, endpoint_path = (
        run / "protocol.json",
        run / "summary.json",
        run / "endpoint.npz",
    )
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert protocol["schema"] == "corrected-neutral-collision-off-rigid6-inverse-v1"
    assert protocol["collision_enabled"] is False
    assert summary["endpoint"]["sha256"] == sha256(endpoint_path)
    expression_name = str(protocol["expression_name"])
    target_path = bound(protocol["sources"]["blendshapes"])
    neutral_endpoint = bound(protocol["sources"]["neutral_endpoint"])
    reference = bound(protocol["sources"]["reference_repair"])
    with np.load(endpoint_path, allow_pickle=False) as archive:
        u_np = np.asarray(archive["displacement_m"], dtype=np.float64)
        q_np = np.asarray(archive["activation_inv"], dtype=np.float64)
        active_ids_np = np.asarray(archive["active_cell_ids"], dtype=np.int64)
        pose_np = np.asarray(archive["pose_rad_m"], dtype=np.float64)
    assert pose_np.shape == (6,)
    assert np.isfinite(u_np).all()
    assert np.isfinite(q_np).all()
    assert np.isfinite(pose_np).all()

    configure_cuda()
    physics, _ = build_rebased_physics(reference.parent, inverse=True)
    model = physics.runtime.forward.model
    # This assertion happens before any endpoint state is installed, so a
    # collision-on runtime cannot silently satisfy this audit.
    model.collision = None
    assert model.collision is None
    isfixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    expected_fixed = np.concatenate(
        (
            np.repeat(isfixed, 3),
            np.ones(model.dof_map.n_full - 3 * len(isfixed), dtype=bool),
        )
    )
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.cpu().numpy(), np.flatnonzero(expected_fixed)
    )
    np.testing.assert_array_equal(
        model.dof_map.free_indices.cpu().numpy(), np.flatnonzero(~expected_fixed)
    )
    lip = np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)
    assert int(lip.sum()) == 3408
    assert not np.any(isfixed[lip])
    baseline, strain_receipt, _ = install_active_strain(model)
    exclusion = exclude_fully_fixed_tetrahedra(physics)
    assert exclusion["excluded_tetrahedra"] == 2249
    baseline = model.get_materials()
    with np.load(
        neutral_endpoint.parent / "active-strain-fields.npz", allow_pickle=False
    ) as saved:
        np.testing.assert_array_equal(
            baseline["skin"]["activation_inv"].cpu().numpy(),
            saved["skin_activation_inverse"],
        )
        np.testing.assert_array_equal(
            baseline["skin"]["mu"].cpu().numpy(), saved["skin_mu_mpa"]
        )
        np.testing.assert_array_equal(
            baseline["skin"]["thickness"].cpu().numpy(), saved["skin_thickness_m"]
        )
    active_ids = physics.base.active_t
    np.testing.assert_array_equal(
        active_ids_np, np.asarray(physics.base.retained_active_cell_ids)
    )
    assert q_np.shape == (len(active_ids), 6)
    materials = {name: dict(fields) for name, fields in baseline.items()}
    activation = baseline["muscle"]["activation_inv"]
    materials["muscle"]["activation_inv"] = activation.index_copy(
        0,
        active_ids,
        torch.as_tensor(q_np, device=activation.device, dtype=activation.dtype),
    )
    model.set_materials(materials)
    pose = torch.as_tensor(pose_np, device=activation.device, dtype=activation.dtype)
    fixed = physics.boundary(pose)
    model.dof_map.fixed_values = fixed.detach().clone()
    u = torch.as_tensor(u_np, device=activation.device, dtype=activation.dtype)
    assert u.shape == physics.full_skull.full_reference_points_m.shape
    fixed_error = float(
        np.abs(
            u.flatten()[model.dof_map.fixed_indices].cpu().numpy() - fixed.cpu().numpy()
        ).max()
    )
    assert fixed_error <= 5e-16
    state = model.State(u=u.detach().clone())
    state.collision = None
    free_force = model.dof_map.to_free_grad(model.grad(state))
    force = float(torch.linalg.vector_norm(free_force))
    threshold = float(protocol["force_contract"]["atol"])
    assert threshold <= 1e-8
    metrics = geometry_metrics(physics, u)
    policy = protocol["inversion_policy"]
    assert policy["maximum_inverted_tetrahedra"] <= 100
    assert policy["maximum_inverted_rest_volume_fraction"] <= 1e-4
    allowed = (
        metrics["inverted_tetrahedra"] <= policy["maximum_inverted_tetrahedra"]
        and metrics["inverted_rest_volume_fraction"]
        <= policy["maximum_inverted_rest_volume_fraction"]
    )

    skin_ids, triangles, neutral_skin, target_skin = expression_target(
        target_path, expression_name
    )
    with np.load(neutral_endpoint, allow_pickle=False) as archive:
        neutral_u = np.asarray(archive["displacement_m"], dtype=np.float64)
    rest = np.asarray(physics.points, dtype=np.float64)
    np.testing.assert_array_equal(rest[skin_ids] + neutral_u[skin_ids], neutral_skin)
    xyz = neutral_skin[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    weights = np.zeros(len(skin_ids), dtype=np.float64)
    np.add.at(weights, triangles.ravel(), np.repeat(area / 3.0, 3))
    weights /= weights.sum()
    fit_rms = float(
        math.sqrt(
            np.sum(
                weights
                * np.sum((rest[skin_ids] + u_np[skin_ids] - target_skin) ** 2, axis=1)
            )
        )
    )
    saved_final: dict[str, Any] = summary["final"]
    np.testing.assert_allclose(
        force * 1e6, saved_final["force_norm_n"], rtol=1e-8, atol=1e-12
    )
    np.testing.assert_allclose(
        fit_rms * 1000, saved_final["fit_rms_mm"], rtol=1e-10, atol=1e-12
    )
    for key, value in metrics.items():
        np.testing.assert_allclose(
            value, saved_final["geometry"][key], rtol=1e-10, atol=1e-12
        )
    result = {
        "schema": "collision-off-expression-independent-audit-v1",
        "scope": "Fresh rebuilt collision-disabled inverse physics and saved endpoint evaluation; no forward or inverse solve.",
        "expression_name": expression_name,
        "inputs": {
            "protocol": record(protocol_path),
            "summary": record(summary_path),
            "endpoint": record(endpoint_path),
            "blendshapes": record(target_path),
            "neutral_endpoint": record(neutral_endpoint),
            "reference": record(reference),
        },
        "collision": {
            "enabled": False,
            "runtime_model_collision_is_none": True,
            "contact_or_intersection_acceptance_gate": False,
            "diagnostics": "Collision-free and contact-valid claims are inapplicable to this collision-off audit.",
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
            "endpoint_fixed_max_abs_error_m": fixed_error,
        },
        "materials": {
            "formulation": strain_receipt["formulation"],
            "skin_prestretch_arrays_match_neutral": True,
            "active_cell_ids_match_endpoint": True,
            "active_cell_ids_semantics": "original retained tetrahedron IDs; activation rows use retained-local active_t",
            "endpoint_raw6_shape": list(q_np.shape),
        },
        "tetrahedron_policy": {
            "protocol": protocol["tetrahedron_policy"],
            "rebuilt_exclusion": exclusion,
            "rebuild_matches_protocol_stable_fields": {
                key: value
                for key, value in protocol["tetrahedron_policy"].items()
                if key != "neutral_free_equation_proof"
            }
            == exclusion,
            "neutral_free_equation_proof": protocol["tetrahedron_policy"].get(
                "neutral_free_equation_proof"
            ),
        },
        "force": {
            "raw_free_force_mpa_m2": force,
            "newtons": force * 1e6,
            "threshold_mpa_m2": threshold,
            "threshold_newtons": threshold * 1e6,
            "converged": force <= threshold,
        },
        "geometry": metrics,
        "inversion_policy": policy,
        "fit": {"weighted_skin_rms_m": fit_rms, "weighted_skin_rms_mm": fit_rms * 1000},
        "saved_final_match": {
            "force_n": True,
            "weighted_skin_rms_mm": True,
            "retained_geometry_metrics": True,
        },
    }
    result["valid_forward"] = result["force"]["converged"] and allowed
    write_json(output, result)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "audit/force_n": force * 1e6,
            "audit/detF_min": metrics["detF_min"],
            "audit/inverted": metrics["inverted_tetrahedra"],
            "audit/fit_rms_mm": fit_rms * 1000,
            "audit/valid_forward": float(result["valid_forward"]),
        }
    )
    assert result["valid_forward"], result


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
