# Copyright (c) 2026 liblaf
"""Independently audit a saved L2-position, normal, and smoothness endpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT)]

from face_shape_activation_objective import (  # noqa: E402
    ActivationSmoothness,
    FaceShapeActivationObjective,
    ObjectiveWeights,
    SkinShapeLoss,
)
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402
from joint_equilibrium import configure_cuda  # noqa: E402
from mouthopen_tet_policy import (  # noqa: E402
    exclude_fully_fixed_tetrahedra,
    geometry_metrics,
)
from neutral_active_strain import install_active_strain  # noqa: E402
from reference_rebase import build_rebased_physics  # noqa: E402


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/inverse-mouthopen-regularized"
    output_name: str = "independent-audit.json"


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


def main(cfg: Config) -> None:  # noqa: PLR0915
    run = cfg.run_dir.resolve()
    output = run / cfg.output_name
    assert not output.exists(), output
    protocol_path, summary_path, endpoint_path = (
        run / "protocol.json",
        run / "summary.json",
        run / "endpoint.npz",
    )
    protocol, summary = (
        json.loads(protocol_path.read_text()),
        json.loads(summary_path.read_text()),
    )
    objective = protocol["objective"]
    assert (
        objective["kind"]
        == "l2_position_plus_oriented_normal_plus_activation_smoothness"
    )
    assert set(objective["weights"]) == {"normal", "smooth"}
    assert float(objective["weights"]["normal"]) > 0
    calibration_control = bool(protocol["config"].get("calibration_control", False))
    calibration_only = bool(protocol["config"].get("calibration_only", False))
    assert float(objective["weights"]["smooth"]) > 0 or (
        calibration_only and calibration_control
    )
    source = objective["source"]
    assert set(source) == {"path", "sha256"}
    local_objective_source = GROUP / "src" / Path(source["path"]).name
    assert local_objective_source == GROUP / "src/face_shape_activation_objective.py"
    assert sha256(local_objective_source) == source["sha256"]
    assert summary["endpoint"]["sha256"] == sha256(endpoint_path)
    target_path = bound(protocol["sources"]["blendshapes"])
    neutral_endpoint = bound(protocol["sources"]["neutral_endpoint"])
    reference = bound(protocol["sources"]["reference_repair"])
    with np.load(endpoint_path, allow_pickle=False) as saved:
        u_np = np.asarray(saved["displacement_m"], dtype=np.float64)
        q_np = np.asarray(saved["activation_inv"], dtype=np.float64)
        active_source_ids = np.asarray(saved["active_cell_ids"], dtype=np.int64)
        pose_np = np.asarray(saved["pose_rad_m"], dtype=np.float64)
    assert u_np.ndim == 2
    assert u_np.shape[1] == 3
    assert np.isfinite(u_np).all()
    assert q_np.ndim == 2
    assert q_np.shape[1] == 6
    assert np.isfinite(q_np).all()
    assert pose_np.shape == (6,)
    assert np.isfinite(pose_np).all()

    configure_cuda()
    physics, _ = build_rebased_physics(reference.parent, inverse=True)
    model = physics.runtime.forward.model
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
    lip = np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)
    assert int(lip.sum()) == 3408
    assert not np.any(isfixed[lip])
    baseline, strain_receipt, _ = install_active_strain(model)
    exclusion = exclude_fully_fixed_tetrahedra(physics)
    assert exclusion["excluded_tetrahedra"] == 2249
    active_ids = physics.base.active_t
    np.testing.assert_array_equal(
        active_source_ids, physics.base.retained_active_cell_ids
    )
    assert q_np.shape == (len(active_ids), 6)
    device, dtype = (
        baseline["muscle"]["activation_inv"].device,
        baseline["muscle"]["activation_inv"].dtype,
    )
    materials = {name: dict(values) for name, values in baseline.items()}
    materials["muscle"]["activation_inv"] = baseline["muscle"][
        "activation_inv"
    ].index_copy(0, active_ids, torch.as_tensor(q_np, device=device, dtype=dtype))
    model.set_materials(materials)
    fixed = physics.boundary(torch.as_tensor(pose_np, device=device, dtype=dtype))
    model.dof_map.fixed_values = fixed.detach().clone()
    u = torch.as_tensor(u_np, device=device, dtype=dtype)
    assert u.shape == physics.full_skull.full_reference_points_m.shape
    fixed_error = float(
        np.abs(
            u.flatten()[model.dof_map.fixed_indices].cpu().numpy() - fixed.cpu().numpy()
        ).max()
    )
    assert fixed_error <= 5e-16
    free_force = model.dof_map.to_free_grad(
        model.grad(model.State(u=u.detach().clone()))
    )
    force = float(torch.linalg.vector_norm(free_force))
    metrics = geometry_metrics(physics, u)

    with np.load(target_path, allow_pickle=False) as data:
        names = [str(value) for value in data["expression_names"]]
        index = names.index(protocol["expression_name"])
        skin_ids = np.asarray(data["skin_global_ids"], dtype=np.int64)
        triangles = np.asarray(data["skin_triangles"], dtype=np.int64)
        neutral_skin = np.asarray(data["new_neutral_points_m"], dtype=np.float64)
        target_skin = np.asarray(data["target_points_m"][index], dtype=np.float64)
    with np.load(neutral_endpoint, allow_pickle=False) as data:
        neutral_u = np.asarray(data["displacement_m"], dtype=np.float64)
    np.testing.assert_array_equal(
        np.asarray(physics.points)[skin_ids] + neutral_u[skin_ids], neutral_skin
    )
    cells = np.asarray(physics.base.tets, dtype=np.int64)
    assert cells.ndim == 2
    assert cells.shape[1] == 4
    fraction = np.asarray(physics.mesh.cell_data["MuscleFraction"], dtype=np.float64)
    material_fraction = baseline["muscle"]["fraction"].detach().cpu().numpy()
    np.testing.assert_array_equal(material_fraction, fraction)
    muscle_id = np.asarray(physics.mesh.cell_data["MuscleId"], dtype=np.int64)
    skin = SkinShapeLoss(
        np.asarray(physics.points)[skin_ids],
        neutral_skin,
        target_skin,
        skin_ids,
        triangles,
        device=device,
        dtype=dtype,
    )
    smoothness = ActivationSmoothness(
        np.asarray(physics.points),
        cells,
        active_source_ids,
        fraction,
        muscle_id,
        device=device,
        dtype=dtype,
    )
    combined = FaceShapeActivationObjective(
        skin, smoothness, ObjectiveWeights(**objective["weights"])
    )
    assert objective["skin_contract"] == skin.contract()
    assert objective["smoothness_contract"] == smoothness.contract()
    computed = combined.metrics(u, torch.as_tensor(q_np, device=device, dtype=dtype))
    final = summary["final"]
    saved_terms = final["objective_components"]
    computed_keys = {
        "position": "position_normalized",
        "normal": "normal_chord2_area_mean",
        "smooth": "activation_smoothness",
        "normal_contribution": "normal_contribution",
        "smooth_contribution": "smooth_contribution",
        "total": "objective",
    }
    for key, computed_key in computed_keys.items():
        np.testing.assert_allclose(
            computed[computed_key], saved_terms[key], rtol=1e-10, atol=1e-12
        )
    np.testing.assert_allclose(
        computed["position_rms_mm"], final["fit_rms_mm"], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        force * 1e6, final["force_norm_n"], rtol=1e-8, atol=1e-12
    )
    for key, value in metrics.items():
        np.testing.assert_allclose(
            value, final["geometry"][key], rtol=1e-10, atol=1e-12
        )
    policy = protocol["inversion_policy"]
    valid = (
        force <= protocol["force_contract"]["atol"]
        and metrics["inverted_tetrahedra"] <= policy["maximum_inverted_tetrahedra"]
        and metrics["inverted_rest_volume_fraction"]
        <= policy["maximum_inverted_rest_volume_fraction"]
    )
    result = {
        "schema": "collision-off-regularized-expression-independent-audit-v1",
        "scope": "Fresh rebuilt collision-disabled forward evaluation and direct regularized objective recomputation; no inverse solve.",
        "inputs": {
            "protocol": record(protocol_path),
            "summary": record(summary_path),
            "endpoint": record(endpoint_path),
            "objective_source": record(
                GROUP / "src/face_shape_activation_objective.py"
            ),
            "blendshapes": record(target_path),
            "neutral_endpoint": record(neutral_endpoint),
            "reference": record(reference),
        },
        "objective": {
            "kind": objective["kind"],
            "weights": objective["weights"],
            "calibration_control": calibration_only and calibration_control,
            "skin_contract": objective["skin_contract"],
            "smoothness_contract": objective["smoothness_contract"],
            "computed": computed,
            "saved_final": saved_terms,
            "position_rms_mm_is_position_only": True,
        },
        "collision": {"enabled": False, "runtime_model_collision_is_none": True},
        "isfixed": {
            "runtime_dof_map_exact": True,
            "all_3408_lip_vertices_free": True,
            "endpoint_fixed_max_abs_error_m": fixed_error,
        },
        "materials": {
            "formulation": strain_receipt["formulation"],
            "active_cell_ids_match_endpoint": True,
        },
        "tetrahedron_policy": {
            "protocol": protocol["tetrahedron_policy"],
            "rebuilt_exclusion": exclusion,
        },
        "force": {
            "raw_free_force_mpa_m2": force,
            "newtons": force * 1e6,
            "threshold_mpa_m2": protocol["force_contract"]["atol"],
        },
        "geometry": metrics,
        "valid_forward": bool(valid),
    }
    write_json(output, result)
    cherries.log_output(output)
    assert valid, result


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
