"""Diagnose fixed-control equilibrium refinement without adopting inverse state."""

# ruff: noqa: E402, PLR0915, SLF001
from __future__ import annotations

import copy
import gc
import json
import logging
import shutil
import sys
import time
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

from joint_common import ProfileJoint, sha256, write_json
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_equilibrium import FeasibleExpressionProblem
from mesh_step_scale import mean_rest_edge_length
from mouthopen_pose_projection import determinant_ratio
from mouthopen_runtime import install_mouthopen_hybrid_runtime
from mouthopen_tet_policy import exclude_fully_fixed_tetrahedra, geometry_metrics
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)
ACCEPTANCE_ATOL = 1e-8


class Config(cherries.BaseConfig):
    run_dirs: tuple[Path, ...] = (
        GROUP / "data/inverse-mouthopen-coupled-007",
        GROUP / "data/inverse-mouthopen-coupled-006",
    )
    output_dir: Path = GROUP / "data/mouthopen-equilibrium-polish-001"
    internal_atol: float = 1e-9
    case_wall_seconds: float = 600.0
    max_newton_steps: int = 3000
    ipc_threads: int = 4


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def bound(item: dict[str, str]) -> Path:
    path = Path(item["path"]).resolve()
    assert record(path) == item, path
    return path


def run_case(cfg: Config, source: Path, output: Path) -> dict:
    """Rebuild and refine one completed, audited checkpoint at fixed controls."""
    source = source.resolve()
    source_refs = {
        name: record(source / name)
        for name in (
            "protocol.json",
            "summary.json",
            "endpoint.npz",
            "checkpoint.pt",
            "independent-audit.json",
        )
    }
    protocol = json.loads((source / "protocol.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    audit = json.loads((source / "independent-audit.json").read_text())
    assert protocol["schema"] == "new-neutral-mouthopen-rigid6-inverse-v1"
    assert summary["status"] != "running"
    assert summary["endpoint"] == source_refs["endpoint.npz"]
    for item in audit["inputs"].values():
        bound(item)
    for name in ("summary", "protocol", "endpoint"):
        suffix = ".npz" if name == "endpoint" else ".json"
        assert bound(audit["inputs"][name]) == source / (name + suffix)
    assert audit["valid_forward"]
    assert audit["force"]["converged"]
    assert audit["collision"]["feasible"]
    assert protocol["force_contract"]["atol"] == ACCEPTANCE_ATOL
    policy = protocol["inversion_policy"]
    assert policy["maximum_inverted_tetrahedra"] == 100
    assert policy["maximum_inverted_rest_volume_fraction"] == 0.0001
    assert policy["orientation_floor"] is None
    checkpoint = torch.load(
        source / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["iteration"] == summary["final"]["iteration"]
    assert checkpoint["optimizer_steps"] == summary["final"]["optimizer_steps"]
    assert len(checkpoint["moments"]) == 4
    with np.load(source / "endpoint.npz", allow_pickle=False) as saved:
        for key in ("activation_inv", "pose_rad_m", "displacement_m"):
            np.testing.assert_array_equal(saved[key], checkpoint[key].numpy())
        active_source_ids = saved["active_cell_ids"].copy()
    reference = bound(protocol["sources"]["reference_repair"])
    target_path = bound(protocol["sources"]["blendshapes"])
    neutral_path = bound(protocol["sources"]["neutral_endpoint"])
    physics, _ = build_rebased_physics(reference.parent, inverse=True)
    model = physics.runtime.forward.model
    isfixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    expected = np.r_[
        np.repeat(isfixed, 3),
        np.ones(model.dof_map.n_full - 3 * len(isfixed), dtype=bool),
    ]
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.cpu().numpy(), np.flatnonzero(expected)
    )
    np.testing.assert_array_equal(
        model.dof_map.free_indices.cpu().numpy(), np.flatnonzero(~expected)
    )
    assert not isfixed[np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)].any()
    install_active_strain(model)
    exclusion = exclude_fully_fixed_tetrahedra(physics)
    saved_exclusion = dict(protocol["tetrahedron_policy"])
    saved_exclusion.pop("neutral_free_equation_proof")
    assert saved_exclusion == exclusion
    baseline = model.get_materials()
    skin_fields = neutral_path.parent / "active-strain-fields.npz"
    with np.load(skin_fields, allow_pickle=False) as saved:
        for key, saved_key in (
            ("activation_inv", "skin_activation_inverse"),
            ("mu", "skin_mu_mpa"),
            ("thickness", "skin_thickness_m"),
        ):
            np.testing.assert_array_equal(
                baseline["skin"][key].cpu().numpy(), saved[saved_key]
            )
    np.testing.assert_array_equal(
        active_source_ids, physics.base.retained_active_cell_ids
    )
    q = checkpoint["activation_inv"].clone().to(device="cuda")
    pose = checkpoint["pose_rad_m"].clone().to(device="cuda")
    old_u = checkpoint["displacement_m"].clone().to(device="cuda")
    materials = {name: dict(fields) for name, fields in baseline.items()}
    materials["muscle"]["activation_inv"] = baseline["muscle"][
        "activation_inv"
    ].index_copy(0, physics.base.active_t, q)
    fixed = physics.boundary(pose).detach().clone()
    torch.testing.assert_close(
        old_u.flatten()[model.dof_map.fixed_indices], fixed, rtol=0, atol=5e-16
    )
    kappa = float(protocol["ipc_stiffness_mpa"])
    assert kappa == 1.3544
    collision = model.collision
    assert collision is not None
    potential = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(), potential.dhat, kappa, collision.use_physical_barrier
    )
    physics.contact_definition["config"]["stiffness_mpa"] = kappa
    runtime = install_mouthopen_hybrid_runtime(
        physics,
        forward_atol=cfg.internal_atol,
        adjoint_rtol=1e-7,
        adjoint_relative_shift=0.001,
        newton_max_steps=cfg.max_newton_steps,
        max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
        fixed_stiffness_mpa=kappa,
    )
    with np.load(target_path, allow_pickle=False) as data:
        index = list(data["expression_names"]).index("MouthOpen")
        skin_ids = data["skin_global_ids"].copy()
        triangles = data["skin_triangles"].copy()
        neutral_points = data["new_neutral_points_m"].copy()
        target_points = data["target_points_m"][index].copy()
    with np.load(neutral_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(
            physics.points[skin_ids] + data["displacement_m"][skin_ids], neutral_points
        )
    xyz = neutral_points[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    weights = np.zeros(len(skin_ids))
    np.add.at(weights, triangles.ravel(), np.repeat(area / 3, 3))
    weights /= weights.sum()
    scale2 = float(
        np.sum(weights * np.sum((target_points - neutral_points) ** 2, axis=1))
    )
    assert scale2 > 0
    retained_ids = np.asarray(physics.base._mouthopen_retained_tetrahedron_ids)
    tets = np.asarray(physics.base.tets)[retained_ids]

    def evaluate(u: torch.Tensor) -> tuple[dict, np.ndarray]:
        model.set_materials(materials)
        model.dof_map.fixed_values = fixed.detach().clone()
        state = model.State(u=u.detach().clone())
        state.collision = collision.state_at(state.u)
        problem = FeasibleExpressionProblem(model=model, collision_step_safety=0.9)
        force = float(torch.linalg.vector_norm(problem.grad(state)))
        contact = runtime._contact_gate(state)
        geometry = geometry_metrics(physics, u)
        geometry_ok = (
            geometry["inverted_tetrahedra"] <= 100
            and geometry["inverted_rest_volume_fraction"] <= 0.0001
        )
        contact_ok = (
            contact["receipt"]["contact_numerically_valid"]
            and contact["no_intersections"]
            and contact["minimum_active_gap_at_least_buffer"]
        )
        u_np = u.detach().cpu().numpy()
        det_f = determinant_ratio(physics.points, tets, u_np)
        assert int((det_f <= 0).sum()) == geometry["inverted_tetrahedra"]
        residual = physics.points[skin_ids] + u_np[skin_ids] - target_points
        squared_error = float(np.sum(weights * np.sum(residual**2, axis=1)))
        fixed_error = float(
            (u.flatten()[model.dof_map.fixed_indices] - fixed).abs().max()
        )
        return {
            "raw_free_force_mpa_m2": force,
            "force_norm_n": force * 1e6,
            "internal_force_target_met": force <= cfg.internal_atol,
            "original_force_gate_met": force <= ACCEPTANCE_ATOL,
            "contact": contact,
            "contact_gate_met": contact_ok,
            "geometry": geometry,
            "geometry_gate_met": geometry_ok,
            "fixed_max_abs_error_m": fixed_error,
            "fit_rms_mm": 1000 * float(np.sqrt(squared_error)),
            "loss": squared_error / scale2,
            "original_acceptance_gates_met": force <= ACCEPTANCE_ATOL
            and contact_ok
            and geometry_ok
            and fixed_error <= 5e-16,
        }, det_f

    initial, old_det = evaluate(old_u)
    assert initial["original_acceptance_gates_met"]
    np.testing.assert_allclose(
        initial["fit_rms_mm"], summary["final"]["fit_rms_mm"], rtol=1e-10
    )
    np.testing.assert_allclose(
        initial["force_norm_n"], summary["final"]["force_norm_n"], rtol=1e-8, atol=1e-12
    )
    output.mkdir()
    case_protocol = {
        "source_refs": source_refs,
        "source_optimizer_steps": checkpoint["optimizer_steps"],
        "optimizer_state_policy": "Source checkpoint and its moments remain unchanged; this diagnostic creates no optimizer checkpoint or iteration.",
        "skin_fields": record(skin_fields),
        "reference": record(reference),
        "target": record(target_path),
        "isfixed_exact": True,
        "skin_fields_exact": True,
        "tetrahedron_exclusion": exclusion,
        "ipc_stiffness_mpa": kappa,
        "internal_force_target": cfg.internal_atol,
        "original_acceptance_force_threshold": ACCEPTANCE_ATOL,
        "inversion_policy": policy,
        "case_wall_seconds": cfg.case_wall_seconds,
        "budget_scope": "Direct physical corrector only; independent rebuild and fresh output evaluation are outside the solve budget.",
        "initial": initial,
        "inverse_adopted": False,
    }
    write_json(output / "protocol.json", case_protocol)
    failure = None
    started = time.perf_counter()
    runtime.deadline = started + cfg.case_wall_seconds
    LOG.info(
        "Refining %s with unchanged controls, internal force target %.3g",
        source.name,
        cfg.internal_atol,
    )
    try:
        candidate = runtime.primal(materials, fixed, old_u.detach().clone())
        candidate_kind = "completed_physical_corrector"
    except ForwardConvergenceError as error:
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "receipt": error.receipt,
        }
        assert hasattr(runtime, "last_failed_displacement"), failure
        candidate = runtime.last_failed_displacement.detach().clone()
        candidate_kind = "unadmitted_failed_corrector_state"
    solve_seconds = time.perf_counter() - started
    forward = copy.deepcopy(runtime.last_forward)
    runtime.deadline = None
    final, new_det = evaluate(candidate)
    np.testing.assert_array_equal(q.cpu().numpy(), checkpoint["activation_inv"].numpy())
    np.testing.assert_array_equal(pose.cpu().numpy(), checkpoint["pose_rad_m"].numpy())
    for name, item in source_refs.items():
        assert record(source / name) == item, name
    displacement_change = (candidate - old_u).detach().cpu().numpy()
    newly_inverted = retained_ids[(old_det > 0) & (new_det <= 0)]
    recovered = retained_ids[(old_det <= 0) & (new_det > 0)]
    sign_changed = retained_ids[(old_det <= 0) != (new_det <= 0)]
    np.savez_compressed(
        output / "endpoint.npz",
        displacement_m=candidate.detach().cpu().numpy(),
        activation_inv=checkpoint["activation_inv"].numpy(),
        active_cell_ids=active_source_ids,
        pose_rad_m=checkpoint["pose_rad_m"].numpy(),
        pose_normalized=checkpoint["pose_normalized"].numpy(),
    )
    np.savez_compressed(
        output / "retained-determinants.npz",
        original_retained_tetrahedron_ids=retained_ids,
        old_det_f=old_det,
        refined_det_f=new_det,
        newly_inverted_original_ids=newly_inverted,
        recovered_original_ids=recovered,
        sign_changed_original_ids=sign_changed,
    )
    succeeded = (
        failure is None
        and final["internal_force_target_met"]
        and final["original_acceptance_gates_met"]
    )
    result = {
        "source": str(source),
        "status": "refined_valid_diagnostic" if succeeded else "refinement_failed",
        "candidate_kind": candidate_kind,
        "failure": failure,
        "solve_seconds": solve_seconds,
        "forward": forward,
        "initial": initial,
        "final": final,
        "refinement_succeeded": succeeded,
        "inverse_adopted": False,
        "inverse_converged": False,
        "candidate_admitted_as_continuation": False,
        "source_checkpoint": source_refs["checkpoint.pt"],
        "source_optimizer_steps": checkpoint["optimizer_steps"],
        "source_hashes_unchanged": True,
        "motion": {
            "maximum_vertex_displacement_m": float(
                np.linalg.norm(displacement_change, axis=1).max()
            ),
            "rms_vertex_displacement_m": float(
                np.sqrt(np.mean(np.sum(displacement_change**2, axis=1)))
            ),
            "maximum_coordinate_displacement_m": float(
                np.abs(displacement_change).max()
            ),
        },
        "newly_inverted_original_ids": newly_inverted.tolist(),
        "recovered_original_ids": recovered.tolist(),
        "sign_changed_original_ids": sign_changed.tolist(),
        "endpoint": record(output / "endpoint.npz"),
        "retained_determinants": record(output / "retained-determinants.npz"),
    }
    write_json(output / "result.json", result)
    LOG.info(
        "Refinement %s: %s; RMS %.8f mm; raw force %.4g; inverted %d",
        source.name,
        result["status"],
        final["fit_rms_mm"],
        final["raw_free_force_mpa_m2"],
        final["geometry"]["inverted_tetrahedra"],
    )
    return result


def main(cfg: Config) -> None:
    assert 0 < cfg.internal_atol < ACCEPTANCE_ATOL
    assert cfg.case_wall_seconds > 0
    assert cfg.max_newton_steps > 0
    assert cfg.run_dirs
    assert len({path.resolve() for path in cfg.run_dirs}) == len(cfg.run_dirs)
    output = cfg.output_dir.resolve()
    assert not output.exists(), output
    output.mkdir(parents=True)
    source_records = {}
    for directory, label in (
        (GROUP / "src", "new-neutral"),
        (SOLVERS, "solver-performance"),
        (JOINT, "joint"),
        (ROOT / "src/liblaf/apple", "apple"),
    ):
        copied = output / "sources" / label
        shutil.copytree(
            directory, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        for path in copied.rglob("*.py"):
            source_records[str(path.relative_to(output / "sources"))] = sha256(path)
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    write_json(
        output / "protocol.json",
        {
            "schema": "mouthopen-fixed-control-equilibrium-polish-v1",
            "config": cfg.model_dump(mode="json"),
            "source": record(Path(__file__)),
            "source_snapshots_sha256": source_records,
            "order_policy": "Try sources in order; later sources run only if the previous fixed-control refinement fails.",
            "inverse_adopted": False,
        },
    )
    results = []
    for index, source in enumerate(cfg.run_dirs):
        cherries.set_step(index)
        write_json(output / "summary.json", {"status": "running", "results": results})
        result = run_case(cfg, source, output / source.name)
        results.append(result)
        cherries.log_metrics(
            {
                "polish/fit_rms_mm": result["final"]["fit_rms_mm"],
                "polish/raw_force": result["final"]["raw_free_force_mpa_m2"],
                "polish/inverted": result["final"]["geometry"]["inverted_tetrahedra"],
                "polish/refinement_succeeded": float(result["refinement_succeeded"]),
            }
        )
        if result["refinement_succeeded"]:
            break
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        output / "summary.json",
        {
            "status": "finished_diagnostic",
            "results": results,
            "inverse_adopted": False,
            "inverse_converged": False,
            "all_attempted_refinements_failed": not any(
                row["refinement_succeeded"] for row in results
            ),
        },
    )
    cherries.log_output(output / "protocol.json")
    cherries.log_output(output / "summary.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
