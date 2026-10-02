"""Measure the frozen iteration-61 determinant tangent at several epsilons.

This is a diagnostic for the rejected joint-projection cache only.  It never
updates Adam state, runs a nonlinear equilibrium, or writes into the inverse
run directory.
"""

# ruff: noqa: E402, PLR0915
from __future__ import annotations

import copy
import json
import math
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
from joint_coupled_predictor import clone_materials
from joint_equilibrium import configure_cuda
from mesh_step_scale import mean_rest_edge_length
from mouthopen_coupled_seed import _damped_equilibrium_tangent
from mouthopen_joint_projection import determinant_directional_derivative
from mouthopen_runtime import install_mouthopen_hybrid_runtime
from mouthopen_tet_policy import exclude_fully_fixed_tetrahedra
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics


class Config(cherries.BaseConfig):
    """Inputs frozen by the rejected iteration-61 projection cache."""

    source_run: Path = GROUP / "data/inverse-mouthopen-coupled-017"
    projection_iteration: int = 61
    output_dir: Path = GROUP / "data/mouthopen-joint-probe-epsilon-001"
    wall_seconds: float = 1800.0
    ipc_threads: int = 4
    epsilons: tuple[float, ...] = (5e-5, 1e-5, 1e-6)
    ordinary_rtol: float = 1e-7
    strict_rtol: float = 1e-9


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _source_record(path: Path) -> dict[str, str]:
    """Bind a path whose source cache must remain immutable."""
    return record(path)


def _determinants(reference: np.ndarray, tets: np.ndarray, u: np.ndarray) -> np.ndarray:
    rest = reference[tets]
    deformed = rest + u[tets]
    return np.linalg.det(deformed[:, 1:] - deformed[:, :1]) / np.linalg.det(
        rest[:, 1:] - rest[:, :1]
    )


def main(cfg: Config) -> None:  # noqa: C901
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.wall_seconds > 0
    assert cfg.epsilons == tuple(sorted(cfg.epsilons, reverse=True))
    assert all(value > 0 for value in cfg.epsilons)
    assert 0 < cfg.strict_rtol < cfg.ordinary_rtol <= 1e-7
    source = cfg.source_run.resolve()
    model_dir = (
        source / "joint-projections" / f"{cfg.projection_iteration:05d}" / "model"
    )
    summary_path = model_dir / "summary.json"
    source_path = model_dir / "source-and-adam.npz"
    probe_path = model_dir / "combined-probe.npz"
    checkpoint_path = source / "checkpoint.pt"
    endpoint_path = source / "endpoint.npz"
    final_summary_path = source / "summary.json"
    progress_path = source / "progress.jsonl"
    independent_audit_path = source / "independent-audit.json"
    for path in (
        summary_path,
        source_path,
        probe_path,
        source / "protocol.json",
        checkpoint_path,
        endpoint_path,
        final_summary_path,
        progress_path,
        independent_audit_path,
    ):
        assert path.is_file(), path
    archived = json.loads(summary_path.read_text())
    assert archived["status"] == "joint_cache_failed"
    comparison = archived["combined_probe_comparison"]
    assert not comparison["passed"]
    assert archived["epsilon"] == cfg.epsilons[0]
    assert archived["comparison_absolute_tolerance"] == 1e-7
    assert archived["comparison_relative_tolerance"] == 0.005
    protocol = json.loads((source / "protocol.json").read_text())
    independent_audit = json.loads(independent_audit_path.read_text())
    assert independent_audit["valid_forward"]
    for name, current in {
        "protocol": source / "protocol.json",
        "summary": final_summary_path,
        "endpoint": endpoint_path,
    }.items():
        assert independent_audit["inputs"][name] == record(current)
    config = protocol["config"]
    assert config["projection_bounded_joint"]
    assert config["exclude_fully_fixed_tets"]
    assert config["adjoint_relative_shift"] > 0
    assert config["adjoint_rtol"] == cfg.ordinary_rtol
    assert config["internal_forward_atol"] == 1e-9
    assert config["predictor_rtol"] == cfg.ordinary_rtol
    cfg.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + cfg.wall_seconds

    def budget() -> None:
        assert time.perf_counter() < deadline, "diagnostic wall budget exhausted"

    with np.load(source_path, allow_pickle=False) as values:
        q_np = np.asarray(values["activation_inv"], dtype=np.float64)
        pose_np = np.asarray(values["pose_normalized"], dtype=np.float64)
        u_np = np.asarray(values["displacement_m"], dtype=np.float64)
        dq_np = np.asarray(values["dq"], dtype=np.float64)
        dp_np = np.asarray(values["dp"], dtype=np.float64)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == checkpoint["local_iteration"] == 60
    assert checkpoint["optimizer_step"] == 373
    assert checkpoint["optimizer_steps"] == {"q": 373, "pose": 373}
    np.testing.assert_array_equal(checkpoint["activation_inv"].numpy(), q_np)
    np.testing.assert_array_equal(checkpoint["pose_normalized"].numpy(), pose_np)
    np.testing.assert_array_equal(checkpoint["displacement_m"].numpy(), u_np)
    with np.load(endpoint_path, allow_pickle=False) as values:
        endpoint_active_ids = np.asarray(values["active_cell_ids"], dtype=np.int64)
        np.testing.assert_array_equal(values["activation_inv"], q_np)
        np.testing.assert_array_equal(values["displacement_m"], u_np)
        np.testing.assert_array_equal(
            values["pose_rad_m"], pose_np * np.array([math.pi / 18] * 3 + [0.01] * 3)
        )
    final = json.loads(final_summary_path.read_text())["final"]
    assert final["iteration"] == final["local_iteration"] == 60
    assert final["optimizer_step"] == 373
    assert final["optimizer_steps"] == {"q": 373, "pose": 373}
    last_progress = json.loads(progress_path.read_text().splitlines()[-1])
    assert last_progress == final
    with np.load(probe_path, allow_pickle=False) as values:
        selected_ids = np.asarray(values["selected_original_ids"], dtype=np.int64)
        archived_adjoint = np.asarray(values["adjoint_directional"], dtype=np.float64)
        archived_probe = np.asarray(values["probe_directional"], dtype=np.float64)
        archived_limit = np.asarray(values["comparison_limit"], dtype=np.float64)
    assert len(selected_ids) == len(archived_adjoint) == 5
    assert 18514 in selected_ids
    assert np.any(np.abs(archived_adjoint - archived_probe) > archived_limit)
    archived_gradients = {}
    for original_id in selected_ids:
        path = model_dir / f"gradient-{original_id}.npz"
        assert path.is_file(), path
        with np.load(path, allow_pickle=False) as values:
            archived_gradients[int(original_id)] = {
                "q": np.asarray(values["q_gradient"], dtype=np.float64),
                "pose": np.asarray(values["pose_gradient"], dtype=np.float64),
            }
    q_dot = np.array(
        [np.vdot(archived_gradients[int(i)]["q"], dq_np) for i in selected_ids]
    )
    pose_dot = np.array(
        [np.vdot(archived_gradients[int(i)]["pose"], dp_np) for i in selected_ids]
    )
    np.testing.assert_allclose(
        q_dot + pose_dot, archived_adjoint, rtol=1e-11, atol=1e-14
    )
    assert abs(
        q_dot[list(selected_ids).index(18514)]
        + pose_dot[list(selected_ids).index(18514)]
    ) < (
        abs(q_dot[list(selected_ids).index(18514)])
        + abs(pose_dot[list(selected_ids).index(18514)])
    )

    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    reference_record = protocol["sources"]["reference_repair"]
    reference_path = Path(reference_record["path"])
    assert record(reference_path) == reference_record
    physics, _ = build_rebased_physics(reference_path.parent, inverse=True)
    model = physics.runtime.forward.model
    baseline, _, _ = install_active_strain(model)
    exclusion = exclude_fully_fixed_tetrahedra(physics)
    baseline = model.get_materials()
    assert exclusion["excluded_tetrahedra"] > 0
    active = physics.base.active_t
    retained_active_ids = np.asarray(
        physics.base.retained_active_cell_ids, dtype=np.int64
    )
    retained_tetrahedron_ids = np.asarray(
        physics.base._mouthopen_retained_tetrahedron_ids,  # noqa: SLF001
        dtype=np.int64,
    )
    assert q_np.shape == (len(active), 6)
    np.testing.assert_array_equal(retained_active_ids, endpoint_active_ids)
    q = torch.as_tensor(q_np, device="cuda", dtype=torch.float64)
    pose = torch.as_tensor(pose_np, device="cuda", dtype=torch.float64)
    u = torch.as_tensor(u_np, device="cuda", dtype=torch.float64)
    dq = torch.as_tensor(dq_np, device="cuda", dtype=torch.float64)
    dp = torch.as_tensor(dp_np, device="cuda", dtype=torch.float64)
    scale = torch.tensor(
        [math.pi / 18] * 3 + [0.01] * 3, device="cuda", dtype=torch.float64
    )

    def physical_pose(value: torch.Tensor) -> torch.Tensor:
        assert value.shape == (6,)
        return value * scale

    def materials(value: torch.Tensor) -> dict:
        result = {name: dict(fields) for name, fields in baseline.items()}
        result["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, active, value)
        return result

    kappa = float(
        json.loads((Path(config["neutral_dir"]) / "stiffness.json").read_text())[
            "final_stiffness"
        ]
    )
    collision = model.collision
    assert collision is not None
    potential = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(), potential.dhat, kappa, collision.use_physical_barrier
    )
    physics.contact_definition["config"]["stiffness_mpa"] = kappa
    runtime = install_mouthopen_hybrid_runtime(
        physics,
        forward_atol=float(config["internal_forward_atol"]),
        adjoint_rtol=cfg.ordinary_rtol,
        adjoint_relative_shift=float(config["adjoint_relative_shift"]),
        newton_max_steps=int(config["max_newton_steps"]),
        max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
        fixed_stiffness_mpa=kappa,
    )
    old_materials = clone_materials(materials(q))
    model.set_materials(old_materials)
    model.dof_map.fixed_values = physics.boundary(physical_pose(pose)).detach().clone()
    torch.testing.assert_close(
        u.flatten()[model.dof_map.fixed_indices],
        model.dof_map.fixed_values,
        rtol=0,
        atol=0,
    )
    reference = np.asarray(physics.points, dtype=np.float64)
    tets = np.asarray(physics.base.tets, dtype=np.int64)[retained_tetrahedron_ids]
    old_j = _determinants(reference, tets, u_np)
    selected_indices = np.array(
        [
            np.flatnonzero(retained_tetrahedron_ids == value).item()
            for value in selected_ids
        ]
    )
    np.testing.assert_allclose(
        old_j[selected_indices],
        [entry["source_J"] for entry in archived["determinant_adjoint_receipts"]],
        rtol=1e-9,
        atol=1e-12,
    )

    variants: dict[str, dict] = {}
    for rtol in (cfg.ordinary_rtol, cfg.strict_rtol):
        runtime.tolerances["adjoint_rtol"] = rtol
        for sign in (1, -1):
            for epsilon in cfg.epsilons:
                budget()
                label = f"rtol-{rtol:.0e}-eps-{sign * epsilon:+.0e}"
                with torch.no_grad():
                    seed, solver_receipt = _damped_equilibrium_tangent(
                        physics,
                        old_materials,
                        materials(q + sign * epsilon * dq),
                        u,
                        physics.boundary(physical_pose(pose + sign * epsilon * dp)),
                        deadline=deadline,
                        predictor_relative_shift=float(
                            config["predictor_relative_shift"]
                        ),
                        predictor_rtol=rtol,
                        material_changed=True,
                    )
                direction = (seed.detach().cpu().numpy() - u_np) / (sign * epsilon)
                finite = determinant_directional_derivative(
                    reference, tets, u_np, direction
                )[selected_indices]
                # The archived adjoint is only comparable to the ordinary rtol run.
                difference = archived_adjoint - finite
                limit = 1e-7 + 0.005 * np.maximum(abs(archived_adjoint), abs(finite))
                variants[label] = {
                    "epsilon": sign * epsilon,
                    "predictor_rtol": rtol,
                    "solver": copy.deepcopy(solver_receipt),
                    "directional_detF": finite.tolist(),
                    "against_archived_adjoint": {
                        "difference": difference.tolist(),
                        "limit": limit.tolist(),
                        "passes_original_gate": bool(np.all(abs(difference) <= limit)),
                        "row18514": {
                            "difference": float(
                                difference[list(selected_ids).index(18514)]
                            ),
                            "limit": float(limit[list(selected_ids).index(18514)]),
                        },
                    },
                }
                variant_path = cfg.output_dir / f"{label}.npz"
                np.savez_compressed(
                    variant_path,
                    selected_original_ids=selected_ids,
                    directional_detF=finite,
                    archived_adjoint_directional=archived_adjoint,
                    difference=difference,
                    limit=limit,
                )
                variants[label]["arrays"] = record(variant_path)
                write_json(
                    cfg.output_dir / "progress.json",
                    {
                        "status": "running",
                        "completed_variants": variants,
                        "source": {
                            "run_protocol": _source_record(source / "protocol.json"),
                            "run_checkpoint": _source_record(checkpoint_path),
                            "run_endpoint": _source_record(endpoint_path),
                            "run_independent_audit": _source_record(
                                independent_audit_path
                            ),
                        },
                    },
                )

    # Central slopes demonstrate finite-epsilon bias without changing the original
    # one-sided acceptance contract.
    central = {}
    for rtol in (cfg.ordinary_rtol, cfg.strict_rtol):
        for epsilon in cfg.epsilons:
            plus = np.asarray(
                variants[f"rtol-{rtol:.0e}-eps-{epsilon:+.0e}"]["directional_detF"]
            )
            minus = np.asarray(
                variants[f"rtol-{rtol:.0e}-eps-{-epsilon:+.0e}"]["directional_detF"]
            )
            average = 0.5 * (plus + minus)
            central[f"rtol-{rtol:.0e}-eps-{epsilon:.0e}"] = {
                "central_directional_detF": average.tolist(),
                "difference_from_archived_adjoint": (
                    archived_adjoint - average
                ).tolist(),
            }

    # Recompute only the failed row under the strict shifted adjoint.  This is a
    # distinct numerical-solver diagnosis, never an optimizer gradient adoption.
    budget()
    runtime.tolerances["adjoint_rtol"] = cfg.strict_rtol
    source_q = q.detach().clone().requires_grad_()
    source_pose = pose.detach().clone().requires_grad_()
    source_u = runtime.solve(
        materials(source_q),
        physics.boundary(physical_pose(source_pose)),
        u,
        key="diagnostic18514",
    )
    torch.testing.assert_close(source_u.detach(), u, rtol=0, atol=0)
    assert runtime.last_forward["newton"]["steps"] == 0
    assert runtime.last_forward["pncg"]["steps"] == 0
    index18514 = np.flatnonzero(retained_tetrahedron_ids == 18514).item()
    tet = torch.as_tensor(tets[index18514], dtype=torch.long, device="cuda")
    rest = torch.as_tensor(reference, dtype=torch.float64, device="cuda")[tet]
    deformed = rest + source_u[tet]
    j = torch.linalg.det((deformed[1:] - deformed[0]).T) / torch.linalg.det(
        (rest[1:] - rest[0]).T
    )
    runtime.drop_warm_adjoint("diagnostic18514")
    jq, jp = torch.autograd.grad(j, (source_q, source_pose))
    strict_adjoint = float((jq * dq).sum() + (jp * dp).sum())
    strict_gradient = cfg.output_dir / "strict-gradient-18514.npz"
    np.savez_compressed(
        strict_gradient,
        q_gradient=jq.detach().cpu().numpy(),
        pose_gradient=jp.detach().cpu().numpy(),
    )
    # The diagnostic has no state handoff: restore and prove the frozen source
    # material and prescribed values before saving its receipt.
    model.set_materials(old_materials)
    source_fixed = physics.boundary(physical_pose(pose)).detach().clone()
    model.dof_map.fixed_values = source_fixed
    torch.testing.assert_close(
        model.get_materials()["muscle"]["activation_inv"],
        old_materials["muscle"]["activation_inv"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(model.dof_map.fixed_values, source_fixed, rtol=0, atol=0)
    receipt = {
        "schema": "mouthopen-joint-probe-epsilon-diagnostic-v1",
        "scope": "Frozen saved iteration-61 q, pose, u, and Adam proposal. No optimizer update, nonlinear candidate solve, or continuation output.",
        "source": {
            "run_protocol": _source_record(source / "protocol.json"),
            "run_checkpoint": _source_record(checkpoint_path),
            "run_endpoint": _source_record(endpoint_path),
            "run_summary": _source_record(final_summary_path),
            "run_progress": _source_record(progress_path),
            "run_independent_audit": _source_record(independent_audit_path),
            "projection_summary": _source_record(summary_path),
            "source_and_adam": _source_record(source_path),
            "combined_probe": _source_record(probe_path),
            "gradient_files": {
                str(i): _source_record(model_dir / f"gradient-{i}.npz")
                for i in selected_ids
            },
        },
        "original_gate": {"atol": 1e-7, "rtol": 0.005, "unchanged": True},
        "selected_original_tetrahedron_ids": selected_ids.tolist(),
        "archived_adjoint_directional": archived_adjoint.tolist(),
        "archived_one_sided_probe": archived_probe.tolist(),
        "archived_q_contribution": q_dot.tolist(),
        "archived_pose_contribution": pose_dot.tolist(),
        "tetrahedron_exclusion": exclusion,
        "variants": variants,
        "central_diagnostics_only": central,
        "strict_adjoint_row18514": {
            "directional": strict_adjoint,
            "archived_ordinary_directional": float(
                archived_adjoint[list(selected_ids).index(18514)]
            ),
            "difference": strict_adjoint
            - float(archived_adjoint[list(selected_ids).index(18514)]),
            "sparse_adjoint": copy.deepcopy(runtime.last_sparse_adjoint),
            "gradient": record(strict_gradient),
        },
        "source_state_restored_exactly": True,
        "seconds": time.perf_counter() - started,
    }
    write_json(cfg.output_dir / "receipt.json", receipt)
    shutil.copy2(Path(__file__), cfg.output_dir / Path(__file__).name)
    cherries.log_output(cfg.output_dir / "receipt.json")
    cherries.log_metrics(
        {
            "diagnostic/row18514_archived_adjoint": float(
                archived_adjoint[list(selected_ids).index(18514)]
            ),
            "diagnostic/row18514_strict_adjoint": strict_adjoint,
            "diagnostic/seconds": receipt["seconds"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
