# ruff: noqa: C901, E402, PLR0912, PLR0915, FBT001, FBT002, FBT003, PT018
"""Joint MouthOpen fit with one hinge angle and unrestricted per-tet Raw6 strain."""

from __future__ import annotations

import copy
import json
import logging
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
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)
ANGLE_SCALE = math.pi / 18.0


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/inverse-mouthopen-003"
    neutral_dir: Path = GROUP / "data/forward-repaired-reference-005"
    blendshape_dir: Path = GROUP / "data/blendshapes-005"
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    maximum_iterations: int = 50
    learning_rate: float = 0.02
    jaw_learning_rate: float = 0.1
    forward_atol: float = 1e-8
    adjoint_rtol: float = 1e-7
    max_newton_steps: int = 3000
    max_backtracks: int = 12
    ipc_threads: int = 4
    resume: bool = False
    initialization_checkpoint: Path | None = None


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def append(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def save_npz(path: Path, **arrays) -> None:
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def save_torch(path: Path, state: dict) -> None:
    temp = path.with_suffix(".tmp.pt")
    torch.save(state, temp)
    temp.replace(path)


def main(cfg: Config) -> None:
    from mesh_step_scale import mean_rest_edge_length
    from mouthopen_runtime import install_mouthopen_hybrid_runtime

    output = cfg.output_dir.resolve()
    assert cfg.maximum_iterations >= 0
    assert cfg.resume or not output.exists(), output
    output.mkdir(parents=True, exist_ok=cfg.resume)
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    neutral_audit = json.loads((cfg.neutral_dir / "independent-audit.json").read_text())
    for name in ("endpoint.npz", "protocol.json", "summary.json"):
        assert (
            sha256(cfg.neutral_dir / name)
            == neutral_audit["run_inputs"][name]["sha256"]
        )
    transfer = json.loads((cfg.blendshape_dir / "manifest.json").read_text())
    assert transfer["transfer_success"]
    target_path = cfg.blendshape_dir / "blendshapes.npz"
    assert sha256(target_path) == transfer["artifacts"]["blendshapes.npz"]["sha256"]
    physics, _ = build_rebased_physics(cfg.reference_dir, inverse=True)
    model = physics.runtime.forward.model
    baseline, strain_receipt, _ = install_active_strain(model)
    model.set_materials(baseline)
    kappa = float(
        json.loads((cfg.neutral_dir / "stiffness.json").read_text())["final_stiffness"]
    )
    collision = model.collision
    potential = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(), potential.dhat, kappa, collision.use_physical_barrier
    )
    physics.contact_definition["config"]["stiffness_mpa"] = kappa
    runtime = install_mouthopen_hybrid_runtime(
        physics,
        forward_atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        newton_max_steps=cfg.max_newton_steps,
        max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
        fixed_stiffness_mpa=kappa,
    )
    with np.load(target_path, allow_pickle=False) as data:
        target_index = list(data["expression_names"]).index("MouthOpen")
        skin_ids = data["skin_global_ids"].copy()
        tri = data["skin_triangles"].copy()
        neutral_points = data["new_neutral_points_m"].copy()
        target_points = data["target_points_m"][target_index].copy()
    with np.load(cfg.neutral_dir / "endpoint.npz") as data:
        neutral_u = torch.as_tensor(data["displacement_m"].copy())
    np.testing.assert_array_equal(
        physics.points[skin_ids] + neutral_u.cpu().numpy()[skin_ids], neutral_points
    )
    seed = physics.full_skull.extend_seed(neutral_u, torch.zeros(6))
    ids = torch.as_tensor(skin_ids, dtype=torch.int64)
    active_ids = physics.base.active_t
    target = torch.as_tensor(target_points - physics.points[skin_ids])
    delta = torch.as_tensor(target_points - neutral_points)
    xyz = neutral_points[tri]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    weights = np.zeros(len(skin_ids))
    np.add.at(weights, tri.ravel(), np.repeat(area / 3.0, 3))
    weights /= weights.sum()
    weights_t = torch.as_tensor(weights)
    scale2 = (weights_t[:, None] * delta.square()).sum()
    assert float(scale2) > 0
    axis = torch.as_tensor(physics.base.arrays["mandible_frame_world"][:, 0])
    assert abs(float(torch.linalg.vector_norm(axis)) - 1) < 1e-12
    q = torch.zeros((len(active_ids), 6))
    jaw = torch.zeros(1)
    moments = [
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros_like(jaw),
        torch.zeros_like(jaw),
    ]
    geometry = physics.full_skull.geometry
    from chin_pose import estimate_chin_pose

    chin = estimate_chin_pose(
        neutral_points,
        target_points,
        tri,
        physics.pivot_t.cpu().numpy(),
        axis.cpu().numpy(),
        geometry.mandible_points_m,
        angle_bounds_rad=(0.0, math.radians(40.0)),
    )
    write_json(output / "chin-estimate.json", chin)
    LOG.info(
        "Chin pose estimate %.6f degrees from %d vertices, patch residual %.4f mm",
        chin["angle_deg"],
        chin["patch_vertices"],
        chin["residual_rms_m"] * 1000,
    )
    rendering = output / "rendering.npz"
    if not cfg.resume:
        save_npz(
            rendering,
            full_reference_points_m=physics.full_skull.full_reference_points_m,
            skin_global_ids=skin_ids,
            skin_triangles=tri,
            cranium_global_ids=geometry.cranium_global_ids,
            cranium_triangles=geometry.cranium_faces,
            mandible_global_ids=geometry.mandible_global_ids,
            mandible_triangles=geometry.mandible_faces,
            eye_global_ids=physics.full_skull.eye_global_ids,
            eye_triangles=physics.eyes.triangles,
        )
        source_records = {}
        source_dir = output / "sources"
        for directory, label in (
            (GROUP / "src", "new-neutral"),
            (SOLVERS, "solver-performance"),
            (JOINT, "joint"),
            (ROOT / "src/liblaf/apple", "apple"),
        ):
            shutil.copytree(
                directory,
                source_dir / label,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        for path in source_dir.rglob("*.py"):
            source_records[str(path.relative_to(source_dir))] = sha256(path)
        protocol = {
            "schema": "new-neutral-mouthopen-raw6-inverse-v1",
            "config": cfg.model_dump(mode="json"),
            "expression_name": "MouthOpen",
            "expression_index": target_index,
            "initialization": {
                "method": "chin estimate followed by contact-aware continuation at zero muscle activation",
                "chin_estimate": record(output / "chin-estimate.json"),
            },
            "sources": {
                "blendshapes": record(target_path),
                "blendshape_manifest": record(cfg.blendshape_dir / "manifest.json"),
                "neutral_endpoint": record(cfg.neutral_dir / "endpoint.npz"),
                "neutral_summary": record(cfg.neutral_dir / "summary.json"),
                "reference_repair": record(
                    cfg.reference_dir / "reference-clearance.npz"
                ),
            },
            "rendering": {
                "archive": record(rendering),
                "layout": "Full runtime coordinates; each mesh uses named global IDs and local zero-based triangles. Jaw angle absolute relative to zero neutral.",
            },
            "parameterization": {
                "activation": "B = I + symmetric(q); six independent dimensionless components per active muscle tetrahedron; no projection, clipping, positivity, rank, magnitude or smoothness penalty",
                "active_cells": len(active_ids),
                "activation_dofs": q.numel(),
                "jaw_dofs": 1,
                "jaw": "hinge rotation about existing mandible pivot and frame first axis; angle = parameter*pi/18; 0 to 40 degrees; maximum proposed update 1 degree",
                "jaw_axis": axis.cpu().tolist(),
                "jaw_pivot_m": physics.pivot_t.cpu().tolist(),
                "packing": "xx, yy, zz, xy, yz, xz",
            },
            "objective": "neutral-skin-area-weighted squared position error divided by weighted squared MouthOpen target motion; no regularization",
            "materials": strain_receipt,
            "ipc_stiffness_mpa": kappa,
            "ipc_policy": "freeze inherited terminal adaptive stiffness for a consistent differentiable objective",
            "force_contract": {
                "rtol": 1e-3,
                "atol": cfg.forward_atol,
                "effective_threshold": cfg.forward_atol,
                "relative_anchor": "original neutral; absolute gate is stricter",
            },
            "source_sha256": source_records,
            "commit_enabled": False,
        }
        write_json(output / "protocol.json", protocol)
    else:
        protocol = json.loads((output / "protocol.json").read_text())
        assert protocol["sources"]["blendshapes"] == record(target_path)
    history = []
    start_iteration = 0
    if cfg.resume:
        state = torch.load(output / "checkpoint.pt", weights_only=False)
        q, jaw, seed = (
            state[key].to(device="cuda")
            for key in ("activation_inv", "jaw", "displacement_m")
        )
        moments = [value.to(device="cuda") for value in state["moments"]]
        start_iteration = state["iteration"]
        history = [
            json.loads(line)
            for line in (output / "progress.jsonl").read_text().splitlines()
        ]
    elif cfg.initialization_checkpoint is not None:
        state = torch.load(cfg.initialization_checkpoint, weights_only=False)
        q, jaw, seed = (
            state[key].to(device="cuda")
            for key in ("activation_inv", "jaw", "displacement_m")
        )
        write_json(
            output / "initialization-parent.json", record(cfg.initialization_checkpoint)
        )

    def materials(value: torch.Tensor):
        result = {name: dict(fields) for name, fields in baseline.items()}
        result["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, active_ids, value)
        return result

    def pose(angle: torch.Tensor):
        return torch.cat((angle[0] * ANGLE_SCALE * axis, torch.zeros_like(axis)))

    def objective(u: torch.Tensor):
        return (weights_t[:, None] * (u[ids] - target).square()).sum() / scale2

    def evaluate(
        value: torch.Tensor,
        angle: torch.Tensor,
        initial: torch.Tensor,
        gradient: bool,
        source_q: torch.Tensor | None = None,
        source_jaw: torch.Tensor | None = None,
        initializing: bool = False,
    ):
        if source_q is not None:
            from mouthopen_seed import prepare_seed

            assert source_jaw is not None

            def seed_checkpoint(
                q_value: torch.Tensor,
                jaw_value: torch.Tensor,
                u_value: torch.Tensor,
                receipt: dict,
            ):
                save_torch(
                    output / "initialization.pt",
                    {
                        "activation_inv": q_value.cpu(),
                        "jaw": jaw_value.cpu(),
                        "displacement_m": u_value.cpu(),
                    },
                )
                write_json(output / "initialization-progress.json", receipt)

            initial, predictor = prepare_seed(
                physics,
                materials=materials,
                pose=pose,
                old_q=source_q,
                new_q=value,
                old_jaw=source_jaw,
                new_jaw=angle,
                seed=initial,
                linear_rtol=cfg.adjoint_rtol,
                callback=seed_checkpoint if initializing else None,
            )
            append(output / "predictors.jsonl", predictor)
        value = value.detach().requires_grad_(gradient)
        angle = angle.detach().requires_grad_(gradient)
        u = runtime.solve(
            materials(value),
            physics.boundary(pose(angle)),
            initial.detach(),
            key="MouthOpen",
        )
        loss = objective(u)
        forward = copy.deepcopy(runtime.last_forward)
        grads = torch.autograd.grad(loss, (value, angle)) if gradient else None
        return {
            "q": value.detach(),
            "jaw": angle.detach(),
            "u": u.detach(),
            "loss": float(loss),
            "grads": grads,
            "forward": forward,
            "adjoint": copy.deepcopy(runtime.last_adjoint) if gradient else None,
            "sparse_adjoint": copy.deepcopy(runtime.last_sparse_adjoint)
            if gradient
            else None,
        }

    def metric(candidate: dict, iteration: int, elapsed: float, **extra):
        u = candidate["u"][: len(physics.points)]
        geom = physics.metrics(u)
        forward = candidate["forward"]
        return {
            "iteration": iteration,
            "loss": candidate["loss"],
            "fit_rms_mm": math.sqrt(candidate["loss"] * float(scale2)) * 1000,
            "jaw_angle_deg": float(candidate["jaw"][0]) * 10,
            "force_norm_n": forward["grad_norm"] * 1e6,
            "force_threshold_n": cfg.forward_atol * 1e6,
            "forward_converged": forward["success"],
            "contact_valid": True,
            "geometry": geom,
            "valid_forward": geom["inverted_tetrahedra"] == 0 and forward["success"],
            "inverse_converged": False,
            "elapsed_seconds": elapsed,
            "activation_rms": float(candidate["q"].square().mean().sqrt()),
            "activation_max_abs": float(candidate["q"].abs().max()),
            **extra,
        }

    def checkpoint(candidate: dict, row: dict):
        save_npz(
            output / "endpoint.npz",
            displacement_m=candidate["u"].cpu().numpy(),
            activation_inv=candidate["q"].cpu().numpy(),
            active_cell_ids=active_ids.cpu().numpy(),
            jaw_angle_rad=np.asarray(float(candidate["jaw"][0]) * ANGLE_SCALE),
        )
        save_torch(
            output / "checkpoint.pt",
            {
                "iteration": row["iteration"],
                "activation_inv": candidate["q"].cpu(),
                "jaw": candidate["jaw"].cpu(),
                "displacement_m": candidate["u"].cpu(),
                "moments": [m.cpu() for m in moments],
            },
        )
        write_json(
            output / "summary.json",
            {
                "status": "running",
                "initial": history[0],
                "final": row,
                "inverse_converged": False,
                "endpoint": record(output / "endpoint.npz"),
            },
        )

    started = time.perf_counter()
    current = evaluate(q, jaw, seed, True)
    from mouthopen_gradient_check import check_joint_pullback

    check = check_joint_pullback(
        physics,
        runtime,
        materials,
        pose,
        current["q"],
        current["jaw"],
        current["u"],
        grads=current["grads"],
        objective=objective,
    )
    write_json(
        output
        / ("resume-gradient-check.json" if cfg.resume else "gradient-check.json"),
        check,
    )
    for parameter in ("q", "jaw"):
        assert min(row["relative_error"] for row in check[parameter].values()) < 1e-3, (
            check[parameter]
        )
    if not cfg.resume:
        write_json(
            output / "neutral-baseline.json",
            metric(current, 0, time.perf_counter() - started),
        )
        current = evaluate(
            q,
            torch.as_tensor([chin["angle_rad"] / ANGLE_SCALE]),
            current["u"],
            True,
            source_q=q,
            source_jaw=jaw,
            initializing=True,
        )
        initialized_check = check_joint_pullback(
            physics,
            runtime,
            materials,
            pose,
            current["q"],
            current["jaw"],
            current["u"],
            grads=current["grads"],
            objective=objective,
        )
        write_json(output / "chin-initialized-gradient-check.json", initialized_check)
        for parameter in ("q", "jaw"):
            assert (
                min(
                    row["relative_error"]
                    for row in initialized_check[parameter].values()
                )
                < 1e-3
            ), initialized_check[parameter]
    gq0, gj0 = current["grads"]
    assert torch.isfinite(gq0).all() and torch.count_nonzero(gq0) > 0
    assert torch.isfinite(gj0).all()
    if not cfg.resume:
        row = metric(current, 0, time.perf_counter() - started)
        history.append(row)
        append(output / "progress.jsonl", row)
        checkpoint(current, row)
        write_json(output / "initial-adjoint.json", current["adjoint"])
    status = "iteration_limit"
    for iteration in range(start_iteration + 1, cfg.maximum_iterations + 1):
        gq, gj = current["grads"]
        mq, vq, mj, vj = moments
        proposed_moments = [
            0.9 * mq + 0.1 * gq,
            0.999 * vq + 0.001 * gq.square(),
            0.9 * mj + 0.1 * gj,
            0.999 * vj + 0.001 * gj.square(),
        ]
        mq, vq, mj, vj = proposed_moments
        dq = (
            -cfg.learning_rate
            * (mq / (1 - 0.9**iteration))
            / ((vq / (1 - 0.999**iteration)).sqrt() + 1e-12)
        )
        dj = (
            -cfg.jaw_learning_rate
            * (mj / (1 - 0.9**iteration))
            / ((vj / (1 - 0.999**iteration)).sqrt() + 1e-12)
        )
        dj = torch.clamp(current["jaw"] + dj.clamp(-0.1, 0.1), 0, 4) - current["jaw"]
        directional = float((gq * dq).sum() + (gj * dj).sum())
        if directional >= 0:
            dq = -cfg.learning_rate * gq / (gq.abs() + 1e-12)
            dj = (
                torch.clamp(current["jaw"] - cfg.jaw_learning_rate * gj.sign(), 0, 4)
                - current["jaw"]
            )
            directional = float((gq * dq).sum() + (gj * dj).sum())
        assert directional < 0
        accepted = None
        for trial in range(cfg.max_backtracks + 1):
            alpha = 0.5**trial
            LOG.info(
                "Iteration %d trial %d: loss %.7g, jaw %.4f deg, alpha %.6g",
                iteration,
                trial,
                current["loss"],
                float(current["jaw"][0]) * 10,
                alpha,
            )
            try:
                candidate = evaluate(
                    current["q"] + alpha * dq,
                    current["jaw"] + alpha * dj,
                    current["u"],
                    False,
                    source_q=current["q"],
                    source_jaw=current["jaw"],
                )
            except ForwardConvergenceError as error:
                append(
                    output / "trials.jsonl",
                    {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "success": False,
                        "failure": str(error),
                        "receipt": runtime.last_forward,
                    },
                )
                continue
            passed = candidate["loss"] <= current["loss"] + 1e-4 * alpha * directional
            append(
                output / "trials.jsonl",
                {
                    "iteration": iteration,
                    "trial": trial,
                    "alpha": alpha,
                    "success": True,
                    "accepted": passed,
                    "loss": candidate["loss"],
                    "forward": candidate["forward"],
                },
            )
            if passed:
                accepted = evaluate(
                    candidate["q"], candidate["jaw"], candidate["u"], True
                )
                break
        if accepted is None:
            status = "line_search_stalled"
            break
        current = accepted
        moments = proposed_moments
        row = metric(
            current,
            iteration,
            time.perf_counter() - started,
            alpha=alpha,
            adjoint_relative_residual=current["adjoint"]["relative_residual"],
        )
        history.append(row)
        append(output / "progress.jsonl", row)
        checkpoint(current, row)
        cherries.set_step(iteration)
        cherries.log_metrics(
            {
                key: row[key]
                for key in (
                    "loss",
                    "fit_rms_mm",
                    "jaw_angle_deg",
                    "force_norm_n",
                    "activation_rms",
                )
            }
        )
        LOG.info(
            "Accepted %d: fit %.6f mm, jaw %.5f deg, force %.6g N, inverted %d",
            iteration,
            row["fit_rms_mm"],
            row["jaw_angle_deg"],
            row["force_norm_n"],
            row["geometry"]["inverted_tetrahedra"],
        )
    summary = json.loads((output / "summary.json").read_text())
    summary["status"] = status
    summary["inverse_converged"] = False
    write_json(output / "summary.json", summary)
    cherries.log_output(output / "summary.json")
    cherries.log_output(output / "progress.jsonl")
    cherries.log_output(output / "endpoint.npz")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
