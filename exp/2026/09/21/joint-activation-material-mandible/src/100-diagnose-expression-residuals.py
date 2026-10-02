"""Separate primal residual error from implicit differentiation error."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_expression_equilibrium import install_expression_runtime
from joint_frozen_neutral import FrozenNeutral, load_script
from joint_rigid_eye_contact import build_eye_collision_physics

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem
from liblaf.apple.inverse._diff_forward import _AdjointProblem

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    validation_dir: Path = GROUP / "data/expression-scale-gradient-validation-001"
    output_dir: Path = GROUP / "data/expression-residual-diagnostic-001"


def comparison(a: float, b: float) -> dict[str, float]:
    return {
        "predicted": a,
        "fd": b,
        "relative_error": abs(a - b) / max(abs(a), abs(b), 1e-12),
    }


def main(cfg: Config) -> None:  # noqa: C901, PLR0915
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    source = json.loads((cfg.validation_dir / "summary.json").read_text())
    assert len(source["solves"]) == 9, len(source["solves"])
    hashes = source["implementation_sha256"] | {
        str(Path(__file__).resolve()): sha256(Path(__file__).resolve())
    }
    for path, digest in hashes.items():
        assert sha256(Path(path)) == digest, path
    load_script("68-run-simple-skin-forward.py").configure_cuda()
    neutral = FrozenNeutral.load(GROUP / "data/frozen-neutral-004")
    physics, _ = build_eye_collision_physics(neutral, GROUP / "data/rigid-eyes-001")
    runtime = install_expression_runtime(physics)
    model = runtime.forward.model
    direction = torch.eye(3).expand(len(physics.base.active_t), -1, -1).clone()
    base_activation = 2e-6 * direction
    base_pose = torch.tensor([1e-7, -1e-7, 1e-7, 1e-7, 0.0, -1e-7])
    pose_direction = torch.tensor([0.2, -0.3, 0.1, 0.4, -0.2, 0.5])
    pose_direction /= torch.linalg.vector_norm(pose_direction)
    ids = torch.as_tensor(neutral.arrays["observation_node_ids"])
    assert len(ids.unique()) == len(ids)
    v = torch.tensor([0.31, -0.27, 0.19])
    rows = []
    report = {
        "schema": "expression-primal-residual-diagnostic-v1",
        "eligible_as_production_validation": False,
        "source_raw_validation_success": source["success"],
        "source_summary_sha256": sha256(cfg.validation_dir / "summary.json"),
        "implementation_sha256": hashes,
        "states": rows,
        "correction_convention": "H p = -J_u; corrected objective = J + p dot free residual",
    }
    write_json(cfg.output_dir / "summary.json", report)
    warm = None
    base_state = base_p = base_free = None
    for receipt in source["solves"]:
        key = receipt["key"]
        path = Path(receipt["checkpoint"]["path"])
        assert sha256(path) == receipt["checkpoint"]["sha256"]
        with np.load(path) as arrays:
            fem = torch.as_tensor(
                np.asarray(arrays["displacement_m"], dtype=np.float64)
            )
            pose = torch.as_tensor(
                np.asarray(arrays["mandible_pose"], dtype=np.float64)
            )
        activation = base_activation
        if key.startswith("activation-"):
            _, index, sign = key.split("-")
            step = source["activation_fd_steps_mpa"][int(index)]
            activation = activation + (1 if sign == "plus" else -1) * step * direction
        full = physics.full_skull.extend_seed(fem, pose)
        model.set_materials(
            physics.expression_materials(
                skin_multiplier=torch.ones(()), active_stress=activation
            )
        )
        model.dof_map.fixed_values = physics.boundary(pose)
        assert torch.allclose(
            full.flatten()[model.dof_map.fixed_indices],
            model.dof_map.fixed_values,
            atol=1e-14,
            rtol=0.0,
        )
        state = model.State(u=full)
        state.collision = model.collision.state_at(full)
        objective_gradient = torch.zeros_like(full)
        objective_gradient[ids] = 1e6 * v / len(ids)
        adjoint = _AdjointProblem(
            b=-model.dof_map.to_free_grad(objective_gradient),
            model=model,
            model_state=state,
        )
        LOG.info("Adjoint residual diagnostic %s", key)
        solution = runtime.solver.solve(
            adjoint, torch.zeros_like(adjoint.b) if warm is None else warm
        )
        assert solution.success, solution
        p = solution.params.detach().clone()
        relative = float(
            torch.linalg.vector_norm(adjoint.matvec(p) - adjoint.b)
            / torch.linalg.vector_norm(adjoint.b)
        )
        assert relative <= runtime.tolerances["adjoint_rtol"] * 1.05, relative
        warm = p
        residual = ForwardProblem(model=model).grad(state)
        raw = float(1e6 * (fem[ids] @ v).mean())
        correction = float(torch.dot(p, residual))
        rows.append(
            {
                "key": key,
                "raw_objective": raw,
                "correction": correction,
                "corrected_objective": raw + correction,
                "force_norm": float(torch.linalg.vector_norm(residual)),
                "adjoint_relative_residual": relative,
            }
        )
        write_json(cfg.output_dir / "summary.json", report)
        LOG.info("%s objective %.12g correction %.12g", key, raw, correction)
        if key == "base":
            base_state, base_p = state, p
            base_free = model.dof_map.to_free(full).detach().clone()
    by_key = {row["key"]: row for row in rows}
    checks = {}
    for parameter, steps in (
        ("activation", source["activation_fd_steps_mpa"]),
        ("pose", source["pose_fd_steps"]),
    ):
        checks[parameter] = []
        for index, step in enumerate(steps):
            plus = by_key[f"{parameter}-{index}-plus"]["corrected_objective"]
            minus = by_key[f"{parameter}-{index}-minus"]["corrected_objective"]
            checks[parameter].append(
                {
                    "step": step,
                    **comparison(
                        source["predicted_gradients"][parameter],
                        (plus - minus) / (2 * step),
                    ),
                }
            )
    report["corrected_plateau"] = {
        parameter: comparison(values[0]["fd"], values[1]["fd"])
        for parameter, values in checks.items()
    }
    report["corrected_fd"] = checks
    write_json(cfg.output_dir / "summary.json", report)
    assert base_state is not None
    assert base_p is not None
    assert base_free is not None

    def residual_and_objective(
        active: torch.Tensor, jaw: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        model.set_materials(
            physics.expression_materials(
                skin_multiplier=torch.ones(()), active_stress=active
            )
        )
        model.dof_map.fixed_values = physics.boundary(jaw)
        state = model.State(u=model.dof_map.to_full(base_free))
        state.collision = model.collision.state_at(state.u)
        return ForwardProblem(model=model).grad(state), float(
            1e6 * (state.u[ids] @ v).mean()
        )

    mechanical = {}
    for parameter, steps in (("activation", (1e-6, 1e-5)), ("pose", (1e-7, 1e-6))):
        mechanical[parameter] = []
        for step in steps:
            if parameter == "activation":
                rp, jp = residual_and_objective(
                    base_activation + step * direction, base_pose
                )
                rm, jm = residual_and_objective(
                    base_activation - step * direction, base_pose
                )
            else:
                rp, jp = residual_and_objective(
                    base_activation, base_pose + step * pose_direction
                )
                rm, jm = residual_and_objective(
                    base_activation, base_pose - step * pose_direction
                )
            estimate = (jp - jm + float(torch.dot(base_p, rp - rm))) / (2 * step)
            mechanical[parameter].append(
                {
                    "step": step,
                    **comparison(source["predicted_gradients"][parameter], estimate),
                }
            )
    report["force_parameter_fd"] = mechanical
    report["completed"] = True
    report["corrected_gradient_check_passed"] = all(
        row["relative_error"] <= 0.05 for group in checks.values() for row in group
    )
    report["corrected_plateau_check_passed"] = all(
        row["relative_error"] <= 0.05 for row in report["corrected_plateau"].values()
    )
    report["force_parameter_check_passed"] = all(
        row["relative_error"] <= 0.05 for group in mechanical.values() for row in group
    )
    write_json(cfg.output_dir / "summary.json", report)
    cherries.log_output(cfg.output_dir)
    LOG.info("Corrected FD %s; force-parameter FD %s", checks, mechanical)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
