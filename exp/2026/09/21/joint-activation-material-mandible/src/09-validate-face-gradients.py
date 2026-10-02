"""Full-mesh directional derivatives; this does not admit oral geometry."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Literal

import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import configure_cuda
from joint_fields import SharedFieldParameters, activation_stresses_mpa
from joint_physics import BULK_NAMES, JointPhysics

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    initial_checkpoint: Path = GROUP / "data/neutral-prestress-001/best-admissible.pt"
    contact_spec: Path | None = None
    output_dir: Path = cherries.output("face-gradient-validation", mkdir=True)
    forward_rtol: float = 1e-6
    forward_atol: float = 1e-12
    adjoint_rtol: float = 1e-7
    max_forward_steps: int = 10000
    forward_method: Literal["pncg", "newton_cg"] = "pncg"
    newton_linear_rtol: float = 1e-3
    newton_max_steps: int = 12


def main(cfg: Config):  # noqa: PLR0915
    global COMPLETED  # noqa: PLW0603
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    archive_sources(output)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    configure_cuda()
    initial = torch.load(cfg.initial_checkpoint, map_location="cpu", weights_only=False)
    materials = initial["materials"]["materials"]
    skin = materials["skin"]
    physics = JointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={name: materials[name]["young_mpa"] for name in BULK_NAMES},
        bulk_nu={name: materials[name]["poisson"] for name in BULK_NAMES},
        skin_young_mpa=skin["reference_map"]["young_mpa"],
        skin_nu=skin["poisson"],
        thickness_m=skin["thickness_m"],
        rtol=cfg.forward_rtol,
        atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        max_steps=cfg.max_forward_steps,
        forward_method=cfg.forward_method,
        newton_linear_rtol=cfg.newton_linear_rtol,
        newton_max_steps=cfg.newton_max_steps,
        contact_config=(
            json.loads(cfg.contact_spec.read_text())
            if cfg.contact_spec is not None
            else None
        ),
    )
    shared = SharedFieldParameters(initial["materials"])
    with torch.no_grad():
        shared.coefficients.copy_(initial["shared_coefficients"])
    q = torch.zeros((len(physics.ids), 6))
    q[:, :3] = 0.01
    q.requires_grad_()
    jaw = torch.zeros(6, requires_grad=True)
    pose_scale = jaw.new_tensor([0.01] * 3 + [0.001] * 3)
    seed = initial["primal"]["neutral"].to(device="cuda")

    def evaluate() -> torch.Tensor:
        u = physics.solve(
            shared.bulk_stresses_mpa(),
            shared.skin_resultant_n_per_m(),
            shared.skin_stiffness_multiplier(),
            activation_stresses_mpa(q, shared.activation_reference_mpa),
            jaw * pose_scale,
            seed,
            key="gradient-check",
        )
        return physics.fit_loss(u, 0)

    value = evaluate()
    base_forward = copy.deepcopy(physics.runtime.last_forward)
    # Every finite-difference side starts independently from this same equilibrium.
    # Do not warm-chain plus -> minus, which would introduce asymmetric solve paths.
    seed = physics.runtime.forward.state.u.detach().clone()
    torch.save(seed.cpu(), output / "base-equilibrium.pt")
    value.backward()
    gradients = {
        "shared": shared.coefficients.grad.detach().clone(),
        "activation": q.grad.detach().clone(),
        "jaw": jaw.grad.detach().clone(),
    }
    targets = {"shared": shared.coefficients, "activation": q, "jaw": jaw}
    directions = []
    for index, name in enumerate(("fat", "aponeurosis", "muscle")):
        direction = torch.zeros_like(shared.coefficients)
        direction[index * 6 : index * 6 + 6] = direction.new_tensor(
            [0.4, -0.3, 0.2, 0.1, -0.2, 0.3]
        )
        directions.append((name, "shared", direction))
    for index, name in ((18, "skin_baseline"), (19, "skin_stiffness")):
        direction = torch.zeros_like(shared.coefficients)
        direction[index] = 1
        directions.append((name, "shared", direction))
    centers = physics.points_t[torch.as_tensor(physics.tets[physics.ids])].mean(dim=1)
    phase = (centers[:, 1] - centers[:, 1].mean()) / centers[:, 1].std()
    direction = (0.8 + 0.2 * torch.sin(phase))[:, None] * q.new_tensor(
        [0.5, 0.2, 0.3, 0.1, -0.15, 0.1]
    )
    directions.append(("dense_activation", "activation", direction))
    for start, name in ((0, "jaw_rotation"), (3, "jaw_translation")):
        direction = torch.zeros_like(jaw)
        direction[start : start + 3] = jaw.new_tensor([0.4, -0.3, 0.5])
        directions.append((name, "jaw", direction))
    rows = []
    for name, block, direction in directions:
        parameter = targets[block]
        original = parameter.detach().clone()
        analytic = float((gradients[block] * direction).sum())
        assert abs(analytic) > 1e-9, (name, analytic)
        for step in (0.003, 0.001):
            LOG.info("Checking full-face %s h=%.3g", name, step)
            with torch.no_grad():
                parameter.copy_(original + step * direction)
                plus = float(evaluate())
                plus_forward = copy.deepcopy(physics.runtime.last_forward)
                parameter.copy_(original - step * direction)
                minus = float(evaluate())
                minus_forward = copy.deepcopy(physics.runtime.last_forward)
                parameter.copy_(original)
            finite = (plus - minus) / (2 * step)
            relative = abs(finite - analytic) / max(abs(finite), abs(analytic))
            rows.append(
                {
                    "name": name,
                    "step": step,
                    "analytic": analytic,
                    "finite_difference": finite,
                    "relative_error": relative,
                    "plus_forward": plus_forward,
                    "minus_forward": minus_forward,
                }
            )
            write_json(output / "checks.json", rows)
            LOG.info("Full-face %s h=%.3g relative error %.6g", name, step, relative)
            assert relative < 0.02, rows[-1]
    write_json(
        output / "summary.json",
        {
            "success": True,
            "scope": "full-mesh numerical derivatives; anatomy remains unvalidated",
            "contact_enabled": cfg.contact_spec is not None,
            "finite_difference_seed_policy": "same frozen converged base equilibrium for each independent plus/minus solve",
            "base_forward": base_forward,
            "contact_spec_sha256": (
                sha256(cfg.contact_spec) if cfg.contact_spec is not None else None
            ),
            "contact_surface_map": physics.contact_definition,
            "last_forward": physics.runtime.last_forward,
            "checks": rows,
            "maximum_relative_error": max(row["relative_error"] for row in rows),
            "forward_tolerances": physics.runtime.tolerances,
            "forward_solver": {
                "method": cfg.forward_method,
                "newton_linear_rtol": cfg.newton_linear_rtol,
                "newton_max_steps": cfg.newton_max_steps,
            },
            "forward_count": physics.runtime.forward_count,
            "adjoint": physics.runtime.last_adjoint,
        },
    )
    cherries.log_output(output)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
