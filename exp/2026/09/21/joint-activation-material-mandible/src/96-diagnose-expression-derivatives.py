"""Isolate exact free-coordinate Hessian products near the eye-neutral state."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_expression_equilibrium import install_expression_runtime
from joint_frozen_neutral import FrozenNeutral, load_script
from joint_rigid_eye_contact import build_eye_collision_physics

from liblaf import cherries


class Config(cherries.BaseConfig):
    neutral: Path = GROUP / "data/frozen-neutral-004"
    eyes: Path = GROUP / "data/rigid-eyes-001"
    forward: Path = GROUP / "data/eye-neutral-forward-002"
    output_dir: Path = GROUP / "data/expression-derivative-diagnostic-001"


def grad(model, state):
    value = torch.zeros_like(state.u)
    model.warp_model.grad(state.u, value)
    model.collision.grad(state.collision, state.u, value)
    return model.dof_map.to_free_grad(value)


def main(cfg: Config):
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    load_script("68-run-simple-skin-forward.py").configure_cuda()
    neutral = FrozenNeutral.load(cfg.neutral)
    physics, baseline = build_eye_collision_physics(neutral, cfg.eyes)
    install_expression_runtime(physics)
    summary = json.loads((cfg.forward / "summary.json").read_text())
    cp = Path(summary["checkpoint"]["path"])
    assert sha256(cp) == summary["checkpoint"]["sha256"]
    with np.load(cp) as a:
        fem = torch.as_tensor(np.asarray(a["displacement_m"], dtype=np.float64))
    model = physics.runtime.forward.model
    model.set_materials(baseline)
    z = torch.zeros(6)
    model.dof_map.fixed_values = physics.boundary(z)
    u = physics.full_skull.extend_seed(fem, z)
    state = model.State(u=u)
    state.collision = model.collision.state_at(u)
    xyz = torch.as_tensor(
        physics.full_skull.full_reference_points_m, dtype=torch.float64
    )
    pfull = torch.stack(
        (
            torch.sin(81 * xyz[:, 0]),
            torch.cos(73 * xyz[:, 1]),
            torch.sin(67 * xyz[:, 2]),
        ),
        1,
    )
    p = model.dof_map.to_free(pfull)
    p /= torch.linalg.vector_norm(p)
    pfull = model.dof_map.to_full_grad(p)
    hp = model.dof_map.to_free_grad(model.hess_prod(state, pfull))
    rows = []
    for h in (1e-4, 3e-5, 1e-5, 3e-6):
        sp = model.State(u=u + h * pfull)
        sm = model.State(u=u - h * pfull)
        sp.collision = model.collision.state_at(sp.u)
        sm.collision = model.collision.state_at(sm.u)
        fd = (grad(model, sp) - grad(model, sm)) / (2 * h)
        err = float(torch.linalg.vector_norm(fd - hp) / torch.linalg.vector_norm(hp))
        rows.append(
            {
                "h": h,
                "relative_error": err,
                "fd_norm": float(torch.linalg.vector_norm(fd)),
            }
        )
    out = {
        "schema": "expression-exact-hvp-diagnostic-v1",
        "success": True,
        "force": float(torch.linalg.vector_norm(grad(model, state))),
        "hvp_norm": float(torch.linalg.vector_norm(hp)),
        "rows": rows,
        "implementation_sha256": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(
                Path(__file__).with_name("joint_expression_equilibrium.py").resolve()
            ): sha256(
                Path(__file__).with_name("joint_expression_equilibrium.py").resolve()
            ),
        },
    }
    write_json(cfg.output_dir / "summary.json", out)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
