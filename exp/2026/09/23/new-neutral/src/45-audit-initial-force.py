# ruff: noqa: SLF001
"""Evaluate saved neutral gradients by component without taking solver steps."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from liblaf import cherries
from liblaf.apple.inverse import DifferentiableForward

GROUP = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "neutral_force_setup", GROUP / "src/30-forward-active-strain.py"
)
assert spec is not None
assert spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/forward-active-strain-001"
    output_dir: Path = GROUP / "data/initial-force-audit-001"


def split_force(model: Any, state: Any) -> dict:
    registry = model.warp_model.__wrapped__.potentials
    potentials = dict(registry)
    full = {}
    try:
        for name, potential in potentials.items():
            registry.clear()
            registry[name] = potential
            gradient = torch.zeros_like(state.u)
            model.warp_model.grad(state.u, gradient)
            full[name] = gradient
    finally:
        registry.clear()
        registry.update(potentials)
    full["contact"] = torch.zeros_like(state.u)
    model.collision.grad(state.collision, state.u, full["contact"])
    full["material_total"] = sum(full[name] for name in potentials)
    full["total"] = model.grad(state)
    free = {name: model.dof_map.to_free_grad(g) for name, g in full.items()}
    difference = free["total"] - free["material_total"] - free["contact"]
    relative_error = float(
        torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(free["total"])
    )
    assert relative_error < 1e-10, relative_error
    total_squared = torch.dot(free["total"], free["total"])
    rows = {}
    for name, g in free.items():
        projected = model.dof_map.to_full_grad(g)
        nodal = 1e6 * torch.linalg.vector_norm(projected, dim=1)
        top = torch.topk(nodal, 5)
        rows[name] = {
            "free_l2_norm_n": 1e6 * float(torch.linalg.vector_norm(g)),
            "projection_on_total_fraction": float(
                torch.dot(g, free["total"]) / total_squared
            ),
            "max_free_node_force_n": float(nodal.max()),
            "top_nodes": [
                {
                    "index": int(i),
                    "force_norm_n": float(value),
                    "displacement_mm": 1000
                    * float(torch.linalg.vector_norm(state.u[i])),
                }
                for value, i in zip(top.values, top.indices, strict=True)
            ],
        }
    return {
        "components": rows,
        "decomposition_relative_error": relative_error,
        "free_dofs": model.n_free,
    }


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    cfg.output_dir.mkdir(parents=True)
    protocol = json.loads((cfg.run_dir / "protocol.json").read_text())
    summary = json.loads((cfg.run_dir / "summary.json").read_text())
    for name in ("neutral_active_strain.py", "active_strain_materials.py"):
        assert (
            runner._sha256(GROUP / "src" / name)
            == protocol["sources"]["active_strain"][name]
        )
    runner.configure_cuda()
    runner.ipctk.set_num_threads(protocol["config"]["ipc_threads"])
    with runner.bind_frozen_neutral_load(
        Path(protocol["config"]["neutral_dir"]),
        cfg.output_dir,
        allow_pncg_curvature_clamps=True,
        allow_isfixed_boundary=True,
        unused_inverse_sha256="0334053c9c21b7b5e7a8d3e084091c68946f1c9eb76dc41c0089530ce5d24ba4",
    ) as neutral:
        physics, baseline = runner.build_eye_collision_physics(
            neutral, Path(protocol["config"]["eyes_dir"])
        )
    if protocol["fixture"]["fem_reference_rebased"]:
        from reference_rebase import rebase_reference

        assert (
            runner._sha256(GROUP / "src/reference_rebase.py")
            == protocol["sources"]["active_strain"]["reference_rebase.py"]
        )
        physics, baseline, _ = rebase_reference(
            physics, baseline, Path(protocol["config"]["reference_dir"]), cfg.output_dir
        )
    model = physics.runtime.forward.model
    model.set_materials(baseline)
    runner._verify_materials(baseline)
    baseline, _, _ = runner.install_active_strain(model)
    model.set_materials(baseline)
    pose = torch.zeros(
        6,
        device=model.dof_map.fixed_values.device,
        dtype=model.dof_map.fixed_values.dtype,
    )
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    collision = model.collision
    collision.min_distance = 1e-8
    outputs = {}
    for label, path, kappa in (
        (
            "initial",
            Path(protocol["fixture"]["seed"]["path"]),
            protocol["contact_stiffness_policy"]["anchor_stiffness_mpa"],
        ),
        (
            "terminal",
            cfg.run_dir / "endpoint.npz",
            summary["result"]["terminal_stiffness_mpa"],
        ),
    ):
        with np.load(path, allow_pickle=False) as archive:
            u = torch.as_tensor(
                archive["displacement_m"], device=pose.device, dtype=pose.dtype
            )
        extended = physics.full_skull.extend_seed(u, pose)
        projected = model.dof_map.to_full(model.dof_map.to_free(extended))
        runner._set_stiffness(collision, kappa)
        state = model.State(u=projected.detach().clone())
        state.collision = collision.state_at(state.u)
        with torch.no_grad():
            outputs[label] = split_force(model, state)
        outputs[label]["kappa_mpa"] = kappa
        outputs[label]["input"] = runner._record(path)
    expected = {
        "initial": 1e6 * protocol["contact_stiffness_policy"]["tolerance_anchor_force"],
        "terminal": 1e6 * summary["result"]["terminal_force"],
    }
    for label, value in expected.items():
        actual = outputs[label]["components"]["total"]["free_l2_norm_n"]
        assert np.isclose(actual, value, rtol=1e-9, atol=1e-9), (label, actual, value)
    receipt = {
        "schema": "neutral-force-component-audit-v1",
        "scope": "Gradient evaluations only at saved states; no forward or inverse solve.",
        "force_unit": "N",
        "norm_definition": "Euclidean norm over all free force components, not resultant force; component norms are not additive.",
        "protocol": runner._record(cfg.run_dir / "protocol.json"),
        "states": outputs,
        "matched_saved_norms": True,
        "seed_contact": protocol["seed_collision"],
        "stiffness_policy": protocol["contact_stiffness_policy"],
        "stiffness_events": summary["result"]["stiffness"]["events"],
    }
    path = cfg.output_dir / "force-components.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_output(cfg.output_dir)
    metrics = {
        f"{label}/{name}_force_n": row["free_l2_norm_n"]
        for label, state in outputs.items()
        for name, row in state["components"].items()
    }
    cherries.log_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    with ExitStack() as guards:
        for name in ("forward", "step", "adjoint_solve", "receipt"):
            guards.enter_context(
                patch.object(DifferentiableForward, name, runner.forbidden_inverse)
            )
        cherries.main(main, profile=runner.benchmark.ProfilePerformance)
