"""Compare one no-contact safeguarded Newton step at two CG budgets."""
# ruff: noqa: BLE001, E402

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT)]

from accelerated_solvers import CachedProblem, safeguarded_newton_step
from hybrid_first_solver import SparseNewtonProblem
from joint_common import ProfileJoint, sha256, write_json
from joint_equilibrium import configure_cuda
from mesh_step_scale import mean_rest_edge_length
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

from liblaf.apple.forward._problem import ForwardProblem


class Config(cherries.BaseConfig):
    input_checkpoint: Path = (
        GROUP / "data/pose-rigid-diagnostic-001/continuation/accepted-partial.pt"
    )
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    output_dir: Path = GROUP / "data/no-contact-newton-probe-001"
    forward_atol: float = 1e-8
    linear_rtol: float = 1e-3
    initial_shift_ratio: float = 1.0
    max_step_fraction_of_mean_edge: float = 0.5


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.input_checkpoint.is_file()
    cfg.output_dir.mkdir(parents=True)
    configure_cuda()
    state = torch.load(cfg.input_checkpoint, map_location="cuda", weights_only=False)
    assert {"activation_inv", "pose_rad_m", "displacement_m"} <= state.keys()
    physics, _ = build_rebased_physics(cfg.reference_dir, inverse=True)
    model = physics.runtime.forward.model
    baseline, _, _ = install_active_strain(model)
    active = physics.base.active_t
    q = state["activation_inv"]
    pose = state["pose_rad_m"]
    initial_u = state["displacement_m"]
    assert q.shape == (len(active), 6)
    assert pose.shape == (6,)
    assert initial_u.shape[1:] == (3,)
    assert torch.isfinite(initial_u).all()
    materials = {name: dict(fields) for name, fields in baseline.items()}
    materials["muscle"]["activation_inv"] = baseline["muscle"][
        "activation_inv"
    ].index_copy(0, active, q)
    model.set_materials(materials)
    model.dof_map.fixed_values = physics.boundary(pose)
    model.collision = None
    max_step = cfg.max_step_fraction_of_mean_edge * mean_rest_edge_length(
        model, physics.points
    )
    endpoints = {}
    for budget in (1000, 3000):
        probe_state = model.State(u=initial_u.detach().clone())
        cached = CachedProblem(ForwardProblem(model=model), exact_curvature=True)
        sparse = SparseNewtonProblem(cached)
        before_energy = float(cached.fun(probe_state))
        before_force = float(torch.linalg.vector_norm(cached.grad(probe_state)))
        started = time.perf_counter()
        try:
            result, receipt = safeguarded_newton_step(
                sparse,
                probe_state,
                atol=cfg.forward_atol,
                linear_rtol=cfg.linear_rtol,
                linear_max_steps=budget,
                max_step_norm=max_step,
                initial_shift_ratio=cfg.initial_shift_ratio,
                shift_policy="reuse",
                reuse_shift_force_ratio=0.0,
                shift_scale_policy="signed_mean",
            )
            assert result is probe_state
            success, failure = True, None
        except Exception as error:
            receipt = getattr(error, "receipt", None)
            success, failure = (
                False,
                {"type": type(error).__name__, "message": str(error)},
            )
        endpoint = cfg.output_dir / f"endpoint-cg{budget}.pt"
        torch.save(
            {
                "activation_inv": q.cpu(),
                "pose_rad_m": pose.cpu(),
                "displacement_m": probe_state.u.detach().cpu(),
            },
            endpoint,
        )
        endpoints[str(budget)] = {
            "success": success,
            "failure": failure,
            "receipt": receipt,
            "energy_before": before_energy,
            "energy_after": float(cached.fun(probe_state)),
            "force_before": before_force,
            "force_after": float(torch.linalg.vector_norm(cached.grad(probe_state))),
            "seconds": time.perf_counter() - started,
            "cached_counts": dict(cached.counts),
            "endpoint": record(endpoint),
        }
    archive = cfg.output_dir / "sources"
    for directory, label in (
        (GROUP / "src", "new-neutral"),
        (SOLVERS, "solver-performance"),
        (JOINT, "joint"),
    ):
        shutil.copytree(
            directory,
            archive / label,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    receipt = {
        "schema": "no-contact-newton-cg-budget-probe-v1",
        "scope": "One copied-state no-contact Newton correction per CG budget; does not change continuation defaults.",
        "inputs": {
            "checkpoint": record(cfg.input_checkpoint),
            "reference": record(cfg.reference_dir / "reference-clearance.npz"),
        },
        "settings": {
            "linear_rtol": cfg.linear_rtol,
            "linear_max_steps": [1000, 3000],
            "atol": cfg.forward_atol,
            "max_step_norm_m": max_step,
            "shift_policy": "reuse",
            "initial_shift_ratio": cfg.initial_shift_ratio,
            "reuse_shift_force_ratio": 0.0,
            "shift_scale_policy": "signed_mean",
            "collision_disabled": True,
        },
        "variants": endpoints,
        "source_sha256": {
            str(path.relative_to(archive)): sha256(path)
            for path in archive.rglob("*.py")
        },
    }
    write_json(cfg.output_dir / "receipt.json", receipt)
    cherries.log_output(cfg.output_dir / "receipt.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
