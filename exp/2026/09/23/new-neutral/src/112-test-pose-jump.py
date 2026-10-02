# ruff: noqa: C901, E402, PLR0915, PT018
"""Test a mandible carry/relax/push initializer with a final contact equilibrium."""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT)]

from accelerated_solvers import CachedProblem, safeguarded_newton
from hybrid_first_solver import SparseNewtonProblem
from joint_common import ProfileJoint, write_json
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from mesh_step_scale import mean_rest_edge_length
from mouthopen_runtime import install_mouthopen_hybrid_runtime
from neutral_active_strain import install_active_strain
from pncg_first import run_pncg_phase
from reference_rebase import build_rebased_physics

spec = importlib.util.spec_from_file_location(
    "mouthopen_fit", GROUP / "src/100-inverse-mouthopen.py"
)
assert spec is not None and spec.loader is not None
fit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fit
spec.loader.exec_module(fit)
LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    source_run: Path = GROUP / "data/inverse-mouthopen-003"
    output_dir: Path = GROUP / "data/pose-jump-001"
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    mode: str = "proposal"
    forward_atol: float = 1e-8
    max_newton_steps: int = 3000
    off_wall_seconds: float = 600.0
    ipc_threads: int = 4
    entrypoint_source: Path | None = None


def relax_without_contact(
    physics: Any,
    materials: dict,
    fixed: torch.Tensor,
    seed: torch.Tensor,
    *,
    atol: float,
    step_cap: float,
    max_steps: int,
    wall_seconds: float,
    linear_max_steps: int = 1000,
    start_from_newton: bool = False,
    checkpoint: Any = None,
):
    assert linear_max_steps > 0
    model = physics.runtime.forward.model
    collision = model.collision
    assert collision is not None
    started = time.perf_counter()
    state = None
    problem = None
    last_accepted: torch.Tensor | None = None

    def publish(kind: str, row: dict[str, Any], current: Any) -> None:
        nonlocal last_accepted
        if checkpoint is None:
            return
        if kind != "failure":
            last_accepted = current.u.detach().clone()
        assert last_accepted is not None
        # Newton trial updates are in-place.  A budget exception can therefore
        # leave ``current`` at a rejected trial; publish only the last state
        # observed after an accepted PNCG/Newton callback.
        accepted = model.State(u=last_accepted)
        checkpoint(
            last_accepted,
            {
                "kind": kind,
                "accepted": kind != "failure",
                "seconds": time.perf_counter() - started,
                "force_norm": float(
                    torch.linalg.vector_norm(problem.delegate.grad(accepted))
                ),
                **row,
            },
        )

    try:
        model.collision = None
        model.set_materials(materials)
        model.dof_map.fixed_values = fixed.detach().clone()
        state = model.State(
            u=model.dof_map.to_full(model.dof_map.to_free(seed)).detach().clone()
        )
        last_accepted = state.u.detach().clone()
        problem = CachedProblem(ForwardProblem(model=model), wall_seconds=wall_seconds)
        if start_from_newton:
            pncg = {
                "reason": "skipped_newton_resume",
                "steps": 0,
                "trace": [],
                "windows": [],
            }
        else:
            state, pncg = run_pncg_phase(
                problem,
                state,
                atol=atol,
                max_step_norm=step_cap,
                callback=lambda row: publish(row["kind"], row, state),
            )
        sparse = SparseNewtonProblem(problem)
        state, newton = safeguarded_newton(
            sparse,
            state,
            atol=atol,
            linear_rtol=1e-3,
            linear_max_steps=linear_max_steps,
            max_steps=max_steps,
            max_step_norm=step_cap,
            preconditioner="diag",
            shift_policy="reuse",
            reuse_shift_force_ratio=0.0,
            shift_scale_policy="signed_mean",
            post_step=lambda current, step: publish("newton", {"step": step}, current),
        )
        force = float(torch.linalg.vector_norm(problem.grad(state)))
        assert force <= atol
        return state.u.detach().clone(), {
            "success": True,
            "collision_enabled": False,
            "force_norm": force,
            "force_threshold": atol,
            "seconds": time.perf_counter() - started,
            "started_from_newton": start_from_newton,
            "pncg": pncg,
            "newton": newton,
            "scope": "initializer only; not the accepted physical equilibrium",
        }
    except ForwardConvergenceError as error:
        if state is not None and problem is not None:
            publish(
                "failure",
                {"failure": str(error), "receipt": error.receipt},
                state,
            )
        raise
    finally:
        model.collision = collision


def main(cfg: Config) -> None:
    from mouthopen_pose_jump import carry_near_mandible, push_out

    output = cfg.output_dir.resolve()
    assert not output.exists(), output
    assert cfg.mode in {"proposal", "baseline", "both"}
    output.mkdir(parents=True)
    source_archive = output / "sources"
    source_archive.mkdir()
    archived_sources = [
        Path(__file__),
        GROUP / "src/mouthopen_pose_jump.py",
        GROUP / "src/mouthopen_runtime.py",
        GROUP / "src/mouthopen_seed.py",
    ]
    if cfg.entrypoint_source is not None:
        archived_sources.append(cfg.entrypoint_source)
    for path in archived_sources:
        shutil.copy2(path, source_archive / path.name)
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    source_path = cfg.source_run / "initialization.pt"
    # Own a frozen copy before any work, so an external continuation cannot alter the fixture.
    shutil.copy2(source_path, output / "fixture.pt")
    state = torch.load(output / "fixture.pt", map_location="cuda", weights_only=False)
    q, old_jaw, source_u = (
        state[name] for name in ("activation_inv", "jaw", "displacement_m")
    )
    chin = json.loads((cfg.source_run / "chin-estimate.json").read_text())
    new_jaw = torch.as_tensor([chin["angle_rad"] / fit.ANGLE_SCALE])
    physics, _ = build_rebased_physics(cfg.reference_dir, inverse=True)
    model = physics.runtime.forward.model
    baseline, _, _ = install_active_strain(model)
    active = physics.base.active_t
    values = {name: dict(fields) for name, fields in baseline.items()}
    values["muscle"]["activation_inv"] = values["muscle"]["activation_inv"].index_copy(
        0, active, q
    )
    model.set_materials(values)
    collision = model.collision
    kappa = 0.3386
    collision.potential = ipctk.BarrierPotential(
        type(collision.potential.barrier)(),
        collision.potential.dhat,
        kappa,
        collision.use_physical_barrier,
    )
    physics.contact_definition["config"]["stiffness_mpa"] = kappa
    cap = 0.5 * mean_rest_edge_length(model, physics.points)
    runtime = install_mouthopen_hybrid_runtime(
        physics,
        forward_atol=cfg.forward_atol,
        adjoint_rtol=1e-7,
        max_step_norm_m=cap,
        newton_max_steps=cfg.max_newton_steps,
        fixed_stiffness_mpa=kappa,
    )
    axis = torch.as_tensor(physics.base.arrays["mandible_frame_world"][:, 0])

    def pose(jaw: torch.Tensor):
        return torch.cat((jaw[0] * fit.ANGLE_SCALE * axis, torch.zeros_like(axis)))

    def materials(value: torch.Tensor):
        result = {name: dict(fields) for name, fields in baseline.items()}
        result["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, active, value)
        return result

    def save_stage(name: str, u: torch.Tensor, jaw: torch.Tensor):
        fit.save_torch(
            output / f"{name}.pt",
            {"activation_inv": q.cpu(), "jaw": jaw.cpu(), "displacement_m": u.cpu()},
        )
        geometry = physics.metrics(u[: len(physics.points)])
        return {"checkpoint": fit.record(output / f"{name}.pt"), "geometry": geometry}

    protocol = {
        "schema": "mouthopen-pose-jump-test-v1",
        "config": cfg.model_dump(mode="json"),
        "fixture": fit.record(output / "fixture.pt"),
        "source_protocol": fit.record(cfg.source_run / "protocol.json"),
        "old_angle_deg": float(old_jaw[0]) * 10,
        "new_angle_deg": float(new_jaw[0]) * 10,
        "dhat_m": collision.potential.dhat,
        "kappa_mpa": kappa,
        "final_contact_force_threshold": cfg.forward_atol,
        "formulation": "unchanged repaired reference and active strain; contact-off is only an initializer",
        "source_archive": {
            path.name: fit.record(path) for path in sorted(source_archive.iterdir())
        },
        "sources": {path.name: fit.record(path) for path in archived_sources},
    }
    write_json(output / "protocol.json", protocol)
    source_u = runtime.primal(values, physics.boundary(pose(old_jaw)), source_u)
    result = {"protocol": protocol, "source": save_stage("source", source_u, old_jaw)}
    try:
        if cfg.mode in {"proposal", "both"}:
            start = time.perf_counter()
            LOG.info(
                "Carry from %.4f to %.4f degrees",
                float(old_jaw[0]) * 10,
                float(new_jaw[0]) * 10,
            )
            carried, carry = carry_near_mandible(
                physics, source_u, old_jaw, new_jaw, pose
            )
            result["carry"] = {**carry, **save_stage("carried", carried, new_jaw)}
            write_json(output / "summary.json", result)
            progress_path = output / "direct-relaxed-force.jsonl"

            def save_partial(u: torch.Tensor, row: dict[str, Any]) -> None:
                row = dict(row)
                last_snapshot = getattr(save_partial, "last_snapshot", -float("inf"))
                snapshot = (
                    row["kind"] == "failure" or row["seconds"] - last_snapshot >= 10.0
                )
                if snapshot:
                    checkpoint_path = output / "direct-relaxed-partial.pt"
                    fit.save_torch(
                        checkpoint_path,
                        {
                            "activation_inv": q.cpu(),
                            "jaw": new_jaw.cpu(),
                            "displacement_m": u.cpu(),
                        },
                    )
                    row["checkpoint"] = fit.record(checkpoint_path)
                    row["geometry"] = physics.metrics(u[: len(physics.points)])
                    save_partial.last_snapshot = row["seconds"]
                with progress_path.open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")

            relaxed, relaxation = relax_without_contact(
                physics,
                values,
                physics.boundary(pose(new_jaw)),
                carried,
                atol=cfg.forward_atol,
                step_cap=cap,
                max_steps=cfg.max_newton_steps,
                wall_seconds=cfg.off_wall_seconds,
                checkpoint=save_partial,
            )
            result["no_contact_relaxation"] = {
                **relaxation,
                **save_stage("relaxed", relaxed, new_jaw),
            }
            write_json(output / "summary.json", result)
            pushed, push = push_out(physics, relaxed, new_jaw, pose, output)
            result["push_out"] = {**push, **save_stage("pushed", pushed, new_jaw)}
            write_json(output / "summary.json", result)
            final = runtime.primal(values, physics.boundary(pose(new_jaw)), pushed)
            result["proposal"] = {
                "seconds": time.perf_counter() - start,
                "forward": copy.deepcopy(runtime.last_forward),
                **save_stage("proposal-final", final, new_jaw),
            }
            write_json(output / "summary.json", result)
        if cfg.mode in {"baseline", "both"}:
            from mouthopen_seed import prepare_seed

            start = time.perf_counter()
            seed, predictor = prepare_seed(
                physics, materials, pose, q, q, old_jaw, new_jaw, source_u
            )
            final = runtime.primal(values, physics.boundary(pose(new_jaw)), seed)
            result["baseline"] = {
                "seconds": time.perf_counter() - start,
                "predictor": predictor,
                "forward": copy.deepcopy(runtime.last_forward),
                **save_stage("baseline-final", final, new_jaw),
            }
            write_json(output / "summary.json", result)
        result["success"] = True
    except Exception as error:
        result["success"] = False
        result["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "receipt": getattr(error, "receipt", None),
        }
        raise
    finally:
        write_json(output / "summary.json", result)
        cherries.log_output(output / "summary.json")
        cherries.log_output(output / "protocol.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
