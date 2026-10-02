# ruff: noqa: C901, E402, PLR0912, PLR0915, PT018
"""Recompute neutral once from a contact-repaired reference with Newton-CG."""

from __future__ import annotations

import faulthandler
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import torch

from liblaf import cherries
from liblaf.apple.forward.hessian import HessianBackend, HessianProblem

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
SOLVERS = GROUP.parent / "solver-performance"
sys.path[:0] = [str(JOINT / "src"), str(SOLVERS / "src")]

from accelerated_solvers import CachedProblem, safeguarded_newton_step
from joint_common import ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_equilibrium import FeasibleExpressionProblem
from joint_physics import moduli
from joint_rigid_eye_contact import build_eye_collision_physics
from mesh_step_scale import mean_rest_edge_length
from model_binding import install_exact_bulk_diagonal, load_current_binding
from smile_collision import audit_collision_state, audit_required_collision

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/forward-001"
    resume_dir: Path | None = None
    seed_dir: Path = GROUP / "data/reference-seed-002"
    neutral_dir: Path = JOINT / "data/frozen-neutral-004"
    eyes_dir: Path = JOINT / "data/rigid-eyes-001"
    atol: float = 1e-8
    rtol: float = 1e-3
    max_steps: int = 100
    ipc_threads: int = 4
    checkpoint_every: int = 10
    hessian_backend: HessianBackend = "gpu_contact"


def record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def verify_materials(baseline: dict) -> dict:
    result = {}
    for name, young in (("fat", 0.0112), ("muscle", 0.012), ("aponeurosis", 1.693)):
        mu, la = moduli(young, 0.49)
        values = baseline[name]
        assert torch.allclose(values["mu"], torch.full_like(values["mu"], mu))
        assert torch.allclose(values["lmbda"], torch.full_like(values["lmbda"], la))
        assert not bool(torch.count_nonzero(values["active_stress"]))
        result[name] = {"young_kpa": 1000 * young, "nu": 0.49, "added_stress_mpa": 0}
    skin = baseline["skin"]
    young = skin["mu"] * (2 * 1.49)
    la_expected = young * 0.49 / (1.49 * 0.02) + skin["mu"]
    assert torch.allclose(skin["lmbda"], la_expected)
    assert torch.allclose(skin["thickness"], torch.full_like(skin["thickness"], 0.001))
    stress = skin["baseline_stress"]
    assert bool(torch.all(stress[:, 0, 1] == 0) & torch.all(stress[:, 1, 0] == 0))
    assert torch.equal(stress[:, 0, 0], stress[:, 1, 1])
    result["skin"] = {
        "young_kpa_range": [float(young.min()) * 1000, float(young.max()) * 1000],
        "nu": 0.49,
        "thickness_m": 0.001,
        "baseline_resultant_n_per_m_range": [
            float(stress[:, 0, 0].min()) * 1e6,
            float(stress[:, 0, 0].max()) * 1e6,
        ],
    }
    result["lambda_convention"] = (
        "lambda_code = E*nu/((1+nu)*(1-2*nu)) + mu; mu = E/(2*(1+nu))"
    )
    result["bulk_energy"] = (
        "mu/2*(trace(F.T@F)-3)-mu*(J-1)+lambda_code/2*(J-1)^2+0.5*Q:(F.T@F-I)"
    )
    result["skin_energy"] = (
        "exact plane-stress Stable Neo-Hookean membrane plus prescribed isotropic baseline stress"
    )
    return result


def save_checkpoint(path: Path, displacement: torch.Tensor) -> dict:
    np.savez_compressed(path, displacement_m=displacement.detach().cpu().numpy())
    return record(path)


def verify_exact_diagonal(problem: Any, state: Any) -> dict:
    """Check assembled diagonal entries against physical coordinate HVPs."""
    diagonal = problem.hess_diag(state)
    indices = sorted(
        {
            int(diagonal.argmin()),
            int(diagonal.argmax()),
            *np.linspace(0, len(diagonal) - 1, 6, dtype=int).tolist(),
        }
    )
    samples = []
    for index in indices:
        coordinate = torch.zeros_like(diagonal)
        coordinate[index] = 1.0
        product = problem.hess_prod(state, coordinate)
        error = float((diagonal[index] - product[index]).abs())
        limit = 1e-12 + 1e-9 * abs(float(product[index]))
        assert error <= limit, (index, error, limit)
        samples.append(
            {
                "free_coordinate": index,
                "diagonal": float(diagonal[index]),
                "coordinate_hvp": float(product[index]),
                "absolute_error": error,
            }
        )
    return {
        "success": True,
        "samples": samples,
        "scope": "selected assembled diagonal entries versus exact physical Hessian-vector products",
    }


def main(cfg: Config) -> None:
    assert cfg.atol == 1e-8 and cfg.rtol == 1e-3 and cfg.max_steps > 0
    assert cfg.ipc_threads > 0 and cfg.checkpoint_every > 0
    parent = None
    parent_protocol = None
    parent_trace = []
    start_iteration = 0
    previous_seconds = 0.0
    if cfg.resume_dir is not None:
        parent = json.loads((cfg.resume_dir / "summary.json").read_text())
        parent_protocol = json.loads((cfg.resume_dir / "protocol.json").read_text())
        assert sha256(cfg.resume_dir / "protocol.json") == parent["protocol"]["sha256"]
        assert sha256(cfg.resume_dir / "terminal.npz") == parent["terminal"]["sha256"]
        parent_trace = [
            json.loads(line)
            for line in (cfg.resume_dir / "trace.jsonl").read_text().splitlines()
        ]
        start_iteration = parent["accepted_steps"]
        assert 0 < start_iteration < cfg.max_steps
        assert [row["iteration"] for row in parent_trace] == list(
            range(start_iteration + 1)
        )
        assert len(parent_trace) == start_iteration + 1
        assert parent_trace[-1]["grad_norm"] == parent["final_grad_norm"]
        for key in ("atol", "rtol", "ipc_threads"):
            assert cfg.model_dump()[key] == parent_protocol["config"][key]
        previous_seconds = parent["forward_seconds"]
        # The driver may gain resume handling; all numerical implementations stay exact.
        parent_sources = json.loads((cfg.resume_dir / "provenance.json").read_text())[
            "sources"
        ]
        source_roots = {
            "experiment": JOINT / "src",
            "apple": ROOT / "src/liblaf/apple",
            "tensor-reference": ROOT / "exp/2026/09/07/tensor-active-stress/src",
            "solver-performance": SOLVERS / "src",
        }
        for name, digest in parent_sources.items():
            family, relative = name.split("/", 1)
            archived = cfg.resume_dir / "sources" / name
            assert sha256(archived) == digest, archived
            if family in source_roots:
                assert sha256(source_roots[family] / relative) == digest, name
            elif relative == "model_binding.py":
                assert sha256(GROUP / "src" / relative) == digest, relative
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = archive_sources(cfg.output_dir)
    for directory, name in (
        (GROUP / "src", "neutral-newton"),
        (SOLVERS / "src", "solver-performance"),
    ):
        shutil.copytree(
            directory,
            cfg.output_dir / "sources" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    provenance["sources"] = {
        str(path.relative_to(cfg.output_dir / "sources")): sha256(path)
        for path in sorted((cfg.output_dir / "sources").rglob("*.py"))
    }
    write_json(cfg.output_dir / "provenance.json", provenance)
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    neutral = load_current_binding(cfg.neutral_dir, cfg.output_dir)
    diagonal_policy = install_exact_bulk_diagonal()
    physics, baseline = build_eye_collision_physics(neutral, cfg.eyes_dir)
    materials = verify_materials(baseline)
    model = physics.runtime.forward.model
    model.set_materials(baseline)
    # The mandible's one rotation coordinate is prescribed to zero for neutral.
    pose = torch.zeros(6)
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    model.collision.narrow_phase_ccd = ipctk.TightInclusionCCD(
        tolerance=1e-10, max_iterations=100000
    )
    coverage = audit_required_collision(physics)
    if parent is None:
        seed_path = cfg.seed_dir / "seed.npz"
        seed_receipt_path = cfg.seed_dir / "summary.json"
        seed_receipt = json.loads(seed_receipt_path.read_text())
        assert seed_receipt["success"]
        assert seed_receipt["prior_equilibrium_used"] is False
        assert seed_receipt["fem_reference_rebased"] is False
        assert seed_receipt["seed"]["sha256"] == sha256(seed_path)
    else:
        seed_path = cfg.resume_dir / "terminal.npz"
        seed_receipt_path = cfg.resume_dir / "summary.json"
    with np.load(seed_path, allow_pickle=False) as source:
        seed_np = source["displacement_m"].copy()
    assert seed_np.shape == physics.points.shape
    assert np.isfinite(seed_np).all()
    seed = torch.as_tensor(seed_np)
    initial_collision = audit_collision_state(physics, seed, pose)
    assert initial_collision["state_feasible"], initial_collision
    full_seed = physics.full_skull.extend_seed(seed, pose)
    projected = model.dof_map.to_full(model.dof_map.to_free(full_seed))
    boundary_roundoff = float((projected - full_seed).abs().max())
    boundary_roundoff_limit = float(
        8 * np.finfo(np.float64).eps * np.abs(physics.points).max()
    )
    assert boundary_roundoff <= boundary_roundoff_limit, (
        boundary_roundoff,
        boundary_roundoff_limit,
    )
    # Zero-pose pivot transforms can leave floating-point subtraction roundoff.
    # Use the exact prescribed values and audit the actual initialized state.
    seed = projected[: len(physics.points)].detach().clone()
    initial_collision = audit_collision_state(physics, seed, pose)
    assert initial_collision["state_feasible"], initial_collision
    state = model.State(u=projected.detach().clone())
    state.collision = model.collision.state_at(state.u)
    delegate = FeasibleExpressionProblem(model=model, collision_step_safety=0.9)
    hessian = HessianProblem(delegate, cfg.hessian_backend)
    problem = CachedProblem(hessian, exact_curvature=True)
    diagonal_validation = verify_exact_diagonal(problem, state)
    edge_mean = mean_rest_edge_length(model, physics.points)
    step_cap = 0.5 * edge_mean
    resume_force = float(torch.linalg.vector_norm(problem.grad(state)))
    g0 = resume_force if parent is None else parent["initial_grad_norm"]
    effective_tolerance = max(cfg.atol, cfg.rtol * g0)
    resume_receipt = None
    if parent is not None:
        assert effective_tolerance == parent["effective_grad_threshold"]
        replay_energy = float(problem.fun(state))
        assert np.isclose(
            resume_force, parent["final_grad_norm"], rtol=1e-10, atol=1e-15
        )
        assert np.isclose(replay_energy, parent["final_energy"], rtol=1e-10, atol=1e-20)
        assert np.array_equal(seed.detach().cpu().numpy(), seed_np)
        resume_receipt = {
            "parent_summary": record(cfg.resume_dir / "summary.json"),
            "parent_protocol": record(cfg.resume_dir / "protocol.json"),
            "parent_trace": record(cfg.resume_dir / "trace.jsonl"),
            "parent_terminal": record(seed_path),
            "start_iteration": start_iteration,
            "original_initial_grad_norm": g0,
            "effective_threshold_preserved": True,
            "numerical_sources_unchanged": True,
            "numerical_source_scope": "all archived experiment, apple, tensor-reference, solver-performance sources and neutral-newton/model_binding.py; driver gains resume handling, renderers and unused geometric initializer are outside this comparison",
            "seed_array_exactly_preserved": True,
            "replay_grad_norm": resume_force,
            "replay_energy": replay_energy,
            "grad_norm_absolute_replay_error": abs(
                resume_force - parent["final_grad_norm"]
            ),
            "energy_absolute_replay_error": abs(replay_energy - parent["final_energy"]),
            "shift_state": "reset to zero every Newton step; no cross-step optimizer history",
        }
    original = json.loads(
        Path(neutral.manifest["sources"]["protocol"]["path"]).read_text()
    )
    old_inputs = original["inputs"]
    # These are the exact geometry artifacts consumed by FrozenNeutral.build_physics.
    prepared = PreparedInputs.load(
        Path(old_inputs["prepared_npz"]), Path(old_inputs["prepared_manifest"])
    )
    protocol = {
        "schema": "single-reference-neutral-newton-v1",
        "config": cfg.model_dump(mode="json"),
        "inputs": {
            "prepared_volume": record(prepared.volume_path),
            "prepared_skin": record(prepared.skin_path),
            "geometry": record(
                Path(old_inputs["geometry"]["geometry"]["geometry_path"])
            ),
            "eyes_dir": str(cfg.eyes_dir.resolve()),
            "eyes": record(cfg.eyes_dir / "eyes.npz"),
            "skin_field": record(Path(old_inputs["skin_field_path"])),
            "neutral_material_binding": record(cfg.neutral_dir / "manifest.json"),
            "current_runtime_binding": record(cfg.output_dir / "model-binding.json"),
            "seed": record(seed_path),
            "seed_receipt": record(seed_receipt_path),
        },
        "start": "constitutive reference plus geometric eye-contact repair; no previous equilibrium displacement used"
        if parent is None
        else f"continue saved iteration {start_iteration}; preserve original cold-start force threshold",
        "resume": resume_receipt,
        "constitutive_reference_changed": False,
        "materials": materials,
        "mandible": {
            "degrees_of_freedom": 1,
            "neutral_rotation_rad": 0.0,
            "optimized_in_this_forward": False,
        },
        "muscle_activation": 0.0,
        "inverse": {
            "requested_algorithm": "Adam",
            "learning_rate": 1.0,
            "updates_in_this_run": 0,
        },
        "collision": coverage,
        "initial_collision": initial_collision,
        "boundary_projection": {
            "maximum_roundoff_m": boundary_roundoff,
            "roundoff_limit_m": boundary_roundoff_limit,
        },
        "solver": {
            "method": "Newton-CG only",
            "atol": cfg.atol,
            "rtol": cfg.rtol,
            "initial_grad_norm": g0,
            "effective_grad_threshold": effective_tolerance,
            "convergence_rule": "L2 norm of true free-coordinate gradient <= max(atol, rtol*initial_gradient_norm)",
            "gradient_units": "MPa*m^2; multiply by 1e6 for N",
            "max_steps": cfg.max_steps,
            "mean_mesh_edge_length_m": edge_mean,
            "max_coordinate_displacement_m": step_cap,
            "line_search": {"armijo": 1e-4, "factor": 0.5, "max_attempts": 8},
            "shift": {
                "initial": 0.0,
                "first": "mean(abs(diag(H)))",
                "multiplier": 10,
                "max_attempts": 8,
                "reset_each_newton_iteration": True,
            },
            "pcg": {
                "preconditioner": "abs(diag(H + shift*I))",
                "rtol": 1e-3,
                "max_steps": 1000,
                "true_residual_check": True,
            },
            "ccd": {
                "tolerance_m": 1e-10,
                "max_iterations": 100000,
                "safety": 0.9,
                "min_distance_m": model.collision.min_distance,
            },
            "physical_hessian": "exact unprojected bulk, membrane and IPC Hessian-vector products; shifts affect search only",
            "forward_solve_count": 1,
        },
        "initial_geometry": physics.metrics(seed),
        "diagonal_validation": diagonal_validation,
        "diagonal_policy": diagonal_policy,
        "hessian_backend": {
            "name": cfg.hessian_backend,
            "scope": "forward Hessian products only; physical energy, gradient, diagonal and contact unchanged",
            "initial_setup": hessian.report(),
        },
        "ipc_threads": int(ipctk.get_num_threads()),
        "runtime": {
            "torch": torch.__version__,
            "ipctk": ipctk.__version__,
            "gpu": torch.cuda.get_device_name(),
            "dtype": str(state.u.dtype),
        },
    }
    if parent_protocol is not None:
        for key in (
            "materials",
            "collision",
            "mandible",
            "muscle_activation",
            "diagonal_policy",
        ):
            assert protocol[key] == parent_protocol[key], key
        assert {
            key: value
            for key, value in protocol["solver"].items()
            if key != "max_steps"
        } == {
            key: value
            for key, value in parent_protocol["solver"].items()
            if key != "max_steps"
        }
        shutil.copyfile(cfg.resume_dir / "trace.jsonl", cfg.output_dir / "trace.jsonl")
    write_json(cfg.output_dir / "protocol.json", jsonable(protocol))
    save_checkpoint(cfg.output_dir / "initial.npz", seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    trace = list(parent_trace)
    accepted_steps = start_iteration
    failure = None
    previous_step = None
    converged = False
    faulthandler.dump_traceback_later(60, repeat=True)
    with torch.no_grad():
        for iteration in range(start_iteration, cfg.max_steps + 1):
            gradient = problem.grad(state)
            force = float(torch.linalg.vector_norm(gradient))
            energy = float(problem.fun(state))
            assert np.isfinite(force) and np.isfinite(energy)
            row = {
                "accepted": True,
                "iteration": iteration,
                "grad_norm": force,
                "energy": energy,
                "elapsed_seconds": previous_seconds + time.perf_counter() - started,
                "newton": previous_step,
            }
            if parent is None or iteration > start_iteration:
                trace.append(row)
                with (cfg.output_dir / "trace.jsonl").open("a") as stream:
                    stream.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
            write_json(
                cfg.output_dir / "status.json",
                {
                    "running": force > effective_tolerance
                    and iteration < cfg.max_steps,
                    **jsonable(row),
                    "effective_grad_threshold": effective_tolerance,
                },
            )
            LOG.info(
                "Accepted Newton %d: gradient %.8g (target %.8g), energy %.9g",
                iteration,
                force,
                effective_tolerance,
                energy,
            )
            cherries.set_step(iteration)
            cherries.log_metrics(
                {
                    "forward/grad_norm": force,
                    "forward/energy": energy,
                    "forward/threshold": effective_tolerance,
                }
            )
            if force <= effective_tolerance:
                converged = True
                break
            if iteration == cfg.max_steps:
                failure = {"reason": "Newton iteration budget exhausted"}
                break
            # Only accepted configurations become persistent output checkpoints.
            if iteration % cfg.checkpoint_every == 0:
                save_checkpoint(
                    cfg.output_dir / f"checkpoint-{iteration:03d}.npz",
                    state.u[: len(physics.points)],
                )
            try:
                state, previous_step = safeguarded_newton_step(
                    problem,
                    state,
                    atol=effective_tolerance,
                    linear_rtol=1e-3,
                    linear_max_steps=1000,
                    max_step_norm=step_cap,
                    armijo=1e-4,
                    max_shift_attempts=8,
                    max_backtracking_trials=8,
                    backtracking_factor=0.5,
                    preconditioner="diag",
                    shift_policy="reset",
                    shift_scale_policy="mean_abs",
                    gradient=gradient,
                )
                accepted_steps += 1
            except ForwardConvergenceError as error:
                failure = {"reason": str(error), "receipt": jsonable(error.receipt)}
                break
    faulthandler.cancel_dump_traceback_later()
    torch.cuda.synchronize()
    segment_seconds = time.perf_counter() - started
    forward_seconds = previous_seconds + segment_seconds
    # Independent fresh evaluation at the persisted terminal configuration.
    final_force = float(torch.linalg.vector_norm(delegate.grad(state)))
    final_energy = float(delegate.fun(state))
    displacement = state.u[: len(physics.points)].detach().clone()
    geometry = physics.metrics(displacement)
    geometry["skin_rms_mm"] = geometry["surface_motion_rms_mm"]
    collision = audit_collision_state(physics, displacement, pose)
    valid = geometry["inverted_tetrahedra"] == 0 and collision["state_feasible"]
    success = converged and final_force <= effective_tolerance and valid
    terminal = save_checkpoint(cfg.output_dir / "terminal.npz", displacement)
    summary = {
        "schema": "single-reference-neutral-newton-result-v1",
        "success": bool(success),
        "status": "converged_valid_neutral"
        if success
        else (
            "force_converged_geometry_invalid"
            if converged
            else "forward_did_not_converge"
        ),
        "failure": failure,
        "forward_solve_count": 1,
        "continuation_segments": 1
        if parent is None
        else parent.get("continuation_segments", 1) + 1,
        "start_iteration": start_iteration,
        "accepted_steps": accepted_steps,
        "segment_accepted_steps": accepted_steps - start_iteration,
        "initial_grad_norm": g0,
        "final_grad_norm": final_force,
        "effective_grad_threshold": effective_tolerance,
        "absolute_tolerance_met": final_force <= cfg.atol,
        "relative_tolerance_met": final_force <= cfg.rtol * g0,
        "final_energy": final_energy,
        "forward_seconds": forward_seconds,
        "segment_forward_seconds": segment_seconds,
        "resume": resume_receipt,
        "geometry": geometry,
        "collision": collision,
        "operation_counts": dict(problem.counts),
        "hessian_backend": hessian.report(),
        "rejected_contact_trials": delegate.rejected_contact_trials,
        "terminal": terminal,
        "protocol": record(cfg.output_dir / "protocol.json"),
        "adoption": "new result saved separately; existing selected neutral unchanged",
    }
    write_json(cfg.output_dir / "summary.json", jsonable(summary))
    write_json(cfg.output_dir / "status.json", {"running": False, **jsonable(summary)})
    cherries.log_output(cfg.output_dir)
    LOG.info("Single-forward result: %s", json.dumps(jsonable(summary)))
    if not success:
        message = f"Single neutral forward finished without a valid equilibrium: {summary['status']}"
        raise RuntimeError(message)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
