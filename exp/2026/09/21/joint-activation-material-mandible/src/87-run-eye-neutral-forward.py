# ruff: noqa: EM102, TRY003
"""Converge the prescribed-skin neutral with exact fixed source-eye collision."""

from __future__ import annotations

import faulthandler
import json
import logging
import signal
import time
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_frozen_neutral import FrozenNeutral, load_script
from joint_rigid_eye_contact import build_eye_collision_physics

from liblaf import cherries

LOG = logging.getLogger(__name__)
runner = load_script("68-run-simple-skin-forward.py")


class Config(cherries.BaseConfig):
    neutral_dir: Path = GROUP / "data/frozen-neutral-004"
    eyes_dir: Path = GROUP / "data/rigid-eyes-001"
    initialization_dir: Path = GROUP / "data/eye-initialization-003"
    contact_validation: Path = (
        GROUP / "data/rigid-eye-contact-validation-004/summary.json"
    )
    output_dir: Path = GROUP / "data/eye-neutral-forward-001"
    restart_checkpoint: Path | None = None
    ipc_threads: int | None = None
    max_steps: int = 100000
    wall_cap_seconds: float = 21600.0
    heartbeat_seconds: float = 15.0
    telemetry_interval_steps: int = 25
    checkpoint_interval_steps: int = 500
    line_search_armijo: float = 0.25
    max_step_norm_m: float = 0.0005
    hessian_damping_initial: float = 0.001
    pncg_restart_interval_steps: int = 200
    ccd_tolerance_m: float = 1e-10
    ccd_max_iterations: int = 100000
    collision_step_safety: float = 0.95


class FeasibleContactProblem(runner.TimedForwardProblem):
    """Reject numerical buffer violations before Armijo can accept a trial."""

    def __init__(self, *args: Any, collision_step_safety: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.collision_step_safety = collision_step_safety
        self.rejected_contact_trials = 0

    def max_step_size(self, state: Any, p: torch.Tensor) -> torch.Tensor:
        fraction = super().max_step_size(state, p)
        if float(fraction) < 1.0:
            fraction = fraction * self.collision_step_safety
        return fraction

    def fun(self, state: Any) -> torch.Tensor:
        energy = super().fun(state)
        collision = self.model.collision
        if len(state.collision.collisions):
            positions = (collision.vertices + state.u[collision.indices]).numpy(
                force=True
            )
            distance_sq = state.collision.collisions.compute_minimum_distance(
                collision.collision_mesh,
                positions,
            )
            if distance_sq <= collision.min_distance**2:
                self.rejected_contact_trials += 1
                return torch.full_like(energy, torch.inf)
        return energy


def intersections(physics: Any) -> bool:
    forward = physics.runtime.forward
    collision = forward.model.collision
    positions = (
        (collision.vertices + forward.state.u[collision.indices]).detach().cpu().numpy()
    )
    return bool(
        ipctk.has_intersections(collision.collision_mesh, positions, ipctk.LBVH())
    )


def record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def main(cfg: Config) -> None:  # noqa: PLR0915
    assert cfg.max_steps > 0
    assert cfg.wall_cap_seconds > 0
    assert cfg.ipc_threads is None or cfg.ipc_threads > 0
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    runner.configure_cuda()
    if cfg.ipc_threads is not None:
        ipctk.set_num_threads(cfg.ipc_threads)
    ipc_threads_actual = int(ipctk.get_num_threads())
    neutral = FrozenNeutral.load(cfg.neutral_dir)
    validation = json.loads(cfg.contact_validation.read_text())
    assert validation["success"] is True
    for path, digest in validation["implementation_sha256"].items():
        assert sha256(Path(path)) == digest
    threshold = neutral.manifest["force_threshold_code"]
    admission_path = cfg.initialization_dir / "summary.json"
    admission = json.loads(admission_path.read_text())
    assert admission["schema"] == "joint-eye-initialization-v1"
    assert admission["success"] is True
    assert admission["fem_reference_rebased"] is False
    assert admission["rigid_coordinates_changed"] is False
    assert admission["eyes"]["sha256"] == sha256(cfg.eyes_dir / "eyes.npz")
    assert admission["frozen_neutral"]["sha256"] == sha256(
        neutral.directory / "state.npz"
    )
    original_protocol = json.loads(
        Path(neutral.manifest["sources"]["protocol"]["path"]).read_text()
    )
    geometry_path = Path(
        original_protocol["inputs"]["geometry"]["geometry"]["geometry_path"]
    )
    assert admission["geometry"]["sha256"] == sha256(geometry_path)
    seed_path = cfg.initialization_dir / "seed.npz"
    # The initialization producer binds the admitted seed by file hash.
    assert sha256(seed_path) == admission["seed"]["sha256"]
    if cfg.restart_checkpoint is not None:
        seed_path = cfg.restart_checkpoint
        receipt = json.loads(seed_path.with_suffix(".json").read_text())
        assert sha256(seed_path) == receipt["sha256"]
    with np.load(seed_path, allow_pickle=False) as archive:
        fem_seed_np = archive["displacement_m"]
    assert fem_seed_np.shape == neutral.arrays["neutral_displacement_m"].shape
    assert np.isfinite(fem_seed_np).all()
    physics, baseline = build_eye_collision_physics(neutral, cfg.eyes_dir)
    forward = physics.runtime.forward
    model = forward.model
    # Resolve CCD more accurately than its 10 nm feasibility buffer. This
    # modifies swept-path numerics only; the IPC barrier energy is unchanged.
    model.collision.narrow_phase_ccd = ipctk.TightInclusionCCD(
        tolerance=cfg.ccd_tolerance_m,
        max_iterations=cfg.ccd_max_iterations,
    )
    physics.contact_definition["config"].update(
        {
            "ccd_tolerance_m": cfg.ccd_tolerance_m,
            "ccd_max_iterations": cfg.ccd_max_iterations,
        }
    )
    model.set_materials(baseline)
    pose = torch.zeros(6)
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    full_seed = physics.full_skull.extend_seed(torch.as_tensor(fem_seed_np), pose)
    projected = model.dof_map.to_full(model.dof_map.to_free(full_seed)).detach()
    assert torch.equal(projected, full_seed), "Initialization moved fixed supports"
    forward.state.u = projected.clone()
    forward.state.collision = model.collision.state_at(projected)
    assert not intersections(physics), "Initial soft versus rigid intersections"
    assert runner.contact_receipt(physics)["contact_numerically_valid"]
    wall_started = time.perf_counter()
    heartbeat = runner.Heartbeat(
        cfg.output_dir / "heartbeat.json",
        interval=cfg.heartbeat_seconds,
        wall_started=wall_started,
    )
    monitor = runner.ForwardMonitor(
        output_dir=cfg.output_dir,
        physics=physics,
        wall_started=wall_started,
        wall_cap_seconds=cfg.wall_cap_seconds,
        telemetry_interval=cfg.telemetry_interval_steps,
        checkpoint_interval=cfg.checkpoint_interval_steps,
        heartbeat=heartbeat,
    )
    monitor.last_accepted_u = projected.clone()
    problem = FeasibleContactProblem(
        model=model,
        monitor=monitor,
        volume_guard=None,
        collision_step_safety=cfg.collision_step_safety,
    )
    optimizer = runner.MonitoredPncg(
        criteria=forward.default_optimizer(
            max_steps=cfg.max_steps, rtol=0.0, atol=threshold
        ).criteria,
        hess_damping=runner.StrictPncg.HessianDamping(
            initial=cfg.hessian_damping_initial
        ),
        line_search=runner.StrictLineSearch(
            armijo=cfg.line_search_armijo,
            max_steps=60,
            max_step_norm=cfg.max_step_norm_m,
        ),
        monitor=monitor,
        restart_interval=cfg.pncg_restart_interval_steps,
    )
    forward.optimizer = optimizer
    protocol = {
        "schema": "joint-fixed-eyes-neutral-forward-protocol-v1",
        "config": cfg.model_dump(mode="json"),
        "neutral_manifest": record(neutral.directory / "manifest.json"),
        "initialization": record(admission_path),
        "contact_validation": record(cfg.contact_validation),
        "seed": record(seed_path),
        "geometry": physics.full_skull_receipt(),
        "runtime": runner.runtime_identity(),
        "ipctk_threads": {
            "requested": cfg.ipc_threads,
            "actual": ipc_threads_actual,
            "scope": "process-global IPCTK setting before rigid-eye collision construction",
        },
        "force_threshold_code": threshold,
        "mechanics": {
            "reference": "original frozen FEM constitutive reference, unchanged",
            "warm_start": "previous loaded neutral after local soft-eye overlap repair",
            "materials": "exact frozen heterogeneous skin and bulk fields",
            "poisson_ratio_all_tissues": 0.49,
            "bulk_additive_stress": 0,
            "activation": "none",
            "skin_baseline_stress": "original prescribed heterogeneous field, unchanged",
            "jaw_pose_rad_m": [0.0] * 6,
            "eyes": "all source triangles at original fixed world coordinates",
            "inversions": "diagnostic only, permitted by user",
        },
        "solver": {
            "method": "strict PNCG",
            "termination": "exact force at accepted state",
            "atol_code": threshold,
            "rtol": 0.0,
            "inverse": False,
            "adjoint": False,
            "ccd_tolerance_m": cfg.ccd_tolerance_m,
            "ccd_max_iterations": cfg.ccd_max_iterations,
            "collision_limited_step_safety": cfg.collision_step_safety,
            "trial_feasibility": "reject minimum active gap <= existing 10 nm CCD buffer before Armijo acceptance",
        },
    }
    write_json(cfg.output_dir / "protocol.json", protocol)
    initial_energy = float(problem.fun(forward.state))
    initial_force = float(torch.linalg.vector_norm(problem.grad(forward.state)))
    assert np.isfinite(initial_energy)
    assert np.isfinite(initial_force)
    monitor.initial_force_norm = monitor.last_force_norm = initial_force
    runner.append_jsonl(
        monitor.trace_path,
        {
            "accepted": True,
            "step": 0,
            "wall_elapsed_seconds": 0.0,
            "energy": initial_energy,
            "accepted_state_free_force_norm": initial_force,
            "contact": runner.contact_receipt(physics),
        },
    )
    monitor.checkpoint(problem, forward.state, None, label="step-00000")
    LOG.info(
        "Eye-inclusive PNCG: initial free force %.9g N; tolerance %.9g N",
        initial_force * 1e6,
        threshold * 1e6,
    )
    failure = None
    completed = initial_force <= threshold
    status = "converged_eye_neutral_forward" if completed else "running"
    heartbeat.start()
    watchdog = (cfg.output_dir / "watchdog-tracebacks.log").open("a")
    prior_signal = signal.getsignal(signal.SIGTERM)

    def request_stop(signum: int, _frame: Any) -> None:
        raise runner.ForwardInterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, request_stop)
    faulthandler.dump_traceback_later(60, repeat=True, file=watchdog)
    try:
        if not completed:
            solution = optimizer.minimize(problem, forward.state, forward.free)
            forward.last_solution = solution
            completed = bool(solution.success)
            status = "converged_eye_neutral_forward" if completed else "solver_failed"
    except (
        runner.ForwardConvergenceError,
        runner.ForwardWallTimeExceededError,
        runner.ForwardInterruptedError,
        KeyboardInterrupt,
    ) as error:
        status = type(error).__name__
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "receipt": getattr(error, "receipt", None),
        }
        LOG.exception("Eye-inclusive forward stopped")
    finally:
        faulthandler.cancel_dump_traceback_later()
        signal.signal(signal.SIGTERM, prior_signal)
        heartbeat.stop()
        write_json(cfg.output_dir / "heartbeat.json", heartbeat.snapshot())
        watchdog.close()
    model.update(forward.state, monitor.last_accepted_u)
    force = float(torch.linalg.vector_norm(problem.grad(forward.state)))
    monitor.last_force_norm = force
    checkpoint = monitor.checkpoint(
        problem,
        forward.state,
        None,
        label=f"terminal-step-{monitor.last_accepted_step:05d}",
    )
    contact = runner.contact_receipt(physics)
    has_intersections = intersections(physics)
    eye_u = forward.state.u[physics.full_skull.eye_global_ids]
    fixed_exact = torch.equal(
        forward.state.u.flatten()[model.dof_map.fixed_indices],
        model.dof_map.fixed_values,
    )
    eyes_fixed = bool(torch.count_nonzero(eye_u) == 0)
    components = runner.force_component_receipt(physics)
    match = abs(components["total_free_gradient_norm"] - force) / max(force, 1e-300)
    finite = bool(torch.isfinite(forward.state.u).all())
    gates = {
        "force_threshold_met": force <= threshold,
        "contact_numerically_valid": contact["contact_numerically_valid"],
        "soft_rigid_intersection_free": not has_intersections,
        "eyes_exactly_fixed": eyes_fixed,
        "all_fixed_dofs_exact": fixed_exact,
        "finite_displacement": finite,
        "force_components_match": match <= 1e-12,
    }
    completed = completed and all(gates.values())
    if not completed and status == "converged_eye_neutral_forward":
        status = "terminal_gate_failed"
    summary = {
        "schema": "joint-fixed-eyes-neutral-forward-v1",
        "success": completed,
        "status": status,
        "failure": failure,
        "accepted_steps": monitor.last_accepted_step,
        "wall_seconds": time.perf_counter() - wall_started,
        "initial_free_force_norm": initial_force,
        "final_free_force_norm": force,
        "force_threshold": threshold,
        "force_units": "MPa m^2; multiply by 1e6 for N",
        "checkpoint": checkpoint,
        "contact": contact,
        "terminal_gates": gates,
        "metrics": physics.metrics(forward.state.u[: len(fem_seed_np)]),
        "force_components": components,
        "force_component_relative_norm_match": match,
        "operation_timings": problem.timing_receipt(),
        "protocol_sha256": sha256(cfg.output_dir / "protocol.json"),
        "inversions_allowed": True,
        "common_poisson": 0.49,
        "activation": "none",
        "bulk_baseline_stress": 0,
        "rigid_eyes": True,
        "rigid_full_skull": True,
        "rejected_contact_trials": problem.rejected_contact_trials,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_output(cfg.output_dir)
    if not completed:
        raise RuntimeError(f"Eye-inclusive neutral did not converge: {status}")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
