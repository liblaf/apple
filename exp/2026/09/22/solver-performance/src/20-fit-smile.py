# ruff: noqa: C901, E402, EM101, PLR0915, PT018, TRY003, TRY300, TRY301
"""Matched, contact-on inverse fit of the ``Smile`` target.

This adapter preserves the reviewed expression fitter and records explicit
objective and solver overrides.  With default options, the paired arms differ
only in their primal equilibrium runtime.  GPU contact products are opt-in;
their default scope is the adjoint linear solve.  A contact-on neutral
refinement is prepared once, hashed, and used as the common initial state.
Its time is reported separately from paired fit times.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

import ipctk
import torch

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
SOURCE_GROUP = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
sys.path.insert(0, str(SOURCE_GROUP / "src"))

from joint_common import sha256, write_json
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_inputs import EyeExpressionInputs

LOG = logging.getLogger(__name__)
SMILE = "Smile"
ARMS = ("original", "hybrid_diag")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = load_module(EXPERIMENT / "src/10-benchmark.py", "smile_benchmark")


class Config(cherries.BaseConfig):
    output_dir: Path = EXPERIMENT / "data/smile-fit-001"
    source_root: Path = EXPERIMENT.parents[4]
    origin_metadata: Path | None = None
    inputs_dir: Path = SOURCE_GROUP / "data/expression-inputs-002"
    calibration_source: Path = SOURCE_GROUP / "data/expression-fitting-007"
    maximum_iterations: int = 20
    wall_cap_seconds: float = 43200.0
    forward_wall_seconds: float | None = None
    forward_atol: float = 1e-8
    adjoint_rtol: float = 1e-7
    learning_rate: float = 0.3
    # None reuses the calibrated 36-expression weight.  A numeric value is an
    # explicit, recorded ablation override; it never edits that calibration.
    smoothness_weight: float | None = None
    magnitude_weight: float = 0.0
    jaw_weight: float = 0.0
    outer_step_policy: Literal["armijo", "full_adam"] = "full_adam"
    max_backtracks: int = 12
    armijo: float = 1e-4
    trial_prescreen: bool = False
    linear_rtol: float = 1e-3
    max_newton_steps: int = 100
    newton_switch_atol: float = 0.0
    newton_shift_policy: Literal["reset", "reuse"] = "reset"
    gpu_contact: bool = False
    gpu_contact_scope: Literal["all", "adjoint"] = "adjoint"
    checkpoint_every: int = 5
    ipc_threads: int | None = 8
    mode: str = "all"  # prepare, fit, or all
    method: str = "all"  # old/original, hybrid/hybrid_diag, or all
    shared_dir: Path | None = None
    resume: bool = False


def record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_fitter_module() -> Any:
    return load_module(SOURCE_GROUP / "src/93-fit-expressions.py", "smile_fitter")


def make_fitter(cfg: Config, arm_dir: Path, *, resume: bool) -> tuple[Any, Any]:
    """Construct a fresh production fitter without calling its all-36 run loop."""
    runner = load_fitter_module()
    fit_cfg = runner.Config(
        _cli_parse_args=False,
        output_dir=arm_dir,
        inputs_dir=cfg.inputs_dir,
        calibration_source=None,
        resume=resume,
        ipc_threads=cfg.ipc_threads,
        maximum_iterations_per_expression=cfg.maximum_iterations,
        wall_cap_seconds=cfg.wall_cap_seconds,
        learning_rate=cfg.learning_rate,
        magnitude_weight=cfg.magnitude_weight,
        jaw_weight=cfg.jaw_weight,
        outer_step_policy=cfg.outer_step_policy,
        forward_atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        max_backtracks=cfg.max_backtracks,
        armijo=cfg.armijo,
        trial_prescreen=cfg.trial_prescreen,
        pose_first=False,
        pose_collision=True,
    )
    fitter = runner.Fitter(fit_cfg)
    # The harness never invokes Fitter.calibrate(): the verified common weight
    # is bound below, so no extra 36-adjoint workload enters either arm.
    return runner, fitter


def calibration_receipt(cfg: Config) -> dict[str, Any]:
    calibration = cfg.calibration_source / "calibration.json"
    protocol = cfg.calibration_source / "protocol.json"
    value = json.loads(calibration.read_text())
    parent = json.loads(protocol.read_text())
    assert value["success"] and value["expression_count"] == 36
    assert parent["inputs_manifest_sha256"] == sha256(cfg.inputs_dir / "manifest.json")
    assert value["probe_scale"] == 0.001 and value["strength_factor"] == 3.0
    return {
        "scope": "reused existing 36-expression calibrated normalized strong_weight exactly",
        "strong_weight": value["strong_weight"],
        "calibration": record(calibration),
        "protocol": record(protocol),
    }


def smoothness_weight_receipt(
    cfg: Config, calibration: dict[str, Any]
) -> dict[str, Any]:
    """Resolve an opt-in objective ablation without mutating calibration evidence."""
    inherited = float(calibration["strong_weight"])
    override = cfg.smoothness_weight
    effective = inherited if override is None else float(override)
    assert effective >= 0
    return {
        "calibrated_strong_weight": inherited,
        "override": override,
        "effective_weight": effective,
        "mode": "inherited_calibration" if override is None else "explicit_override",
    }


def bind_smoothness_weight(fitter: Any, weight: dict[str, Any]) -> None:
    """Set the production fitter's actual objective coefficient once per arm."""
    fitter.smooth_weight = weight["effective_weight"]
    assert fitter.smooth_weight == weight["effective_weight"]


def check_collision(physics: Any) -> dict[str, Any]:
    """Require the companion audit; no collision-off fallback is permitted."""
    from smile_collision import audit_required_collision

    receipt = audit_required_collision(physics)
    assert receipt["success"], receipt
    return receipt


def check_collision_state(
    physics: Any, displacement: torch.Tensor, jaw: torch.Tensor
) -> dict[str, Any]:
    from smile_collision import audit_collision_state
    from smile_fitter import hinge_pose

    receipt = audit_collision_state(
        physics,
        displacement,
        hinge_pose(
            jaw, torch.as_tensor(physics.base.arrays["mandible_frame_world"][:, 0])
        ),
    )
    assert receipt["state_feasible"], receipt
    return receipt


def arm_summary(
    *,
    arm: str,
    fitter: Any,
    directory: Path,
    collision: dict[str, Any],
    shared_neutral: dict[str, Any],
    terminal: str,
    timing: list[dict[str, Any]],
) -> dict[str, Any]:
    latest = directory / "expressions" / SMILE / "latest.pt"
    status = fitter.status["expressions"][SMILE]
    forward = status.get("forward", {})
    shape = status.get("shape", {})
    return {
        "schema": "smile-solver-arm-v1",
        "inverse_converged": terminal == "converged",
        "success": terminal
        in {"converged", "iteration_budget_reached", "wall_budget_reached"}
        and forward.get("success", False)
        and shape.get("inverted_tetrahedra", 0) == 0,
        "arm": arm,
        "target_expression": SMILE,
        "source_expression_index": 12,
        "terminal": terminal,
        "status": status,
        "latest_checkpoint": record(latest) if latest.exists() else None,
        "collision": collision,
        "shared_neutral": shared_neutral,
        "timing": {
            "fit_step_seconds": sum(row["fit_step_seconds"] for row in timing),
            "forward_seconds": sum(row.get("forward_seconds", 0.0) for row in timing),
            "adjoint_seconds": sum(row.get("adjoint_seconds", 0.0) for row in timing),
            "iterations_observed": len(timing),
        },
        "physical_terminal": {
            "force_norm": forward.get("grad_norm"),
            "force_threshold": forward.get("force_threshold"),
            "contact": forward.get("contact"),
            "inverted_tetrahedra": shape.get("inverted_tetrahedra"),
            "detF_min": shape.get("detF_min"),
        },
    }


def write_arm_summary(path: Path, **kwargs: Any) -> dict[str, Any]:
    summary = arm_summary(**kwargs)
    write_json(path / "arm-summary.json", summary)
    return summary


def prepare_shared_neutral(
    cfg: Config,
    smoothness_weight: dict[str, Any],
) -> dict[str, Any]:
    """Produce one tight, contact-on seed outside the paired fit time budget."""
    output = cfg.output_dir / "shared-neutral-preparation"
    artifact = cfg.output_dir / "shared-neutral-init.pt"
    receipt_path = cfg.output_dir / "shared-neutral-init.json"
    if artifact.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["forward_atol"] == cfg.forward_atol
        assert sha256(artifact) == receipt["artifact"]["sha256"]
        return receipt
    output.mkdir()
    runner, fitter = make_fitter(cfg, output, resume=False)
    bind_smoothness_weight(fitter, smoothness_weight)
    collision = check_collision(fitter.physics)
    from accelerated_solvers import accelerate_runtime

    # This input is an already-converged neutral endpoint.  A relative PNCG
    # transition would equal the final tolerance here and do no Newton work.
    fitter.runtime = accelerate_runtime(
        fitter.runtime,
        "newton_diag",
        rest_points=fitter.physics.points,
        wall_seconds=cfg.forward_wall_seconds,
        linear_rtol=cfg.linear_rtol,
        max_newton_steps=cfg.max_newton_steps,
        newton_switch_atol=cfg.newton_switch_atol,
    )
    fitter.physics.runtime = fitter.runtime
    started = time.perf_counter()
    try:
        fitter.refine_neutral()
    except BaseException as error:
        failure = {
            "success": False,
            "error_type": type(error).__name__,
            "failure": str(error),
            "seconds": time.perf_counter() - started,
            "collision": collision,
        }
        write_json(receipt_path, failure)
        raise
    runner.atomic_torch(
        artifact,
        runner.cpu_tree({"displacement_m": fitter.neutral_seed.detach().clone()}),
    )
    collision["shared_neutral_state"] = check_collision_state(
        fitter.physics,
        fitter.neutral_seed,
        torch.zeros(1, device=fitter.neutral.device),
    )
    receipt = {
        "success": True,
        "scope": "contact-on numerical initialization; excluded from paired arm times",
        "primal_method": "newton_diag refinement of the already-converged neutral; excluded from inverse-fit arm timings",
        "forward_atol": cfg.forward_atol,
        "smoothness_weight": smoothness_weight,
        "seconds": time.perf_counter() - started,
        "artifact": record(artifact),
        "displacement_tensor_sha256": tensor_sha256(fitter.neutral_seed),
        "forward": copy.deepcopy(fitter.runtime.last_forward),
        "collision": collision,
    }
    write_json(receipt_path, receipt)
    return receipt


def install_arm_runtime(cfg: Config, fitter: Any, arm: str) -> None:
    if arm != "original":
        from accelerated_solvers import accelerate_runtime

        runtime = accelerate_runtime(
            fitter.runtime,
            "hybrid_diag",
            rest_points=fitter.physics.points,
            wall_seconds=cfg.forward_wall_seconds,
            linear_rtol=cfg.linear_rtol,
            max_newton_steps=cfg.max_newton_steps,
            newton_switch_atol=cfg.newton_switch_atol,
            shift_policy=cfg.newton_shift_policy,
        )
        fitter.runtime = runtime
        fitter.physics.runtime = runtime
    if cfg.gpu_contact:
        if cfg.gpu_contact_scope == "all":
            from gpu_contact import install_gpu_contact

            fitter.gpu_contact_handle = install_gpu_contact(fitter.physics)
        else:
            from gpu_contact import install_adjoint_gpu_contact

            fitter.gpu_contact_handle = install_adjoint_gpu_contact(
                fitter.runtime, fitter.physics
            )


def release_arm_runtime(fitter: Any) -> None:
    handle = getattr(fitter, "gpu_contact_handle", None)
    if handle is not None:
        handle.uninstall()
        fitter.gpu_contact_handle = None


def run_arm(
    cfg: Config,
    arm: str,
    calibration: dict[str, Any],
    smoothness_weight: dict[str, Any],
    shared_neutral: dict[str, Any],
    shared_dir: Path,
) -> dict[str, Any]:
    directory = cfg.output_dir / "arms" / arm
    expression_dir = directory / "expressions" / SMILE
    if not cfg.resume:
        if not directory.exists():
            directory.mkdir(parents=True)
    else:
        assert directory.is_dir(), directory
    runner, fitter = make_fitter(cfg, directory, resume=cfg.resume)
    install_arm_runtime(cfg, fitter, arm)
    collision = check_collision(fitter.physics)
    seed = torch.load(
        shared_dir / "shared-neutral-init.pt",
        map_location=fitter.neutral.device,
        weights_only=False,
    )["displacement_m"]
    fitter.neutral_seed = seed.detach().clone()
    assert (
        tensor_sha256(fitter.neutral_seed)
        == shared_neutral["displacement_tensor_sha256"]
    )
    bind_smoothness_weight(fitter, smoothness_weight)
    original_evaluate = fitter.evaluate
    evaluation_calls: list[dict[str, Any]] = []

    def evaluate_with_geometry_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        started = time.perf_counter()
        candidate = None
        failure = None
        try:
            candidate = original_evaluate(*args, **kwargs)
            inversions = candidate["metrics"]["shape"].get("inverted_tetrahedra", 0)
            if inversions > 0:
                raise ForwardConvergenceError(
                    "candidate has inverted tetrahedra",
                    receipt={"inverted_tetrahedra": inversions},
                )
            return candidate
        except BaseException as error:
            failure = {"error_type": type(error).__name__, "failure": str(error)}
            if isinstance(error, runner.TrialObjectiveRejectedError):
                failure["objective_prescreen"] = error.receipt
            raise
        finally:
            prescreen = failure is not None and "objective_prescreen" in failure
            prescreen_stage = (
                failure["objective_prescreen"]["stage"] if prescreen else None
            )
            forward = (
                {}
                if prescreen_stage == "before_forward"
                else copy.deepcopy(getattr(fitter.runtime, "last_forward", {}))
            )
            adjoint = (
                {}
                if prescreen
                else copy.deepcopy(getattr(fitter.runtime, "last_adjoint", {}))
            )
            evaluation_calls.append(
                {
                    "total_seconds": time.perf_counter() - started,
                    "forward_seconds": forward.get("seconds", 0.0),
                    "adjoint_seconds": adjoint.get("seconds", 0.0)
                    if candidate is not None
                    else 0.0,
                    "forward": forward,
                    "adjoint": adjoint if candidate is not None else None,
                    "success": candidate is not None and failure is None,
                    "failure": failure,
                }
            )

    fitter.evaluate = evaluate_with_geometry_gate
    fitter.status["smoothness_weight"] = fitter.smooth_weight
    fitter.status["solver_comparison"] = {
        "arm": arm,
        "primal": "production AcceptedForcePncg"
        if arm == "original"
        else "hybrid_diag PNCG-to-Newton-CG",
        "target_expression": SMILE,
        "source_expression_index": 12,
        "forward_atol": cfg.forward_atol,
        "adjoint_rtol": cfg.adjoint_rtol,
        "newton_shift_policy": cfg.newton_shift_policy,
        "gpu_contact": cfg.gpu_contact,
        "gpu_contact_scope": cfg.gpu_contact_scope if cfg.gpu_contact else None,
        "learning_rate": cfg.learning_rate,
        "smoothness_weight": smoothness_weight,
        "magnitude_weight": cfg.magnitude_weight,
        "jaw_weight": cfg.jaw_weight,
        "outer_step_policy": cfg.outer_step_policy,
        "trial_prescreen": cfg.trial_prescreen,
        "trial_prescreen_policy": (
            "skip a trial before forward when its non-negative prior exceeds the "
            "raw Armijo ceiling; after forward, skip only its adjoint when its raw "
            "objective exceeds that ceiling; retain the existing residual-corrected "
            "acceptance checks for every potentially acceptable trial"
            if cfg.trial_prescreen
            else "disabled"
        ),
        "shared_neutral": shared_neutral["artifact"],
        "collision": collision,
    }
    write_json(directory / "calibration-reuse.json", calibration)
    write_json(directory / "collision.json", collision)
    smile_index = fitter.names.index(SMILE)
    assert smile_index == 12
    timing_path = directory / "iteration-timing.jsonl"
    timings = read_jsonl(timing_path)
    terminal = {
        "converged",
        "line_search_failed",
        "stationary_primal_unresolved",
        "descent_unresolved",
        "requires_contact_recovery",
    }
    started = time.perf_counter()
    while True:
        current = fitter.status["expressions"][SMILE]
        if current.get("status") in terminal:
            break
        if current.get("accepted_steps", 0) >= cfg.maximum_iterations:
            fitter.status["expressions"][SMILE] = {
                **current,
                "status": "iteration_budget_reached",
                "maximum_iterations": cfg.maximum_iterations,
            }
            break
        if time.perf_counter() - started > cfg.wall_cap_seconds:
            fitter.status["expressions"][SMILE] = {
                **current,
                "status": "wall_budget_reached",
            }
            break
        before_trials = read_jsonl(expression_dir / "trials.jsonl")
        before_evaluations = len(evaluation_calls)
        before = current.get("accepted_steps", 0)
        iteration_started = time.perf_counter()
        try:
            fitter.fit_step(smile_index)
        except BaseException as error:
            fitter.status["expressions"][SMILE] = {
                **fitter.status["expressions"][SMILE],
                "status": "failed",
                "error_type": type(error).__name__,
                "failure": str(error),
            }
            write_json(directory / "failure.json", fitter.status["expressions"][SMILE])
            release_arm_runtime(fitter)
            raise
        elapsed = time.perf_counter() - iteration_started
        after_trials = read_jsonl(expression_dir / "trials.jsonl")
        emitted = after_trials[len(before_trials) :]
        calls = evaluation_calls[before_evaluations:]
        latest = torch.load(
            expression_dir / "latest.pt", map_location="cpu", weights_only=False
        )
        metrics = latest["metrics"]
        timing = {
            "from_accepted_step": before,
            "accepted_step": latest["accepted_steps"],
            "fit_step_seconds": elapsed,
            "total_seconds": elapsed,
            "forward_seconds": sum(call["forward_seconds"] for call in calls),
            "adjoint_seconds": sum(call["adjoint_seconds"] for call in calls),
            "evaluation_seconds": sum(call["total_seconds"] for call in calls),
            "evaluations": calls,
            "accepted_trials": sum(bool(row.get("accepted")) for row in emitted),
            "rejected_trials": sum(not bool(row.get("accepted")) for row in emitted),
            "trials": emitted,
            "objective": metrics["objective"],
            "fit_rms_mm": metrics["fit_rms_mm"],
            "physical": {
                "force_norm": metrics["forward"].get("grad_norm"),
                "force_threshold": metrics["forward"].get("force_threshold"),
                "contact": metrics["forward"].get("contact"),
                "inverted_tetrahedra": metrics["shape"].get("inverted_tetrahedra"),
                "detF_min": metrics["shape"].get("detF_min"),
            },
        }
        append(timing_path, timing)
        timings.append(timing)
        if (
            latest["accepted_steps"]
            and latest["accepted_steps"] % cfg.checkpoint_every == 0
        ):
            runner.atomic_torch(
                expression_dir / f"comparison-step-{latest['accepted_steps']:05d}.pt",
                runner.cpu_tree(latest),
            )
        terminal_status = fitter.status["expressions"][SMILE].get("status", "fitting")
        write_arm_summary(
            directory,
            arm=arm,
            fitter=fitter,
            directory=directory,
            collision=collision,
            shared_neutral=shared_neutral,
            terminal=terminal_status,
            timing=timings,
        )
        LOG.info(
            "%s Smile step %d: RMS %.6f mm in %.3fs",
            arm,
            latest["accepted_steps"],
            metrics["fit_rms_mm"],
            elapsed,
        )
    fitter.status["running"] = False
    terminal_status = fitter.status["expressions"][SMILE]["status"]
    latest = expression_dir / "latest.pt"
    if latest.exists():
        state = torch.load(
            latest, map_location=fitter.neutral.device, weights_only=False
        )
        collision["terminal_state"] = check_collision_state(
            fitter.physics, state["displacement_m"], state["jaw_normalized"]
        )
    fitter.status["phase"] = terminal_status
    write_json(directory / "status.json", fitter.status)
    result = write_arm_summary(
        directory,
        arm=arm,
        fitter=fitter,
        directory=directory,
        collision=collision,
        shared_neutral=shared_neutral,
        terminal=terminal_status,
        timing=timings,
    )
    release_arm_runtime(fitter)
    return result


def main(cfg: Config) -> None:
    assert cfg.maximum_iterations > 0 and cfg.wall_cap_seconds > 0
    assert cfg.forward_wall_seconds is None or cfg.forward_wall_seconds > 0
    assert cfg.forward_atol > 0
    assert cfg.smoothness_weight is None or cfg.smoothness_weight >= 0
    assert cfg.magnitude_weight >= 0 and cfg.jaw_weight >= 0
    assert cfg.outer_step_policy != "full_adam" or not cfg.trial_prescreen
    assert cfg.newton_switch_atol >= 0
    assert cfg.checkpoint_every > 0 and 0 < cfg.linear_rtol < 1
    assert cfg.ipc_threads is None or cfg.ipc_threads > 0
    assert cfg.inputs_dir.is_dir() and cfg.calibration_source.is_dir()
    assert cfg.mode in {"prepare", "fit", "all"}
    assert cfg.method in {"old", "original", "hybrid", "hybrid_diag", "all"}
    selected = {
        "old": "original",
        "original": "original",
        "hybrid": "hybrid_diag",
        "hybrid_diag": "hybrid_diag",
    }
    requested_arms = ARMS if cfg.method == "all" else (selected[cfg.method],)
    saved_protocol = None
    if cfg.resume:
        assert cfg.output_dir.is_dir(), cfg.output_dir
        saved_protocol = json.loads((cfg.output_dir / "protocol.json").read_text())
        assert saved_protocol["frozen_inputs"]["inputs_manifest"]["sha256"] == sha256(
            cfg.inputs_dir / "manifest.json"
        )
    elif not cfg.output_dir.exists():
        cfg.output_dir.mkdir(parents=True, exist_ok=False)
        BENCHMARK.archive_benchmark_sources(cfg)
    from remote_paths import install_loader_path_relocation

    install_loader_path_relocation(source_root=cfg.source_root)
    if cfg.ipc_threads is not None:
        ipctk.set_num_threads(cfg.ipc_threads)
    configure_cuda()
    inputs = EyeExpressionInputs.load(cfg.inputs_dir)
    smile_index = inputs.expression_names.index(SMILE)
    assert smile_index == 12
    calibration = calibration_receipt(cfg)
    smoothness_weight = smoothness_weight_receipt(cfg, calibration)
    frozen = {
        "inputs_manifest": record(cfg.inputs_dir / "manifest.json"),
        "inputs_state": record(cfg.inputs_dir / "state.npz"),
        "calibration": calibration,
    }
    protocol = {
        "schema": "smile-solver-comparison-v1",
        "scope": "matched contact-on inverse physics; one Smile target, shared recorded optimizer/objective/adjoint options, paired primal runtimes",
        "target_expression": SMILE,
        "source_expression_index": smile_index,
        "arms": list(ARMS),
        "requested_arms": list(requested_arms),
        "common_initialization": "one newton_diag contact-on tight neutral refinement is frozen and excluded from paired arm timing",
        "common_contact": "required full skull, mandible, and rigid-eye collision model; no collision-off stage",
        "candidate_geometry_gate": (
            "an invalid physical evaluation stops the run visibly; no smaller-step retry"
            if cfg.outer_step_policy == "full_adam"
            else "reject any inverse proposal with inverted_tetrahedra > 0 before Armijo acceptance"
        ),
        "outer_step_policy": (
            "one full projected Adam update per iteration; no loss, slope, roughness-budget, or residual-error rejection; no outer backtracking; physical solve failures stop visibly"
            if cfg.outer_step_policy == "full_adam"
            else "projected Adam with descent safeguard and residual-corrected Armijo backtracking"
        ),
        "objective": {
            "data": "normalized area-weighted skin-position MSE",
            "smoothness_weight": smoothness_weight["effective_weight"],
            "calibrated_smoothness_weight": smoothness_weight[
                "calibrated_strong_weight"
            ],
            "smoothness_weight_override": smoothness_weight["override"],
            "smoothness_weight_mode": smoothness_weight["mode"],
            "magnitude_weight": cfg.magnitude_weight,
            "jaw_weight": cfg.jaw_weight,
        },
        "trial_prescreen": "enabled" if cfg.trial_prescreen else "disabled",
        "newton_switch_policy": "hybrid warm-up ends at max(final force tolerance, 1e-3 times initial force, configured newton_switch_atol); the optional absolute floor is an empirical transition heuristic, not a global convergence guarantee. Newton still must satisfy the unchanged final force, contact, and inversion gates.",
        "active_stress": "six-component symmetric material stress Q: active energy is 0.5 Q:(F^T F-I), yielding P_active=FQ through activation_stresses_mpa(q, REFERENCE_MPA); fixed materials; no active-strain substitution",
        "common_forward_atol": cfg.forward_atol,
        "config": cfg.model_dump(mode="json"),
        "comparison_config": {
            key: getattr(cfg, key)
            for key in (
                "forward_atol",
                "adjoint_rtol",
                "learning_rate",
                "smoothness_weight",
                "magnitude_weight",
                "jaw_weight",
                "outer_step_policy",
                "max_backtracks",
                "armijo",
                "trial_prescreen",
                "linear_rtol",
                "max_newton_steps",
                "newton_switch_atol",
                "newton_shift_policy",
                "gpu_contact",
                "gpu_contact_scope",
                "checkpoint_every",
                "ipc_threads",
            )
        }
        | {"effective_smoothness_weight": smoothness_weight["effective_weight"]},
        "frozen_inputs": frozen,
        "implementation": {
            "20-fit-smile.py": sha256(Path(__file__)),
            "93-fit-expressions.py": sha256(SOURCE_GROUP / "src/93-fit-expressions.py"),
            "accelerated_solvers.py": sha256(
                Path(__file__).with_name("accelerated_solvers.py")
            ),
            "gpu_contact.py": sha256(Path(__file__).with_name("gpu_contact.py"))
            if cfg.gpu_contact
            else None,
        },
        "ipctk_threads_actual": int(ipctk.get_num_threads()),
    }
    if not cfg.resume and not (cfg.output_dir / "protocol.json").exists():
        write_json(cfg.output_dir / "protocol.json", protocol)
        write_json(cfg.output_dir / "frozen-inputs.json", frozen)
    if cfg.mode == "prepare":
        shared_neutral = prepare_shared_neutral(cfg, smoothness_weight)
        write_json(
            cfg.output_dir / "summary.json",
            {**protocol, "shared_neutral_init": shared_neutral, "arms": {}},
        )
        return
    shared_dir = cfg.shared_dir or cfg.output_dir
    assert shared_dir.is_dir(), shared_dir
    if cfg.mode == "all" and not (shared_dir / "shared-neutral-init.pt").exists():
        shared_neutral = prepare_shared_neutral(cfg, smoothness_weight)
    else:
        shared_neutral = json.loads(
            (shared_dir / "shared-neutral-init.json").read_text()
        )
    assert shared_neutral["success"]
    assert (
        sha256(shared_dir / "shared-neutral-init.pt")
        == shared_neutral["artifact"]["sha256"]
    )
    # A stricter common seed remains valid when both paired arms use a looser
    # production stopping tolerance. The seed bytes and provenance still match.
    assert shared_neutral["forward_atol"] <= cfg.forward_atol
    assert shared_neutral["forward"]["grad_norm"] <= cfg.forward_atol
    shared_protocol = json.loads((shared_dir / "protocol.json").read_text())
    assert shared_protocol["frozen_inputs"] == frozen
    if saved_protocol is not None:
        assert saved_protocol["comparison_config"] == protocol["comparison_config"]
        assert saved_protocol["frozen_inputs"] == frozen
        assert saved_protocol["implementation"] == protocol["implementation"]
        saved_summary = json.loads((cfg.output_dir / "summary.json").read_text())
        previous_seed = saved_summary.get("shared_neutral_init", {}).get("artifact")
        if previous_seed is not None:
            assert previous_seed == shared_neutral["artifact"]
    existing_summary = cfg.output_dir / "summary.json"
    summaries = (
        json.loads(existing_summary.read_text()).get("arms", {})
        if existing_summary.exists()
        else {}
    )
    arms = requested_arms
    for arm in arms:
        summaries[arm] = run_arm(
            cfg, arm, calibration, smoothness_weight, shared_neutral, shared_dir
        )
        write_json(
            cfg.output_dir / "summary.json",
            {**protocol, "shared_neutral_init": shared_neutral, "arms": summaries},
        )
    paired_comparison_complete = set(summaries) == set(ARMS)
    success = all(summaries[arm]["success"] for arm in arms)
    summary = {
        **protocol,
        "success": success,
        "paired_comparison_complete": paired_comparison_complete,
        "shared_neutral_init": shared_neutral,
        "arms": summaries,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_metrics(
        {
            "smile_fit/arms_completed": len(summaries),
            "smile_fit/success": float(success),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=BENCHMARK.ProfilePerformance)
