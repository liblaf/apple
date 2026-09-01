"""Separate unbounded shared-control fallback after a stalled canonical h020 run.

This is deliberately a new ledger.  It re-verifies the saved canonical shared
endpoint, then uses only inverse-BFGS, accepted-equilibrium continuation, and
Armijo backtracking.  Failed/non-finite trials are receipts, never objectives.
"""

from __future__ import annotations

# ruff: noqa: ARG001, BLE001, C901, E702, EM101, EM102, FBT003, PLR0912, PLR0915, SLF001, TRY003
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import scipy.sparse.linalg as spla

from liblaf import cherries

RUNNER_PATH = Path(__file__).with_name("20-run-canonical-h020.py")
SPEC = importlib.util.spec_from_file_location("pork_canonical", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {RUNNER_PATH}")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    canonical_root: Path = Path("data/20-canonical-h020")
    output_dir: Path = cherries.output("30-shared-fallback-h020", mkdir=True)
    max_accepted: int = 100
    max_trials: int = 1000
    max_no_progress_restarts: int = 5
    failed_trial_eligibility_cap: int = 10
    armijo: float = 1e-4
    alpha_floor: float = 2.0**-40
    forward_tolerance: float = 1e-10
    forward_max_iterations: int = 3000
    gradient_inf_tolerance: float = 2e-8
    gradient_rms_tolerance: float = 1e-8


def sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def evaluate(
    case: Any, controls: np.ndarray, initial_u: np.ndarray, cfg: Config
) -> tuple[Any, float, np.ndarray]:
    runner._SHARED = True
    mesh = runner.build_mesh(case)
    runner._GROUPS = runner.group_map(mesh, True)
    strict = runner.Config(
        validate_derivatives=False,
        forward_tolerance=cfg.forward_tolerance,
        forward_max_iterations=cfg.forward_max_iterations,
    )
    state = runner.core.solve(mesh, controls, "stable", initial_u, strict)
    if not state.converged:
        raise RuntimeError(
            f"forward nonconvergence: iterations={state.iterations}, failure={state.failure!r}"
        )
    value, du, _ = runner.loss(mesh, state.u, case.height, "l2", case.length)
    _, _, _, mixed, *_ = runner.assembly(mesh, state.u, controls, "stable", False, True)
    gradient = np.asarray(mixed.T @ spla.spsolve(state.h, -du)).ravel()
    if not (np.isfinite(value) and np.isfinite(gradient).all()):
        raise FloatingPointError("non-finite objective or gradient")
    return state, float(value), gradient


def row(mesh: Any, case: Any, state: Any, controls: np.ndarray, value: float, gradient: np.ndarray, step: int, phase: str) -> dict[str, Any]:
    detf = np.asarray(state.det_f)
    deta = np.asarray(state.det_a)
    detg = np.asarray(state.det_g)
    def fraction(mask: np.ndarray) -> float:
        return float(mesh.area[mask].sum() / mesh.area.sum())
    return {
        "step": step,
        "phase": phase,
        "objective": value,
        "gradient_inf": float(np.linalg.norm(gradient, np.inf)),
        "gradient_rms": float(np.linalg.norm(gradient) / math.sqrt(gradient.size)),
        "equilibrium_residual_rms": float(np.linalg.norm(state.r) / math.sqrt(mesh.nfree)),
        "forward_iterations": int(state.iterations),
        "min_det_f": float(detf.min()),
        "min_det_ainv": float(deta.min()),
        "min_det_g": float(detg.min()),
        "inverted_rest_measure_fraction": fraction(detf < 0),
        "ainv_negative_rest_measure_fraction": fraction(deta < 0),
        "g_negative_rest_measure_fraction": fraction(detg < 0),
        "double_inverted_rest_measure_fraction": fraction((detf < 0) & (deta < 0)),
        "activation_sha256": sha(controls),
        "displacement_sha256": sha(state.u),
    }


def main(cfg: Config) -> None:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(cfg.output_dir)
    canonical = cfg.canonical_root.resolve() / "h020-shared"
    summary_path, seed_path = canonical / "summary.json", canonical / "final-state.npz"
    if not summary_path.is_file() or not seed_path.is_file():
        raise FileNotFoundError("canonical shared summary/final-state is not complete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["case"]["name"] != "h020-shared":
        raise ValueError("fallback accepts only canonical h020-shared")
    if summary["inverse"]["convergence"]["practical_stationarity_gate"]:
        raise RuntimeError("canonical shared is stationary; fallback is unnecessary")
    seed = np.load(seed_path)
    controls = np.asarray(seed["controls"], dtype=float)
    stored_u = np.asarray(seed["displacement_free"], dtype=float)
    case = next(case for case in runner.cases() if case.name == "h020-shared")
    mesh = runner.build_mesh(case)
    cfg.output_dir.mkdir(parents=True)
    state, value, gradient = evaluate(case, controls, stored_u, cfg)
    u_delta = float(np.linalg.norm(state.u - stored_u, np.inf))
    if u_delta > 1e-8 or abs(value - summary["metrics"]["final"]["objective"]) > 1e-10:
        raise RuntimeError("canonical shared endpoint fails strict fallback preflight")
    np.savez_compressed(
        cfg.output_dir / "initial-seed.npz",
        controls=controls,
        displacement_free=state.u,
    )
    initial_seed = {
        "path": "initial-seed.npz",
        "controls_sha256": sha(controls),
        "displacement_sha256": sha(state.u),
    }
    accepted = [row(mesh, case, state, controls, value, gradient, 0, "strict_seed")]
    trials: list[dict[str, Any]] = []
    inverse_hessian = np.eye(controls.size)
    no_progress = 0
    while len(accepted) - 1 < cfg.max_accepted:
        current = accepted[-1]
        if (current["gradient_inf"] <= cfg.gradient_inf_tolerance and current["gradient_rms"] <= cfg.gradient_rms_tolerance and current["equilibrium_residual_rms"] <= cfg.forward_tolerance):
            termination = "physical_stationarity"
            break
        direction = -inverse_hessian @ gradient
        directional = float(gradient @ direction)
        if not math.isfinite(directional) or directional >= 0:
            inverse_hessian = np.eye(controls.size)
            direction, directional = -gradient, -float(gradient @ gradient)
        alpha, accepted_trial = 1.0, None
        while alpha >= cfg.alpha_floor and len(trials) < cfg.max_trials:
            candidate = controls + alpha * direction
            try:
                trial_state, trial_value, trial_gradient = evaluate(case, candidate, state.u, cfg)
                accepted_armijo = trial_value <= value + cfg.armijo * alpha * directional
                trials.append({"trial": len(trials), "alpha": alpha, "accepted": accepted_armijo, "objective": trial_value, "gradient_inf": float(np.linalg.norm(trial_gradient, np.inf)), "forward_iterations": int(trial_state.iterations), "min_det_f": float(trial_state.det_f.min()), "controls_sha256": sha(candidate), "displacement_sha256": sha(trial_state.u)})
                if accepted_armijo:
                    accepted_trial = (candidate, trial_state, trial_value, trial_gradient)
                    break
            except Exception as error:  # Trial failure is evidence, not a surrogate value.
                trials.append({"trial": len(trials), "alpha": alpha, "accepted": False, "error": repr(error), "controls_sha256": sha(candidate)})
            alpha *= 0.5
        if accepted_trial is None:
            no_progress += 1
            inverse_hessian = np.eye(controls.size)
            if no_progress >= cfg.max_no_progress_restarts or len(trials) >= cfg.max_trials:
                termination = "no_progress_restart_limit"
                break
            continue
        next_controls, next_state, next_value, next_gradient = accepted_trial
        step_vector, gradient_delta = next_controls - controls, next_gradient - gradient
        curvature = float(gradient_delta @ step_vector)
        if curvature > 1e-14 * np.linalg.norm(step_vector) * np.linalg.norm(gradient_delta):
            rho = 1.0 / curvature
            identity = np.eye(controls.size)
            v = identity - rho * np.outer(step_vector, gradient_delta)
            inverse_hessian = v @ inverse_hessian @ v.T + rho * np.outer(step_vector, step_vector)
        else:
            inverse_hessian = np.eye(controls.size)
        no_progress = 0
        controls, state, value, gradient = next_controls, next_state, next_value, next_gradient
        accepted.append(row(mesh, case, state, controls, value, gradient, len(accepted), "inverse_bfgs"))
    else:
        termination = "accepted_iteration_budget"
    with (cfg.output_dir / "accepted.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=accepted[0].keys()); writer.writeheader(); writer.writerows(accepted)
    fields = sorted({key for item in trials for key in item})
    with (cfg.output_dir / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(trials)
    stationary = termination == "physical_stationarity"
    failed_trials = sum("error" in item for item in trials)
    eligible = stationary and failed_trials <= cfg.failed_trial_eligibility_cap
    report = {"canonical_attempt": {"summary": {"path": str(summary_path), "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest()}, "final_state": {"path": str(seed_path), "sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest()}, "runner_source": {"path": str(RUNNER_PATH), "sha256": hashlib.sha256(RUNNER_PATH.read_bytes()).hexdigest()}, "stationary": False}, "fallback": {"method": "unbounded inverse-BFGS with accepted-u Armijo continuation", "bounds": None, "determinant_constraint": None, "skin": None, "stationarity": stationary, "termination": termination, "accepted_states": len(accepted), "trial_evaluations": len(trials), "failed_trials": failed_trials, "rejected_trials": sum(not item.get("accepted", False) for item in trials), "seed_strict_u_inf_delta": u_delta, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cost": {"strict_seed_forward_adjoint": 1, "trial_forward_evaluations_including_failed": len(trials), "trial_adjoint_evaluations": sum("error" not in item for item in trials), "aggregate_forward_evaluations": 1 + len(trials), "aggregate_adjoint_evaluations": 1 + sum("error" not in item for item in trials)}}, "initial_seed": initial_seed, "stationary_seed": None, "releases": {"eligible": eligible, "failed_trial_eligibility_cap": cfg.failed_trial_eligibility_cap, "rule": "release requires fallback stationarity and failed trials at or below the explicit cap; both use tiled controls, strict receipts, and fresh optimizer moments"}, "final": accepted[-1]}
    write(cfg.output_dir / "summary.json", report)
    if not stationary:
        return
    np.savez_compressed(
        cfg.output_dir / "stationary-seed.npz",
        controls=controls,
        displacement_free=state.u,
    )
    stationary_seed = np.load(cfg.output_dir / "stationary-seed.npz")
    if not (
        sha(stationary_seed["controls"]) == sha(controls)
        and sha(stationary_seed["displacement_free"]) == sha(state.u)
    ):
        raise RuntimeError("stationary seed verification failed")
    report["stationary_seed"] = {
        "path": "stationary-seed.npz",
        "controls_sha256": sha(controls),
        "displacement_sha256": sha(state.u),
        "verified": True,
    }
    write(cfg.output_dir / "summary.json", report)
    if not eligible:
        return
    release_cfg = runner.Config(
        output_dir=cfg.output_dir / "releases",
        validate_derivatives=False,
        forward_tolerance=1e-8,
        forward_max_iterations=cfg.forward_max_iterations,
    )
    shared_case = case
    per_cell = {
        item.name: item
        for item in runner.cases()
        if item.name in ("h020-shared-release", "h020-shared-release_zero_u")
    }
    release_seed = np.load(cfg.output_dir / "stationary-seed.npz")
    controls = np.asarray(release_seed["controls"], dtype=float)
    state_u = np.asarray(release_seed["displacement_free"], dtype=float)
    expanded = np.tile(controls, int(mesh.muscle.sum()))
    shared_strict, shared_receipt = runner.strict_observation(
        shared_case, controls, state_u, release_cfg
    )
    shared_u_strict, shared_u_receipt = runner.strict_observation(
        per_cell["h020-shared-release"], expanded, shared_strict.u, release_cfg
    )
    _zero_strict, zero_receipt = runner.strict_observation(
        per_cell["h020-shared-release_zero_u"], expanded, np.zeros(mesh.nfree), release_cfg
    )
    shared_u_deltas = {
        "u_inf": float(np.linalg.norm(shared_u_strict.u - shared_strict.u, np.inf)),
        "objective": abs(shared_u_receipt["objective"] - shared_receipt["objective"]),
        **{
            name: abs(shared_u_receipt[name] - shared_receipt[name])
            for name in ("min_det_f", "min_det_ainv", "min_det_g")
        },
    }
    if (
        shared_u_deltas["u_inf"] > 1e-8
        or shared_u_deltas["objective"] > 1e-10
        or max(shared_u_deltas[name] for name in ("min_det_f", "min_det_ainv", "min_det_g")) > 1e-8
    ):
        raise RuntimeError(f"strict tiled shared-u reproduction failed: {shared_u_deltas}")
    zero_branch_gap = {
        "u_inf": float(np.linalg.norm(_zero_strict.u - shared_u_strict.u, np.inf)),
        "objective": float(zero_receipt["objective"] - shared_u_receipt["objective"]),
        **{
            name: float(zero_receipt[name] - shared_u_receipt[name])
            for name in ("min_det_f", "min_det_ainv", "min_det_g")
        },
        "asserted": False,
    }
    release_reports = {}
    for name, initial_u, receipt, strict_state in (
        ("h020-shared-release", shared_u_strict.u, shared_u_receipt, shared_u_strict),
        ("h020-shared-release_zero_u", None, zero_receipt, _zero_strict),
    ):
        release_reports[name] = runner.run_case(
            per_cell[name],
            release_cfg,
            initial_controls=expanded,
            initial_u=initial_u,
            initialization="stationary fallback seed; fresh Adam moments and variance",
        )
        normal_first = release_reports[name]["metrics"]["initial"]
        normal_u = np.asarray(
            runner.core.pv.read(release_cfg.output_dir / name / "frames" / "step-0000.vtu").point_data["Displacement"],
            dtype=float,
        )[:, :2].ravel()[mesh.free]
        release_reports[name]["fallback_handoff"] = {
            "shared_strict": shared_receipt,
            "shared_u_or_zero_strict": receipt,
            "expanded_controls_sha256": sha(expanded),
            "normal_tolerance_first_row_vs_strict_preflight": {
                "normal_tolerance": release_cfg.forward_tolerance,
                "strict_tolerance": cfg.forward_tolerance,
                "u_inf_delta": float(np.linalg.norm(normal_u - strict_state.u, np.inf)),
                "objective_delta": float(normal_first["objective"] - receipt["objective"]),
                "min_det_deltas": {
                    key: float(normal_first[key] - receipt[key])
                    for key in ("min_det_f", "min_det_ainv", "min_det_g")
                },
                "asserted_branch_identity": False,
            },
        }
        runner.write_json(release_cfg.output_dir / name / "summary.json", release_reports[name])
    report["releases"].update({"strict_shared": shared_receipt, "strict_shared_u": shared_u_receipt, "strict_zero_u": zero_receipt, "strict_tiled_shared_u_reproduction": {"deltas": shared_u_deltas, "asserted": True}, "strict_zero_u_branch_gap": zero_branch_gap, "reports": release_reports, "handoff_forward_adjoint_evaluations": 3})
    report["fallback"]["cost"]["aggregate_forward_evaluations"] += 3
    report["fallback"]["cost"]["aggregate_adjoint_evaluations"] += 3
    for release in release_reports.values():
        cost = release["inverse"]["cost"]
        report["fallback"]["cost"]["aggregate_forward_evaluations"] += (
            cost["adam_objective_forward_adjoint_evaluations"]
            + cost["strict_refinement_forward_adjoint_evaluations"]
            + cost["lbfgs_actual_forward_evaluations"]
        )
        report["fallback"]["cost"]["aggregate_adjoint_evaluations"] += (
            cost["adam_objective_forward_adjoint_evaluations"]
            + cost["strict_refinement_forward_adjoint_evaluations"]
            + cost["lbfgs_actual_adjoint_evaluations"]
        )
    write(cfg.output_dir / "summary.json", report)


if __name__ == "__main__":
    cherries.main(main)
