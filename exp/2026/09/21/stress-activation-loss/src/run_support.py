"""Projected Adam with approximate-solve receipts and bounded failure recovery."""
# ruff: noqa: ANN001, C901, PLR0912, PLR0915

from __future__ import annotations

import copy
import csv
import datetime
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from activation_models import (
    STRESS_REF_MPA,
    controls_from_matrix,
    initialize_learned_zero_amplitude_axes,
    learned_controls_from_fixed,
    matrices,
    project_,
)
from stress_physics import ForwardConvergenceError
from stress_study import ROOT, InvalidEquilibriumError

from liblaf.apple.inverse import ImplicitNumericalError, ImplicitSolveError

LOG = logging.getLogger(__name__)
MODES = ("symmetric6", "psd6", "rankone_fixed", "rankone_learned")
NUMERICAL_FAILURES = (
    ImplicitSolveError,
    InvalidEquilibriumError,
    ForwardConvergenceError,
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n"
    )
    temporary.replace(path)


def receipt(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def archive(out):
    """Archive only numerical runner sources, never reports or post-processing."""
    source = Path(__file__).parent
    repository = ROOT.parent
    files = {
        source / name
        for name in (
            "activation_models.py",
            "run_support.py",
            "stress_material.py",
            "stress_physics.py",
            "stress_study.py",
            "stress_regularization.py",
            "12-freeze-validation.py",
            "11-finalize-physics-validation.py",
            "10-validate-physics.py",
            "20-calibrate-smoothness.py",
            "40-run-chains.py",
            "41-fit-l2-unrestricted.py",
        )
        if (source / name).exists()
    }
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__name__", "")
        locations = (
            getattr(module, "__file__", None),
            getattr(getattr(module, "__spec__", None), "origin", None),
        )
        path = next(
            (
                Path(location).resolve()
                for location in locations
                if isinstance(location, str)
                and Path(location).is_absolute()
                and Path(location).suffix == ".py"
                and Path(location).is_file()
            ),
            None,
        )
        relevant_package = name.startswith(("liblaf.apple", "liblaf.peach"))
        if relevant_package:
            assert path is not None, (
                f"Cannot archive actual source for loaded numerical module {name}: "
                f"{locations}"
            )
        if path is not None:
            if path.is_relative_to(source) and path.name.startswith(
                ("05-", "06-", "30-", "50-", "60-", "80-")
            ):
                continue
            if (
                (
                    path.is_relative_to(source)
                    and path.name
                    not in {"05-validate-activation.py", "30-build-report.py"}
                )
                or path.is_relative_to(ROOT / "src")
                or path.is_relative_to(ROOT / "exp")
                or relevant_package
                or path.is_relative_to(repository / "peach")
            ):
                files.add(path)
    result = {}
    for path in sorted(files):
        assert path.is_file(), path
        dest = out / "sources" / path.relative_to(repository)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        result[str(path.resolve())] = {**receipt(path), "snapshot": str(dest.resolve())}
    write_json(out / "sources.json", result)
    return result


def verify_sources(records):
    for value in records.values():
        assert receipt(value["path"])["sha256"] == value["sha256"], value["path"]
        assert receipt(value["snapshot"])["sha256"] == value["sha256"], value[
            "snapshot"
        ]


def metrics(result):
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in result.items()
        if isinstance(value, (int, float, bool, str, np.generic))
    }


def append_jsonl(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, allow_nan=False, default=_json_default) + "\n")


def release_zero_axes(q, mode, result, optimizer):
    """Change the chart at zero stress, then pull back the same tensor gradient."""
    if mode != "rankone_learned":
        return 0
    updated = initialize_learned_zero_amplitude_axes(q, result["tensor_gradient"])
    changed = (updated != q.detach()).any(dim=-1)
    count = int(changed.sum())
    if count:
        with torch.no_grad():
            q.copy_(updated)
            for value in optimizer.state.get(q, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == q.shape:
                    value[changed] = 0
        q.grad = torch.autograd.grad(
            matrices(q, mode), q, grad_outputs=result["tensor_gradient"]
        )[0].detach()
        result["gradient"] = q.grad.clone()
        result["gradient_rms"] = float(q.grad.square().mean().sqrt())
    return count


def save_state(path, q, axes, result, mode, step, activation_model="stress"):
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    matrix = matrices(q, mode, axes).detach().cpu().numpy()
    state = {
        "q": q.detach().cpu().numpy(),
        "fixed_axes": np.empty((0, 3)) if axes is None else axes.detach().cpu().numpy(),
        "u": result["u"],
        "mode": mode,
        "step": step,
        "activation_model": activation_model,
        "solver_valid": result["solver_valid"],
    }
    if activation_model == "stress":
        state["Qhat"] = matrix
        state["stress_reference_MPa"] = STRESS_REF_MPA
        state["control_matrix_kind"] = "normalized_active_stress"
    else:
        assert activation_model == "strain"
        state["S"] = matrix
        state["B"] = matrix + np.eye(3)
        state["control_matrix_kind"] = "dimensionless_active_strain"
    np.savez_compressed(
        temporary,
        **state,
    )
    temporary.replace(path)


def update_site(site, stage_id, stage_summary, rows):
    path = site / "status.json"
    state = json.loads(path.read_text())
    state["updated_at"] = datetime.datetime.now().astimezone().isoformat()
    matched = False
    for stage in state["stages"]:
        if stage["id"] == stage_id:
            matched = True
            stage.update(
                {
                    "status": stage_summary["status"],
                    "step": stage_summary["last_step"],
                    "budget": stage_summary["budget"],
                    "metrics": stage_summary["last_metrics"] or {},
                    "history": rows,
                    "failure": stage_summary["failure"],
                }
            )
    if not matched:
        state["phase"] = (
            f"Smoothness calibration: {stage_id}, update "
            f"{stage_summary['last_step']}/{stage_summary['budget']}"
        )
        state["calibration_progress"] = {
            "id": stage_id,
            "status": stage_summary["status"],
            "step": stage_summary["last_step"],
            "budget": stage_summary["budget"],
            "metrics": stage_summary["last_metrics"] or {},
            "history": rows,
            "failure": stage_summary["failure"],
        }
    write_json(path, state)


def _check_usable(q, result):
    """Unconverged is usable; nonfinite quantities are not."""
    finite = (
        bool(torch.isfinite(q).all())
        and np.isfinite(result["objective"])
        and np.isfinite(result["u"]).all()
        and bool(torch.isfinite(result["gradient"]).all())
        and bool(torch.isfinite(result["tensor_gradient"]).all())
    )
    if not finite:
        message = "Nonfinite inverse state or gradient"
        raise ImplicitNumericalError(message)


def _save_optimizer(
    folder, q, axes, optimizer, result, mode, step, attempt, activation_model
):
    temporary = folder / "optimizer-latest.tmp.pt"
    torch.save(
        {
            "q": q.detach().cpu(),
            "fixed_axes": None if axes is None else axes.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "attempted_steps": attempt,
            "mode": mode,
            "activation_model": activation_model,
            "u": result["u"],
            "solver_valid": result["solver_valid"],
        },
        temporary,
    )
    temporary.replace(folder / "optimizer-latest.pt")


def run_stage(
    study,
    folder,
    mode,
    normal_weight,
    smooth_weight,
    Q_initial,
    seed,
    *,
    steps=200,
    learning_rate=0.05,
    adam_eps=1e-8,
    site=None,
    stage_id=None,
    parent_axes=None,
    resume_checkpoint=None,
    activation_model="stress",
):
    """Spend a bounded Adam attempt budget, allowing finite approximate gradients.

    Objective increases and nonconvergence do not reject updates. An unusable
    proposal restores controls and moments, halves the learning rate, and retries
    on the next iteration. Only solver-converged states enter best-valid.npz.
    """
    assert steps >= 0
    assert activation_model in {"stress", "strain"}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    controls, axes = controls_from_matrix(Q_initial, mode)
    if parent_axes is not None:
        assert mode == "rankone_learned"
        controls = learned_controls_from_fixed(controls[..., :1], parent_axes)
    q = torch.nn.Parameter(controls.detach().clone())
    axes = None if axes is None else axes.detach().clone()
    optimizer = torch.optim.Adam(
        [q], lr=learning_rate, eps=adam_eps, betas=(0.9, 0.999)
    )
    start_step = 0
    resumed = None
    if resume_checkpoint is not None:
        resumed = torch.load(resume_checkpoint, weights_only=False)
        assert resumed["mode"] == mode
        assert resumed.get("activation_model", "stress") == activation_model
        assert int(resumed["step"]) < steps
        with torch.no_grad():
            q.copy_(resumed["q"].to(dtype=q.dtype, device=q.device))
        if axes is not None and resumed["fixed_axes"] is not None:
            axes.copy_(resumed["fixed_axes"].to(dtype=axes.dtype, device=axes.device))
        optimizer.load_state_dict(resumed["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        seed = np.asarray(resumed["u"]).copy()
        start_step = int(resumed["step"])
    initial_Q = matrices(q, mode, axes).detach().clone()
    write_json(
        folder / "initialization.json",
        {
            "mode": mode,
            "activation_model": activation_model,
            "control_matrix_kind": (
                "normalized_active_stress"
                if activation_model == "stress"
                else "dimensionless_active_strain"
            ),
            "normal_weight": normal_weight,
            "smooth_weight": smooth_weight,
            "fresh_adam": resumed is None,
            "resume_checkpoint": None
            if resume_checkpoint is None
            else receipt(resume_checkpoint),
            "resume_step": start_step if resumed is not None else None,
            "resume_learning_rate": learning_rate if resumed is not None else None,
            "solve_policy": "finite approximate gradients allowed; convergence recorded",
            "initial_tensor_projection_frobenius_rms": float(
                torch.sqrt((initial_Q - Q_initial).square().sum((-2, -1)).mean())
            ),
            (
                "zero_stress_cells"
                if activation_model == "stress"
                else "zero_strain_cells"
            ): int(torch.count_nonzero(initial_Q.square().sum((-2, -1)) == 0)),
        },
    )
    rows = []
    result = None
    status = "running"
    failure = None
    last_failure = None
    start = time.perf_counter()
    attempted_steps = optimizer_updates = skipped_steps = approximate_steps = 0
    best_objective = best_valid_objective = float("inf")
    best_valid_step = None

    def summarize():
        return {
            "status": status,
            "last_step": rows[-1]["step"] if rows else None,
            "start_step": start_step,
            "resumed": resumed is not None,
            "attempted_steps": attempted_steps,
            "attempts_this_run": attempted_steps - start_step + 1,
            "optimizer_updates": optimizer_updates,
            "skipped_steps": skipped_steps,
            "approximate_steps": approximate_steps,
            "best_valid_step": best_valid_step,
            "budget": steps,
            "initial_metrics": rows[0] if rows else None,
            "last_metrics": rows[-1] if rows else None,
            "failure": failure,
            "last_numerical_failure": last_failure,
            "elapsed_seconds": time.perf_counter() - start,
        }

    def persist_summary():
        summary = summarize()
        write_json(folder / "summary.json", summary)
        if site is not None:
            update_site(site, stage_id, summary, rows)

    try:
        with study.physics.approximate_solves():
            for step in range(start_step, steps + 1):
                attempted_steps = step
                old_q = q.detach().clone()
                old_optimizer = copy.deepcopy(optimizer.state_dict())
                has_proposal = result is not None
                try:
                    if has_proposal:
                        q.grad = result["gradient"].clone()
                        optimizer.step()
                        q.grad = None
                        if not bool(torch.isfinite(q).all()):
                            message = "Adam returned nonfinite controls"
                            raise ImplicitNumericalError(message)
                        with torch.no_grad():
                            project_(q, mode)
                    candidate = study.evaluate(
                        q, mode, axes, seed, normal_weight, smooth_weight, backward=True
                    )
                    assert (
                        candidate.get("activation_model", activation_model)
                        == activation_model
                    )
                    _check_usable(q, candidate)
                except NUMERICAL_FAILURES as error:
                    with torch.no_grad():
                        q.copy_(old_q)
                    q.grad = None
                    optimizer.load_state_dict(old_optimizer)
                    # No gradient is invented. Keep moments intact, and reduce only
                    # an unusable proposal to avoid deterministically repeating it.
                    if has_proposal:
                        for group in optimizer.param_groups:
                            group["lr"] *= 0.5
                    if result is not None:
                        _save_optimizer(
                            folder,
                            q,
                            axes,
                            optimizer,
                            result,
                            mode,
                            rows[-1]["step"],
                            step,
                            activation_model,
                        )
                    skipped_steps += 1
                    last_failure = {
                        "step": step,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                    append_jsonl(
                        folder / "proposals.jsonl",
                        {
                            **last_failure,
                            "accepted": False,
                            "reason": "unusable_solve_or_gradient",
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                    )
                    LOG.warning(
                        "%s attempt %d/%d skipped: %s", folder.name, step, steps, error
                    )
                    persist_summary()
                    continue
                candidate["zero_amplitude_axis_updates"] = release_zero_axes(
                    q, mode, candidate, optimizer
                )
                candidate["direction_kind"] = "adam" if has_proposal else "initial"
                if has_proposal:
                    optimizer_updates += 1
                if not candidate["solver_valid"]:
                    approximate_steps += 1
                append_jsonl(
                    folder / "proposals.jsonl",
                    {
                        "step": step,
                        "accepted": True,
                        "reason": "usable_gradient",
                        "solver_valid": candidate["solver_valid"],
                        "objective": candidate["objective"],
                        "objective_increased": result is not None
                        and candidate["objective"] > result["objective"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                )
                result = candidate
                seed = result["u"].copy()
                row = {
                    "step": step,
                    **metrics(result),
                    "elapsed_seconds": time.perf_counter() - start,
                }
                rows.append(row)
                save_state(
                    folder / "last.npz",
                    q,
                    axes,
                    result,
                    mode,
                    step,
                    activation_model,
                )
                if len(rows) == 1:
                    save_state(
                        folder / "initial-state.npz",
                        q,
                        axes,
                        result,
                        mode,
                        step,
                        activation_model,
                    )
                if step % 50 == 0 or step == steps:
                    save_state(
                        folder / f"step-{step:04d}.npz",
                        q,
                        axes,
                        result,
                        mode,
                        step,
                        activation_model,
                    )
                _save_optimizer(
                    folder,
                    q,
                    axes,
                    optimizer,
                    result,
                    mode,
                    step,
                    step,
                    activation_model,
                )
                if result["objective"] < best_objective:
                    best_objective = result["objective"]
                    save_state(
                        folder / "best-available.npz",
                        q,
                        axes,
                        result,
                        mode,
                        step,
                        activation_model,
                    )
                if (
                    result["solver_valid"]
                    and result["objective"] < best_valid_objective
                ):
                    best_valid_objective = result["objective"]
                    best_valid_step = step
                    save_state(
                        folder / "best-valid.npz",
                        q,
                        axes,
                        result,
                        mode,
                        step,
                        activation_model,
                    )
                    shutil.copy2(
                        folder / "optimizer-latest.pt",
                        folder / "optimizer-best-valid.pt",
                    )
                with (folder / "trace.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=sorted(set().union(*(r.keys() for r in rows)))
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                append_jsonl(
                    folder / "solver-receipts.jsonl",
                    {
                        "step": step,
                        "solver_valid": result["solver_valid"],
                        "forward": result["forward"],
                        "adjoint": result["adjoint"],
                    },
                )
                persist_summary()
                LOG.info(
                    "%s attempt %d/%d objective=%.6g solver_valid=%s",
                    folder.name,
                    step,
                    steps,
                    result["objective"],
                    result["solver_valid"],
                )
        status = (
            "completed_budget_not_convergence_certified"
            if rows
            else "exhausted_without_usable_state"
        )
    except Exception as error:
        status = "failed"
        failure = {"type": type(error).__name__, "message": str(error)}
        persist_summary()
        raise
    persist_summary()
    return summarize()
