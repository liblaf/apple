"""Freeze one face optimizer probe using the same checkpoint and true gradient."""

# ruff: noqa: PLR0915

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from continuation_helpers import (
    BASE,
    adam_metrics,
    initial_replay_metrics,
    load_checkpoint,
    projected_gradient_metrics,
    rms,
    tensor_rms_mpa,
    verify_parent_sources,
)
from experiment_profile import ProfileCometNoCommit
from face_physics import FacePhysics, configure
from tensor_controls import project

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
COMPLETED = False


class Config(BASE.Config):
    output_dir: Path = cherries.output("73-face-step-calibration", mkdir=True)
    resume: Path = HERE / "data/21-psd/optimizer-latest.pt"
    small_recovery_summary: Path
    reduced_eps: float
    physical_step_multiplier: float = 2.0
    projected_gradient_eta: float = 1.0


def run(cfg: Config) -> None:
    global COMPLETED  # noqa: PLW0603
    assert cfg.model == "tensor"
    assert cfg.smoothness_weight == cfg.magnitude_weight == cfg.rank_weight == 0
    assert 0 < cfg.reduced_eps < cfg.adam_eps
    assert cfg.physical_step_multiplier == 2.0
    assert cfg.projected_gradient_eta == 1.0
    small = json.loads(cfg.small_recovery_summary.read_text())
    assert small["status"] == "passed"
    assert small["selected"]["adam_eps"] == cfg.reduced_eps
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir())
    state = load_checkpoint(cfg.resume, cfg)
    assert int(state["step"]) == 64
    assert state["config"]["learning_rate"] == cfg.learning_rate
    assert state["config"]["adam_eps"] == cfg.adam_eps
    BASE.write_json(out / "config.json", cfg.model_dump(mode="json"))
    provenance = BASE.archive(out, cfg)
    parent = json.loads((cfg.resume.parent / "provenance.json").read_text())
    parent_summary = json.loads((cfg.resume.parent / "summary.json").read_text())
    verify_parent_sources(parent, provenance)
    configure()
    physics = FacePhysics(cfg.fixture, activation_model="tensor")
    assert np.array_equal(physics.ids, state["active_ids"])
    objective = BASE.Objective(physics, cfg)
    q = torch.nn.Parameter(state["q"].to(device="cuda").clone())
    initial_q = q.detach().clone()
    result = objective(q, np.asarray(state["u"]).copy())
    assert q.grad is not None
    initial_metrics = BASE.metrics(physics, initial_q, result, cfg)
    parent_endpoint = parent_summary["primary_endpoint"]
    replay = initial_replay_metrics(
        {**initial_metrics, "data_objective_mm2": result["data_objective_mm2"]},
        parent_endpoint,
    )
    BASE.write_json(out / "initial-replay.json", replay)
    assert replay["passed"], replay
    gradient = q.grad.detach().clone()
    maximum = cfg.stress_cap_mpa / cfg.stress_reference_mpa
    baseline_opt = torch.optim.Adam([q], lr=cfg.learning_rate, eps=cfg.adam_eps)
    baseline_opt.load_state_dict(copy.deepcopy(state["optimizer"]))
    initial_adam = adam_metrics(baseline_opt)
    moment = baseline_opt.state[q]
    gradient_sha256 = hashlib.sha256(gradient.cpu().numpy().tobytes()).hexdigest()
    moment_sha256 = {
        name: hashlib.sha256(moment[name].cpu().numpy().tobytes()).hexdigest()
        for name in ("exp_avg", "exp_avg_sq")
    }
    step = int(moment["step"]) + 1
    group = baseline_opt.param_groups[0]
    beta1, beta2 = group["betas"]
    assert group["weight_decay"] == 0
    assert not group["amsgrad"]
    assert not group["maximize"]
    mhat = (beta1 * moment["exp_avg"] + (1 - beta1) * gradient) / (1 - beta1**step)
    vhat = (beta2 * moment["exp_avg_sq"] + (1 - beta2) * gradient.square()) / (
        1 - beta2**step
    )
    sqrt_v = vhat.sqrt()
    baseline_direction = -mhat / (sqrt_v + cfg.adam_eps)
    reduced_direction = -mhat / (sqrt_v + cfg.reduced_eps)

    @torch.no_grad()
    def trial(
        direction: torch.Tensor, learning_rate: float
    ) -> tuple[torch.Tensor, dict]:
        candidate = initial_q + learning_rate * direction
        projection = project(candidate, maximum)
        delta = candidate - initial_q
        return delta, {
            "learning_rate": learning_rate,
            "actual_update_rms": rms(delta),
            "physical_update_rms_mpa": tensor_rms_mpa(delta, cfg.stress_reference_mpa),
            "unprojected_update_rms": rms(learning_rate * direction),
            "physical_update_frobenius_max_mpa": float(
                cfg.stress_reference_mpa * delta.square().sum(-1).sqrt().max()
            ),
            **projection,
        }

    baseline_delta, baseline = trial(baseline_direction, cfg.learning_rate)
    # Confirm the closed form against the installed Adam implementation.
    baseline_opt.step()
    project(q, maximum)
    closed_form_error = float(((q.detach() - initial_q) - baseline_delta).abs().max())
    assert closed_form_error < 1e-12
    target = cfg.physical_step_multiplier * baseline["physical_update_rms_mpa"]
    assert target > 0
    lower, upper = 0.0, cfg.learning_rate
    trials: list[dict] = []
    _, entry = trial(reduced_direction, upper)
    trials.append(entry)
    if entry["physical_update_rms_mpa"] < target:
        message = (
            "The reduced-epsilon projected step is not bracketed below the "
            "baseline learning-rate cap."
        )
        raise RuntimeError(message)
    for _ in range(30):
        middle = 0.5 * (lower + upper)
        _, entry = trial(reduced_direction, middle)
        trials.append(entry)
        if entry["physical_update_rms_mpa"] < target:
            lower = middle
        else:
            upper = middle
    selected_lr = 0.5 * (lower + upper)
    selected_delta, selected = trial(reduced_direction, selected_lr)
    trials.append(selected)
    ordered = sorted(trials, key=lambda item: item["learning_rate"])
    for left, right in itertools.pairwise(ordered):
        if right["physical_update_rms_mpa"] + 1e-12 < left["physical_update_rms_mpa"]:
            message = "Projected physical-step calibration is nonmonotone."
            raise RuntimeError(message)
    assert abs(selected["physical_update_rms_mpa"] / target - 1) < 1e-5
    selected["adam_eps"] = cfg.reduced_eps
    selected["update_cosine_with_baseline"] = float(
        (selected_delta * baseline_delta).sum()
        / (selected_delta.square().sum() * baseline_delta.square().sum()).sqrt()
    )
    selected["actual_physical_step_multiplier"] = (
        selected["physical_update_rms_mpa"] / baseline["physical_update_rms_mpa"]
    )
    replay_q = torch.nn.Parameter(initial_q.detach().clone())
    replay_opt = torch.optim.Adam([replay_q], lr=selected_lr, eps=cfg.reduced_eps)
    replay_opt.load_state_dict(copy.deepcopy(state["optimizer"]))
    replay_opt.param_groups[0]["lr"] = selected_lr
    replay_opt.param_groups[0]["eps"] = cfg.reduced_eps
    replay_q.grad = gradient.detach().clone()
    replay_opt.step()
    project(replay_q, maximum)
    replay_delta = replay_q.detach() - initial_q
    replay_physical = tensor_rms_mpa(replay_delta, cfg.stress_reference_mpa)
    replay_max_abs = float((replay_delta - selected_delta).abs().max())
    replay_relative_physical_error = abs(
        replay_physical / selected["physical_update_rms_mpa"] - 1
    )
    assert replay_max_abs < 1e-10
    assert replay_relative_physical_error < 1e-5
    selected["installed_adam_replay"] = {
        "control_max_abs_error": replay_max_abs,
        "relative_physical_update_error": replay_relative_physical_error,
        "adam_step_before": initial_adam["adam_step"],
        "adam_step_after": int(replay_opt.state[replay_q]["step"]),
    }
    baseline["adam_eps"] = cfg.adam_eps
    # The initial state and gradient belong to the unmodified parent, not a trial.
    np.savez_compressed(
        out / "initial-gradient.npz",
        q=initial_q.cpu().numpy(),
        gradient=gradient.cpu().numpy(),
        u=result["u"],
        step=np.asarray(state["step"]),
    )
    summary = {
        "status": "frozen_before_probe_runs",
        "parent_checkpoint": {
            "path": str(cfg.resume.resolve()),
            "sha256": BASE.sha256(cfg.resume),
        },
        "small_recovery": {
            "path": str(cfg.small_recovery_summary.resolve()),
            "sha256": BASE.sha256(cfg.small_recovery_summary),
        },
        "parent_global_step": int(state["step"]),
        "rule": "Use the small-model selected epsilon; choose learning rate by bisection so the next projected physical stress step is the declared multiple of the baseline step at the same true face gradient and Adam moments. No trial displacement or final fit outcome is used.",
        "physical_step_multiplier": cfg.physical_step_multiplier,
        "physical_update_definition": "uniform-cell RMS Frobenius norm of ΔQ in MPa",
        "baseline": baseline,
        "selected": selected,
        "closed_form_vs_installed_adam_max_abs": closed_form_error,
        "bisection_trials": trials,
        "initial_metrics": initial_metrics,
        "initial_replay": replay,
        "gradient_rms": rms(gradient),
        "gradient_sha256": gradient_sha256,
        "adam_moment_sha256": moment_sha256,
        "initial_adam": initial_adam,
        "projected_gradient": projected_gradient_metrics(
            initial_q, gradient, maximum, cfg.projected_gradient_eta
        ),
        "forward": result["forward"],
        "adjoint": result["adjoint"],
        "initial_control_sha256": hashlib.sha256(
            initial_q.cpu().numpy().tobytes()
        ).hexdigest(),
        "provenance": provenance,
    }
    BASE.write_json(out / "summary.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "baseline",
                    "selected",
                    "closed_form_vs_installed_adam_max_abs",
                )
            }
        ),
        flush=True,
    )
    cherries.log_metrics({"baseline": baseline, "selected": selected})
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(run, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
