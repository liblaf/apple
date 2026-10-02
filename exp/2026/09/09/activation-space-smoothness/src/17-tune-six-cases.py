"""Extend validated settings to learned axes and a common physical initialization."""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import torch
from activation_controls import common_initial_controls, learned_axis_raw6, project_psd_
from experiment_profile import ProfileCometNoCommit
from study_metrics import StudyMetrics
from study_physics import FacePhysics, ForwardConvergenceError, configure
from study_runner import (
    FIXTURE,
    GROUP,
    Objective,
    StudySolveError,
    archive_runtime,
    control_z,
    digest,
    volume_metrics,
    write_json,
    write_trace,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)
DONE = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = FIXTURE
    previous_settings: Path = GROUP / "data/15-calibration/summary.json"
    controls_validation: Path = GROUP / "data/11-learned-axis-validation/summary.json"
    output_dir: Path = GROUP / "data/17-six-case-calibration"
    pilot_steps: int = 8
    initialization_seed: int = 20260909
    initial_strength: float = 0.001


def main(cfg: Config) -> None:
    global DONE
    assert cfg.pilot_steps == 8 and cfg.initial_strength > 0
    old = json.loads(cfg.previous_settings.read_text())
    assert old["status"] == "frozen_before_primary_runs"
    assert json.loads(cfg.controls_validation.read_text())["status"] == "passed"
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Refuse to overwrite calibration: {out}"
    configure()
    start = time.perf_counter()
    diagnostics = StudyMetrics(fixture=cfg.fixture)
    models = ("raw6", "tensor", "learned-axis")
    settings = {
        **{k: old[k] for k in ("adam_eps", "betas", "smooth_length_m", "inputs")},
        "status": "tuning_in_progress",
        "pilot_steps": cfg.pilot_steps,
        "skin_enabled": False,
        "magnitude_weight": 0.0,
        "rank_weight": 0.0,
        "upper_stress_cap": None,
        "initialization_seed": cfg.initialization_seed,
        "initial_strength": cfg.initial_strength,
        "initialization": "Identical B=I+s0 ff^T across models; one seeded isotropic unit axis per muscle label; fresh Adam and zero displacement solve seed. Each cell's controls are then optimized independently.",
        "learned_axis_model": "B=I+vv^T; Z=(2+||v||^2)vv^T; v has three free coordinates per active tetrahedron. Transverse active lengths remain one.",
        "lr_selection_rule": "Reuse the previously calibrated Raw6 and tensor rates after a fresh 8-update check at common initialization. Match learned-axis initial physical Delta Z RMS to Raw6 rates .3, .9, 3.0; choose the smallest rate within 5% of best stable fitting progress. Probe the physical equivalent of Raw6 9.0 if the upper edge has the best valid progress.",
        "weight_selection_rule": "Recheck the previous shared weight in all three models at the common start. Require solver-valid stable objective, >=80% fit-progress retention, >=90% target-projection retention, and reduced Z smoothness in every model. If it fails, test the previous weight divided by 4, then 16 in all three; select the largest passing weight. Fail if none passes.",
        "limits": "Pilots establish short-run effectiveness, not stationarity. Physical starting state is shared, but optimizer coordinates differ. The learned axes are effective directions, not validated anatomical fibers.",
        "previous_settings": {
            "path": str(cfg.previous_settings),
            "sha256": digest(cfg.previous_settings),
        },
        "controls_validation": {
            "path": str(cfg.controls_validation),
            "sha256": digest(cfg.controls_validation),
        },
    }
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    write_json(out / "selection-protocol.json", settings)
    provenance = archive_runtime(out, cfg.fixture)
    write_json(out / "provenance.json", provenance)
    for name, record in old["inputs"].items():
        assert digest(cfg.fixture / name) == record["sha256"]

    def initial_q(physics, model):
        return torch.nn.Parameter(
            common_initial_controls(
                physics.region_t,
                model=model,
                seed=cfg.initialization_seed,
                strength=cfg.initial_strength,
            )
        )

    initial = {}
    gradients = {}
    controls = {}
    for model in models:
        physics = FacePhysics(cfg.fixture, activation_model=model, skin_factor=0.0)
        q = initial_q(physics, model)
        initial[model] = Objective(physics)(q, np.zeros_like(physics.points))
        gradients[model] = q.grad.detach().clone()
        controls[model] = q.detach().clone()
        mass = torch.as_tensor(physics.volumes / physics.volumes.sum())
        del physics, q
        gc.collect()
        torch.cuda.empty_cache()
    reference = initial["raw6"]
    mapped = controls["learned-axis"].clone().requires_grad_()
    predicted = torch.autograd.grad(
        (learned_axis_raw6(mapped) * gradients["raw6"]).sum(), mapped
    )[0]
    gradient_error = float(
        (predicted - gradients["learned-axis"]).norm()
        / gradients["learned-axis"].norm()
    )
    assert gradient_error < 0.005, gradient_error
    initial_checks = {"learned_axis_chain_relative_error": gradient_error, "models": {}}
    for model in models:
        geometry_error = float(
            1000
            * np.linalg.norm(initial[model]["u"] - reference["u"])
            / np.sqrt(len(reference["u"]))
        )
        z_error = float(np.max(np.abs(initial[model]["Z"] - reference["Z"])))
        assert geometry_error < 0.001 and z_error < 1e-12
        initial_checks["models"][model] = {
            "geometry_rms_difference_mm": geometry_error,
            "Z_max_abs_difference": z_error,
            "fit_rms_mm": float(np.sqrt(3 * initial[model]["objective_mm2"])),
            "forward": initial[model]["forward"],
            "adjoint": initial[model]["adjoint"],
        }
    write_json(out / "initial-equivalence.json", initial_checks)

    def rms(delta):
        return float((mass * delta.square().sum((-1, -2))).sum().sqrt())

    matches = []

    def axis_rate_for(base_lr):
        base_q = controls["raw6"]
        base_g = gradients["raw6"]
        target = rms(
            control_z(
                base_q - base_lr * base_g / (base_g.abs() + old["adam_eps"]), "raw6"
            )
            - control_z(base_q, "raw6")
        )
        axis_q = controls["learned-axis"]
        axis_g = gradients["learned-axis"]
        direction = -axis_g / (axis_g.abs() + old["adam_eps"])
        z0 = control_z(axis_q, "learned-axis")

        def delta_for(lr):
            return rms(control_z(axis_q + lr * direction, "learned-axis") - z0)

        lo, hi = 0.0, 1.0
        while delta_for(hi) < target:
            hi *= 2
            assert hi < 1e8
        for _ in range(48):
            mid = (lo + hi) / 2
            if delta_for(mid) < target:
                lo = mid
            else:
                hi = mid
        lr = (lo + hi) / 2
        actual = delta_for(lr)
        assert abs(actual / target - 1) < 1e-10
        matches.append(
            {
                "raw6_rate": base_lr,
                "learned_axis_rate": lr,
                "raw6_delta_Z_rms": target,
                "learned_axis_delta_Z_rms": actual,
            }
        )
        write_json(out / "initial-step-matching.json", matches)
        return lr

    trials = []

    def trial(model, lr, weight, name):
        trial_out = out / name
        trial_out.mkdir()
        physics = FacePhysics(cfg.fixture, activation_model=model, skin_factor=0.0)
        objective = Objective(physics, smoothness_weight=weight)
        q = initial_q(physics, model)
        optimizer = torch.optim.Adam(
            [q], lr=lr, eps=old["adam_eps"], betas=(0.9, 0.999)
        )
        assert not optimizer.state
        seed = np.zeros_like(physics.points)
        previous_u, previous_z = seed, None
        trace, failure, endpoint = [], None, None
        tick = time.perf_counter()
        try:
            for step in range(cfg.pilot_steps + 1):
                result = objective(q, seed)
                row = {
                    "step": step,
                    "model": model,
                    "learning_rate": lr,
                    **{
                        k: result[k]
                        for k in (
                            "smoothness_weight",
                            "objective_mm2",
                            "objective_total",
                            "smoothness",
                            "gradient_rms",
                            "fit_gradient_rms",
                            "regularizer_gradient_rms",
                            "objective_s",
                        )
                    },
                    "forward_steps": result["forward"]["steps"],
                    "forward_success": True,
                    "adjoint_success": True,
                    **diagnostics.evaluate(
                        result["u"], result["Z"], previous_u, previous_z
                    ),
                    **volume_metrics(physics, result["u"]),
                }
                trace.append(row)
                write_trace(trial_out / "trace.csv", trace)
                with (trial_out / "solver-receipts.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "step": step,
                                "forward": result["forward"],
                                "adjoint": result["adjoint"],
                            }
                        )
                        + "\n"
                    )
                LOG.info(
                    "%s %d/%d: lr %.6g, lambda %.6g, fit %.4f mm, S %.5g; %.1f s",
                    name,
                    step,
                    cfg.pilot_steps,
                    lr,
                    weight,
                    row["fit_rms_mm"],
                    row["smoothness"],
                    row["objective_s"],
                )
                cherries.log_metrics(
                    {
                        f"{name}/fit_rms_mm": row["fit_rms_mm"],
                        f"{name}/smoothness": row["smoothness"],
                    },
                    step=step,
                )
                if step == cfg.pilot_steps:
                    endpoint = {
                        "q": q.detach().cpu().numpy(),
                        "Z": result["Z"],
                        "u": result["u"],
                    }
                    np.savez_compressed(trial_out / "endpoint.npz", **endpoint)
                    break
                previous_u, previous_z, seed = result["u"], result["Z"], result["u"]
                optimizer.step()
                if model == "tensor":
                    project_psd_(q)
        except (StudySolveError, ForwardConvergenceError) as error:
            failure = {
                "type": type(error).__name__,
                "message": str(error),
                "step": step,
                "forward": getattr(physics, "last_forward", None),
            }
            np.savez_compressed(
                trial_out / "failure-controls.npz",
                q=q.detach().cpu().numpy(),
                Z=control_z(q.detach(), model).cpu().numpy(),
                step=np.array(step),
            )
            LOG.error("Rejected solver-invalid pilot %s: %s", name, error)
        complete = endpoint is not None
        progress = (
            trace[0]["objective_mm2"] - trace[-1]["objective_mm2"] if complete else None
        )
        key = "objective_total" if weight else "objective_mm2"
        stable = bool(
            complete
            and progress > 0
            and trace[-1][key] <= 1.01 * min(r[key] for r in trace[1:])
        )
        record = {
            "name": name,
            "model": model,
            "learning_rate": lr,
            "smoothness_weight": weight,
            "status": "completed" if complete else "rejected_solver_failure",
            "stable": stable,
            "progress": progress,
            "initial": trace[0] if trace else None,
            "final": trace[-1] if trace else None,
            "failure": failure,
            "elapsed_s": time.perf_counter() - tick,
        }
        write_json(trial_out / "summary.json", record)
        trials.append(record)
        write_json(out / "pilots.json", trials)
        del physics, objective, q, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return record

    selected = {}
    for model in ("raw6", "tensor"):
        selected[model] = trial(
            model, old["learning_rates"][model], 0.0, f"confirm-lr-{model}"
        )
        assert selected[model]["stable"], (
            f"Prior rate did not pass common-start validation: {model}"
        )
    candidates = [
        trial("learned-axis", axis_rate_for(rate), 0.0, f"lr-learned-axis-{i}")
        for i, rate in enumerate((0.3, 0.9, 3.0))
    ]
    valid = [r for r in candidates if r["stable"]]
    assert valid, "No valid learned-axis rate"
    if max(valid, key=lambda r: r["progress"]) is candidates[-1]:
        candidates.append(
            trial("learned-axis", axis_rate_for(9.0), 0.0, "lr-learned-axis-3")
        )
        valid = [r for r in candidates if r["stable"]]
    best_gain = max(r["progress"] for r in valid)
    selected["learned-axis"] = min(
        (r for r in valid if r["progress"] >= 0.95 * best_gain),
        key=lambda r: r["learning_rate"],
    )
    weights = []
    chosen = None
    for index, weight in enumerate(
        old["smoothness_weight"] / np.array([1.0, 4.0, 16.0])
    ):
        records = {
            model: trial(
                model,
                selected[model]["learning_rate"],
                float(weight),
                f"weight-{index}-{model}",
            )
            for model in models
        }
        effects = {}
        for model, record in records.items():
            reference = selected[model]
            if record["status"] == "completed":
                assert (
                    reference["final"]["target_projection"] > 0
                    and reference["final"]["smoothness"] > 0
                )
                effects[model] = {
                    "fit_progress_retention": record["progress"]
                    / reference["progress"],
                    "target_projection_retention": record["final"]["target_projection"]
                    / reference["final"]["target_projection"],
                    "smoothness_reduction_fraction": 1
                    - record["final"]["smoothness"] / reference["final"]["smoothness"],
                }
        eligible = (
            len(effects) == 3
            and all(r["stable"] for r in records.values())
            and min(e["fit_progress_retention"] for e in effects.values()) >= 0.80
            and min(e["target_projection_retention"] for e in effects.values()) >= 0.90
            and min(e["smoothness_reduction_fraction"] for e in effects.values()) > 0
        )
        candidate = {
            "smoothness_weight": float(weight),
            "eligible": bool(eligible),
            "effects": effects,
            "pilots": {model: r["name"] for model, r in records.items()},
        }
        weights.append(candidate)
        write_json(out / "weight-comparison.json", weights)
        if eligible:
            chosen = candidate
            break
    assert chosen is not None, (
        "No shared positive weight passed; inspect rejected pilots"
    )
    settings.update(
        {
            "status": "frozen_before_primary_runs",
            "learning_rates": {
                model: r["learning_rate"] for model, r in selected.items()
            },
            "selected_lr_pilots": {model: r["name"] for model, r in selected.items()},
            "smoothness_weight": chosen["smoothness_weight"],
            "selected_weight_pilots": chosen["pilots"],
            "selected_weight_effects": chosen["effects"],
            "initial_equivalence": initial_checks,
            "lr_candidate_matching": matches,
            "elapsed_s": time.perf_counter() - start,
        }
    )
    write_json(out / "summary.json", settings)
    for name in (
        "config.json",
        "selection-protocol.json",
        "initial-equivalence.json",
        "initial-step-matching.json",
        "provenance.json",
        "pilots.json",
        "weight-comparison.json",
        "summary.json",
    ):
        cherries.log_output(out / name)
    LOG.info(
        "Frozen six-case rates %s, shared lambda %.8g",
        settings["learning_rates"],
        settings["smoothness_weight"],
    )
    DONE = True


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
    if not DONE:
        raise SystemExit(1)
