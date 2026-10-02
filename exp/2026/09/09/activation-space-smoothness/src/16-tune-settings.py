"""Choose effective Adam rates and one shared smoothness weight using discarded pilots."""

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
from activation_controls import baseline_z, project_psd_
from experiment_profile import ProfileCometNoCommit
from study_metrics import StudyMetrics
from study_physics import FacePhysics, ForwardConvergenceError, configure
from study_runner import (
    FIXTURE,
    GROUP,
    Objective,
    StudySolveError,
    archive_runtime,
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
    scale_probe: Path = GROUP / "data/15-initial-probe"
    output_dir: Path = GROUP / "data/15-calibration"
    pilot_steps: int = 8
    minimum_fit_progress_retention: float = 0.80
    minimum_target_projection_retention: float = 0.90


def main(cfg: Config) -> None:
    global DONE
    assert cfg.pilot_steps in {8, 16}
    scale = json.loads((cfg.scale_probe / "summary.json").read_text())
    assert scale["status"] == "completed_initial_scale_probe"
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Refuse to overwrite tuning evidence: {out}"
    configure()
    diagnostics = StudyMetrics(fixture=cfg.fixture)
    start = time.perf_counter()
    trials = []
    settings = {
        "status": "tuning_in_progress",
        "pilot_steps": cfg.pilot_steps,
        "inputs": scale["inputs"],
        "adam_eps": scale["adam_eps"],
        "betas": [0.9, 0.999],
        "smooth_length_m": scale["smooth_length_m"],
        "skin_enabled": False,
        "magnitude_weight": 0.0,
        "rank_weight": 0.0,
        "upper_stress_cap": None,
        "initialization": "All pilot and primary cases start inactive at rest with fresh Adam moments. Pilot states are discarded.",
        "lr_selection_rule": "For each model, choose the smallest tested rate within 5% of the largest eight-update data-objective decrease among solver-valid pilots whose last objective is within 1% of their best post-update objective. Test three physically matched candidates; probe one larger candidate if the best remains at the upper edge.",
        "weight_selection_rule": "Test lambda0/4, lambda0, 4lambda0 in both models at their selected rates; lambda0=0.25*||grad_Z data||/||grad_Z smoothness|| at the selected tensile pilot endpoint. Require >=80% fit-progress retention and >=90% target-projection retention for both models; maximize the weaker model's relative smoothness reduction, choosing the smaller weight within 5 percentage points of the best reduction.",
        "minimum_fit_progress_retention": cfg.minimum_fit_progress_retention,
        "minimum_target_projection_retention": cfg.minimum_target_projection_retention,
        "inversion_policy": "Record detF signs as diagnostics; reject only actual solver failure, nonfinite values, or the declared loss-progress checks.",
        "limits": "Short pilot effectiveness is not convergence proof. Initial physical-step matching calibrates the candidate grid; independently selected rates and subsequent Adam geometry can differ.",
    }
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    write_json(out / "selection-protocol.json", settings)

    def trial(model, lr, weight, name):
        trial_out = out / name
        trial_out.mkdir()
        physics = FacePhysics(cfg.fixture, activation_model=model, skin_factor=0.0)
        objective = Objective(physics, smoothness_weight=weight)
        q = torch.nn.Parameter(torch.zeros((len(physics.ids), 6)))
        seed = np.zeros_like(physics.points)
        optimizer = torch.optim.Adam(
            [q], lr=lr, eps=scale["adam_eps"], betas=(0.9, 0.999)
        )
        assert not optimizer.state
        trace, previous_z = [], None
        previous_u = seed
        tick = time.perf_counter()
        failure = None
        endpoint = None
        try:
            for step in range(cfg.pilot_steps + 1):
                result = objective(q, seed, component_gradients=step == cfg.pilot_steps)
                row = {
                    "step": step,
                    "model": model,
                    "learning_rate": lr,
                    "smoothness_weight": weight,
                    "objective_mm2": result["objective_mm2"],
                    "objective_total": result["objective_total"],
                    "smoothness": result["smoothness"],
                    "gradient_rms": result["gradient_rms"],
                    "fit_gradient_rms": result["fit_gradient_rms"],
                    "regularizer_gradient_rms": result["regularizer_gradient_rms"],
                    "objective_s": result["objective_s"],
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
                    "%s %d/%d: lr %.6g, lambda %.6g, fit %.4f mm, S %.5g",
                    name,
                    step,
                    cfg.pilot_steps,
                    lr,
                    weight,
                    row["fit_rms_mm"],
                    row["smoothness"],
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
                        "q": q.detach().cpu().numpy().copy(),
                        "Z": result["Z"],
                        "u": result["u"],
                        "fit_gradient": result["fit_gradient"].cpu().numpy(),
                        "smooth_gradient": result["smooth_gradient"].cpu().numpy(),
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
            LOG.error("Rejected solver-invalid pilot %s: %s", name, error)
        complete = endpoint is not None
        progress = (
            trace[0]["objective_mm2"] - trace[-1]["objective_mm2"] if complete else None
        )
        final = trace[-1] if trace else None
        # Regularized pilots are checked on their own objective, while comparing
        # unpenalized fitting progress against the fit-only reference.
        key = "objective_total" if weight else "objective_mm2"
        stable = bool(
            complete
            and progress > 0
            and final[key] <= 1.01 * min(r[key] for r in trace[1:])
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
            "final": final,
            "elapsed_s": time.perf_counter() - tick,
            "failure": failure,
        }
        write_json(trial_out / "summary.json", record)
        trials.append(record)
        write_json(out / "pilots.json", trials)
        del physics, objective, q, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return record

    # Match each candidate's exact nonlinear Raw6 first Delta Z, not merely its LR multiplier.
    with np.load(cfg.scale_probe / "pilot-final.npz") as loaded:
        assert len(loaded["q"]) == scale["active_cells"]
    base_rates = [0.3, 0.9, 3.0]
    lr_candidates = {"raw6": base_rates, "tensor": []}
    # The scale probe archived the first-step field RMS at base=.3. Re-evaluate
    # the zero-state gradient for exact candidate matching when Raw6 steps grow.
    p = FacePhysics(cfg.fixture, activation_model="raw6", skin_factor=0.0)
    q0 = torch.nn.Parameter(torch.zeros((len(p.ids), 6)))
    initial = Objective(p)(q0, np.zeros_like(p.points))
    g0 = q0.grad.detach().clone()
    mass = torch.as_tensor(p.volumes / p.volumes.sum())
    candidate_receipts = []

    def tensor_rate_for(base_lr):
        qb = -base_lr * g0 / (g0.abs() + scale["adam_eps"])
        zb = baseline_z(qb)
        rms = float((mass * zb.square().sum((-1, -2))).sum().sqrt())
        lr = (
            scale["learning_rates"]["tensor"]
            * rms
            / scale["first_step"]["baseline_volume_weighted_delta_Z_frobenius_rms"]
        )
        candidate_receipts.append(
            {
                "raw6_lr": base_lr,
                "tensor_lr": lr,
                "matched_initial_delta_Z_volume_weighted_frobenius_rms": rms,
            }
        )
        return lr

    lr_candidates["tensor"] = [tensor_rate_for(lr) for lr in base_rates]
    del p, q0, initial
    gc.collect()
    selected = {}

    def select_lr(records):
        valid = [r for r in records if r["stable"]]
        assert valid, "No valid improving learning-rate pilot"
        gain = max(r["progress"] for r in valid)
        return min(
            (r for r in valid if r["progress"] >= 0.95 * gain),
            key=lambda r: r["learning_rate"],
        )

    for model in ("raw6", "tensor"):
        records = [
            trial(model, lr, 0.0, f"lr-{model}-{index}")
            for index, lr in enumerate(lr_candidates[model])
        ]
        winner = select_lr(records)
        if (
            max(
                (record for record in records if record["stable"]),
                key=lambda record: record["progress"],
            )
            is records[-1]
        ):
            lr = 9.0 if model == "raw6" else tensor_rate_for(9.0)
            records.append(trial(model, lr, 0.0, f"lr-{model}-3"))
            winner = select_lr(records)
        selected[model] = winner
    with np.load(out / selected["tensor"]["name"] / "endpoint.npz") as endpoint:
        fit_norm = float(np.linalg.norm(endpoint["fit_gradient"]))
        smooth_norm = float(np.linalg.norm(endpoint["smooth_gradient"]))
    assert fit_norm > 0 and smooth_norm > 0
    lambda0 = 0.25 * fit_norm / smooth_norm
    weights = [lambda0 / 4, lambda0, 4 * lambda0]
    pairs = []
    for index, weight in enumerate(weights):
        records = {
            model: trial(
                model,
                selected[model]["learning_rate"],
                weight,
                f"weight-{index}-{model}",
            )
            for model in ("raw6", "tensor")
        }
        effects = {}
        for model, record in records.items():
            reference = selected[model]
            if record["status"] == "completed":
                assert reference["final"]["target_projection"] > 0
                effects[model] = {
                    "fit_progress_retention": record["progress"]
                    / reference["progress"],
                    "target_projection_retention": record["final"]["target_projection"]
                    / reference["final"]["target_projection"],
                    "smoothness_reduction_fraction": 1
                    - record["final"]["smoothness"] / reference["final"]["smoothness"],
                }
        eligible = (
            len(effects) == 2
            and all(r["stable"] for r in records.values())
            and min(v["fit_progress_retention"] for v in effects.values())
            >= cfg.minimum_fit_progress_retention
            and min(v["target_projection_retention"] for v in effects.values())
            >= cfg.minimum_target_projection_retention
            and min(v["smoothness_reduction_fraction"] for v in effects.values()) > 0
        )
        pairs.append(
            {
                "smoothness_weight": weight,
                "eligible": bool(eligible),
                "effects": effects,
                "pilots": {model: record["name"] for model, record in records.items()},
            }
        )
        write_json(out / "weight-comparison.json", pairs)
    eligible = [pair for pair in pairs if pair["eligible"]]
    assert eligible, (
        "No effective shared positive weight passed the declared pilot criteria; inspect pilots before extending the bracket or horizon"
    )
    reduction = max(
        min(v["smoothness_reduction_fraction"] for v in pair["effects"].values())
        for pair in eligible
    )
    chosen = min(
        (
            pair
            for pair in eligible
            if min(v["smoothness_reduction_fraction"] for v in pair["effects"].values())
            >= reduction - 0.05
        ),
        key=lambda pair: pair["smoothness_weight"],
    )
    provenance = archive_runtime(out, cfg.fixture)
    settings.update(
        {
            "status": "frozen_before_primary_runs",
            "learning_rates": {
                model: selected[model]["learning_rate"] for model in selected
            },
            "smoothness_weight": chosen["smoothness_weight"],
            "lr_candidate_matching": candidate_receipts,
            "selected_lr_pilots": {
                model: selected[model]["name"] for model in selected
            },
            "selected_weight_pilots": chosen["pilots"],
            "selected_weight_effects": chosen["effects"],
            "lambda0": lambda0,
            "lambda0_gradient_norms": {"fit": fit_norm, "smoothness": smooth_norm},
            "secondary_weight_multiplier": 4.0,
            "scale_probe": {
                "path": str(cfg.scale_probe / "summary.json"),
                "sha256": digest(cfg.scale_probe / "summary.json"),
            },
            "controls_validation": scale["controls_validation"],
            "elapsed_s": time.perf_counter() - start,
        }
    )
    write_json(out / "provenance.json", provenance)
    write_json(out / "summary.json", settings)
    for name in (
        "config.json",
        "selection-protocol.json",
        "pilots.json",
        "weight-comparison.json",
        "provenance.json",
        "summary.json",
    ):
        cherries.log_output(out / name)
    LOG.info(
        "Selected rates %s and common smoothness weight %.8g",
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
