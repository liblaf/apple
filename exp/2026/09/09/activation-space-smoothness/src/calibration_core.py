"""Bounded C-regularization pilots with complete local evidence."""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from activation_controls import common_initial_controls, control_c
from study_metrics import StudyMetrics
from study_physics import FacePhysics, ForwardConvergenceError
from study_runner import (
    Objective,
    StudySolveError,
    archived_initial_state,
    control_metrics,
    control_z,
    make_adam,
    volume_metrics,
    write_json,
    write_trace,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Calibration:
    def __init__(
        self, out: Path, fixture: Path, settings: dict, model: str, steps: int
    ):
        self.out, self.fixture, self.settings = out, fixture, settings
        self.model, self.steps = model, steps
        self.diagnostics = StudyMetrics(fixture=fixture)
        self.trials: list[dict] = []

    def initialize(self, physics):
        if self.model == "raw6":
            q, seed = archived_initial_state(
                physics, self.settings["archive_initialization"]
            )
        else:
            q = common_initial_controls(
                physics.region_t,
                model=self.model,
                seed=self.settings["initialization_seed"],
                strength=self.settings["initial_strength"],
            )
            seed = np.zeros_like(physics.points)
        return torch.nn.Parameter(q.clone()), seed

    def trial(self, lr: float, weight: float, name: str) -> dict:
        out = self.out / name
        out.mkdir()
        physics = FacePhysics(
            self.fixture, activation_model=self.model, skin_factor=0.0
        )
        objective = Objective(physics, smoothness_weight=weight, smoothness_field="C")
        q, seed = self.initialize(physics)
        optimizer = make_adam(q, self.settings, lr)
        assert not optimizer.state
        previous_u, previous_z = seed, None
        trace, failure, endpoint = [], None, None
        tick = time.perf_counter()
        try:
            for step in range(self.steps + 1):
                result = objective(q, seed, component_gradients=step == self.steps)
                q_cpu = q.detach().cpu().numpy().copy()
                row = {
                    "step": step,
                    "model": self.model,
                    "learning_rate": lr,
                    **{
                        k: result[k]
                        for k in (
                            "smoothness_weight",
                            "objective_mm2",
                            "objective_total",
                            "smoothness",
                            "smoothness_C",
                            "smoothness_Z",
                            "gradient_rms",
                            "fit_gradient_rms",
                            "regularizer_gradient_rms",
                            "objective_s",
                        )
                    },
                    "forward_steps": result["forward"]["steps"],
                    "forward_success": True,
                    "adjoint_success": True,
                    **self.diagnostics.evaluate(
                        result["u"], result["Z"], previous_u, previous_z
                    ),
                    **control_metrics(q_cpu, result["C"], self.model),
                    **volume_metrics(physics, result["u"]),
                }
                trace.append(row)
                write_trace(out / "trace.csv", trace)
                with (out / "solver-receipts.jsonl").open("a") as stream:
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
                    "%s %d/%d: lr %.7g, lambda %.7g, fit %.4f mm, S_C %.5g; %.1f s",
                    name,
                    step,
                    self.steps,
                    lr,
                    weight,
                    row["fit_rms_mm"],
                    row["smoothness_C"],
                    row["objective_s"],
                )
                cherries.log_metrics(
                    {
                        f"{name}/fit_rms_mm": row["fit_rms_mm"],
                        f"{name}/smoothness_C": row["smoothness_C"],
                    },
                    step=step,
                )
                if step == self.steps:
                    endpoint = {
                        "q": q_cpu,
                        "C": result["C"],
                        "Z": result["Z"],
                        "u": result["u"],
                        "fit_gradient": result["fit_gradient"].cpu().numpy(),
                        "smooth_gradient": result["smooth_gradient"].cpu().numpy(),
                    }
                    np.savez_compressed(out / "endpoint.npz", **endpoint)
                    break
                previous_u, previous_z, seed = result["u"], result["Z"], result["u"]
                optimizer.step()
        except (StudySolveError, ForwardConvergenceError) as error:
            failure = {
                "type": type(error).__name__,
                "message": str(error),
                "step": step,
                "forward": getattr(physics, "last_forward", None),
            }
            np.savez_compressed(
                out / "failure-controls.npz",
                q=q.detach().cpu().numpy(),
                C=control_c(q.detach(), self.model).cpu().numpy(),
                Z=control_z(q.detach(), self.model).cpu().numpy(),
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
            and trace[-1][key] < trace[0][key]
            and trace[-1][key] <= 1.01 * min(r[key] for r in trace[1:])
        )
        record = {
            "name": name,
            "model": self.model,
            "learning_rate": lr,
            "smoothness_weight": weight,
            "status": "completed" if complete else "rejected_solver_failure",
            "stable": stable,
            "progress": progress,
            "fractional_progress": progress / trace[0]["objective_mm2"]
            if complete
            else None,
            "initial": trace[0] if trace else None,
            "final": trace[-1] if trace else None,
            "failure": failure,
            "elapsed_s": time.perf_counter() - tick,
        }
        write_json(out / "summary.json", record)
        self.trials.append(record)
        write_json(self.out / "pilots.json", self.trials)
        del physics, objective, q, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return record


def weight_effect(record: dict, reference: dict) -> dict:
    effect = {
        "name": record["name"],
        "smoothness_weight": record["smoothness_weight"],
        "eligible": False,
    }
    if record["status"] != "completed":
        return effect
    initial, final, off = record["initial"], record["final"], reference["final"]
    assert abs(initial["objective_mm2"] - reference["initial"]["objective_mm2"]) < 1e-7
    assert (
        abs(initial["target_projection"] - reference["initial"]["target_projection"])
        < 1e-7
    )
    projection_progress = (
        off["target_projection"] - reference["initial"]["target_projection"]
    )
    if not (
        reference["progress"] > 0
        and projection_progress > 0
        and off["smoothness_C"] > 0
    ):
        return effect
    effect.update(
        {
            "fit_progress_retention": record["progress"] / reference["progress"],
            "target_projection_progress_retention": (
                final["target_projection"] - initial["target_projection"]
            )
            / projection_progress,
            "smoothness_reduction_fraction": 1
            - final["smoothness_C"] / off["smoothness_C"],
        }
    )
    effect["eligible"] = bool(
        record["stable"]
        and effect["fit_progress_retention"] >= 0.8
        and effect["target_projection_progress_retention"] >= 0.9
        and effect["smoothness_reduction_fraction"] > 0
    )
    return effect


def select_weight(
    calibration: Calibration, lr: float, scale: float, reference: dict
) -> dict:
    assert np.isfinite(scale) and scale > 0
    effects = []
    for index, weight in enumerate(scale * np.array([0.25, 1.0, 4.0])):
        record = calibration.trial(lr, float(weight), f"weight-{index}")
        effects.append(weight_effect(record, reference))
        write_json(calibration.out / "weight-comparison.json", effects)
    eligible = [e for e in effects if e["eligible"]]
    assert eligible, "No positive C-smoothness weight passed the bounded calibration"
    best = max(e["smoothness_reduction_fraction"] for e in eligible)
    return min(
        (e for e in eligible if e["smoothness_reduction_fraction"] >= best - 0.05),
        key=lambda e: e["smoothness_weight"],
    )
