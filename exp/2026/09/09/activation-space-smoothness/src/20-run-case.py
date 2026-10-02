"""Run one frozen no-skin activation-space and smoothness comparison."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pydantic_settings as ps
import torch
from activation_controls import common_initial_controls, control_c
from experiment_profile import ProfileCometNoCommit
from study_metrics import StudyMetrics
from study_physics import FacePhysics, configure
from study_runner import (
    FIXTURE,
    GROUP,
    Objective,
    archive_runtime,
    archived_initial_state,
    control_metrics,
    control_z,
    digest,
    make_adam,
    save_state,
    volume_metrics,
    write_json,
    write_trace,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)
DONE = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    case: Literal["baseline-smooth", "learned-axis", "learned-axis-smooth"]
    settings: Path = GROUP / "data/18-learned-axis-calibration-refined/summary.json"
    fixture: Path = FIXTURE
    steps: int = 256
    checkpoint_interval: int = 16
    output_dir: Path | None = None
    smoothness_multiplier: float = 1.0
    resume: Path | None = None
    initialization_seed: int = 20260909


def main(cfg: Config) -> None:
    global DONE
    assert cfg.steps > 0 and cfg.checkpoint_interval > 0
    assert cfg.smoothness_multiplier == 1.0
    settings = json.loads(cfg.settings.read_text())
    assert settings["status"] == "frozen_before_primary_runs"
    controls_validation = Path(settings["controls_validation"]["path"])
    assert json.loads(controls_validation.read_text())["status"] == "passed"
    assert digest(controls_validation) == settings["controls_validation"]["sha256"]
    assert settings["smoothness_field"] == "C"
    model = "raw6" if cfg.case.startswith("baseline") else "learned-axis"
    smooth_weight = (
        settings["smoothness_weight"] * cfg.smoothness_multiplier
        if cfg.case.endswith("smooth")
        else 0.0
    )
    learning_rate = settings["learning_rates"][model]
    eps = settings["adam_eps"]
    names = {
        "baseline-smooth": "28-raw6-smooth",
        "learned-axis": "24-learned-axis",
        "learned-axis-smooth": "25-learned-axis-smooth",
    }
    out = cfg.output_dir or GROUP / "data" / names[cfg.case]
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Refuse to overwrite run: {out}"
    configure()
    physics = FacePhysics(cfg.fixture, activation_model=model, skin_factor=0.0)
    diagnostics = StudyMetrics(fixture=cfg.fixture)
    assert len(physics.ids) == 288235 and len(physics.top) == 15302
    objective = Objective(
        physics, smoothness_weight=smooth_weight, smoothness_field="C"
    )
    if model == "raw6":
        assert (
            settings["initialization_mode"] == "archived_controls_and_seed_fresh_adam"
        )
        initial_q, seed = archived_initial_state(
            physics, settings["archive_initialization"]
        )
        initialization_seed = None
    else:
        assert cfg.initialization_seed == settings["initialization_seed"]
        initial_q = common_initial_controls(
            physics.region_t,
            model=model,
            seed=cfg.initialization_seed,
            strength=settings["initial_strength"],
        )
        seed = np.zeros_like(physics.points)
        initialization_seed = cfg.initialization_seed
    q = torch.nn.Parameter(initial_q.clone())
    optimizer = make_adam(q, settings, learning_rate)
    start_step = 0
    parent = None
    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        assert checkpoint["case"] == cfg.case and checkpoint["model"] == model
        assert checkpoint["smoothness_weight"] == smooth_weight
        assert checkpoint["settings_sha256"] == digest(cfg.settings)
        assert checkpoint["initialization_seed"] == initialization_seed
        assert np.array_equal(checkpoint["active_ids"], physics.ids)
        q.data.copy_(checkpoint["q"].to(q))
        optimizer.load_state_dict(checkpoint["optimizer"])
        q.grad = checkpoint["gradient"].to(q).clone()
        assert optimizer.param_groups[0]["lr"] == learning_rate
        seed = checkpoint["u"].copy()
        start_step = int(checkpoint["step"])
        parent = {
            "checkpoint": str(cfg.resume),
            "sha256": digest(cfg.resume),
            "step": start_step,
            "continuation": "Restore saved gradient, controls, Adam state and solve seed; next evaluation follows one Adam update. Parent states are not re-evaluated.",
        }
    assert cfg.steps > start_step
    if start_step == 0:
        assert not optimizer.state and torch.equal(q, initial_q)
    sources = archive_runtime(out, cfg.fixture)
    for name, record in settings["inputs"].items():
        assert digest(cfg.fixture / name) == record["sha256"]
    provenance = {
        **sources,
        "case": cfg.case,
        "model": model,
        "materials": physics.material_spec,
        "settings": {"path": str(cfg.settings), "sha256": digest(cfg.settings)},
        "controls_validation": {
            "path": str(controls_validation),
            "sha256": digest(controls_validation),
        },
        "diagnostics": diagnostics.provenance,
        "forward_tolerances": physics.forward_tolerance,
        "objective": "uniform finite-IsFace displacement MSE * 1e6 + lambda * same-muscle smoothness of C=B-I",
        "physical_field": "Z=BB^T-I; raw6 B=I+sym(q), learned-axis B=I+vv^T; tensile Q=mu Z",
        "skin_enabled": False,
        "magnitude_weight": 0.0,
        "rank_weight": 0.0,
        "upper_stress_cap": None,
        "smoothness_weight": smooth_weight,
        "optimizer": {
            "name": "Adam",
            "lr": learning_rate,
            "eps": eps,
            "betas": settings["betas"],
            "weight_decay": 0,
            "amsgrad": False,
            "maximize": False,
            "foreach": optimizer.param_groups[0]["foreach"],
            "fused": optimizer.param_groups[0]["fused"],
            "target_step": cfg.steps,
            "start_step": start_step,
            "initialization": settings["initialization_mode"]
            if parent is None
            else parent["continuation"],
            "initialization_seed": initialization_seed,
            "initial_strength": settings.get("initial_strength"),
            "archive_initialization": settings.get("archive_initialization"),
        },
        "parent": parent,
        "failure_policy": "Stop on nonfinite fields or failed forward/adjoint. Record inversions and stress spectrum as diagnostics.",
        "inverse_stationarity_claimed": False,
    }
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    write_json(out / "provenance.json", provenance)
    trace, best_fit, best_objective = [], None, None
    elapsed_offset = 0.0
    if parent is not None:
        parent_dir = cfg.resume.parent
        parent_summary = json.loads((parent_dir / "summary.json").read_text())
        assert parent_summary["last_evaluated_step"] == start_step
        with (parent_dir / "trace.csv").open() as stream:
            for raw in csv.DictReader(stream):
                trace.append(
                    {
                        k: (
                            v == "True"
                            if v in {"True", "False"}
                            else int(v)
                            if v.lstrip("-").isdigit()
                            else float(v)
                        )
                        for k, v in raw.items()
                    }
                )
        assert trace[-1]["step"] == start_step
        elapsed_offset = trace[-1]["elapsed_s"]
        for pattern in (
            "surface-*.npz",
            "step-*.npz",
            "best*.npz",
            "solver-receipts.jsonl",
        ):
            for source in parent_dir.glob(pattern):
                shutil.copy2(source, out / source.name)

        def load_best(name, metric_key):
            with np.load(parent_dir / name) as saved:
                state = {key: saved[key].copy() for key in ("q", "C", "Z", "u")}
                state["step"] = int(saved["step"])
            state["metrics"] = parent_summary[metric_key]
            return state

        best_fit = load_best("best.npz", "best_metrics")
        best_objective = load_best("best-objective.npz", "best_objective_metrics")
        write_trace(out / "trace.csv", trace)
    start = time.perf_counter()
    previous_u = seed.copy()
    previous_z = control_z(q.detach(), model).cpu().numpy().copy() if parent else None
    status, failure = "running", None
    projection = {"projection_rms": 0.0, "projected_negative_eigenvalue_fraction": 0.0}
    step = start_step + int(parent is not None)
    try:
        if parent is not None:
            optimizer.step()
        for step in range(start_step + int(parent is not None), cfg.steps + 1):
            tick = time.perf_counter()
            result = objective(q, seed)
            u, z = result["u"], result["Z"]
            row = {
                "step": step,
                **{
                    key: result[key]
                    for key in (
                        "objective_mm2",
                        "objective_total",
                        "smoothness",
                        "smoothness_weight",
                        "smoothness_C",
                        "smoothness_Z",
                        "fit_gradient_rms",
                        "gradient_rms",
                        "regularizer_gradient_rms",
                        "objective_s",
                    )
                },
                "forward_steps": result["forward"]["steps"],
                "forward_grad_norm": result["forward"]["grad_norm"],
                "forward_success": True,
                "adjoint_success": True,
                **diagnostics.evaluate(u, z, previous_u, previous_z),
                **control_metrics(q.detach().cpu().numpy(), result["C"], model),
                **volume_metrics(physics, u),
                **projection,
                "elapsed_s": elapsed_offset + time.perf_counter() - start,
                "step_s": time.perf_counter() - tick,
            }
            assert math.isclose(
                row["objective_mm2"] * 3, row["fit_rms_mm"] ** 2, rel_tol=1e-10
            )
            state = {
                "step": step,
                "q": q.detach().cpu().numpy().copy(),
                "C": result["C"],
                "Z": z,
                "u": u,
                "metrics": row.copy(),
            }
            if (
                best_fit is None
                or row["objective_mm2"] < best_fit["metrics"]["objective_mm2"]
            ):
                best_fit = state
            if (
                best_objective is None
                or row["objective_total"] < best_objective["metrics"]["objective_total"]
            ):
                best_objective = state
            row["best_step"] = best_fit["step"]
            row["best_objective_step"] = best_objective["step"]
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
            np.savez_compressed(
                out / f"surface-{step:04d}.npz",
                step=np.array(step),
                point_ids=diagnostics.skin_ids,
                u=u[diagnostics.skin_ids],
            )
            if step == 1 or step % cfg.checkpoint_interval == 0 or step == cfg.steps:
                save_state(out / f"step-{step:04d}.npz", physics, state)
                save_state(out / "best.npz", physics, best_fit)
                save_state(out / "best-objective.npz", physics, best_objective)
            temporary = out / "optimizer-latest.tmp"
            torch.save(
                {
                    "case": cfg.case,
                    "model": model,
                    "step": step,
                    "q": q.detach().cpu(),
                    "u": u,
                    "active_ids": physics.ids,
                    "gradient": q.grad.detach().cpu(),
                    "optimizer": optimizer.state_dict(),
                    "smoothness_weight": smooth_weight,
                    "settings_sha256": digest(cfg.settings),
                    "initialization_seed": initialization_seed,
                    "best_fit_step": best_fit["step"],
                    "best_objective_step": best_objective["step"],
                },
                temporary,
            )
            temporary.replace(out / "optimizer-latest.pt")
            cherries.log_metrics(
                {
                    key: value
                    for key, value in row.items()
                    if isinstance(value, (int, float))
                },
                step=step,
            )
            LOG.info(
                "%s %04d/%d: fit %.4f mm, motion %.4f mm, smoothness %.5g; %.1f s",
                cfg.case,
                step,
                cfg.steps,
                row["fit_rms_mm"],
                row["motion_rms_mm"],
                row["smoothness"],
                row["step_s"],
            )
            if step == cfg.steps:
                save_state(out / "last.npz", physics, state)
                status = "completed_fixed_budget_not_stationarity_certified"
                break
            previous_u, previous_z, seed = u, z, u
            optimizer.step()
    except BaseException as error:
        status = "failed_before_completion"
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "step": step,
            "forward": getattr(physics, "last_forward", None),
        }
        write_json(out / "failure.json", failure)
        np.savez_compressed(
            out / "failure-controls.npz",
            step=np.array(step),
            q=q.detach().cpu().numpy(),
            C=control_c(q.detach(), model).cpu().numpy(),
            Z=control_z(q.detach(), model).cpu().numpy(),
            active_ids=physics.ids,
            solver_valid=np.array(False),
        )
        raise
    finally:
        if best_fit is not None:
            save_state(out / "best.npz", physics, best_fit)
            save_state(out / "best-objective.npz", physics, best_objective)
        summary = {
            "case": cfg.case,
            "status": status,
            "last_evaluated_step": None if not trace else trace[-1]["step"],
            "best_step": None if best_fit is None else best_fit["step"],
            "best_metrics": None if best_fit is None else best_fit["metrics"],
            "best_objective_step": None
            if best_objective is None
            else best_objective["step"],
            "best_objective_metrics": None
            if best_objective is None
            else best_objective["metrics"],
            "last_metrics": None if not trace else trace[-1],
            "failure": failure,
            "elapsed_s": time.perf_counter() - start,
            "parent": parent,
            "smoothness_weight": smooth_weight,
        }
        write_json(out / "summary.json", summary)
        for name in (
            "config.json",
            "provenance.json",
            "summary.json",
            "trace.csv",
            "solver-receipts.jsonl",
        ):
            if (out / name).exists():
                cherries.log_output(out / name)
        LOG.info("%s: %s; %s", cfg.case, status, out)
    DONE = True


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
    if not DONE:
        raise SystemExit(1)
