"""Neutral-start full-face comparison: surface L2 versus gradient matching alone."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import comet_ml
import numpy as np
import pydantic_settings as ps
import torch
from face_study import FIXTURE, LEGACY, ROOT, FaceStudy
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Comet(plugins.Comet):
    @core.impl
    def start(self):
        experiment = comet_ml.start(
            project_name=self.run.project_name,
            experiment_config=comet_ml.ExperimentConfig(
                disabled=os.environ.get("DEBUG") == "1",
                name=self.run.run_name,
                tags=self.run.tags,
                log_env_details=False,
                log_git_patch=False,
                auto_log_co2=False,
            ),
        )
        self.run.log_other("cherries/comet/url", experiment.url)


class ProfileFaceExperiment(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(Comet(run=run))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Logging(run=run))
        run.plugins.register(plugins.Local(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("10-comparison")
    steps: int = 100
    learning_rate: float = 0.3
    adam_eps: float = 0.01
    checkpoint_interval: int = 10
    validation_only: bool = False
    calibration: Path | None = None
    branches: str = "l2,gradient"


def write_json(path: Path, data: object):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive(out: Path) -> dict:
    sources = {}
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if not source or not source.endswith(".py"):
            continue
        path = Path(source).resolve()
        if path.parent in {Path(__file__).parent, LEGACY}:
            relative = Path("experiment") / path.name
        elif name.startswith(("liblaf.apple", "liblaf.peach")):
            relative = Path("runtime") / Path(*name.split(".")).with_suffix(".py")
        else:
            continue
        snapshot = out / "sources" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
        sources[name] = {
            "path": str(path),
            "sha256": digest(path),
            "snapshot": str(snapshot),
        }
    return sources


def calibrate(study: FaceStudy, cfg: Config, out: Path):
    seed = np.zeros_like(study.physics.points)
    q = torch.nn.Parameter(torch.zeros((len(study.physics.ids), 6)))
    reference = study.evaluate(q, seed, "l2", 1.0, backward=True)
    raw = study.evaluate(q, seed, "gradient", 1.0, backward=True)
    g_l2, g_gradient = reference["gradient"], raw["gradient"]
    target_rms = study.physical_step_rms(
        -cfg.learning_rate * g_l2 / (np.abs(g_l2) + cfg.adam_eps)
    )

    def first_step_rms(scale: float):
        g = scale * g_gradient
        return study.physical_step_rms(
            -cfg.learning_rate * g / (np.abs(g) + cfg.adam_eps)
        )

    low, high = 0.0, 1.0
    while first_step_rms(high) < target_rms:
        high *= 10
        assert high < 1e20
    for _ in range(70):
        middle = (low + high) / 2
        if first_step_rms(middle) < target_rms:
            low = middle
        else:
            high = middle
    scale = (low + high) / 2
    result = {
        "gradient_scale": scale,
        "reference_l2_first_step_physical_B_rms": target_rms,
        "gradient_first_step_physical_B_rms": first_step_rms(scale),
        "raw_gradient_first_step_physical_B_rms": first_step_rms(1.0),
        "learning_rate": cfg.learning_rate,
        "adam_eps": cfg.adam_eps,
        "meaning": "One positive constant matches initial active-volume-weighted Frobenius RMS of Adam's delta B. No positional term is added; directions and subsequent steps remain different.",
        "unscaled_initial_surface_gradient_loss": raw["surface_gradient_loss"],
        "initial_l2_component_mm2": reference["position_loss_component_mm2"],
    }
    write_json(out / "calibration.json", result)
    np.savez_compressed(out / "calibration-gradients.npz", l2=g_l2, gradient=g_gradient)
    LOG.info(
        "Gradient scalar %.9g; matched initial delta-B RMS %.9g", scale, target_rms
    )
    return result, raw


def gradient_check(study: FaceStudy, scale: float, initial: dict, out: Path):
    grad = scale * initial["gradient"]
    rng = np.random.default_rng(20260919)
    by_region = rng.choice([-1.0, 1.0], size=(study.physics.n_regions, 6))
    directions = {
        "gradient_sign": np.sign(grad),
        "muscle_coherent": by_region[study.physics.region],
    }
    rows = []
    for name, direction in directions.items():
        analytic = float(np.sum(grad * direction))
        assert abs(analytic) > 1e-9
        slopes = []
        for epsilon in (0.01, 0.005):
            values = []
            for sign in (-1, 1):
                LOG.info(
                    "Full-face derivative check %s epsilon=%g sign=%d",
                    name,
                    epsilon,
                    sign,
                )
                result = study.evaluate(
                    torch.as_tensor(sign * epsilon * direction),
                    initial["u"],
                    "gradient",
                    scale,
                    backward=False,
                )
                values.append(result["objective"])
            numeric = (values[1] - values[0]) / (2 * epsilon)
            slopes.append(numeric)
            rows.append(
                {
                    "direction": name,
                    "epsilon": epsilon,
                    "analytic": analytic,
                    "numeric": numeric,
                    "relative_error": abs(numeric - analytic) / abs(analytic),
                }
            )
            write_json(
                out / "gradient-validation.json", {"status": "running", "checks": rows}
            )
        assert abs(slopes[-1] - analytic) / abs(analytic) < 0.02, rows
        assert abs(slopes[-1] - slopes[0]) / abs(analytic) < 0.02, rows
    write_json(
        out / "gradient-validation.json",
        {"status": "passed", "checks": rows, "tolerance": 0.02},
    )


def save_state(path: Path, q: np.ndarray, result: dict, step: int):
    np.savez_compressed(
        path,
        q=q,
        u=result["u"],
        step=step,
        solver_valid=True,
        physical_volume_energy=True,
    )


def run_branch(study: FaceStudy, cfg: Config, out: Path, kind: str, scale: float):
    folder = out / kind
    folder.mkdir()
    q = torch.nn.Parameter(torch.zeros((len(study.physics.ids), 6)))
    optimizer = torch.optim.Adam(
        [q], lr=cfg.learning_rate, eps=cfg.adam_eps, betas=(0.9, 0.999)
    )
    seed = np.zeros_like(study.physics.points)
    rows, best = [], None
    best_noninverted = None
    status, failure = "running", None
    start = time.perf_counter()
    try:
        for step in range(cfg.steps + 1):
            tick = time.perf_counter()
            result = study.evaluate(q, seed, kind, scale, backward=True)
            q_numpy = q.detach().cpu().numpy().copy()
            row = {
                "step": step,
                **study.metrics(q_numpy, result),
                "elapsed_seconds": time.perf_counter() - start,
            }
            row["step_seconds"] = time.perf_counter() - tick
            rows.append(row)
            if best is None or row["objective"] < best["objective"]:
                best = row.copy()
                save_state(folder / "best.npz", q_numpy, result, step)
            if row["inverted_all_cells"] == 0 and (
                best_noninverted is None
                or row["objective"] < best_noninverted["objective"]
            ):
                best_noninverted = row.copy()
                save_state(folder / "best-noninverted.npz", q_numpy, result, step)
            if step % cfg.checkpoint_interval == 0 or step == cfg.steps:
                save_state(folder / f"step-{step:04d}.npz", q_numpy, result, step)
            save_state(folder / "last.npz", q_numpy, result, step)
            torch.save(
                {
                    "step": step,
                    "q": q.detach().cpu(),
                    "u": result["u"],
                    "optimizer": optimizer.state_dict(),
                    "kind": kind,
                    "gradient_scale": scale,
                },
                folder / "optimizer-latest.pt",
            )
            with (folder / "trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            with (folder / "solver-receipts.jsonl").open("a") as stream:
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
            write_json(
                folder / "summary.json",
                {
                    "status": status,
                    "last_step": step,
                    "initial_metrics": rows[0],
                    "last_metrics": row,
                    "best_metrics": best,
                    "best_noninverted_metrics": best_noninverted,
                    "failure": failure,
                },
            )
            cherries.log_metrics(
                {
                    f"{kind}/{key}": row[key]
                    for key in (
                        "objective",
                        "fit_rms_mm",
                        "surface_gradient_rms",
                        "motion_rms_mm",
                        "inverted_all_cells",
                    )
                },
                step=step,
            )
            LOG.info(
                "%s %d/%d: fit %.4f mm, surface-gradient RMS %.6f, motion %.4f mm, inversions %d, forward %d; %.1fs",
                kind,
                step,
                cfg.steps,
                row["fit_rms_mm"],
                row["surface_gradient_rms"],
                row["motion_rms_mm"],
                row["inverted_all_cells"],
                row["forward_steps"],
                row["step_seconds"],
            )
            if step == cfg.steps:
                status = "completed_budget_not_convergence_certified"
                break
            seed = result["u"].copy()
            optimizer.step()
    except Exception as error:
        status = "failed_before_budget_completed"
        failure = {"type": type(error).__name__, "message": str(error), "step": step}
        write_json(folder / "failure.json", failure)
        np.savez_compressed(
            folder / "failed-proposal.npz",
            q=q.detach().cpu().numpy(),
            step=step,
            solver_valid=False,
        )
        raise
    finally:
        write_json(
            folder / "summary.json",
            {
                "status": status,
                "last_step": rows[-1]["step"] if rows else None,
                "initial_metrics": rows[0] if rows else None,
                "last_metrics": rows[-1] if rows else None,
                "best_metrics": best,
                "best_noninverted_metrics": best_noninverted,
                "failure": failure,
                "elapsed_seconds": time.perf_counter() - start,
            },
        )


def main(cfg: Config):
    assert cfg.steps >= 0
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    study = FaceStudy()
    p = study.physics
    np.savez_compressed(
        out / "mesh.npz",
        rest_points=p.points,
        skin_ids=study.skin_ids,
        triangles=study.triangles,
        target_displacement_skin=p.target[study.skin_ids],
        skin_vertex_weights=study.weights,
        initial_u=np.zeros_like(p.points),
        active_ids=p.ids,
    )
    sources = archive(out)
    if cfg.calibration is None:
        calibration, raw = calibrate(study, cfg, out)
        gradient_check(study, calibration["gradient_scale"], raw, out)
    else:
        gate = cfg.calibration.parent / "gradient-validation.json"
        assert json.loads(gate.read_text())["status"] == "passed"
        calibration = json.loads(cfg.calibration.read_text())
        assert calibration["learning_rate"] == cfg.learning_rate
        assert calibration["adam_eps"] == cfg.adam_eps
        shutil.copy2(cfg.calibration, out / "calibration.json")
        shutil.copy2(gate, out / "gradient-validation.json")
    protocol = {
        "config": cfg.model_dump(mode="json"),
        "start": "neutral: q=0, B=I, zero displacement seed, fresh Adam moments for each branch",
        "activation": "Raw6 unrestricted symmetric B=I+sym(qxx,qyy,qzz,qxy,qyz,qxz); same model for both branches; no projections or activation regularizers",
        "position_objective": "1e6/3 times rest-area-lumped mean squared skin position residual in meters; all 15299 skin vertices",
        "gradient_objective": "fixed positive scalar times total-rest-area-normalized triangle surface-gradient Frobenius error; no positional or activation penalty",
        "gradient_scale_calibration": calibration,
        "translation": "surface gradient has a constant-translation nullspace; skull fixation retained; report mean residual and centered fit separately; no extra positional anchor",
        "optimizer": "constant-lr Adam with matched initial physical delta-B RMS; same lr/betas/epsilon; no decay or inverse line search",
        "materials": p.material_spec,
        "forward_tolerance": p.forward_tolerance,
        "surface_points": len(study.skin_ids),
        "surface_triangles": len(study.triangles),
        "volume_points": len(p.points),
        "tetrahedra": len(p.tets),
        "active_cells": len(p.ids),
        "skin_unsupported_legacy_L2_ids": np.setdiff1d(p.top, study.skin_ids).tolist(),
        "fixture": {
            name: {"path": str(FIXTURE / name), "sha256": digest(FIXTURE / name)}
            for name in ("volume.vtu", "skin.vtp", "summary.json")
        },
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": [sys.executable, *sys.argv],
        },
        "sources": sources,
        "inverse_stationarity_claimed": False,
    }
    write_json(out / "protocol.json", protocol)
    if cfg.validation_only:
        return
    for kind in cfg.branches.split(","):
        assert kind in {"l2", "gradient"}
        run_branch(study, cfg, out, kind, calibration["gradient_scale"])


if __name__ == "__main__":
    cherries.main(main, profile=ProfileFaceExperiment)
