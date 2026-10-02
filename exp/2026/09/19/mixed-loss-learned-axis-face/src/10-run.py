"""Neutral-start mixed position/gradient face loss with matched learned-axis activation."""

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
from liblaf.cherries import core, plugins, profiles
from study import BASE, FIXTURE, LEGACY, ROOT, Study, pack, project_

from liblaf import cherries

LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parents[1]
AXIS = GROUP.parent / "learned-axis-gradient-face"
AXIS_PREP = AXIS / "data/06-preparation-tight"


class Comet(plugins.Comet):
    @core.impl
    def start(self) -> None:
        experiment = comet_ml.start(
            project_name=self.run.project_name,
            experiment_config=comet_ml.ExperimentConfig(
                disabled=self.disabled or os.environ.get("DEBUG") == "1",
                name=self.run.run_name,
                tags=self.run.tags,
                log_env_details=False,
                log_git_patch=False,
                auto_log_co2=False,
            ),
        )
        self.run.log_other("cherries/comet/url", experiment.url)


class ProfileAxisExperiment(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(Comet(run=run))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Logging(run=run))
        run.plugins.register(plugins.Local(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    phase: str = "validate"
    output: Path = Path("06-validation")
    preparation: Path = GROUP / "data/10-preparation"
    validation: Path = GROUP / "data/06-validation"
    beta: float = 1.0
    steps: int = 200
    probe_steps: int = 8
    pilot_steps: int = 20
    coefficient: float | None = None
    learning_rate: float = 0.3
    adam_eps: float = 0.01
    smooth_length_m: float = 0.005
    checkpoint_interval: int = 25
    convergence_window: int = 25
    min_steps: int = 100
    plateau_tolerance: float = 0.001
    stationarity_ratio: float = 0.01


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive(out: Path) -> dict:
    records = {}
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if not source or not source.endswith(".py"):
            continue
        path = Path(source).resolve()
        if path.parent in {Path(__file__).parent, LEGACY, BASE / "src", AXIS / "src"}:
            relative = Path("experiment") / path.parent.parent.name / path.name
        elif name.startswith(("liblaf.apple", "liblaf.peach")):
            relative = Path("runtime") / Path(*name.split(".")).with_suffix(".py")
        else:
            continue
        target = out / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records[name] = {
            "path": str(path),
            "sha256": digest(path),
            "snapshot": str(target),
        }
    return records


def parameters(axes: np.ndarray) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    return torch.nn.Parameter(torch.zeros(len(axes))), torch.nn.Parameter(
        torch.as_tensor(axes.copy())
    )


def optimizer_for(s: torch.Tensor, n: torch.Tensor, cfg: Config):
    return torch.optim.Adam(
        [s, n], lr=cfg.learning_rate, eps=cfg.adam_eps, betas=(0.9, 0.999)
    )


def save_state(path: Path, case: dict, result: dict, step: int) -> None:
    np.savez(
        path,
        s=case["s"].detach().cpu().numpy(),
        n=case["n"].detach().cpu().numpy(),
        q=result["q"],
        u=result["u"],
        step=step,
        solver_valid=True,
        physical_volume_energy=True,
        smoothness_coefficient=case["coefficient"],
    )


def prepare(study: Study, cfg: Config, out: Path, axes: np.ndarray) -> None:
    p = study.physics
    s, n = parameters(axes)
    optimizer = optimizer_for(s, n, cfg)
    seed = np.zeros_like(p.points)
    rows = []
    for step in range(cfg.probe_steps + 1):
        result = study.evaluate_axis(s, n, seed, 0.0, backward=True)
        rows.append(
            {
                "step": step,
                "data_objective": result["data_objective"],
                "activation_smoothness": result["activation_smoothness"],
            }
        )
        LOG.info(
            "Coefficient calibration probe %d/%d: data %.8g, smoothness %.8g",
            step,
            cfg.probe_steps,
            result["data_objective"],
            result["activation_smoothness"],
        )
        if step == cfg.probe_steps:
            break
        seed = result["u"].copy()
        optimizer.step()
        project_(s, n)
    reg = study.regularizer(pack(s, n))
    gs, gn = torch.autograd.grad(reg, (s, n))
    data_norm = float(
        np.sqrt(
            np.sum(result["strength_gradient"] ** 2)
            + np.sum(result["axis_gradient"] ** 2)
        )
    )
    regularizer_norm = float(torch.sqrt(gs.square().sum() + gn.square().sum()))
    assert regularizer_norm > 0
    coefficient = data_norm / regularizer_norm
    calibration = {
        "base_coefficient": coefficient,
        "data_gradient_norm": data_norm,
        "regularizer_gradient_norm": regularizer_norm,
        "probe_steps": cfg.probe_steps,
        "probe": rows,
        "beta": cfg.beta,
        "loss_normalization": study.normalization,
        "initial_data_objective": rows[0]["data_objective"],
        "smooth_length_m": cfg.smooth_length_m,
        "learning_rate": cfg.learning_rate,
        "adam_eps": cfg.adam_eps,
        "axis_initialization": "Shared previous neutral gradient-derived axes; exact s=0 and u=0; no fitted deformation.",
        "coefficient_rule": "Base coefficient equalizes Euclidean strength-plus-tangent-axis gradient norms at the fixed 8-update data-only probe; all compared branches restart neutral.",
    }
    write_json(out / "calibration.json", calibration)
    save_state(
        out / "coefficient-probe.npz",
        {"s": s, "n": n, "coefficient": 0.0},
        result,
        cfg.probe_steps,
    )
    try:
        derivative_check(study, axes, out)
    except Exception as error:
        path = out / "gradient-validation.json"
        gate = json.loads(path.read_text()) if path.exists() else {}
        gate.update(status="failed", failure=f"{type(error).__name__}: {error}")
        write_json(path, gate)
        raise


def derivative_check(study: Study, axes: np.ndarray, out: Path) -> None:
    # Small axis perturbations require equilibrium accuracy beyond training tolerance.
    p = study.physics
    saved_optimizer = p.forward.optimizer
    saved_tolerance = p.forward_tolerance.copy()
    p.forward.optimizer = p.forward.default_optimizer(
        max_steps=10000, rtol=1e-6, atol=1e-12
    )
    p.forward_tolerance.update(max_steps=10000, rtol=1e-6, atol=1e-12)
    s, n = parameters(axes)
    with torch.no_grad():
        s.fill_(0.05)
    center = study.evaluate_axis(
        s, n, np.zeros_like(study.physics.points), 0.0, backward=True
    )
    gs, gn = center["strength_gradient"], center["axis_gradient"]
    tangent = gn - np.sum(gn * axes, axis=1)[:, None] * axes
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-30)
    directions = {
        "strength": (np.sign(gs), np.zeros_like(axes)),
        "axis": (np.zeros(len(axes)), tangent),
    }
    checks = []
    for name, (ds, dn) in directions.items():
        analytic = float(np.sum(gs * ds) + np.sum(gn * dn))
        assert abs(analytic) > 1e-8
        slopes = []
        for epsilon in (0.01, 0.005):
            values = []
            for sign in (-1, 1):
                LOG.info(
                    "Full-chain derivative %s epsilon=%g sign=%d", name, epsilon, sign
                )
                trial_s = torch.as_tensor(
                    np.full(len(axes), 0.05) + sign * epsilon * ds
                )
                trial_n = torch.as_tensor(axes + sign * epsilon * dn)
                result = study.evaluate_axis(
                    trial_s, trial_n, center["u"], 0.0, backward=False
                )
                values.append(result["objective"])
            numeric = (values[1] - values[0]) / (2 * epsilon)
            slopes.append(numeric)
            checks.append(
                {
                    "direction": name,
                    "epsilon": epsilon,
                    "analytic": analytic,
                    "numeric": numeric,
                    "relative_error": abs(numeric - analytic) / abs(analytic),
                }
            )
            write_json(
                out / "gradient-validation.json",
                {"status": "running", "checks": checks},
            )
        assert abs(slopes[-1] - analytic) / abs(analytic) < 0.02, checks
        assert abs(slopes[-1] - slopes[0]) / abs(analytic) < 0.02, checks
    write_json(
        out / "gradient-validation.json",
        {
            "status": "passed",
            "checks": checks,
            "relative_tolerance": 0.02,
            "strength_at_check": 0.05,
            "validation_forward_tolerance": p.forward_tolerance,
            "training_forward_tolerance": saved_tolerance,
        },
    )
    p.forward.optimizer = saved_optimizer
    p.forward_tolerance = saved_tolerance


def persist(case: dict, folder: Path, row: dict, result: dict, cfg: Config) -> None:
    step = row["step"]
    case["rows"].append(row)
    if case["best"] is None or row["objective"] < case["best"]["objective"]:
        case["best"] = row.copy()
        save_state(folder / "best.npz", case, result, step)
    if row["inverted_all_cells"] == 0 and (
        case["best_noninverted"] is None
        or row["objective"] < case["best_noninverted"]["objective"]
    ):
        case["best_noninverted"] = row.copy()
        save_state(folder / "best-noninverted.npz", case, result, step)
    save_state(folder / "last.npz", case, result, step)
    if step % cfg.checkpoint_interval == 0 or step == cfg.steps:
        save_state(folder / f"step-{step:04d}.npz", case, result, step)
    torch.save(
        {
            "step": step,
            "s": case["s"].detach().cpu(),
            "n": case["n"].detach().cpu(),
            "u": result["u"],
            "optimizer": case["optimizer"].state_dict(),
            "scheduler": case["scheduler"].state_dict(),
            "coefficient": case["coefficient"],
        },
        folder / "optimizer-latest.pt",
    )
    with (folder / "trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(case["rows"])
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
            "status": case["status"],
            "last_step": step,
            "coefficient": case["coefficient"],
            "initial_metrics": case["rows"][0],
            "last_metrics": row,
            "best_metrics": case["best"],
            "best_noninverted_metrics": case["best_noninverted"],
            "failure": None,
        },
    )


def run_cases(  # noqa: C901, PLR0912, PLR0915
    study: Study,
    cfg: Config,
    out: Path,
    axes: np.ndarray,
    branches: dict[str, dict],
) -> None:
    cases = {}
    for name, branch in branches.items():
        coefficient = branch["coefficient"]
        (out / name).mkdir()
        s, n = parameters(axes)
        optimizer = optimizer_for(s, n, cfg)
        cases[name] = {
            "s": s,
            "n": n,
            "kind": branch["kind"],
            "beta": branch["beta"],
            "optimizer": optimizer,
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=100, gamma=0.5
            ),
            "seed": np.zeros_like(study.physics.points),
            "coefficient": coefficient,
            "rows": [],
            "best": None,
            "best_noninverted": None,
            "status": "running",
            "convergence_hits": 0,
            "seconds": 0.0,
        }
    initial_steps = {}
    for step in range(cfg.steps + 1):
        for name, case in cases.items():
            if case["status"] != "running":
                continue
            study.set_loss(case["kind"], case["beta"])
            tick = time.perf_counter()
            try:
                result = study.evaluate_axis(
                    case["s"],
                    case["n"],
                    case["seed"],
                    case["coefficient"],
                    backward=True,
                )
                row = {
                    "step": step,
                    **study.axis_metrics(
                        case["s"].detach().cpu().numpy(),
                        case["n"].detach().cpu().numpy(),
                        result,
                        axes,
                    ),
                }
                row["step_seconds"] = time.perf_counter() - tick
                case["seconds"] += row["step_seconds"]
                row["elapsed_seconds"] = case["seconds"]
                row["learning_rate"] = case["optimizer"].param_groups[0]["lr"]
                initial_pg = (
                    row["projected_gradient_rms"]
                    if not case["rows"]
                    else case["rows"][0]["projected_gradient_rms"]
                )
                row["projected_gradient_ratio"] = (
                    row["projected_gradient_rms"] / initial_pg
                )
                row["recent_objective_relative_span"] = 1.0
                if step >= cfg.min_steps and step % cfg.convergence_window == 0:
                    recent = [
                        r["objective"] for r in case["rows"][-cfg.convergence_window :]
                    ] + [row["objective"]]
                    span = (max(recent) - min(recent)) / max(abs(recent[0]), 1e-30)
                    row["recent_objective_relative_span"] = span
                    passed = (
                        span < cfg.plateau_tolerance
                        and row["projected_gradient_ratio"] < cfg.stationarity_ratio
                    )
                    case["convergence_hits"] = (
                        case["convergence_hits"] + 1 if passed else 0
                    )
                    if case["convergence_hits"] >= 2:
                        case["status"] = "stationarity_and_plateau_thresholds_met"
                if step == cfg.steps and case["status"] == "running":
                    case["status"] = "completed_budget_not_convergence_certified"
                persist(case, out / name, row, result, cfg)
                cherries.log_metrics(
                    {
                        f"{name}/{key}": row[key]
                        for key in (
                            "objective",
                            "data_objective",
                            "activation_smoothness",
                            "fit_rms_mm",
                            "surface_gradient_rms",
                            "projected_gradient_ratio",
                            "inverted_all_cells",
                        )
                    },
                    step=step,
                )
                LOG.info(
                    "%s %d/%d: fit %.4f mm, grad %.6f, R %.6g, PG ratio %.4g, inversions %d, %.1fs",
                    name,
                    step,
                    cfg.steps,
                    row["fit_rms_mm"],
                    row["surface_gradient_rms"],
                    row["activation_smoothness"],
                    row["projected_gradient_ratio"],
                    row["inverted_all_cells"],
                    row["step_seconds"],
                )
                if step == 0:
                    with torch.no_grad():
                        proposed_s = torch.clamp(
                            -cfg.learning_rate
                            * case["s"].grad
                            / (case["s"].grad.abs() + cfg.adam_eps),
                            min=0,
                        )
                        proposed_q = pack(proposed_s, case["n"]).cpu().numpy()
                    initial_steps[name] = proposed_q
                    np.savez_compressed(
                        out / f"initial-update-{name}.npz", q=proposed_q
                    )
                if case["status"] == "running":
                    case["seed"] = result["u"].copy()
                    case["optimizer"].step()
                    project_(case["s"], case["n"])
                    case["scheduler"].step()
            except Exception as error:
                failure = {
                    "step": step,
                    "type": type(error).__name__,
                    "message": str(error),
                }
                write_json(out / name / "failure.json", failure)
                summary_path = out / name / "summary.json"
                summary = (
                    json.loads(summary_path.read_text())
                    if summary_path.exists()
                    else {"last_step": None}
                )
                summary.update(status="failed", failure=failure)
                write_json(summary_path, summary)
                raise
        if all(case["status"] != "running" for case in cases.values()):
            break

    step_diagnostics = {}
    for name, dq in initial_steps.items():
        weights = study.active_weights
        component_weights = np.array([1, 1, 1, 2, 2, 2])
        norm2 = float(np.sum(weights[:, None] * component_weights * dq**2))
        cosine = {}
        for other, other_dq in initial_steps.items():
            other_norm2 = float(
                np.sum(weights[:, None] * component_weights * other_dq**2)
            )
            assert norm2 > 0
            assert other_norm2 > 0
            cosine[other] = float(
                np.sum(weights[:, None] * component_weights * dq * other_dq)
                / np.sqrt(norm2 * other_norm2)
            )
        step_diagnostics[name] = {
            "physical_B_step_rms": np.sqrt(norm2),
            "cosine_to_other_initial_updates": cosine,
        }
    write_json(out / "initial-update-diagnostics.json", step_diagnostics)


def shared_axes(out: Path) -> np.ndarray:
    gate = json.loads((AXIS_PREP / "gradient-validation.json").read_text())
    assert gate["status"] == "passed"
    activation_gate = AXIS / "data/05-activation/checks.json"
    assert json.loads(activation_gate.read_text())["passed"]
    source = AXIS_PREP / "initialization.npz"
    shutil.copy2(source, out / "initialization.npz")
    axes = np.load(source)["axes"]
    assert np.max(np.abs(np.linalg.norm(axes, axis=1) - 1)) < 1e-12
    return axes


def assert_gate(preparation: Path, protocol: dict) -> dict:
    gate = json.loads((preparation / "gradient-validation.json").read_text())
    assert gate["status"] == "passed", gate
    prepared = json.loads((preparation / "protocol.json").read_text())
    for key, source in prepared["sources"].items():
        assert protocol["sources"][key]["sha256"] == source["sha256"], key
    assert prepared["fixture"] == protocol["fixture"]
    assert prepared["loss_normalization"] == protocol["loss_normalization"]
    assert prepared["initial_axes"] == protocol["initial_axes"]
    return prepared


def main(cfg: Config) -> None:
    assert cfg.phase in {"validate", "loss-pilot", "prepare", "smooth-pilot", "compare"}
    assert cfg.steps >= 0
    assert cfg.beta > 0
    if cfg.phase in {"loss-pilot", "smooth-pilot"}:
        cfg.steps = cfg.pilot_steps
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    study = Study(cfg.smooth_length_m)
    study.normalization["formula"] = (
        "D_beta=K*(L2/L20+beta*Lg/Lg0)/(1+beta); "
        "pure L2=L2; pure gradient=K*Lg/Lg0; K=L20"
    )
    study.set_loss("mixed", cfg.beta)
    cpu_gate_path = GROUP / "data/05-loss/checks.json"
    cpu_gate = json.loads(cpu_gate_path.read_text())
    assert cpu_gate["passed"]
    for filename, sha256 in cpu_gate["source_sha256"].items():
        assert digest(GROUP / "src" / filename) == sha256
    axes = shared_axes(out)
    p = study.physics
    assert len(axes) == len(p.ids)
    np.savez_compressed(
        out / "mesh.npz",
        rest_points=p.points,
        skin_ids=study.skin_ids,
        triangles=study.triangles,
        target_displacement_skin=p.target[study.skin_ids],
        skin_vertex_weights=study.weights,
        initial_u=np.zeros_like(p.points),
        active_ids=p.ids,
        active_volumes=p.volumes,
        edge_i=p.graph[0],
        edge_j=p.graph[1],
        edge_weight=p.graph[2],
    )
    protocol = {
        "config": cfg.model_dump(mode="json"),
        "activation": "B=I+snnT; s>=0, unit n; learned strength and axis; 3 physical DoF per active cell; clamp/normalize after Adam",
        "start": "Exact neutral s=0, B=I, u=0; same inherited neutral-gradient axes and fresh Adam for every branch",
        "initial_axes": {
            "source": str(AXIS_PREP / "initialization.npz"),
            "sha256": digest(AXIS_PREP / "initialization.npz"),
        },
        "activation_gate": {
            "path": str(AXIS / "data/05-activation/checks.json"),
            "sha256": digest(AXIS / "data/05-activation/checks.json"),
        },
        "loss_normalization": study.normalization,
        "selected_beta": cfg.beta,
        "learning_rate_schedule": "Same Adam lr=0.3, eps=0.01, betas=(0.9,0.999), halved every100updates; no objective-dependent scheduling",
        "smoothness": "ell^2/V_active times same-muscle face conductance weighted squared Frobenius differences in B; ell=5mm",
        "materials": p.material_spec,
        "forward_tolerance": p.forward_tolerance,
        "fixture": {
            name: {"path": str(FIXTURE / name), "sha256": digest(FIXTURE / name)}
            for name in ("volume.vtu", "skin.vtp", "summary.json")
        },
        "sources": archive(out),
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": [sys.executable, *sys.argv],
        },
        "convergence": "Two consecutive25update checks after100 with own-objective relative span<0.001 and physical rank-one projected-gradient RMS ratio<0.01; no mechanical stability certification",
        "checkpoint_policy": "Uncompressed NPZ endpoint/best files and every25update checkpoints; scalar traces and solver receipts every evaluation",
    }
    write_json(out / "protocol.json", protocol)
    if cfg.phase == "validate":
        try:
            derivative_check(study, axes, out)
        except Exception as error:
            path = out / "gradient-validation.json"
            gate = json.loads(path.read_text()) if path.exists() else {}
            gate.update(status="failed", failure=f"{type(error).__name__}: {error}")
            write_json(path, gate)
            raise
        return
    assert_gate(cfg.validation, protocol)
    if cfg.phase == "prepare":
        prepare(study, cfg, out, axes)
        return
    if cfg.phase == "loss-pilot":
        branches = {
            "l2": {"kind": "l2", "beta": 0.0, "coefficient": 0.0},
            "gradient": {"kind": "gradient", "beta": 0.0, "coefficient": 0.0},
            **{
                f"beta-{beta:g}": {"kind": "mixed", "beta": beta, "coefficient": 0.0}
                for beta in (0.25, 1.0, 4.0)
            },
        }
    else:
        prepared = assert_gate(cfg.preparation, protocol)
        assert prepared["selected_beta"] == cfg.beta
        calibration = json.loads((cfg.preparation / "calibration.json").read_text())
        for key in ("beta", "learning_rate", "adam_eps", "smooth_length_m"):
            assert calibration[key] == getattr(cfg, key)
        shutil.copy2(cfg.preparation / "calibration.json", out / "calibration.json")
        if cfg.phase == "smooth-pilot":
            coefficients = {
                "off": 0.0,
                **{
                    f"factor-{factor:g}": factor * calibration["base_coefficient"]
                    for factor in (0.1, 1.0, 10.0)
                },
            }
        else:
            assert cfg.coefficient is not None
            assert cfg.coefficient > 0
            coefficients = {"mixed-off": 0.0, "mixed-on": cfg.coefficient}
        branches = {
            name: {"kind": "mixed", "beta": cfg.beta, "coefficient": coefficient}
            for name, coefficient in coefficients.items()
        }
        write_json(out / "coefficients.json", coefficients)
    protocol["branches"] = branches
    write_json(out / "protocol.json", protocol)
    run_cases(study, cfg, out, axes, branches)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileAxisExperiment)
