"""Fresh four-way unrestricted face comparison with positional and normal losses."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import torch
from experiment import Profile
from study import FIXTURE, ROOT, Study

from liblaf import cherries

LOG = logging.getLogger(__name__)
VARIANTS = ("smooth-off-l2", "smooth-off-normal", "smooth-on-l2", "smooth-on-normal")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("10-comparison")
    steps: int = 100
    learning_rate: float = 0.3
    adam_eps: float = 0.01
    beta: float = 0.05
    smooth_coefficient: float = 0.003214147722027223
    smooth_length_m: float = 0.005
    checkpoint_interval: int = 10
    validation_only: bool = False
    gate: Path = Path("06-validation")
    branches: str = ",".join(VARIANTS)


def write_json(path: Path, data: object) -> None:
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
        if not source or not source.endswith(".py") or not Path(source).is_absolute():
            continue
        path = Path(source).resolve()
        if path.is_relative_to(ROOT / "exp") or name.startswith(
            ("liblaf.apple", "liblaf.peach")
        ):
            relative = (
                path.relative_to(ROOT)
                if path.is_relative_to(ROOT)
                else Path("runtime") / Path(*name.split(".")).with_suffix(".py")
            )
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


def gradient_check(study: Study, cfg: Config, out: Path) -> None:
    p = study.physics
    original_optimizer = p.forward.optimizer
    original_tolerance = p.forward_tolerance.copy()
    p.forward.optimizer = p.forward.default_optimizer(
        max_steps=10000, rtol=1e-6, atol=1e-12
    )
    p.forward_tolerance.update(max_steps=10000, rtol=1e-6, atol=1e-12)
    rng = np.random.default_rng(20260921)
    center_np = 0.005 * rng.choice([-1.0, 1.0], size=(p.n_regions, 6))[p.region]
    center_np += 0.001 * rng.standard_normal(center_np.shape)
    q = torch.nn.Parameter(torch.as_tensor(center_np))
    rows = []
    try:
        for kind, beta, coefficient, normal_only in (
            ("normal-only", 1.0, 0.0, True),
            ("combined-with-smoothness", cfg.beta, cfg.smooth_coefficient, False),
        ):
            center = study.evaluate_normal(
                q,
                np.zeros_like(p.points),
                beta,
                coefficient,
                backward=True,
                normal_only=normal_only,
            )
            grad = center["gradient"]
            directions = {
                "gradient-sign": np.sign(grad),
                "muscle-coherent": rng.choice([-1.0, 1.0], size=(p.n_regions, 6))[
                    p.region
                ],
            }
            for name, direction in directions.items():
                analytic = float(np.sum(grad * direction))
                assert abs(analytic) > 1e-8
                slopes = []
                for epsilon in (0.002, 0.001):
                    values = []
                    for sign in (-1, 1):
                        LOG.info(
                            "Implicit check %s %s eps=%g sign=%d",
                            kind,
                            name,
                            epsilon,
                            sign,
                        )
                        result = study.evaluate_normal(
                            torch.as_tensor(center_np + sign * epsilon * direction),
                            center["u"],
                            beta,
                            coefficient,
                            backward=False,
                            normal_only=normal_only,
                        )
                        values.append(result["objective"])
                    numeric = (values[1] - values[0]) / (2 * epsilon)
                    slopes.append(numeric)
                    rows.append(
                        {
                            "objective": kind,
                            "direction": name,
                            "epsilon": epsilon,
                            "analytic": analytic,
                            "numeric": numeric,
                            "relative_error": abs(numeric - analytic) / abs(analytic),
                        }
                    )
                    write_json(
                        out / "gradient-validation.json",
                        {"status": "running", "checks": rows},
                    )
                assert abs(slopes[-1] - analytic) / abs(analytic) < 0.02, rows
                assert abs(slopes[-1] - slopes[0]) / abs(analytic) < 0.02, rows
        write_json(
            out / "gradient-validation.json",
            {
                "status": "passed",
                "checks": rows,
                "tolerance": 0.02,
                "forward_tolerance": dict(p.forward_tolerance),
                "seed": 20260921,
            },
        )
    except Exception as error:
        write_json(
            out / "gradient-validation.json",
            {"status": "failed", "checks": rows, "error": str(error)},
        )
        raise
    finally:
        p.forward.optimizer = original_optimizer
        p.forward_tolerance = original_tolerance


def save_state(path: Path, q: np.ndarray, result: dict, step: int) -> None:
    np.savez_compressed(
        path,
        q=q,
        u=result["u"],
        step=step,
        solver_valid=True,
        physical_volume_energy=True,
    )


def run_branch(study: Study, cfg: Config, out: Path, branch: str) -> dict:  # noqa: PLR0915
    folder = out / branch
    folder.mkdir()
    beta = cfg.beta if branch.endswith("normal") else 0.0
    coefficient = cfg.smooth_coefficient if branch.startswith("smooth-on") else 0.0
    study.physics.diff.last_adjoint_solution = None
    q = torch.nn.Parameter(torch.zeros((len(study.physics.ids), 6)))
    optimizer = torch.optim.Adam(
        [q], lr=cfg.learning_rate, eps=cfg.adam_eps, betas=(0.9, 0.999)
    )
    seed = np.zeros_like(study.physics.points)
    initial_q = q.detach().cpu().numpy().copy()
    assert not optimizer.state
    assert np.count_nonzero(initial_q) == 0
    np.savez_compressed(
        folder / "initial-state.npz",
        q=initial_q,
        u=seed,
        m=initial_q,
        v=initial_q,
        step=0,
        activation_identity=True,
        adjoint_initial_guess_zero=True,
    )
    rows = []
    best = best_noninverted = None
    status, failure = "running", None
    start = time.perf_counter()
    initial_gradient = None
    try:
        for step in range(cfg.steps + 1):
            tick = time.perf_counter()
            result = study.evaluate_normal(q, seed, beta, coefficient, backward=True)
            q_numpy = q.detach().cpu().numpy().copy()
            metrics = study.normal_metrics(q_numpy, result)
            if initial_gradient is None:
                initial_gradient = metrics["physical_gradient_rms"]
                assert initial_gradient > 0
                delta = (
                    -cfg.learning_rate
                    * result["gradient"]
                    / (np.abs(result["gradient"]) + cfg.adam_eps)
                )
                np.savez_compressed(
                    folder / "initial-gradient.npz",
                    gradient=result["gradient"],
                    adam_delta=delta,
                )
                write_json(
                    folder / "initial-update.json",
                    {
                        "physical_B_update_rms": study.physical_step_rms(delta),
                        "beta": beta,
                        "smooth_coefficient": coefficient,
                    },
                )
            row = {
                "step": step,
                **metrics,
                "physical_gradient_ratio": metrics["physical_gradient_rms"]
                / initial_gradient,
                "learning_rate": cfg.learning_rate,
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
                    "branch": branch,
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
                    f"{branch}/{key}": row[key]
                    for key in (
                        "objective",
                        "fit_rms_mm",
                        "normal_angle_rms_deg",
                        "activation_smoothness",
                        "inverted_all_cells",
                        "physical_gradient_ratio",
                    )
                },
                step=step,
            )
            LOG.info(
                "%s %d/%d: fit %.4f mm, normal %.3f deg, R %.5g, inversions %d, gradient ratio %.4g; %.1fs",
                branch,
                step,
                cfg.steps,
                row["fit_rms_mm"],
                row["normal_angle_rms_deg"],
                row["activation_smoothness"],
                row["inverted_all_cells"],
                row["physical_gradient_ratio"],
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
        LOG.exception(
            "Branch %s failed; preserving failure and running independent remaining branches",
            branch,
        )
    summary = {
        "status": status,
        "last_step": rows[-1]["step"] if rows else None,
        "initial_metrics": rows[0] if rows else None,
        "last_metrics": rows[-1] if rows else None,
        "best_metrics": best,
        "best_noninverted_metrics": best_noninverted,
        "failure": failure,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json(folder / "summary.json", summary)
    return summary


def main(cfg: Config) -> None:
    assert cfg.steps >= 0
    assert cfg.checkpoint_interval > 0
    assert cfg.beta > 0
    assert cfg.smooth_coefficient > 0
    branches = cfg.branches.split(",")
    assert len(set(branches)) == len(branches)
    assert set(branches) <= set(VARIANTS)
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    study = Study(cfg.smooth_length_m)
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
        tets=p.tets,
        edge_i=p.graph[0],
        edge_j=p.graph[1],
        edge_weight=p.graph[2],
        active_volume_weights=study.active_weights,
        regularizer_factor=study.regularizer_factor,
        fixed_mask=np.asarray(p.mesh.point_data["FixedMask"], bool),
        fixed_values=np.asarray(p.mesh.point_data["FixedValue"]),
    )
    sources = archive(out)
    fixture = {
        name: {"path": str(FIXTURE / name), "sha256": digest(FIXTURE / name)}
        for name in ("volume.vtu", "skin.vtp", "summary.json")
    }
    protocol = {
        "config": cfg.model_dump(mode="json"),
        "normalization": study.normalization,
        "start": "q=0, B=I, u=0, fresh zero Adam moments; no resume or fitted inputs",
        "activation": "288235 independent Raw6 symmetric tensors, no projections",
        "objective": "L2 + beta*(L20/N0)*normal_chord_squared + smooth_coefficient*R",
        "materials": p.material_spec,
        "forward_tolerance": p.forward_tolerance,
        "surface_points": len(study.skin_ids),
        "surface_triangles": len(study.triangles),
        "volume_points": len(p.points),
        "tetrahedra": len(p.tets),
        "active_cells": len(p.ids),
        "fixture": fixture,
        "sources": sources,
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": [sys.executable, *sys.argv],
        },
        "inverse_stationarity_claimed": False,
        "mechanical_stability_claimed": False,
    }
    write_json(out / "protocol.json", protocol)
    shutil.copy2(Path(__file__).parents[1] / "docs/00-protocol.md", out / "protocol.md")
    if cfg.validation_only:
        gradient_check(study, cfg, out)
        return
    gate_dir = cherries.input(cfg.gate)
    gate = json.loads((gate_dir / "gradient-validation.json").read_text())
    assert gate["status"] == "passed"
    gate_protocol = json.loads((gate_dir / "protocol.json").read_text())
    assert {key: value["sha256"] for key, value in sources.items()} == {
        key: value["sha256"] for key, value in gate_protocol["sources"].items()
    }
    assert fixture == gate_protocol["fixture"]
    assert study.normalization == gate_protocol["normalization"]
    for key in (
        "learning_rate",
        "adam_eps",
        "beta",
        "smooth_coefficient",
        "smooth_length_m",
    ):
        assert cfg.model_dump()[key] == gate_protocol["config"][key]
    shutil.copy2(
        gate_dir / "gradient-validation.json", out / "gradient-validation.json"
    )
    cpu_gate = Path(__file__).parents[1] / "data/05-normal-verification/checks.json"
    assert json.loads(cpu_gate.read_text())["passed"]
    shutil.copy2(cpu_gate, out / "normal-validation.json")
    summaries = {branch: run_branch(study, cfg, out, branch) for branch in branches}
    write_json(out / "summary.json", summaries)
    assert all(row["failure"] is None for row in summaries.values()), (
        "One or more independent branches failed; inspect preserved receipts."
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
