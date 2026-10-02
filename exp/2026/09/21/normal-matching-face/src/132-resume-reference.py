"""Recover the interrupted reference-length fit, then fit smooth-on from neutral."""

# ruff: noqa: PLR0915

from __future__ import annotations

import copy
import csv
import importlib.util
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
from reference_study import ReferenceStudy
from study import ROOT

from liblaf import cherries

spec = importlib.util.spec_from_file_location(
    "original_reference_runner", Path(__file__).with_name("120-run-reference.py")
)
assert spec is not None
assert spec.loader is not None
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
initial = reference.legacy
LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    parent_dir: Path = Path("130-reference-fit")
    output: Path = Path("132-reference-continuation")
    gate: Path = Path("131-reference-resume-checks/checks.json")
    steps: int = 200


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def receipt(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": initial.digest(path)}


def load_trace(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = [
            {
                key: int(value) if key == "step" else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(stream)
        ]
    assert [row["step"] for row in rows] == list(range(len(rows)))
    return rows


def assert_restored(q: torch.Tensor, optimizer: torch.optim.Adam, saved: dict) -> dict:
    state = optimizer.state[q]
    original = next(iter(saved["optimizer"]["state"].values()))
    result = {
        "step": int(state["step"]),
        "q_equal": torch.equal(q.detach().cpu(), saved["q"]),
        "m_equal": torch.equal(state["exp_avg"].cpu(), original["exp_avg"]),
        "v_equal": torch.equal(state["exp_avg_sq"].cpu(), original["exp_avg_sq"]),
    }
    assert result["step"] == saved["step"]
    assert result["q_equal"]
    assert result["m_equal"]
    assert result["v_equal"]
    assert optimizer.param_groups[0]["lr"] == 0.3
    assert (
        optimizer.param_groups[0]["eps"] == saved["optimizer"]["param_groups"][0]["eps"]
    )
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    return result


def replay(
    study: ReferenceStudy,
    q: torch.Tensor,
    optimizer: torch.optim.Adam,
    saved: dict,
    old_metrics: dict,
    beta: float,
    coefficient: float,
    folder: Path,
) -> dict:
    study.physics.diff.last_adjoint_solution = None
    restored = assert_restored(q, optimizer, saved)
    result = study.evaluate_normal(
        q, saved["u"].copy(), beta, coefficient, backward=True
    )
    metrics = study.normal_metrics(q.detach().cpu().numpy(), result)
    displacement_error = float(np.max(np.abs(result["u"] - saved["u"])))
    limits = {
        "objective": 1e-4 * abs(old_metrics["objective"]),
        "position_contribution": 1e-4 * abs(old_metrics["position_contribution"]),
        "fit_rms_mm": 1e-3,
        "normal_angle_rms_deg": 0.01,
        "normal_loss": 1e-4 * abs(old_metrics["normal_loss"]),
        "activation_smoothness": 1e-12
        * max(1.0, abs(old_metrics["activation_smoothness"])),
        "detF_min": 0.001,
        "inverted_all_cells": 0.0,
    }
    errors = {key: abs(float(metrics[key]) - float(old_metrics[key])) for key in limits}
    gradient_difference = abs(
        metrics["physical_gradient_rms"] / old_metrics["physical_gradient_rms"] - 1
    )
    errors["forward_displacement_max_abs_difference_m"] = displacement_error
    errors["physical_gradient_rms_relative_difference"] = gradient_difference
    limits["forward_displacement_max_abs_difference_m"] = 1e-6
    limits["physical_gradient_rms_relative_difference"] = 0.02
    passed = all(error <= limits[key] for key, error in errors.items())
    assert assert_restored(q, optimizer, saved) == restored
    np.savez_compressed(
        folder / "resume-gradient.npz",
        q=q.detach().cpu().numpy(),
        u=result["u"],
        gradient=result["gradient"],
        step=saved["step"],
    )
    initial.write_json(
        folder / "resume-replay.json",
        {
            "passed": passed,
            "step": saved["step"],
            "metrics": metrics,
            "forward": result["forward"],
            "adjoint": result["adjoint"],
            "errors": errors,
            "error_limits": limits,
            "physical_gradient_rms_relative_difference": gradient_difference,
            "forward_displacement_max_abs_difference_m": displacement_error,
            "optimizer_state": restored,
        },
    )
    assert passed, errors
    LOG.info(
        "%s replay %d: displacement difference %.3g m, gradient RMS difference %.3g%%",
        folder.name,
        saved["step"],
        displacement_error,
        100 * gradient_difference,
    )
    return result


def run_branch(
    study: ReferenceStudy,
    cfg: Config,
    settings: dict,
    source: Path,
    out: Path,
    branch: str,
) -> dict:
    parent = source / branch
    folder = out / branch
    folder.mkdir()
    for name in (
        "initial-state.npz",
        "initial-gradient.npz",
        "initial-update.json",
        "best.npz",
        "best-noninverted.npz",
        "last.npz",
        "optimizer-latest.pt",
        "trace.csv",
        "solver-receipts.jsonl",
    ):
        shutil.copy2(parent / name, folder / name)
    for path in parent.glob("step-*.npz"):
        shutil.copy2(path, folder / path.name)
    old = read_json(parent / "summary.json")
    shutil.copy2(parent / "summary.json", folder / "parent-summary.json")
    saved = torch.load(
        parent / "optimizer-latest.pt", map_location="cpu", weights_only=False
    )
    start_step = int(saved["step"])
    assert saved["branch"] == branch
    assert start_step == old["last_step"] == 102
    assert old["status"] == "running"
    assert old["failure"] is None
    assert cfg.steps > start_step
    with np.load(parent / "last.npz", allow_pickle=False) as last:
        assert int(last["step"]) == start_step
        assert np.array_equal(saved["q"].numpy(), last["q"])
        assert np.array_equal(saved["u"], last["u"])
    torch.save(saved, folder / "continuation-start.pt")
    q = torch.nn.Parameter(saved["q"].to(device=study.target.device).clone())
    optimizer = torch.optim.Adam(
        [q], lr=settings["learning_rate"], eps=settings["adam_eps"], betas=(0.9, 0.999)
    )
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    assert_restored(q, optimizer, saved)
    rows = load_trace(parent / "trace.csv")
    assert rows[-1]["step"] == start_step
    best = old["best_metrics"].copy()
    best_noninverted = old["best_noninverted_metrics"].copy()
    beta = settings["beta"] if branch.endswith("normal") else 0.0
    coefficient = (
        settings["smooth_coefficient"] if branch.startswith("smooth-on") else 0.0
    )
    initial_gradient = rows[0]["physical_gradient_rms"]
    offset = rows[-1]["elapsed_seconds"]
    started = time.perf_counter()
    status, failure, step = "running", None, start_step

    def summary() -> dict:
        return {
            "status": status,
            "last_step": rows[-1]["step"],
            "initial_metrics": old["initial_metrics"],
            "last_metrics": rows[-1],
            "best_metrics": best,
            "best_noninverted_metrics": best_noninverted,
            "failure": failure,
            "elapsed_seconds": offset + time.perf_counter() - started,
            "continuation_parent": str(parent),
            "continuation_from_step": start_step,
            "continuation_updates": rows[-1]["step"] - start_step,
            "continuation_elapsed_seconds": time.perf_counter() - started,
        }

    try:
        result = replay(
            study, q, optimizer, saved, old["last_metrics"], beta, coefficient, folder
        )
        seed = result["u"].copy()
        optimizer.step()
        for step in range(start_step + 1, cfg.steps + 1):
            tick = time.perf_counter()
            result = study.evaluate_normal(q, seed, beta, coefficient, backward=True)
            q_numpy = q.detach().cpu().numpy().copy()
            metrics = study.normal_metrics(q_numpy, result)
            assert int(optimizer.state[q]["step"]) == step
            row = {
                "step": step,
                **metrics,
                "physical_gradient_ratio": metrics["physical_gradient_rms"]
                / initial_gradient,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": offset + time.perf_counter() - started,
                "step_seconds": time.perf_counter() - tick,
            }
            assert set(row) == set(rows[0])
            rows.append(row)
            if row["objective"] < best["objective"]:
                best = row.copy()
                initial.save_state(folder / "best.npz", q_numpy, result, step)
            if (
                row["inverted_all_cells"] == 0
                and row["objective"] < best_noninverted["objective"]
            ):
                best_noninverted = row.copy()
                initial.save_state(
                    folder / "best-noninverted.npz", q_numpy, result, step
                )
            if (
                step == start_step + 1
                or step % settings["checkpoint_interval"] == 0
                or step == cfg.steps
            ):
                initial.save_state(
                    folder / f"step-{step:04d}.npz", q_numpy, result, step
                )
            initial.save_state(folder / "last.npz", q_numpy, result, step)
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
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
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
            initial.write_json(folder / "summary.json", summary())
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
        initial.write_json(folder / "failure.json", failure)
        np.savez_compressed(
            folder / "failed-proposal.npz",
            q=q.detach().cpu().numpy(),
            step=step,
            solver_valid=False,
        )
        LOG.exception(
            "%s stopped; preserving its last accepted state and continuing the independent branches",
            branch,
        )
    result_summary = summary()
    initial.write_json(folder / "summary.json", result_summary)
    return result_summary


def main(cfg: Config) -> None:
    source = cherries.input(cfg.parent_dir).resolve()
    parent_protocol = read_json(source / "protocol.json")
    cpu_path = cherries.input(cfg.gate).resolve()
    cpu_gate = read_json(cpu_path)
    assert cpu_gate["passed"] is True
    assert cpu_gate["parent_protocol_record"] == receipt(source / "protocol.json")
    for record in cpu_gate["parent_artifacts"]:
        assert receipt(Path(record["path"])) == record
    for record in parent_protocol["sources"].values():
        assert initial.digest(Path(record["path"])) == record["sha256"]
        assert initial.digest(Path(record["snapshot"])) == record["sha256"]
    for record in parent_protocol["fixture"].values():
        assert receipt(Path(record["path"])) == record
    preflight = read_json(source / "reference-preflight.json")
    for record in preflight.values():
        assert receipt(Path(record["path"])) == record
    assert cfg.steps == parent_protocol["config"]["steps"] == 200
    assert not (source / "smooth-on-normal").exists()
    assert cpu_gate["branches"]["smooth-off-normal"]["step"] == 102
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    study = ReferenceStudy(
        parent_protocol["selected_loss"]["l_ref_mm"],
        parent_protocol["config"]["smooth_length_m"],
    )
    assert study.normalization == parent_protocol["normalization"]
    assert study.physics.forward_tolerance == parent_protocol["forward_tolerance"]
    assert study.physics.material_spec == parent_protocol["materials"]
    for name in (
        "mesh.npz",
        "loss-config.json",
        "gradient-validation.json",
        "normal-validation.json",
        "reference-preflight.json",
    ):
        shutil.copy2(source / name, out / name)
    protocol_path = Path(__file__).parents[1] / "docs/131-reference-resume-protocol.md"
    protocol = copy.deepcopy(parent_protocol)
    protocol["config"].update(output=str(cfg.output))
    protocol["start"] = (
        "Both branches originate from neutral; smooth-off resumes from q102/u102 "
        "and saved Adam m/v/t=102 after interruption; smooth-on starts fresh neutral. "
        "Adjoint warm start reset at resume; not a bitwise-continuous off trajectory."
    )
    protocol["resume"] = {
        "parent": str(source),
        "parent_protocol_record": receipt(source / "protocol.json"),
        "cpu_gate": receipt(cpu_path),
        "protocol_record": receipt(protocol_path),
        "interruption_record": receipt(source / "interruption.json"),
        "from_step": 102,
        "to_step": cfg.steps,
        "resumed_branches": ["smooth-off-normal"],
        "fresh_branches": ["smooth-on-normal"],
        "parent_artifacts": cpu_gate["parent_artifacts"],
        "request_config": cfg.model_dump(mode="json"),
    }
    protocol["sources"] = initial.archive(out)
    parent_sources = {
        v["path"]: v["sha256"] for v in parent_protocol["sources"].values()
    }
    current_sources = {v["path"]: v["sha256"] for v in protocol["sources"].values()}
    assert parent_sources.items() <= current_sources.items()
    protocol["runtime"] = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": [sys.executable, *sys.argv],
    }
    initial.write_json(out / "protocol.json", protocol)
    initial.write_json(out / "config.json", protocol["config"])
    shutil.copy2(source / "protocol.md", out / "protocol.md")
    shutil.copy2(protocol_path, out / "resume-protocol.md")
    summaries = {}
    summaries["smooth-off-normal"] = run_branch(
        study, cfg, protocol["config"], source, out, "smooth-off-normal"
    )
    effective = initial.Config(_cli_parse_args=False, **protocol["config"])
    summaries["smooth-on-normal"] = initial.run_branch(
        study, effective, out, "smooth-on-normal"
    )
    initial.write_json(out / "summary.json", summaries)
    assert all(row["failure"] is None for row in summaries.values()), summaries
    assert all(row["last_step"] == cfg.steps for row in summaries.values()), summaries


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
