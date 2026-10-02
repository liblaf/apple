"""Run one four-stage, no-skin active-strain inverse chain."""

from __future__ import annotations

import datetime
import math
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import numpy as np
import pydantic_settings as ps
import torch
from experiment import Profile
from run_support import NUMERICAL_FAILURES, archive, receipt, run_stage, write_json
from stress_study import FIXTURE, L_REF_MM, SMOOTH_LENGTH_M, StressStudy

from liblaf import cherries

LOSS = Literal["l2", "l2-normal"]
STAGES = ("symmetric6", "psd6", "rankone_fixed", "rankone_learned")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("46-active-strain-chain")
    mechanics_provenance: Path | None = None
    loss: LOSS = "l2"
    steps: int = 200
    learning_rate: float = 0.05
    adam_eps: float = 1e-8
    smooth_weight: float = 7.2e-7


def portable_source_freeze(out: Path, records: dict[str, dict]) -> dict[str, dict]:
    """Refer only to files inside the output so a copied run remains auditable."""
    portable = {}
    for record in records.values():
        snapshot = Path(record["snapshot"])
        portable[str(snapshot.relative_to(out))] = {
            "sha256": record["sha256"],
            "logical_source_path": str(snapshot.relative_to(out / "sources")),
        }
    write_json(out / "source-freeze.json", portable)
    return portable


def stage_id(loss: LOSS, mode: str) -> str:
    assert mode in STAGES
    return f"{loss}-{mode}"


def normal_weight(loss: LOSS) -> float:
    """Match 2 mm vector RMS to a uniform five-degree normal error."""
    if loss == "l2":
        return 0.0
    return (2.0**2 / (3.0 * L_REF_MM**2)) / (2.0 * (1.0 - math.cos(math.radians(5.0))))


def package_version(*names: str) -> str | None:
    for name in names:
        try:
            return version(name)
        except PackageNotFoundError:
            pass
    return None


def runtime_receipt(out: Path) -> dict[str, str | None]:
    runtime = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "warp": package_version("warp-lang"),
        "cupy": package_version("cupy-cuda13x", "cupy-cuda12x", "cupy"),
        "device": (
            f"cuda:{torch.cuda.get_device_name()}"
            if torch.cuda.is_available()
            else "cpu"
        ),
    }
    write_json(out / "runtime.json", runtime)
    return receipt(out / "runtime.json")


def write_chain_status(
    out: Path,
    *,
    loss: LOSS,
    current: str | None,
    completed: list[str],
    errors: list[dict[str, object]],
    summaries: dict[str, dict],
    timestamps: dict[str, dict[str, str]],
    steps_per_stage: int,
) -> None:
    queue = [stage_id(loss, mode) for mode in STAGES]
    stage_rows = []
    for index, item in enumerate(queue, start=1):
        summary = summaries.get(item, {})
        times = timestamps.get(item, {})
        status = (
            "completed"
            if item in completed
            else "running"
            if item == current
            else "failed"
            if any(error["stage"] == item for error in errors)
            else "queued"
        )
        stage_rows.append(
            {
                "index": index,
                "name": item.removeprefix(f"{loss}-"),
                "status": status,
                "stage_dir": item,
                "budget": summary.get("budget", steps_per_stage),
                "last_step": summary.get("last_step"),
                "started_at": times.get("started_at"),
                "updated_at": times.get("updated_at"),
                "completed_at": times.get("completed_at"),
            }
        )
    write_json(
        out / "chain-status.json",
        {
            "schema": "active-strain-chain-status-v1",
            "chain_id": out.name,
            "activation_model": "strain",
            "loss": loss,
            "status": "failed"
            if errors
            else "completed"
            if len(completed) == len(queue)
            else "running"
            if current is not None or completed
            else "queued",
            "queue": queue,
            "current_stage": None if current is None else queue.index(current) + 1,
            "completed": completed,
            "errors": errors,
            "stages": stage_rows,
        },
    )


def endpoint_diagnostic(
    study: StressStudy,
    folder: Path,
    mode: str,
    normal_weight: float,
    smooth_weight: float,
    summary: dict,
) -> None:
    """Save endpoint components, avoiding an extra solve for the L2-only column."""
    if normal_weight == 0:
        metrics = summary["last_metrics"]
        assert metrics is not None
        write_json(
            folder / "gradient-balance.json",
            {
                "status": "recorded_from_endpoint",
                "checkpoint": receipt(folder / "last.npz"),
                "activation_model": "strain",
                "solver_valid": bool(metrics["solver_valid"]),
                "l2_component": "runner endpoint; no additional component solve for L2-only objective",
                **{
                    key: metrics[key]
                    for key in (
                        "l2_gradient_dual_norm",
                        "smoothness_gradient_dual_norm",
                        "weighted_smoothness_gradient_dual_norm",
                        "smoothness_to_l2_gradient_ratio",
                    )
                    if key in metrics
                },
            },
        )
        return
    with np.load(folder / "last.npz", allow_pickle=False) as state:
        controls = torch.nn.Parameter(torch.as_tensor(state["q"]))
        axes = torch.as_tensor(state["fixed_axes"])
        seed = state["u"].copy()
    try:
        with study.physics.approximate_solves():
            result = study.evaluate(
                controls,
                mode,
                axes if axes.numel() else None,
                seed,
                normal_weight,
                smooth_weight,
                component_gradients=True,
            )
        write_json(
            folder / "gradient-balance.json",
            {
                "status": "available",
                "checkpoint": receipt(folder / "last.npz"),
                "activation_model": "strain",
                "gradient_metric": "ambient symmetric strain tensor; dual effective-volume norm",
                "l2_component": "same equilibrium as the L2-plus-normal endpoint",
                "solver_valid": result["solver_valid"],
                **(
                    {"l2_adjoint": result["l2_adjoint"]}
                    if "l2_adjoint" in result
                    else {}
                ),
                **{
                    key: result[key]
                    for key in (
                        "l2_gradient_dual_norm",
                        "smoothness_gradient_dual_norm",
                        "weighted_smoothness_gradient_dual_norm",
                        "smoothness_to_l2_gradient_ratio",
                        "forward",
                        "adjoint",
                    )
                },
            },
        )
    except NUMERICAL_FAILURES as error:
        write_json(
            folder / "gradient-balance.json",
            {
                "status": "unavailable",
                "checkpoint": receipt(folder / "last.npz"),
                "activation_model": "strain",
                "failure": {"type": type(error).__name__, "message": str(error)},
            },
        )


def main(cfg: Config) -> None:  # noqa: PLR0915
    assert cfg.steps > 0
    assert cfg.learning_rate > 0
    assert cfg.adam_eps > 0
    assert cfg.smooth_weight >= 0
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    study = StressStudy(activation_model="strain")
    physics = study.physics
    sources = archive(out)
    study.save_geometry(out / "mesh.npz")
    runtime = runtime_receipt(out)
    provenance = (
        None
        if cfg.mechanics_provenance is None
        else receipt(cherries.input(cfg.mechanics_provenance))
    )
    beta = normal_weight(cfg.loss)
    protocol = {
        "schema": "active-strain-chain-v1",
        "activation_model": "strain",
        "activation_units": "dimensionless active strain S with B = I + S",
        "stage_sequence": list(STAGES),
        "loss": {
            "name": cfg.loss,
            "position_weight": 1.0,
            "normal_weight": beta,
            "normal_balance": {
                "position_vector_rms_mm": 2.0,
                "normal_angle_deg": 5.0,
                "formula": "(2^2/(3*L_REF_MM^2))/(2*(1-cos(5 degrees)))",
            },
            "smooth_weight": cfg.smooth_weight,
        },
        "l_ref_mm": L_REF_MM,
        "smooth_length_m": SMOOTH_LENGTH_M,
        "steps_per_stage": cfg.steps,
        "optimizer": {
            "name": "projected Adam with finite approximate gradients",
            "learning_rate": cfg.learning_rate,
            "eps": cfg.adam_eps,
            "betas": [0.9, 0.999],
            "fresh_state_per_stage": True,
            "finite_approximate_policy": "accepted and transferable, including finite inverted states",
        },
        "initialization": "symmetric6 starts S=0, B=I, zero displacement; later stages project the preceding accepted finite endpoint and retain its displacement seed",
        "mechanics_provenance": provenance,
        "mechanics_provenance_scope": (
            "Optional portable provenance receipt only; this remote chain does not "
            "claim an absolute-path source-matched validation gate."
        ),
        "materials": physics.material_spec,
        "forward_tolerance": physics.forward_tolerance,
        "fixture": {
            "volume": receipt(FIXTURE / "volume.vtu"),
            "skin": receipt(FIXTURE / "skin.vtp"),
        },
        "runtime": runtime,
        "sources": portable_source_freeze(out, sources),
    }
    write_json(out / "protocol.json", protocol)
    completed: list[str] = []
    errors: list[dict[str, object]] = []
    summaries: dict[str, dict] = {}
    timestamps: dict[str, dict[str, str]] = {}
    write_chain_status(
        out,
        loss=cfg.loss,
        current=None,
        completed=completed,
        errors=errors,
        summaries=summaries,
        timestamps=timestamps,
        steps_per_stage=cfg.steps,
    )
    Q0 = torch.zeros((len(physics.ids), 3, 3))
    seed = np.zeros_like(physics.points)
    parent: tuple[str, dict] | None = None
    for mode in STAGES:
        current = stage_id(cfg.loss, mode)
        timestamps[current] = {
            "started_at": datetime.datetime.now().astimezone().isoformat()
        }
        write_chain_status(
            out,
            loss=cfg.loss,
            current=current,
            completed=completed,
            errors=errors,
            summaries=summaries,
            timestamps=timestamps,
            steps_per_stage=cfg.steps,
        )
        parent_axes = None
        if parent is not None:
            parent_id, parent_summary = parent
            if (
                parent_summary["status"] != "completed_budget_not_convergence_certified"
                or parent_summary["last_step"] is None
            ):
                error = {
                    "stage": current,
                    "type": "ParentEndpointUnavailable",
                    "message": "Parent did not complete with a usable finite endpoint.",
                }
                errors.append(error)
                summaries[current] = {
                    "status": "blocked_by_parent_failure",
                    "last_step": None,
                    "budget": cfg.steps,
                    "failure": error,
                }
                write_json(out / current / "summary.json", summaries[current])
                write_json(out / "summary.json", summaries)
                write_chain_status(
                    out,
                    loss=cfg.loss,
                    current=None,
                    completed=completed,
                    errors=errors,
                    summaries=summaries,
                    timestamps=timestamps,
                    steps_per_stage=cfg.steps,
                )
                break
            with np.load(out / parent_id / "last.npz", allow_pickle=False) as state:
                assert str(state["activation_model"]) == "strain"
                Q0 = torch.as_tensor(state["S"])
                seed = state["u"].copy()
                parent_solver_valid = bool(state["solver_valid"])
                if mode == "rankone_learned":
                    parent_axes = torch.as_tensor(state["fixed_axes"])
            write_json(
                out / f"{current}-parent.json",
                {
                    "parent": parent_id,
                    "state": receipt(out / parent_id / "last.npz"),
                    "parent_status": parent_summary["status"],
                    "parent_solver_valid": parent_solver_valid,
                    "transfer": "latest accepted finite endpoint, irrespective of solver_valid or inverted-cell diagnostics",
                },
            )
        try:
            summary = run_stage(
                study,
                out / current,
                mode,
                beta,
                cfg.smooth_weight,
                Q0,
                seed,
                steps=cfg.steps,
                learning_rate=cfg.learning_rate,
                adam_eps=cfg.adam_eps,
                stage_id=current,
                parent_axes=parent_axes,
                activation_model="strain",
            )
        except Exception as error:
            errors.append(
                {"stage": current, "type": type(error).__name__, "message": str(error)}
            )
            write_chain_status(
                out,
                loss=cfg.loss,
                current=None,
                completed=completed,
                errors=errors,
                summaries=summaries,
                timestamps=timestamps,
                steps_per_stage=cfg.steps,
            )
            raise
        summaries[current] = summary
        usable_completion = (
            summary["status"] == "completed_budget_not_convergence_certified"
            and summary["last_step"] is not None
        )
        if usable_completion:
            endpoint_diagnostic(
                study, out / current, mode, beta, cfg.smooth_weight, summary
            )
        timestamps[current].update(
            updated_at=datetime.datetime.now().astimezone().isoformat(),
        )
        if not usable_completion:
            error = {
                "stage": current,
                "type": "StageEndpointUnavailable",
                "message": "Stage did not complete with a usable finite endpoint.",
            }
            errors.append(error)
            write_json(out / "summary.json", summaries)
            write_chain_status(
                out,
                loss=cfg.loss,
                current=None,
                completed=completed,
                errors=errors,
                summaries=summaries,
                timestamps=timestamps,
                steps_per_stage=cfg.steps,
            )
            break
        completed.append(current)
        timestamps[current]["completed_at"] = (
            datetime.datetime.now().astimezone().isoformat()
        )
        parent = (current, summary)
        write_json(out / "summary.json", summaries)
        write_chain_status(
            out,
            loss=cfg.loss,
            current=None,
            completed=completed,
            errors=errors,
            summaries=summaries,
            timestamps=timestamps,
            steps_per_stage=cfg.steps,
        )
    cherries.log_metrics(
        {
            "completed_stages": len(completed),
            "chain_errors": len(errors),
            "normal_weight": beta,
            "smooth_weight": cfg.smooth_weight,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
