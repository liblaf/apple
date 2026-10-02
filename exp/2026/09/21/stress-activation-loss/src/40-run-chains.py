"""Run two independent four-stage active-stress chains with frozen regularization."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import torch
from activation_models import STRESS_REF_MPA
from experiment import Profile
from run_support import (
    MODES,
    NUMERICAL_FAILURES,
    archive,
    receipt,
    run_stage,
    update_site,
    verify_sources,
    write_json,
)
from stress_study import L_REF_MM, SMOOTH_LENGTH_M, StressStudy

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("40-runs-inverse-v2")
    validation: Path = Path("12-validation-inverse-v2")
    calibration: Path = Path("20-calibration-inverse-v2/calibration.json")
    steps: int = 200
    site: Path | None = None
    render: bool = False


def main(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915
    group = Path(__file__).parents[1]
    site = cfg.site
    gate = cherries.input(cfg.validation)
    calibration_path = cherries.input(cfg.calibration)
    gate_protocol = json.loads((gate / "protocol.json").read_text())
    assert json.loads((gate / "checks.json").read_text())["passed"]
    verify_sources(gate_protocol["sources"])
    calibration = json.loads(calibration_path.read_text())
    assert calibration["passed"]
    assert calibration["schema"] == "smile-stress-gradient-calibration-v2"
    verify_sources(calibration["sources"])
    assert calibration["validation"] == receipt(gate / "checks.json")
    weight = calibration["selected_weight"]
    assert weight > 0
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    study = StressStudy()
    p = study.physics
    study.save_geometry(out / "mesh.npz")
    protocol = {
        "materials": p.material_spec,
        "activation_reference_MPa": STRESS_REF_MPA,
        "l_ref_mm": L_REF_MM,
        "smooth_length_m": SMOOTH_LENGTH_M,
        "smooth_weight": weight,
        "steps_per_stage": cfg.steps,
        "optimizer": {
            "name": "projected Adam with finite approximate gradients",
            "learning_rate": calibration["learning_rate"],
            "eps": calibration["adam_eps"],
            "betas": [0.9, 0.999],
            "fresh_state_per_stage": True,
            "budget_counts": "attempted updates, including skipped unusable solves",
            "solve_policy": "finite unconverged forward and adjoint results are usable; convergence status recorded",
            "objective_policy": "loss increases allowed; no Armijo test or descent fallback",
            "unusable_policy": "restore controls and moments, halve learning rate, continue next attempt",
            "zero_amplitude_axis_policy": "negative minimum eigenvector of total physical tensor gradient; exact zero tensor retained; affected cell moments reset",
        },
        "losses": {"l2": 0.0, "normal": 1.0},
        "modes": MODES,
        "initialization": "Type1 Q=0 neutral; type2 PSD projection of same-loss type1; type3 strongest nonnegative eigenmode of same-loss type2; type4 exact same-loss type3 stress then release its axis",
        "sources": archive(out),
        "gate": receipt(gate / "checks.json"),
        "calibration": receipt(calibration_path),
        "fixture": gate_protocol["fixture"],
        "forward_tolerance": p.forward_tolerance,
    }
    write_json(out / "protocol.json", protocol)
    if site is not None:
        state = json.loads((site / "status.json").read_text())
        state["status"] = "running"
        state["phase"] = "eight staged fits"
        state["protocol"]["smooth_weight"] = weight
        state["protocol"]["steps_per_stage"] = cfg.steps
        state["protocol"]["calibration"] = calibration
        write_json(site / "status.json", state)
    summaries = {}
    for loss, beta in (("l2", 0.0), ("normal", 1.0)):
        Q0 = torch.zeros((len(p.ids), 3, 3))
        seed = np.zeros_like(p.points)
        parent = None
        for mode in MODES:
            parent_axes = None
            stage_id = f"{loss}-{mode}"
            folder = out / stage_id
            if parent is not None and (
                parent["last_step"] is None
                or parent["attempted_steps"] != cfg.steps
                or parent["status"] != "completed_budget_not_convergence_certified"
            ):
                summary = {
                    "status": "blocked_by_parent_failure",
                    "last_step": None,
                    "attempted_steps": 0,
                    "budget": cfg.steps,
                    "last_metrics": None,
                    "failure": {
                        "message": "Parent has no usable endpoint or did not finish its attempt budget"
                    },
                }
                folder.mkdir()
                write_json(folder / "summary.json", summary)
                summaries[stage_id] = summary
                if site is not None:
                    update_site(site, stage_id, summary, [])
                continue
            if parent is not None:
                with np.load(out / parent["id"] / "last.npz", allow_pickle=False) as z:
                    Q0 = torch.as_tensor(z["Qhat"])
                    seed = z["u"].copy()
                    parent_solver_valid = bool(z["solver_valid"])
                    if mode == "rankone_learned":
                        parent_axes = torch.as_tensor(z["fixed_axes"])
                write_json(
                    out / f"{stage_id}-parent.json",
                    {
                        "parent": parent["id"],
                        "state": receipt(out / parent["id"] / "last.npz"),
                        "parent_status": parent["status"],
                        "parent_solver_valid": parent_solver_valid,
                    },
                )
            summary = run_stage(
                study,
                folder,
                mode,
                beta,
                weight,
                Q0,
                seed,
                steps=cfg.steps,
                learning_rate=calibration["learning_rate"],
                adam_eps=calibration["adam_eps"],
                site=site,
                stage_id=stage_id,
                parent_axes=parent_axes,
            )
            if summary["last_step"] is not None:
                # Common tensor-space endpoint diagnostic, including an L2-only
                # adjoint in the normal column. No weight is changed here.
                try:
                    with np.load(folder / "last.npz", allow_pickle=False) as endpoint:
                        controls = torch.nn.Parameter(torch.as_tensor(endpoint["q"]))
                        fixed = torch.as_tensor(endpoint["fixed_axes"])
                        diagnostic = study.evaluate(
                            controls,
                            mode,
                            fixed if fixed.numel() else None,
                            endpoint["u"].copy(),
                            beta,
                            weight,
                            component_gradients=True,
                        )
                    write_json(
                        folder / "gradient-balance.json",
                        {
                            "status": "available",
                            "checkpoint": receipt(folder / "last.npz"),
                            "gradient_metric": "ambient symmetric tensor; dual effective-volume norm",
                            "convergence_certificate": False,
                            "weight": weight,
                            **{
                                key: diagnostic[key]
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
                            "convergence_certificate": False,
                            "failure": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        },
                    )
            summaries[stage_id] = summary
            parent = {**summary, "id": stage_id}
            write_json(out / "summary.json", summaries)
            if cfg.render and summary["last_step"] is not None:
                subprocess.run(
                    [
                        sys.executable,
                        str(group / "src/50-render-stage.py"),
                        "--source",
                        str(out.resolve()),
                        "--stage",
                        stage_id,
                    ],
                    check=True,
                    cwd=group,
                )
    write_json(out / "summary.json", summaries)
    if cfg.render:
        for entry in ("60-verify-results.py", "70-compare-results.py"):
            subprocess.run(
                [
                    sys.executable,
                    str(group / "src" / entry),
                    "--source",
                    str(out.resolve()),
                ],
                check=True,
                cwd=group,
            )
    if site is not None:
        state = json.loads((site / "status.json").read_text())
        state["updated_at"] = datetime.datetime.now().astimezone().isoformat()
        state["status"] = (
            "completed"
            if all(
                s["status"] == "completed_budget_not_convergence_certified"
                for s in summaries.values()
            )
            else "finished_with_incomplete_stages"
        )
        state["phase"] = (
            "results available; finite budget is not convergence certification"
        )
        write_json(site / "status.json", state)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
