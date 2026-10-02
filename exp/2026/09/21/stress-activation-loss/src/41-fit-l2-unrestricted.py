"""Run the calibrated no-skin L2 fit with unrestricted symmetric active stress."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from activation_models import STRESS_REF_MPA
from experiment import Profile
from run_support import (
    NUMERICAL_FAILURES,
    archive,
    metrics,
    receipt,
    run_stage,
    verify_sources,
    write_json,
)
from stress_study import L_REF_MM, SMOOTH_LENGTH_M, StressStudy

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("41-l2-unrestricted-inverse-v2")
    validation: Path = Path("12-validation-inverse-v2")
    calibration: Path = Path("20-calibration-inverse-v2/calibration.json")
    mechanics_reference: Path | None = None
    pilot_reference: Path | None = None
    resume_checkpoint: Path | None = None
    steps: int = 200
    pilot_steps: int = 8
    learning_rate: float = 0.05
    adam_eps: float = 1e-8
    target_gradient_ratio: float = 0.1
    smooth_weight: float | None = None
    activation_model: Literal["stress", "strain"] = "stress"


def reference_scope(reference: Path) -> dict[str, object]:
    """Check reference receipts without requiring changed live sources to match."""
    checks = json.loads((reference / "checks.json").read_text())
    assert checks["passed"]
    protocol = json.loads((reference / "protocol.json").read_text())
    assert checks["source_protocol"] == receipt(reference / "protocol.json")
    diffs = {}
    for path, record in protocol["sources"].items():
        snapshot = Path(record["snapshot"])
        assert receipt(snapshot)["sha256"] == record["sha256"]
        live = Path(path)
        current = receipt(live)["sha256"] if live.is_file() else None
        diffs[path] = {
            "validated_sha256": record["sha256"],
            "snapshot_sha256": receipt(snapshot)["sha256"],
            "current_sha256": current,
            "matches_validated": current == record["sha256"],
        }
    return {
        "reference_checks": receipt(reference / "checks.json"),
        "reference_protocol": receipt(reference / "protocol.json"),
        "source_snapshot_integrity": "passed",
        "live_source_diffs": diffs,
        "scope": "Converged mechanics derivatives before the user-requested Newton initial-shift change; fitting allows approximate solves.",
        "fixture": protocol["fixture"],
    }


def source_drift(records: dict) -> dict[str, object]:
    """Record current-versus-parent numerical source hashes without exemptions."""
    drift = {}
    for path, record in records.items():
        assert receipt(record["snapshot"])["sha256"] == record["sha256"]
        live = Path(path)
        current = receipt(live)["sha256"] if live.is_file() else None
        drift[path] = {
            "parent_sha256": record["sha256"],
            "current_sha256": current,
            "changed": current != record["sha256"],
        }
    return drift


def verify_pilot_reference(pilot_dir: Path) -> None:
    """Accept only a completed unregularized symmetric6 pilot from this physics."""
    parent_sources = json.loads((pilot_dir.parent / "sources.json").read_text())
    recipe = Path(__file__).resolve()
    for path, record in parent_sources.items():
        assert receipt(record["snapshot"])["sha256"] == record["sha256"]
        live = Path(path)
        if live.resolve() != recipe:
            assert receipt(live)["sha256"] == record["sha256"], path
    summary = json.loads((pilot_dir / "summary.json").read_text())
    initial = json.loads((pilot_dir / "initialization.json").read_text())
    assert summary["status"] == "completed_budget_not_convergence_certified"
    assert summary["last_step"] is not None
    assert initial["mode"] == "symmetric6"
    assert initial["normal_weight"] == 0.0
    assert initial["smooth_weight"] == 0.0


def calibrate_current_runtime(study: StressStudy, out: Path, cfg: Config) -> dict:
    """Freeze eta from recorded pilot-state gradient norms without another solve."""
    physics = study.physics
    if cfg.pilot_reference is None:
        pilot_dir = out / "pilot-l2"
        pilot = run_stage(
            study,
            pilot_dir,
            "symmetric6",
            0.0,
            0.0,
            torch.zeros((len(physics.ids), 3, 3)),
            np.zeros_like(physics.points),
            steps=cfg.pilot_steps,
            learning_rate=cfg.learning_rate,
            adam_eps=cfg.adam_eps,
            stage_id="pilot-l2",
        )
    else:
        pilot_dir = cherries.input(cfg.pilot_reference)
        verify_pilot_reference(pilot_dir)
        pilot = json.loads((pilot_dir / "summary.json").read_text())
    assert pilot["last_step"] is not None, pilot
    metrics_at_pilot = pilot["last_metrics"]
    assert metrics_at_pilot is not None
    l2_norm = float(metrics_at_pilot["l2_gradient_dual_norm"])
    smooth_norm = float(metrics_at_pilot["smoothness_gradient_dual_norm"])
    assert np.isfinite(l2_norm)
    assert l2_norm > 0
    assert np.isfinite(smooth_norm)
    assert smooth_norm > 0
    weight = cfg.target_gradient_ratio * l2_norm / smooth_norm
    assert np.isfinite(weight)
    assert weight > 0
    receipts = [
        json.loads(line)
        for line in (pilot_dir / "solver-receipts.jsonl").read_text().splitlines()
    ]
    receipt_at_pilot = next(
        row for row in receipts if row["step"] == pilot["last_step"]
    )
    with np.load(pilot_dir / "last.npz", allow_pickle=False) as last:
        solver_valid = bool(last["solver_valid"])
    calibration = {
        "schema": "smile-stress-gradient-calibration-v2",
        "passed": True,
        "selected_weight": weight,
        "target_gradient_ratio": cfg.target_gradient_ratio,
        "calibration_balance": {
            "l2_gradient_dual_norm": l2_norm,
            "smoothness_gradient_dual_norm": smooth_norm,
            "weighted_smoothness_gradient_dual_norm": weight * smooth_norm,
            "smoothness_to_l2_gradient_ratio": weight * smooth_norm / l2_norm,
        },
        "gradient_metric": "ambient symmetric tensor; dual normalized effective-volume norm",
        "gradient_numerator": "L2 only; normal loss excluded",
        "calibration_state": receipt(pilot_dir / "last.npz"),
        "solver_valid": solver_valid,
        "forward": receipt_at_pilot["forward"],
        "adjoint": receipt_at_pilot["adjoint"],
        "pilot": pilot,
        "pilot_reference": None
        if cfg.pilot_reference is None
        else receipt(pilot_dir / "summary.json"),
        "pilot_steps": cfg.pilot_steps,
        "learning_rate": cfg.learning_rate,
        "adam_eps": cfg.adam_eps,
        "current_source_snapshot": receipt(out / "sources.json"),
    }
    write_json(out / "calibration.json", calibration)
    return calibration


def explicit_weight_selection(out: Path, cfg: Config) -> dict:
    """Record a user-selected objective weight without calling it calibrated."""
    assert cfg.smooth_weight is not None
    selection = {
        "schema": "smile-stress-objective-weight-v1",
        "passed": True,
        "selection": "explicit_config_override",
        "selected_weight": cfg.smooth_weight,
        "calibrated_weight": None,
        "calibration_status": "not_run",
        "reason": "User selected the smoothness weight explicitly.",
        "learning_rate": cfg.learning_rate,
        "adam_eps": cfg.adam_eps,
        "current_source_snapshot": receipt(out / "sources.json"),
    }
    write_json(out / "calibration.json", selection)
    return selection


def save_activation_mesh(
    physics: Any, path: Path, displacement: np.ndarray, matrix: np.ndarray, model: str
) -> None:
    """Export the physical activation field without assigning strain stress units."""
    if model == "stress":
        physics.save_mesh(path, displacement, active_stress=matrix * STRESS_REF_MPA)
        return
    assert model == "strain"
    physics.save_mesh(path, displacement)
    mesh = pv.read(path)
    strain = np.zeros((len(physics.tets), 3, 3))
    strain[physics.ids] = matrix
    deformation = np.broadcast_to(np.eye(3), strain.shape).copy()
    deformation[physics.ids] += matrix
    mesh.cell_data["ActiveStrainMatrix"] = strain.reshape(-1, 9)
    mesh.cell_data["ActiveStrainTrace"] = np.trace(strain, axis1=1, axis2=2)
    mesh.cell_data["ActivationDeformationB"] = deformation.reshape(-1, 9)
    mesh.save(path)


def main(cfg: Config) -> None:  # noqa: PLR0915
    assert cfg.steps >= 0
    assert cfg.pilot_steps > 0
    assert 0 < cfg.target_gradient_ratio < 1
    assert cfg.smooth_weight is None or (
        np.isfinite(cfg.smooth_weight) and cfg.smooth_weight >= 0
    )
    assert cfg.pilot_reference is None or cfg.mechanics_reference is not None
    assert cfg.resume_checkpoint is None or cfg.mechanics_reference is not None
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    study = (
        StressStudy()
        if cfg.activation_model == "stress"
        else StressStudy(activation_model=cfg.activation_model)
    )
    physics = study.physics
    source_records = archive(out)
    study.save_geometry(out / "mesh.npz")
    reference = None
    calibration_path = None
    calibration_receipt = None
    resume_checkpoint = (
        None if cfg.resume_checkpoint is None else cherries.input(cfg.resume_checkpoint)
    )
    if cfg.smooth_weight is not None:
        if cfg.mechanics_reference is None:
            gate = cherries.input(cfg.validation)
            gate_protocol = json.loads((gate / "protocol.json").read_text())
            assert json.loads((gate / "checks.json").read_text())["passed"]
            verify_sources(gate_protocol["sources"])
            reference = {
                "gate": receipt(gate / "checks.json"),
                "fixture": gate_protocol["fixture"],
            }
        else:
            reference = reference_scope(cherries.input(cfg.mechanics_reference))
        calibration = explicit_weight_selection(out, cfg)
        calibration_receipt = receipt(out / "calibration.json")
    elif cfg.mechanics_reference is None:
        gate = cherries.input(cfg.validation)
        calibration_path = cherries.input(cfg.calibration)
        assert json.loads((gate / "checks.json").read_text())["passed"]
        gate_protocol = json.loads((gate / "protocol.json").read_text())
        verify_sources(gate_protocol["sources"])
        calibration = json.loads(calibration_path.read_text())
        calibration_receipt = receipt(calibration_path)
        assert calibration["passed"]
        assert calibration["schema"] == "smile-stress-gradient-calibration-v2"
        assert calibration["validation"] == receipt(gate / "checks.json")
        verify_sources(calibration["sources"])
        reference = {
            "gate": receipt(gate / "checks.json"),
            "fixture": gate_protocol["fixture"],
        }
    else:
        reference = reference_scope(cherries.input(cfg.mechanics_reference))
        if cfg.resume_checkpoint is None:
            calibration = calibrate_current_runtime(study, out, cfg)
            calibration_receipt = receipt(out / "calibration.json")
        else:
            parent = resume_checkpoint.parent.parent
            calibration = json.loads((parent / "calibration.json").read_text())
            assert calibration["passed"]
            assert calibration["schema"] in {
                "smile-stress-gradient-calibration-v2",
                "smile-stress-objective-weight-v1",
            }
            calibration = {
                **calibration,
                "inherited_from": receipt(parent / "calibration.json"),
                "resume_checkpoint": receipt(resume_checkpoint),
            }
            write_json(out / "calibration.json", calibration)
            calibration_receipt = receipt(out / "calibration.json")
            reference["resume_lineage"] = {
                "checkpoint": receipt(resume_checkpoint),
                "parent_last_state": receipt(resume_checkpoint.parent / "last.npz"),
                "parent_calibration": receipt(parent / "calibration.json"),
                "parent_sources": receipt(parent / "sources.json"),
                "source_drift": source_drift(
                    json.loads((parent / "sources.json").read_text())
                ),
                "accepted_checkpoint_continuation": "Continues from the accepted checkpoint; prior attempts remain in the parent lineage and are not replayed.",
            }
    if cfg.activation_model == "strain":
        reference["activation_model_scope"] = (
            "The mechanics reference validates the archived baseline derivatives; "
            "it does not validate the active-strain constitutive wiring."
        )
    weight = float(calibration["selected_weight"])
    assert weight >= 0
    assert calibration_receipt is not None
    fit_learning_rate = (
        cfg.learning_rate
        if cfg.resume_checkpoint is not None or cfg.smooth_weight is not None
        else calibration["learning_rate"]
    )
    protocol = {
        "stage": "l2-symmetric6",
        "materials": physics.material_spec,
        "activation_model": cfg.activation_model,
        "activation_units": (
            "normalized active stress times reference MPa"
            if cfg.activation_model == "stress"
            else "dimensionless active strain S with B = I + S"
        ),
        "l_ref_mm": L_REF_MM,
        "smooth_length_m": SMOOTH_LENGTH_M,
        "objective": {
            "position_weight": 1.0,
            "normal_weight": 0.0,
            "smooth_weight": weight,
        },
        "smooth_weight": weight,
        "weight_selection": {
            "effective_smooth_weight": weight,
            "config_override": cfg.smooth_weight,
            "calibration_status": calibration.get("calibration_status", "calibrated"),
        },
        "steps": cfg.steps,
        "mode": "symmetric6",
        "initialization": (
            "zero active stress and zero displacement seed"
            if cfg.activation_model == "stress"
            else "zero active strain (B = I) and zero displacement seed"
        )
        if cfg.resume_checkpoint is None
        else "restored controls, displacement seed, and Adam moments from resume checkpoint",
        "optimizer": {
            "name": "projected Adam with finite approximate gradients",
            "learning_rate": fit_learning_rate,
            "eps": calibration["adam_eps"],
            "betas": [0.9, 0.999],
            "fresh_state": cfg.resume_checkpoint is None,
            "budget_counts": "attempts, including skipped unusable solves",
            "solve_policy": "finite unconverged forward and adjoint results are usable and recorded",
            "objective_policy": "loss increases allowed; no Armijo or descent fallback",
            "unusable_policy": "restore controls and moments, halve learning rate, continue next attempt",
        },
        "mechanics_reference": reference,
        "calibration": calibration_receipt,
        "fixture": reference["fixture"],
        "forward_tolerance": physics.forward_tolerance,
        "sources": source_records,
    }
    if cfg.activation_model == "stress":
        protocol["activation_reference_MPa"] = STRESS_REF_MPA
    write_json(out / "protocol.json", protocol)
    summary = run_stage(
        study,
        out / "l2-symmetric6",
        "symmetric6",
        0.0,
        weight,
        torch.zeros((len(physics.ids), 3, 3)),
        np.zeros_like(physics.points),
        steps=cfg.steps,
        learning_rate=fit_learning_rate,
        adam_eps=calibration["adam_eps"],
        stage_id="l2-symmetric6",
        resume_checkpoint=None if cfg.resume_checkpoint is None else resume_checkpoint,
        activation_model=cfg.activation_model,
    )
    stage = out / "l2-symmetric6"
    diagnostic: dict[str, object] = {"status": "unavailable"}
    if summary["last_step"] is not None:
        with np.load(stage / "last.npz", allow_pickle=False) as last:
            controls, matrix, seed, u = (
                last["q"],
                last["Qhat"] if cfg.activation_model == "stress" else last["S"],
                last["u"].copy(),
                last["u"],
            )
        save_activation_mesh(
            physics,
            out
            / (
                "last-displacement-active-stress.vtu"
                if cfg.activation_model == "stress"
                else "last-displacement-active-strain.vtu"
            ),
            u,
            matrix,
            cfg.activation_model,
        )
        try:
            q = torch.nn.Parameter(torch.as_tensor(controls))
            result = study.evaluate(
                q, "symmetric6", None, seed, 0.0, weight, component_gradients=True
            )
            diagnostic = {
                "status": "available",
                "checkpoint": receipt(stage / "last.npz"),
                "gradient_metric": "ambient symmetric tensor; dual effective-volume norm",
                "convergence_certificate": False,
                "weight": weight,
                "reequilibrated_metrics": metrics(result),
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
            }
        except NUMERICAL_FAILURES as error:
            diagnostic = {
                "status": "unavailable",
                "checkpoint": receipt(stage / "last.npz"),
                "convergence_certificate": False,
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
    write_json(stage / "gradient-balance.json", diagnostic)
    write_json(out / "summary.json", {"l2-symmetric6": summary})
    cherries.log_metrics(
        {
            key: value
            for key, value in summary.items()
            if isinstance(value, (bool, int, float, str))
        }
    )
    if summary["last_metrics"] is not None:
        cherries.log_metrics(summary["last_metrics"])


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
