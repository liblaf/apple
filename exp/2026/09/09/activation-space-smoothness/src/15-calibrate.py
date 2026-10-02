"""Measure initial physical-step and penalty scales for subsequent tuning pilots."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import torch
from activation_controls import baseline_z, project_psd_, tensile_z
from experiment_profile import ProfileCometNoCommit
from study_physics import FacePhysics, configure
from study_runner import (
    FIXTURE,
    GROUP,
    LENGTH_M,
    MU,
    Objective,
    archive_runtime,
    digest,
    write_json,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)
DONE = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = FIXTURE
    output_dir: Path = GROUP / "data/15-initial-probe"
    pilot_steps: int = 8
    baseline_learning_rate: float = 0.3
    adam_eps: float = 0.01
    smooth_gradient_ratio: float = 0.25


def main(cfg: Config) -> None:
    global DONE
    assert cfg.pilot_steps == 8 and cfg.smooth_gradient_ratio == 0.25
    validation_path = GROUP / "data/10-controls-validation/summary.json"
    assert json.loads(validation_path.read_text())["status"] == "passed"
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Refuse to overwrite calibration: {out}"
    configure()
    start = time.perf_counter()
    baseline = FacePhysics(cfg.fixture, activation_model="raw6", skin_factor=0.0)
    base_objective = Objective(baseline)
    qb = torch.nn.Parameter(torch.zeros((len(baseline.ids), 6)))
    zero_u = np.zeros_like(baseline.points)
    initial_base = base_objective(qb, zero_u, component_gradients=True)
    base_gradient = qb.grad.detach().clone()
    base_optimizer = torch.optim.Adam(
        [qb], lr=cfg.baseline_learning_rate, eps=cfg.adam_eps
    )
    base_optimizer.step()
    delta_b = baseline_z(qb).detach()
    mass = torch.as_tensor(baseline.volumes / baseline.volumes.sum())

    def physical_rms(z):
        return float((mass * z.square().sum((-1, -2))).sum().sqrt())

    target_step = physical_rms(delta_b)
    assert target_step > 0
    tensor = FacePhysics(cfg.fixture, activation_model="tensor", skin_factor=0.0)
    assert np.array_equal(tensor.ids, baseline.ids)
    assert np.array_equal(tensor.target, baseline.target)
    for lhs, rhs in zip(baseline.graph, tensor.graph, strict=True):
        assert np.array_equal(lhs, rhs)
    tensor_objective = Objective(tensor)
    qt = torch.nn.Parameter(torch.zeros_like(qb))
    initial_tensor = tensor_objective(qt, zero_u, component_gradients=True)
    tensor_gradient = qt.grad.detach().clone()
    # At inactivity dZ=2*dB. Raw off-diagonals are not orthonormal coordinates.
    factors = qb.new_tensor([2, 2, 2, 2 * 2**0.5, 2 * 2**0.5, 2 * 2**0.5])
    predicted_base_gradient = tensor_gradient * factors
    gradient_relative_error = float(
        (base_gradient - predicted_base_gradient).norm() / base_gradient.norm()
    )
    initial_geometry_error_mm = float(
        1000
        * np.linalg.norm(initial_base["u"] - initial_tensor["u"])
        / np.sqrt(len(zero_u))
    )
    assert gradient_relative_error < 0.005, gradient_relative_error
    assert initial_geometry_error_mm < 0.001, initial_geometry_error_mm
    unit_step = -tensor_gradient / (tensor_gradient.abs() + cfg.adam_eps)
    project_psd_(unit_step)
    # PSD projection and direct Z are positively homogeneous at the zero start.
    tensor_lr = target_step / physical_rms(tensile_z(unit_step))
    assert np.isfinite(tensor_lr) and tensor_lr > 0
    optimizer = torch.optim.Adam([qt], lr=tensor_lr, eps=cfg.adam_eps)
    rows = []
    result = initial_tensor
    seed = zero_u
    for step in range(cfg.pilot_steps + 1):
        if step:
            result = tensor_objective(
                qt, seed, component_gradients=step == cfg.pilot_steps
            )
        fit_rms = float(np.sqrt(3 * result["objective_mm2"]))
        row = {
            "step": step,
            "fit_rms_mm": fit_rms,
            "smoothness": result["smoothness"],
            "objective_s": result["objective_s"],
            "forward_steps": result["forward"]["steps"],
            "forward_success": result["forward"]["success"],
            "adjoint_success": result["adjoint"]["success"],
        }
        rows.append(row)
        cherries.log_metrics(row, step=step)
        LOG.info(
            "Discarded tensile pilot %d/%d: fit %.4f mm, %.1f s",
            step,
            cfg.pilot_steps,
            fit_rms,
            result["objective_s"],
        )
        if step == cfg.pilot_steps:
            break
        seed = result["u"]
        optimizer.step()
        projection = project_psd_(qt)
        if step == 0:
            actual_first_step = physical_rms(tensile_z(qt))
            first_step_error = abs(actual_first_step / target_step - 1)
            assert first_step_error < 1e-10, first_step_error
            first_step_receipt = {
                "baseline_volume_weighted_delta_Z_frobenius_rms": target_step,
                "tensor_volume_weighted_delta_Z_frobenius_rms": actual_first_step,
                "relative_matching_error": first_step_error,
                "baseline_max_delta_Z_frobenius": float(
                    delta_b.square().sum((-1, -2)).sqrt().max()
                ),
                "tensor_max_delta_Z_frobenius": float(
                    tensile_z(qt).square().sum((-1, -2)).sqrt().max()
                ),
                "baseline_delta_Q_rms_mpa": MU * target_step,
                "tensor_delta_Q_rms_mpa": MU * actual_first_step,
                "tensor_projection": projection,
            }
    fit_norm = float(result["fit_gradient"].norm())
    smooth_norm = float(result["smooth_gradient"].norm())
    assert fit_norm > 0 and smooth_norm > 0
    smooth_weight = cfg.smooth_gradient_ratio * fit_norm / smooth_norm
    # The pilot uses orthonormal Z coordinates; these are physical-field norms.
    np.savez_compressed(
        out / "pilot-final.npz",
        q=qt.detach().cpu().numpy(),
        Z=result["Z"],
        u=result["u"],
        fit_gradient=result["fit_gradient"].cpu().numpy(),
        smooth_gradient=result["smooth_gradient"].cpu().numpy(),
    )
    # Verify the actual two face pipelines at the same nonzero physical field.
    values, vectors = np.linalg.eigh(np.eye(3) + result["Z"])
    assert values.min() > 0
    b = (vectors * np.sqrt(values)[:, None, :]) @ np.swapaxes(vectors, -1, -2)
    mapped_q = np.column_stack(
        (
            b[:, 0, 0] - 1,
            b[:, 1, 1] - 1,
            b[:, 2, 2] - 1,
            b[:, 0, 1],
            b[:, 1, 2],
            b[:, 0, 2],
        )
    )
    qb_mapped = torch.nn.Parameter(torch.as_tensor(mapped_q))
    mapped = base_objective(qb_mapped, result["u"])
    mapped_geometry_error_mm = float(
        1000 * np.linalg.norm(mapped["u"] - result["u"]) / np.sqrt(len(zero_u))
    )
    mapped_fit_difference_mm = float(
        abs(np.sqrt(3 * mapped["objective_mm2"]) - np.sqrt(3 * result["objective_mm2"]))
    )
    mapped_z_error = float(np.max(np.abs(mapped["Z"] - result["Z"])))
    assert mapped_z_error < 1e-12, mapped_z_error
    assert mapped_geometry_error_mm < 0.01, mapped_geometry_error_mm
    assert mapped_fit_difference_mm < 0.01, mapped_fit_difference_mm
    provenance = archive_runtime(out, cfg.fixture)
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    write_json(out / "provenance.json", provenance)
    summary = {
        "status": "completed_initial_scale_probe",
        "inputs": provenance["inputs"],
        "learning_rates": {"raw6": cfg.baseline_learning_rate, "tensor": tensor_lr},
        "adam_eps": cfg.adam_eps,
        "betas": [0.9, 0.999],
        "smoothness_weight": smooth_weight,
        "smooth_length_m": LENGTH_M,
        "magnitude_weight": 0.0,
        "rank_weight": 0.0,
        "skin_enabled": False,
        "upper_stress_cap": None,
        "initialization": "Every primary run starts from inactive controls, rest displacement and fresh Adam moments; pilot discarded",
        "step_selection": "Keep historical raw6 Adam lr/eps; match first post-PSD-projection volume-weighted physical Delta Z Frobenius RMS for tensile controls",
        "weight_selection": "lambda = 0.25 * norm(physical-Z data gradient) / norm(physical-Z smoothness gradient) at discarded tensile pilot step 8",
        "secondary_weight_multiplier": 4.0,
        "first_step": first_step_receipt,
        "physical_gradient_norms_at_pilot": {
            "fit": fit_norm,
            "smoothness": smooth_norm,
        },
        "initial_pipeline_check": {
            "gradient_relative_error": gradient_relative_error,
            "geometry_rms_difference_mm": initial_geometry_error_mm,
        },
        "nonzero_pipeline_check": {
            "mapped_Z_max_abs_error": mapped_z_error,
            "geometry_rms_difference_mm": mapped_geometry_error_mm,
            "fit_rms_difference_mm": mapped_fit_difference_mm,
        },
        "pilot": rows,
        "pilot_steps": cfg.pilot_steps,
        "active_cells": len(tensor.ids),
        "same_muscle_edges": len(tensor.graph[0]),
        "total_active_volume_m3": float(tensor.volumes.sum()),
        "elapsed_s": time.perf_counter() - start,
        "controls_validation": {
            "path": str(validation_path),
            "sha256": digest(validation_path),
        },
        "limits": "Initial physical RMS matching does not equalize subsequent Adam geometry or directions; smoothness permits constants on each connected muscle component. No upper activation bound or magnitude penalty is imposed.",
    }
    write_json(out / "summary.json", summary)
    for name in ("config.json", "provenance.json", "summary.json"):
        cherries.log_output(out / name)
    LOG.info(
        "Provisional rate scales: %s; penalty scale %.8g; further tuning required",
        summary["learning_rates"],
        smooth_weight,
    )
    DONE = True


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
    if not DONE:
        raise SystemExit(1)
