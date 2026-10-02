"""Full-mesh neutral prestress continuation and equilibrium pilot."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import configure_cuda
from joint_fields import (
    BULK_TISSUES,
    SharedFieldParameters,
    research_informed_material_config,
)
from joint_physics import JointPhysics

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    initial_checkpoint: Path | None = None
    output_dir: Path = cherries.output("neutral-pilot", mkdir=True)
    updates: int = 8
    learning_rate: float = 0.001
    skin_prestress_fraction: float = 0.01
    forward_rtol: float = 5e-4
    forward_atol: float = 1e-10
    adjoint_rtol: float = 5e-4
    max_forward_steps: int = 5000


def make_physics(cfg: Config, prepared: PreparedInputs, spec: dict):
    materials = spec["materials"]
    skin = materials["skin"]
    return JointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={name: materials[name]["young_mpa"] for name in BULK_TISSUES},
        bulk_nu={name: materials[name]["poisson"] for name in BULK_TISSUES},
        skin_young_mpa=skin["reference_map"]["young_mpa"],
        skin_nu=skin["poisson"],
        thickness_m=skin["thickness_m"],
        rtol=cfg.forward_rtol,
        atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        max_steps=cfg.max_forward_steps,
    )


def main(cfg: Config):  # noqa: PLR0915
    global COMPLETED  # noqa: PLW0603
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    archive_sources(output)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz",
        cfg.prepared_dir / "manifest.json",
        verify_sources=True,
    )
    configure_cuda()
    spec = research_informed_material_config()
    write_json(output / "material-config.json", spec)
    physics = make_physics(cfg, prepared, spec)
    shared = SharedFieldParameters(spec)
    initial = None
    if cfg.initial_checkpoint is not None:
        initial = torch.load(
            cfg.initial_checkpoint, map_location="cpu", weights_only=False
        )
        assert initial["materials"] == spec
        with torch.no_grad():
            shared.coefficients.copy_(initial["shared_coefficients"])
    optimizer = torch.optim.Adam([shared.coefficients], lr=cfg.learning_rate, eps=1e-8)
    # A declared 1%-of-proxy continuation probe is not a full prestress fit.
    target_resultant = cfg.skin_prestress_fraction * (89.4 + 71.8) / 2
    with torch.no_grad():
        shared.coefficients[18] = target_resultant / shared.skin_resultant_scale_n_per_m
    pose = torch.zeros(6)
    seed = torch.zeros_like(physics.points_t)
    if initial is not None:
        seed = initial["primal"]["neutral"].to(device="cuda")
    trace = []
    started = time.perf_counter()
    last_valid = None
    protocol = {
        "skin_prestress_fraction": cfg.skin_prestress_fraction,
        "skin_target_n_per_m": target_resultant,
        "status": "neutral computational continuation probe; not full literature prestress recovery",
        "neutral_surface_budget_mm": 0.25,
        "neutral_muscle_centroid_budget_mm": 0.5,
        "detF_lower": 0.25,
        "detF_upper": 2.0,
        "skin_area_ratio_lower": 0.25,
        "input_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "input_arrays_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "basis": "18 constant bulk tensor coefficients + isotropic skin + global skin log multiplier",
    }
    write_json(output / "protocol.json", protocol)
    try:
        for update in range(cfg.updates + 1):
            optimizer.zero_grad(set_to_none=True)
            u = physics.solve(
                shared.bulk_stresses_mpa(),
                shared.skin_resultant_n_per_m(),
                shared.skin_stiffness_multiplier(),
                None,
                pose,
                seed,
                key="neutral",
            )
            metrics = physics.metrics(u)
            assert metrics["inverted_tetrahedra"] == 0, metrics
            assert metrics["detF_min"] >= 0.25, metrics
            assert metrics["detF_max"] <= 2, metrics
            assert metrics["skin_area_ratio_min"] >= 0.25, metrics
            objective = (
                physics.neutral_loss(u) + 0.001 * shared.regularizers()["prior_total"]
            )
            objective.backward()
            gradient = shared.coefficients.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            row = {
                "update": update,
                "elapsed_seconds": time.perf_counter() - started,
                "objective": float(objective.detach()),
                "gradient_norm": float(torch.linalg.vector_norm(gradient)),
                "skin_resultant_n_per_m": float(
                    shared.skin_resultant_n_per_m()[0, 0].detach()
                ),
                "skin_multiplier": float(shared.skin_stiffness_multiplier().detach()),
                "shared": shared.coefficients.detach().cpu().tolist(),
                "metrics": metrics,
                "forward": physics.runtime.last_forward,
                "adjoint": physics.runtime.last_adjoint,
            }
            trace.append(row)
            write_json(output / "trace.json", trace)
            cherries.set_step(update)
            cherries.log_metrics({"neutral": metrics, "objective": row["objective"]})
            LOG.info(
                "Neutral update %d: surface %.6f mm, muscle %.6f mm",
                update,
                metrics["surface_motion_rms_mm"],
                metrics["muscle_centroid_motion_rms_mm"],
            )
            seed = u.detach().clone()
            valid = (
                metrics["surface_motion_rms_mm"] <= 0.25
                and metrics["muscle_centroid_motion_rms_mm"] <= 0.5
            )
            checkpoint = {
                "schema": "joint-inverse-checkpoint-v1",
                "shared_coefficients": shared.coefficients.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "next_update": update,
                "primal": {"neutral": seed.cpu()},
                "adjoint": {
                    key: value.cpu()
                    for key, value in physics.runtime.warm_adjoints.items()
                },
                "protocol": protocol,
                "materials": spec,
                "metrics": row,
                "neutral_budget_met": valid,
                "stage": "neutral",
                "inverse_converged": False,
            }
            torch.save(checkpoint, output / "terminal.pt")
            if valid and (
                last_valid is None or row["objective"] < last_valid["objective"]
            ):
                torch.save(checkpoint, output / "best-admissible.pt")
                last_valid = row
            if update == cfg.updates:
                break
            # Keep the selected continuation stress fixed during neutral balance.
            gradient[18] = 0
            optimizer.step()
            shared.project_()
        summary = {
            "success": last_valid is not None,
            "scope": protocol["status"],
            "accepted_evaluations": len(trace),
            "stop_reason": "neutral pilot update budget",
            "first": trace[0],
            "terminal": trace[-1],
            "best_admissible": last_valid,
            "full_joint_run_ready": False,
            "remaining_gates": [
                "oral/jaw envelope",
                "all-field face derivatives",
                "strong smoothness calibration",
                "deformed timing",
            ],
        }
        write_json(output / "summary.json", summary)
        assert last_valid is not None, (
            "Neutral surface/muscle budgets not met; inspect the saved trace"
        )
        COMPLETED = True
        cherries.log_output(output)
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_evaluations": len(trace),
                "forward": physics.runtime.last_forward,
                "adjoint": physics.runtime.last_adjoint,
            },
        )
        raise


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
