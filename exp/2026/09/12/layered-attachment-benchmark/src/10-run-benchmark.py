"""Compare localized attachment mechanics against calibrated uniform stiffness."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp
from liblaf.cherries import core, plugins, profiles
from patch_model import MOBILE_E, Case, Load, configure, solve
from scipy.optimize import brentq

from liblaf import cherries

mpl.use("Agg")

logger = logging.getLogger(__name__)


class ProfileRecordWithoutCommit(profiles.Profile):
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("10-benchmark")
    smoke: bool = False
    maximum_refinement: int = 3


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915
    configure()
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, snapshots, solve_records = [], {}, []
    manifest = {
        "question": "Can uniform stiffness matching one known-load compliance reproduce a localized attachment model under another load?",
        "evidence_class": "synthetic controlled mechanics, not human calibration",
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "warp": wp.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(),
        },
        "units": {
            "length": "mm",
            "force": "N",
            "young_modulus": "N/mm^2 = MPa",
            "energy": "N mm",
            "link_stiffness": "N/mm",
        },
        "anatomical_claim": "None: regular patch, layer boundaries, fiber topology and coefficients are illustrative hypotheses.",
        "attachments": "Opposed oblique tension-only axial families across 0.5 mm continuum layer; Gaussian band with constant total link stiffness; distinct endpoint DOFs.",
        "boundary": "All displacement components fixed at bottom z=0; other surfaces free except Gaussian distributed tangential traction on the top.",
        "fixed_controls": "No activation or fitting of a target expression; passive known-load equilibrium.",
        "reference_state": "Initially stress free; no gravity, rest prestress, contact, rate dependence or bending.",
        "library_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[6])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in [
                Path(__file__).resolve().parents[6]
                / "src/liblaf/apple/warp/fem/_koiter.py",
                Path(__file__).resolve().parents[6]
                / "src/liblaf/apple/warp/potential/_fiber_spring.py",
            ]
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    write_json(output / "protocol.json", manifest)
    source_snapshot = output / "source"
    source_snapshot.mkdir(exist_ok=True)
    for relative in manifest["library_sha256"]:
        original = Path(__file__).resolve().parents[6] / relative
        shutil.copy2(original, source_snapshot / original.name)
    for original in Path(__file__).parent.glob("*.py"):
        shutil.copy2(original, source_snapshot / original.name)
    calibration = Load("calibration_x_right", 0, 15.0)

    def run(case: Case, load: Load, refinement: int = 1, *, save: bool = True):
        metrics, mesh, skin, info, _ = solve(case, load, refinement)
        record = {
            "case_parameters": asdict(case),
            "load_parameters": asdict(load),
            **metrics,
        }
        solve_records.append(record)
        write_json(output / "all-solves.json", solve_records)
        cherries.log_metrics(
            {
                "solve/displacement_mm": metrics["load_displacement_mm"],
                "solve/residual_n": metrics["residual_force_n"],
            }
        )
        cherries.set_step(len(solve_records))
        if save:
            name = f"{case.name}--{load.name}--r{refinement}"
            mesh.save(output / f"{name}.vtu")
            skin.save(output / f"{name}.vtp")
            if len(info["pairs"]):
                lines = pv.PolyData(
                    mesh.points,
                    lines=np.column_stack(
                        [np.full(len(info["pairs"]), 2), info["pairs"]]
                    ).ravel(),
                )
                lines.cell_data["StiffnessNPerMm"] = info["link_stiffness"]
                lines.save(output / f"{name}--links.vtp")
            rows.append(record)
            snapshots[(case.name, load.name, refinement)] = (skin, info)
        return metrics

    cases = [
        Case("homogeneous", layer_young=0.003),
        Case("mobile_layer"),
        Case("localized_attachment", attachment_center=10.0),
        Case("shifted_attachment", attachment_center=5.0),
    ]
    for case in cases[:1] if cfg.smoke else cases:
        run(case, calibration)
    if cfg.smoke:
        write_json(output / "summary.json", {"smoke": True, "runs": rows})
        return
    target = next(
        r["load_displacement_mm"] for r in rows if r["case"] == "localized_attachment"
    )
    calibration_trials = []

    def mismatch(log_scale: float) -> float:
        young = MOBILE_E * np.exp(log_scale)
        result = run(
            Case("uniform_calibration_trial", layer_young=young),
            calibration,
            save=False,
        )
        error = result["load_displacement_mm"] - target
        calibration_trials.append(
            {
                "young_mpa": young,
                "displacement_mm": result["load_displacement_mm"],
                "error_mm": error,
            }
        )
        logger.info(
            "Uniform material calibration: E=%.7g MPa, compliance mismatch %.6g mm",
            young,
            error,
        )
        return error

    matched_log_scale = brentq(mismatch, 0.0, np.log(100.0), xtol=2e-4)
    matched_young = MOBILE_E * np.exp(matched_log_scale)
    matched = Case("matched_uniform", layer_young=matched_young)
    cases.append(matched)
    matched_result = run(matched, calibration)
    matched_error = abs(matched_result["load_displacement_mm"] / target - 1)
    assert matched_error < 5e-4
    holdouts = [
        Load("holdout_y_right", 1, 15.0),
        Load("holdout_x_over_band", 0, 10.0),
        Load("holdout_x_larger", 0, 15.0, force=0.006),
    ]
    for load in holdouts:
        for case in cases:
            run(case, load)
    field_case = Case(
        "regional_thickness", attachment_center=10.0, thickness_field=True
    )
    run(field_case, calibration)
    run(
        Case(
            "eh_equivalent",
            attachment_center=10.0,
            thickness_field=True,
            skin_young_scale=2.0,
            thickness_scale=0.5,
        ),
        calibration,
    )
    run(
        Case(
            "twice_stiffness_known_force", attachment_center=10.0, stiffness_scale=2.0
        ),
        calibration,
    )
    for refinement in range(2, cfg.maximum_refinement + 1):
        for case in [cases[1], cases[2], matched]:
            for load in [calibration, holdouts[0]]:
                run(case, load, refinement)
    predictions = []
    for load in [calibration, *holdouts]:
        truth = next(
            r
            for r in rows
            if r["case"] == "localized_attachment"
            and r["load"] == load.name
            and r["refinement"] == 1
        )
        comparison = next(
            r
            for r in rows
            if r["case"] == "matched_uniform"
            and r["load"] == load.name
            and r["refinement"] == 1
        )
        reference_skin = snapshots[("localized_attachment", load.name, 1)][0]
        candidate_skin, candidate_info = snapshots[("matched_uniform", load.name, 1)]
        delta = (
            candidate_skin.point_data["DisplacementMm"]
            - reference_skin.point_data["DisplacementMm"]
        )
        predictions.append(
            {
                "load": load.name,
                "reference_displacement_mm": truth["load_displacement_mm"],
                "matched_uniform_displacement_mm": comparison["load_displacement_mm"],
                "relative_compliance_error": comparison["load_displacement_mm"]
                / truth["load_displacement_mm"]
                - 1,
                "full_surface_displacement_rms_error_mm": float(
                    np.sqrt(
                        np.average(
                            np.sum(delta**2, axis=1),
                            weights=candidate_info["node_areas"],
                        )
                    )
                ),
            }
        )
    field_u = snapshots[("regional_thickness", calibration.name, 1)][0].point_data[
        "DisplacementMm"
    ]
    equivalent_u = snapshots[("eh_equivalent", calibration.name, 1)][0].point_data[
        "DisplacementMm"
    ]
    eh_error = float(np.max(np.abs(field_u - equivalent_u)))
    assert eh_error < 1e-7
    sensitivity = np.empty((2, 2))
    delta = 0.02
    reference_case = cases[2]
    for column, parameter in enumerate(["layer_young", "attachment_stiffness_scale"]):
        center_value = getattr(reference_case, parameter)
        for row, load in enumerate([calibration, holdouts[0]]):
            values = []
            for sign in [-1, 1]:
                perturbed = replace(
                    reference_case,
                    name=f"sensitivity_{parameter}_{sign}",
                    **{parameter: center_value * np.exp(sign * delta)},
                )
                values.append(run(perturbed, load, save=False)["load_displacement_mm"])
            sensitivity[row, column] = (np.log(values[1]) - np.log(values[0])) / (
                2 * delta
            )
    singular_values = np.linalg.svd(sensitivity, compute_uv=False)
    summary = {
        "protocol": manifest,
        "runs": rows,
        "uniform_calibration": {
            "target_case": "localized_attachment",
            "observable": "force-weighted displacement along applied force",
            "matched_young_mpa": matched_young,
            "relative_calibration_error": matched_error,
            "trials": calibration_trials,
        },
        "held_out_predictions": predictions,
        "eh_equivalence_max_displacement_error_mm": eh_error,
        "completed_forward_solves": len(solve_records),
        "local_identifiability": {
            "observable_rows": ["x_compliance", "y_compliance"],
            "parameter_columns": ["layer_young", "attachment_stiffness_scale"],
            "method": "central differences of log observables versus log parameters, perturbation 0.02; illustrative synthetic local sensitivity",
            "normalized_sensitivity_matrix": sensitivity.tolist(),
            "singular_values": singular_values.tolist(),
            "two_load_condition_number": float(singular_values[0] / singular_values[1]),
            "single_scalar_maximum_rank": 1,
        },
    }
    write_json(output / "summary.json", summary)
    render(output, rows, predictions, snapshots)


def render(output: Path, rows: list, predictions: list, snapshots: dict) -> None:
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = {
        "homogeneous": "#777777",
        "mobile_layer": "#62a5b5",
        "localized_attachment": "#b45c3b",
        "shifted_attachment": "#ab85b5",
        "matched_uniform": "#283f74",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    order = list(colors)
    cal = [
        next(
            r
            for r in rows
            if r["case"] == name
            and r["load"] == "calibration_x_right"
            and r["refinement"] == 1
        )
        for name in order
    ]
    labels = [name.replace("_", "\n") for name in order]
    axes[0].bar(
        labels, [r["load_displacement_mm"] for r in cal], color=list(colors.values())
    )
    axes[0].set(
        ylabel="Loaded-patch displacement (mm)", title="Same known x load: 2 mN"
    )
    axes[1].bar(
        [
            p["load"]
            .replace("calibration_", "fit\n")
            .replace("holdout_", "test\n")
            .replace("_", " ")
            for p in predictions
        ],
        [100 * p["relative_compliance_error"] for p in predictions],
        color="#283f74",
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set(
        ylabel="Uniform model compliance error (%)",
        title="Fit one scalar; predict other loads",
    )
    for name in ["mobile_layer", "localized_attachment", "matched_uniform"]:
        values = sorted(
            [r for r in rows if r["case"] == name and r["load"] == "holdout_y_right"],
            key=lambda r: r["refinement"],
        )
        axes[2].plot(
            [r["refinement"] for r in values],
            [r["load_displacement_mm"] for r in values],
            "o-",
            label=name.replace("_", " "),
            color=colors[name],
        )
    axes[2].set(
        xlabel="Mesh refinement factor",
        ylabel="Displacement under y load (mm)",
        title="Fixed r1 parameters across meshes",
        xticks=[1, 2, 3],
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "Synthetic layered patch: localized attachments versus uniform material",
        fontsize=14,
    )
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    names = ["mobile_layer", "localized_attachment", "matched_uniform"]
    max_u = max(
        float(
            np.max(
                snapshots[(name, "calibration_x_right", 1)][0].point_data[
                    "DisplacementMm"
                ][:, 0]
            )
        )
        for name in names
    )
    for ax, name in zip(axes, names, strict=True):
        skin, _ = snapshots[(name, "calibration_x_right", 1)]
        uv = skin.point_data["DisplacementMm"]
        tri = skin.faces.reshape(-1, 4)[:, 1:]
        plot = ax.tripcolor(
            skin.points[:, 0],
            skin.points[:, 1],
            tri,
            uv[:, 0],
            shading="gouraud",
            vmin=0,
            vmax=max_u,
            cmap="viridis",
        )
        ax.set(
            xlabel="x (mm)",
            ylabel="y (mm)",
            title=name.replace("_", " "),
            aspect="equal",
        )
        ax.plot([10, 10], [0, 10], "--", color="white", linewidth=1)
    fig.colorbar(plot, ax=axes, label="Surface x displacement (mm)", shrink=0.7)
    fig.suptitle("Same external load; dashed line marks the attachment-band center")
    fig.savefig(output / "surface-response.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    cherries.main(
        main,
        profile="debug"
        if os.environ.get("DEBUG") == "1"
        else ProfileRecordWithoutCommit,
    )
