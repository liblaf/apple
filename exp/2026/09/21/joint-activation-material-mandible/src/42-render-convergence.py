"""Render preparation convergence and physical shared-parameter trajectories."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_field_visuals import coefficient_summary

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    run_dir: Path
    output_dir: Path = GROUP / "data/convergence-visuals"


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=180)
    plt.close(fig)


def main(cfg: Config) -> None:  # noqa: C901, PLR0915 - compose related figure panels.
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = cfg.run_dir / "trace.json"
    trace = json.loads(trace_path.read_text())
    assert trace
    checkpoint_path = cfg.run_dir / "terminal.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["stage"] == "neutral"
    # A running experiment may have published its next trace before checkpointing.
    trace = [row for row in trace if row["update"] <= checkpoint["update"]]
    assert trace[-1]["update"] == checkpoint["update"]
    coefficients = np.asarray([row["shared"] for row in trace])
    assert coefficients.shape == (
        len(trace),
        checkpoint["materials"]["parameterization"]["shared_coefficient_count"],
    )
    updates = np.asarray([row["update"] for row in trace])
    seconds = np.asarray([row["elapsed_seconds"] for row in trace])
    pg = np.asarray([row["projected_gradient_inf"] for row in trace])
    objective = np.asarray([row["objective"] for row in trace])
    assert np.isfinite(coefficients).all()
    assert np.isfinite(pg).all()
    metrics = [row["metrics"] for row in trace]
    colors = ("#007d85", "#c26724", "#6845a3", "#426488")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), layout="constrained")
    axes[0, 0].plot(updates, objective, color=colors[0], label="Total objective")
    for key, label, color in zip(
        ("surface_loss", "muscle_loss", "weighted_prior"),
        ("Surface", "Muscle", "Weighted prior"),
        colors[1:],
        strict=True,
    ):
        axes[0, 0].plot(
            updates, [row["terms"][key] for row in trace], color=color, label=label
        )
    if checkpoint["materials"]["schema"] == "joint-additive-spatial-stress-fields-v1":
        axes[0, 0].plot(
            updates,
            [row["terms"]["weighted_spatial_roughness"] for row in trace],
            color="#946f1b",
            label="Weighted spatial smoothness",
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Objective and its components", loc="left")
    axes[0, 0].set_ylabel("Dimensionless loss")
    axes[0, 0].legend()
    axes[0, 1].semilogy(updates, pg, color=colors[0], label="Projected gradient, max")
    axes[0, 1].axhline(
        trace[-1]["convergence"]["projected_gradient_inf_tolerance"],
        color=colors[1],
        linestyle="--",
        label="Required threshold",
    )
    axes[0, 1].set_title("Constrained stationarity", loc="left")
    axes[0, 1].set_ylabel("Unit-step gradient mapping")
    axes[0, 1].legend()
    axes[1, 0].plot(
        updates,
        [row["surface_motion_rms_mm"] for row in metrics],
        color=colors[0],
        label="Surface RMS",
    )
    axes[1, 0].plot(
        updates,
        [row["muscle_centroid_motion_rms_mm"] for row in metrics],
        color=colors[1],
        label="Muscle-centroid RMS",
    )
    axes[1, 0].axhline(0.25, color=colors[0], linestyle="--", label="Surface budget")
    axes[1, 0].axhline(0.5, color=colors[1], linestyle="--", label="Muscle budget")
    axes[1, 0].set_title("Neutral geometry", loc="left")
    axes[1, 0].set_ylabel("Reference displacement (mm)")
    axes[1, 0].legend(fontsize=8)
    for key, label, color in zip(
        ("detF_min", "detF_p001", "detF_max"),
        ("Minimum", "0.1% quantile", "Maximum"),
        colors,
        strict=False,
    ):
        axes[1, 1].plot(
            updates, [row[key] for row in metrics], label=label, color=color
        )
    axes[1, 1].axhline(1.0, color="#7f8892", linestyle="--")
    axes[1, 1].set_title("Physical tetrahedral volume ratios", loc="left")
    axes[1, 1].set_ylabel("det(F)")
    axes[1, 1].legend(fontsize=8)
    axes[2, 0].semilogy(
        updates,
        [row["forward"]["grad_norm"] for row in trace],
        label="Forward free-force norm",
        color=colors[0],
    )
    axes[2, 0].set_title("Equilibrium accuracy", loc="left")
    axes[2, 0].set_ylabel("Model force units")
    axes[2, 0].axhline(
        1e-12, color=colors[1], linestyle="--", label="Absolute tolerance"
    )
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].plot(seconds / 60, objective, color=colors[0])
    axes[2, 1].set_title("Preparation cost", loc="left")
    axes[2, 1].set_xlabel("Elapsed minutes")
    axes[2, 1].set_ylabel("Total objective")
    axes[3, 0].semilogy(
        updates,
        [row["convergence"]["objective_relative_range"] for row in trace],
        color=colors[0],
        label="Objective range over the required window",
    )
    axes[3, 0].axhline(
        trace[-1]["convergence"]["objective_range_tolerance"],
        color=colors[1],
        linestyle="--",
        label="Required threshold",
    )
    axes[3, 0].set_title("Objective stabilization", loc="left")
    axes[3, 0].set_ylabel("Relative range")
    axes[3, 0].legend(fontsize=8)
    axes[3, 0].set_xlim(updates[0] - 0.1, max(1, updates[-1]) + 0.1)
    if len(trace) < trace[-1]["convergence"]["stabilization_window"]:
        axes[3, 0].text(
            0.5,
            0.65,
            "Required window not yet complete",
            ha="center",
            transform=axes[3, 0].transAxes,
        )
    axes[3, 1].plot(
        updates,
        [
            row["contact"].get("minimum_active_distance_m", np.nan) * 1e3
            if row["contact"].get("minimum_active_distance_m") is not None
            else np.nan
            for row in trace
        ],
        color=colors[0],
        label="Minimum active IPC gap",
    )
    axes[3, 1].set_title("Soft tissue-bone contact", loc="left")
    axes[3, 1].set_ylabel("Active contact gap (mm)")
    axes[3, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        if axis is not axes[2, 1]:
            axis.set_xlabel("Accepted preparation update")
    converged = bool(checkpoint["neutral_converged"])
    fig.suptitle(
        f"Neutral preparation · {'converged' if converged else 'not converged'} · {cfg.run_dir.name}",
        fontsize=14,
    )
    save_figure(fig, cfg.output_dir, "convergence")

    field_summaries = [
        coefficient_summary(row, checkpoint["materials"]) for row in coefficients
    ]
    spectra = np.asarray(
        [row["principal_anchor_min_mean_max_kpa"] for row in field_summaries]
    )
    resultants = [row["skin_resultant_n_per_m"] for row in field_summaries]
    stiffness = [row["skin_stiffness_multiplier"] for row in field_summaries]
    anchor_norms = np.asarray(
        [row["anchor_coordinate_rms_frobenius"] for row in field_summaries]
    )
    spatial = field_summaries[0]["spatial"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    for index, tissue in enumerate(("Fat", "Aponeurosis", "Muscle")):
        axis = axes[0, index]
        for principal in range(3):
            axis.plot(
                updates,
                spectra[:, index, 1, principal],
                label=f"Principal {principal + 1}"
                + (" anchor mean" if spatial else ""),
                color=colors[principal],
            )
            if spatial:
                axis.fill_between(
                    updates,
                    spectra[:, index, 0, principal],
                    spectra[:, index, 2, principal],
                    color=colors[principal],
                    alpha=0.15,
                )
        axis.axhline(0, color="#7f8892", linewidth=0.8)
        axis.set_title(f"{tissue}: signed baseline stress", loc="left")
        axis.set_ylabel("kPa; positive = tension")
        axis.legend(fontsize=8)
    axes[1, 0].plot(updates, resultants, color=colors[0])
    axes[1, 0].axhline(80.6, color=colors[1], linestyle="--", label="Literature proxy")
    axes[1, 0].set_title("Prescribed skin prestress", loc="left")
    axes[1, 0].set_ylabel("Membrane resultant (N/m)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(updates, stiffness, color=colors[0])
    axes[1, 1].axhline(1, color="#7f8892", linestyle="--", label="Reference modulus")
    axes[1, 1].set_title("Skin stiffness", loc="left")
    axes[1, 1].set_ylabel("Multiplier of the frozen reference")
    axes[1, 1].legend(fontsize=8)
    for index, tissue in enumerate(("Fat", "Aponeurosis", "Muscle")):
        axes[1, 2].plot(
            updates,
            anchor_norms[:, index],
            color=colors[index],
            label=tissue,
        )
    axes[1, 2].set_title("Anchor RMS of normalized baseline tensors", loc="left")
    axes[1, 2].set_ylabel("Frobenius norm / tissue shear modulus")
    axes[1, 2].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xlabel("Accepted preparation update")
    fig.suptitle(
        "Material preparation · signed shared stresses and stiffness"
        + (
            "\nLines: unweighted anchor means; shading: anchor ranges"
            if spatial
            else ""
        ),
        fontsize=14,
    )
    save_figure(fig, cfg.output_dir, "material-trajectories")
    write_json(
        cfg.output_dir / "summary.json",
        {
            "schema": "joint-preparation-plots-v1",
            "run_dir": str(cfg.run_dir.resolve()),
            "trace_sha256": sha256(trace_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "accepted_evaluations": len(trace),
            "neutral_converged": converged,
            "neutral_budget_met": checkpoint["neutral_budget_met"],
            "figures": ["convergence.png", "material-trajectories.png"],
            "scope": "preparation diagnostics; not final joint-optimization trends",
            "field_summary_scope": field_summaries[0]["scope"],
        },
    )
    LOG.info("Rendered preparation convergence and physical material trajectories")
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
