"""Render the measured rigid-bone collision fractions for jaw proposals."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_data import PreparedInputs

from liblaf import cherries


class Config(cherries.BaseConfig):
    validation: Path = GROUP / "data/rigid-bone-ccd-validation-003/summary.json"
    jaw_preflight: Path | None = None
    prepared_dir: Path = GROUP / "data/prepared"
    output_dir: Path = GROUP / "data/jaw-domain-visuals-001"


def probe_label(row: dict) -> str:
    label = row["label"]
    if label == "zero":
        return "Zero pose"
    if label == "legacy_one_degree_center":
        return "Rotation x +1° (legacy diagnostic)"
    if label == "world_x_positive_0p01_degree":
        return "Rotation x +0.01°"
    coordinate = int(label.split("_")[1])
    value = row["pose_rad_m"][coordinate]
    axis = "xyz"[coordinate % 3]
    return (
        f"Rotation {axis} {np.rad2deg(value):+g}°"
        if coordinate < 3
        else f"Translation {axis} {1000 * value:+g} mm"
    )


def render_preflight(cfg: Config) -> tuple[dict, list[dict]]:  # noqa: C901, PLR0912, PLR0915
    assert cfg.jaw_preflight is not None
    receipt = json.loads(cfg.jaw_preflight.read_text())
    assert receipt["schema"] in ("joint-jaw-preflight-v1", "joint-jaw-preflight-v2")
    assert receipt["input_arrays_sha256"] == sha256(cfg.prepared_dir / "inputs.npz")
    assert receipt["input_manifest_sha256"] == sha256(
        cfg.prepared_dir / "manifest.json"
    )
    assert receipt["rigid_bone_ccd_validation_sha256"] == sha256(cfg.validation)
    assert len(receipt["rows"]) == receipt["probe_count"]
    assert receipt["diagnostic_only"] == (
        receipt["admission_mode"] == "diagnostic_only"
    )
    assert receipt["final_launch_ready"] == (
        receipt["success"] and receipt["admission_mode"] == "final_launch_ready"
    )
    assets = []
    rows = receipt["rows"]
    for row in rows:
        assert row["numerically_admissible"] == (row["status"] == "accepted")
        assert row["seed_sha256"] == receipt["seed_sha256"]
        assert row["seed_matches_neutral"] is True
        if row["numerically_admissible"]:
            assert row["forward"]["success"] is True
            assert row["contact"]["contact_numerically_valid"] is True
            assert row["geometry"]["numerical_geometry_admissible"] is True
            assert all(flag is True for flag in row["shape_checks"].values())
        else:
            assert "state" not in row
    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    labels = [probe_label(row) for row in rows]
    colors = ["#25776d" if row["numerically_admissible"] else "#b45543" for row in rows]
    axis.scatter(np.zeros(len(rows)), np.arange(len(rows)), c=colors, s=65)
    for index, row in enumerate(rows):
        axis.text(
            0.04, index, row["status"].replace("_", " "), va="center", fontsize=10
        )
    axis.set_yticks(np.arange(len(rows)), labels=labels)
    axis.invert_yaxis()
    axis.set_xticks([])
    axis.set_xlim(-0.02, 1)
    axis.set_title(
        f"Full-face jaw preflight: {receipt['status'].replace('_', ' ')}\nIndependent neutral seed for every probe",
        loc="left",
        fontsize=14,
    )
    if receipt["schema"] == "joint-jaw-preflight-v2":
        reproducibility = receipt["geometry_reproducibility"]
        metrics = reproducibility["metrics"]
        if metrics is not None:
            maximum = metrics["maximum_euclidean_nodal_displacement_difference_m"]
            figure.supxlabel(
                f"Zero-pose repeatability: maximum nodal difference {maximum * 1e6:.3g} µm; "
                f"declared budget {reproducibility['budget_m'] * 1e6:g} µm.\n"
                "Large proposal failures remain diagnostics; accepted means numerical checks passed.",
                fontsize=10,
            )
    else:
        figure.supxlabel(
            "Historical v1: accepted zero solve failed an incorrectly dimensioned identity test.\n"
            "The v2 rerun uses a separately declared geometry repeatability budget.",
            fontsize=10,
        )
    filename = "02-contact-jaw-preflight-status.png"
    figure.savefig(cfg.output_dir / filename, dpi=180)
    plt.close(figure)
    assets.append(
        {
            "filename": filename,
            "caption": "Actual contact-enabled preflight outcomes. Red rows are rejected or failed candidates, not equilibrium shapes. Only accepted cases have saved displacement renders. Diagnostic-only results cannot authorize the final launch.",
        }
    )

    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz",
        cfg.prepared_dir / "manifest.json",
        verify_sources=True,
    )
    volume = pv.read(prepared.volume_path)
    skin = pv.read(prepared.skin_path).triangulate()
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    accepted = []
    for row in rows:
        if not row["numerically_admissible"]:
            continue
        state = row["state"]
        state_path = Path(state["path"])
        assert sha256(state_path) == state["sha256"]
        with np.load(state_path, allow_pickle=False) as archive:
            assert str(archive["schema"].item()) == "joint-jaw-preflight-state-v1"
            assert str(archive["label"].item()) == row["label"]
            assert str(archive["role"].item()) == row["role"]
            for key in (
                "neutral_checkpoint_sha256",
                "input_arrays_sha256",
                "input_manifest_sha256",
                "seed_sha256",
            ):
                assert str(archive[key].item()) == receipt[key], key
            displacement = archive["full_displacement_m"].copy()
            assert displacement.dtype == np.dtype("float64")
            assert displacement.shape == (volume.n_points, 3)
            assert list(displacement.shape) == state["displacement_shape"]
            assert displacement.dtype.str == state["displacement_dtype"]
            assert np.isfinite(displacement).all()
            assert np.array_equal(archive["pose_rad_m"], row["pose_rad_m"])
            assert np.array_equal(archive["normalized_pose"], row["normalized_pose"])
        accepted.append((row, displacement))
    assert accepted, "no accepted preflight state available for geometry rendering"
    source = Path(__file__).with_name("45-render-neutral-state.py")
    spec = importlib.util.spec_from_file_location("jaw_preflight_plotting", source)
    assert spec is not None
    assert spec.loader is not None
    plotting = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plotting
    spec.loader.exec_module(plotting)
    limit = max(
        float(np.linalg.norm(displacement[ids], axis=1).max()) * 1000
        for _, displacement in accepted
    )
    assert limit > 0
    for row, displacement in accepted:
        deformed = skin.copy(deep=True)
        deformed.points = np.asarray(skin.points) + displacement[ids]
        deformed.point_data["reference_motion_mm"] = (
            np.linalg.norm(displacement[ids], axis=1) * 1000
        )
        for view in ("front", "side"):
            filename = f"03-{row['label']}-{view}.png"
            plot = plotting.plotter(
                f"Jaw preflight: {probe_label(row)} — {view}\n{receipt['admission_mode'].replace('_', ' ')}; numerically accepted"
            )
            plot.add_mesh(
                deformed,
                scalars="reference_motion_mm",
                cmap="viridis",
                clim=(0, limit),
                smooth_shading=True,
                scalar_bar_args=plotting.scalar_bar("motion (mm)"),
            )
            plotting.save(plot, cfg.output_dir / filename, plotting.camera(skin, view))
            assets.append(
                {
                    "filename": filename,
                    "caption": f"Accepted {probe_label(row)} preflight geometry, {view}. Actual displacement from the FEM reference, in mm, on the same 0 to {limit:.5g} scale across accepted cases. Mode: {receipt['admission_mode'].replace('_', ' ')}. No deformation magnification; numerical acceptance does not establish anatomical validity.",
                }
            )
    zero_displacement = next(
        displacement for row, displacement in accepted if row["label"] == "zero"
    )
    for row, displacement in accepted:
        if row["label"] == "zero":
            continue
        increment_um = (
            np.linalg.norm(displacement[ids] - zero_displacement[ids], axis=1) * 1e6
        )
        incremental_limit = float(increment_um.max())
        assert incremental_limit > 0
        deformed = skin.copy(deep=True)
        deformed.points = np.asarray(skin.points) + displacement[ids]
        deformed.point_data["jaw_increment_um"] = increment_um
        for view in ("front", "side"):
            filename = f"04-{row['label']}-increment-{view}.png"
            plot = plotting.plotter(
                f"Jaw-induced change: {probe_label(row)} — {view}\nRelative to resolved zero; actual scale"
            )
            plot.add_mesh(
                deformed,
                scalars="jaw_increment_um",
                cmap="viridis",
                clim=(0, incremental_limit),
                smooth_shading=True,
                scalar_bar_args=plotting.scalar_bar("increment (µm)"),
            )
            plotting.save(plot, cfg.output_dir / filename, plotting.camera(skin, view))
            assets.append(
                {
                    "filename": filename,
                    "caption": f"{probe_label(row)}, {view}: Euclidean surface displacement difference from the independently resolved zero-pose state, in micrometres. Both states share the same materials and starting seed. Skin RMS difference {np.sqrt(np.mean(increment_um**2)):.5g} µm; maximum {incremental_limit:.5g} µm. Geometry is not magnified.",
                }
            )
    for asset in assets:
        asset["sha256"] = sha256(cfg.output_dir / asset["filename"])
    return {
        "path": str(cfg.jaw_preflight.resolve()),
        "sha256": sha256(cfg.jaw_preflight),
        "schema": receipt["schema"],
        "success": receipt["success"],
        "admission_mode": receipt["admission_mode"],
        "diagnostic_only": receipt["diagnostic_only"],
        "status": receipt["status"],
        "final_launch_ready": receipt["final_launch_ready"],
        "accepted_state_count": len(accepted),
    }, assets


def main(cfg: Config) -> None:
    receipt = json.loads(cfg.validation.read_text())
    assert receipt["schema"] == "joint-rigid-bone-ccd-validation-v1"
    assert receipt["success"] is True
    assert receipt["whole_pose_box_validated"] is False
    for source, digest in receipt["sources"].items():
        assert sha256(Path(source)) == digest, source
    rows = receipt["probes"]
    assert len(rows) == 13
    assert rows[0]["label"] == "zero"
    assert rows[0]["numerically_admissible"]
    labels, values = [], []
    for coordinate, name in enumerate(
        (
            "Rotation x",
            "Rotation y",
            "Rotation z",
            "Translation x",
            "Translation y",
            "Translation z",
        )
    ):
        for sign in (-1, 1):
            row = rows[1 + 2 * coordinate + (sign == 1)]
            assert row["label"] == f"coordinate_{coordinate}_{sign:+d}"
            amount = "10°" if coordinate < 3 else "5 mm"
            labels.append(f"{name}  {'-' if sign == -1 else '+'}{amount}")
            values.append(row["collision_free_fraction"])
    values = np.asarray(values)
    assert np.all(values > 0)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    figure, axis = plt.subplots(figsize=(10, 7.4), constrained_layout=True)
    y = np.arange(len(labels))
    axis.hlines(y, 0.005, values, color="#bf6554", linewidth=5, alpha=0.75)
    axis.scatter(values, y, color="#943b31", s=40, zorder=3)
    for i, value in enumerate(values):
        axis.annotate(
            f"{100 * value:.2f}%",
            (value, i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )
    axis.axvline(1.0, color="#25776d", linestyle="--", linewidth=1.5)
    axis.set_xscale("log")
    axis.set_xlim(0.005, 1.6)
    axis.set_xticks([0.01, 0.1, 1.0], labels=["1%", "10%", "100%"])
    axis.set_yticks(y, labels=labels)
    axis.invert_yaxis()
    axis.grid(axis="x", which="both", alpha=0.15)
    axis.set_xlabel(
        "Conservative collision-free fraction of the attempted linear vertex motion"
    )
    axis.set_title(
        "Jaw proposal bounds are not a feasible motion box\nPure FEM mandible versus cranium; all 12 extreme proposals rejected",
        loc="left",
        fontsize=14,
        pad=16,
    )
    figure.supxlabel(
        "Zero pose and the small rigid-only seed test pass; full-face contact checks are separate.\nFractions are numerical path limits, not physiological ranges or complete-source-bone validation.",
        fontsize=10,
    )
    filename = "01-jaw-rigid-bone-ccd-proposals.png"
    figure.savefig(cfg.output_dir / filename, dpi=180)
    plt.close(figure)
    preflight = None
    preflight_assets = []
    if cfg.jaw_preflight is not None:
        preflight, preflight_assets = render_preflight(cfg)
    write_json(
        cfg.output_dir / "summary.json",
        {
            "schema": "joint-jaw-domain-visuals-v1",
            "validation": {
                "path": str(cfg.validation.resolve()),
                "sha256": sha256(cfg.validation),
            },
            "preflight": preflight,
            "assets": [
                {
                    "filename": filename,
                    "caption": "Rigid-bone CCD rejects all 12 extreme jaw proposals from zero. Each mark is IPC's conservative safe fraction along the solver's linear vertex path. The computational box remains a proposal bound; no whole-box or anatomical validity is claimed.",
                    "sha256": sha256(cfg.output_dir / filename),
                },
                *preflight_assets,
            ],
            "whole_pose_box_validated": False,
            "scientific_scope": "numerical collision geometry and explicitly bound accepted preflight states when supplied",
        },
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
