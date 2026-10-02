"""Visualize prescribed-skin PNCG evidence without amplifying deformation."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_data import PreparedInputs
from matplotlib.ticker import MaxNLocator

from liblaf import cherries


class Config(cherries.BaseConfig):
    input_dir: Path = GROUP / "data/simple-skin-forward-inputs-001"
    run_dir: Path = GROUP / "data/simple-skin-forward-011"
    output_dir: Path = GROUP / "data/simple-skin-forward-visuals-011"
    skin_field: Path | None = (
        GROUP / "data/simple-skin-forward-nu049-inputs-001/skin-field.npz"
    )


def load_live_jsonl(path: Path) -> list[dict]:
    """Read complete JSONL objects while permitting one incomplete live tail."""
    rows = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            break
        if not isinstance(value, dict):
            message = f"JSONL row {index + 1} in {path} is not an object"
            raise TypeError(message)
        rows.append(value)
    return rows


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(
        r"checkpoint-(?:(?:terminal|wall-cap)-)?step-(\d+)\.npz", path.name
    )
    assert match is not None, path
    return int(match.group(1))


def load_trace_lineage(run_dir: Path, end_step: int) -> tuple[list[dict], list[dict]]:
    """Follow restart checkpoints and assign cumulative accepted iterations."""
    protocol = json.loads((run_dir / "protocol.json").read_text())
    restart_checkpoint = protocol["inputs"].get("restart_checkpoint")
    rows: list[dict] = []
    segments: list[dict] = []
    cumulative_start = 0
    if restart_checkpoint:
        restart_path = Path(restart_checkpoint)
        parent_rows, segments = load_trace_lineage(
            restart_path.parent, checkpoint_step(restart_path)
        )
        rows.extend(parent_rows)
        cumulative_start = int(segments[-1]["cumulative_end"])
    source = [
        row
        for row in load_live_jsonl(run_dir / "trace.jsonl")
        if int(row["step"]) <= end_step
    ]
    assert source
    run_id = run_dir.name.removeprefix("simple-skin-forward-")
    for row in source:
        local_step = int(row["step"])
        if segments and local_step == 0:
            continue
        rows.append(
            {
                **row,
                "source_run": run_id,
                "segment_step": local_step,
                "cumulative_step": cumulative_start + local_step,
            }
        )
    segment_end = max(int(row["step"]) for row in source)
    segments.append(
        {
            "run": run_id,
            "local_start": 0,
            "local_end": segment_end,
            "cumulative_start": cumulative_start,
            "cumulative_end": cumulative_start + segment_end,
            "trace_path": str((run_dir / "trace.jsonl").resolve()),
        }
    )
    return rows, segments


def main(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915 - compose scientific review panels
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.run_dir / "summary.json"
    result = json.loads(summary_path.read_text()) if summary_path.exists() else None
    protocol = json.loads((cfg.run_dir / "protocol.json").read_text())
    inversions_allowed = (
        protocol["solver"]["inversion_policy"]
        == "diagnostic only; user permits inverted cells"
    )
    if result:
        checkpoint = result["checkpoint"]
    else:
        records = [
            json.loads(p.read_text())
            for p in [
                *cfg.run_dir.glob("checkpoint-step-*.json"),
                *cfg.run_dir.glob("checkpoint-terminal-step-*.json"),
                *cfg.run_dir.glob("checkpoint-wall-cap-step-*.json"),
            ]
        ]
        checkpoint = max(records, key=lambda row: row["step"])
    path = Path(checkpoint["path"])
    assert sha256(path) == checkpoint["sha256"]
    lineage, trace_segments = load_trace_lineage(cfg.run_dir, int(checkpoint["step"]))
    u = np.load(path)["displacement_m"]
    prepared = PreparedInputs.load(
        cfg.input_dir / "prepared/inputs.npz", cfg.input_dir / "prepared/manifest.json"
    )
    mesh = pv.read(prepared.volume_path)
    skin = pv.read(prepared.skin_path)
    points = np.asarray(mesh.points)
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    deformed = skin.copy()
    deformed.points = points[ids] + u[ids]
    deformed.point_data["motion_mm"] = 1000 * np.linalg.norm(u[ids], axis=1)
    cells = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    dm = np.transpose(points[cells[:, 1:]] - points[cells[:, :1]], (0, 2, 1))
    current = points + u
    ds = np.transpose(current[cells[:, 1:]] - current[cells[:, :1]], (0, 2, 1))
    detf = np.linalg.det(ds) / np.linalg.det(dm)
    mesh.points = current
    mesh.cell_data["PhysicalDetF"] = detf
    mesh.point_data["ForwardDisplacement_m"] = u
    mesh.save(cfg.output_dir / "deformed-volume.vtu")
    deformed.save(cfg.output_dir / "deformed-skin.vtp")
    spec = importlib.util.spec_from_file_location(
        "visuals", Path(__file__).with_name("17-audit-source-bone-contact.py")
    )
    assert spec is not None
    assert spec.loader is not None
    visuals = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visuals)
    run_id = cfg.run_dir.name.removeprefix("simple-skin-forward-")
    restart_checkpoint = protocol["inputs"].get("restart_checkpoint")
    parent_frame = None
    if restart_checkpoint:
        restart_path = Path(restart_checkpoint)
        parent_frame = (
            f"parent {restart_path.parent.name.removeprefix('simple-skin-forward-')} "
            f"step {checkpoint_step(restart_path)}"
        )
    segment_frame = f"run {run_id} segment-local step {checkpoint['step']}"
    state_label = segment_frame + (
        " — converged" if result and result["success"] else " — not converged"
    )
    scalar_bar = {
        "vertical": False,
        "position_x": 0.28,
        "position_y": 0.035,
        "width": 0.44,
        "height": 0.06,
        "title_font_size": 18,
        "label_font_size": 16,
        "n_labels": 5,
    }
    assets = []
    with np.load(cfg.skin_field or cfg.input_dir / "skin-field.npz") as archive:
        skin.cell_data["E_kPa"] = archive["E_mpa"] * 1000
        skin.cell_data["N0_N_per_m"] = archive["baseline_N_per_m"][:, 0, 0]
    for name, title, label in (
        ("E_kPa", "Prescribed heterogeneous skin stiffness", "Derived E (kPa)"),
        (
            "N0_N_per_m",
            "Prescribed heterogeneous skin prestress",
            "Isotropic resultant (N/m)",
        ),
    ):
        plot = visuals.plotter(
            f"{title}\nFlynn regional fits + manual anchors + smooth interpolation"
        )
        plot.add_mesh(
            skin,
            scalars=name,
            cmap="viridis",
            smooth_shading=True,
            scalar_bar_args={**scalar_bar, "title": label},
        )
        filename = f"00-skin-{name}.png"
        visuals.save(
            plot,
            cfg.output_dir / filename,
            visuals.camera(skin, "front", zoom=0.82),
        )
        assets.append(
            {
                "filename": filename,
                "caption": f"{title}. Prescribed prior from sparse literature inverse fits with a 20 mm smooth spatial blend; not a measured subject map.",
            }
        )
    with np.load(cfg.input_dir / "geometry.npz") as archive:
        bones = {
            name: pv.PolyData(
                archive[f"{name}_points_m"],
                np.column_stack(
                    (
                        np.full(len(archive[f"{name}_faces"]), 3),
                        archive[f"{name}_faces"],
                    )
                ),
            )
            for name in ("cranium", "mandible")
        }
    for view in ("front", "side"):
        plot = visuals.plotter(
            f"Prescribed skin stress · {state_label}\n{view.title()} · true scale, no displacement amplification"
        )
        plot.add_mesh(
            deformed,
            scalars="motion_mm",
            cmap="viridis",
            clim=(0, max(float(deformed["motion_mm"].max()), 1e-6)),
            smooth_shading=True,
            scalar_bar_args={**scalar_bar, "title": "Displacement (mm)"},
        )
        filename = f"02-displacement-{view}.png"
        visuals.save(
            plot,
            cfg.output_dir / filename,
            visuals.camera(skin, view, zoom=0.82),
        )
        assets.append(
            {
                "filename": filename,
                "caption": f"{state_label}. Surface displacement relative to the repaired reference, at actual scale.",
            }
        )
        plot = pv.Plotter(off_screen=True, shape=(1, 2), window_size=visuals.WINDOW)
        plot.set_background(visuals.BACKGROUND)
        plot.enable_anti_aliasing("ssaa")
        comparison_camera = visuals.camera(skin, view, zoom=0.88)
        for column, surface, color, panel_title in (
            (0, skin, "#858585", "Repaired reference"),
            (1, deformed, "#168b8b", f"Run {run_id} current"),
        ):
            plot.subplot(0, column)
            plot.add_text(
                f"{panel_title}\n{view.title()} · true scale",
                position="upper_left",
                font_size=15,
                color="#202124",
            )
            plot.add_mesh(surface, color=color, smooth_shading=True)
            plot.camera_position = comparison_camera["position"]
            plot.camera.parallel_projection = True
            plot.camera.parallel_scale = comparison_camera["parallel_scale"]
            plot.reset_camera_clipping_range()
        filename = f"03-overlay-{view}.png"
        plot.show(screenshot=cfg.output_dir / filename, auto_close=True)
        assets.append(
            {
                "filename": filename,
                "caption": f"Side-by-side repaired reference and current forward surface at the same camera and true length scale. This avoids transparency z-fighting; surface facets are mesh rendering, not inferred wrinkles. Frame: {segment_frame}"
                + (f", restarted from {parent_frame}." if parent_frame else "."),
            }
        )
    plot = visuals.plotter(
        f"Full source skull + forward skin · {state_label}\nComplete bones fixed; soft-bone collision enabled"
    )
    plot.add_mesh(bones["cranium"], color="#ddc69d", smooth_shading=True)
    plot.add_mesh(bones["mandible"], color="#598999", smooth_shading=True)
    plot.add_mesh(deformed, color="#de937d", opacity=0.22)
    filename = "04-full-skull-contact-context.png"
    visuals.save(plot, cfg.output_dir / filename, visuals.camera(skin, "side"))
    assets.append(
        {
            "filename": filename,
            "caption": "Complete registered cranium and mandible with translucent deformed skin. The bones remain fixed; bone-bone collision is outside this forward test.",
        }
    )
    invalid_ids = np.flatnonzero(detf <= 0)
    if len(invalid_ids):
        invalid_surface = mesh.extract_cells(invalid_ids).extract_surface(
            algorithm=None
        )
        plot = visuals.plotter(
            f"Cell inversion diagnostic: {len(invalid_ids)} tetrahedra\nRed markers: inverted-tet locations (enlarged for visibility)"
        )
        plot.add_mesh(bones["cranium"], color="#ddc69d", opacity=0.18)
        plot.add_mesh(bones["mandible"], color="#598999", opacity=0.18)
        plot.add_mesh(skin, color="#999999", opacity=0.08)
        plot.add_mesh(invalid_surface, color="#e32929", show_edges=True, line_width=3)
        filename = "05-inverted-tetrahedra-location.png"
        visuals.save(plot, cfg.output_dir / filename, visuals.camera(skin, "front"))
        assets.append(
            {
                "filename": filename,
                "caption": f"{len(invalid_ids)} inverted tetrahedra at the saved accepted PNCG step. Red identifies their location. "
                + (
                    "The user permits inversions for this diagnostic; force convergence is assessed separately."
                    if inversions_allowed
                    else "The positive-volume preparation criterion is not met."
                ),
            }
        )
    limiting_id = int(np.argmin(detf))
    if detf[limiting_id] < 0.01:
        tet = cells[limiting_id]
        center = points[tet].mean(axis=0)
        reference_tet = (points[tet] - center) * 1e6
        current_tet = (current[tet] - center) * 1e6
        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
        axis = fig.add_subplot(projection="3d")
        for xyz, color, label in (
            (reference_tet, "#777777", "Reference"),
            (current_tet, "#cf3e2d", "Saved PNCG state"),
        ):
            axis.scatter(*xyz.T, color=color, s=28, label=label)
            for a in range(4):
                for b in range(a + 1, 4):
                    axis.plot(*xyz[[a, b]].T, color=color, linewidth=1.6)
        axis.set(xlabel="X (µm)", ylabel="Y (µm)", zlabel="Z (µm)")
        axis.set_box_aspect(
            np.maximum(np.ptp(np.vstack((reference_tet, current_tet)), axis=0), 1)
        )
        for coord_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            coord_axis.set_major_locator(MaxNLocator(3))
        axis.legend(loc="upper right")
        axis.view_init(elev=24, azim=38)
        fig.suptitle(
            f"Limiting tetrahedron {limiting_id}: det(F) = {detf[limiting_id]:.3g}\nRelative coordinates; same length scale in all axes"
        )
        filename = "05-limiting-tetrahedron.png"
        fig.savefig(
            cfg.output_dir / filename, dpi=180, bbox_inches="tight", pad_inches=0.3
        )
        plt.close(fig)
        assets.append(
            {
                "filename": filename,
                "caption": f"Reference and saved shapes of cell {limiting_id}, shown in micrometres relative to its reference centroid. Cell orientation and distortion are reported separately from the force-convergence criterion.",
            }
        )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    exact = [row for row in lineage if "accepted_state_free_force_norm" in row]
    current_segment = trace_segments[-1]
    if result and exact[-1]["cumulative_step"] != current_segment["cumulative_end"]:
        exact.append(
            {
                "source_run": run_id,
                "segment_step": checkpoint["step"],
                "cumulative_step": current_segment["cumulative_end"],
                "accepted_state_free_force_norm": result["final_free_force_norm"],
            }
        )
    axes[0].semilogy(
        [row["cumulative_step"] for row in exact],
        [row["accepted_state_free_force_norm"] * 1e6 for row in exact],
        color="#087d81",
    )
    force_threshold = (
        result["force_threshold"]
        if result
        else protocol["solver"]["force_threshold_override"]
    )
    if force_threshold is not None:
        axes[0].axhline(
            force_threshold * 1e6,
            color="#ae614a",
            linestyle="--",
            label=f"force target: {force_threshold * 1e6:.4g} N",
        )
        axes[0].legend(fontsize=8)
    axes[0].set(
        xlabel="Cumulative accepted iterations",
        ylabel="Free force L2 norm (N)",
        title="Independently evaluated force",
    )
    axes[1].plot(
        [row["cumulative_step"] for row in lineage],
        [row["energy"] * 1e6 for row in lineage],
        color="#087d81",
    )
    axes[1].set(
        xlabel="Cumulative accepted iterations",
        ylabel="Energy (J)",
        title="Total model energy",
    )
    for axis in axes[:2]:
        for segment in trace_segments[1:]:
            boundary = segment["cumulative_start"]
            axis.axvline(boundary, color="#777777", linestyle=":", linewidth=0.9)
            axis.text(
                boundary,
                0.98,
                f"run {segment['run']} · local 0",
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color="#555555",
                transform=axis.get_xaxis_transform(),
            )
    axes[2].hist(detf, bins=150, color="#7da5ac", log=True)
    axes[2].axvline(1, color="#555555", linewidth=0.8)
    axes[2].set(
        xlabel="Physical det(F)",
        ylabel="Tet count (log)",
        title=f"J range {detf.min():.4g} to {detf.max():.4g}",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle(f"Simple forward: {state_label}")
    filename = "01-solver-and-volume-trends.png"
    fig.savefig(cfg.output_dir / filename, dpi=180)
    plt.close(fig)
    assets.insert(
        0,
        {
            "filename": filename,
            "caption": "Checkpoint-ancestry energy and independently recomputed free force use cumulative accepted iterations; dotted boundaries label each source run's local step zero. The volume histogram is the saved current state. Force convergence is not a mechanical stability proof.",
        },
    )
    weights = prepared.arrays["observation_weight_normalized"]
    observations = prepared.arrays["observation_node_ids"]
    write_json(
        cfg.output_dir / "summary.json",
        {
            "schema": "joint-simple-skin-forward-visuals-v1",
            "result": result,
            "inversions_allowed": inversions_allowed,
            "poisson_ratios": protocol["mechanics"].get("poisson_ratios"),
            "checkpoint": checkpoint,
            "step_frame": {
                "current": segment_frame,
                "parent": parent_frame,
                "definition": "current step is segment-local; parent step is the restart checkpoint",
            },
            "trace_segments": trace_segments,
            "force_threshold_n": force_threshold * 1e6,
            "input_manifest_sha256": sha256(prepared.manifest_path),
            "detF_min": float(detf.min()),
            "detF_max": float(detf.max()),
            "inverted_tetrahedra": int(np.count_nonzero(detf <= 0)),
            "surface_motion_rms_mm": float(
                1000 * np.sqrt(np.sum(weights[:, None] * u[observations] ** 2))
            ),
            "assets": assets,
        },
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
