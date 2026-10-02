"""Visualize all three signed effective-activation eigenmodes on reference geometry."""

# ruff: noqa: PERF401, PLR0915

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from matplotlib import colors
from muscle_glyph_context import (
    RegionVisibility,
    build_muscle_region_context,
    save_region_visibility,
    set_parallel_camera,
    visible_region_mask,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[6]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
WINDOW = (1500, 1500)
PITCH = 17
LINE_PIXELS = 13
ZERO_TOL = 1e-8
GAP_TOL = 1e-6
BACKGROUND = "#f4f2ed"
NORM = colors.SymLogNorm(linthresh=0.01, linscale=0.5, vmin=-40, vmax=40, base=10)
CMAP = colors.LinearSegmentedColormap.from_list(
    "signed_activation", ["#053061", "#4393c3", "#b5b6b4", "#d6604d", "#67001f"]
)
TICKS = [-40, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 40]
TICK_LABELS = ["-40", "-1", "-0.1", "-0.01", "0", "0.01", "0.1", "1", "40"]
LABELS = [
    ("mode-1-principal", "1  Principal mode", "Strongest contraction direction"),
    ("mode-2-residual", "2  Residual component", "Intermediate signed mode"),
    ("mode-3-residual", "3  Residual component", "Most expansive direction"),
]


class Config(cherries.BaseConfig):
    source: Path = cherries.input("10-forward/baseline-replay.npz")
    output_dir: Path = cherries.output("71-eigenmodes", mkdir=True)


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def choose_sample(
    visibility: RegionVisibility, centers: np.ndarray, camera: dict
) -> np.ndarray:
    ids = np.flatnonzero(visibility.mask)
    pixel = visibility.projected_pixel_xy[ids]
    bins = (pixel[:, 0] // PITCH) + ((WINDOW[0] + PITCH - 1) // PITCH) * (
        pixel[:, 1] // PITCH
    )
    backward = np.asarray(camera["position"]) - np.asarray(camera["focal_point"])
    backward /= np.linalg.norm(backward)
    depth = centers[ids] @ backward
    order = np.lexsort((ids, -depth, bins))
    sorted_bins = bins[order]
    first = np.r_[True, sorted_bins[1:] != sorted_bins[:-1]]
    return np.sort(ids[order[first]])


def line_mesh(
    centers: np.ndarray,
    axes: np.ndarray,
    coefficients: np.ndarray,
    global_ids: np.ndarray,
    length: float,
) -> pv.PolyData:
    offsets = 0.5 * length * axes
    points = np.empty((2 * len(centers), 3))
    points[0::2], points[1::2] = centers - offsets, centers + offsets
    indices = np.arange(len(centers))
    edges = np.c_[np.full(len(centers), 2), 2 * indices, 2 * indices + 1]
    mesh = pv.PolyData(points, lines=edges.ravel())
    mesh.cell_data["Z_eigenvalue"] = coefficients
    mesh.cell_data["GlobalCellId"] = global_ids
    mesh.cell_data["ReferenceAxis"] = axes
    mesh.cell_data["ColorRGB"] = np.round(CMAP(NORM(coefficients))[:, :3] * 255).astype(
        np.uint8
    )
    return mesh


def raw_panel(
    skin: pv.PolyData,
    centers: np.ndarray,
    axes: np.ndarray,
    values: np.ndarray,
    unique: np.ndarray,
    ids: np.ndarray,
    sample: np.ndarray,
    camera: dict,
    path: Path,
) -> dict:
    nonzero = np.abs(values[sample]) > ZERO_TOL
    point_ids = sample[nonzero]
    line_ids = sample[nonzero & unique[sample]]
    length = LINE_PIXELS * 2 * camera["parallel_scale"] / WINDOW[1]
    glyphs = line_mesh(
        centers[line_ids], axes[line_ids], values[line_ids], ids[line_ids], length
    )
    glyphs.save(path.with_suffix(".vtp"), binary=True)
    point_cloud = pv.PolyData(centers[point_ids])
    point_cloud.point_data["ColorRGB"] = np.round(
        CMAP(NORM(values[point_ids]))[:, :3] * 255
    ).astype(np.uint8)
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="three lights")
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(skin, color="#9ca3a3", opacity=0.28, smooth_shading=False)
    plotter.add_mesh(
        glyphs,
        scalars="ColorRGB",
        rgb=True,
        preference="cell",
        lighting=False,
        line_width=3.5,
        show_scalar_bar=False,
        render_lines_as_tubes=False,
    )
    plotter.add_mesh(
        point_cloud,
        scalars="ColorRGB",
        rgb=True,
        lighting=False,
        point_size=3.6,
        show_scalar_bar=False,
        render_points_as_spheres=False,
    )
    set_parallel_camera(plotter, camera)
    plotter.screenshot(path)
    plotter.close()
    return {
        "common_sample_count": len(sample),
        "nonzero_dots": len(point_ids),
        "direction_lines": len(line_ids),
        "degenerate_direction_dots_only": len(point_ids) - len(line_ids),
        "neutral_omitted": len(sample) - len(point_ids),
        "glyph_length_m": length,
        "line_mesh": record(path.with_suffix(".vtp")),
        "raw_image": record(path),
    }


def add_colorbar(figure: plt.Figure, axes: plt.Axes | np.ndarray) -> None:
    bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=NORM, cmap=CMAP),
        ax=axes,
        orientation="horizontal",
        fraction=0.047,
        pad=0.025,
        aspect=55,
    )
    bar.set_ticks(TICKS, labels=TICK_LABELS)
    bar.ax.tick_params(labelsize=10)
    bar.set_label(
        "Signed effective activation z   |   blue: expansion   gray: near zero   red: contraction\n"
        "Shared symmetric-log scale; same color means the same value in every panel",
        fontsize=11,
    )


def labeled_figures(raw_paths: list[Path], output: Path, view: str) -> list[Path]:
    assets = []
    for index, (identifier, title, subtitle) in enumerate(LABELS):
        fig, ax = plt.subplots(figsize=(7.5, 8.7), layout="constrained")
        fig.patch.set_facecolor(BACKGROUND)
        ax.imshow(plt.imread(raw_paths[index]))
        ax.set_axis_off()
        ax.set_title(f"{title}\n{subtitle}", fontsize=18, pad=10)
        add_colorbar(fig, ax)
        fig.suptitle("Full fitted activation | reference geometry", fontsize=12)
        path = output / f"{view}-{identifier}.png"
        fig.savefig(path, dpi=190, facecolor=BACKGROUND)
        plt.close(fig)
        assets.append(path)
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    for ax, raw, (_, title, subtitle) in zip(axes, raw_paths, LABELS, strict=True):
        ax.imshow(plt.imread(raw))
        ax.set_axis_off()
        ax.set_title(f"{title}\n{subtitle}", fontsize=19, pad=9)
    add_colorbar(fig, axes)
    fig.suptitle(
        "One full activation tensor, split into three orthogonal modes\n"
        "Same reference shape and sampled tetrahedra | line direction = axis; color = signed strength",
        fontsize=18,
    )
    path = output / f"{view}-triptych.png"
    fig.savefig(path, dpi=190, facecolor=BACKGROUND)
    plt.close(fig)
    return [*assets, path]


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    (out / "raw").mkdir()
    (out / "sources").mkdir()
    sources = {}
    for name in [
        Path(__file__).name,
        "muscle_glyph_context.py",
        "experiment_profile.py",
    ]:
        source = Path(__file__).parent / name
        saved = out / "sources" / name
        shutil.copyfile(source, saved)
        sources[name] = {"source": record(source), "snapshot": record(saved)}
    assert (
        record(cfg.source)["sha256"]
        == "07efff9f6a96ff7d4556df723f6f6386c31c111ad89ac65ebff21ded82050201"
    )
    with np.load(cfg.source, allow_pickle=False) as saved:
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        ids, rest, b, z = (
            saved["active_ids"],
            saved["rest_points"],
            saved["B"],
            saved["Z"],
        )
    assert np.max(np.abs(b @ b.swapaxes(-1, -2) - np.eye(3) - z)) < 2e-13
    values, axes = np.linalg.eigh(z)
    values, axes = values[:, ::-1].copy(), axes[:, :, ::-1].copy()
    assert np.all(values[:, 0] >= -ZERO_TOL)
    assert np.all(values[:, 2] <= ZERO_TOL)
    assert np.max(np.abs(axes @ np.swapaxes(axes, -1, -2) - np.eye(3))) < 3e-15
    reconstruction = np.einsum("nik,nk,njk->nij", axes, values, axes)
    error = float(np.max(np.abs(reconstruction - z)))
    assert error < 2e-13
    gap = (values[:, :-1] - values[:, 1:]) / np.maximum(
        1, np.max(np.abs(values), axis=1, keepdims=True)
    )
    unique = np.c_[
        gap[:, 0] > GAP_TOL, np.min(gap, axis=1) > GAP_TOL, gap[:, 1] > GAP_TOL
    ]
    mesh = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    assert np.array_equal(mesh.points, rest)
    assert np.array_equal(np.flatnonzero(mesh.cell_data["ActivationMask"]), ids)
    tets = mesh.cells.reshape(-1, 5)[:, 1:]
    centers = rest[tets[ids]].mean(axis=1)
    control_ids = np.asarray(mesh.cell_data["ActivationControlId"])[ids]
    volume = (
        np.asarray(mesh.cell_data["Volume"])[ids]
        * np.asarray(mesh.cell_data["MuscleFraction"])[ids]
    )
    mode_norm2 = np.sum(values**2, axis=0)
    weighted_norm2 = np.sum(volume[:, None] * values**2, axis=0)
    np.savez_compressed(
        out / "eigenmodes.npz",
        global_cell_ids=ids,
        centers_rest=centers,
        eigenvalues_descending=values,
        reference_axes_columns=axes,
        axis_unique=unique,
        muscle_volume_weights=volume,
    )
    cloud = pv.PolyData(centers)
    cloud.point_data["GlobalCellId"] = ids
    cloud.point_data["MuscleId"] = np.asarray(mesh.cell_data["MuscleId"])[ids]
    cloud.point_data["ResidualFrobeniusNorm"] = np.sqrt(
        values[:, 1] ** 2 + values[:, 2] ** 2
    )
    for j in range(3):
        cloud.point_data[f"Mode{j + 1}Eigenvalue"] = values[:, j]
        cloud.point_data[f"Mode{j + 1}ReferenceAxis"] = axes[:, :, j]
    cloud.save(out / "all-active-cell-modes.vtp", binary=True)
    context = build_muscle_region_context(mesh)
    cameras = json.loads(CAMERAS.read_text())
    views = {}
    LOG.info("Loaded %d active tetrahedra; reconstruction error %.3g", len(ids), error)
    for view in cameras["views"]:
        name, camera = view["id"], view["camera"]
        if name not in ["side-context", "region1-mouth-corner"]:
            continue
        visibility = visible_region_mask(
            context, centers, ids, control_ids, camera, window_size=WINDOW
        )
        save_region_visibility(visibility, out / f"{name}-visibility.npz")
        sample = choose_sample(visibility, centers, camera)
        np.savez_compressed(
            out / f"{name}-sample.npz",
            active_array_indices=sample,
            global_cell_ids=ids[sample],
            centers=centers[sample],
        )
        raw_paths, modes = [], []
        for j, (identifier, _, _) in enumerate(LABELS):
            path = out / "raw" / f"{name}-{identifier}.png"
            modes.append(
                raw_panel(
                    skin,
                    centers,
                    axes[:, :, j],
                    values[:, j],
                    unique[:, j],
                    ids,
                    sample,
                    camera,
                    path,
                )
            )
            raw_paths.append(path)
        plots = labeled_figures(raw_paths, out, name)
        views[name] = {
            "camera": camera,
            "visibility_candidates": visibility.retained_count,
            "sample_count": len(sample),
            "modes": modes,
            "figures": [record(p) for p in plots],
        }
        LOG.info("Rendered %s with %d shared sample locations", name, len(sample))
    metrics = []
    for j in range(3):
        metrics.append(
            {
                "mode": j + 1,
                "positive_cells": int(np.sum(values[:, j] > ZERO_TOL)),
                "negative_cells": int(np.sum(values[:, j] < -ZERO_TOL)),
                "neutral_cells": int(np.sum(np.abs(values[:, j]) <= ZERO_TOL)),
                "nonunique_axes": int(np.sum(~unique[:, j])),
                "eigenvalue_percentiles_0_1_10_50_90_99_100": np.percentile(
                    values[:, j], [0, 1, 10, 50, 90, 99, 100]
                ).tolist(),
                "squared_frobenius_share_unweighted": float(
                    mode_norm2[j] / mode_norm2.sum()
                ),
                "squared_frobenius_share_muscle_volume_weighted": float(
                    weighted_norm2[j] / weighted_norm2.sum()
                ),
            }
        )
    summary = {
        "status": "completed",
        "source": record(cfg.source),
        "inputs": [
            record(FIXTURE / "volume.vtu"),
            record(FIXTURE / "skin.vtp"),
            record(CAMERAS),
        ],
        "sources": sources,
        "active_cell_count": len(ids),
        "mode_statistics": metrics,
        "Z_reconstruction_max_abs_error": error,
        "views": views,
        "encoding": {
            "frame": "reference/rest",
            "ordering": "z1>=z2>=z3, largest algebraic first",
            "residual": "Z_res=z2*n2*n2T+z3*n3*n3T",
            "color": "common SymLogNorm(-40,40), linthresh .01, linscale .5, base 10; no clipping in data range",
            "direction": "centered unoriented constant-length lines; projected length can foreshorten",
            "sampling": f"front-visible muscle region, then frontmost centroid in each {PITCH}px grid bin; independent of activation",
            "neutral_tolerance": ZERO_TOL,
            "relative_eigenvalue_gap_tolerance": GAP_TOL,
            "dots": "nonzero mode centers; repeated eigenvalues retain dot but omit nonunique axis",
            "length": f"{LINE_PIXELS}px equivalent before directional foreshortening; no amplitude encoded in length",
        },
        "limitations": [
            "Spectral decomposition of effective activation Z, not displacement, measured strain, or anatomy.",
            "Squared-norm shares are not energy or fit-contribution shares.",
            "Display samples only the camera-facing portion; statistics and NPZ/VTP cover all active cells.",
        ],
        "runtime": {
            "software_rendering": os.environ.get("LIBGL_ALWAYS_SOFTWARE"),
            "pyvista": pv.__version__,
        },
    }
    write_json(out / "summary.json", summary)
    cherries.log_metrics(
        {
            "active_cells": len(ids),
            "Z_reconstruction_max_abs_error": error,
            "residual_squared_norm_fraction": float(
                mode_norm2[1:].sum() / mode_norm2.sum()
            ),
        }
    )
    LOG.info("Completed three-mode visualization: %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
