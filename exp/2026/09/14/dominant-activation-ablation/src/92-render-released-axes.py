"""Render the saved released-axis refit and its optimization history on CPU."""

# ruff: noqa: C901, PLR0915

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from muscle_glyph_context import (
    build_muscle_region_context,
    save_region_visibility,
    set_parallel_camera,
    visible_region_mask,
)
from PIL import Image

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]

LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
WINDOW = (1800, 1800)
BACKGROUND = "#f4f2ed"
SHAPE_COLOR = "#8d969b"
MAX_LENGTH = 0.0045
ZERO_TOL = 1e-8
STATE_FILES = (
    "initial.npz",
    "initial-converted.npz",
    "step-0128.npz",
    "step-0200.npz",
    "best.npz",
    "last.npz",
)


class Config(cherries.BaseConfig):
    released_dir: Path = cherries.input("90-released-axes")
    fixed_dir: Path = cherries.input("42-fixed-directions-400")
    full_state: Path = cherries.input("10-forward/baseline-replay.npz")
    signed_figures: Path = cherries.input("77-signed-meeting-eigenmodes")
    fixed_figures: Path = cherries.input("78-fixed-axis-activation")
    output_dir: Path = cherries.output("92-released-axes-visualization", mkdir=True)


def record(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def load_state(path: Path) -> dict[str, Any]:
    required = {
        "v",
        "u",
        "active_ids",
        "rest_points",
        "step",
        "source_fixed_step",
        "Z",
        "C",
        "B",
    }
    with np.load(path, allow_pickle=False) as saved:
        assert required.issubset(saved.files), (path, required - set(saved.files))
        state = {key: np.asarray(saved[key]).copy() for key in required}
        for flag in ("solver_valid", "physical_volume_energy"):
            if flag in saved:
                assert bool(saved[flag]), (path, flag)
        flags = {
            flag: bool(saved[flag])
            for flag in (
                "solver_valid",
                "physical_volume_energy",
                "physical_noninverted",
            )
            if flag in saved
        }
    v = np.asarray(state["v"], dtype=np.float64)
    u = np.asarray(state["u"], dtype=np.float64)
    ids = np.asarray(state["active_ids"], dtype=np.int64)
    rest = np.asarray(state["rest_points"], dtype=np.float64)
    assert v.shape == (len(ids), 3)
    assert u.shape == rest.shape
    assert rest.ndim == 2
    assert rest.shape[1] == 3
    assert all(
        np.asarray(state[key]).shape == (len(ids), 3, 3) for key in ("Z", "C", "B")
    )
    assert np.isfinite(v).all()
    assert np.isfinite(u).all()
    assert int(state["source_fixed_step"]) == 400
    c = v[:, :, None] * v[:, None, :]
    b = np.eye(3) + c
    z = b @ b.swapaxes(-1, -2) - np.eye(3)
    c_error = float(np.max(np.abs(state["C"] - c)))
    b_error = float(np.max(np.abs(state["B"] - b)))
    z_delta = np.abs(state["Z"] - z)
    z_error = float(np.max(z_delta))
    z_scaled_error = float(np.max(z_delta / np.maximum(1.0, np.abs(z))))
    assert c_error < 2e-13
    assert b_error < 2e-13
    assert np.allclose(state["Z"], z, rtol=2e-14, atol=3e-13)
    return {
        **state,
        "v": v,
        "u": u,
        "active_ids": ids,
        "rest_points": rest,
        "C": c,
        "B": b,
        "Z": z,
        "step": int(state["step"]),
        "source_fixed_step": int(state["source_fixed_step"]),
        "tensor_validation": {
            "C_max_abs_error": c_error,
            "B_max_abs_error": b_error,
            "Z_max_abs_error": z_error,
            "Z_max_scaled_error": z_scaled_error,
            "Z_scaled_error_denominator": "max(1, abs(reconstructed_Z)) elementwise",
            "Z_allclose_rtol": 2e-14,
            "Z_allclose_atol": 3e-13,
        },
        **flags,
    }


def make_skin(template: pv.PolyData, rest: np.ndarray, u: np.ndarray) -> pv.PolyData:
    mesh = template.copy(deep=True)
    point_ids = np.asarray(mesh.point_data["GlobalPointId"], dtype=np.int64)
    mesh.points = (rest + u)[point_ids]
    return mesh


def render_shape(
    skin: pv.PolyData,
    camera: dict[str, Any],
    caption: str,
    color: str,
    path: Path,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="three lights")
    plotter.set_background(BACKGROUND)
    actor = plotter.add_mesh(
        skin,
        color=color,
        opacity=1.0,
        smooth_shading=False,
        show_edges=False,
    )
    assert actor.GetProperty().GetInterpolation() == 0
    label = plotter.add_text(
        caption, position="upper_left", color="black", font_size=14
    )
    label.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    label.GetTextProperty().SetBackgroundOpacity(0.90)
    set_parallel_camera(plotter, camera)
    plotter.screenshot(path)
    plotter.close()


def render_glyphs(
    skin: pv.PolyData,
    glyphs: pv.UnstructuredGrid,
    camera: dict[str, Any],
    palette: list[str],
    step: int,
    inversions: int,
    path: Path,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="three lights")
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(skin, color="#8d969b", opacity=0.06, smooth_shading=False)
    plotter.add_mesh(
        glyphs,
        scalars="SignedDisplayMagnitudePercent",
        cmap=LinearSegmentedColormap.from_list("signed_activation", palette),
        clim=(-100, 100),
        lighting=False,
        line_width=1.0,
        render_lines_as_tubes=False,
        scalar_bar_args={
            "title": "Signed display\nmagnitude (%)",
            "color": "black",
            "title_font_size": 17,
            "label_font_size": 14,
            "n_labels": 5,
            "vertical": True,
            "background_color": BACKGROUND,
            "fill": True,
            "position_x": 0.80,
            "position_y": 0.10,
            "width": 0.08,
            "height": 0.45,
        },
    )
    warning = (
        f"\nWARNING: saved state contains {inversions} inverted tetrahedra"
        if inversions
        else "\nSaved state has zero inverted tetrahedra"
    )
    label = plotter.add_text(
        f"Released-axis refit - best update {step} - saved deformed shape\n"
        "One nonnegative rank-one contraction per active tetrahedron\n"
        "Red: contraction-like; blue is unused; 100% magnitude = 4.5 mm"
        f"{warning}",
        position="upper_left",
        color="black",
        font_size=12,
    )
    label.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    label.GetTextProperty().SetBackgroundOpacity(0.90)
    set_parallel_camera(plotter, camera)
    plotter.screenshot(path)
    plotter.close()


def row(paths: list[Path], output: Path) -> None:
    canvas = Image.new("RGB", (len(paths) * WINDOW[0], WINDOW[1]), BACKGROUND)
    for index, path in enumerate(paths):
        with Image.open(path) as panel:
            assert panel.size == WINDOW
            canvas.paste(panel.convert("RGB"), (index * WINDOW[0], 0))
    canvas.save(output)


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert "step" in rows[0]
    result = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }
    assert np.all(np.diff(result["step"]) > 0)
    assert all(np.isfinite(values).all() for values in result.values())
    return result


def render_history(trace: dict[str, np.ndarray], path: Path) -> dict[str, list[str]]:
    fit = ["uniform_fit_rms_mm", "area_weighted_fit_rms_mm"]
    roughness = [
        "primary_union_normal_displacement_highpass_5mm_rms_mm",
        "primary_union_normal_residual_highpass_5mm_rms_mm",
    ]
    inversions = [
        key
        for key in (
            "inverted_all_cells",
            "inverted_active_cells",
            "inverted_pure_muscle_cells",
        )
        if key in trace
    ]
    drift = [
        "direction_drift_p50_deg",
        "direction_drift_p90_deg",
        "direction_drift_p99_deg",
        "direction_drift_volume_weighted_mean_deg",
    ]
    assert all(key in trace for key in fit)
    assert all(key in trace for key in roughness)
    assert inversions
    assert all(key in trace for key in drift)
    groups = {
        "fit": fit,
        "roughness": roughness,
        "inversions": inversions,
        "drift": drift,
    }
    labels = {
        "uniform_fit_rms_mm": "uniform fit RMS",
        "area_weighted_fit_rms_mm": "area-weighted fit RMS",
        "primary_union_normal_displacement_highpass_5mm_rms_mm": (
            "predicted surface detail (5 mm high-pass RMS)"
        ),
        "primary_union_normal_residual_highpass_5mm_rms_mm": (
            "target residual detail (5 mm high-pass RMS)"
        ),
        "inverted_all_cells": "all inverted cells",
        "inverted_active_cells": "active inverted cells",
        "inverted_pure_muscle_cells": "pure-muscle inverted cells",
        "direction_drift_p50_deg": "median",
        "direction_drift_p90_deg": "p90",
        "direction_drift_p99_deg": "p99",
        "direction_drift_volume_weighted_mean_deg": "volume-weighted mean",
    }
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 14), sharex=True, constrained_layout=True
    )
    for axis, (title, keys) in zip(axes, groups.items(), strict=True):
        for key in keys:
            axis.plot(trace["step"], trace[key], linewidth=1.7, label=labels[key])
        axis.set_title(title.capitalize())
        axis.set_ylabel(
            "count"
            if title == "inversions"
            else "degrees"
            if title == "drift"
            else "mm"
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("released-axis optimizer update")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return groups


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    source_dir = out / "sources"
    source_dir.mkdir()
    sources = []
    for source in (
        Path(__file__),
        Path(__file__).with_name("muscle_glyph_context.py"),
        Path(__file__).with_name("experiment_profile.py"),
    ):
        snapshot = source_dir / source.name
        shutil.copyfile(source, snapshot)
        sources.append({"source": record(source), "snapshot": record(snapshot)})

    summary_path = cfg.released_dir / "summary.json"
    trace_path = cfg.released_dir / "trace.csv"
    released_summary = json.loads(summary_path.read_text())
    assert released_summary["status"].startswith("completed")
    inputs = [record(summary_path), record(trace_path)]
    states = {}
    for name in STATE_FILES:
        path = cfg.released_dir / name
        states[name] = load_state(path)
        inputs.append(record(path))
    initial = states["initial.npz"]
    converted = states["initial-converted.npz"]
    best = states["best.npz"]
    last = states["last.npz"]
    for state in states.values():
        assert np.array_equal(state["active_ids"], initial["active_ids"])
        assert np.array_equal(state["rest_points"], initial["rest_points"])
    assert initial["step"] == 0
    assert converted["step"] == 0
    assert states["step-0128.npz"]["step"] == 128
    assert states["step-0200.npz"]["step"] == 200
    assert last["step"] == 200
    assert best["step"] == int(released_summary["best_step"])
    assert "physical_noninverted" in best

    with np.load(cfg.fixed_dir / "best.npz", allow_pickle=False) as saved:
        fixed_s = np.asarray(saved["s"], dtype=np.float64)
        fixed_u = np.asarray(saved["u"], dtype=np.float64)
        fixed_ids = np.asarray(saved["active_ids"], dtype=np.int64)
        assert int(saved["step"]) == 400
    with np.load(cfg.fixed_dir / "initialization.npz", allow_pickle=False) as saved:
        fixed_axes = np.asarray(saved["axes"], dtype=np.float64)
        fixed_rest = np.asarray(saved["rest_points"], dtype=np.float64)
    expected_initial_v = np.sqrt(fixed_s)[:, None] * fixed_axes
    assert np.array_equal(initial["active_ids"], fixed_ids)
    assert np.array_equal(initial["rest_points"], fixed_rest)
    assert np.array_equal(converted["u"], fixed_u)
    assert np.max(np.abs(initial["v"] - expected_initial_v)) < 2e-15
    assert np.array_equal(converted["v"], expected_initial_v)
    inputs.extend(
        [
            record(cfg.fixed_dir / "best.npz"),
            record(cfg.fixed_dir / "initialization.npz"),
        ]
    )

    with np.load(cfg.full_state, allow_pickle=False) as saved:
        full_rest = np.asarray(saved["rest_points"], dtype=np.float64)
        full_u = np.asarray(saved["u"], dtype=np.float64)
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
    assert np.array_equal(full_rest, initial["rest_points"])
    inputs.append(record(cfg.full_state))

    supplemental_states = {}
    for name in ("best-noninverted.npz", "first-inverted.npz"):
        path = cfg.released_dir / name
        if path.exists():
            supplemental_states[name] = load_state(path)
            inputs.append(record(path))
    displayed = best
    displayed_name = "best.npz"
    best_inversions = int(released_summary["best_metrics"]["inverted_all_cells"])
    assert best["physical_noninverted"] == (best_inversions == 0)
    if best_inversions:
        assert "best-noninverted.npz" in supplemental_states
        assert supplemental_states["best-noninverted.npz"]["physical_noninverted"]

    volume = pv.read(FIXTURE / "volume.vtu")
    skin_template = pv.read(FIXTURE / "skin.vtp")
    ids = initial["active_ids"]
    rest = initial["rest_points"]
    assert np.array_equal(volume.points, rest)
    assert np.array_equal(np.flatnonzero(volume.cell_data["ActivationMask"]), ids)
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:][ids]
    rest_tets = rest[tets]
    deformed_tets = (rest + displayed["u"])[tets]
    f = (deformed_tets[:, 1:] - deformed_tets[:, :1]).swapaxes(1, 2) @ np.linalg.inv(
        (rest_tets[:, 1:] - rest_tets[:, :1]).swapaxes(1, 2)
    )
    strength = np.sum(displayed["v"] ** 2, axis=1)
    z_scalar = 2 * strength + strength**2
    nonzero = z_scalar > ZERO_TOL
    reference_axes = np.zeros_like(displayed["v"])
    reference_axes[nonzero] = displayed["v"][nonzero] / np.sqrt(strength[nonzero, None])
    transported = np.einsum("nij,nj->ni", f[nonzero], reference_axes[nonzero])
    spatial_axes = np.zeros_like(reference_axes)
    spatial_axes[nonzero] = transported / np.linalg.norm(transported, axis=1)[:, None]
    centers = deformed_tets.mean(axis=1)
    magnitude = strength / (1 + strength)
    lengths = MAX_LENGTH * magnitude * nonzero
    half = 0.5 * lengths[:, None] * spatial_axes
    points = np.empty((2 * len(ids), 3), dtype=np.float64)
    points[0::2], points[1::2] = centers - half, centers + half
    index = np.arange(len(ids), dtype=np.int64)
    glyph = pv.PolyData(
        points, lines=np.c_[np.full(len(ids), 2), 2 * index, 2 * index + 1].ravel()
    )
    control_ids = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)[
        ids
    ]
    for key, value in {
        "GlobalCellId": ids,
        "ActivationControlId": control_ids,
        "VectorControlV": displayed["v"],
        "ScalarStrengthS": strength,
        "Z_eigenvalue": z_scalar,
        "ReferenceAxis": reference_axes,
        "SpatialAxis": spatial_axes,
        "SignedDisplayMagnitudePercent": 100 * magnitude,
        "DisplayLengthM": lengths,
    }.items():
        glyph.cell_data[key] = value
    glyph.field_data["State"] = np.asarray(
        [f"released axes, {displayed_name}, update {displayed['step']}"]
    )
    glyph.save(out / "all-active-released-best.vtp", binary=True)
    np.savez_compressed(
        out / "released-best-field.npz",
        global_cell_ids=ids,
        vector_control_v=displayed["v"],
        scalar_strength=strength,
        reference_axes=reference_axes,
        spatial_axes=spatial_axes,
        deformation_gradient=f,
        centers_deformed=centers,
        eigenvalues_descending=np.c_[z_scalar, np.zeros((len(ids), 2))],
        display_magnitude=magnitude,
        display_length_m=lengths,
    )

    signed_summary_path = cfg.signed_figures / "summary.json"
    fixed_summary_path = cfg.fixed_figures / "summary.json"
    signed_summary = json.loads(signed_summary_path.read_text())
    fixed_summary = json.loads(fixed_summary_path.read_text())
    inputs.extend([record(signed_summary_path), record(fixed_summary_path)])
    palette = signed_summary["encoding"]["color_stops"]
    released_skin = make_skin(skin_template, rest, displayed["u"])
    fixed_skin = make_skin(skin_template, fixed_rest, fixed_u)
    full_skin = make_skin(skin_template, full_rest, full_u)
    volume.points = rest + displayed["u"]
    context = build_muscle_region_context(volume)
    views = {}
    for name, signed_view in signed_summary["views"].items():
        camera = signed_view["camera"]
        visibility = visible_region_mask(
            context, centers, ids, control_ids, camera, window_size=WINDOW
        )
        visibility_path = out / f"{name}-visibility.npz"
        save_region_visibility(visibility, visibility_path)
        shown = glyph.extract_cells(visibility.mask & nonzero)
        glyph_path = out / f"{name}-released-best.png"
        render_glyphs(
            released_skin,
            shown,
            camera,
            palette,
            displayed["step"],
            best_inversions,
            glyph_path,
        )
        fixed_glyph = Path(fixed_summary["views"][name]["fixed_axis_refit"]["path"])
        assert record(fixed_glyph) == fixed_summary["views"][name]["fixed_axis_refit"]
        inputs.append(record(fixed_glyph))
        glyph_pair = out / f"{name}-fixed-vs-released-glyphs.png"
        row([fixed_glyph, glyph_path], glyph_pair)

        shape_paths = []
        for identifier, mesh, caption, color in (
            ("full", full_skin, "Original corrected full fit", SHAPE_COLOR),
            (
                "fixed",
                fixed_skin,
                "Fixed-axis scalar refit - update 400",
                SHAPE_COLOR,
            ),
            (
                "released",
                released_skin,
                f"Released-axis best fit - update {displayed['step']} - "
                + (
                    f"WARNING: {best_inversions} inverted tetrahedra"
                    if best_inversions
                    else "zero inverted tetrahedra"
                ),
                SHAPE_COLOR,
            ),
        ):
            path = out / f"{name}-shape-{identifier}.png"
            render_shape(mesh, camera, caption, color, path)
            shape_paths.append(path)
        shape_comparison = out / f"{name}-shape-comparison.png"
        row(shape_paths, shape_comparison)
        noninverted_comparison = None
        best_noninverted = supplemental_states.get("best-noninverted.npz")
        if best_inversions and best_noninverted is not None:
            noninverted_skin = make_skin(skin_template, rest, best_noninverted["u"])
            noninverted_path = out / f"{name}-shape-best-noninverted.png"
            render_shape(
                noninverted_skin,
                camera,
                f"Best noninverted released-axis state - update {best_noninverted['step']}",
                SHAPE_COLOR,
                noninverted_path,
            )
            noninverted_comparison_path = (
                out / f"{name}-best-vs-best-noninverted-shapes.png"
            )
            row([shape_paths[-1], noninverted_path], noninverted_comparison_path)
            noninverted_comparison = {
                "best_noninverted_shape": record(noninverted_path),
                "best_vs_best_noninverted": record(noninverted_comparison_path),
            }
        views[name] = {
            "camera": camera,
            "visibility_candidates": visibility.retained_count,
            "shown_released_lines": shown.n_cells,
            "visibility": record(visibility_path),
            "released_glyphs": record(glyph_path),
            "fixed_vs_released_glyphs": record(glyph_pair),
            "shape_panels": [record(path) for path in shape_paths],
            "shape_comparison": record(shape_comparison),
            "inverted_best_comparison": noninverted_comparison,
        }

    trace = load_trace(trace_path)
    history_path = out / "optimization-history.png"
    history_columns = render_history(trace, history_path)
    zero_initial = np.linalg.norm(initial["v"], axis=1) == 0
    zero_best = np.linalg.norm(best["v"], axis=1) == 0
    summary = {
        "status": "completed",
        "inputs": inputs,
        "sources": sources,
        "states": {
            name: {
                "step": state["step"],
                "record": record(cfg.released_dir / name),
                "tensor_validation": state["tensor_validation"],
            }
            for name, state in {**states, **supplemental_states}.items()
        },
        "field_npz": record(out / "released-best-field.npz"),
        "glyph_vtp": record(out / "all-active-released-best.vtp"),
        "views": views,
        "history": {"image": record(history_path), "columns": history_columns},
        "displayed": {
            "source": displayed_name,
            "step": displayed["step"],
            "physical_noninverted": displayed.get("physical_noninverted"),
            "inverted_all_cells": best_inversions,
            "warning": None
            if best_inversions == 0
            else f"Primary best-fit state contains {best_inversions} inverted tetrahedra",
        },
        "best": {
            "step": best["step"],
            "physical_noninverted": best.get("physical_noninverted"),
            "active_cells": len(ids),
            "nonzero_lines": int(np.sum(nonzero)),
            "zero_initial_v_cells": int(np.sum(zero_initial)),
            "zero_initial_still_zero_cells": int(np.sum(zero_initial & zero_best)),
            "zero_initial_now_nonzero_cells": int(np.sum(zero_initial & ~zero_best)),
            "z_range": [float(z_scalar.min()), float(z_scalar.max())],
            "signed_display_range": [
                float((100 * magnitude).min()),
                float((100 * magnitude).max()),
            ],
            "line_length_m_range": [float(lengths.min()), float(lengths.max())],
        },
        "encoding": {
            "tensor": "C=v*vT; B=I+C; Z=B*BT-I; one nonnegative rank-one mode per active tetrahedron",
            "direction": "n=v/||v|| in reference frame; spatial glyph direction normalize(F_state*n)",
            "signed_display": "100*s/(1+s), s=||v||^2; nonnegative on shared [-100,100] palette",
            "length": "4.5 mm*s/(1+s); zero when 2*s+s^2<=1e-8",
            "shape": "opaque, flat-shaded saved equilibrium surfaces under identical frozen cameras",
        },
        "limitations": [
            "Exactly zero initial v has zero first derivative under C=v*vT and cannot release without perturbation.",
            "The sign of v and n is arbitrary; glyph lines encode an unoriented axis.",
            "Transported material axes are not spatial stress or strain eigenvectors.",
            "Each state uses its own deformed geometry, so spatial differences combine activation and deformation.",
            "The released-axis endpoint is a numerical fit, not an anatomy or stationarity certificate.",
        ],
    }
    summary_file = out / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    cherries.log_metrics(
        {
            "best_step": best["step"],
            "active_cells": len(ids),
            "nonzero_lines": int(np.sum(nonzero)),
            "zero_initial_now_nonzero_cells": int(np.sum(zero_initial & ~zero_best)),
        }
    )
    for path in (summary_file, history_path):
        cherries.log_output(path)
    LOG.info("Completed released-axis visualization: %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
