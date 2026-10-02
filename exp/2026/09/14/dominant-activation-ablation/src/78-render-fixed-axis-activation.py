"""Add the saved fixed-axis scalar refit to the signed activation figures."""

# ruff: noqa: PLR0915

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
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
WINDOW = (1800, 1800)
BACKGROUND = "#f4f2ed"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
ZERO_TOL = 1e-8
GAP_TOL = 1e-6


class Config(cherries.BaseConfig):
    refit_dir: Path = cherries.input("42-fixed-directions-400")
    verification: Path = cherries.input("62-verification-400/receipt.json")
    full_figures: Path = cherries.input("77-signed-meeting-eigenmodes")
    output_dir: Path = cherries.output("78-fixed-axis-activation", mkdir=True)


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def checked_record(expected: dict) -> dict:
    actual = record(Path(expected["path"]))
    assert actual == expected
    return actual


def render_panel(
    skin: pv.PolyData,
    shown: pv.UnstructuredGrid,
    camera: dict,
    palette: list[str],
    path: Path,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="three lights")
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(skin, color="#8d969b", opacity=0.06, smooth_shading=False)
    plotter.add_mesh(
        shown,
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
    actor = plotter.add_text(
        "Corrected fixed-axis refit - update 400 - saved deformed shape\n"
        "One nonnegative contraction strength per frozen reference axis\n"
        "Red: contraction-like; blue: extension-like; 100% magnitude = 4.5 mm",
        position="upper_left",
        color="black",
        font_size=12,
    )
    actor.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    actor.GetTextProperty().SetBackgroundOpacity(0.90)
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


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    (out / "sources").mkdir()
    sources = []
    for source in [
        Path(__file__),
        Path(__file__).with_name("experiment_profile.py"),
        Path(__file__).with_name("muscle_glyph_context.py"),
    ]:
        destination = out / "sources" / source.name
        shutil.copyfile(source, destination)
        sources.append({"source": record(source), "snapshot": record(destination)})
    verified = json.loads(cfg.verification.read_text())
    assert verified["status"] == "passed"
    inputs = [record(cfg.verification)]
    for key in ["best", "initialization", "summary", "protocol"]:
        inputs.append(checked_record(verified["inputs"][key]))
        assert Path(verified["inputs"][key]["path"]).parent == cfg.refit_dir
    with np.load(cfg.refit_dir / "best.npz", allow_pickle=False) as best:
        s, u, ids = best["s"], best["u"], best["active_ids"]
        assert best["step"].item() == 400
        assert best["solver_valid"].item()
        assert best["physical_volume_energy"].item()
        axes_hash = best["axes_sha256"].item()
    with np.load(cfg.refit_dir / "initialization.npz", allow_pickle=False) as initial:
        assert np.array_equal(ids, initial["active_ids"])
        assert axes_hash == initial["axes_sha256"].item()
        axes, rest, original_eigenvalues = (
            initial["axes"],
            initial["rest_points"],
            initial["eigenvalues_Z0"],
        )
    assert np.all(s >= 0)
    assert hashlib.sha256(np.ascontiguousarray(axes).tobytes()).hexdigest() == axes_hash
    assert np.max(np.abs(np.linalg.norm(axes, axis=1) - 1)) < 2e-15
    z = 2 * s + s**2
    magnitude = s / (1 + s)
    magnitude_error = float(np.max(np.abs(magnitude - (1 - 1 / np.sqrt(1 + z)))))
    assert magnitude_error < 3e-16
    gap = (original_eigenvalues[:, -1] - original_eigenvalues[:, -2]) / np.maximum(
        1, np.max(np.abs(original_eigenvalues), axis=1)
    )
    original_axis_unique = gap > GAP_TOL
    lengths = 0.0045 * magnitude * original_axis_unique * (z > ZERO_TOL)
    protocol = json.loads((cfg.refit_dir / "protocol.json").read_text())
    for name in ["volume.vtu", "skin.vtp"]:
        inputs.append(checked_record(protocol["inputs"][name]))
        assert Path(protocol["inputs"][name]["path"]) == FIXTURE / name
    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    assert np.array_equal(volume.points, rest)
    assert np.array_equal(np.flatnonzero(volume.cell_data["ActivationMask"]), ids)
    tets = volume.cells.reshape(-1, 5)[:, 1:][ids]
    rest_tets, def_tets = rest[tets], (rest + u)[tets]
    f = (def_tets[:, 1:] - def_tets[:, :1]).swapaxes(1, 2) @ np.linalg.inv(
        (rest_tets[:, 1:] - rest_tets[:, :1]).swapaxes(1, 2)
    )
    transported = np.einsum("nij,nj->ni", f, axes)
    norms = np.linalg.norm(transported, axis=1)
    assert np.all(norms > 0)
    assert np.all(np.isfinite(norms))
    spatial_axes = transported / norms[:, None]
    centers = def_tets.mean(axis=1)
    control_ids = np.asarray(volume.cell_data["ActivationControlId"])[ids]
    half = 0.5 * lengths[:, None] * spatial_axes
    points = np.empty((2 * len(ids), 3))
    points[0::2], points[1::2] = centers - half, centers + half
    index = np.arange(len(ids))
    glyph = pv.PolyData(
        points, lines=np.c_[np.full(len(ids), 2), 2 * index, 2 * index + 1].ravel()
    )
    for key, value in {
        "GlobalCellId": ids,
        "ActivationControlId": control_ids,
        "ScalarStrengthS": s,
        "Z_eigenvalue": z,
        "ReferenceAxis": axes,
        "SpatialAxis": spatial_axes,
        "ReferenceAxisUnique": original_axis_unique.astype(np.uint8),
        "SignedDisplayMagnitudePercent": 100 * magnitude,
        "DisplayLengthM": lengths,
    }.items():
        glyph.cell_data[key] = value
    glyph.field_data["State"] = np.asarray(
        ["fixed reference axes, refitted scalar strengths, update 400"]
    )
    glyph.field_data["TransverseActivation"] = np.asarray(
        ["two transverse effective eigenvalues are zero by construction"]
    )
    glyph.save(out / "all-active-fixed-axis-refit.vtp", binary=True)
    np.savez_compressed(
        out / "fixed-axis-field.npz",
        global_cell_ids=ids,
        scalar_strength=s,
        reference_axes=axes,
        spatial_axes=spatial_axes,
        deformation_gradient=f,
        centers_deformed=centers,
        eigenvalues_descending=np.c_[z, np.zeros((len(ids), 2))],
        reference_axis_unique=original_axis_unique,
        original_relative_top_gap=gap,
        display_magnitude=magnitude,
        display_length_m=lengths,
    )
    volume.points = rest + u
    skin.points = (rest + u)[np.asarray(skin.point_data["GlobalPointId"])]
    context = build_muscle_region_context(volume)
    full_summary = json.loads((cfg.full_figures / "summary.json").read_text())
    inputs.append(record(cfg.full_figures / "summary.json"))
    views = {}
    for name, full_view in full_summary["views"].items():
        camera = full_view["camera"]
        visibility = visible_region_mask(
            context, centers, ids, control_ids, camera, window_size=WINDOW
        )
        save_region_visibility(visibility, out / f"{name}-visibility.npz")
        shown = glyph.extract_cells(visibility.mask & (lengths > 0))
        refit_path = out / f"{name}-fixed-axis-refit.png"
        render_panel(
            skin, shown, camera, full_summary["encoding"]["color_stops"], refit_path
        )
        original_paths = []
        for mode in full_view["modes"]:
            inputs.append(checked_record(mode["image"]))
            original_paths.append(Path(mode["image"]["path"]))
        pair_path, four_path = (
            out / f"{name}-principal-vs-refit.png",
            out / f"{name}-four-panel.png",
        )
        row([original_paths[0], refit_path], pair_path)
        row([*original_paths, refit_path], four_path)
        views[name] = {
            "camera": camera,
            "refit_visibility_candidates": visibility.retained_count,
            "shown_refit_lines": shown.n_cells,
            "fixed_axis_refit": record(refit_path),
            "principal_vs_refit": record(pair_path),
            "four_panel": record(four_path),
        }
        LOG.info("Rendered %s: %d refit lines", name, shown.n_cells)
    refit_summary = json.loads((cfg.refit_dir / "summary.json").read_text())
    summary = {
        "status": "completed",
        "inputs": inputs,
        "sources": sources,
        "views": views,
        "field_npz": record(out / "fixed-axis-field.npz"),
        "glyph_vtp": record(out / "all-active-fixed-axis-refit.vtp"),
        "state": {
            "step": 400,
            "axes_sha256": axes_hash,
            "active_cells": len(ids),
            "zero_strength_cells": int(np.sum(s == 0)),
            "z_range": [float(z.min()), float(z.max())],
            "signed_display_range": [
                float(100 * magnitude.min()),
                float(100 * magnitude.max()),
            ],
            "nonunique_source_axes": int(np.sum(~original_axis_unique)),
            "zero_length_cells": int(np.sum(lengths == 0)),
            "magnitude_formula_max_abs_error": magnitude_error,
            "metrics": refit_summary["best_metrics"],
        },
        "encoding": {
            "tensor": "B=I+s*n*nT; s>=0; reference n frozen; Z=(2s+s^2)*n*nT",
            "signed_display": "100*s/(1+s)=100*(1-1/sqrt(1+z)), nonnegative",
            "color_stops": full_summary["encoding"]["color_stops"],
            "color_limits": [-100, 100],
            "length": "4.5 mm*s/(1+s); zero for neutral z or nonunique original axes",
            "geometry": "each state shown on its own saved equilibrium shape; normalize(F_state @ n)",
            "visibility": "same rule and camera, recomputed on refit deformed muscle regions",
            "neutral_tolerance": ZERO_TOL,
            "source_axis_gap_tolerance": GAP_TOL,
        },
        "limitations": [
            "Projected axes may look different because each state has different F; reference axes are frozen.",
            "The refit has no transverse activation; passive transverse deformation is still allowed.",
            "Prescribed active shortening is not total deformation strain.",
            "Update 400 is a finite-budget endpoint, not a stationarity or anatomy certificate.",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    cherries.log_metrics(
        {
            "refit_update": 400,
            "active_cells": len(ids),
            "max_display_percent": float(100 * magnitude.max()),
        }
    )
    LOG.info("Completed fixed-axis activation visualization: %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
