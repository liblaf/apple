"""Render the full corrected activation modes in the prior meeting's dense style."""

# ruff: noqa: PLR0915

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from muscle_glyph_context import (
    build_muscle_region_context,
    save_region_visibility,
    set_parallel_camera,
    visible_region_mask,
)
from PIL import Image

from liblaf import cherries

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[6]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
STYLE_SOURCE = (
    ROOT / "exp/2026/09/09/activation-space-smoothness/src/83-render-idea-activation.py"
)
STYLE_IMAGE = Path("/tmp/codex-clipboard-5d03b17c-2f8d-44f4-ba8b-d145fac3ddda.png")  # noqa: S108
WINDOW = (1800, 1800)
BACKGROUND = "#f4f2ed"
MAX_LENGTH = 0.0045
ZERO_TOL = 1e-8
GAP_TOL = 1e-6
LABELS = [
    ("mode-1-principal", "Principal mode - maximum contraction direction"),
    ("mode-2-residual", "Residual mode 2 - mixed sign (see sign view)"),
    ("mode-3-residual", "Residual mode 3 - extension-like effective mode"),
]


class Config(cherries.BaseConfig):
    source: Path = cherries.input("10-forward/baseline-replay.npz")
    output_dir: Path = cherries.output("74-meeting-eigenmodes", mkdir=True)


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def line_mesh(
    centers: np.ndarray,
    spatial_axes: np.ndarray,
    lengths: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> pv.PolyData:
    half = 0.5 * lengths[:, None] * spatial_axes
    points = np.empty((2 * len(centers), 3))
    points[0::2], points[1::2] = centers - half, centers + half
    indices = np.arange(len(centers))
    edges = np.c_[np.full(len(centers), 2), 2 * indices, 2 * indices + 1]
    mesh = pv.PolyData(points, lines=edges.ravel())
    for key, value in arrays.items():
        mesh.cell_data[key] = value
    mesh.field_data["CoordinateFrame"] = np.asarray(["saved deformed coordinates"])
    mesh.field_data["DirectionSemantics"] = np.asarray(
        ["unoriented reference eigenaxis carried by F and normalized"]
    )
    mesh.field_data["MagnitudeSemantics"] = np.asarray(
        ["1-1/sqrt(1+abs(z)); display magnitude, not signed physical strain"]
    )
    mesh.field_data["LengthSemantics"] = np.asarray(
        ["0.0045 m * display magnitude; neutral or nonunique axes have zero length"]
    )
    return mesh


def render_panel(
    skin: pv.PolyData,
    glyphs: pv.PolyData,
    mask: np.ndarray,
    camera: dict,
    title: str,
    path: Path,
    *,
    sign: bool = False,
) -> dict:
    shown = glyphs.extract_cells(mask & (glyphs.cell_data["DisplayLengthM"] > 0))
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="three lights")
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(skin, color="#8d969b", opacity=0.06, smooth_shading=False)
    style = {"lighting": False, "line_width": 1.0, "render_lines_as_tubes": False}
    if sign:
        plotter.add_mesh(
            shown, scalars="SignRGB", rgb=True, show_scalar_bar=False, **style
        )
        legend = plotter.add_text(
            "Red: positive z (contraction-like)\nBlue: negative z (extension-like)",
            position="lower_right",
            font_size=12,
            color="black",
        )
        legend.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
        legend.GetTextProperty().SetBackgroundOpacity(0.90)
    else:
        plotter.add_mesh(
            shown,
            scalars="DisplayMagnitudePercent",
            cmap="viridis",
            clim=(0, 100),
            scalar_bar_args={
                "title": "Display magnitude (%)",
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
            **style,
        )
    actor = plotter.add_text(
        "Corrected full fit - no regularization - saved deformed shape\n"
        f"{title}\n"
        "Dense visible-tetrahedron axes - 100% = 4.5 mm",
        position="upper_left",
        color="black",
        font_size=12,
    )
    actor.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    actor.GetTextProperty().SetBackgroundOpacity(0.90)
    set_parallel_camera(plotter, camera)
    plotter.screenshot(path)
    plotter.close()
    return {"shown_nonzero_unique_lines": shown.n_cells, "image": record(path)}


def triptych(paths: list[Path], output: Path) -> None:
    canvas = Image.new("RGB", (3 * WINDOW[0], WINDOW[1]), BACKGROUND)
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
    sources = {}
    for source in [
        Path(__file__),
        Path(__file__).with_name("muscle_glyph_context.py"),
        Path(__file__).with_name("experiment_profile.py"),
        STYLE_SOURCE,
    ]:
        destination = out / "sources" / source.name
        shutil.copyfile(source, destination)
        sources[source.name] = {
            "source": record(source),
            "snapshot": record(destination),
        }
    shutil.copyfile(STYLE_IMAGE, out / "style-reference.png")
    assert (
        record(cfg.source)["sha256"]
        == "07efff9f6a96ff7d4556df723f6f6386c31c111ad89ac65ebff21ded82050201"
    )
    with np.load(cfg.source, allow_pickle=False) as saved:
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        ids, rest, u, b, z = (
            saved[k] for k in ["active_ids", "rest_points", "u", "B", "Z"]
        )
    assert np.max(np.abs(b @ b.swapaxes(-1, -2) - np.eye(3) - z)) < 2e-13
    values, axes = np.linalg.eigh(z)
    values, axes = values[:, ::-1].copy(), axes[:, :, ::-1].copy()
    assert np.all(values[:, 0] >= -ZERO_TOL)
    assert np.all(values[:, 2] <= ZERO_TOL)
    error = float(np.max(np.abs(np.einsum("nik,nk,njk->nij", axes, values, axes) - z)))
    assert error < 2e-13
    gaps = (values[:, :-1] - values[:, 1:]) / np.maximum(
        1, np.max(np.abs(values), axis=1, keepdims=True)
    )
    unique = np.c_[
        gaps[:, 0] > GAP_TOL, np.min(gaps, axis=1) > GAP_TOL, gaps[:, 1] > GAP_TOL
    ]
    magnitude = 1 - 1 / np.sqrt(1 + np.abs(values))
    lengths = MAX_LENGTH * magnitude * unique * (np.abs(values) > ZERO_TOL)
    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    assert np.array_equal(volume.points, rest)
    assert np.array_equal(np.flatnonzero(volume.cell_data["ActivationMask"]), ids)
    tets = volume.cells.reshape(-1, 5)[:, 1:][ids]
    rest_tets, deformed_tets = rest[tets], (rest + u)[tets]
    rest_edges = (rest_tets[:, 1:] - rest_tets[:, :1]).swapaxes(1, 2)
    deformed_edges = (deformed_tets[:, 1:] - deformed_tets[:, :1]).swapaxes(1, 2)
    f = deformed_edges @ np.linalg.inv(rest_edges)
    transported = f @ axes
    norms = np.linalg.norm(transported, axis=1)
    assert np.all(np.isfinite(norms))
    assert np.all(norms > 0)
    spatial_axes = transported / norms[:, None, :]
    centers = deformed_tets.mean(axis=1)
    control_ids = np.asarray(volume.cell_data["ActivationControlId"])[ids]
    weights = (
        np.asarray(volume.cell_data["Volume"])[ids]
        * np.asarray(volume.cell_data["MuscleFraction"])[ids]
    )
    volume.points = rest + u
    skin.points = (rest + u)[np.asarray(skin.point_data["GlobalPointId"])]
    np.savez_compressed(
        out / "eigenmodes.npz",
        global_cell_ids=ids,
        centers_rest=rest_tets.mean(axis=1),
        centers_deformed=centers,
        eigenvalues_descending=values,
        reference_axes_columns=axes,
        spatial_axes_columns=spatial_axes,
        deformation_gradient=f,
        normalized_adjacent_eigenvalue_gaps=gaps,
        axis_unique=unique,
        display_magnitude=magnitude,
        display_length_m=lengths,
        muscle_volume_weights=weights,
    )
    glyphs = []
    for j, (identifier, _) in enumerate(LABELS):
        sign_rgb = np.where(
            (values[:, j] > 0)[:, None], [178, 24, 43], [33, 102, 172]
        ).astype(np.uint8)
        glyph = line_mesh(
            centers,
            spatial_axes[:, :, j],
            lengths[:, j],
            {
                "GlobalCellId": ids,
                "ActivationControlId": control_ids,
                "Z_eigenvalue": values[:, j],
                "ReferenceAxis": axes[:, :, j],
                "SpatialAxis": spatial_axes[:, :, j],
                "AxisUnique": unique[:, j].astype(np.uint8),
                "DisplayMagnitudePercent": 100 * magnitude[:, j],
                "DisplayLengthM": lengths[:, j],
                "SignRGB": sign_rgb,
            },
        )
        glyph.save(out / f"all-active-{identifier}.vtp", binary=True)
        glyphs.append(glyph)
    LOG.info("Loaded %d active tetrahedra; reconstruct error %.3g", len(ids), error)
    context = build_muscle_region_context(volume)
    views = {}
    for view in json.loads(CAMERAS.read_text())["views"]:
        name, camera = view["id"], view["camera"]
        if name not in ["side-context", "region1-mouth-corner"]:
            continue
        visibility = visible_region_mask(
            context, centers, ids, control_ids, camera, window_size=WINDOW
        )
        save_region_visibility(visibility, out / f"{name}-visibility.npz")
        panels, images = [], []
        for glyph, (identifier, title) in zip(glyphs, LABELS, strict=True):
            path = out / f"{name}-{identifier}.png"
            panels.append(
                render_panel(skin, glyph, visibility.mask, camera, title, path)
            )
            images.append(path)
        triptych_path = out / f"{name}-triptych.png"
        triptych(images, triptych_path)
        sign_panel = render_panel(
            skin,
            glyphs[1],
            visibility.mask,
            camera,
            "Residual mode 2 - sign; line length still shows magnitude",
            out / f"{name}-mode-2-sign.png",
            sign=True,
        )
        views[name] = {
            "camera": camera,
            "common_visible_candidates": visibility.retained_count,
            "modes": panels,
            "triptych": record(triptych_path),
            "mode_2_sign": sign_panel,
        }
        LOG.info(
            "Rendered %s with %d shared visibility candidates",
            name,
            visibility.retained_count,
        )
    norm2 = np.sum(values**2, axis=0)
    weighted_norm2 = np.sum(weights[:, None] * values**2, axis=0)
    statistics = [
        {
            "mode": j + 1,
            "positive_cells": int(np.sum(values[:, j] > ZERO_TOL)),
            "negative_cells": int(np.sum(values[:, j] < -ZERO_TOL)),
            "neutral_cells": int(np.sum(np.abs(values[:, j]) <= ZERO_TOL)),
            "nonunique_axes": int(np.sum(~unique[:, j])),
            "squared_frobenius_share_unweighted": float(norm2[j] / norm2.sum()),
            "squared_frobenius_share_muscle_volume_weighted": float(
                weighted_norm2[j] / weighted_norm2.sum()
            ),
        }
        for j in range(3)
    ]
    summary = {
        "status": "completed",
        "source": record(cfg.source),
        "sources": sources,
        "inputs": [
            record(FIXTURE / "volume.vtu"),
            record(FIXTURE / "skin.vtp"),
            record(CAMERAS),
        ],
        "style_reference": record(out / "style-reference.png"),
        "active_cell_count": len(ids),
        "Z_reconstruction_max_abs_error": error,
        "mode_statistics": statistics,
        "views": views,
        "encoding": {
            "frame": "saved deformed full-fit geometry; directions normalize(F @ reference_axis)",
            "ordering": "z1 >= z2 >= z3, largest algebraic first",
            "residual": "Z_res = z2*n2*n2T + z3*n3*n3T",
            "magnitude": "a = 1-1/sqrt(1+abs(z)); common bounded display transform",
            "color": "Viridis, 100*a, common limits [0,100]; does not encode sign",
            "length": "0.0045 m * a; no floor; zero for neutral or nonunique axes",
            "mode_2_sign": "red positive z, blue negative z; identical geometry and lengths",
            "sampling": "all camera-facing region-matched active tetrahedra; no grid sampling; same mask for all modes",
            "skin_opacity": 0.06,
            "line_width": 1.0,
            "neutral_tolerance": ZERO_TOL,
            "relative_eigenvalue_gap_tolerance": GAP_TOL,
        },
        "limitations": [
            "Display magnitude for negative z is not expansion strain or shortening percent.",
            "Transported axes need not remain orthogonal and are not total spatial stress eigenvectors.",
            "Modes 2 and 3 jointly form the residual; equilibrium shapes are not additive.",
            "Tensor norm shares are not energy or shape-fit contribution shares.",
            "Near repeated eigenvalues do not define unique individual directions; their lines are suppressed.",
        ],
        "runtime": {
            "software_rendering": os.environ.get("LIBGL_ALWAYS_SOFTWARE"),
            "pyvista": pv.__version__,
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    cherries.log_metrics(
        {"active_cells": len(ids), "Z_reconstruction_max_abs_error": error}
    )
    LOG.info("Completed meeting-style visualization: %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
