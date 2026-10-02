"""Render aligned shapes and principal activation fields for five saved states."""

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
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from muscle_glyph_context import (
    build_muscle_region_context,
    save_region_visibility,
    set_parallel_camera,
    visible_region_mask,
)
from PIL import Image, ImageDraw, ImageFont

from liblaf import cherries

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[6]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
SCRATCH = (
    ROOT / "exp/2026/09/09/activation-space-smoothness/data/25-learned-axis-smooth"
)
WINDOW = (1800, 1800)
BACKGROUND = "#f4f2ed"
GRAY = "#8d969b"
PALETTE = ["#053061", "#4393c3", "#b5b6b4", "#d6604d", "#67001f"]
CMAP = LinearSegmentedColormap.from_list("signed_activation", PALETTE)
MAX_LENGTH = 0.0045
ZERO_TOL = 1e-8
GAP_TOL = 1e-6
FONT = Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans")))
BOLD_FONT = Path(
    font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
)


class Config(cherries.BaseConfig):
    free: Path = cherries.input("10-forward/baseline-replay.npz")
    dominant: Path = cherries.input("10-forward/dominant-only.npz")
    fixed: Path = cherries.input("42-fixed-directions-400/best.npz")
    fixed_initialization: Path = cherries.input(
        "42-fixed-directions-400/initialization.npz"
    )
    released: Path = cherries.input("90-released-axes/best.npz")
    scratch: Path = SCRATCH / "step-0128.npz"
    cameras: Path = cherries.input("77-signed-meeting-eigenmodes/summary.json")
    output_dir: Path = cherries.output("96-aligned-five-way", mkdir=True)


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def load_state(path: Path, kind: str, initial: dict) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        a = {key: saved[key] for key in saved.files}
    assert bool(a["solver_valid"])
    if "physical_volume_energy" in a:
        assert bool(a["physical_volume_energy"])
    if kind == "fixed":
        assert int(a["step"]) == 400
        assert str(a["axes_sha256"]) == str(initial["axes_sha256"])
        assert np.array_equal(a["active_ids"], initial["active_ids"])
        assert np.min(a["s"]) >= 0
        a["rest_points"] = initial["rest_points"]
        axes = initial["axes"]
        a["B"] = np.eye(3) + a["s"][:, None, None] * axes[:, :, None] * axes[:, None, :]
    elif kind in {"released", "scratch"}:
        vector = a["v"] if kind == "released" else a["q"]
        assert vector.shape == (len(a["active_ids"]), 3)
        assert int(a["step"]) == (200 if kind == "released" else 128)
        c = vector[:, :, None] * vector[:, None, :]
        assert np.allclose(a["C"], c, rtol=2e-14, atol=3e-13)
        b = np.eye(3) + c
        if "B" in a:
            assert np.allclose(a["B"], b, rtol=2e-14, atol=3e-13)
        a["B"] = b
    z = a["B"] @ a["B"].swapaxes(1, 2) - np.eye(3)
    error = 0.0
    if "Z" in a:
        error = float(np.max(np.abs(a["Z"] - z)))
        assert np.allclose(a["Z"], z, rtol=2e-14, atol=3e-13), (kind, error)
    assert all(np.isfinite(a[key]).all() for key in ("u", "rest_points", "B"))
    return {**a, "Z": z, "tensor_reconstruction_max_abs_error": error}


def make_glyphs(
    state: dict, rest: np.ndarray, tets: np.ndarray, dm_inv: np.ndarray
) -> tuple:
    ids = state["active_ids"]
    deformed = rest + state["u"]
    dt = deformed[tets[ids]]
    f = (dt[:, 1:] - dt[:, :1]).swapaxes(1, 2) @ dm_inv[ids]
    values, axes = np.linalg.eigh(state["Z"])
    eigenvalue, reference_axis = values[:, -1], axes[:, :, -1]
    scale = np.maximum(1, np.max(np.abs(values), axis=1))
    unique = (values[:, -1] - values[:, -2]) / scale > GAP_TOL
    nonneutral = np.abs(eigenvalue) > ZERO_TOL
    shown = unique & nonneutral
    residual = (
        np.einsum("nij,nj->ni", state["Z"], reference_axis)
        - eigenvalue[:, None] * reference_axis
    )
    eig_error = float(np.max(np.linalg.norm(residual, axis=1) / scale))
    assert eig_error < 5e-14
    transported = np.einsum("nij,nj->ni", f, reference_axis)
    norms = np.linalg.norm(transported, axis=1)
    assert np.isfinite(norms).all()
    assert np.all(norms > 0)
    spatial_axis = transported / norms[:, None]
    signed = 100 * np.sign(eigenvalue) * (1 - 1 / np.sqrt(1 + np.abs(eigenvalue)))
    lengths = MAX_LENGTH * np.abs(signed) / 100 * shown
    centers = dt.mean(axis=1)
    points = np.empty((2 * len(ids), 3))
    half = 0.5 * lengths[:, None] * spatial_axis
    points[0::2], points[1::2] = centers - half, centers + half
    index = np.arange(len(ids))
    glyphs = pv.PolyData(
        points, lines=np.c_[np.full(len(ids), 2), 2 * index, 2 * index + 1].ravel()
    )
    for key, value in {
        "GlobalCellId": ids,
        "PrincipalEigenvalueZ": eigenvalue,
        "ReferencePrincipalAxis": reference_axis,
        "SpatialPrincipalAxis": spatial_axis,
        "PrincipalAxisUnique": unique.astype(np.uint8),
        "SignedDisplayMagnitudePercent": signed,
        "DisplayLengthM": lengths,
    }.items():
        glyphs.cell_data[key] = value
    return (
        glyphs,
        centers,
        shown,
        {
            "normalized_eigenpair_residual_max": eig_error,
            "nonunique_axes": int(np.sum(~unique)),
            "neutral_modes": int(np.sum(~nonneutral)),
            "nonzero_unique_lines": int(np.sum(shown)),
            "signed_display_range_percent": [float(signed.min()), float(signed.max())],
            "eigenvalue_range": [float(eigenvalue.min()), float(eigenvalue.max())],
        },
    )


def render(
    skin: pv.PolyData,
    camera: dict,
    path: Path,
    glyphs: pv.UnstructuredGrid | None = None,
) -> dict:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="three lights")
    plotter.set_background(BACKGROUND)
    actor = plotter.add_mesh(
        skin, color=GRAY, opacity=1.0 if glyphs is None else 0.06, smooth_shading=False
    )
    assert actor.GetProperty().GetInterpolation() == 0
    if glyphs is not None:
        plotter.add_mesh(
            glyphs,
            scalars="SignedDisplayMagnitudePercent",
            cmap=CMAP,
            clim=(-100, 100),
            lighting=False,
            line_width=1.0,
            render_lines_as_tubes=False,
            show_scalar_bar=False,
        )
    set_parallel_camera(plotter, camera)
    projected = np.asarray(
        plotter.camera.GetCompositeProjectionTransformMatrix(1.0, -1.0, 1.0).GetData()
    ).reshape(4, 4)
    # The x/y/w rows establish shared screen alignment despite state-specific clipping ranges.
    alignment = {
        "position": list(plotter.camera.position),
        "focal_point": list(plotter.camera.focal_point),
        "view_up": list(plotter.camera.up),
        "parallel_scale": plotter.camera.parallel_scale,
        "screen_projection_rows": projected[[0, 1, 3]].tolist(),
    }
    plotter.screenshot(path)
    plotter.close()
    return alignment


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD_FONT if bold else FONT), size)


def centered(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    size: int,
    *,
    bold: bool = False,
    color: str = "#252525",
) -> None:
    draw.text((x, y), text, font=font(size, bold=bold), fill=color, anchor="mt")


def compose(states: list[dict], view: str, out: Path, *, single: bool = False) -> Path:
    width, height = len(states) * WINDOW[0], 4380
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title = (
        "Aligned shape and principal activation"
        if single
        else "Deformed shape and principal activation | aligned five-state comparison"
    )
    centered(draw, width // 2, 25, title, 66 if single else 80, bold=True)
    centered(
        draw,
        width // 2,
        125,
        "Mouth-corner close-up"
        if view == "region1-mouth-corner"
        else "Whole-face view",
        46,
    )
    shape_y, field_y = 470, 2350
    for column, state in enumerate(states):
        x, center = column * WINDOW[0], (column * 2 + 1) * WINDOW[0] // 2
        centered(draw, center, 210, state["title"], 62, bold=True)
        centered(draw, center, 290, state["subtitle"], 46)
        centered(
            draw,
            center,
            350,
            f"Fit RMS {state['metrics']['uniform_fit_rms_mm']:.3f} mm   |   Inversions {state['metrics']['inverted_all_cells']}",
            46,
            color="#8b2826" if state["metrics"]["inverted_all_cells"] else "#252525",
        )
        for kind, y in (("shape", shape_y), ("principal", field_y)):
            with Image.open(state["views"][view][kind]["path"]) as panel:
                assert panel.size == WINDOW
                canvas.paste(panel.convert("RGB"), (x, y))
        if column:
            draw.line([(x, 195), (x, 4150)], fill="#d7d4ce", width=3)
    centered(draw, width // 2, 421, "DEFORMED SHAPE", 34, bold=True)
    centered(
        draw,
        width // 2,
        2290,
        "PRINCIPAL ACTIVATION ON THE SAME DEFORMED SHAPE",
        34,
        bold=True,
    )
    bar_width = min(width - 320, 1500)
    bar_x, bar_y = (width - bar_width) // 2, 4195
    gradient = (CMAP(np.linspace(0, 1, bar_width))[:, :3] * 255).astype(np.uint8)
    canvas.paste(
        Image.fromarray(np.tile(gradient[None, :, :], (35, 1, 1))), (bar_x, bar_y)
    )
    for tick in (-100, -50, 0, 50, 100):
        tx = bar_x + int((tick + 100) / 200 * bar_width)
        draw.line([(tx, bar_y + 35), (tx, bar_y + 44)], fill="#252525", width=2)
        centered(draw, tx, bar_y + 50, str(tick), 32)
    centered(draw, width // 2, 4150, "Signed display magnitude (%)", 32)
    centered(
        draw,
        width // 2,
        4290,
        "Blue: extension-like   |   Red: contraction-like   |   100% line length = 4.5 mm",
        32,
    )
    centered(
        draw,
        width // 2,
        4340,
        "Shared camera, scale and color limits. Principal mode shown; each shape uses its full saved activation.",
        27,
    )
    filename = (
        f"{view}-{states[0]['id']}-aligned.png"
        if single
        else f"{view}-aligned-five-way.png"
    )
    path = out / filename
    canvas.save(path)
    return path


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
        dest = out / "sources" / source.name
        shutil.copyfile(source, dest)
        sources.append({"source": record(source), "snapshot": record(dest)})
    inputs = [
        record(p)
        for p in (
            cfg.free,
            cfg.dominant,
            cfg.fixed,
            cfg.fixed_initialization,
            cfg.released,
            cfg.scratch,
            cfg.cameras,
            FIXTURE / "volume.vtu",
            FIXTURE / "skin.vtp",
            cfg.scratch.parent / "config.json",
            cfg.scratch.parent / "provenance.json",
        )
    ]
    with np.load(cfg.fixed_initialization, allow_pickle=False) as a:
        initial = {key: a[key] for key in a.files}
    cameras = {
        name: view["camera"]
        for name, view in json.loads(cfg.cameras.read_text())["views"].items()
    }
    assert set(cameras) == {"side-context", "region1-mouth-corner"}
    volume = pv.read(FIXTURE / "volume.vtu")
    skin_template = pv.read(FIXTURE / "skin.vtp")
    rest = np.asarray(volume.points).copy()
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    ids = np.flatnonzero(volume.cell_data["ActivationMask"])
    control = np.asarray(volume.cell_data["ActivationControlId"])[ids]
    skin_ids = np.asarray(skin_template.point_data["GlobalPointId"], dtype=np.int64)
    target = np.asarray(volume.point_data["Smile"])
    top = np.flatnonzero(
        np.asarray(volume.point_data["IsFace"], bool) & np.isfinite(target).all(axis=1)
    )
    dm = (rest[tets[:, 1:]] - rest[tets[:, :1]]).swapaxes(1, 2)
    assert np.all(np.linalg.det(dm) > 0)
    dm_inv = np.linalg.inv(dm)
    del dm
    specs = [
        ("free", "Free activation", "6 DoF; update 200; no regularization", cfg.free),
        (
            "dominant",
            "Dominant only",
            "Principal contraction retained; re-equilibrated",
            cfg.dominant,
        ),
        (
            "fixed",
            "Fixed-axis refit",
            "1 DoF; update 400; no regularization",
            cfg.fixed,
        ),
        (
            "released",
            "Released-axis continuation",
            "3 DoF; +200 updates; no regularization",
            cfg.released,
        ),
        (
            "scratch",
            "Learned-axis from scratch",
            "3 DoF; update 128; smoothness ON",
            cfg.scratch,
        ),
    ]
    states, alignments = [], {}
    full_principal_tensor = None
    dominant_projection_error = None
    for index, (kind, title, subtitle, path) in enumerate(specs):
        cherries.set_step(index)
        LOG.info("Preparing %s from %s", title, path)
        a = load_state(path, kind, initial)
        assert np.array_equal(a["rest_points"], rest)
        assert np.array_equal(a["active_ids"], ids)
        assert a["u"].shape == rest.shape
        deformed = rest + a["u"]
        df = (deformed[tets[:, 1:]] - deformed[tets[:, :1]]).swapaxes(1, 2) @ dm_inv
        determinant = np.linalg.det(df)
        del df
        metrics = {
            "uniform_fit_rms_mm": float(
                1000
                * np.sqrt(np.mean(np.sum((a["u"][top] - target[top]) ** 2, axis=1)))
            ),
            "inverted_all_cells": int(np.sum(determinant <= 0)),
            "inverted_active_cells": int(np.sum(determinant[ids] <= 0)),
            "detF_min": float(determinant.min()),
        }
        glyphs, centers, eligible, field = make_glyphs(a, rest, tets, dm_inv)
        if kind == "free":
            axis = np.asarray(glyphs.cell_data["ReferencePrincipalAxis"])
            value = np.maximum(glyphs.cell_data["PrincipalEigenvalueZ"], 0)
            full_principal_tensor = (
                value[:, None, None] * axis[:, :, None] * axis[:, None, :]
            )
        elif kind == "dominant":
            assert full_principal_tensor is not None
            dominant_projection_error = float(
                np.max(np.abs(a["Z"] - full_principal_tensor))
            )
            assert np.allclose(a["Z"], full_principal_tensor, rtol=2e-13, atol=3e-13)
        glyphs.cell_data["ActivationControlId"] = control
        field_path = out / f"{kind}-principal-field.vtp"
        glyphs.save(field_path, binary=True)
        skin = skin_template.copy(deep=True)
        skin.points = deformed[skin_ids]
        skin_path = out / f"{kind}-deformed-skin.vtp"
        skin.save(skin_path, binary=True)
        volume.points = deformed
        context = build_muscle_region_context(volume)
        views = {}
        for view, camera in cameras.items():
            LOG.info("Rendering %s: %s", kind, view)
            visibility = visible_region_mask(
                context, centers, ids, control, camera, window_size=WINDOW
            )
            visibility_path = out / f"{view}-{kind}-visibility.npz"
            save_region_visibility(visibility, visibility_path)
            mask = visibility.mask & eligible
            shown = glyphs.extract_cells(mask)
            assert shown.n_cells == int(np.sum(mask))
            shape_path, principal_path = (
                out / f"{view}-{kind}-shape.png",
                out / f"{view}-{kind}-principal.png",
            )
            shape_alignment = render(skin, camera, shape_path)
            glyph_alignment = render(skin, camera, principal_path, shown)
            assert shape_alignment == glyph_alignment
            if view in alignments:
                assert shape_alignment == alignments[view], (
                    kind,
                    view,
                    shape_alignment,
                    alignments[view],
                )
            else:
                alignments[view] = shape_alignment
            views[view] = {
                "shape": record(shape_path),
                "principal": record(principal_path),
                "visibility": record(visibility_path),
                "visible_candidates": int(visibility.retained_count),
                "shown_lines": shown.n_cells,
                "camera_alignment": shape_alignment,
            }
        states.append(
            {
                "id": kind,
                "title": title,
                "subtitle": subtitle,
                "source": record(path),
                "metrics": metrics,
                "tensor_reconstruction_max_abs_error": a[
                    "tensor_reconstruction_max_abs_error"
                ],
                "field": field,
                "field_vtp": record(field_path),
                "skin_vtp": record(skin_path),
                "views": views,
            }
        )
        LOG.info("%s metrics: %s", kind, metrics)
        cherries.log_metrics({f"{kind}/{key}": value for key, value in metrics.items()})
    composites = {}
    for view in cameras:
        plate = compose(states, view, out)
        with Image.open(plate) as original:
            preview = out / f"{view}-aligned-five-way-preview.png"
            original.resize((4500, 2190), Image.Resampling.LANCZOS).save(preview)
        columns = [record(compose([state], view, out, single=True)) for state in states]
        composites[view] = {
            "full_resolution": record(plate),
            "preview": record(preview),
            "individual_columns": columns,
        }
    summary = {
        "status": "completed",
        "inputs": inputs,
        "sources": sources,
        "fonts": [record(FONT), record(BOLD_FONT)],
        "states": states,
        "cameras": cameras,
        "verified_alignment": alignments,
        "dominant_matches_positive_full_principal_max_abs_error": dominant_projection_error,
        "composites": composites,
        "encoding": {
            "principal": "largest algebraic eigenvalue of Z=B*BT-I, with its unoriented reference eigenaxis",
            "direction": "normalize(F*n) on each saved deformed geometry",
            "color_stops": PALETTE,
            "signed_display": "100*sign(z)*(1-1/sqrt(1+abs(z)))",
            "color_limits_percent": [-100, 100],
            "line_length": "4.5 mm*abs(signed_display)/100",
            "neutral_tolerance": ZERO_TOL,
            "principal_gap_tolerance": GAP_TOL,
            "gap_scale": "max(1,max(abs(all_eigenvalues)))",
            "visibility": "same muscle-region front-label rule, recomputed for each saved deformation",
            "shape_opacity": 1.0,
            "activation_context_opacity": 0.06,
            "flat_shading": True,
            "panel_pixels": list(WINDOW),
            "composite_pixels": [9000, 4380],
        },
        "limitations": [
            "Saved states have different optimization budgets and regularization; this is a visualization comparison.",
            "Free-activation shape uses the complete tensor; only its principal field is displayed.",
            "Direction projection includes each state's deformation; reference-axis differences require comparing reference fields.",
            "Inversions are included and labeled; absence of inversions does not establish good element quality.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for image in out.glob("*.png"):
        with Image.open(image) as panel:
            panel.verify()
    for artifact in out.glob("*.png"):
        cherries.log_output(artifact)
    cherries.log_output(out / "summary.json")
    LOG.info("Completed aligned comparison in %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
