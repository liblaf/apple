"""Render two slide-sized, four-stage active-strain comparison figures."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from activation_scene import (
    PrincipalGlyphs,
    contraction_colormap,
    principal_glyphs,
)
from experiment import Profile
from PIL import Image, ImageDraw, ImageFont
from shape_scene import (
    CRANIUM_PATH,
    EYES_PATH,
    MANDIBLE_PATH,
    VOLUME_PATH,
    ShapeScene,
    deformed_exterior,
    load_static_context,
)

from liblaf import cherries

REPO = Path(__file__).resolve().parents[6]
CONTEXT_SRC = REPO / "exp/2026/09/14/dominant-activation-ablation/src"
sys.path.insert(0, str(CONTEXT_SRC))
from muscle_glyph_context import (  # noqa: E402
    build_muscle_region_context,
    save_region_visibility,
    set_parallel_camera,
    visible_region_mask,
)

LOGGER = logging.getLogger(__name__)
RESOLUTION_SCALE = 2
PANEL_LAYOUT = (1230, 1000)
PAGE_LAYOUT = (5120, 2880)
WINDOW = (2460, 2000)
ACTIVATION_WINDOW = (4428, 3600)
ACTIVATION_LINE_WIDTH_PX = 2.0
ACTIVATION_CONTEXT_OPACITY = 0.06
COLOR_AMPLITUDE_MAX = 60.0
COLOR_TICKS = (0, 1, 3, 10, 30, 60)
PAGE = (10240, 5760)
PANEL_BG = "#f4f2ed"
SHAPE_COLOR = "#8d969b"
STAGES = ("symmetric6", "psd6", "rankone_fixed", "rankone_learned")
HEADINGS = (
    "Unrestricted\nactivation",
    "Contraction\nonly",
    "Fixed-axis\ncontraction",
    "Released-axis\ncontraction",
)
DOFS = ("6 DoF", "6 DoF", "1 DoF", "3 DoF")
FONT_DIR = Path("/usr/share/fonts/TTF")


class Config(cherries.BaseConfig):
    source: Path = Path("51-visualization-checkpoints-002")
    output: Path = Path("52-four-stage-figures-010")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(
        str(FONT_DIR / f"DejaVuSans{suffix}.ttf"), size * RESOLUTION_SCALE
    )


def scaled_point(x: float, y: float) -> tuple[int, int]:
    """Map layout coordinates to the native high-resolution canvas."""
    return round(x * RESOLUTION_SCALE), round(y * RESOLUTION_SCALE)


def load_states(source: Path) -> list[dict]:
    states = []
    for loss in ("l2", "l2-normal"):
        branch = source / loss
        manifest = json.loads((branch / "manifest.json").read_text())
        assert len(manifest["stages"]) == 4
        for i, stage in enumerate(STAGES):
            folder = branch / f"{loss}-{stage}"
            checkpoint = folder / "last.npz"
            record = manifest["stages"][i]
            assert record["stage_dir"] == folder.name
            assert sha256(checkpoint) == record["last_sha256"]
            with np.load(checkpoint, allow_pickle=False) as archive:
                state = {key: archive[key].copy() for key in archive.files}
            step = int(state["step"])
            assert step == record["step"]
            assert str(state["activation_model"]) == "strain"
            assert str(state["mode"]) == stage
            rows = list(csv.DictReader((folder / "trace.csv").open()))
            matches = [row for row in rows if int(row["step"]) == step]
            assert len(matches) == 1
            states.append(
                {
                    "loss": loss,
                    "stage": stage,
                    "index": i,
                    "step": step,
                    "u": state["u"],
                    "B": state["B"],
                    "solver_valid": bool(state["solver_valid"]),
                    "trace": matches[0],
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": record["last_sha256"],
                    "captured_at": manifest["captured_at"],
                    "status_at_capture": manifest["current_status"]["stages"][i][
                        "status"
                    ],
                }
            )
    return states


def common_camera(scene: ShapeScene, states: list[dict]) -> dict:
    # A three-quarter view that reveals both actual eyes through the eyelids.
    toward_camera = np.array([0.65, 0.03, 1.0])
    toward_camera /= np.linalg.norm(toward_camera)
    right = np.cross([0.0, 1.0, 0.0], toward_camera)
    right /= np.linalg.norm(right)
    up = np.cross(toward_camera, right)
    basis = np.column_stack((right, up, toward_camera))
    point_sets = [mesh.points for mesh in scene.bones.values()] + [scene.eyes.points]
    point_sets += [
        scene.rest_points[scene.exterior_point_ids]
        + state["u"][scene.exterior_point_ids]
        for state in states
    ]
    projected = np.vstack(point_sets) @ basis
    low, high = projected.min(axis=0), projected.max(axis=0)
    focal = basis @ ((low + high) / 2.0)
    half_extent = (high - low) / 2.0
    parallel_scale = 1.055 * max(
        half_extent[1], half_extent[0] / (WINDOW[0] / WINDOW[1])
    )
    return {
        "position": (focal + 0.6 * toward_camera).tolist(),
        "focal_point": focal.tolist(),
        "view_up": up.tolist(),
        "parallel_scale": float(parallel_scale),
        "projected_bounds_m": [low.tolist(), high.tolist()],
        "window_size": list(WINDOW),
        "fit_padding_factor": 1.055,
        "unit_direction_toward_camera": toward_camera.tolist(),
        "policy": "one orthographic camera fitted to all eight full surfaces and static anatomy",
    }


def capture(plotter: pv.Plotter, camera: dict, path: Path) -> np.ndarray:
    set_parallel_camera(plotter, camera)
    matrix = plotter.camera.GetCompositeProjectionTransformMatrix(
        WINDOW[0] / WINDOW[1], -1, 1
    )
    screen_projection = np.array(
        [[matrix.GetElement(row, col) for col in range(4)] for row in (0, 1, 3)]
    )
    with Image.fromarray(plotter.screenshot(return_img=True)) as rendered:
        if rendered.size != WINDOW:
            rendered.save(path.with_name(f"{path.stem}-native.png"))
            rendered.resize(WINDOW, Image.Resampling.LANCZOS).save(path)
        else:
            rendered.save(path)
    plotter.close()
    assert Image.open(path).size == WINDOW
    return screen_projection


def new_plotter(window_size: tuple[int, int] = WINDOW) -> pv.Plotter:
    plotter = pv.Plotter(
        off_screen=True, window_size=window_size, lighting="three lights"
    )
    plotter.set_background(PANEL_BG)
    plotter.ren_win.SetMultiSamples(8)
    return plotter


def render_shape(
    scene: ShapeScene, surface: pv.PolyData, camera: dict, path: Path
) -> np.ndarray:
    plotter = new_plotter()
    actor = plotter.add_mesh(
        surface, color=SHAPE_COLOR, smooth_shading=False, opacity=1.0
    )
    assert actor.GetProperty().GetInterpolation() == 0
    for mesh in scene.bones.values():
        plotter.add_mesh(mesh, color="#e6dfcf", smooth_shading=True)
    plotter.add_mesh(
        scene.eyes, color="#fffdf8", smooth_shading=True, ambient=0.3, diffuse=0.7
    )
    return capture(plotter, camera, path)


def render_activation(
    skin: pv.PolyData,
    glyph: PrincipalGlyphs,
    mask: np.ndarray,
    camera: dict,
    path: Path,
) -> np.ndarray:
    plotter = new_plotter(ACTIVATION_WINDOW)
    plotter.add_mesh(
        skin,
        color=SHAPE_COLOR,
        opacity=ACTIVATION_CONTEXT_OPACITY,
        smooth_shading=False,
    )
    amplitude = np.sqrt(1.0 + glyph.eigenvalues_z[mask]) - 1.0
    color_fraction = log_color_fraction(amplitude)
    lengths = 0.0045 * color_fraction
    half = 0.5 * lengths[:, None] * glyph.spatial_axes[mask]
    ends = np.stack((glyph.centers[mask] - half, glyph.centers[mask] + half), axis=1)
    n = len(ends)
    assert n > 0
    lines = np.column_stack(
        (np.full(n, 2), 2 * np.arange(n), 2 * np.arange(n) + 1)
    ).ravel()
    poly = pv.PolyData(ends.reshape(-1, 3), lines=lines)
    poly.cell_data["LogActivationColor"] = color_fraction
    plotter.add_mesh(
        poly,
        scalars="LogActivationColor",
        preference="cell",
        cmap=contraction_colormap(),
        clim=(0.0, 1.0),
        lighting=False,
        line_width=ACTIVATION_LINE_WIDTH_PX,
        render_lines_as_tubes=False,
        show_scalar_bar=False,
    )
    return capture(plotter, camera, path)


def log_color_fraction(amplitude: np.ndarray) -> np.ndarray:
    """Map dimensionless principal amplitude to a shared zero-preserving log scale."""
    amplitude = np.asarray(amplitude, dtype=np.float64)
    assert np.isfinite(amplitude).all()
    assert np.all(amplitude >= 0.0)
    assert np.all(amplitude <= COLOR_AMPLITUDE_MAX)
    return np.log1p(amplitude) / np.log1p(COLOR_AMPLITUDE_MAX)


def compose(loss: str, states: list[dict], out: Path) -> dict:
    page = Image.new("RGB", PAGE, "#000000")
    draw = ImageDraw.Draw(page)
    name = "L2" if loss == "l2" else "L2 + normal"
    draw.text(
        scaled_point(PAGE_LAYOUT[0] / 2, 38),
        f"{name}: shapes and principal activation",
        font=font(92, bold=True),
        fill="white",
        anchor="mt",
    )
    draw.text(
        scaled_point(PAGE_LAYOUT[0] / 2, 167),
        "Deformed shape (top) and activation on the same shape (bottom)",
        font=font(43),
        fill="#d9d9d9",
        anchor="mt",
    )
    for i, state in enumerate(states):
        x = 64 + i * (PANEL_LAYOUT[0] + 24)
        center = x + PANEL_LAYOUT[0] / 2
        draw.multiline_text(
            scaled_point(center, 252),
            HEADINGS[i],
            font=font(64, bold=True),
            fill="white",
            anchor="ma",
            align="center",
            spacing=5 * RESOLUTION_SCALE,
        )
        suffix = " (snapshot)" if state["status_at_capture"] != "completed" else ""
        draw.text(
            scaled_point(center, 412),
            f"{DOFS[i]}  ·  update {state['step']}/200{suffix}",
            font=font(38),
            fill="#f1cc83" if suffix else "#dfdfdf",
            anchor="mt",
        )
        draw.text(
            scaled_point(center, 478),
            f"Fit {state['fit_rms_mm']:.3f} mm  ·  normal {state['normal_angle_rms_deg']:.2f}°",
            font=font(38),
            fill="white",
            anchor="mt",
        )
        stem = f"{loss}-{STAGES[i]}"
        for row, y in (("shape", 550), ("activation", 1576)):
            with Image.open(out / "panels" / f"{stem}-{row}.png") as panel:
                assert panel.size == WINDOW
                page.paste(panel.convert("RGB"), scaled_point(x, y))
    draw.text(
        scaled_point(64, 2605),
        "Principal activation a (dimensionless)",
        font=font(39),
        fill="white",
    )
    colorbar_x, colorbar_y, colorbar_w = 1710, 2613, 1700
    ramp = contraction_colormap()(
        np.linspace(0, 1, colorbar_w * RESOLUTION_SCALE), bytes=True
    )[:, :3]
    bar = Image.fromarray(np.tile(ramp[None, :, :], (40 * RESOLUTION_SCALE, 1, 1)))
    page.paste(bar, scaled_point(colorbar_x, colorbar_y))
    for value in COLOR_TICKS:
        x = colorbar_x + float(log_color_fraction(value)) * colorbar_w
        draw.line(
            [
                scaled_point(x, colorbar_y + 40),
                scaled_point(x, colorbar_y + 49),
            ],
            fill="#dfdfdf",
            width=2 * RESOLUTION_SCALE,
        )
        draw.text(
            scaled_point(x, colorbar_y + 53),
            str(value),
            font=font(35),
            fill="#dfdfdf",
            anchor="mt",
        )
    draw.text(
        scaled_point(4140, 2612),
        "Red: contraction-like",
        font=font(38),
        fill="#df6255",
    )
    draw.text(
        scaled_point(64, 2680),
        "Color: log(1 + a);  a = σ_max(B) − 1",  # noqa: RUF001
        font=font(37),
        fill="#dfdfdf",
        anchor="lt",
    )
    draw.text(
        scaled_point(PAGE_LAYOUT[0] - 64, 2680),
        "Line length: 4.5 mm × log(1 + a) / log(61)",  # noqa: RUF001
        font=font(37),
        fill="#dfdfdf",
        anchor="rt",
    )
    draw.text(
        scaled_point(64, 2768),
        "Positive principal mode shown; full tensor drives the shape.  Common camera and scales across both figures.",
        font=font(33),
        fill="#bdbdbd",
    )
    path = out / f"{loss}-four-stages-16x9.png"
    page.save(path, dpi=(300, 300))
    preview = out / f"{loss}-four-stages-preview.png"
    page.resize((1920, 1080), Image.Resampling.LANCZOS).save(preview)
    return {
        "path": str(path.resolve()),
        "size": list(page.size),
        "sha256": sha256(path),
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    source = cherries.input(cfg.source)
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "panels").mkdir()
    scene = load_static_context()
    states = load_states(source)
    with np.load(source / "l2/mesh.npz", allow_pickle=False) as archive:
        mesh = {key: archive[key].copy() for key in archive.files}
    mesh_identity = json.loads((source / "shared-mesh-identity.json").read_text())
    assert sha256(source / "l2/mesh.npz") == mesh_identity["local_sha256"]
    for loss in ("l2", "l2-normal"):
        for key in (
            "rest_points",
            "tets",
            "active_ids",
            "skin_ids",
            "triangles",
            "target_displacement_skin",
            "skin_vertex_weights",
        ):
            digest = hashlib.sha256(
                np.ascontiguousarray(mesh[key]).tobytes()
            ).hexdigest()
            assert (
                digest == mesh_identity["remote_sources"][loss]["arrays"][key]["sha256"]
            )
    rest, tets, active_ids = mesh["rest_points"], mesh["tets"], mesh["active_ids"]
    np.testing.assert_array_equal(rest, scene.rest_points)
    np.testing.assert_array_equal(rest, scene.volume.points)
    np.testing.assert_array_equal(tets, scene.volume.cells_dict[pv.CellType.TETRA])
    np.testing.assert_array_equal(
        active_ids, np.flatnonzero(scene.volume.cell_data["ActivationMask"])
    )
    skin_ids, weights = mesh["skin_ids"], mesh["skin_vertex_weights"]
    skin_faces = np.column_stack(
        (np.full(len(mesh["triangles"]), 3), mesh["triangles"])
    ).ravel()
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    target = rest[skin_ids] + mesh["target_displacement_skin"]
    active_tets = tets[active_ids]
    rest_tet = rest[active_tets]
    dm = (rest_tet[:, 1:] - rest_tet[:, :1]).transpose(0, 2, 1)
    inverse_dm = np.linalg.inv(dm)
    camera = common_camera(scene, states)
    control_ids = np.asarray(scene.volume.cell_data["ActivationControlId"])[active_ids]
    receipts, projections = [], []
    for state in states:
        stem = f"{state['loss']}-{state['stage']}"
        LOGGER.info("Rendering %s at update %s", stem, state["step"])
        deformed = rest + state["u"]
        delta = deformed[skin_ids] - target
        rms = float(1000 * np.sqrt(np.sum(weights * np.sum(delta * delta, axis=1))))
        np.testing.assert_allclose(rms, float(state["trace"]["fit_rms_mm"]), rtol=1e-10)
        state["fit_rms_mm"] = rms
        state["normal_angle_rms_deg"] = float(state["trace"]["normal_angle_rms_deg"])
        tet = deformed[active_tets]
        f = (tet[:, 1:] - tet[:, :1]).transpose(0, 2, 1) @ inverse_dm
        centers = tet.mean(axis=1)
        glyph = principal_glyphs(state["B"], f, centers)
        surface = deformed_exterior(scene, state["u"])
        expected_surface = deformed[scene.exterior_point_ids]
        np.testing.assert_array_equal(surface.points, expected_surface)
        projections.append(
            render_shape(scene, surface, camera, out / "panels" / f"{stem}-shape.png")
        )
        # vtkPoints is shared by shallow copies: assigning points would mutate
        # the reference volume and contaminate all following shape panels.
        volume = scene.volume.copy(deep=True)
        volume.points = deformed
        np.testing.assert_array_equal(scene.volume.points, rest)
        context = build_muscle_region_context(volume)
        visibility = visible_region_mask(
            context,
            centers,
            active_ids,
            control_ids,
            camera,
            window_size=ACTIVATION_WINDOW,
        )
        mask = visibility.mask & glyph.eligible & (glyph.signed_display_percent > 0)
        raw_display = glyph.signed_display_percent[mask]
        amplitude = np.sqrt(1.0 + glyph.eigenvalues_z[mask]) - 1.0
        color_fraction = log_color_fraction(amplitude)
        save_region_visibility(visibility, out / "panels" / f"{stem}-visibility.npz")
        activation_context = pv.PolyData(deformed[skin_ids], skin_faces)
        projections.append(
            render_activation(
                activation_context,
                glyph,
                mask,
                camera,
                out / "panels" / f"{stem}-activation.png",
            )
        )
        np.testing.assert_array_equal(surface.points, expected_surface)
        np.testing.assert_array_equal(activation_context.points, deformed[skin_ids])
        np.testing.assert_array_equal(scene.volume.points, rest)
        np.testing.assert_array_equal(scene.rest_points, rest)
        np.savez_compressed(
            out / "panels" / f"{stem}-rendered-surface.npz",
            points=np.asarray(surface.points),
            global_point_ids=scene.exterior_point_ids,
        )
        receipt = {k: v for k, v in state.items() if k not in {"u", "B"}}
        receipt.update(
            {
                "activation": glyph.receipt,
                "activation_visual_transfer": {
                    "raw_percent_quantiles_25_50_75_95": np.quantile(
                        raw_display, [0.25, 0.5, 0.75, 0.95]
                    ).tolist(),
                    "principal_amplitude_quantiles_25_50_75_95": np.quantile(
                        amplitude, [0.25, 0.5, 0.75, 0.95]
                    ).tolist(),
                    "principal_amplitude_max": float(amplitude.max()),
                    "color_fraction_quantiles_25_50_75_95": np.quantile(
                        color_fraction, [0.25, 0.5, 0.75, 0.95]
                    ).tolist(),
                    "rendered_line_length_mm_quantiles_25_50_75_95": (
                        4.5 * np.quantile(color_fraction, [0.25, 0.5, 0.75, 0.95])
                    ).tolist(),
                    "amplitude": "a=sqrt(1+z)-1=sigma_max(B)-1; positive principal modes only",
                    "color_formula": "color=cmap(log1p(a)/log1p(60))",
                    "length_formula": "length_m=0.0045*log1p(a)/log1p(60)",
                    "legend": "dimensionless amplitude a at zero-preserving logarithmic tick positions",
                },
                "visible_lines": int(mask.sum()),
                "visible_region_centroids": visibility.retained_count,
                "hidden_extension_like_glyphs": int(
                    np.count_nonzero(
                        visibility.mask
                        & glyph.eligible
                        & (glyph.signed_display_percent < 0)
                    )
                ),
                "fully_occluded_control_ids": list(
                    visibility.fully_occluded_control_ids
                ),
                "physical_active_inverted_cells": int((np.linalg.det(f) <= 0).sum()),
                "full_exterior_points": surface.n_points,
                "full_exterior_triangles": surface.n_cells,
                "rendered_surface_equals_rest_plus_u_exactly": True,
                "rendered_surface_displacement_error_max_m": float(
                    np.max(np.abs(np.asarray(surface.points) - expected_surface))
                ),
                "reference_volume_unchanged_after_render": True,
            }
        )
        receipts.append(receipt)
        cherries.log_metrics(
            {
                stem: {
                    "step": state["step"],
                    "fit_rms_mm": rms,
                    "visible_lines": int(mask.sum()),
                }
            }
        )
        del volume, context, visibility, glyph, surface
    for projection in projections:
        np.testing.assert_allclose(projection, projections[0], rtol=0, atol=1e-12)
    figures = [
        compose(loss, [s for s in states if s["loss"] == loss], out)
        for loss in ("l2", "l2-normal")
    ]
    summary = {
        "purpose": "Four-stage shape and principal active-strain display, one 16:9 figure per loss branch",
        "camera": camera,
        "screen_projection": projections[0].tolist(),
        "identical_screen_projection_all_16_panels": True,
        "figures": figures,
        "states": receipts,
        "anatomy_sources": [
            {"path": str(p), "sha256": sha256(p)}
            for p in (VOLUME_PATH, CRANIUM_PATH, MANDIBLE_PATH, EYES_PATH)
        ],
        "source_code": [
            {"path": str(p), "sha256": sha256(p)}
            for p in (
                Path(__file__),
                Path(__file__).with_name("shape_scene.py"),
                Path(__file__).with_name("activation_scene.py"),
                CONTEXT_SRC / "muscle_glyph_context.py",
            )
        ],
        "render": {
            "surface": "full tetrahedral volume exterior; original point IDs; exact X+u; no smoothing or remeshing",
            "static_context": "cranium, mandible and eyes in the registered reference frame",
            "shading": "flat FEM surface; smooth static anatomy",
            "activation_visibility": "front muscle-region label at each projected deformed tet centroid; principal-unique and nonneutral modes",
            "field": "principal eigenpair of B B.T - I transported by physical F",
            "color_amplitude_limits_dimensionless": [0.0, COLOR_AMPLITUDE_MAX],
            "display_mode": "positive principal contraction only; negative extension-like glyphs omitted",
            "display_colormap": "neutral to red, positive half of the reference diverging palette",
            "color_transfer": "log1p(a)/log1p(60), a=sigma_max(B)-1, shared across all eight states",
            "color_amplitude_ticks_dimensionless": list(COLOR_TICKS),
            "color_zero_policy": "log1p maps exact zero to zero; no epsilon or clipping",
            "length_transfer": "4.5 mm*log1p(a)/log1p(60); same normalized logarithmic fraction as color",
            "shared_color_and_length_normalization": True,
            "capture_date_shown": False,
            "max_line_length_m": 0.0045,
            "activation_style_reference": str(
                CONTEXT_SRC / "96-render-aligned-activation-comparison.py"
            ),
            "activation_style_reference_sha256": sha256(
                CONTEXT_SRC / "96-render-aligned-activation-comparison.py"
            ),
            "activation_native_pixels": list(ACTIVATION_WINDOW),
            "shape_native_pixels": list(WINDOW),
            "final_figure_pixels": list(PAGE),
            "resolution_policy": "fresh mesh and glyph rasterization at doubled dimensions; native font and legend drawing; no upscaled prior panel images",
            "activation_line_width_native_pixels": ACTIVATION_LINE_WIDTH_PX,
            "activation_context_opacity": ACTIVATION_CONTEXT_OPACITY,
            "activation_context_surface": "fitting skin at exact X+u, matching the previous clean figure",
            "activation_downsampling": "Lanczos from 3600 pixels high to the shared 2000-pixel layout panel",
            "glyph_sampling": "all eligible positive front-region-visible glyphs; no thinning or field smoothing",
            "fit_metric": "reference-surface area-weighted position RMS in mm",
            "normal_metric": "reference-surface area-weighted normal-angle RMS in degrees from matching trace row",
            "render_geometry_and_metric_arrays_equal_across_branches": True,
            "mesh_identity_receipt": str(
                (source / "shared-mesh-identity.json").resolve()
            ),
            "unused_mesh_array_difference": "active_volume_weights differs between hosts; not used in rendering or reported RMS metrics",
            "numerical_runs_changed": False,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Wrote both %s x %s figures to %s", *PAGE, out)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
