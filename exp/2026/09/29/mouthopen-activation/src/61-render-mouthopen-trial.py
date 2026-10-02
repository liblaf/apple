# ruff: noqa: C901, E402, EM101, EM102, PLR0912, PLR0915, TRY003
"""Render the contact-off MouthOpen continuation and optional fitted field."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
STRESS_SRC = ROOT / "exp/2026/09/21/stress-activation-loss/src"
CONTEXT_SRC = ROOT / "exp/2026/09/14/dominant-activation-ablation/src"
sys.path.extend((str(STRESS_SRC), str(CONTEXT_SRC)))

from activation_scene import contraction_colormap, principal_glyphs
from experiment import Profile
from shape_scene import CRANIUM_PATH, EYES_PATH, MANDIBLE_PATH

spec = importlib.util.spec_from_file_location(
    "mouthopen_figure_helpers", STRESS_SRC / "52-render-four-stage-figures.py"
)
assert spec is not None
assert spec.loader is not None
FIG = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = FIG
spec.loader.exec_module(FIG)

WINDOW = (1800, 1460)
BACKGROUND = "#f4f2ed"
FONT_DIR = Path("/usr/share/fonts/TTF")


class Config(cherries.BaseConfig):
    forward: Path = Path("49-forward-contact-off")
    fixture: Path = Path("30-pruned-fixture")
    pose_source: Path = Path("10-mandible/prepared.npz")
    fit: Path | None = None
    output: Path = Path("61-mouthopen-jaw-only")


def record(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def metric_text(value: Any, unit: str) -> str:
    if value == "n/a":
        return f"n/a {unit}".strip()
    return f"{float(value):.3f} {unit}".strip()


def rigid(points: np.ndarray, pivot: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (
        (points - pivot) @ Rotation.from_rotvec(pose[:3]).as_matrix().T
        + pivot
        + pose[3:]
    )


def terminal_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    status = str(summary.get("status", "unknown"))
    if status in {"running", "initializing", "unknown"}:
        raise RuntimeError(f"input run is not terminal: {path} status={status!r}")
    return summary


def camera_for(
    cranium: pv.PolyData,
    eyes: pv.PolyData,
    skins: list[np.ndarray],
    jaws: list[np.ndarray],
) -> dict[str, Any]:
    toward = np.array([0.65, 0.03, 1.0], dtype=np.float64)
    toward /= np.linalg.norm(toward)
    right = np.cross([0.0, 1.0, 0.0], toward)
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    basis = np.column_stack((right, up, toward))
    point_sets = [
        np.asarray(cranium.points),
        np.asarray(eyes.points),
        *jaws,
        *skins,
    ]
    projected = np.vstack(point_sets) @ basis
    low, high = projected.min(axis=0), projected.max(axis=0)
    center = basis @ ((low + high) / 2.0)
    half = (high - low) / 2.0
    return {
        "position": (center + 0.6 * toward).tolist(),
        "focal_point": center.tolist(),
        "view_up": up.tolist(),
        "parallel_scale": float(
            1.055 * max(half[1], half[0] / (WINDOW[0] / WINDOW[1]))
        ),
        "projected_bounds_m": [low.tolist(), high.tolist()],
        "window_size": list(WINDOW),
        "policy": "shared orthographic camera fitted to every rendered skin and jaw pose",
    }


def render_panel(
    output: Path,
    camera: dict[str, Any],
    cranium: pv.PolyData,
    eyes: pv.PolyData,
    jaw: pv.PolyData,
    skin: pv.PolyData,
    *,
    glyph: Any | None = None,
    glyph_mask: np.ndarray | None = None,
    amplitude_cap: float = 60.0,
) -> None:
    plot = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="three lights")
    plot.set_background(BACKGROUND)
    plot.ren_win.SetMultiSamples(8)
    plot.add_mesh(cranium, color="#e6dfcf", smooth_shading=True)
    plot.add_mesh(jaw, color="#e6dfcf", smooth_shading=True)
    plot.add_mesh(eyes, color="#fffdf8", smooth_shading=True, ambient=0.3, diffuse=0.7)
    plot.add_mesh(
        skin,
        color="#8d969b",
        smooth_shading=False,
        opacity=0.24 if glyph is not None else 1.0,
    )
    if glyph is not None and glyph_mask is not None and glyph_mask.any():
        amplitude = np.sqrt(1.0 + glyph.eigenvalues_z[glyph_mask]) - 1.0
        color_fraction = np.log1p(amplitude) / np.log1p(amplitude_cap)
        half = 0.00225 * color_fraction[:, None] * glyph.spatial_axes[glyph_mask]
        endpoints = np.stack(
            (glyph.centers[glyph_mask] - half, glyph.centers[glyph_mask] + half),
            axis=1,
        )
        count = len(endpoints)
        lines = np.column_stack(
            (np.full(count, 2), 2 * np.arange(count), 2 * np.arange(count) + 1)
        ).ravel()
        line_mesh = pv.PolyData(endpoints.reshape(-1, 3), lines=lines)
        line_mesh.cell_data["LogActivationColor"] = color_fraction
        plot.add_mesh(
            line_mesh,
            scalars="LogActivationColor",
            preference="cell",
            cmap=contraction_colormap(),
            clim=(0.0, 1.0),
            lighting=False,
            line_width=2.0,
            render_lines_as_tubes=False,
            show_scalar_bar=False,
        )
    plot.camera.position = camera["position"]
    plot.camera.focal_point = camera["focal_point"]
    plot.camera.up = camera["view_up"]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = camera["parallel_scale"]
    plot.reset_camera_clipping_range()
    plot.enable_anti_aliasing("ssaa")
    plot.screenshot(output)
    plot.close()


def render_activation_panel(
    output: Path,
    camera: dict[str, Any],
    skin: pv.PolyData,
    glyph: Any,
    glyph_mask: np.ndarray,
    amplitude_cap: float,
) -> None:
    """Render positive principal modes over a subdued fitted-skin context."""
    plot = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="three lights")
    plot.set_background(BACKGROUND)
    plot.ren_win.SetMultiSamples(8)
    plot.add_mesh(skin, color="#8d969b", opacity=0.16, smooth_shading=False)
    if glyph_mask.any():
        amplitude = np.sqrt(1.0 + glyph.eigenvalues_z[glyph_mask]) - 1.0
        color_fraction = np.log1p(amplitude) / np.log1p(amplitude_cap)
        half = 0.00225 * color_fraction[:, None] * glyph.spatial_axes[glyph_mask]
        endpoints = np.stack(
            (glyph.centers[glyph_mask] - half, glyph.centers[glyph_mask] + half),
            axis=1,
        )
        count = len(endpoints)
        lines = np.column_stack(
            (np.full(count, 2), 2 * np.arange(count), 2 * np.arange(count) + 1)
        ).ravel()
        line_mesh = pv.PolyData(endpoints.reshape(-1, 3), lines=lines)
        line_mesh.cell_data["LogActivationColor"] = color_fraction
        plot.add_mesh(
            line_mesh,
            scalars="LogActivationColor",
            preference="cell",
            cmap=contraction_colormap(),
            clim=(0.0, 1.0),
            lighting=False,
            line_width=2.0,
            render_lines_as_tubes=False,
            show_scalar_bar=False,
        )
    plot.camera.position = camera["position"]
    plot.camera.focal_point = camera["focal_point"]
    plot.camera.up = camera["view_up"]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = camera["parallel_scale"]
    plot.reset_camera_clipping_range()
    plot.enable_anti_aliasing("ssaa")
    plot.screenshot(output)
    plot.close()


def render_direction_panel(
    output: Path,
    camera: dict[str, Any],
    skin: pv.PolyData,
    glyph: Any,
    selected: np.ndarray,
    amplitude_cap: float,
) -> None:
    """Render a spatially sampled direction view with fixed-length lines."""
    plot = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="three lights")
    plot.set_background(BACKGROUND)
    plot.ren_win.SetMultiSamples(8)
    plot.add_mesh(skin, color="#8d969b", opacity=0.13, smooth_shading=False)
    if selected.any():
        amplitude = np.sqrt(1.0 + glyph.eigenvalues_z[selected]) - 1.0
        color_fraction = np.log1p(amplitude) / np.log1p(amplitude_cap)
        half = 0.001 * glyph.spatial_axes[selected]
        endpoints = np.stack(
            (glyph.centers[selected] - half, glyph.centers[selected] + half), axis=1
        )
        count = len(endpoints)
        lines = np.column_stack(
            (np.full(count, 2), 2 * np.arange(count), 2 * np.arange(count) + 1)
        ).ravel()
        line_mesh = pv.PolyData(endpoints.reshape(-1, 3), lines=lines)
        line_mesh.cell_data["LogActivationColor"] = color_fraction
        plot.add_mesh(
            line_mesh,
            scalars="LogActivationColor",
            preference="cell",
            cmap=contraction_colormap(),
            clim=(0.0, 1.0),
            lighting=False,
            line_width=3.0,
            render_lines_as_tubes=False,
            show_scalar_bar=False,
        )
    plot.camera.position = camera["position"]
    plot.camera.focal_point = camera["focal_point"]
    plot.camera.up = camera["view_up"]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = camera["parallel_scale"]
    plot.reset_camera_clipping_range()
    plot.enable_anti_aliasing("ssaa")
    plot.screenshot(output)
    plot.close()


def sample_spatial_bins(
    glyph: Any,
    candidates: np.ndarray,
    active_cell_ids: np.ndarray,
    bin_size_m: float,
) -> np.ndarray:
    """Choose the lowest active cell ID from each orthographic screen bin."""
    if bin_size_m <= 0:
        raise ValueError("projected bin size must be positive")
    toward = np.array([0.65, 0.03, 1.0], dtype=np.float64)
    toward /= np.linalg.norm(toward)
    right = np.cross([0.0, 1.0, 0.0], toward)
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    basis = np.column_stack((right, up, toward))
    xy = glyph.centers[candidates] @ basis[:, :2]
    bins = np.floor(xy / bin_size_m).astype(np.int64)
    order = np.argsort(active_cell_ids[candidates], kind="stable")
    chosen: dict[tuple[int, int], int] = {}
    for candidate_index in order:
        key = (int(bins[candidate_index, 0]), int(bins[candidate_index, 1]))
        chosen.setdefault(key, int(candidates[candidate_index]))
    return np.asarray(list(chosen.values()), dtype=np.int64)


def compose_direction_review(
    panel_path: Path,
    output: Path,
    *,
    selected_count: int,
    bin_size_m: float,
    cell_id_sha256: str,
    amplitude_cap: float,
) -> tuple[Path, Path]:
    """Add a readable caption and color scale to the sampled field panel."""
    panel = Image.open(panel_path).convert("RGB")
    canvas = Image.new("RGB", (WINDOW[0], WINDOW[1] + 270), "#101315")
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 36)
    small = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 23)
    draw.text(
        (WINDOW[0] // 2, 12),
        "Fitted principal directions · spatially sampled view",
        font=heading,
        fill="white",
        anchor="mt",
    )
    draw.text(
        (WINDOW[0] // 2, 58),
        f"One visible positive mode per {bin_size_m * 1000:.1f} mm projected bin · fixed 2 mm line length · {selected_count} lines",
        font=small,
        fill="#d7dfe2",
        anchor="mt",
    )
    canvas.paste(panel, (0, 92))
    ramp = contraction_colormap()(np.linspace(0.0, 1.0, 450), bytes=True)[:, :3]
    bar = Image.fromarray(np.tile(ramp[None, :, :], (24, 1, 1)))
    bar_left = WINDOW[0] - 570
    bar_top = WINDOW[1] + 132
    canvas.paste(bar, (bar_left, bar_top))
    draw.text(
        (35, WINDOW[1] + 126),
        "Color = amplitude a (shared log scale)",
        font=small,
        fill="#d7dfe2",
    )
    draw.text((bar_left, bar_top + 29), "0", font=small, fill="#d7dfe2")
    draw.text(
        (bar_left + 450, bar_top + 29),
        f"{amplitude_cap:g}",
        font=small,
        fill="#d7dfe2",
        anchor="ra",
    )
    draw.text(
        (35, WINDOW[1] + 190),
        "Selection rule: one visible positive mode per occupied projected bin; lowest active cell ID wins; no direction filtering.",
        font=small,
        fill="#bdc7cb",
    )
    draw.text(
        (35, WINDOW[1] + 232),
        f"Selected cell-ID list SHA-256: {cell_id_sha256}",
        font=small,
        fill="#bdc7cb",
    )
    figure = output / "mouthopen-principal-directions.png"
    preview = output / "mouthopen-principal-directions-preview.png"
    canvas.save(figure)
    canvas.resize(
        (1920, round(1920 * canvas.height / canvas.width)), Image.Resampling.LANCZOS
    ).save(preview)
    panel.close()
    return figure, preview


def load_fit(
    path: Path, volume: pv.UnstructuredGrid
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = path / "summary.json"
    summary = terminal_summary(summary_path)
    checkpoint_path = path / "last.npz"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"fit checkpoint is missing: {checkpoint_path}")
    checkpoint_receipt = summary.get("final_checkpoint")
    if (
        checkpoint_receipt
        and checkpoint_receipt.get("sha256")
        and record(checkpoint_path)["sha256"] != checkpoint_receipt["sha256"]
    ):
        raise ValueError("fit checkpoint does not match summary receipt")
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        required = {"u", "B", "activation_model", "mode"}
        if not required.issubset(archive.files):
            raise ValueError(
                f"fit checkpoint missing keys: {sorted(required - set(archive.files))}"
            )
        state = {key: archive[key].copy() for key in archive.files}
    if str(state["activation_model"]) != "strain":
        raise ValueError("fit checkpoint is not an active-strain state")
    if str(state["mode"]) != "psd6":
        raise ValueError("fit checkpoint is not a full-S PSD6 state")
    if state["u"].shape != (volume.n_points, 3):
        raise ValueError("fit displacement does not match the rendering fixture")
    if state["B"].ndim != 3 or state["B"].shape[1:] != (3, 3):
        raise ValueError("fit tensor B must have shape (active cells, 3, 3)")
    return {
        "summary": summary,
        "summary_path": summary_path,
        "checkpoint_path": checkpoint_path,
    }, state


def compose(
    output: Path,
    panels: list[dict[str, str]],
    fit_lines: int | None,
    note: str,
    *,
    fit_metrics: dict[str, Any] | None = None,
    amplitude_cap: float | None = None,
) -> tuple[Path, Path]:
    has_fit = fit_lines is not None
    columns = 3 if has_fit else len(panels)
    page_height = 3900 if has_fit else 2030
    page = Image.new("RGB", (columns * 1840, page_height), "#101315")
    draw = ImageDraw.Draw(page)
    title = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 80)
    heading = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 42)
    small = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 29)
    draw.text(
        (page.width // 2, 22),
        "MouthOpen: exploratory contact-off continuation",
        font=title,
        fill="white",
        anchor="mt",
    )
    top_panels = panels[:3]
    for i, panel in enumerate(top_panels):
        left = i * 1840 + 20
        center = left + 900
        draw.text(
            (center, 130), panel["heading"], font=heading, fill="white", anchor="mt"
        )
        draw.text(
            (center, 186), panel["subtitle"], font=small, fill="#bdc7cb", anchor="mt"
        )
        with Image.open(output / panel["file"]) as image:
            if image.size != WINDOW:
                raise ValueError(f"unexpected panel size {image.size}")
            page.paste(image, (left, 240))
    if has_fit:
        for index, panel in enumerate(panels[3:5]):
            left = index * 1840 + 20
            center = left + 900
            draw.text(
                (center, 1775),
                panel["heading"],
                font=heading,
                fill="white",
                anchor="mt",
            )
            draw.text(
                (center, 1831),
                panel["subtitle"],
                font=small,
                fill="#bdc7cb",
                anchor="mt",
            )
            with Image.open(output / panel["file"]) as image:
                if image.size != WINDOW:
                    raise ValueError(f"unexpected panel size {image.size}")
                page.paste(image, (left, 1885))
        x = 2 * 1840 + 55
        draw.text((x, 1800), "Fit and activation scale", font=heading, fill="white")
        details = [
            fit_metrics.get("status_caption", ""),
            f"Fit RMS: {metric_text(fit_metrics.get('fit_rms_mm', 'n/a'), 'mm')}",
            f"Normal RMS: {metric_text(fit_metrics.get('normal_angle_rms_deg', 'n/a'), '°')}",
            f"Optimizer updates: {fit_metrics.get('optimizer_updates', 'n/a')}",
            f"Solver valid: {fit_metrics.get('solver_valid', 'n/a')}",
            f"Visible positive principal lines: {fit_lines}",
            "Amplitude a = sigma_max(B) - 1 (dimensionless)",
            f"Line length: 4.5 mm x log(1+a) / log(1+{amplitude_cap:g})",
        ]
        for row, text in enumerate(details):
            draw.text((x, 1875 + row * 43), text, font=small, fill="#d7dfe2")
        assert amplitude_cap is not None
        assert amplitude_cap > 0
        bar_x, bar_y, bar_width = x, 2220, 1500
        ramp = contraction_colormap()(np.linspace(0.0, 1.0, bar_width), bytes=True)[
            :, :3
        ]
        bar = Image.fromarray(np.tile(ramp[None, :, :], (40, 1, 1)))
        page.paste(bar, (bar_x, bar_y))
        ticks = [0.0, 1.0, 3.0, 10.0, 30.0, 60.0]
        if amplitude_cap > 60.0:
            ticks.append(amplitude_cap)
        for value in ticks:
            if value > amplitude_cap:
                continue
            position = np.log1p(value) / np.log1p(amplitude_cap)
            tick_x = round(bar_x + position * bar_width)
            draw.line((tick_x, bar_y + 40, tick_x, bar_y + 51), fill="#dfdfdf", width=2)
            draw.text(
                (tick_x, bar_y + 58),
                f"{value:g}",
                font=small,
                fill="#dfdfdf",
                anchor="mt",
            )
        draw.text(
            (bar_x, bar_y + 112), "Red: contraction-like", font=small, fill="#d6604d"
        )
        draw.text((35, 3455), note, font=small, fill="#d7dfe2")
        draw.text(
            (35, 3510),
            "No contact forces were modeled. Finite inversions or detected boundary intersections leave this result mechanically unvalidated.",
            font=small,
            fill="#f1cc83",
        )
        draw.text(
            (35, 3565),
            "No bone-contact obstacle or containment test is included.",
            font=small,
            fill="#bdc7cb",
        )
    else:
        draw.text((35, 1745), note, font=small, fill="#d7dfe2")
        draw.text(
            (35, 1800),
            "No contact forces were modeled. Finite inversions or detected boundary intersections leave this result mechanically unvalidated.",
            font=small,
            fill="#f1cc83",
        )
        draw.text(
            (35, 1855),
            "No bone-contact obstacle or containment test is included.",
            font=small,
            fill="#bdc7cb",
        )
    figure = output / "mouthopen-trial.png"
    preview = output / "mouthopen-trial-preview.png"
    page.save(figure)
    page.resize(
        (1920, round(1920 * page.height / page.width)), Image.Resampling.LANCZOS
    ).save(preview)
    return figure, preview


def main(cfg: Config) -> None:
    fixture = cherries.input(cfg.fixture)
    forward = cherries.input(cfg.forward)
    pose_source = cherries.input(cfg.pose_source)
    fit_path = cherries.input(cfg.fit) if cfg.fit is not None else None
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "panels").mkdir()

    forward_summary_path = forward / "summary.json"
    forward_checkpoint_path = forward / "final.npz"
    forward_summary = terminal_summary(forward_summary_path)
    if not forward_checkpoint_path.is_file():
        raise FileNotFoundError("contact-off continuation has no final checkpoint")
    forward_receipt = forward_summary.get("final_checkpoint", {})
    if (
        forward_receipt.get("sha256")
        and record(forward_checkpoint_path)["sha256"] != forward_receipt["sha256"]
    ):
        raise ValueError("contact-off checkpoint does not match summary receipt")
    with np.load(forward_checkpoint_path, allow_pickle=False) as archive:
        required = {"displacement", "pose", "fraction"}
        if not required.issubset(archive.files):
            raise ValueError(
                f"forward checkpoint missing keys: {sorted(required - set(archive.files))}"
            )
        final_u = np.asarray(archive["displacement"], dtype=np.float64).copy()
        final_pose = np.asarray(archive["pose"], dtype=np.float64).copy()
        fraction = float(archive["fraction"])
    if not np.isfinite(final_u).all() or not np.isfinite(final_pose).all():
        raise ValueError("contact-off checkpoint contains nonfinite arrays")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("contact-off pose fraction is outside [0, 1]")

    with np.load(pose_source, allow_pickle=False) as archive:
        prepared = {key: archive[key].copy() for key in archive.files}
    full_pose = np.asarray(prepared["pose"], dtype=np.float64)
    pivot = np.asarray(prepared["pivot"], dtype=np.float64)
    np.testing.assert_allclose(final_pose, fraction * full_pose, rtol=0, atol=1e-12)

    volume = pv.read(fixture / "volume.vtu")
    skin_mesh = pv.read(fixture / "skin.vtp")
    points = np.asarray(volume.points, dtype=np.float64)
    skin_ids = np.asarray(skin_mesh.point_data["GlobalPointId"], dtype=np.int64)
    target_skin = np.asarray(prepared["target_skin"], dtype=np.float64)
    if final_u.shape != points.shape or target_skin.shape != skin_mesh.points.shape:
        raise ValueError("forward or target skin shape does not match the fixture")
    np.testing.assert_array_equal(skin_mesh.points, points[skin_ids])

    cranium = pv.read(CRANIUM_PATH)
    eyes = pv.read(EYES_PATH)
    mandible = pv.read(MANDIBLE_PATH)
    neutral_jaw = mandible.copy(deep=True)
    target_jaw = mandible.copy(deep=True)
    target_jaw.points = rigid(np.asarray(mandible.points), pivot, full_pose)
    final_jaw = mandible.copy(deep=True)
    # The checkpoint pose already includes its accepted fraction: apply it once.
    final_jaw.points = rigid(np.asarray(mandible.points), pivot, final_pose)

    states = [
        {
            "label": "neutral",
            "skin": np.asarray(skin_mesh.points).copy(),
            "jaw": neutral_jaw,
        },
        {"label": "target", "skin": target_skin, "jaw": target_jaw},
        {
            "label": "contact-off",
            "skin": points[skin_ids] + final_u[skin_ids],
            "jaw": final_jaw,
        },
    ]
    fit_receipt: dict[str, Any] | None = None
    fit_state: dict[str, Any] | None = None
    if fit_path is not None:
        fit_receipt, fit_state = load_fit(fit_path, volume)
        fit_jaw = mandible.copy(deep=True)
        fit_jaw.points = rigid(np.asarray(mandible.points), pivot, full_pose)
        states.append(
            {
                "label": "fitted",
                "skin": points[skin_ids] + fit_state["u"][skin_ids],
                "jaw": fit_jaw,
            }
        )

    camera = camera_for(
        cranium,
        eyes,
        [state["skin"] for state in states],
        [np.asarray(state["jaw"].points) for state in states],
    )
    panel_records: list[dict[str, str]] = []
    for index, state in enumerate(states[:3]):
        path = output / "panels" / f"shape-{state['label']}.png"
        skin = pv.PolyData(
            np.asarray(state["skin"]), np.asarray(skin_mesh.faces).copy()
        )
        render_panel(path, camera, cranium, eyes, state["jaw"], skin)
        if index == 0:
            heading, subtitle = "Historical neutral", "Source skin and neutral jaw"
        elif index == 1:
            heading, subtitle = (
                "Full MouthOpen target",
                "Transferred blendshape; full chin-derived jaw pose",
            )
        else:
            inverted = int(forward_summary.get("final", {}).get("inverted_cells", -1))
            inv_text = (
                f"{inverted} inverted tets"
                if inverted >= 0
                else "inversions reported in manifest"
            )
            heading = "49 contact-off forward"
            subtitle = f"Jaw {fraction:.1%} pose · S=0 · {inv_text}"
        panel_records.append(
            {"file": f"panels/{path.name}", "heading": heading, "subtitle": subtitle}
        )

    fit_glyph_receipt: dict[str, Any] | None = None
    direction_review: dict[str, Any] | None = None
    fit_lines: int | None = None
    fit_inverted: int | None = None
    fit_metrics: dict[str, Any] | None = None
    amplitude_cap: float | None = None
    if fit_state is not None:
        fit_inverted = int(
            fit_receipt["summary"].get("last_metrics", {}).get("inverted_all_cells", -1)
        )
        active_ids = np.flatnonzero(
            np.asarray(volume.cell_data["ActivationMask"], dtype=bool)
        )
        if len(active_ids) != len(fit_state["B"]):
            raise ValueError(
                "fit tensor count does not match the fixture activation mask"
            )
        tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
        active_tets = tets[active_ids]
        deformed_points = points + fit_state["u"]
        rest_tet = points[active_tets]
        deformed_tet = deformed_points[active_tets]
        dm = (rest_tet[:, 1:] - rest_tet[:, :1]).transpose(0, 2, 1)
        f = (deformed_tet[:, 1:] - deformed_tet[:, :1]).transpose(
            0, 2, 1
        ) @ np.linalg.inv(dm)
        glyph = principal_glyphs(fit_state["B"], f, deformed_tet.mean(axis=1))
        positive = glyph.eligible & (glyph.eigenvalues_z > 0)
        amplitudes = np.sqrt(1.0 + glyph.eigenvalues_z[positive]) - 1.0
        amplitude_cap = max(60.0, float(amplitudes.max()) if len(amplitudes) else 0.0)
        volume.points = deformed_points
        glyph_context = FIG.build_muscle_region_context(volume)
        visibility = FIG.visible_region_mask(
            glyph_context,
            glyph.centers,
            active_ids,
            np.asarray(volume.cell_data["ActivationControlId"])[active_ids],
            camera,
            window_size=FIG.ACTIVATION_WINDOW,
        )
        glyph_mask = visibility.mask & positive
        fit_lines = int(glyph_mask.sum())
        candidates = np.flatnonzero(glyph_mask)
        sample_bin_size_m = 0.0018
        selected_glyph_indices = sample_spatial_bins(
            glyph, candidates, active_ids, sample_bin_size_m
        )
        selected_cell_ids = np.asarray(active_ids[selected_glyph_indices], dtype="<i8")
        selected_cell_ids_sha256 = hashlib.sha256(
            selected_cell_ids.tobytes()
        ).hexdigest()
        shape_path = output / "panels" / "shape-fitted.png"
        fit_skin = pv.PolyData(
            np.asarray(states[3]["skin"]), np.asarray(skin_mesh.faces).copy()
        )
        render_panel(
            shape_path,
            camera,
            cranium,
            eyes,
            states[3]["jaw"],
            fit_skin,
        )
        activation_path = output / "panels" / "activation-fitted.png"
        render_activation_panel(
            activation_path, camera, fit_skin, glyph, glyph_mask, amplitude_cap
        )
        direction_panel_path = output / "panels" / "directions-sampled.png"
        render_direction_panel(
            direction_panel_path,
            camera,
            fit_skin,
            glyph,
            selected_glyph_indices,
            amplitude_cap,
        )
        direction_figure, direction_preview = compose_direction_review(
            direction_panel_path,
            output,
            selected_count=len(selected_glyph_indices),
            bin_size_m=sample_bin_size_m,
            cell_id_sha256=selected_cell_ids_sha256,
            amplitude_cap=amplitude_cap,
        )
        direction_review = {
            "figure": record(direction_figure),
            "preview": record(direction_preview),
            "panel": record(direction_panel_path),
            "selection_rule": "one visible positive principal mode per orthographic projected bin; choose the smallest active cell ID in each occupied bin; no selection based on apparent direction",
            "bin_size_projected_m": sample_bin_size_m,
            "fixed_line_length_m": 0.002,
            "selected_count": len(selected_glyph_indices),
            "selected_active_cell_ids_sha256": selected_cell_ids_sha256,
            "selected_active_cell_ids": selected_cell_ids.tolist(),
            "amplitude_color_cap": amplitude_cap,
        }
        fit_glyph_receipt = {
            **glyph.receipt,
            "positive_modes": int(positive.sum()),
            "visible_positive_modes": fit_lines,
            "shared_amplitude_cap": amplitude_cap,
            "definition": "a = sigma_max(B) - 1; line direction is normalized F n",
        }
        last_metrics = fit_receipt["summary"].get("last_metrics", {})
        optimizer_updates = fit_receipt["summary"].get("optimizer_updates", "n/a")
        attempted_updates = fit_receipt["summary"].get("attempted_updates", "n/a")
        fit_metrics = {
            "fit_rms_mm": last_metrics.get("fit_rms_mm", "n/a"),
            "normal_angle_rms_deg": last_metrics.get("normal_angle_rms_deg", "n/a"),
            "optimizer_updates": optimizer_updates,
            "attempted_updates": attempted_updates,
            "status_caption": (
                f"Interim checkpoint · attempt {attempted_updates}/200"
                if fit_receipt["summary"].get("is_interim_only")
                else f"Terminal run · {fit_receipt['summary']['status']}"
            ),
            "solver_valid": bool(fit_state.get("solver_valid", False)),
        }
        panel_records.extend(
            [
                {
                    "file": f"panels/{shape_path.name}",
                    "heading": "Activation fit geometry",
                    "subtitle": (
                        f"{fit_metrics['status_caption']} · Full jaw pose · PSD6 strain · {optimizer_updates} updates "
                        f"from {attempted_updates} attempts · "
                        f"{fit_inverted if fit_inverted >= 0 else 'unknown'} inverted tets"
                    ),
                },
                {
                    "file": f"panels/{activation_path.name}",
                    "heading": "Fitted principal activation",
                    "subtitle": f"{fit_metrics['status_caption']} · Fitted skin context · {fit_lines} visible positive modes",
                },
            ]
        )

    inverted = int(forward_summary.get("final", {}).get("inverted_cells", -1))
    full_boundary_intersections = bool(
        forward_summary.get("final", {}).get("has_intersections", False)
    )
    intersection_text = (
        "Complete FEM boundary audit detected self-intersections."
        if full_boundary_intersections
        else "Complete FEM boundary audit reported no self-intersections."
    )
    note = (
        f"Contact-off run status: {forward_summary['status']}; accepted jaw pose fraction {fraction:.6f}; "
        f"finite inverted cells at endpoint: {inverted if inverted >= 0 else 'see run summary'}. "
        f"{intersection_text}"
    )
    figure, preview = compose(
        output,
        panel_records,
        fit_lines,
        note,
        fit_metrics=fit_metrics,
        amplitude_cap=amplitude_cap,
    )
    source_paths = [
        Path(__file__),
        STRESS_SRC / "shape_scene.py",
        STRESS_SRC / "activation_scene.py",
        STRESS_SRC / "52-render-four-stage-figures.py",
        CONTEXT_SRC / "muscle_glyph_context.py",
        CRANIUM_PATH,
        MANDIBLE_PATH,
        EYES_PATH,
    ]
    input_paths = [
        fixture / "summary.json",
        fixture / "volume.vtu",
        fixture / "skin.vtp",
        fixture / "mapping.npz",
        pose_source,
        forward_summary_path,
        forward_checkpoint_path,
    ]
    if fit_receipt is not None:
        input_paths.extend(
            [fit_receipt["summary_path"], fit_receipt["checkpoint_path"]]
        )
    manifest = {
        "schema": "mouthopen-trial-render-v1",
        "forward_status": forward_summary["status"],
        "forward_completed_pose_fraction": fraction,
        "forward_final_inverted_cells": inverted,
        "forward_full_boundary_has_intersections": full_boundary_intersections,
        "forward_physical_validity_claim": False,
        "fit_status": None if fit_receipt is None else fit_receipt["summary"]["status"],
        "fit_is_interim_only": (
            None
            if fit_receipt is None
            else bool(fit_receipt["summary"].get("is_interim_only", False))
        ),
        "fit_inverted_cells": fit_inverted,
        "inputs": {str(path.resolve()): record(path) for path in input_paths},
        "sources": [record(path) for path in source_paths],
        "camera": camera,
        "panels": panel_records,
        "fitted_activation_field": fit_glyph_receipt,
        "direction_review": direction_review,
        "jaw_pose_semantics": {
            "full_target": "prepared full pose applied directly",
            "contact_off_endpoint": "checkpoint pose applied directly; it already encodes the accepted fraction",
            "fitted_state": "prepared full pose applied directly; the fit holds the full jaw target fixed",
        },
        "model_scope": "exploratory contact-off bulk result; no contact forces, interfaces not mechanically validated, finite inverted cells may remain",
        "figure": record(figure),
        "preview": record(preview),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "source.py").write_text(Path(__file__).read_text())
    cherries.log_metrics(
        {
            "completed_pose_fraction": fraction,
            "inverted_cells": inverted,
            "fitted_visible_principal_lines": 0 if fit_lines is None else fit_lines,
            "fitted_sampled_direction_lines": (
                0 if direction_review is None else direction_review["selected_count"]
            ),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
