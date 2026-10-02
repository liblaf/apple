# Copyright (c) 2026 liblaf
"""Render the fixed-axis parent and three released-axis smoothness endpoints."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
REPO = GROUP.parents[4]
HISTORICAL = REPO / "exp/2026/09/21/stress-activation-loss"
HISTORICAL_SRC = HISTORICAL / "src"
sys.path.insert(0, str(HISTORICAL_SRC))

from activation_scene import contraction_colormap, principal_glyphs  # noqa: E402
from shape_scene import (  # noqa: E402
    CRANIUM_PATH,
    EYES_PATH,
    MANDIBLE_PATH,
    VOLUME_PATH,
    deformed_exterior,
    load_static_context,
)

spec = importlib.util.spec_from_file_location(
    "historical_figure_52", HISTORICAL_SRC / "52-render-four-stage-figures.py"
)
assert spec is not None
assert spec.loader is not None
FIG = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = FIG
spec.loader.exec_module(FIG)

LOGGER = logging.getLogger(__name__)
MULTIPLIERS = (1, 3, 10)
HEADINGS = (
    "Fixed axis\nparent",
    "Released axis\n1x",
    "Released axis\n3x",
    "Released axis\n10x",
)


class Config(cherries.BaseConfig):
    source: Path = Path("10-sweep")
    output: Path = Path("30-comparison")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(path: Path, mode: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        assert str(archive["activation_model"]) == "strain"
        assert str(archive["mode"]) == mode
        assert int(archive["step"]) == 200
        return {
            "u": archive["u"].copy(),
            "B": archive["B"].copy(),
            "step": int(archive["step"]),
            "solver_valid": bool(archive["solver_valid"]),
            "checkpoint": {"path": str(path.resolve()), "sha256": sha256(path)},
        }


def trace_row(path: Path, step: int) -> dict:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["step"]) == step]
    assert len(rows) == 1, (path, step)
    return rows[0]


def states_from(source: Path, protocol: dict) -> list[dict]:
    parent_path = Path(protocol["parent_checkpoint"]["path"])
    assert sha256(parent_path) == protocol["parent_checkpoint"]["sha256"]
    parent = checkpoint(parent_path, "rankone_fixed")
    historical_trace = (
        HISTORICAL
        / "data/51-visualization-checkpoints-002/l2-normal/l2-normal-rankone_fixed/trace.csv"
    )
    parent.update(
        label="parent",
        multiplier=None,
        smooth_weight=protocol["base_smooth_weight"],
        trace=trace_row(historical_trace, 200),
        trace_path=str(historical_trace.resolve()),
        trace_sha256=sha256(historical_trace),
    )
    states = [parent]
    for multiplier in MULTIPLIERS:
        folder = source / f"multiplier-{multiplier}"
        branch = json.loads((folder / "protocol.json").read_text())
        assert branch["multiplier"] == multiplier
        assert branch["steps"] == 200
        assert branch["parent_checkpoint"]["sha256"] == parent["checkpoint"]["sha256"]
        assert branch["smooth_weight"] == protocol["base_smooth_weight"] * multiplier
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["status"] == "completed_budget_not_convergence_certified"
        assert summary["last_step"] == 200
        path = folder / "stage/last.npz"
        state = checkpoint(path, "rankone_learned")
        trace = folder / "stage/trace.csv"
        state.update(
            label=f"multiplier-{multiplier}",
            multiplier=multiplier,
            smooth_weight=branch["smooth_weight"],
            trace=trace_row(trace, 200),
            trace_path=str(trace.resolve()),
            trace_sha256=sha256(trace),
        )
        states.append(state)
    return states


def compose(states: list[dict], output: Path, cap: float) -> dict:
    page = Image.new("RGB", FIG.PAGE, "#000000")
    draw = ImageDraw.Draw(page)
    point = FIG.scaled_point
    draw.text(
        point(FIG.PAGE_LAYOUT[0] / 2, 38),
        "Released-axis smoothness continuation",
        font=FIG.font(88, bold=True),
        fill="white",
        anchor="mt",
    )
    draw.text(
        point(FIG.PAGE_LAYOUT[0] / 2, 160),
        "One fixed-axis starting point · 200 released-axis updates per branch",
        font=FIG.font(40),
        fill="#d9d9d9",
        anchor="mt",
    )
    for index, state in enumerate(states):
        x = 64 + index * (FIG.PANEL_LAYOUT[0] + 24)
        center = x + FIG.PANEL_LAYOUT[0] / 2
        draw.multiline_text(
            point(center, 250),
            HEADINGS[index],
            font=FIG.font(63, bold=True),
            fill="white",
            anchor="ma",
            align="center",
            spacing=5 * FIG.RESOLUTION_SCALE,
        )
        draw.text(
            point(center, 408),
            f"λ = {state['smooth_weight']:.2g}  ·  update 200",
            font=FIG.font(38),
            fill="#dfdfdf",
            anchor="mt",
        )
        draw.text(
            point(center, 476),
            f"Fit {state['fit_rms_mm']:.3f} mm  ·  normal {state['normal_angle_rms_deg']:.2f}°",
            font=FIG.font(37),
            fill="white",
            anchor="mt",
        )
        for row, y in (("shape", 550), ("activation", 1576)):
            with Image.open(output / "panels" / f"{state['label']}-{row}.png") as panel:
                assert panel.size == FIG.WINDOW
                page.paste(panel.convert("RGB"), point(x, y))
    draw.text(
        point(64, 2605),
        "Principal activation a (dimensionless)",
        font=FIG.font(39),
        fill="white",
    )
    bar_x, bar_y, bar_w = 1710, 2613, 1700
    ramp = contraction_colormap()(
        np.linspace(0, 1, bar_w * FIG.RESOLUTION_SCALE), bytes=True
    )[:, :3]
    bar = Image.fromarray(np.tile(ramp[None, :, :], (40 * FIG.RESOLUTION_SCALE, 1, 1)))
    page.paste(bar, point(bar_x, bar_y))
    ticks = list(FIG.COLOR_TICKS)
    if cap > 60:
        ticks.append(cap)
    for value in ticks:
        x = bar_x + np.log1p(value) / np.log1p(cap) * bar_w
        draw.line(
            [point(x, bar_y + 40), point(x, bar_y + 49)],
            fill="#dfdfdf",
            width=2 * FIG.RESOLUTION_SCALE,
        )
        draw.text(
            point(x, bar_y + 53),
            f"{value:g}",
            font=FIG.font(35),
            fill="#dfdfdf",
            anchor="mt",
        )
    draw.text(
        point(64, 2680),
        "Color and length: log(1 + a) on one scale · a = sigma_max(B) - 1",
        font=FIG.font(37),
        fill="#dfdfdf",
    )
    draw.text(
        point(64, 2768),
        "Positive principal mode shown; the full tensor drives each shape. Shared camera and scale.",
        font=FIG.font(33),
        fill="#bdbdbd",
    )
    path = output / "smoothness-comparison-16x9.png"
    preview = output / "smoothness-comparison-preview.png"
    page.save(path, dpi=(300, 300))
    page.resize((1920, 1080), Image.Resampling.LANCZOS).save(preview)
    return {
        "figure": {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "pixels": list(page.size),
        },
        "preview": {
            "path": str(preview.resolve()),
            "sha256": sha256(preview),
            "pixels": [1920, 1080],
        },
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    source = cherries.input(cfg.source)
    output = cherries.output(cfg.output)
    protocol_path = source / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert tuple(protocol["multipliers"]) == MULTIPLIERS
    assert protocol["steps"] == 200
    mesh_path = Path(protocol["mesh"]["path"])
    assert sha256(mesh_path) == protocol["mesh"]["sha256"]
    states = states_from(source, protocol)
    output.mkdir(parents=True, exist_ok=False)
    (output / "panels").mkdir()
    scene = load_static_context()
    with np.load(mesh_path, allow_pickle=False) as archive:
        mesh = {key: archive[key].copy() for key in archive.files}
    rest, tets, active_ids = mesh["rest_points"], mesh["tets"], mesh["active_ids"]
    np.testing.assert_array_equal(rest, scene.rest_points)
    np.testing.assert_array_equal(tets, scene.volume.cells_dict[pv.CellType.TETRA])
    np.testing.assert_array_equal(
        active_ids, np.flatnonzero(scene.volume.cell_data["ActivationMask"])
    )
    skin_ids, weights = mesh["skin_ids"], mesh["skin_vertex_weights"]
    target = rest[skin_ids] + mesh["target_displacement_skin"]
    skin_faces = np.column_stack(
        (np.full(len(mesh["triangles"]), 3), mesh["triangles"])
    ).ravel()
    active_tets = tets[active_ids]
    rest_tet = rest[active_tets]
    dm = (rest_tet[:, 1:] - rest_tet[:, :1]).transpose(0, 2, 1)
    inverse_dm = np.linalg.inv(dm)
    camera = FIG.common_camera(scene, states)
    control_ids = np.asarray(scene.volume.cell_data["ActivationControlId"])[active_ids]
    cap_max = 0.0
    for state in states:
        deformed = rest + state["u"]
        delta = deformed[skin_ids] - target
        rms = float(1000 * np.sqrt(np.sum(weights * np.sum(delta * delta, axis=1))))
        np.testing.assert_allclose(rms, float(state["trace"]["fit_rms_mm"]), rtol=1e-10)
        state["fit_rms_mm"] = rms
        state["normal_angle_rms_deg"] = float(state["trace"]["normal_angle_rms_deg"])
        tet = deformed[active_tets]
        f = (tet[:, 1:] - tet[:, :1]).transpose(0, 2, 1) @ inverse_dm
        state["glyph"] = principal_glyphs(state["B"], f, tet.mean(axis=1))
        glyph = state["glyph"]
        positive = glyph.eligible & (glyph.eigenvalues_z > 0)
        cap_max = max(
            cap_max, float(np.sqrt(1.0 + glyph.eigenvalues_z[positive]).max() - 1.0)
        )
    cap = max(60.0, cap_max)
    FIG.COLOR_AMPLITUDE_MAX = cap
    projections = []
    receipts = []
    for state in states:
        LOGGER.info("Rendering %s", state["label"])
        deformed = rest + state["u"]
        expected = deformed[scene.exterior_point_ids]
        surface = deformed_exterior(scene, state["u"])
        np.testing.assert_array_equal(surface.points, expected)
        projections.append(
            FIG.render_shape(
                scene,
                surface,
                camera,
                output / "panels" / f"{state['label']}-shape.png",
            )
        )
        volume = scene.volume.copy(deep=True)
        volume.points = deformed
        context = FIG.build_muscle_region_context(volume)
        visibility = FIG.visible_region_mask(
            context,
            state["glyph"].centers,
            active_ids,
            control_ids,
            camera,
            window_size=FIG.ACTIVATION_WINDOW,
        )
        glyph = state["glyph"]
        mask = visibility.mask & glyph.eligible & (glyph.eigenvalues_z > 0)
        assert mask.any(), state["label"]
        activation_skin = pv.PolyData(deformed[skin_ids], skin_faces)
        projections.append(
            FIG.render_activation(
                activation_skin,
                glyph,
                mask,
                camera,
                output / "panels" / f"{state['label']}-activation.png",
            )
        )
        np.testing.assert_array_equal(surface.points, expected)
        np.testing.assert_array_equal(activation_skin.points, deformed[skin_ids])
        np.testing.assert_array_equal(scene.volume.points, rest)
        receipts.append(
            {
                **{
                    key: value
                    for key, value in state.items()
                    if key not in {"u", "B", "glyph", "trace"}
                },
                "visible_positive_principal_lines": int(mask.sum()),
                "principal_amplitude_max_all_eligible": float(
                    np.sqrt(
                        1.0
                        + glyph.eigenvalues_z[
                            glyph.eligible & (glyph.eigenvalues_z > 0)
                        ]
                    ).max()
                    - 1.0
                ),
                "activation": glyph.receipt,
                "surface_is_exact_X_plus_u": True,
                "surface_displacement_error_max_m": float(
                    np.max(np.abs(surface.points - expected))
                ),
                "shape_panel": {
                    "path": f"panels/{state['label']}-shape.png",
                    "sha256": sha256(output / "panels" / f"{state['label']}-shape.png"),
                },
                "activation_panel": {
                    "path": f"panels/{state['label']}-activation.png",
                    "sha256": sha256(
                        output / "panels" / f"{state['label']}-activation.png"
                    ),
                },
            }
        )
        cherries.log_metrics(
            {
                state["label"]: {
                    "fit_rms_mm": state["fit_rms_mm"],
                    "normal_angle_rms_deg": state["normal_angle_rms_deg"],
                    "visible_lines": int(mask.sum()),
                }
            }
        )
        del volume, context, visibility, surface
    for projection in projections:
        np.testing.assert_allclose(projection, projections[0], rtol=0, atol=1e-12)
    figures = compose(states, output, cap)
    manifest = {
        "purpose": "Fixed-axis parent and matched released-axis 1x, 3x, 10x endpoints at update 200",
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256(protocol_path),
        },
        "mesh": {"path": str(mesh_path.resolve()), "sha256": sha256(mesh_path)},
        "states": receipts,
        "camera": camera,
        "screen_projection": projections[0].tolist(),
        "shared_amplitude_cap": cap,
        "cap_policy": "60 unless a positive eligible principal amplitude exceeds 60; then use the observed maximum",
        "display": "Positive principal mode only; a=sigma_max(B)-1, color and line length use log1p(a)/log1p(cap). Full B drives shape.",
        "anatomy_sources": [
            {"path": str(p), "sha256": sha256(p)}
            for p in (VOLUME_PATH, CRANIUM_PATH, MANDIBLE_PATH, EYES_PATH)
        ],
        "source_code": [
            {"path": str(p), "sha256": sha256(p)}
            for p in (
                Path(__file__),
                HISTORICAL_SRC / "shape_scene.py",
                HISTORICAL_SRC / "activation_scene.py",
                HISTORICAL_SRC / "52-render-four-stage-figures.py",
                FIG.CONTEXT_SRC / "muscle_glyph_context.py",
            )
        ],
        **figures,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    cherries.main(main, profile=FIG.Profile)
