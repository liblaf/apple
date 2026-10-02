"""Include activation sign in all three dense meeting-style eigenmode panels."""

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
from muscle_glyph_context import set_parallel_camera
from PIL import Image

from liblaf import cherries

LOG = logging.getLogger(__name__)
WINDOW = (1800, 1800)
BACKGROUND = "#f4f2ed"
CMAP = LinearSegmentedColormap.from_list(
    "signed_activation",
    ["#053061", "#4393c3", "#b5b6b4", "#d6604d", "#67001f"],
)
LABELS = [
    ("mode-1-principal", "Principal mode - maximum contraction direction"),
    ("mode-2-residual", "Residual mode 2 - intermediate direction"),
    ("mode-3-residual", "Residual mode 3 - extension-like effective mode"),
]


class Config(cherries.BaseConfig):
    input_dir: Path = cherries.input("74-meeting-eigenmodes")
    verification: Path = cherries.input(
        "82-meeting-eigenmodes-verification/receipt.json"
    )
    output_dir: Path = cherries.output("77-signed-meeting-eigenmodes", mkdir=True)


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
    title: str,
    path: Path,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="three lights")
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(skin, color="#8d969b", opacity=0.06, smooth_shading=False)
    plotter.add_mesh(
        shown,
        scalars="SignedDisplayMagnitudePercent",
        cmap=CMAP,
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
        "Corrected full fit - no regularization - saved deformed shape\n"
        f"{title}\n"
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
    sources = []
    for source in [
        Path(__file__),
        Path(__file__).with_name("experiment_profile.py"),
        Path(__file__).with_name("muscle_glyph_context.py"),
    ]:
        destination = out / "sources" / source.name
        shutil.copyfile(source, destination)
        sources.append({"source": record(source), "snapshot": record(destination)})
    previous = json.loads((cfg.input_dir / "summary.json").read_text())
    verified = json.loads(cfg.verification.read_text())
    assert previous["status"] == "completed"
    assert verified["status"] == "passed"
    inputs = [record(cfg.verification), checked_record(verified["inputs"]["summary"])]
    baseline = checked_record(verified["inputs"]["baseline"])
    inputs.append(baseline)
    with np.load(baseline["path"], allow_pickle=False) as saved:
        deformed = saved["rest_points"] + saved["u"]
        global_ids = saved["active_ids"]
    skin_record = previous["inputs"][1]
    inputs.append(checked_record(skin_record))
    skin = pv.read(skin_record["path"])
    skin.points = deformed[np.asarray(skin.point_data["GlobalPointId"])]
    glyphs, signed_values = [], []
    for expected, (identifier, _) in zip(
        verified["inputs"]["glyphs"], LABELS, strict=True
    ):
        inputs.append(checked_record(expected))
        assert Path(expected["path"]) == cfg.input_dir / f"all-active-{identifier}.vtp"
        glyph = pv.read(expected["path"])
        assert np.array_equal(glyph.cell_data["GlobalCellId"], global_ids)
        z = np.asarray(glyph.cell_data["Z_eigenvalue"])
        signed = 100 * np.sign(z) * (1 - 1 / np.sqrt(1 + np.abs(z)))
        assert (
            np.max(np.abs(np.abs(signed) - glyph.cell_data["DisplayMagnitudePercent"]))
            == 0
        )
        nonneutral = np.abs(z) > previous["encoding"]["neutral_tolerance"]
        assert np.array_equal(np.sign(signed[nonneutral]), np.sign(z[nonneutral]))
        glyph.cell_data["SignedDisplayMagnitudePercent"] = signed
        glyphs.append(glyph)
        signed_values.append(signed)
    np.savez_compressed(
        out / "signed-color-values.npz",
        global_cell_ids=global_ids,
        activation_control_ids=glyphs[0].cell_data["ActivationControlId"],
        mode_indices=np.arange(1, 4),
        eigenvalues_descending=np.stack(
            [g.cell_data["Z_eigenvalue"] for g in glyphs], axis=1
        ),
        signed_display_magnitude_percent=np.stack(signed_values, axis=1),
    )
    views = {}
    for view_name, view in previous["views"].items():
        visibility_path = cfg.input_dir / f"{view_name}-visibility.npz"
        inputs.append(record(visibility_path))
        with np.load(visibility_path, allow_pickle=False) as visibility:
            assert np.array_equal(visibility["global_cell_ids"], global_ids)
            mask = visibility["mask"]
        panels, paths = [], []
        for index, (glyph, (identifier, title)) in enumerate(
            zip(glyphs, LABELS, strict=True)
        ):
            visible_mask = mask & (glyph.cell_data["DisplayLengthM"] > 0)
            shown = glyph.extract_cells(visible_mask)
            assert shown.n_cells == view["modes"][index]["shown_nonzero_unique_lines"]
            assert np.array_equal(
                shown.cell_data["SignedDisplayMagnitudePercent"],
                signed_values[index][visible_mask],
            )
            path = out / f"{view_name}-{identifier}.png"
            render_panel(skin, shown, view["camera"], title, path)
            panels.append(
                {"mode": index + 1, "image": record(path), "shown_lines": shown.n_cells}
            )
            paths.append(path)
        triptych_path = out / f"{view_name}-triptych.png"
        triptych(paths, triptych_path)
        views[view_name] = {
            "camera": view["camera"],
            "modes": panels,
            "triptych": record(triptych_path),
        }
        LOG.info("Rendered all three signed modes for %s", view_name)
    summary = {
        "status": "completed",
        "inputs": inputs,
        "sources": sources,
        "views": views,
        "signed_values": record(out / "signed-color-values.npz"),
        "encoding": {
            "signed_display_scalar": "100*sign(z)*(1-1/sqrt(1+abs(z)))",
            "color": "linear diverging palette on shared [-100,100]; positive red, negative blue, near zero gray",
            "color_stops": ["#053061", "#4393c3", "#b5b6b4", "#d6604d", "#67001f"],
            "length": "unchanged from verified input: 4.5 mm*(1-1/sqrt(1+abs(z)))",
            "geometry": "verified input lines, deformed skin, and visibility masks reused unchanged",
            "sign_semantics": "sign of activation eigenvalue z, independent of arbitrary eigenvector orientation",
            "neutral_or_nonunique_axes": "same zero-length suppression as verified input",
            "neutral_tolerance": previous["encoding"]["neutral_tolerance"],
        },
        "limitations": [
            "Signed display percentage is a visualization transform, not physical strain.",
            "The negative-z range approaches -1, so the most negative display value approaches -29.3%; this is not 29.3% extension.",
            "Modes 2 and 3 together form the residual tensor; equilibrium shapes do not add.",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    cherries.log_metrics({"active_cells": len(global_ids), "signed_panel_count": 6})
    LOG.info("Completed signed meeting-style figures: %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
