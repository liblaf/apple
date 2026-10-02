"""Aligned whole-shape and activation views of the saved h=0.20 endpoints."""

# ruff: noqa: RUF001, SLF001

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

GROUP = Path(__file__).resolve().parents[1]
SOURCE = GROUP / "src/30-render.py"
SPEC = importlib.util.spec_from_file_location("activation_figures", SOURCE)
assert SPEC is not None
assert SPEC.loader is not None
render = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = render
SPEC.loader.exec_module(render)
LOG = logging.getLogger(__name__)
HEIGHT = 0.20
LIMITS = (-0.02, 1.02, -0.01, 0.31)
LABELS = (
    "Free activation",
    "Contraction only\nfree directions",
    "Contraction only\nlearned direction",
    "Contraction only\nfixed x-direction",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("80-aligned-h200")
    slide_format: bool = False
    slide_dpi: int = 240
    vector_export: bool = False


def save_aligned_figure(
    fig: plt.Figure,
    output: Path,
    stem: str,
    *,
    slide_format: bool,
    dpi: int,
    vector_export: bool,
) -> list[str]:
    if not slide_format:
        return render._save_pair(fig, output, stem)
    files = [f"{stem}.png", f"{stem}.pdf"]
    if vector_export:
        files.append(f"{stem}.svg")
        for artist in fig.findobj():
            if artist.get_rasterized():
                artist.set_rasterized(False)
    for filename in files:
        with mpl.rc_context({"svg.fonttype": "path"}):
            fig.savefig(output / filename, dpi=dpi, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return files


def aligned_figure(  # noqa: C901
    cases: list[Any],
    weights: tuple[float, ...],
    output: Path,
    maximum: float,
    glyph_length: float,
    stem: str,
    *,
    slide_format: bool = False,
    dpi: int = 240,
    vector_export: bool = False,
    failure_label: str | None = None,
    footnote: str = "* Free activation / smoothness off: last valid step 262; forward solve failed at 263. Other results: step 1200.",
) -> dict[str, Any]:
    ncols = 2 * len(weights)
    fig, axes = plt.subplots(
        4,
        ncols,
        figsize=(16, 9) if slide_format else (20 if ncols == 4 else 12, 8.4),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.15 if slide_format else (0.105 if ncols == 4 else 0.16),
        right=0.98 if slide_format else 0.99,
        bottom=0.20 if slide_format else 0.16,
        top=0.80 if slide_format else 0.87,
        wspace=0.07,
        hspace=0.40 if slide_format else 0.18,
    )
    panel_records = []
    for row, mode in enumerate(render.MODES):
        for group, weight in enumerate(weights):
            case = render._case(cases, HEIGHT, mode, weight)
            shape, activation = axes[row, 2 * group : 2 * group + 2]
            render._draw_deformation(shape, case, LIMITS)
            render._draw_activation(activation, case, maximum, glyph_length, LIMITS)
            points, _, _, top, _, _ = render._history_contract(case)
            x, y = render._target(points, top, HEIGHT)
            activation.plot(x, y, "k--", lw=1.0)
            for axis in (shape, activation):
                for annotation in list(axis.texts):
                    annotation.remove()
                axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
                axis.set_yticks([0, 0.1, 0.2, 0.3])
                axis.tick_params(labelsize=9 if slide_format else 8, length=3)
                if row == 3:
                    axis.set_xlabel("deformed x", fontsize=10 if slide_format else 9)
            shape.text(
                0.02,
                0.96,
                f"L2/h² {case.summary['final']['normalized_loss']:.4f}",
                transform=shape.transAxes,
                fontsize=8,
                va="top",
            )
            activation.text(
                0.02,
                0.96,
                f"step {int(case.trace['step'][-1])}",
                transform=activation.transAxes,
                fontsize=8,
                va="top",
            )
            if case.failure:
                for axis in (shape, activation):
                    axis.text(
                        0.98,
                        0.96,
                        failure_label
                        or ("FAILED*" if slide_format else "FAILED · last valid*"),
                        color="#a71930",
                        ha="right",
                        va="top",
                        transform=axis.transAxes,
                        fontsize=8,
                        weight="bold",
                    )
            if row == 0:
                shape.set_title(
                    "Final shape", fontsize=13 if slide_format else 12, pad=8
                )
                activation.set_title(
                    "Activation" if slide_format else "Activation on final shape",
                    fontsize=13 if slide_format else 12,
                    pad=8,
                )
            panel_records.append(
                {
                    "row": row,
                    "mode": mode,
                    "weight": weight,
                    "case": case.name,
                    "step": int(case.trace["step"][-1]),
                    "failed": bool(case.failure),
                }
            )
        axes[row, 0].set_ylabel(
            LABELS[row],
            rotation=0,
            ha="right",
            va="center",
            labelpad=14,
            fontsize=12 if slide_format else 10,
        )
    fig.suptitle(
        "Final shape and activation · h = 0.20"
        if slide_format
        else "Final shape and activation · h = 0.20 · whole-domain overview",
        y=0.975 if slide_format else 0.995,
        fontsize=23 if slide_format else 17,
    )
    fig.text(
        0.55,
        0.925 if slide_format else 0.947,
        "Whole-domain view · identical physical scales · dashed curve: target",
        ha="center",
        fontsize=12 if slide_format else 10,
    )
    fig.canvas.draw()
    for group, weight in enumerate(weights):
        left = axes[0, 2 * group].get_position().x0
        right = axes[0, 2 * group + 1].get_position().x1
        label = "Smoothness off" if weight == 0 else "Smoothness on (α = 1)"
        fig.text(
            (left + right) / 2,
            0.86 if slide_format else 0.901,
            label,
            ha="center",
            fontsize=16 if slide_format else 13,
            weight="bold",
        )
    cax = fig.add_axes(
        [
            0.34 if ncols == 4 else 0.30,
            0.105 if slide_format else 0.073,
            0.42 if ncols == 4 else 0.48,
            0.018,
        ]
    )
    bar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(-maximum, maximum), cmap="RdBu_r"
        ),
        cax=cax,
        orientation="horizontal",
    )
    bar.set_label(
        "Signed activation λ(B−I) · thin equal-length axes show orientation",
        fontsize=11 if slide_format else 10,
    )
    bar.ax.tick_params(labelsize=9 if slide_format else 8)
    fig.text(
        0.55,
        0.025 if slide_format else 0.012,
        footnote,
        ha="center",
        fontsize=10 if slide_format else 9,
    )
    fig.canvas.draw()
    bounds = np.asarray([axis.get_window_extent().bounds for axis in axes.flat])
    pixel_sizes = bounds[:, 2:]
    np.testing.assert_allclose(
        pixel_sizes,
        np.broadcast_to(pixel_sizes[0], pixel_sizes.shape),
        atol=1e-8,
        rtol=0,
    )
    for row in range(4):
        np.testing.assert_allclose(
            bounds[row * ncols : (row + 1) * ncols, 1],
            bounds[row * ncols, 1],
            atol=1e-8,
        )
    for axis in axes.flat:
        np.testing.assert_array_equal(axis.get_xlim(), LIMITS[:2])
        np.testing.assert_array_equal(axis.get_ylim(), LIMITS[2:])
        transform = axis.transData.transform(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )
        np.testing.assert_allclose(
            transform[1, 0] - transform[0, 0],
            transform[2, 1] - transform[0, 1],
            atol=1e-8,
        )
    return {
        "files": save_aligned_figure(
            fig,
            output,
            stem,
            slide_format=slide_format,
            dpi=dpi,
            vector_export=vector_export,
        ),
        "slide_format": slide_format,
        "canvas_inches": [16, 9] if slide_format else None,
        "png_pixels": [16 * dpi, 9 * dpi] if slide_format else None,
        "vector_export": vector_export,
        "panels": panel_records,
        "data_limits": LIMITS,
        "equal_panel_dimensions": True,
        "equal_physical_xy_scale": True,
        "panel_size_range_pixels": np.ptp(pixel_sizes, axis=0).tolist(),
    }


def main(cfg: Config) -> None:
    assert cfg.slide_dpi > 0
    assert not cfg.vector_export or cfg.slide_format
    cases = render._load_cases([GROUP / "data/tune-w0", GROUP / "data/tune-w1"])
    selected = render._select(cases, 1.0)
    maximum, glyph_length = render._activation_scale(selected)
    cases = [case for case in selected if np.isclose(case.height, HEIGHT)]
    assert len(cases) == 8
    for case in cases:
        geometry = render._activation_geometry(case)
        with np.load(
            GROUP / "data/70-readable-glyphs" / f"{case.name}-glyphs.npz"
        ) as prior:
            for key, value in geometry.items():
                np.testing.assert_array_equal(value, prior[key])
        points, _, _, top, _, _ = render._history_contract(case)
        x, y = render._target(points, top, HEIGHT)
        cloud = np.vstack([geometry["points"], np.column_stack([x, y])])
        assert np.all(cloud.min(axis=0) > [LIMITS[0], LIMITS[2]])
        assert np.all(cloud.max(axis=0) < [LIMITS[1], LIMITS[3]])
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    for source in (Path(__file__), SOURCE):
        shutil.copy2(source, output / source.name)
    selections = (
        [((0.0, 1.0), "aligned-shape-activation-h200-16x9")]
        if cfg.slide_format
        else [
            ((0.0, 1.0), "aligned-shape-activation-h200"),
            ((0.0,), "aligned-shape-activation-h200-off"),
            ((1.0,), "aligned-shape-activation-h200-on"),
        ]
    )
    records = [
        aligned_figure(
            cases,
            weights,
            output,
            maximum,
            glyph_length,
            stem,
            slide_format=cfg.slide_format,
            dpi=cfg.slide_dpi,
            vector_export=cfg.vector_export,
        )
        for weights, stem in selections
    ]
    titles = [
        "All four models: smoothness off and on",
        "Smoothness off",
        "Smoothness on (α = 1)",
    ]
    render._write_gallery(
        output,
        [
            (
                title,
                record["files"],
                "Whole-domain views with identical physical axes. Shape and activation use the same saved endpoint; glyph directions follow F n / ||F n||.",
            )
            for title, record in zip(titles[: len(records)], records, strict=True)
        ],
        cases,
        1.0,
    )
    (output / "alignment-checks.json").write_text(
        json.dumps(
            {
                "height": HEIGHT,
                "cases": len(cases),
                "geometry_matches_previous_render_exactly": True,
                "global_color_range": [-maximum, maximum],
                "glyph_full_length": glyph_length,
                "glyph_linewidth_points": 0.45,
                "figures": records,
            },
            indent=2,
        )
        + "\n"
    )
    cherries.log_metrics(
        {
            "figure_pairs": len(records),
            "selected_cases": len(cases),
            "alignment_checks_passed": True,
        }
    )
    LOG.info(
        "Saved %d aligned whole-domain PNG/PDF figure pairs to %s", len(records), output
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else render.ProfileFigures
    )
