"""Slide-ready comparison of four target heights from saved free/off runs."""

# ruff: noqa: RUF001, SLF001

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import study
from PIL import Image

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

GROUP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "height_figures", GROUP / "src/30-render.py"
)
assert SPEC is not None
assert SPEC.loader is not None
render = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = render
SPEC.loader.exec_module(render)
HEIGHTS = (0.05, 0.10, 0.15, 0.20)
LIMITS = (-0.02, 1.02, -0.01, 0.31)
STEM = "free-activation-height-comparison-16x9"
LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("170-free-height-figures")
    dpi: int = 240


def verify(cases: list, output: Path) -> dict:
    mesh = study.ph.build_mesh(100, 10)
    common_step = min(case.summary["accepted_iterations"] for case in cases)
    common_saved_step = max(
        set.intersection(*(set(case.history["steps"]) for case in cases))
    )
    records = []
    for case in cases:
        u, B = case.checkpoint["u"], case.checkpoint["B"]
        np.testing.assert_array_equal(case.history["points"], mesh.p)
        np.testing.assert_array_equal(case.history["triangles"], mesh.tri)
        np.testing.assert_array_equal(case.history["muscle"], mesh.muscle)
        np.testing.assert_array_equal(
            study.matrices(mesh, case.checkpoint["controls"], case.mode), B
        )
        free_u = u.ravel()[mesh.free]
        _, residual, _, J = study.ph.assemble(mesh, free_u, B, hessian=False)
        loss, _, _ = study.ph.loss(mesh, free_u, case.height, "l2")
        np.testing.assert_allclose(
            loss / case.height**2, case.summary["final"]["normalized_loss"], rtol=1e-12
        )
        np.testing.assert_allclose(
            J.min(), case.summary["final"]["min_J"], rtol=1e-10, atol=1e-13
        )
        assert np.linalg.norm(residual, np.inf) <= 1e-10
        index = np.flatnonzero(case.trace["step"] == common_step).item()
        records.append(
            {
                "height": case.height,
                "source": str(case.source / case.name),
                "step": int(case.checkpoint["step"]),
                "failure": case.failure,
                "normalized_L2": loss / case.height**2,
                "fit_rms": float(np.sqrt(loss)),
                "activation_neighbor_rms": case.summary["final"]["tensor_neighbor_rms"],
                "min_physical_J": float(J.min()),
                "inverted_cells": int(np.count_nonzero(J <= 0)),
                "force_residual_inf": float(np.linalg.norm(residual, np.inf)),
                "common_step": common_step,
                "common_step_normalized_L2": float(
                    case.trace["normalized_loss"][index]
                ),
                "common_step_activation_neighbor_rms": float(
                    case.trace["tensor_neighbor_rms"][index]
                ),
                "checkpoint_sha256": hashlib.sha256(
                    (case.source / case.name / "checkpoint.npz").read_bytes()
                ).hexdigest(),
            }
        )
        np.savez_compressed(
            output / f"{case.name}-glyphs.npz", **render._activation_geometry(case)
        )
    result = {
        "cases": records,
        "common_saved_step": int(common_saved_step),
        "mechanical_stability_tested": False,
        "inverse_convergence_demonstrated": False,
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    return result


def plot(  # noqa: C901, PLR0912, PLR0915
    cases: list, output: Path, dpi: int, completion: dict | None = None
) -> dict:
    maximum, glyph_length = render._activation_scale(cases)
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(
        0.5,
        0.958,
        "Free activation across target heights",
        ha="center",
        fontsize=27,
    )
    fig.text(
        0.5,
        0.911,
        "Smoothness off  ·  h = peak target displacement  ·  identical physical scales"
        if completion is None
        else "Smoothness off  ·  1,200 Adam updates at every height  ·  identical physical scales",
        ha="center",
        fontsize=14,
    )
    for x, title in (
        (4.15, "Last accepted shape" if completion is None else "Final shape"),
        (9.90, "Activation"),
        (14.23, "Fit and run status"),
    ):
        fig.text(x / 16, 0.867, title, ha="center", fontsize=17, weight="bold")
    axes = []
    width = 5.10
    height = width * (LIMITS[3] - LIMITS[2]) / (LIMITS[1] - LIMITS[0])
    for row, case in enumerate(cases):
        bottom = 7.60 - height - row * 1.63
        pair = [
            fig.add_axes([left / 16, bottom / 9, width / 16, height / 9])
            for left in (1.60, 7.35)
        ]
        render._draw_deformation(pair[0], case, LIMITS)
        render._draw_activation(pair[1], case, maximum, glyph_length, LIMITS)
        points, _, _, top, _, _ = render._history_contract(case)
        x, y = render._target(points, top, case.height)
        pair[1].plot(x, y, "k--", lw=1.1)
        cloud = np.vstack([points + case.checkpoint["u"], np.column_stack([x, y])])
        assert np.all(cloud.min(axis=0) > [LIMITS[0], LIMITS[2]])
        assert np.all(cloud.max(axis=0) < [LIMITS[1], LIMITS[3]])
        for col, axis in enumerate(pair):
            for annotation in list(axis.texts):
                annotation.remove()
            axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
            axis.set_yticks([0, 0.1, 0.2, 0.3])
            axis.tick_params(
                labelsize=10, length=3, labelbottom=row == 3, labelleft=col == 0
            )
            if row == 3:
                axis.set_xlabel("deformed x", fontsize=11, labelpad=2)
            if not case.failure:
                for spine in axis.spines.values():
                    spine.set_color("#777777")
                    spine.set_linewidth(0.7)
        axes.extend(pair)
        center = bottom + height / 2
        fig.text(
            1.28 / 16,
            center / 9,
            f"h = {case.height:.2f}",
            ha="right",
            va="center",
            fontsize=17,
            weight="bold",
        )
        fig.text(
            13.05 / 16,
            (center + 0.42) / 9,
            f"L2/h² = {case.summary['final']['normalized_loss']:.4f}"
            + (
                "†"
                if completion is not None
                and not completion[case.height]["final_forward_converged"]
                else ""
            ),
            fontsize=16,
        )
        step = int(case.checkpoint["step"])
        if completion is not None:
            record = completion[case.height]
            fig.text(
                13.05 / 16,
                center / 9,
                f"Adam {step:,} · {record['failed_solves']} failed solves",
                fontsize=12,
            )
            converged = record["final_forward_converged"]
            unstable = converged and record["minimum_hessian_eigenvalue"] < -1e-10
            status = (
                "Unstable equilibrium*"
                if unstable
                else ("Forward equilibrium" if converged else "Forward not converged")
            )
            fig.text(
                13.05 / 16,
                (center - 0.33) / 9,
                status,
                fontsize=12,
                color="#a71930" if unstable or not converged else "#345951",
                weight="bold",
            )
            fig.text(
                13.05 / 16,
                (center - 0.61) / 9,
                f"Force residual {record['force_residual_inf']:.2g}",
                fontsize=10,
                color="#555555",
            )
        elif case.failure:
            fig.text(13.05 / 16, center / 9, f"Last accepted step {step}", fontsize=13)
            fig.text(
                13.05 / 16,
                (center - 0.33) / 9,
                f"FAILED at {case.failure['step']}",
                fontsize=13,
                color="#a71930",
                weight="bold",
            )
            cause = (
                "Forward line search"
                if "line search" in case.failure["reason"]
                else "Forward iteration limit"
            )
            if case.height == 0.15:
                cause = "Iteration limit; residual 1.032×10⁻¹⁰"
            fig.text(
                13.05 / 16,
                (center - 0.61) / 9,
                cause,
                fontsize=10,
                color="#555555",
            )
        else:
            fig.text(13.05 / 16, center / 9, f"Step {step}", fontsize=13)
            fig.text(
                13.05 / 16,
                (center - 0.33) / 9,
                "Budget completed",
                fontsize=12,
                color="#555555",
            )
    cax = fig.add_axes([0.29, 0.058, 0.39, 0.017])
    bar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(-maximum, maximum), cmap="RdBu_r"
        ),
        cax=cax,
        orientation="horizontal",
    )
    bar.ax.tick_params(labelsize=10, length=2, pad=2)
    bar.set_label(
        "Signed activation λ(B−I) · equal-length axes show orientation",
        fontsize=11,
        labelpad=2,
    )
    fig.text(
        0.985,
        0.019,
        "Dashed: target\nUnequal run lengths; convergence not established"
        if completion is None
        else "Dashed: target · failed solves use approximate gradients\n"
        "Next solve resets to rest · † Off-equilibrium loss\n"
        "* Negative Hessian curvature\n"
        "Inverse convergence not established",
        ha="right",
        fontsize=9,
        color="#444444",
    )
    fig.canvas.draw()
    bounds = np.array([axis.get_window_extent().bounds for axis in axes])
    np.testing.assert_allclose(
        bounds[:, 2:], np.broadcast_to(bounds[0, 2:], (8, 2)), atol=1e-8
    )
    for axis in axes:
        transform = axis.transData.transform([[0, 0], [1, 0], [0, 1]])
        np.testing.assert_allclose(
            transform[1, 0] - transform[0, 0],
            transform[2, 1] - transform[0, 1],
            atol=1e-8,
        )
    for artist in fig.findobj():
        if artist.get_rasterized():
            artist.set_rasterized(False)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output / f"{STEM}.{suffix}", dpi=dpi, bbox_inches=None, facecolor="white"
        )
    fig.savefig(
        output / f"{STEM}-preview.png", dpi=120, bbox_inches=None, facecolor="white"
    )
    plt.close(fig)
    with Image.open(output / f"{STEM}.png") as image:
        assert image.size == (16 * dpi, 9 * dpi)
        image.verify()
    assert "<image " not in (output / f"{STEM}.svg").read_text()
    return {
        "png_pixels": [16 * dpi, 9 * dpi],
        "data_limits": LIMITS,
        "equal_physical_xy_scale": True,
        "equal_panel_sizes": True,
        "activation_color_range": [-maximum, maximum],
        "glyph_length": glyph_length,
        "vector_svg_no_embedded_images": True,
    }


def main(cfg: Config) -> None:
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = [GROUP / "data/tune-w0", GROUP / "data/160-free-height-sweep"]
    loaded = render._load_cases(sources)
    cases = [render._case(loaded, h, "unconstrained", 0) for h in HEIGHTS]
    comparison = verify(cases, output)
    checks = plot(cases, output, cfg.dpi)
    (output / "delivery-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    snapshot = output / "source"
    snapshot.mkdir()
    for source in (Path(__file__), Path(render.__file__)):
        shutil.copy2(source, snapshot / source.name)
    for row in comparison["cases"]:
        LOG.info(
            "h=%.2f step=%d L2/h²=%.6f failure=%s",
            row["height"],
            row["step"],
            row["normalized_L2"],
            row["failure"],
        )
    LOG.info("Saved 16:9 PNG, PDF, SVG to %s", output)


if __name__ == "__main__":
    cherries.main(main, profile=render.ProfileFigures)
