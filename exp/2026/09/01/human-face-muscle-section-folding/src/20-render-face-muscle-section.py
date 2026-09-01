"""Render static, local-coordinate evidence for the id64 folding section."""

from __future__ import annotations

# ruff: noqa: EM101, TRY003
import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "10-face-muscle-section"
OUT = ROOT / "data" / "20-face-muscle-section-render"
PRIMARY = "20-human-face-smile-no-skin-lr3"
SMOOTH = "20-human-face-smile-skin-estimated-plus-tightening-lr1"


def digest(path: Path) -> dict[str, object]:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": h}


def camera(
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, object]:
    span = np.array(
        (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    )
    center = np.array(
        (
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        )
    )
    return {
        "position": (center + 2.6 * max(span) * np.array((1.0, 0.8, 0.65))).tolist(),
        "focal_point": center.tolist(),
        "view_up": [0, 0, 1],
        "parallel_projection": True,
        "parallel_scale": float(0.62 * max(span)),
    }


def setup(plotter: pv.Plotter, spec: dict[str, object]) -> None:
    plotter.set_background("white")
    plotter.enable_parallel_projection()
    plotter.camera.position = spec["position"]
    plotter.camera.focal_point = spec["focal_point"]
    plotter.camera.up = spec["view_up"]
    plotter.camera.parallel_scale = spec["parallel_scale"]


def section_panel(
    plotter: pv.Plotter,
    ref: pv.UnstructuredGrid,
    deformed: pv.UnstructuredGrid,
    scalar: str,
    rng: tuple[float, float],
    title: str,
    spec: dict[str, object],
) -> None:
    setup(plotter, spec)
    plotter.add_mesh(
        ref, color="#9aa0a6", style="wireframe", line_width=1.2, opacity=0.72
    )
    plotter.add_mesh(
        deformed,
        scalars=scalar,
        clim=rng,
        cmap="coolwarm",
        show_edges=True,
        edge_color="#202124",
        line_width=0.35,
        scalar_bar_args={"title": scalar},
    )
    bad = deformed.threshold((0.5, 1.5), scalars="DoubleInverted")
    if bad.n_cells:
        plotter.add_mesh(
            bad,
            color="#ff00b8",
            show_edges=True,
            edge_color="black",
            line_width=1.5,
            opacity=1.0,
        )
    plotter.add_text(title, font_size=13, color="black", position="upper_left")


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    summary = json.loads((DATA / "summary.json").read_text())
    if (
        summary["selection"]["primary_case"] != PRIMARY
        or summary["selection"]["comparator_case"] != SMOOTH
    ):
        raise ValueError("unexpected builder case semantics")
    if summary["selection"]["slab"]["source_cell_count"] != 31:
        raise ValueError("expected the matched 31-cell section")
    files = {name: DATA / name for name in summary["exports"]}
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError("builder export missing")
    p_ref, p_def = (
        pv.read(DATA / f"{PRIMARY}-section-{state}.vtu")
        for state in ("reference", "deformed")
    )
    s_ref, s_def = (
        pv.read(DATA / f"{SMOOTH}-section-{state}.vtu")
        for state in ("reference", "deformed")
    )
    if p_def.n_cells != s_def.n_cells != 31 or set(p_def.cell_data) < {
        "DetF",
        "DetAinv",
        "DetG",
        "DoubleInverted",
        "ActivationNorm",
    }:
        raise ValueError("section array contract mismatch")
    bounds = tuple(
        float(x)
        for x in np.r_[p_ref.bounds, p_def.bounds, s_ref.bounds, s_def.bounds]
        .reshape(4, 6)
        .T[[0, 1, 2, 3, 4, 5]]
        .reshape(-1)
    )
    # Explicit union is clearer than relying on a plotting-side auto-fit.
    b = np.vstack([p_ref.bounds, p_def.bounds, s_ref.bounds, s_def.bounds])
    bounds = (
        b[:, 0].min(),
        b[:, 1].max(),
        b[:, 2].min(),
        b[:, 3].max(),
        b[:, 4].min(),
        b[:, 5].max(),
    )
    spec = camera(bounds)
    ranges = {
        name: (
            min(0.0, *(float(g.cell_data[name].min()) for g in (p_def, s_def))),
            max(0.0, *(float(g.cell_data[name].max()) for g in (p_def, s_def))),
        )
        for name in ("DetF", "DetAinv", "DetG")
    }
    OUT.mkdir(parents=True)
    plot = pv.Plotter(
        shape=(1, 3), off_screen=True, window_size=(3000, 1000), lighting="light kit"
    )
    for i, name in enumerate(("DetF", "DetAinv", "DetG")):
        plot.subplot(0, i)
        section_panel(
            plot,
            p_ref,
            p_def,
            name,
            ranges[name],
            f"Bumpy primary | {name} | id64",
            spec,
        )
    plot.screenshot(OUT / "primary-mechanism.png")
    plot.close()
    plot = pv.Plotter(
        shape=(1, 2), off_screen=True, window_size=(2400, 1100), lighting="light kit"
    )
    for i, (label, ref, deformed) in enumerate(
        (
            ("Bumpy primary", p_ref, p_def),
            ("Historical smooth-surface context (not causal proof)", s_ref, s_def),
        )
    ):
        plot.subplot(0, i)
        section_panel(plot, ref, deformed, "DetF", ranges["DetF"], label, spec)
    plot.screenshot(OUT / "matched-section-comparison.png")
    plot.close()
    source = Path(summary["source_summary"]).parent / f"{PRIMARY}.vtu"
    face = pv.read(source)
    axes = np.asarray(summary["selection"]["pca_axes_columns"], dtype=float)
    origin = np.asarray(summary["selection"]["pca_origin"], dtype=float)
    locator = p_ref.copy(deep=True)
    locator.points = np.asarray(locator.points) @ axes.T + origin
    plot = pv.Plotter(off_screen=True, window_size=(2600, 1800), lighting="light kit")
    plot.set_background("white")
    plot.add_mesh(
        face.extract_surface(), color="#d9b49b", opacity=0.82, smooth_shading=True
    )
    plot.add_mesh(locator, color="#d91c5c", show_edges=True, edge_color="black")
    plot.add_text(
        "Rest/global face locator: Zygomaticus major id64 (red)\n"
        "Final deformed local section is shown separately; no deformation exaggeration",
        font_size=16,
        color="black",
    )
    plot.view_isometric()
    plot.screenshot(OUT / "face-context-id64.png")
    plot.close()
    from paraview.simple import SaveState

    SaveState(str(OUT / "face-muscle-section.pvsm"))
    receipt = {
        "status": "ok",
        "primary_case": PRIMARY,
        "primary_best_step": 194,
        "comparator_case": SMOOTH,
        "comparator_best_step": 192,
        "dimensions": {
            "section_cells": 31,
            "png_size": {
                "mechanism": [3000, 1000],
                "comparison": [2400, 1100],
                "context": [2600, 1800],
            },
        },
        "arrays": ["DetF", "DetAinv", "DetG", "DoubleInverted", "ActivationNorm"],
        "scalar_ranges": ranges,
        "camera": spec,
        "sources": {
            "builder_summary": digest(DATA / "summary.json"),
            **{name: digest(path) for name, path in files.items()},
        },
        "outputs": {p.name: digest(p) for p in OUT.glob("*.png")},
        "no_exaggeration": True,
    }
    (OUT / "render-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
