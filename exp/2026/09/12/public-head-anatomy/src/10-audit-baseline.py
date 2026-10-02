# Copyright 2026 liblaf
"""Measure and visualize the existing semantic regions and fiber directions."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from anatomy_common import BASELINE, ProfileCometNoCommit, camera, sha256, write_json

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = BASELINE
    output: Path = cherries.output("10-baseline", mkdir=True)


def main(cfg: Config) -> None:
    output = cfg.output
    output.mkdir(parents=True, exist_ok=True)
    mesh = pv.read(cfg.fixture / "volume.vtu")
    skin = pv.read(cfg.fixture / "skin.vtp")
    centers = mesh.cell_centers().points
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    labels = np.asarray(mesh.cell_data["MuscleId"], dtype=int)
    fibers = np.asarray(mesh.cell_data["ActivationFiber"])
    weights = np.asarray(mesh.cell_data["Volume"] * mesh.cell_data["MuscleFraction"])
    names = mesh.field_data["MuscleName"]
    rows = []
    for muscle_id in np.unique(labels[active]):
        selected = active & (labels == muscle_id)
        f = fibers[selected]
        w = weights[selected]
        tensor = np.einsum("n,ni,nj->ij", w / w.sum(), f, f)
        values, vectors = np.linalg.eigh(tensor)
        axis = vectors[:, -1]
        axis *= 1 if axis[np.argmax(np.abs(axis))] >= 0 else -1
        rows.append(
            {
                "muscle_id": int(muscle_id),
                "name": str(names[muscle_id]),
                "active_cells": int(selected.sum()),
                "muscle_volume_mm3": float(w.sum() * 1e9),
                "centroid_m": np.average(centers[selected], weights=w, axis=0).tolist(),
                "extent_mm": (np.ptp(centers[selected], axis=0) * 1000).tolist(),
                "principal_fiber_axis": axis.tolist(),
                "fiber_second_moment_eigenvalues": values.tolist(),
                "mean_squared_superior_alignment": float(tensor[1, 1]),
            }
        )
    write_json(
        output / "regions.json",
        {
            "source_volume": str(cfg.fixture / "volume.vtu"),
            "source_volume_sha256": sha256(cfg.fixture / "volume.vtu"),
            "source_skin_sha256": sha256(cfg.fixture / "skin.vtp"),
            "units": "m",
            "axes": {"x": "lateral", "y": "superior", "z": "anterior"},
            "active_cells": int(active.sum()),
            "regions": rows,
            "interpretation": "Existing heuristic field; this audit does not validate anatomy.",
        },
    )
    skin.save(output / "skin.vtp")
    forehead = active & (labels == 28)
    patch = (
        mesh.extract_cells(forehead)
        .extract_surface(algorithm="dataset_surface")
        .triangulate()
    )
    patch.save(output / "forehead.vtp")
    rng = np.random.default_rng(0)
    ids = rng.choice(
        np.flatnonzero(forehead), size=min(250, int(forehead.sum())), replace=False
    )
    glyphs = pv.PolyData(centers[ids])
    glyphs.point_data["fiber"] = fibers[ids]
    glyphs.save(output / "forehead-fibers.vtp")
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 1100))
    plotter.set_background("white")
    plotter.add_mesh(skin, color="#b7b9bc", opacity=0.18)
    plotter.add_mesh(patch, color="#d88a70", opacity=0.8)
    plotter.add_mesh(
        glyphs.glyph(orient="fiber", scale=False, factor=0.007), color="#253a57"
    )
    plotter.add_text(
        "Current forehead region and PCA fibers", color="#252525", font_size=14
    )
    camera(plotter, skin.center)
    plotter.camera.zoom(1.12)
    plotter.show(screenshot=output / "forehead-fibers.png", auto_close=True)
    cherries.log_metrics(
        {
            "baseline/active_cells": int(active.sum()),
            "baseline/forehead_active_cells": int(forehead.sum()),
            "baseline/forehead_superior_alignment_squared": next(
                r["mean_squared_superior_alignment"]
                for r in rows
                if r["muscle_id"] == 28
            ),
        }
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
