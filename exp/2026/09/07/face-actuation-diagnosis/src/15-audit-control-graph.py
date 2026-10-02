"""Measure the support of the full per-tetrahedron smoothness prior."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from face_physics import active_graph
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = (
        Path(__file__).resolve().parents[2]
        / "face-activation-materials/data/10-fixture"
    )
    output_dir: Path = cherries.output("15-control-graph", mkdir=True)


def main(cfg: Config):
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir())
    mesh = pv.read(cfg.fixture / "volume.vtu")
    points = np.asarray(mesh.points)
    tets = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    ids = np.flatnonzero(mesh.cell_data["ActivationMask"])
    region = np.asarray(mesh.cell_data["ActivationControlId"])[ids]
    fraction = np.asarray(mesh.cell_data["MuscleFraction"])
    volume = (
        np.linalg.det(
            np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
        )
        / 6
    )
    mass = volume[ids] * fraction[ids]
    i, j, weight = active_graph(points, tets, ids, region, fraction)
    assert np.all(weight > 0) and np.all(region[i] == region[j])
    graph = coo_array(
        (np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(len(ids), len(ids))
    ).tocsr()
    nc, component = connected_components(graph, directed=False)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    fixture_summary = json.loads((cfg.fixture / "summary.json").read_text())
    region_rows = fixture_summary["fiber_estimate"]["regions"]
    names = {r["control_id"]: r["name"] for r in region_rows}
    rows = []
    for r in np.unique(region):
        selected = region == r
        comp = np.unique(component[selected])
        cmass = np.bincount(component[selected], weights=mass[selected], minlength=nc)
        rows.append(
            dict(
                region=int(r),
                name=names[int(r)],
                cells=int(selected.sum()),
                edges=int((region[i] == r).sum()),
                components=len(comp),
                isolated_cells=int((selected & (degree == 0)).sum()),
                isolated_mass_fraction=float(
                    mass[selected & (degree == 0)].sum() / mass[selected].sum()
                ),
                largest_component_mass_fraction=float(
                    cmass.max() / mass[selected].sum()
                ),
            )
        )
    summary = dict(
        n_active=len(ids),
        raw6_controls=6 * len(ids),
        g5_controls=5 * len(ids),
        fiber_controls=len(ids),
        edges=len(i),
        components=nc,
        regions=len(rows),
        isolated_cells=int((degree == 0).sum()),
        isolated_mass_fraction=float(mass[degree == 0].sum() / mass.sum()),
        edge_weight_quantiles=dict(
            zip(
                ["min", "q01", "median", "q99", "max"],
                np.quantile(weight, [0, 0.01, 0.5, 0.99, 1]).tolist(),
                strict=True,
            )
        ),
        rows=rows,
        interpretation="A finite smoothness weight couples shared-face neighbors inside each named muscle without reducing the number of optimization coordinates. Each disconnected component retains a constant null mode; isolated cells have no graph penalty.",
        input_sha256=hashlib.sha256(
            (cfg.fixture / "volume.vtu").read_bytes()
        ).hexdigest(),
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    sources = out / "sources"
    sources.mkdir()
    for name in (Path(__file__).name, "face_physics.py", "experiment_profile.py"):
        shutil.copy2(Path(__file__).parent / name, sources / name)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2) + "\n")
    cherries.log_metrics(
        {
            "graph/active_cells": len(ids),
            "graph/edges": len(i),
            "graph/components": nc,
            "graph/isolated_mass_fraction": summary["isolated_mass_fraction"],
        }
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "n_active",
                    "edges",
                    "components",
                    "isolated_cells",
                    "isolated_mass_fraction",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
