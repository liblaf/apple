"""Add the exact target skin to the collected comparison meshes."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    comparison: Path = cherries.input("98-aligned-errors/summary.json")
    output: Path = cherries.output("five-deformed-meshes/00-target.vtp", mkdir=True)
    manifest: Path = cherries.output("five-deformed-meshes/target-manifest.json")


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def main(cfg: Config) -> None:
    assert not cfg.output.exists(), cfg.output
    assert not cfg.manifest.exists(), cfg.manifest
    previous = json.loads(cfg.comparison.read_text())
    inputs = []
    for name in ("volume.vtu", "skin.vtp"):
        item = next(x for x in previous["inputs"] if Path(x["path"]).name == name)
        assert record(Path(item["path"])) == item
        inputs.append(item)
    volume, skin = [pv.read(item["path"]) for item in inputs]
    rest = np.asarray(volume.points)
    target_displacement = np.asarray(volume.point_data["Smile"])
    valid = np.isfinite(target_displacement).all(axis=1)
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    assert len(np.unique(ids)) == len(ids)
    assert np.array_equal(skin.points, rest[ids])
    assert valid[ids].all()
    target = rest[ids] + target_displacement[ids]
    for state in previous["states"]:
        item = state["error_skin"]
        assert record(Path(item["path"])) == item
        comparison_skin = pv.read(item["path"])
        assert np.array_equal(comparison_skin.point_data["GlobalPointId"], ids)
        assert np.array_equal(comparison_skin.point_data["TargetPosition"], target)
    triangles = np.asarray(skin.faces).copy()
    skin.points = target
    skin.point_data["RestPosition"] = rest[ids]
    skin.point_data["Displacement"] = target_displacement[ids]
    skin.point_data["TargetPosition"] = target
    skin.point_data["PointToPointErrorMm"] = np.zeros(len(ids))
    skin.save(cfg.output, binary=True)
    reopened = pv.read(cfg.output)
    assert np.isfinite(reopened.points).all()
    assert np.array_equal(reopened.points, target)
    assert np.array_equal(reopened.faces, triangles)
    assert np.array_equal(reopened.point_data["GlobalPointId"], ids)
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    counts = {
        "surface_vertices": skin.n_points,
        "surface_cells": skin.n_cells,
        "volume_vertices_without_target": int(np.sum(~valid)),
        "tetrahedra_touching_missing_targets": int(np.sum(~valid[tets].all(axis=1))),
    }
    summary = {
        "status": "completed",
        "target_surface": record(cfg.output),
        "definition": "X + Smile at each surface GlobalPointId",
        "coordinate_units": "m",
        "counts": counts,
        "inputs": inputs + [record(cfg.comparison)],
        "source": record(Path(__file__)),
        "matches_target_positions_of_all_five_results_exactly": True,
        "tetmesh": {
            "exported": False,
            "reason": "Source target displacement is nonfinite at 33200 volume vertices; missing positions are not imputed.",
        },
    }
    cfg.manifest.write_text(json.dumps(summary, indent=2) + "\n")
    cherries.log_metrics(counts)
    LOG.info("Exported exact target surface: %s; %s", cfg.output, counts)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
