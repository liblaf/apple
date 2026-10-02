"""Freeze every registered source-eye triangle as a rigid IPC obstacle."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyvista as pv
from joint_common import ProfileJoint, archive_sources, sha256, write_json
from joint_data import array_sha256

from liblaf import cherries


class Config(cherries.BaseConfig):
    source: Path = Path(os.environ["APPLE_MELON_HEAD"]) / "20-eye.ply"
    output_dir: Path = cherries.output("rigid-eyes-001", mkdir=True)


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    source = pv.read(cfg.source)
    faces = np.asarray(source.faces).reshape(-1, 4)
    assert np.all(faces[:, 0] == 3)
    points = np.ascontiguousarray(np.asarray(source.points, dtype="<f8"))
    triangles = np.ascontiguousarray(faces[:, 1:], dtype="<i8")
    # Connectivity is only an ID label; neither coordinates nor source triangles change.
    components = source.connectivity().point_data["RegionId"]
    unique = np.unique(components)
    remap = {value: index for index, value in enumerate(unique)}
    vertex_components = np.asarray([remap[value] for value in components], dtype="<i8")
    triangle_components = vertex_components[triangles[:, 0]]
    assert np.all(vertex_components[triangles] == triangle_components[:, None])
    arrays = {
        "points_m": points,
        "triangles": triangles,
        "source_vertex_ids": np.arange(len(points), dtype="<i8"),
        "source_triangle_ids": np.arange(len(triangles), dtype="<i8"),
        "vertex_component_ids": vertex_components,
        "triangle_component_ids": triangle_components,
    }
    path = cfg.output_dir / "eyes.npz"
    np.savez_compressed(path, **arrays)
    source.save(cfg.output_dir / "eyes.vtp")
    manifest = {
        "schema": "joint-rigid-eyes-v1",
        "success": True,
        "units": "metres",
        "frame": "unchanged registered source and FEM world frame",
        "source": record(cfg.source),
        "artifacts": {
            "eyes.npz": record(path),
            "eyes.vtp": record(cfg.output_dir / "eyes.vtp"),
        },
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": array_sha256(value),
            }
            for key, value in arrays.items()
        },
        "vertices": len(points),
        "triangles": len(triangles),
        "components": len(unique),
        "all_source_triangles_retained": True,
        "source_coordinates_changed": False,
        "component_labels": "connectivity labels only; collision uses original points_m and triangles",
        "provenance": record(cfg.output_dir / "provenance.json"),
    }
    write_json(cfg.output_dir / "manifest.json", manifest)
    cherries.log_metrics(
        {
            "eyes/vertices": len(points),
            "eyes/triangles": len(triangles),
            "eyes/components": len(unique),
        }
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
