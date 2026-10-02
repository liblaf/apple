"""Measure full-source skull compatibility without altering source geometry."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pyvista as pv
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs, _collision_geometry

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    prepared_dir: Path = GROUP / "data/prepared"
    output_dir: Path = cherries.output("full-skull-initialization-audit", mkdir=True)


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    module_path = Path(__file__).with_name("17-audit-source-bone-contact.py")
    spec = importlib.util.spec_from_file_location(
        "source_bone_geometry_audit", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    volume = pv.read(prepared.volume_path)
    points = np.asarray(volume.points, dtype=np.float64)
    surfaces = geometry.boundary_partition(volume, points)
    soft = surfaces["soft"]
    soft_ids = np.asarray(soft.point_data["GlobalPointId"], dtype=np.int64)
    fixed = np.unique(
        np.concatenate(
            [
                prepared.arrays["historical_fixed_node_ids"],
                prepared.arrays["cranium_node_ids"],
                prepared.arrays["mandible_node_ids"],
            ]
        )
    )
    soft_fixed = np.isin(soft_ids, fixed)
    arrays = {
        "fem_reference_points_m": points,
        "soft_global_ids": soft_ids,
        "soft_faces": np.asarray(soft.faces).reshape(-1, 4)[:, 1:].astype(np.int64),
        "fixed_global_ids": fixed,
        "mandible_pivot_m": prepared.arrays["mandible_pivot_m"],
    }
    results = {}
    bones = {}
    for name, source_key in (
        ("cranium", "cranium_surface"),
        ("mandible", "mandible_surface"),
    ):
        record = prepared.manifest["sources"][source_key]
        path = Path(record["path"])
        bone = pv.read(path).triangulate()
        assert bone.is_manifold
        assert bone.n_open_edges == 0
        bones[name] = bone
        signed = geometry.signed_clearance(
            np.asarray(soft.points, dtype=np.float64), bone
        )
        cell_ids, closest = bone.find_closest_cell(
            soft.points, return_closest_point=True
        )
        pairs, segments, _, _ = _collision_geometry(soft, bone)
        tolerance = geometry.source_tolerance(bone, points)
        penetrating = signed < -tolerance
        arrays.update(
            {
                f"{name}_points_m": np.asarray(bone.points, dtype=np.float64),
                f"{name}_faces": np.asarray(bone.faces)
                .reshape(-1, 4)[:, 1:]
                .astype(np.int64),
                f"{name}_source_vertex_ids": np.arange(bone.n_points, dtype=np.int64),
                f"{name}_source_triangle_ids": np.arange(bone.n_cells, dtype=np.int64),
                f"{name}_soft_signed_distance_m": signed,
                f"{name}_soft_closest_points_m": np.asarray(closest, dtype=np.float64),
                f"{name}_soft_closest_triangle_ids": np.asarray(
                    cell_ids, dtype=np.int64
                ),
                f"{name}_intersection_pairs": pairs,
                f"{name}_intersection_segments_m": segments,
            }
        )
        results[name] = {
            "source": {"path": str(path), "sha256": sha256(path)},
            "vertices": bone.n_points,
            "triangles": bone.n_cells,
            "all_source_triangles_retained": True,
            "source_coordinates_changed": False,
            "soft_signed_clearance_mm": geometry.quantiles(signed * 1000),
            "strictly_inside_soft_nodes": int(np.sum(signed < 0)),
            "inside_beyond_source_tolerance_soft_nodes": int(penetrating.sum()),
            "inside_beyond_tolerance_fixed_soft_nodes": int(
                np.sum(penetrating & soft_fixed)
            ),
            "source_tolerance_m": tolerance,
            "raw_soft_bone_triangle_intersection_pairs": len(pairs),
            "intersecting_soft_triangles": len(np.unique(pairs[:, 0])),
        }
        LOG.info("%s: %s", name, results[name])
    bone_pairs, _, _, _ = _collision_geometry(bones["cranium"], bones["mandible"])
    arrays["bone_bone_intersection_pairs"] = bone_pairs
    geometry_path = cfg.output_dir / "geometry.npz"
    np.savez_compressed(geometry_path, **arrays)
    write_json(
        cfg.output_dir / "summary.json",
        {
            "schema": "joint-full-skull-initialization-audit-v1",
            "input_arrays_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
            "input_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
            "units": "metres",
            "frame": "unchanged registered source and FEM world frame",
            "fem_nodes": len(points),
            "soft_boundary_nodes": len(soft_ids),
            "soft_boundary_triangles": soft.n_cells,
            "fixed_soft_boundary_nodes": int(soft_fixed.sum()),
            "bones": results,
            "raw_cranium_mandible_intersection_pairs": len(bone_pairs),
            "geometry": {
                "path": str(geometry_path.resolve()),
                "sha256": sha256(geometry_path),
            },
            "initialization_repaired": False,
            "full_skull_contact_admitted": False,
            "purpose": "all-source-triangle compatibility audit; no mechanics or geometry mutation",
        },
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
