"""Separate complete-source bone crossings from soft-bone contact admission."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import ipctk
import numpy as np
import pyvista as pv
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import _collision_geometry

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    audit_path: Path = GROUP / "data/full-skull-initialization-audit-001/summary.json"
    output_dir: Path = GROUP / "data/full-skull-bone-bone-audit-001"


def plane_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    normal = np.cross(first[:, 1] - first[:, 0], first[:, 2] - first[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    return np.einsum("nvd,nd->nv", second - first[:, :1], normal)


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    audit = json.loads(cfg.audit_path.read_text())
    geometry_path = Path(audit["geometry"]["path"])
    assert sha256(geometry_path) == audit["geometry"]["sha256"]
    with np.load(geometry_path) as archive:
        a = {key: archive[key] for key in archive.files}
    spec = importlib.util.spec_from_file_location(
        "bone_pair_geometry",
        Path(__file__).with_name("17-audit-source-bone-contact.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    bones = {}
    for name in ("cranium", "mandible"):
        faces = a[f"{name}_faces"]
        bones[name] = pv.PolyData(
            a[f"{name}_points_m"], np.column_stack((np.full(len(faces), 3), faces))
        )
    pairs, segments, _, lengths = _collision_geometry(
        bones["cranium"], bones["mandible"]
    )
    assert len(pairs) == audit["raw_cranium_mandible_intersection_pairs"]
    c = a["cranium_points_m"][a["cranium_faces"][pairs[:, 0]]]
    m = a["mandible_points_m"][a["mandible_faces"][pairs[:, 1]]]
    cm, mc = plane_distances(c, m), plane_distances(m, c)
    source_tolerance = max(audit["bones"][name]["source_tolerance_m"] for name in bones)
    straddling = {}
    for label, tolerance in (
        ("double_precision", 1e-12),
        ("source_precision", source_tolerance),
    ):
        crosses = (
            (cm.min(axis=1) < -tolerance)
            & (cm.max(axis=1) > tolerance)
            & (mc.min(axis=1) < -tolerance)
            & (mc.max(axis=1) > tolerance)
            & (lengths > tolerance)
        )
        straddling[label] = {
            "tolerance_m": tolerance,
            "mutual_plane_straddling_pairs": int(crosses.sum()),
        }
    signed = {}
    for name, other in (("cranium", "mandible"), ("mandible", "cranium")):
        distance = geometry.signed_clearance(a[f"{name}_points_m"], bones[other])
        signed[name] = {
            "against": other,
            "minimum_signed_distance_mm": float(distance.min() * 1000),
            "inside_vertices_beyond_source_tolerance": int(
                np.sum(distance < -source_tolerance)
            ),
        }
    vertices = np.concatenate((a["cranium_points_m"], a["mandible_points_m"]))
    nc = len(a["cranium_points_m"])
    faces = np.concatenate((a["cranium_faces"], a["mandible_faces"] + nc)).astype(
        np.int32
    )
    mesh = ipctk.CollisionMesh(vertices, ipctk.edges(faces), faces)
    mesh.can_collide = ipctk.make_vertex_patches_filter(
        np.r_[
            np.zeros(nc, dtype=np.int32),
            np.ones(len(a["mandible_points_m"]), dtype=np.int32),
        ]
    )
    mesh.init_adjacencies()
    intersects = bool(ipctk.has_intersections(mesh, vertices, ipctk.LBVH()))
    np.savez_compressed(
        cfg.output_dir / "pairs.npz",
        pairs=pairs,
        segments_m=segments,
        lengths_m=lengths,
        cranium_plane_distances_m=cm,
        mandible_plane_distances_m=mc,
    )
    result = {
        "schema": "joint-full-skull-bone-pair-audit-v1",
        "geometry_sha256": sha256(geometry_path),
        "geometry_audit_sha256": sha256(cfg.audit_path),
        "coordinates_or_topology_changed": False,
        "raw_intersection_pairs": len(pairs),
        "intersection_segment_length_mm": geometry.quantiles(lengths * 1000),
        "mutual_plane_straddling": straddling,
        "signed_vertex_distances": signed,
        "ipc_cross_bone_intersection": intersects,
        "complete_source_bone_bone_ccd_start_valid": not intersects,
        "soft_bone_contact_implication": "none: the soft-bone adapter excludes bone-bone pairs; these remain separate jaw-domain geometry failures",
    }
    write_json(cfg.output_dir / "summary.json", result)
    LOG.info("Complete source bone-pair audit: %s", result)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
