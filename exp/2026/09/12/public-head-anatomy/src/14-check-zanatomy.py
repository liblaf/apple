# Copyright 2026 liblaf
"""Check topology, material dependence, and bilateral symmetry of Z-Anatomy."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import trimesh as tm
from anatomy_common import ProfileCometNoCommit, sha256, write_json
from scipy.spatial import cKDTree

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    atlas: Path = cherries.input("12-public-models/zanatomy/extracted")
    output: Path = cherries.output(
        "12-public-models/zanatomy/geometry-qa.json", mkdir=True
    )


def edge_incidence(mesh: tm.Trimesh) -> np.ndarray:
    return np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))


def topology_record(
    atlas: Path, record: dict[str, object]
) -> tuple[dict[str, object], tm.Trimesh]:
    path = atlas / str(record["output_ply"])
    assert sha256(path) == record["output_ply_sha256"]
    loaded = tm.load_mesh(path, process=False)
    assert isinstance(loaded, tm.Trimesh)
    assert len(loaded.vertices) == record["vertex_count"]
    assert len(loaded.faces) == record["triangle_count"]
    incidence = edge_incidence(loaded)
    result = {
        "source_object_name": record["source_object_name"],
        "role": record["role"],
        "side_label": record["side_label"],
        "output_ply": record["output_ply"],
        "output_ply_sha256": record["output_ply_sha256"],
        "vertex_count": len(loaded.vertices),
        "triangle_count": len(loaded.faces),
        "connected_component_count": int(loaded.body_count),
        "watertight": bool(loaded.is_watertight),
        "winding_consistent": bool(loaded.is_winding_consistent),
        "boundary_edge_count": int(np.count_nonzero(incidence == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(incidence > 2)),
        "degenerate_face_count": int(np.count_nonzero(~loaded.nondegenerate_faces())),
        "duplicate_face_count": int(np.count_nonzero(~loaded.unique_faces())),
        "materials": record["materials"],
        "image_textures": record["image_textures"],
    }
    return result, loaded


def vertex_set_digest(mesh: tm.Trimesh) -> str:
    vertices = np.round(np.asarray(mesh.vertices), decimals=8)
    vertices = vertices[np.lexsort(vertices.T[::-1])]
    return hashlib.sha256(vertices.tobytes()).hexdigest()


def mirrored_pair_record(
    base_name: str, right: tm.Trimesh, left: tm.Trimesh, role: str
) -> dict[str, object]:
    right_points = np.asarray(right.vertices)
    reflected_left = np.asarray(left.vertices).copy()
    reflected_left[:, 0] *= -1
    right_to_left = cKDTree(reflected_left).query(right_points)[0]
    left_to_right = cKDTree(right_points).query(reflected_left)[0]
    symmetric_rms = np.sqrt((np.mean(right_to_left**2) + np.mean(left_to_right**2)) / 2)
    return {
        "base_name": base_name,
        "role": role,
        "right_name": f"{base_name}.r",
        "left_name": f"{base_name}.l",
        "right_vertex_count": len(right_points),
        "left_vertex_count": len(reflected_left),
        "reflected_nearest_vertex_distance_um": {
            "right_to_left_max": float(right_to_left.max() * 1e6),
            "left_to_right_max": float(left_to_right.max() * 1e6),
            "symmetric_rms": float(symmetric_rms * 1e6),
        },
    }


def role_summaries(rows: list[dict[str, object]]) -> dict[str, object]:
    summaries = {}
    for role in sorted({str(row["role"]) for row in rows}):
        selected = [row for row in rows if row["role"] == role]
        components = Counter(int(row["connected_component_count"]) for row in selected)
        summaries[role] = {
            "mesh_count": len(selected),
            "watertight_mesh_count": sum(bool(row["watertight"]) for row in selected),
            "winding_consistent_mesh_count": sum(
                bool(row["winding_consistent"]) for row in selected
            ),
            "boundary_edge_count": sum(
                int(row["boundary_edge_count"]) for row in selected
            ),
            "nonmanifold_edge_count": sum(
                int(row["nonmanifold_edge_count"]) for row in selected
            ),
            "degenerate_face_count": sum(
                int(row["degenerate_face_count"]) for row in selected
            ),
            "duplicate_face_count": sum(
                int(row["duplicate_face_count"]) for row in selected
            ),
            "connected_component_count_histogram": {
                str(key): value for key, value in sorted(components.items())
            },
        }
    return summaries


def main(cfg: Config) -> None:
    manifest_path = cfg.atlas / "zanatomy-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["outputs"]["mesh_count"] == 61
    topology, meshes = [], {}
    for record in manifest["objects"]:
        result, mesh = topology_record(cfg.atlas, record)
        topology.append(result)
        meshes[str(record["source_object_name"])] = mesh

    digest_groups: dict[tuple[int, int, str], list[str]] = defaultdict(list)
    for name, mesh in meshes.items():
        key = (len(mesh.vertices), len(mesh.faces), vertex_set_digest(mesh))
        digest_groups[key].append(name)
    exact_duplicates = [names for names in digest_groups.values() if len(names) > 1]

    roles = {str(row["source_object_name"]): str(row["role"]) for row in topology}
    mirrored_pairs = []
    for right_name, right in meshes.items():
        if not right_name.endswith(".r"):
            continue
        base_name = right_name[:-2]
        left_name = f"{base_name}.l"
        if left_name in meshes:
            assert roles[right_name] == roles[left_name]
            mirrored_pairs.append(
                mirrored_pair_record(
                    base_name, right, meshes[left_name], roles[right_name]
                )
            )

    material_counts = Counter(
        str(material)
        for record in manifest["objects"]
        for material in record["materials"]
    )
    textured = [
        str(record["source_object_name"])
        for record in manifest["objects"]
        if record["image_textures"]
    ]
    mirror_threshold_um = 1.0
    mirror_counts = {}
    for role in sorted({str(pair["role"]) for pair in mirrored_pairs}):
        selected = [pair for pair in mirrored_pairs if pair["role"] == role]
        mirror_counts[role] = {
            "pair_count": len(selected),
            "pairs_with_max_distance_below_1_um": sum(
                max(
                    pair["reflected_nearest_vertex_distance_um"]["right_to_left_max"],
                    pair["reflected_nearest_vertex_distance_um"]["left_to_right_max"],
                )
                < mirror_threshold_um
                for pair in selected
            ),
        }

    result = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "methods": {
            "topology": "trimesh process=False; edge incidence one is boundary and greater than two is nonmanifold",
            "exact_duplicate": "SHA-256 of lexicographically sorted vertices rounded to 1e-8 m, grouped with equal vertex and triangle counts",
            "mirror": "reflect each left mesh across atlas x=0 and compute bidirectional nearest-vertex distances",
            "mirror_threshold_um": mirror_threshold_um,
        },
        "summary_by_role": role_summaries(topology),
        "mirror_summary_by_role": mirror_counts,
        "exact_in_place_vertex_set_duplicate_groups": exact_duplicates,
        "material_slot_counts": dict(sorted(material_counts.items())),
        "objects_with_image_textures": textured,
        "objects": topology,
        "mirrored_pairs": mirrored_pairs,
        "limitations": [
            "Topology checks describe the extracted polygon surfaces; they do not validate anatomical accuracy.",
            "Nearest-vertex mirror distances test geometric symmetry, not source specimen identity.",
            "Open fascia sheets can serve as references but are not closed simulation volumes.",
            "Material slot names are labels; no selected mesh references an image texture.",
        ],
    }
    write_json(cfg.output, result)
    cherries.log_metrics(
        {
            "meshes": len(topology),
            "watertight_muscles": result["summary_by_role"]["facial_expression_muscle"][
                "watertight_mesh_count"
            ],
            "watertight_fascia": result["summary_by_role"][
                "head_fascia_or_aponeurosis"
            ]["watertight_mesh_count"],
            "mirrored_muscle_pairs_below_1um": mirror_counts[
                "facial_expression_muscle"
            ]["pairs_with_max_distance_below_1_um"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
