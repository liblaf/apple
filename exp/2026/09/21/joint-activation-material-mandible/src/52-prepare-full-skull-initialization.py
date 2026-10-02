"""Test a minimum-offset soft-tissue initialization against unchanged full bones."""

from __future__ import annotations

import importlib.util
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs, _collision_geometry

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    prepared_dir: Path = GROUP / "data/prepared"
    geometry_audit: Path = (
        GROUP / "data/full-skull-initialization-audit-001/summary.json"
    )
    clearance_m: float = 1e-5
    max_iterations: int = 8
    output_dir: Path = cherries.output("full-skull-initialization", mkdir=True)


def main(cfg: Config) -> None:  # noqa: PLR0915
    assert cfg.clearance_m > 0
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    spec = importlib.util.spec_from_file_location(
        "source_bone_geometry_audit",
        Path(__file__).with_name("17-audit-source-bone-contact.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    audit = json.loads(cfg.geometry_audit.read_text())
    assert audit["schema"] == "joint-full-skull-initialization-audit-v1"
    geometry_path = Path(audit["geometry"]["path"])
    assert sha256(geometry_path) == audit["geometry"]["sha256"]
    with np.load(geometry_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    assert audit["input_arrays_sha256"] == sha256(cfg.prepared_dir / "inputs.npz")
    volume = pv.read(prepared.volume_path)
    points = arrays["fem_reference_points_m"]
    assert np.array_equal(points, volume.points)
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    dm = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    dm_inv = np.linalg.inv(dm)
    cell_volume = np.linalg.det(dm) / 6
    assert np.all(cell_volume > 0)
    soft_ids = arrays["soft_global_ids"]
    fixed = arrays["fixed_global_ids"]
    assert not np.intersect1d(soft_ids, fixed).size
    soft_faces = arrays["soft_faces"]
    bones = {}
    for name in ("cranium", "mandible"):
        faces = arrays[f"{name}_faces"]
        bone = pv.PolyData(
            arrays[f"{name}_points_m"], np.column_stack((np.full(len(faces), 3), faces))
        )
        bones[name] = bone.compute_normals(
            cell_normals=True,
            point_normals=False,
            auto_orient_normals=True,
            consistent_normals=True,
            split_vertices=False,
        )
        assert np.array_equal(bones[name].faces, bone.faces)
        assert np.array_equal(bones[name].points, bone.points)

    # A positive scalar graph extends the surface displacement to the volume;
    # this is an initialization proposal, not a replacement material or equilibrium.
    edges = np.concatenate(
        [tets[:, (a, b)] for a, b in itertools.combinations(range(4), 2)]
    )
    lengths_sq = np.sum((points[edges[:, 0]] - points[edges[:, 1]]) ** 2, axis=1)
    weights = np.tile(cell_volume, 6) / lengths_sq
    adjacency = sp.coo_matrix(
        (
            np.r_[weights, weights],
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(len(points), len(points)),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = sp.diags(degree) - adjacency
    boundary = np.union1d(soft_ids, fixed)
    interior = np.setdiff1d(np.arange(len(points)), boundary)
    matrix = laplacian[interior][:, interior].tocsr()
    coupling = laplacian[interior][:, boundary].tocsr()
    preconditioner = sp.diags(1 / matrix.diagonal())
    candidate_soft = points[soft_ids].copy()
    displacement = np.zeros_like(points)
    trace = []
    complete = False
    for iteration in range(cfg.max_iterations):
        for name, bone in bones.items():
            signed = geometry.signed_clearance(candidate_soft, bone)
            closest_ids, closest = bone.find_closest_cell(
                candidate_soft, return_closest_point=True
            )
            normals = np.asarray(bone.cell_normals)[closest_ids]
            direction = candidate_soft - closest
            distance = np.linalg.norm(direction, axis=1)
            nonzero = distance > 1e-12
            direction[nonzero] /= distance[nonzero, None]
            direction[nonzero & (signed < 0)] *= -1
            direction[~nonzero] = normals[~nonzero]
            gap = np.maximum(cfg.clearance_m - signed, 0)
            candidate_soft += gap[:, None] * direction
            surface = pv.PolyData(
                candidate_soft,
                np.column_stack((np.full(len(soft_faces), 3), soft_faces)),
            )
            pairs, _, _, _ = _collision_geometry(surface, bone)
            # Whole intersecting soft triangles must leave the contacted bone plane.
            for soft_face, bone_face in pairs:
                ids = soft_faces[soft_face]
                normal = np.asarray(bone.cell_normals)[bone_face]
                base = np.asarray(bone.points)[arrays[f"{name}_faces"][bone_face, 0]]
                signed_plane = (candidate_soft[ids] - base) @ normal
                correction = np.maximum(cfg.clearance_m - signed_plane, 0)
                candidate_soft[ids] += correction[:, None] * normal
        displacement.fill(0)
        displacement[soft_ids] = candidate_soft - points[soft_ids]
        for axis in range(3):
            rhs = -(coupling @ displacement[boundary, axis])
            solution, info = spla.cg(
                matrix, rhs, M=preconditioner, rtol=1e-8, atol=0, maxiter=2000
            )
            assert info == 0, info
            displacement[interior, axis] = solution
        x = points + displacement
        ds = np.transpose(x[tets[:, 1:]] - x[tets[:, :1]], (0, 2, 1))
        determinant = np.linalg.det(ds @ dm_inv)
        contacts = {}
        surface = pv.PolyData(
            x[soft_ids], np.column_stack((np.full(len(soft_faces), 3), soft_faces))
        )
        for name, bone in bones.items():
            pairs, _, _, _ = _collision_geometry(surface, bone)
            signed = geometry.signed_clearance(x[soft_ids], bone)
            contacts[name] = {
                "intersection_pairs": len(pairs),
                "minimum_node_signed_distance_m": float(signed.min()),
            }
        obs = prepared.arrays["observation_node_ids"]
        weights_obs = prepared.arrays["observation_weight_normalized"]
        row = {
            "iteration": iteration + 1,
            "detF_min": float(determinant.min()),
            "detF_max": float(determinant.max()),
            "inverted_tetrahedra": int(np.sum(determinant <= 0)),
            "maximum_nodal_displacement_m": float(
                np.linalg.norm(displacement, axis=1).max()
            ),
            "surface_motion_rms_mm": float(
                np.sqrt(np.sum(weights_obs * np.sum(displacement[obs] ** 2, axis=1)))
                * 1000
            ),
            "contacts": contacts,
        }
        trace.append(row)
        write_json(cfg.output_dir / "trace.json", trace)
        LOG.info("initialization candidate: %s", row)
        complete = (
            all(
                value["intersection_pairs"] == 0
                and value["minimum_node_signed_distance_m"] > 0
                for value in contacts.values()
            )
            and row["detF_min"] >= 0.25
            and row["detF_max"] <= 2
            and row["surface_motion_rms_mm"] <= 0.25
        )
        if complete:
            break
    state_path = cfg.output_dir / "candidate.npz"
    np.savez_compressed(state_path, initial_displacement_m=displacement)
    write_json(
        cfg.output_dir / "summary.json",
        {
            "schema": "joint-full-skull-initialization-candidate-v1",
            "geometry_audit_sha256": sha256(cfg.geometry_audit),
            "geometry_sha256": sha256(geometry_path),
            "candidate": {
                "path": str(state_path.resolve()),
                "sha256": sha256(state_path),
            },
            "clearance_target_m": cfg.clearance_m,
            "method": "outward surface projection and positive graph harmonic volume extension",
            "all_source_triangles_retained": True,
            "bone_coordinates_changed": False,
            "fixed_original_nodes_changed": bool(np.any(displacement[fixed] != 0)),
            "geometric_candidate_passed": complete,
            "full_skull_contact_admitted": False,
            "ipc_candidate_validation_pending": True,
            "equilibrium_converged": False,
            "trace": trace,
        },
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
