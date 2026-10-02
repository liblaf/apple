"""Rebase the admitted repaired geometry for a passive-bulk forward experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path

import ipctk
import numpy as np
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs, array_sha256
from joint_full_skull_contact import (
    build_full_skull_contact,
    load_admitted_initialization,
    load_full_skull_geometry,
    validate_full_skull_admission,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/simple-skin-forward-inputs-001"


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main(cfg: Config) -> None:  # noqa: PLR0915 - auditable preparation sequence
    out = cfg.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    archive_sources(out)
    prepared = PreparedInputs.load()
    audit_path = GROUP / "data/full-skull-initialization-audit-001/summary.json"
    audit0 = json.loads(audit_path.read_text())
    geometry0 = load_full_skull_geometry(Path(audit0["geometry"]["path"]), audit_path)
    admission0_path = (
        GROUP / "data/full-skull-initialization-candidate-002/admission.json"
    )
    admission0 = json.loads(admission0_path.read_text())
    repair = load_admitted_initialization(admission0, geometry0)
    mesh = pv.read(prepared.volume_path)
    skin = pv.read(prepared.skin_path)
    old_points = np.asarray(mesh.points).copy()
    points = old_points + repair
    fixed = geometry0.fixed_global_ids
    observation = prepared.arrays["observation_node_ids"]
    assert np.array_equal(points[fixed], old_points[fixed])
    assert np.array_equal(points[observation], old_points[observation])
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    assert np.array_equal(points[ids], old_points[ids])
    cells = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    dm = np.transpose(points[cells[:, 1:]] - points[cells[:, :1]], (0, 2, 1))
    volumes = np.linalg.det(dm) / 6
    assert np.all(volumes > 0)
    mesh.points = points
    skin.points = points[ids]
    fixture = out / "prepared"
    fixture.mkdir()
    mesh.save(fixture / "volume.vtu")
    skin.save(fixture / "skin.vtp")
    arrays = {key: value.copy() for key, value in prepared.arrays.items()}
    arrays["active_effective_volume_m3"] = (
        volumes[arrays["active_cell_ids"]] * arrays["active_muscle_fraction"]
    )
    np.savez_compressed(fixture / "inputs.npz", **arrays)
    manifest = copy.deepcopy(prepared.manifest)
    manifest["purpose"] = (
        "standalone prescribed-skin forward; repaired soft geometry rebased as FEM reference"
    )
    for name, extension in (("volume", "vtu"), ("skin", "vtp")):
        path = fixture / f"{name}.{extension}"
        manifest["sources"][name] = record(path)
        manifest["fixture"][f"{name}_path"] = str(path)
    manifest["artifact"] = record(fixture / "inputs.npz")
    manifest["arrays"] = {
        key: {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": array_sha256(a)}
        for key, a in arrays.items()
    }
    rebase = {
        "parent_prepared_inputs": record(prepared.npz_path),
        "parent_prepared_manifest": record(prepared.manifest_path),
        "parent_geometry_admission": record(admission0_path),
        "definition": "X_new = X_original + admitted candidate002 displacement; initial u = 0",
        "source_bones_changed": False,
        "fixed_nodes_changed": False,
        "outer_skin_changed": False,
        "parent_repair_detF_range": [admission0["detF_min"], admission0["detF_max"]],
        "activation_graph": "retained historical graph is unused in this forward; not validated for a rebased inverse run",
        "joint_inverse_ready": False,
    }
    manifest["reference_rebase"] = rebase
    write_json(fixture / "manifest.json", manifest)
    PreparedInputs.load(fixture / "inputs.npz", fixture / "manifest.json")
    keys = (
        "fem_reference_points_m",
        "soft_global_ids",
        "soft_faces",
        "fixed_global_ids",
        "mandible_pivot_m",
        "cranium_points_m",
        "cranium_faces",
        "cranium_source_vertex_ids",
        "cranium_source_triangle_ids",
        "mandible_points_m",
        "mandible_faces",
        "mandible_source_vertex_ids",
        "mandible_source_triangle_ids",
    )
    geometry_arrays = {key: getattr(geometry0, key).copy() for key in keys}
    geometry_arrays["fem_reference_points_m"] = points
    np.savez_compressed(out / "geometry.npz", **geometry_arrays)
    audit = {
        key: copy.deepcopy(audit0[key])
        for key in (
            "schema",
            "units",
            "frame",
            "fem_nodes",
            "soft_boundary_nodes",
            "soft_boundary_triangles",
            "bones",
        )
    }
    # Bone records describe unchanged source assets. Old soft-tissue diagnostics are deliberately omitted.
    audit.update(
        {
            "geometry": record(out / "geometry.npz"),
            "input_arrays_sha256": sha256(fixture / "inputs.npz"),
            "input_manifest_sha256": sha256(fixture / "manifest.json"),
            "reference_rebase": rebase,
        }
    )
    write_json(out / "geometry-audit.json", audit)
    geometry = load_full_skull_geometry(
        out / "geometry.npz", out / "geometry-audit.json"
    )
    contact_config = {
        "schema": "joint-full-source-bone-contact-v1",
        "enabled": True,
        "surface_selection": "pure-soft-vs-complete-source-bones",
        "attachment_policy": "no-source-triangle-exclusions",
        "friction": "frictionless",
        "dhat_m": 0.0001,
        "stiffness_mpa": 0.01,
    }
    LOG.info("Checking full-source soft-bone contact on rebased reference")
    contact = build_full_skull_contact(geometry, contact_config).collision
    zero = torch.zeros((geometry.full_node_count, 3), dtype=torch.float64)
    state = contact.state_at(zero)
    diagnostic = contact.diagnostics(state, zero)
    intersects = bool(
        ipctk.has_intersections(
            contact.collision_mesh, contact.vertices.numpy(force=True), ipctk.LBVH()
        )
    )
    assert not intersects
    assert diagnostic["contact_numerically_valid"]
    initial = np.zeros_like(points)
    np.savez_compressed(out / "initialization.npz", initial_displacement_m=initial)
    admission = {
        "schema": "joint-full-skull-contact-admission-v2",
        "success": True,
        "scope": "rebased standalone forward initialization; no equilibrium or joint-inverse admission",
        "geometry_sha256": sha256(out / "geometry.npz"),
        "geometry_audit_sha256": sha256(out / "geometry-audit.json"),
        "initialization_path": str(out / "initialization.npz"),
        "initialization_sha256": sha256(out / "initialization.npz"),
        "initialization_displacement_sha256": hashlib.sha256(
            initial.tobytes()
        ).hexdigest(),
        "initialization_array_key": "initial_displacement_m",
        "complete_source_triangles_retained": True,
        "source_coordinates_changed": False,
        "excluded_source_triangles": 0,
        "initialization_intersection_free": True,
        "fixed_nodes_unchanged": True,
        "detF_min": 1.0,
        "detF_max": 1.0,
        "observation_surface_rms_m": 0.0,
        "ipc_initialization": diagnostic,
        "equilibrium_converged": False,
        "final_launch_ready": False,
        "reference_rebase": rebase,
    }
    validate_full_skull_admission(admission, geometry)
    write_json(out / "admission.json", admission)
    write_json(
        out / "preparation-summary.json",
        {
            "success": True,
            "reference_rebase": rebase,
            "contact": diagnostic,
            "soft_bone_intersections": intersects,
            "minimum_reference_tet_volume_m3": float(volumes.min()),
            "contact_config": contact_config,
        },
    )
    LOG.info("Prepared reference: %s", out)
    cherries.log_output(out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
