"""Preserve all 36 source expression offsets on the new neutral geometry."""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/forward-repaired-reference-005"
    review_dir: Path = GROUP / "data/review-repaired-reference-005"
    output_dir: Path = GROUP / "data/blendshapes-005"
    isfixed_audit_filename: str = "independent-audit-isfixed-verified.json"


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def verified(item: dict) -> Path:
    path = Path(item["path"]).resolve()
    assert path.is_file(), path
    assert sha256(path) == item["sha256"], path
    return path


def main(cfg: Config) -> None:  # noqa: PLR0915
    run, review, output = (
        cfg.run_dir.resolve(),
        cfg.review_dir.resolve(),
        cfg.output_dir.resolve(),
    )
    assert not output.exists(), output
    neutral_audit_path = run / "independent-audit.json"
    neutral_audit = json.loads(neutral_audit_path.read_text())
    isfixed_audit_path = run / cfg.isfixed_audit_filename
    isfixed_audit = json.loads(isfixed_audit_path.read_text())
    assert (
        isfixed_audit["endpoint"]["sha256"]
        == neutral_audit["run_inputs"]["endpoint.npz"]["sha256"]
    )
    assert isfixed_audit["solver_gates"]["recomputed_valid_forward"]
    assert isfixed_audit["isfixed_boundary"]["isfixed_equality_verified"]
    assert isfixed_audit["isfixed_boundary"]["saved_fixedmask_equals_isfixed"]
    protocol_path = verified(neutral_audit["run_inputs"]["protocol.json"])
    summary_path = verified(neutral_audit["run_inputs"]["summary.json"])
    endpoint_path = verified(neutral_audit["run_inputs"]["endpoint.npz"])
    assert protocol_path == run / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    result = summary["result"]
    assert summary["protocol"] == protocol
    assert result["success"]
    neutral_manifest_path = verified(protocol["fixture"]["neutral_manifest"])
    neutral_manifest = json.loads(neutral_manifest_path.read_text())
    source_volume_path = verified(neutral_manifest["sources"]["constitutive_volume"])
    source_skin_path = verified(neutral_manifest["sources"]["constitutive_skin"])
    source_volume, source_skin = pv.read(source_volume_path), pv.read(source_skin_path)
    configuration = protocol["reference_configuration"]
    repaired_volume_path = verified(configuration["constitutive_volume"])
    repaired_skin_path = verified(configuration["constitutive_skin"])
    repaired_volume, repaired_skin = (
        pv.read(repaired_volume_path),
        pv.read(repaired_skin_path),
    )
    np.testing.assert_array_equal(source_volume.cells, repaired_volume.cells)
    np.testing.assert_array_equal(source_skin.faces, repaired_skin.faces)
    ids = np.asarray(source_skin.point_data["GlobalPointId"], dtype=np.int64)
    np.testing.assert_array_equal(ids, repaired_skin.point_data["GlobalPointId"])
    assert np.unique(ids).size == ids.size
    np.testing.assert_array_equal(source_skin.points, source_volume.points[ids])
    np.testing.assert_array_equal(repaired_skin.points, repaired_volume.points[ids])
    names = np.asarray(source_volume.field_data["ExpressionName"], dtype=str)
    assert len(names) == 36
    assert len(set(names)) == len(names)
    vector_names = {
        name
        for name, values in source_volume.point_data.items()
        if values.shape == (source_volume.n_points, 3)
        and name not in {"FixedValue", "FixedMask"}
    }
    assert set(names) == vector_names
    with np.load(endpoint_path, allow_pickle=False) as endpoint:
        neutral_displacement = endpoint["displacement_m"].copy()
    assert neutral_displacement.shape == repaired_volume.points.shape
    assert np.isfinite(neutral_displacement).all()
    new_neutral_full = np.asarray(repaired_volume.points) + neutral_displacement
    new_neutral_skin = new_neutral_full[ids]
    deltas = np.stack(
        [np.asarray(source_volume.point_data[name])[ids] for name in names]
    )
    assert np.isfinite(deltas).all()
    targets = new_neutral_skin[None] + deltas
    coordinate_roundoff = 4 * np.finfo(targets.dtype).eps * np.abs(targets).max()
    np.testing.assert_allclose(
        targets - new_neutral_skin[None], deltas, rtol=0, atol=coordinate_roundoff
    )
    review_path = review / "receipt.json"
    review_receipt = json.loads(review_path.read_text())
    assert verified(review_receipt["run"]["endpoint"]) == endpoint_path
    np.testing.assert_array_equal(
        new_neutral_skin, pv.read(review / "neutral-skin.vtp").points
    )
    neutral_status = {
        "solver_converged": bool(result["success"]),
        "valid_forward": bool(result["valid_forward"]),
        "inverted_tetrahedra": result["geometry"]["inverted_tetrahedra"],
        "detF_min": result["geometry"]["detF_min"],
        "contact_valid": bool(result["collision"]["state_feasible"]),
    }
    faces = np.asarray(source_skin.faces).reshape(-1, 4)
    assert np.all(faces[:, 0] == 3)
    triangles = faces[:, 1:].copy()
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / Path(__file__).name)
    arrays = {
        "expression_names": names,
        "skin_global_ids": ids,
        "skin_triangles": triangles,
        "source_neutral_points_m": np.asarray(source_skin.points).copy(),
        "new_neutral_points_m": new_neutral_skin,
        "expression_displacement_m": deltas,
        "target_points_m": targets,
    }
    np.savez_compressed(output / "blendshapes.npz", **arrays)
    neutral_skin = pv.PolyData(new_neutral_skin, source_skin.faces)
    neutral_skin.point_data["GlobalPointId"] = ids
    neutral_skin.field_data["ExpressionName"] = names
    neutral_skin.field_data["CoordinateUnits"] = ["meters"]
    neutral_skin.field_data["CoordinateContract"] = [
        "new_neutral + sum(weight * original_expression_displacement)"
    ]
    neutral_skin.field_data["NeutralValidForward"] = [
        int(neutral_status["valid_forward"])
    ]
    for index, name in enumerate(names):
        neutral_skin.point_data[name] = deltas[index]
    neutral_skin.save(output / "neutral-with-blendshapes.vtp")
    # Preserve partial full-volume fields exactly, including their undefined entries.
    neutral_volume = source_volume.copy(deep=True)
    neutral_volume.points = new_neutral_full
    valid_masks = [
        np.isfinite(source_volume.point_data[name]).all(axis=1) for name in names
    ]
    assert all(np.array_equal(valid_masks[0], mask) for mask in valid_masks)
    defined = valid_masks[0]
    assert np.all(defined[ids])
    for name in names:
        field = np.asarray(source_volume.point_data[name])
        assert np.isnan(field[~defined]).all()
        np.testing.assert_array_equal(neutral_volume.point_data[name], field)
    neutral_volume.point_data["BlendshapeDefined"] = defined.astype(np.uint8)
    neutral_volume.field_data["CoordinateContract"] = [
        "deformed neutral geometry with original expression offsets; not a stress-free FEM reference"
    ]
    neutral_volume.field_data["NeutralValidForward"] = [
        int(neutral_status["valid_forward"])
    ]
    cells = np.asarray(source_volume.cells).reshape(-1, 5)[:, 1:]
    rest = np.asarray(repaired_volume.points)
    detf = np.linalg.det(
        new_neutral_full[cells[:, 1:]] - new_neutral_full[cells[:, :1]]
    ) / np.linalg.det(rest[cells[:, 1:]] - rest[cells[:, :1]])
    assert int(np.count_nonzero(detf <= 0)) == neutral_status["inverted_tetrahedra"]
    neutral_volume.cell_data["PhysicalDetF"] = detf
    neutral_volume.save(output / "neutral-with-blendshapes.vtu")
    target_dir = output / "targets"
    target_dir.mkdir()
    target_meshes, stats = {}, {}
    for index, name in enumerate(names):
        target = pv.PolyData(targets[index], source_skin.faces)
        target.point_data["GlobalPointId"] = ids
        target.point_data["ExpressionDisplacementM"] = deltas[index]
        target.field_data["ExpressionName"] = [name]
        target.field_data["CoordinateUnits"] = ["meters"]
        target.field_data["NeutralValidForward"] = [
            int(neutral_status["valid_forward"])
        ]
        path = target_dir / f"{name}.vtp"
        target.save(path)
        target_meshes[name] = record(path)
        magnitude = np.linalg.norm(deltas[index], axis=1) * 1000
        stats[name] = {
            "skin_rms_mm": float(np.sqrt(np.mean(magnitude**2))),
            "skin_max_mm": float(magnitude.max()),
        }
    readme = f"""# Blendshapes on the new neutral

36 original neutral-relative displacement fields are preserved exactly.
For skin vertex i and weights w: x_i = new_neutral_i + sum(w_j * delta_ji).
All coordinates and offsets are in meters. Faces and GlobalPointId ordering
are unchanged. Each targets/<name>.vtp is that single shape at weight 1.
neutral-with-blendshapes.vtp contains the new neutral and all 36 vector fields.
targets.zip contains the neutral VTP and all 36 target VTPs.

blendshapes.npz contains ordered names, local triangle connectivity, volume
GlobalPointIds, source/new neutral skin points, unchanged deltas, and targets.
The VTU contains the full deformed neutral volume and original vector fields.
BlendshapeDefined identifies defined full-volume offsets; undefined entries
remain NaN. All skin offsets are defined. This asset is not a stress-free FEM
reference. Transferred shapes are kinematic targets, not equilibrium solves.

The base neutral status is: valid_forward={neutral_status["valid_forward"]},
inverted_tetrahedra={neutral_status["inverted_tetrahedra"]}, and
detF_min={neutral_status["detF_min"]:.17g}.  The transfer preserves this
diagnostic status; expression contact and volume validity have not been
established. See manifest.json for exact provenance.
"""
    (output / "README.md").write_text(readme)
    with zipfile.ZipFile(
        output / "targets.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in [
            output / "README.md",
            output / "neutral-with-blendshapes.vtp",
            *sorted(target_dir.glob("*.vtp")),
        ]:
            archive.write(path, arcname=str(path.relative_to(output)))
    sources = {
        "protocol": protocol_path,
        "summary": summary_path,
        "endpoint": endpoint_path,
        "independent_neutral_audit": neutral_audit_path,
        "independent_neutral_isfixed_audit": isfixed_audit_path,
        "neutral_manifest": neutral_manifest_path,
        "source_volume": source_volume_path,
        "source_skin": source_skin_path,
        "repaired_volume": repaired_volume_path,
        "repaired_skin": repaired_skin_path,
        "neutral_review": review_path,
    }
    manifest = {
        "schema": "new-neutral-blendshape-transfer-v1",
        "transfer_success": True,
        "method": "preserve_original_expression_displacements",
        "coordinate_contract": "target_points_m = X_repaired[skin_global_ids] + u005[skin_global_ids] + original_expression_displacement_m",
        "blend_contract": "points(weights) = new_neutral_points_m + sum(weights[:, None, None] * expression_displacement_m, axis=0)",
        "coordinate_units": "meters",
        "topology_convention": "skin_triangles contains local zero-based indices into skin_global_ids",
        "expression_names": names.tolist(),
        "neutral_status": neutral_status,
        "scope": "Kinematic blendshape transfer only; no expression forward solves, contact validation, or FEM-reference adoption.",
        "sources": {key: record(path) for key, path in sources.items()},
        "artifacts": {
            path.name: record(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
        "target_meshes": target_meshes,
        "expression_statistics": stats,
        "transfer_validation": {
            "skin_vertices": len(ids),
            "skin_triangles": len(triangles),
            "expressions": len(names),
            "source_offsets_preserved_exactly": True,
            "full_volume_defined_vertices": int(defined.sum()),
            "full_volume_undefined_vertices": int((~defined).sum()),
            "full_volume_undefined_offsets_preserved_as_nan": True,
            "max_offset_reconstruction_error_m": float(
                np.abs(targets - new_neutral_skin[None] - deltas).max()
            ),
            "coordinate_roundoff_bound_m": float(coordinate_roundoff),
            "new_neutral_matches_review_exactly": True,
        },
    }
    write_json(output / "manifest.json", manifest)
    cherries.log_metrics(
        {
            "blendshapes/count": len(names),
            "blendshapes/skin_vertices": len(ids),
            "blendshapes/transfer_error_m": manifest["transfer_validation"][
                "max_offset_reconstruction_error_m"
            ],
            "neutral/valid_forward": float(result["valid_forward"]),
        }
    )
    cherries.log_output(output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
