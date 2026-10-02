# ruff: noqa: PLR0915
"""Independently audit the saved blendshape transfer onto neutral 005."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from liblaf import cherries

HERE = Path(__file__).resolve().parent
GROUP = HERE.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
sys.path.insert(0, str(JOINT / "src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    bundle_dir: Path = GROUP / "data/blendshapes-005"


def _record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _input(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"])
    assert path.is_file(), f"{label} is missing: {path}"
    assert sha256(path) == record["sha256"], f"{label} SHA-256 mismatch: {path}"
    return path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _triangles(mesh: pv.PolyData) -> np.ndarray:
    packed = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
    assert np.all(packed[:, 0] == 3)
    return packed[:, 1:]


def _tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    packed = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)
    assert np.all(packed[:, 0] == 4)
    return packed[:, 1:]


def _equal_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(left, right, equal_nan=True)


def main(cfg: Config) -> None:
    bundle = cfg.bundle_dir.resolve()
    output = bundle / "independent-audit.json"
    assert not output.exists(), output
    manifest_path = bundle / "manifest.json"
    manifest = _json(manifest_path)
    assert manifest["schema"] == "new-neutral-blendshape-transfer-v1"
    assert (
        manifest["topology_convention"]
        == "skin_triangles contains local zero-based indices into skin_global_ids"
    )
    sources = {
        name: _input(record, f"source {name}")
        for name, record in manifest["sources"].items()
    }
    artifacts = {
        name: _input(record, f"artifact {name}")
        for name, record in manifest["artifacts"].items()
    }
    required_sources = {
        "protocol",
        "summary",
        "endpoint",
        "independent_neutral_audit",
        "source_volume",
        "source_skin",
        "repaired_volume",
        "repaired_skin",
        "neutral_review",
        "independent_neutral_isfixed_audit",
    }
    assert required_sources <= sources.keys()
    assert {
        "blendshapes.npz",
        "neutral-with-blendshapes.vtp",
        "neutral-with-blendshapes.vtu",
        "targets.zip",
    } <= artifacts.keys()

    archive_path = artifacts["blendshapes.npz"]
    expected_keys = {
        "expression_names",
        "skin_global_ids",
        "skin_triangles",
        "source_neutral_points_m",
        "new_neutral_points_m",
        "expression_displacement_m",
        "target_points_m",
    }
    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == expected_keys
        names = [str(name) for name in archive["expression_names"].tolist()]
        global_ids = np.asarray(archive["skin_global_ids"], dtype=np.int64)
        triangles = np.asarray(archive["skin_triangles"], dtype=np.int64)
        source_neutral = np.asarray(
            archive["source_neutral_points_m"], dtype=np.float64
        )
        new_neutral = np.asarray(archive["new_neutral_points_m"], dtype=np.float64)
        displacement = np.asarray(
            archive["expression_displacement_m"], dtype=np.float64
        )
        targets = np.asarray(archive["target_points_m"], dtype=np.float64)
    assert names == manifest["expression_names"]
    assert len(names) == 36
    assert len(set(names)) == 36
    assert global_ids.ndim == 1
    assert np.all(global_ids >= 0)
    assert triangles.ndim == 2
    assert triangles.shape[1] == 3
    assert triangles.min() >= 0
    assert triangles.max() < len(global_ids)
    assert source_neutral.shape == new_neutral.shape == (len(global_ids), 3)
    assert displacement.shape == targets.shape == (len(names), len(global_ids), 3)
    assert np.isfinite(source_neutral).all()
    assert np.isfinite(new_neutral).all()

    source_volume = pv.read(sources["source_volume"])
    source_skin = pv.read(sources["source_skin"])
    repaired_volume = pv.read(sources["repaired_volume"])
    repaired_skin = pv.read(sources["repaired_skin"])
    assert np.array_equal(global_ids, source_skin.point_data["GlobalPointId"])
    assert np.array_equal(global_ids, repaired_skin.point_data["GlobalPointId"])
    assert np.array_equal(triangles, _triangles(source_skin))
    assert np.array_equal(triangles, _triangles(repaired_skin))
    assert np.array_equal(source_neutral, np.asarray(source_volume.points)[global_ids])
    with np.load(sources["endpoint"], allow_pickle=False) as endpoint:
        endpoint_u = np.asarray(endpoint["displacement_m"], dtype=np.float64)
    assert endpoint_u.shape == np.asarray(repaired_volume.points).shape
    assert np.array_equal(
        new_neutral,
        np.asarray(repaired_volume.points)[global_ids] + endpoint_u[global_ids],
    )
    assert np.array_equal(
        new_neutral, np.asarray(repaired_skin.points) + endpoint_u[global_ids]
    )

    source_fields = {
        name: np.asarray(source_volume.point_data[name], dtype=np.float64)
        for name in names
    }
    defined = np.isfinite(source_fields[names[0]]).all(axis=1)
    assert np.count_nonzero(defined) == 195460
    assert np.count_nonzero(~defined) == 33200
    for name, value in source_fields.items():
        assert np.array_equal(np.isfinite(value).all(axis=1), defined), name
    for index, name in enumerate(names):
        assert name in source_volume.point_data, name
        original = source_fields[name][global_ids]
        assert _equal_nan(displacement[index], original), name
        assert _equal_nan(targets[index], new_neutral + original), name
    target_meshes = {
        name: _input(record, f"target mesh {name}")
        for name, record in manifest["target_meshes"].items()
    }
    assert list(target_meshes) == names
    for index, name in enumerate(names):
        target = pv.read(target_meshes[name])
        assert _equal_nan(np.asarray(target.points), targets[index]), name
        assert np.array_equal(_triangles(target), triangles), name
        assert np.array_equal(target.point_data["GlobalPointId"], global_ids), name

    neutral_surface = pv.read(artifacts["neutral-with-blendshapes.vtp"])
    neutral_volume = pv.read(artifacts["neutral-with-blendshapes.vtu"])
    assert np.array_equal(np.asarray(neutral_surface.points), new_neutral)
    assert np.array_equal(_triangles(neutral_surface), triangles)
    assert np.array_equal(neutral_surface.point_data["GlobalPointId"], global_ids)
    assert np.array_equal(
        np.asarray(neutral_volume.points),
        np.asarray(repaired_volume.points) + endpoint_u,
    )
    assert np.array_equal(_tetrahedra(neutral_volume), _tetrahedra(repaired_volume))
    volume_defined = np.asarray(neutral_volume.point_data["BlendshapeDefined"])
    assert volume_defined.dtype == np.uint8
    assert np.array_equal(volume_defined, defined.astype(np.uint8))
    for name, original in source_fields.items():
        assert np.isfinite(original[global_ids]).all(), name
        assert _equal_nan(neutral_surface.point_data[name], original[global_ids]), name
        assert _equal_nan(neutral_volume.point_data[name], original), name

    archive_members = {
        "README.md": bundle / "README.md",
        "neutral-with-blendshapes.vtp": artifacts["neutral-with-blendshapes.vtp"],
    }
    archive_members.update(
        {f"targets/{name}.vtp": target_meshes[name] for name in names}
    )
    with zipfile.ZipFile(artifacts["targets.zip"]) as zip_file:
        assert set(zip_file.namelist()) == set(archive_members)
        assert len(zip_file.infolist()) == 38
        for name, path in archive_members.items():
            assert zip_file.read(name) == path.read_bytes(), name

    protocol = _json(sources["protocol"])
    summary = _json(sources["summary"])
    neutral_audit = _json(sources["independent_neutral_audit"])
    isfixed_audit = _json(sources["independent_neutral_isfixed_audit"])
    status = manifest["neutral_status"]
    assert summary["protocol"] == protocol
    assert status["solver_converged"] is True
    assert status["valid_forward"] == summary["result"]["valid_forward"]
    assert (
        status["inverted_tetrahedra"]
        == summary["result"]["geometry"]["inverted_tetrahedra"]
    )
    assert np.isclose(status["detF_min"], summary["result"]["geometry"]["detF_min"])
    assert status["valid_forward"] == summary["result"]["valid_forward"]
    assert (
        status["inverted_tetrahedra"]
        == neutral_audit["geometry"]["inverted_tetrahedra"]
    )
    assert np.isclose(status["detF_min"], neutral_audit["geometry"]["detF_min"])
    assert (
        isfixed_audit["solver_gates"]["recomputed_valid_forward"]
        == status["valid_forward"]
    )
    assert (
        isfixed_audit["geometry"]["inverted_tetrahedra"]
        == status["inverted_tetrahedra"]
    )
    assert np.isclose(isfixed_audit["geometry"]["detF_min"], status["detF_min"])
    assert isfixed_audit["isfixed_boundary"]["isfixed_equality_verified"]
    assert isfixed_audit["isfixed_boundary"]["saved_fixedmask_equals_isfixed"]
    receipt = {
        "schema": "new-neutral-blendshape-transfer-independent-audit-v1",
        "scope": "Coordinate/topology transfer audit; it does not validate expression biomechanics.",
        "manifest": _record(manifest_path),
        "sources": {name: _record(path) for name, path in sources.items()},
        "artifacts": {name: _record(path) for name, path in artifacts.items()},
        "target_meshes": {name: _record(path) for name, path in target_meshes.items()},
        "expression_count": len(names),
        "expression_names": names,
        "coordinates_and_ordering_verified": True,
        "source_deltas_preserved_exactly": True,
        "blendshape_defined_mask_matches_source_finiteness": True,
        "targets_equal_new_neutral_plus_source_delta": True,
        "zero_weight_returns_new_neutral": True,
        "neutral_status": status,
        "neutral_isfixed_boundary_verified": True,
        "transfer_physics_validation": "not performed; target geometry inherits the recorded neutral geometry status and remains a kinematic target.",
    }
    write_json(output, receipt)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "audit/expression_count": float(len(names)),
            "audit/neutral_inverted_tetrahedra": float(status["inverted_tetrahedra"]),
            "audit/neutral_valid_forward": float(status["valid_forward"]),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
