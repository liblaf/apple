"""Build prescribed heterogeneous skin fields from reviewed literature anchors."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_data import array_sha256

from liblaf import cherries


class Config(cherries.BaseConfig):
    prepared_dir: Path = GROUP / "data/simple-skin-forward-inputs-001/prepared"
    anchor_review: Path = GROUP / "data/simple-skin-anchor-review-001/anchors.json"
    literature_report: Path = GROUP / "docs/66-skin-forward-literature.md"
    gaussian_sigma_m: float = 0.020
    skin_poisson_ratio: float = 0.46
    thickness_m: float = 0.001
    output_dir: Path = cherries.output("simple-skin-forward-inputs", mkdir=True)


def _array_records(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": array_sha256(value),
        }
        for name, value in arrays.items()
    }


def main(cfg: Config) -> None:  # noqa: PLR0915 - linear fail-fast artifact build.
    assert cfg.gaussian_sigma_m == 0.020
    assert cfg.skin_poisson_ratio == 0.46
    assert cfg.thickness_m == 0.001
    assert cfg.output_dir.is_dir(), (
        "skin field must append to the preserved simple-forward preparation bundle"
    )
    assert cfg.prepared_dir.resolve() == (cfg.output_dir / "prepared").resolve(), (
        "skin field must bind the mechanics inputs copied into its output bundle"
    )
    geometry_provenance_path = cfg.output_dir / "provenance.json"
    assert geometry_provenance_path.is_file()
    geometry_provenance = json.loads(geometry_provenance_path.read_text())

    prepared_manifest_path = cfg.prepared_dir / "manifest.json"
    prepared_npz_path = cfg.prepared_dir / "inputs.npz"
    prepared_manifest = json.loads(prepared_manifest_path.read_text())
    assert sha256(prepared_npz_path) == prepared_manifest["artifact"]["sha256"]
    skin_path = Path(prepared_manifest["sources"]["skin"]["path"])
    assert sha256(skin_path) == prepared_manifest["sources"]["skin"]["sha256"]

    anchor_review = json.loads(cfg.anchor_review.read_text())
    assert anchor_review["schema"] == "joint-simple-skin-anchor-review-v1"
    assert anchor_review["success"] is True
    anchor_skin_path = Path(anchor_review["skin_path"])
    assert anchor_review["skin_sha256"] == sha256(anchor_skin_path)
    anchors = anchor_review["anchors"]
    assert len(anchors) == 11
    anchor_points = np.asarray([row["point_m"] for row in anchors], dtype=np.float64)
    anchor_young = np.asarray([row["young_mpa"] for row in anchors], dtype=np.float64)
    anchor_prestress = np.asarray(
        [row["prestress_n_per_m"] for row in anchors], dtype=np.float64
    )
    assert np.isfinite(anchor_points).all()
    assert np.all(anchor_young > 0)
    assert np.all(anchor_prestress > 0)

    skin = pv.read(skin_path)
    anchor_skin = pv.read(anchor_skin_path)
    assert sha256(skin_path) != sha256(anchor_skin_path), (
        "rebased skin is expected to have distinct serialized bytes"
    )
    assert np.array_equal(np.asarray(skin.points), np.asarray(anchor_skin.points))
    assert np.array_equal(np.asarray(skin.faces), np.asarray(anchor_skin.faces))
    for name in ("GlobalPointId", "SourcePointId"):
        assert np.array_equal(
            np.asarray(skin.point_data[name]), np.asarray(anchor_skin.point_data[name])
        )
    points = np.asarray(skin.points, dtype=np.float64)
    packed_faces = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    assert np.all(packed_faces[:, 0] == 3)
    local_triangles = np.ascontiguousarray(packed_faces[:, 1:], dtype=np.int64)
    global_point_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    skin_triangles = np.ascontiguousarray(global_point_ids[local_triangles])
    centroids = points[local_triangles].mean(axis=1)

    difference = centroids[:, None, :] - anchor_points[None, :, :]
    squared_distance = np.einsum("fai,fai->fa", difference, difference)
    unnormalized = np.exp(
        -0.5 * squared_distance / (cfg.gaussian_sigma_m * cfg.gaussian_sigma_m)
    )
    denominators = unnormalized.sum(axis=1, keepdims=True)
    assert np.all(np.isfinite(denominators))
    assert np.all(denominators > 0)
    weights = unnormalized / denominators
    assert np.max(np.abs(weights.sum(axis=1) - 1.0)) <= 5e-15
    assert np.all(weights > 0)

    young_mpa = np.ascontiguousarray(weights @ anchor_young, dtype=np.float64)
    scalar_resultant = np.ascontiguousarray(
        weights @ anchor_prestress, dtype=np.float64
    )
    poisson = np.full(len(local_triangles), cfg.skin_poisson_ratio, dtype=np.float64)
    thickness = np.full(len(local_triangles), cfg.thickness_m, dtype=np.float64)
    baseline = np.zeros((len(local_triangles), 2, 2), dtype=np.float64)
    baseline[:, 0, 0] = scalar_resultant
    baseline[:, 1, 1] = scalar_resultant

    arrays = {
        "E_mpa": young_mpa,
        "nu": poisson,
        "h_m": thickness,
        "baseline_N_per_m": baseline,
        "skin_triangles": skin_triangles,
    }
    assert set(arrays) == {
        "E_mpa",
        "nu",
        "h_m",
        "baseline_N_per_m",
        "skin_triangles",
    }
    assert all(
        value.dtype == np.float64
        for value in arrays.values()
        if value is not skin_triangles
    )
    assert skin_triangles.dtype == np.int64
    assert np.isfinite(young_mpa).all()
    assert np.all(young_mpa > 0)
    assert np.isfinite(baseline).all()
    assert np.all(scalar_resultant > 0)

    artifact_path = cfg.output_dir / "skin-field.npz"
    manifest_path = cfg.output_dir / "skin-field-manifest.json"
    assert not manifest_path.exists(), manifest_path
    temporary = artifact_path.with_suffix(".npz.tmp")
    if artifact_path.exists():
        with np.load(artifact_path, allow_pickle=False) as loaded:
            assert set(loaded.files) == set(arrays)
            for name, expected in arrays.items():
                actual = np.asarray(loaded[name])
                assert actual.dtype == expected.dtype
                assert np.array_equal(actual, expected)
    else:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(artifact_path)

    manifest = {
        "schema": "joint-prescribed-skin-field-v1",
        "success": True,
        "status": (
            "prescribed Flynn-derived scalar regional transfer; manual anchors and "
            "interpolation assumptions, not a measured registered map"
        ),
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": sha256(artifact_path),
        },
        "prepared_inputs": {
            "npz_path": str(prepared_npz_path.resolve()),
            "npz_sha256": sha256(prepared_npz_path),
            "manifest_path": str(prepared_manifest_path.resolve()),
            "manifest_sha256": sha256(prepared_manifest_path),
        },
        "skin_reference": {
            "path": str(skin_path.resolve()),
            "sha256": sha256(skin_path),
            "anchor_source_path": str(anchor_skin_path.resolve()),
            "anchor_source_sha256": sha256(anchor_skin_path),
            "equivalence": (
                "exact points, packed triangle connectivity, GlobalPointId, and "
                "SourcePointId arrays"
            ),
        },
        "arrays": _array_records(arrays),
        "constitutive_contract": {
            "law": "exact plane-stress polynomial stable Neo-Hookean membrane",
            "baseline_units": "N/m",
            "baseline_frame": "frozen neutral per-triangle tangent frame",
            "elastic_modulus_units": "MPa",
            "thickness_units": "m",
            "triangle_order": "skin_triangles exact global FEM point ids",
        },
        "field_construction": {
            "interpolation": "positive normalized isotropic 3D Gaussian weights at triangle centroids",
            "gaussian_sigma_m": cfg.gaussian_sigma_m,
            "distance_space": "Euclidean world XYZ on the frozen reference skin",
            "weight_formula": "w_fa = exp(-||c_f-p_a||^2/(2 sigma^2)) / sum_b exp(-||c_f-p_b||^2/(2 sigma^2))",
            "skin_poisson_ratio": cfg.skin_poisson_ratio,
            "thickness_m": cfg.thickness_m,
            "prestress_transfer": (
                "isotropic mean of Flynn sigma_X and sigma_Y, preserving 3D stress "
                "at the current 1 mm thickness"
            ),
            "young_transfer": (
                "derived incompressible zero-stress Ogden tangent "
                "E0=(3/2)*sum_i(mu_i*alpha_i)"
            ),
            "actual_field_ranges": {
                "E_mpa": [float(young_mpa.min()), float(young_mpa.max())],
                "baseline_N_per_m": [
                    float(scalar_resultant.min()),
                    float(scalar_resultant.max()),
                ],
                "nu": [float(poisson.min()), float(poisson.max())],
                "h_m": [float(thickness.min()), float(thickness.max())],
                "minimum_anchor_weight": float(weights.min()),
                "maximum_anchor_weight": float(weights.max()),
            },
            "interpretation": (
                "Gaussian blending is convex and smooth but does not reproduce the "
                "input site values exactly at finite sigma"
            ),
        },
        "source": {
            "citation": (
                "Flynn C, Taberner AJ, Nielsen PMF, Fels S. Simulating the "
                "three-dimensional deformation of in vivo facial skin. J Mech "
                "Behav Biomed Mater. 2013;28:484-494."
            ),
            "doi": "10.1016/j.jmbbm.2013.03.004",
            "doi_url": "https://doi.org/10.1016/j.jmbbm.2013.03.004",
            "table": "Table 3, six regional inverse Ogden-QLV fits on one volunteer",
            "literature_report_path": str(cfg.literature_report.resolve()),
            "literature_report_sha256": sha256(cfg.literature_report),
            "source_status": (
                "inverse-model estimates; derived scalar transfer values, not direct "
                "measurements on the current subject"
            ),
        },
        "anchor_review": {
            "path": str(cfg.anchor_review.resolve()),
            "sha256": sha256(cfg.anchor_review),
            "schema": anchor_review["schema"],
            "anchors": anchors,
            "mapping_status": (
                "manual anatomical candidates accepted for this diagnostic; Flynn "
                "probe centers are not registered to this subject"
            ),
            "bilateral_policy": (
                "right-side regional values mirrored to the left as an explicit "
                "symmetry assumption"
            ),
            "extrapolated_regions": [
                "nose",
                "lips and vermilion",
                "eyelids",
                "ears",
                "scalp outside the forehead anchor neighborhood",
                "all skin between or outside the six sampled anatomical sites",
            ],
        },
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "git_sha": geometry_provenance["git_sha"],
            "commit_enabled": False,
            "geometry_preparation_provenance_path": str(
                geometry_provenance_path.resolve()
            ),
            "geometry_preparation_provenance_sha256": sha256(geometry_provenance_path),
            "implementation_sha256": {
                "71-build-simple-skin-forward-inputs.py": sha256(Path(__file__)),
                "joint_common.py": sha256(Path(__file__).with_name("joint_common.py")),
                "joint_data.py": sha256(Path(__file__).with_name("joint_data.py")),
            },
        },
    }

    legacy_manifest_path = cfg.output_dir / "manifest.json"
    rejected_manifest_path = (
        cfg.output_dir / "skin-field-manifest-rejected-parent-binding.json"
    )
    if legacy_manifest_path.exists():
        assert not rejected_manifest_path.exists(), rejected_manifest_path
        rejected_manifest = json.loads(legacy_manifest_path.read_text())
        assert rejected_manifest["schema"] == "joint-prescribed-skin-field-v1"
        assert rejected_manifest["success"] is True
        assert rejected_manifest["artifact"]["sha256"] == sha256(artifact_path)
        assert rejected_manifest["prepared_inputs"]["npz_sha256"] != sha256(
            prepared_npz_path
        )
        write_json(
            rejected_manifest_path,
            {
                "schema": "joint-prescribed-skin-field-rejection-v1",
                "success": False,
                "status": (
                    "rejected: parent prepared-input hashes did not name the copied "
                    "mechanics inputs consumed by the simple forward runner"
                ),
                "rejected_manifest_sha256": sha256(legacy_manifest_path),
                "rejected_manifest": rejected_manifest,
                "correct_prepared_inputs": manifest["prepared_inputs"],
            },
        )
        legacy_manifest_path.unlink()
    write_json(manifest_path, manifest)

    with np.load(artifact_path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for name, expected in arrays.items():
            actual = np.asarray(loaded[name])
            assert actual.dtype == expected.dtype
            assert np.array_equal(actual, expected)
            assert array_sha256(actual) == manifest["arrays"][name]["sha256"]
    cherries.log_output(artifact_path)
    cherries.log_output(manifest_path)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
