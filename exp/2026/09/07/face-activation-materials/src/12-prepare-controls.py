# ruff: noqa: EM101, EM102, TRY003
"""Prepare a target-independent four-control basis for each named muscle."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from control_basis import (
    CONTROLS_PER_REGION,
    LOCAL_CONTROL_LABELS,
    build_control_basis,
)
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    """Pinned fixture and Cherries-managed control-basis outputs."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = cherries.input("10-fixture")
    output_npz: Path = cherries.output("12-controls.npz", mkdir=True)
    output_json: Path = cherries.output("12-controls.json", mkdir=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def main(cfg: Config) -> None:
    for output in (cfg.output_npz, cfg.output_json):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite control basis: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    fixture_paths = {
        name: cfg.fixture / name for name in ("volume.vtu", "skin.vtp", "summary.json")
    }
    mesh = pv.read(fixture_paths["volume.vtu"])
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError("fixture volume must be an UnstructuredGrid")
    points = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:].astype(np.int64, copy=False)
    activation_mask = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    active_ids = np.flatnonzero(activation_mask).astype(np.int64, copy=False)
    active_region_ids = np.asarray(
        mesh.cell_data["ActivationControlId"], dtype=np.int64
    )[active_ids]
    active_muscle_ids = np.asarray(mesh.cell_data["MuscleId"], dtype=np.int64)[
        active_ids
    ]
    active_fraction = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)[
        active_ids
    ]
    basis = build_control_basis(
        points,
        tets,
        active_ids,
        active_region_ids,
        active_muscle_ids,
        active_fraction,
    )

    field_region_muscle_ids = np.asarray(
        mesh.field_data["ActivationRegionMuscleId"], dtype=np.int64
    )
    region_muscle_ids = np.asarray(
        [
            np.unique(active_muscle_ids[active_region_ids == region_id]).item()
            for region_id in range(basis.n_regions)
        ],
        dtype=np.int64,
    )
    if not np.array_equal(region_muscle_ids, field_region_muscle_ids):
        raise ValueError(
            "region-to-muscle ordering differs from the fixture field data"
        )
    region_names = [str(name) for name in mesh.field_data["ActivationRegionName"]]
    if len(region_names) != basis.n_regions:
        raise ValueError("fixture region names do not match the active region count")

    arrays = basis.arrays()
    arrays["region_muscle_ids"] = region_muscle_ids
    np.savez_compressed(cfg.output_npz, **arrays)
    regions = []
    for region_id, (muscle_id, name) in enumerate(
        zip(region_muscle_ids, region_names, strict=True)
    ):
        rows = active_region_ids == region_id
        controls = slice(
            CONTROLS_PER_REGION * region_id,
            CONTROLS_PER_REGION * (region_id + 1),
        )
        regions.append(
            {
                "region_id": region_id,
                "muscle_id": int(muscle_id),
                "name": name,
                "active_cells": int(rows.sum()),
                "active_mass_m3": float(basis.active_mass[rows].sum()),
                "centroid_m": basis.region_centroids[region_id].tolist(),
                "pca_eigenvalues_m2_descending": basis.region_eigenvalues[
                    region_id
                ].tolist(),
                "major_minor_axes": basis.region_axes[region_id].tolist(),
                "sigma_m": float(basis.region_sigma[region_id]),
                "control_ids": basis.control_ids[controls].tolist(),
                "control_labels": list(LOCAL_CONTROL_LABELS),
                "control_centers_m": basis.control_centers[controls].tolist(),
                "parameter_mass_m3": basis.parameter_mass[controls].tolist(),
                "weight_min": float(basis.weights[rows].min()),
                "weight_max": float(basis.weights[rows].max()),
            }
        )

    source_paths = {
        name: Path(__file__).parent / name
        for name in ("12-prepare-controls.py", "control_basis.py")
    }
    metadata = {
        "schema_version": 1,
        "design": "four-control weighted-PCA Gaussian partition per named muscle",
        "claim": (
            "smooth kinematic spatial modes for later scalar fibers or log tensors; "
            "not anatomical compartments"
        ),
        "target_independence": {
            "target_used": False,
            "reason": (
                "the pure builder accepts only rest points, tetrahedra, active/region/"
                "muscle IDs, and MuscleFraction"
            ),
            "consumed_fixture_arrays": [
                "points",
                "tetrahedra",
                "ActivationMask",
                "ActivationControlId",
                "MuscleId",
                "MuscleFraction",
                "field_data/ActivationRegionMuscleId",
                "field_data/ActivationRegionName",
            ],
            "consumed_target_arrays": [],
        },
        "formula": {
            "cell_location": "rest tetrahedron vertex mean",
            "cell_mass": "rest tetrahedron volume * MuscleFraction",
            "pca": "mass-weighted covariance about each region centroid",
            "axis_sign": "largest-absolute Cartesian component is positive",
            "control_order": list(LOCAL_CONTROL_LABELS),
            "centers": (
                "centroid +/- sqrt(lambda_major)*major and centroid +/- "
                "sqrt(lambda_minor)*minor"
            ),
            "sigma": "sqrt((lambda_major + lambda_minor) / 2)",
            "unnormalized_weight": "exp(-||cell_center-control_center||^2/(2*sigma^2))",
            "weight": "unnormalized weight divided by the four-row sum",
            "parameter_mass": "sum_cell(cell_mass * weight)",
        },
        "counts": {
            "active_cells": int(active_ids.size),
            "regions": basis.n_regions,
            "controls_per_region": CONTROLS_PER_REGION,
            "global_controls": basis.n_controls,
        },
        "validations": {
            "active_ids_strictly_increasing": True,
            "region_ids_contiguous": True,
            "one_muscle_id_per_region": True,
            "weights_finite_nonnegative": True,
            "partition_max_abs_error": float(
                np.max(np.abs(basis.weights.sum(axis=1) - 1.0))
            ),
            "uniform_regional_controls_recover_regional_field": True,
            "convex_interpolation_preserves_bounds": True,
            "all_parameter_masses_positive": bool(np.all(basis.parameter_mass > 0.0)),
            "mass_conservation_relative_error": float(
                abs(basis.parameter_mass.sum() - basis.active_mass.sum())
                / basis.active_mass.sum()
            ),
        },
        "hashes": {
            "fixture_files": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for name, path in fixture_paths.items()
            },
            "geometry_inputs": {
                "points": array_sha256(points),
                "tetrahedra": array_sha256(tets),
                "active_ids": array_sha256(active_ids),
                "active_region_ids": array_sha256(active_region_ids),
                "active_muscle_ids": array_sha256(active_muscle_ids),
                "active_fraction": array_sha256(active_fraction),
            },
            "sources": {name: file_sha256(path) for name, path in source_paths.items()},
            "output_npz": file_sha256(cfg.output_npz),
        },
        "regions": regions,
    }
    cfg.output_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cherries.log_metrics(
        {
            "controls/active_cells": active_ids.size,
            "controls/regions": basis.n_regions,
            "controls/global_controls": basis.n_controls,
            "controls/partition_error": metadata["validations"][
                "partition_max_abs_error"
            ],
            "controls/min_parameter_mass_m3": float(basis.parameter_mass.min()),
        }
    )
    LOG.info("Wrote %s", cfg.output_npz)
    LOG.info("Wrote %s", cfg.output_json)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
