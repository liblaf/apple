# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Fit target-independent latent-vector controls to the prepared rest fibers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import learned_fiber_models as lfm
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
RING_MUSCLE_IDS = frozenset({97, 98, 254})


class Config(cherries.BaseConfig):
    """Pinned rest fixture, four-control basis, and latent initialization outputs."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    fixture: Path = cherries.input("10-fixture")
    controls: Path = cherries.input("12-controls.npz")
    output_npz: Path = cherries.output("13-latent-controls.npz", mkdir=True)
    output_json: Path = cherries.output("13-latent-controls.json", mkdir=True)
    initial_activation: float = 0.02
    amax: float = -math.log(0.65)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot have a stable byte-level receipt")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
        + "\n",
        encoding="utf-8",
    )


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...] = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0),
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        raise ValueError("weighted quantile has no positive finite support")
    value = values[valid]
    weight = weights[valid]
    order = np.argsort(value)
    value = value[order]
    weight = weight[order]
    cumulative = np.cumsum(weight)
    cumulative = (cumulative - 0.5 * weight) / cumulative[-1]
    result = np.interp(np.asarray(probabilities), cumulative, value)
    return {
        f"q{probability:g}": float(item)
        for probability, item in zip(probabilities, result, strict=True)
    }


def log_tensor(vectors: np.ndarray) -> np.ndarray:
    amplitude = np.sum(vectors * vectors, axis=1)
    identity = np.eye(3, dtype=np.float64)[None, :, :]
    return (
        1.5 * np.einsum("ni,nj->nij", vectors, vectors)
        - 0.5 * amplitude[:, None, None] * identity
    )


def approximation_metrics(
    local: np.ndarray,
    reference: np.ndarray,
    fibers: np.ndarray,
    mass: np.ndarray,
) -> dict[str, Any]:
    total_mass = float(mass.sum())
    if total_mass <= 0.0:
        raise ValueError("approximation metric needs positive mass")
    vector_error = np.linalg.norm(local - reference, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    vector_rms = math.sqrt(float(np.sum(mass * vector_error**2) / total_mass))
    reference_vector_rms = math.sqrt(
        float(np.sum(mass * reference_norm**2) / total_mass)
    )

    local_H = log_tensor(local)
    reference_H = log_tensor(reference)
    H_error = np.linalg.norm(local_H - reference_H, axis=(1, 2))
    reference_H_norm = np.linalg.norm(reference_H, axis=(1, 2))
    H_rms = math.sqrt(float(np.sum(mass * H_error**2) / total_mass))
    reference_H_rms = math.sqrt(float(np.sum(mass * reference_H_norm**2) / total_mass))

    local_norm = np.linalg.norm(local, axis=1)
    amplitude = local_norm**2
    nonzero = local_norm > 0.0
    output: dict[str, Any] = {
        "mass_m3": total_mass,
        "vector": {
            "weighted_rms": vector_rms,
            "weighted_relative_rms": vector_rms / reference_vector_rms,
            "max": float(vector_error.max()),
        },
        "H_tensor": {
            "weighted_frobenius_rms": H_rms,
            "weighted_relative_frobenius_rms": H_rms / reference_H_rms,
            "frobenius_max": float(H_error.max()),
        },
        "local_amplitude": {
            "min": float(amplitude.min()),
            "max": float(amplitude.max()),
            "weighted_quantiles": weighted_quantiles(amplitude, mass),
        },
        "zero_local_vectors": int((~nonzero).sum()),
    }
    if np.any(nonzero):
        cosine = np.abs(
            np.einsum("ni,ni->n", local[nonzero], fibers[nonzero]) / local_norm[nonzero]
        )
        angle_degrees = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
        output["unoriented_axis_angle_degrees"] = {
            "support_cells": int(nonzero.sum()),
            "weighted_quantiles": weighted_quantiles(angle_degrees, mass[nonzero]),
        }
    return output


def main(cfg: Config) -> None:
    if not 0.0 < cfg.initial_activation <= cfg.amax or not math.isfinite(cfg.amax):
        raise ValueError("require 0 < initial_activation <= finite amax")
    for output in (cfg.output_npz, cfg.output_json):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite latent controls: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    volume_path = cfg.fixture / "volume.vtu"
    mesh = pv.read(volume_path)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError("fixture volume must be an UnstructuredGrid")
    with np.load(cfg.controls) as source:
        basis = {name: source[name].copy() for name in source.files}

    consumed_mesh_arrays = {
        "cell_data/ActivationMask": np.asarray(mesh.cell_data["ActivationMask"]),
        "cell_data/ActivationControlId": np.asarray(
            mesh.cell_data["ActivationControlId"]
        ),
        "cell_data/ActivationFiber": np.asarray(
            mesh.cell_data["ActivationFiber"], dtype=np.float64
        ),
        "cell_data/MuscleFraction": np.asarray(
            mesh.cell_data["MuscleFraction"], dtype=np.float64
        ),
        "cell_data/Volume": np.asarray(mesh.cell_data["Volume"], dtype=np.float64),
        "field_data/ActivationRegionName": np.asarray(
            mesh.field_data["ActivationRegionName"]
        ),
        "field_data/ActivationRegionMuscleId": np.asarray(
            mesh.field_data["ActivationRegionMuscleId"], dtype=np.int64
        ),
        "field_data/ExpressionName": np.asarray(mesh.field_data["ExpressionName"]),
    }
    consumed_basis_names = (
        "active_ids",
        "active_region_ids",
        "active_mass",
        "control_indices",
        "weights",
        "control_ids",
        "control_region_ids",
        "control_local_ids",
    )
    consumed_basis_arrays = {
        name: np.asarray(basis[name]) for name in consumed_basis_names
    }

    active_ids = consumed_basis_arrays["active_ids"].astype(np.int64, copy=False)
    region_ids = consumed_basis_arrays["active_region_ids"].astype(np.int64, copy=False)
    active_mass = consumed_basis_arrays["active_mass"].astype(np.float64, copy=False)
    control_indices = consumed_basis_arrays["control_indices"].astype(
        np.int64, copy=False
    )
    weights = consumed_basis_arrays["weights"].astype(np.float64, copy=False)
    control_ids = consumed_basis_arrays["control_ids"].astype(np.int64, copy=False)
    control_region_ids = consumed_basis_arrays["control_region_ids"].astype(
        np.int64, copy=False
    )
    control_local_ids = consumed_basis_arrays["control_local_ids"].astype(
        np.int64, copy=False
    )
    activation_mask = consumed_mesh_arrays["cell_data/ActivationMask"].astype(
        bool, copy=False
    )
    activation_control = consumed_mesh_arrays["cell_data/ActivationControlId"].astype(
        np.int64, copy=False
    )
    fibers_all = consumed_mesh_arrays["cell_data/ActivationFiber"]
    fraction_all = consumed_mesh_arrays["cell_data/MuscleFraction"]
    volume_all = consumed_mesh_arrays["cell_data/Volume"]
    region_names = consumed_mesh_arrays["field_data/ActivationRegionName"]
    region_muscle_ids = consumed_mesh_arrays["field_data/ActivationRegionMuscleId"]
    expression_names = {
        str(name)
        for name in consumed_mesh_arrays["field_data/ExpressionName"].reshape(-1)
    }

    if not np.array_equal(active_ids, np.flatnonzero(activation_mask)):
        raise ValueError("control basis active IDs differ from the fixture")
    if not np.array_equal(region_ids, activation_control[active_ids]):
        raise ValueError("control basis region IDs differ from the fixture")
    reference_mass = volume_all[active_ids] * fraction_all[active_ids]
    mass_absolute_error = np.abs(active_mass - reference_mass)
    mass_relative_error = mass_absolute_error / reference_mass
    if not np.allclose(active_mass, reference_mass, rtol=1e-12, atol=0.0):
        raise ValueError("control basis mass differs from Volume * MuscleFraction")
    active_mass = reference_mass
    fibers = fibers_all[active_ids]
    if not np.allclose(np.linalg.norm(fibers, axis=1), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("prepared active-cell fibers must be unit vectors")
    n_regions = int(region_ids.max() + 1)
    n_controls = len(control_ids)
    if (
        n_regions != len(region_names)
        or n_regions != len(region_muscle_ids)
        or n_controls != 4 * n_regions
        or not np.array_equal(control_ids, np.arange(n_controls))
        or not np.array_equal(control_region_ids, np.repeat(np.arange(n_regions), 4))
        or not np.array_equal(control_local_ids, np.tile(np.arange(4), n_regions))
    ):
        raise ValueError("fixture and four-control region ordering differ")

    consumed_point_data_arrays: tuple[str, ...] = ()
    if expression_names.intersection(consumed_point_data_arrays):
        raise AssertionError("an expression target array was consumed")

    scale = math.sqrt(cfg.initial_activation)
    reference_v = scale * fibers
    raw_q = np.empty(lfm.shape(n_controls), dtype=np.float64)
    condition_by_region: list[dict[str, Any]] = []
    for region_id in range(n_regions):
        rows = region_ids == region_id
        region_mass = active_mass[rows]
        local_control_ids = np.arange(4 * region_id, 4 * (region_id + 1))
        if not np.all(control_indices[rows] == local_control_ids[None, :]):
            raise ValueError(f"region {region_id} control ordering is not local")
        sqrt_mass_fraction = np.sqrt(region_mass / region_mass.sum())
        design = sqrt_mass_fraction[:, None] * weights[rows]
        response = sqrt_mass_fraction[:, None] * reference_v[rows]
        solution, _, rank, singular_values = np.linalg.lstsq(
            design, response, rcond=None
        )
        gram_eigenvalues = np.linalg.eigvalsh(design.T @ design)
        if rank != 4 or singular_values[-1] <= 0.0 or gram_eigenvalues[0] <= 0.0:
            raise ValueError(f"region {region_id} four-control fit is rank deficient")
        raw_q[local_control_ids] = solution
        raw_local = weights[rows] @ solution
        weighted_residual = float(
            np.sum(region_mass * np.sum((raw_local - reference_v[rows]) ** 2, axis=1))
            / region_mass.sum()
        )
        condition_by_region.append(
            {
                "region_id": region_id,
                "muscle_id": int(region_muscle_ids[region_id]),
                "name": str(region_names[region_id]),
                "ring": int(region_muscle_ids[region_id]) in RING_MUSCLE_IDS,
                "cells": int(rows.sum()),
                "least_squares": {
                    "rank": int(rank),
                    "singular_values": singular_values.tolist(),
                    "condition_number": float(singular_values[0] / singular_values[-1]),
                    "weighted_residual_mean_square_before_projection": weighted_residual,
                },
                "gram": {
                    "eigenvalues": gram_eigenvalues.tolist(),
                    "condition_number": float(
                        gram_eigenvalues[-1] / gram_eigenvalues[0]
                    ),
                },
            }
        )

    raw_norm = np.linalg.norm(raw_q, axis=1)
    radius = math.sqrt(cfg.amax)
    projected_mask = raw_norm > radius
    q = lfm.project(torch.from_numpy(raw_q), cfg.amax).numpy(force=True)
    local_v = lfm.interpolate(
        torch.from_numpy(q),
        torch.from_numpy(control_indices),
        torch.from_numpy(weights),
    ).numpy(force=True)
    if q.shape != (140, 3):
        raise ValueError(f"expected q shape (140, 3), got {q.shape}")
    if np.any(np.linalg.norm(local_v, axis=1) > radius + 1e-12):
        raise ValueError("convex interpolation exceeded the latent-vector bound")

    regions: list[dict[str, Any]] = []
    for condition in condition_by_region:
        region_id = condition["region_id"]
        rows = region_ids == region_id
        local_control_ids = np.arange(4 * region_id, 4 * (region_id + 1))
        regions.append(
            {
                **condition,
                "projected_controls": int(projected_mask[local_control_ids].sum()),
                "control_norm_before_projection": raw_norm[local_control_ids].tolist(),
                "control_norm_after_projection": np.linalg.norm(
                    q[local_control_ids], axis=1
                ).tolist(),
                "approximation_after_projection": approximation_metrics(
                    local_v[rows], reference_v[rows], fibers[rows], active_mass[rows]
                ),
            }
        )

    global_metrics = approximation_metrics(local_v, reference_v, fibers, active_mass)
    design_conditions = np.asarray(
        [region["least_squares"]["condition_number"] for region in regions]
    )
    gram_conditions = np.asarray(
        [region["gram"]["condition_number"] for region in regions]
    )
    ring_regions = [region for region in regions if region["ring"]]
    output_arrays = {
        "q": q,
        "q_unprojected": raw_q,
        "projected_control_mask": projected_mask,
        "local_v": local_v,
        "local_amplitude": np.sum(local_v * local_v, axis=1),
        "control_ids": control_ids,
        "control_region_ids": control_region_ids,
        "control_local_ids": control_local_ids,
        "active_ids": active_ids,
        "active_region_ids": region_ids,
        "initial_activation": np.asarray(cfg.initial_activation),
        "amax": np.asarray(cfg.amax),
    }
    np.savez_compressed(cfg.output_npz, **output_arrays)

    source_paths = (
        Path(__file__),
        HERE / "learned_fiber_models.py",
        HERE / "experiment_profile.py",
    )
    metadata = {
        "schema_version": 1,
        "design": (
            "four signed latent vectors per named region fitted by rest "
            "muscle-fraction-volume-weighted least squares"
        ),
        "claim": "latent kinematic directions; not identified anatomical fibers",
        "config": {
            "initial_activation": cfg.initial_activation,
            "amax": cfg.amax,
            "control_radius": radius,
        },
        "counts": {
            "active_cells": len(active_ids),
            "regions": n_regions,
            "controls": n_controls,
            "parameters": int(q.size),
            "projected_controls": int(projected_mask.sum()),
            "zero_local_vectors": global_metrics["zero_local_vectors"],
        },
        "target_independence": {
            "target_used": False,
            "point_data_arrays_consumed": list(consumed_point_data_arrays),
            "expression_target_arrays_present": sorted(expression_names),
            "assertion": "no expression target array is in the consumed array set",
            "fit_source": "prepared signed ActivationFiber from rest geometry",
        },
        "mass_validation": {
            "fit_weights": "fixture cell_data/Volume * cell_data/MuscleFraction",
            "control_basis_active_mass_max_absolute_error": float(
                mass_absolute_error.max()
            ),
            "control_basis_active_mass_max_relative_error": float(
                mass_relative_error.max()
            ),
            "accepted_relative_tolerance": 1e-12,
        },
        "global_approximation_after_projection": global_metrics,
        "conditioning": {
            "least_squares_condition_number": {
                "min": float(design_conditions.min()),
                "max": float(design_conditions.max()),
                "weighted_quantiles_by_region_mass": weighted_quantiles(
                    design_conditions,
                    np.asarray(
                        [
                            region["approximation_after_projection"]["mass_m3"]
                            for region in regions
                        ]
                    ),
                ),
            },
            "gram_condition_number": {
                "min": float(gram_conditions.min()),
                "max": float(gram_conditions.max()),
                "weighted_quantiles_by_region_mass": weighted_quantiles(
                    gram_conditions,
                    np.asarray(
                        [
                            region["approximation_after_projection"]["mass_m3"]
                            for region in regions
                        ]
                    ),
                ),
            },
        },
        "ring_summary": [
            {
                "region_id": region["region_id"],
                "muscle_id": region["muscle_id"],
                "name": region["name"],
                "approximation_after_projection": region[
                    "approximation_after_projection"
                ],
            }
            for region in ring_regions
        ],
        "regions": regions,
        "hashes": {
            "inputs": {
                "volume.vtu": sha256(volume_path),
                "12-controls.npz": sha256(cfg.controls),
            },
            "sources": {path.name: sha256(path) for path in source_paths},
            "consumed_arrays": {
                **{
                    name: array_sha256(array)
                    for name, array in consumed_mesh_arrays.items()
                },
                **{
                    f"12-controls.npz/{name}": array_sha256(array)
                    for name, array in consumed_basis_arrays.items()
                },
            },
            "output_npz": sha256(cfg.output_npz),
        },
    }
    write_json(cfg.output_json, metadata)
    cherries.log_metrics(
        {
            "latent/vector_relative_rms": global_metrics["vector"][
                "weighted_relative_rms"
            ],
            "latent/H_relative_rms": global_metrics["H_tensor"][
                "weighted_relative_frobenius_rms"
            ],
            "latent/projected_controls": int(projected_mask.sum()),
            "latent/zero_local_vectors": global_metrics["zero_local_vectors"],
        }
    )
    LOG.info("Wrote %s and %s", cfg.output_npz, cfg.output_json)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
