# ruff: noqa: C901, EM101, EM102, PLR0915, TRY003
"""Target-independent smooth control bases for named facial muscle regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

CONTROLS_PER_REGION = 4
LOCAL_CONTROL_LABELS = ("+major", "-major", "+minor", "-minor")


@dataclass(frozen=True)
class ControlBasis:
    """Four convex spatial controls per named muscle region."""

    active_ids: NDArray[np.int64]
    active_region_ids: NDArray[np.int64]
    active_muscle_ids: NDArray[np.int64]
    active_mass: NDArray[np.float64]
    control_indices: NDArray[np.int64]
    weights: NDArray[np.float64]
    control_ids: NDArray[np.int64]
    control_region_ids: NDArray[np.int64]
    control_local_ids: NDArray[np.int64]
    control_centers: NDArray[np.float64]
    parameter_mass: NDArray[np.float64]
    region_centroids: NDArray[np.float64]
    region_axes: NDArray[np.float64]
    region_eigenvalues: NDArray[np.float64]
    region_sigma: NDArray[np.float64]

    @property
    def n_regions(self) -> int:
        return self.region_centroids.shape[0]

    @property
    def n_controls(self) -> int:
        return self.control_ids.size

    def expand(self, controls: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate global control rows to the active cells."""
        controls = np.asarray(controls, dtype=np.float64)
        if controls.shape[0] != self.n_controls:
            raise ValueError(
                f"controls need {self.n_controls} rows, got {controls.shape[0]}"
            )
        return np.sum(
            self.weights[(...,) + (None,) * (controls.ndim - 1)]
            * controls[self.control_indices],
            axis=1,
        )

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return the stable, pickle-free NPZ schema."""
        return {
            "active_ids": self.active_ids,
            "active_region_ids": self.active_region_ids,
            "active_muscle_ids": self.active_muscle_ids,
            "active_mass": self.active_mass,
            "control_indices": self.control_indices,
            "weights": self.weights,
            "control_ids": self.control_ids,
            "control_region_ids": self.control_region_ids,
            "control_local_ids": self.control_local_ids,
            "control_centers": self.control_centers,
            "parameter_mass": self.parameter_mass,
            "parameter_mass_fraction": self.parameter_mass / self.parameter_mass.sum(),
            "region_centroids": self.region_centroids,
            "region_axes": self.region_axes,
            "region_eigenvalues": self.region_eigenvalues,
            "region_sigma": self.region_sigma,
        }


def build_control_basis(
    points: NDArray[np.float64],
    tets: NDArray[np.int64],
    active_ids: NDArray[np.int64],
    active_region_ids: NDArray[np.int64],
    active_muscle_ids: NDArray[np.int64],
    active_fraction: NDArray[np.float64],
) -> ControlBasis:
    """Build four smooth convex controls per region from rest geometry only.

    Cell centers and ``tet_volume * muscle_fraction`` define a weighted PCA in
    each region. Control centers are one standard deviation along the positive
    and negative major and minor axes. The isotropic Gaussian width is the RMS
    of those two standard deviations. Normalized Gaussian values form a
    nonnegative partition of unity.
    """
    points = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    active_ids = np.asarray(active_ids, dtype=np.int64)
    active_region_ids = np.asarray(active_region_ids, dtype=np.int64)
    active_muscle_ids = np.asarray(active_muscle_ids, dtype=np.int64)
    active_fraction = np.asarray(active_fraction, dtype=np.float64)
    _validate_inputs(
        points,
        tets,
        active_ids,
        active_region_ids,
        active_muscle_ids,
        active_fraction,
    )

    active_tets = tets[active_ids]
    vertices = points[active_tets]
    cell_centers = vertices.mean(axis=1)
    dm = np.transpose(vertices[:, 1:] - vertices[:, :1], (0, 2, 1))
    volumes = np.linalg.det(dm) / 6.0
    if np.any(volumes <= 0.0):
        raise ValueError("active tetrahedra must have positive orientation and volume")
    active_mass = volumes * active_fraction
    if np.any(active_mass <= 0.0):
        raise ValueError("every active cell must have positive muscle-volume mass")

    region_values = np.unique(active_region_ids)
    if not np.array_equal(region_values, np.arange(region_values.size)):
        raise ValueError("active region IDs must be contiguous from zero")
    n_regions = region_values.size
    n_controls = CONTROLS_PER_REGION * n_regions
    region_centroids = np.empty((n_regions, 3), dtype=np.float64)
    region_axes = np.empty((n_regions, 2, 3), dtype=np.float64)
    region_eigenvalues = np.empty((n_regions, 3), dtype=np.float64)
    region_sigma = np.empty(n_regions, dtype=np.float64)
    control_centers = np.empty((n_controls, 3), dtype=np.float64)
    weights = np.empty((active_ids.size, CONTROLS_PER_REGION), dtype=np.float64)

    for region_id in range(n_regions):
        rows = np.flatnonzero(active_region_ids == region_id)
        xyz = cell_centers[rows]
        mass = active_mass[rows]
        centroid = np.average(xyz, axis=0, weights=mass)
        centered = xyz - centroid
        covariance = np.einsum("n,ni,nj->ij", mass, centered, centered) / mass.sum()
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]
        if values[1] <= np.finfo(np.float64).eps * values[0]:
            raise ValueError(f"region {region_id} lacks two resolved PCA directions")
        major = _canonical_axis(vectors[:, 0])
        minor = _canonical_axis(vectors[:, 1])
        sigma = np.sqrt((values[0] + values[1]) / 2.0)
        centers = np.stack(
            (
                centroid + np.sqrt(values[0]) * major,
                centroid - np.sqrt(values[0]) * major,
                centroid + np.sqrt(values[1]) * minor,
                centroid - np.sqrt(values[1]) * minor,
            )
        )
        distance_squared = ((xyz[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        logits = -distance_squared / (2.0 * sigma**2)
        logits -= logits.max(axis=1, keepdims=True)
        phi = np.exp(logits)
        phi /= phi.sum(axis=1, keepdims=True)

        control_slice = slice(
            CONTROLS_PER_REGION * region_id,
            CONTROLS_PER_REGION * (region_id + 1),
        )
        region_centroids[region_id] = centroid
        region_axes[region_id] = (major, minor)
        region_eigenvalues[region_id] = values
        region_sigma[region_id] = sigma
        control_centers[control_slice] = centers
        weights[rows] = phi

    control_ids = np.arange(n_controls, dtype=np.int64)
    control_region_ids = np.repeat(
        np.arange(n_regions, dtype=np.int64), CONTROLS_PER_REGION
    )
    control_local_ids = np.tile(
        np.arange(CONTROLS_PER_REGION, dtype=np.int64), n_regions
    )
    control_indices = (
        CONTROLS_PER_REGION * active_region_ids[:, None]
        + np.arange(CONTROLS_PER_REGION, dtype=np.int64)[None, :]
    )
    parameter_mass = np.bincount(
        control_indices.ravel(),
        weights=(active_mass[:, None] * weights).ravel(),
        minlength=n_controls,
    )
    basis = ControlBasis(
        active_ids=active_ids,
        active_region_ids=active_region_ids,
        active_muscle_ids=active_muscle_ids,
        active_mass=active_mass,
        control_indices=control_indices,
        weights=weights,
        control_ids=control_ids,
        control_region_ids=control_region_ids,
        control_local_ids=control_local_ids,
        control_centers=control_centers,
        parameter_mass=parameter_mass,
        region_centroids=region_centroids,
        region_axes=region_axes,
        region_eigenvalues=region_eigenvalues,
        region_sigma=region_sigma,
    )
    validate_control_basis(basis)
    return basis


def validate_control_basis(basis: ControlBasis) -> None:
    """Check ordering, convexity, regional recovery, and lumped masses."""
    n_active = basis.active_ids.size
    n_regions = basis.n_regions
    n_controls = CONTROLS_PER_REGION * n_regions
    if basis.control_indices.shape != (n_active, CONTROLS_PER_REGION):
        raise ValueError("control_indices has the wrong shape")
    if basis.weights.shape != (n_active, CONTROLS_PER_REGION):
        raise ValueError("weights has the wrong shape")
    if not np.array_equal(basis.control_ids, np.arange(n_controls)):
        raise ValueError("global control IDs must be contiguous")
    expected_indices = (
        CONTROLS_PER_REGION * basis.active_region_ids[:, None]
        + np.arange(CONTROLS_PER_REGION)[None, :]
    )
    if not np.array_equal(basis.control_indices, expected_indices):
        raise ValueError("each active cell must reference its region's four controls")
    if not np.isfinite(basis.weights).all() or np.any(basis.weights < 0.0):
        raise ValueError("partition weights must be finite and nonnegative")
    if not np.allclose(basis.weights.sum(axis=1), 1.0, rtol=0.0, atol=2e-15):
        raise ValueError("partition weights must sum to one per active cell")
    if not np.isfinite(basis.parameter_mass).all() or np.any(
        basis.parameter_mass <= 0.0
    ):
        raise ValueError("every global control must have positive lumped mass")
    if not np.isclose(
        basis.parameter_mass.sum(), basis.active_mass.sum(), rtol=2e-15, atol=0.0
    ):
        raise ValueError("parameter masses must conserve active muscle-volume mass")

    regional = np.linspace(0.1, 0.9, n_regions, dtype=np.float64)
    uniform_controls = np.repeat(regional, CONTROLS_PER_REGION)[:, None]
    recovered = basis.expand(uniform_controls)[:, 0]
    if not np.allclose(
        recovered, regional[basis.active_region_ids], rtol=0.0, atol=2e-15
    ):
        raise ValueError("uniform local controls must recover the regional field")
    bounded_controls = (basis.control_ids % 2).astype(np.float64)[:, None]
    bounded_field = basis.expand(bounded_controls)[:, 0]
    if np.any(bounded_field < 0.0) or np.any(bounded_field > 1.0):
        raise ValueError("convex interpolation must preserve control bounds")


def _canonical_axis(axis: NDArray[np.float64]) -> NDArray[np.float64]:
    axis = np.asarray(axis, dtype=np.float64).copy()
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis *= -1.0
    return axis


def _validate_inputs(
    points: NDArray[np.float64],
    tets: NDArray[np.int64],
    active_ids: NDArray[np.int64],
    active_region_ids: NDArray[np.int64],
    active_muscle_ids: NDArray[np.int64],
    active_fraction: NDArray[np.float64],
) -> None:
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite with shape (n_points, 3)")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tets must have shape (n_cells, 4)")
    if np.any(tets < 0) or np.any(tets >= points.shape[0]):
        raise ValueError("tetrahedron point indices are out of bounds")
    n_active = active_ids.size
    if any(
        array.shape != (n_active,)
        for array in (active_region_ids, active_muscle_ids, active_fraction)
    ):
        raise ValueError("active metadata arrays must have one row per active cell")
    if not np.array_equal(active_ids, np.unique(active_ids)):
        raise ValueError("active IDs must be unique and strictly increasing")
    if np.any(active_ids < 0) or np.any(active_ids >= tets.shape[0]):
        raise ValueError("active IDs are out of bounds")
    if np.any(active_fraction <= 0.0) or np.any(active_fraction > 1.0):
        raise ValueError("active muscle fractions must be in (0, 1]")
    for region_id in np.unique(active_region_ids):
        muscle_ids = np.unique(active_muscle_ids[active_region_ids == region_id])
        if muscle_ids.size != 1:
            raise ValueError(f"region {region_id} must contain exactly one MuscleId")
