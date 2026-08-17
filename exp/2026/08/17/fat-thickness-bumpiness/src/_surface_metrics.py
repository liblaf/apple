from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv
from _common import toy
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class ResampledSurface:
    x: np.ndarray
    z: np.ndarray
    rest_y: np.ndarray
    displacement: np.ndarray
    target: np.ndarray
    valid: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        return self.displacement - self.target


def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(array))))


def vector_rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.sum(np.square(array), axis=-1))))


def top_surface_ids(mesh: pv.UnstructuredGrid) -> np.ndarray:
    triangles = toy.top_surface_triangles(mesh)
    if triangles.size == 0:
        msg = "mesh has no top-surface triangles"
        raise ValueError(msg)
    return np.unique(triangles.ravel())


def top_xz_bounds(mesh: pv.UnstructuredGrid) -> tuple[float, float, float, float]:
    ids = top_surface_ids(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)[ids]
    return (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 2].min()),
        float(points[:, 2].max()),
    )


def common_xz_bounds(
    meshes: list[pv.UnstructuredGrid],
) -> tuple[float, float, float, float]:
    bounds = np.asarray([top_xz_bounds(mesh) for mesh in meshes], dtype=np.float64)
    domain = toy.ALL_BOUNDS
    common = (
        max(float(bounds[:, 0].max()), float(domain[0])),
        min(float(bounds[:, 1].min()), float(domain[1])),
        max(float(bounds[:, 2].max()), float(domain[4])),
        min(float(bounds[:, 3].min()), float(domain[5])),
    )
    if common[0] >= common[1] or common[2] >= common[3]:
        msg = f"top surfaces have no common x-z domain: {common}"
        raise ValueError(msg)
    return common


def interpolation_values(
    points_xz: np.ndarray,
    values: np.ndarray,
    query_xz: np.ndarray,
) -> np.ndarray:
    linear = LinearNDInterpolator(points_xz, values, fill_value=np.nan)
    result = np.asarray(linear(query_xz), dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    missing = ~np.isfinite(result).all(axis=1)
    if np.any(missing):
        msg = (
            f"{int(missing.sum())} common-grid points lie outside the top-surface "
            "linear interpolation domain"
        )
        raise ValueError(msg)
    return result


def resample_top_surface(
    mesh: pv.UnstructuredGrid,
    *,
    bounds: tuple[float, float, float, float],
    grid_size: int,
) -> ResampledSurface:
    if grid_size < 5:
        msg = f"grid_size must be at least 5, got {grid_size}"
        raise ValueError(msg)
    ids = top_surface_ids(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)[ids]
    points_xz = points[:, (0, 2)]
    if np.unique(points_xz, axis=0).shape[0] != points_xz.shape[0]:
        msg = "top surface is not a single-valued graph over x-z"
        raise ValueError(msg)

    displacement_all = np.asarray(mesh.point_data.get("Displacement"), dtype=np.float64)
    if displacement_all.shape != (mesh.n_points, 3):
        msg = (
            "result mesh must have point_data['Displacement'] with shape "
            f"({mesh.n_points}, 3), got {displacement_all.shape}"
        )
        raise ValueError(msg)
    target_raw = mesh.point_data.get("TargetDisplacement")
    target_all = (
        np.zeros_like(displacement_all)
        if target_raw is None
        else np.asarray(target_raw, dtype=np.float64)
    )
    if target_all.shape != displacement_all.shape:
        msg = (
            "TargetDisplacement must match Displacement, got "
            f"{target_all.shape} and {displacement_all.shape}"
        )
        raise ValueError(msg)

    xmin, xmax, zmin, zmax = bounds
    x_axis = np.linspace(xmin, xmax, grid_size)
    z_axis = np.linspace(zmin, zmax, grid_size)
    x, z = np.meshgrid(x_axis, z_axis, indexing="ij")
    query_xz = np.column_stack((x.ravel(), z.ravel()))
    packed = np.column_stack(
        (
            points[:, 1],
            displacement_all[ids],
            target_all[ids],
        )
    )
    sampled = interpolation_values(points_xz, packed, query_xz).reshape(
        grid_size, grid_size, -1
    )
    rest_y = sampled[..., 0]
    displacement = sampled[..., 1:4]
    target = sampled[..., 4:7]
    valid = np.isfinite(sampled).all(axis=-1)
    return ResampledSurface(
        x=x,
        z=z,
        rest_y=rest_y,
        displacement=displacement,
        target=target,
        valid=valid,
    )


def finite_difference_laplacian(
    field: np.ndarray,
    valid: np.ndarray,
    *,
    dx: float,
    dz: float,
) -> np.ndarray:
    center = field[1:-1, 1:-1]
    laplacian = (field[2:, 1:-1] - 2.0 * center + field[:-2, 1:-1]) / dx**2 + (
        field[1:-1, 2:] - 2.0 * center + field[1:-1, :-2]
    ) / dz**2
    stencil_valid = (
        valid[1:-1, 1:-1]
        & valid[2:, 1:-1]
        & valid[:-2, 1:-1]
        & valid[1:-1, 2:]
        & valid[1:-1, :-2]
        & np.isfinite(laplacian)
    )
    return laplacian[stencil_valid]


def high_frequency_power_fraction(
    field: np.ndarray,
    *,
    dx: float,
    dz: float,
    cutoff_cycles_per_unit: float,
) -> float:
    if cutoff_cycles_per_unit <= 0.0:
        msg = f"cutoff_cycles_per_unit must be positive, got {cutoff_cycles_per_unit}"
        raise ValueError(msg)
    values = np.asarray(field, dtype=np.float64)
    values = values - values.mean()
    if np.allclose(values, 0.0):
        return 0.0
    window = np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    spectrum = np.fft.rfft2(values * window)
    power = np.square(np.abs(spectrum))
    freq_x = np.fft.fftfreq(values.shape[0], d=dx)
    freq_z = np.fft.rfftfreq(values.shape[1], d=dz)
    radius = np.sqrt(freq_x[:, None] ** 2 + freq_z[None, :] ** 2)
    nonzero = radius > 0.0
    denominator = power[nonzero].sum()
    if denominator <= 0.0:
        return 0.0
    return float(power[radius >= cutoff_cycles_per_unit].sum() / denominator)


def gaussian_smooth(
    field: np.ndarray,
    *,
    dx: float,
    dz: float,
    smoothing_length: float,
) -> np.ndarray:
    if smoothing_length <= 0.0:
        msg = f"smoothing_length must be positive, got {smoothing_length}"
        raise ValueError(msg)
    sigma = (smoothing_length / dx, smoothing_length / dz)
    return np.asarray(gaussian_filter(field, sigma=sigma, mode="nearest"))


def graph_surface_area(
    x: np.ndarray, z: np.ndarray, y: np.ndarray, valid: np.ndarray
) -> float:
    points = np.stack((x, y, z), axis=-1)
    p00 = points[:-1, :-1]
    p10 = points[1:, :-1]
    p01 = points[:-1, 1:]
    p11 = points[1:, 1:]
    cell_valid = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    area0 = 0.5 * np.linalg.norm(np.cross(p10 - p00, p11 - p00), axis=-1)
    area1 = 0.5 * np.linalg.norm(np.cross(p11 - p00, p01 - p00), axis=-1)
    return float((area0[cell_valid] + area1[cell_valid]).sum())


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def surface_metrics(
    sample: ResampledSurface,
    *,
    high_frequency_cutoff_cycles: float,
    laplacian_smoothing_length: float,
    muscle_bounds: tuple[float, float, float, float, float, float],
) -> dict[str, float | int]:
    if laplacian_smoothing_length <= 0.0:
        msg = (
            "laplacian_smoothing_length must be positive, got "
            f"{laplacian_smoothing_length}"
        )
        raise ValueError(msg)
    valid = sample.valid
    displacement = sample.displacement
    target = sample.target
    residual = sample.residual
    dx = float(sample.x[1, 0] - sample.x[0, 0])
    dz = float(sample.z[0, 1] - sample.z[0, 0])
    domain_length = math.sqrt(
        float((sample.x.max() - sample.x.min()) * (sample.z.max() - sample.z.min()))
    )

    displacement_rms = vector_rms(displacement[valid])
    target_rms = vector_rms(target[valid])
    residual_rms = vector_rms(residual[valid])
    displacement_y_rms = rms(displacement[..., 1][valid])
    target_y_rms = rms(target[..., 1][valid])
    residual_y_rms = rms(residual[..., 1][valid])

    displacement_lap_raw = finite_difference_laplacian(
        displacement[..., 1], valid, dx=dx, dz=dz
    )
    residual_lap_raw = finite_difference_laplacian(
        residual[..., 1], valid, dx=dx, dz=dz
    )
    displacement_y_smooth = gaussian_smooth(
        displacement[..., 1],
        dx=dx,
        dz=dz,
        smoothing_length=laplacian_smoothing_length,
    )
    residual_y_smooth = gaussian_smooth(
        residual[..., 1],
        dx=dx,
        dz=dz,
        smoothing_length=laplacian_smoothing_length,
    )
    displacement_y_highpass = displacement[..., 1] - displacement_y_smooth
    residual_y_highpass = residual[..., 1] - residual_y_smooth
    displacement_lap = finite_difference_laplacian(
        displacement_y_smooth, valid, dx=dx, dz=dz
    )
    residual_lap = finite_difference_laplacian(residual_y_smooth, valid, dx=dx, dz=dz)
    displacement_lap_rms = rms(displacement_lap)
    residual_lap_rms = rms(residual_lap)

    rest_area = graph_surface_area(sample.x, sample.z, sample.rest_y, valid)
    deformed_area = graph_surface_area(
        sample.x + displacement[..., 0],
        sample.z + displacement[..., 2],
        sample.rest_y + displacement[..., 1],
        valid,
    )
    target_area = graph_surface_area(
        sample.x + target[..., 0],
        sample.z + target[..., 2],
        sample.rest_y + target[..., 1],
        valid,
    )
    muscle_mask = (
        valid
        & (sample.x >= muscle_bounds[0])
        & (sample.x <= muscle_bounds[1])
        & (sample.z >= muscle_bounds[4])
        & (sample.z <= muscle_bounds[5])
    )
    near_target_rms = vector_rms(target[muscle_mask])
    near_residual_rms = vector_rms(residual[muscle_mask])
    result: dict[str, float | int] = {
        "grid/n": int(sample.x.shape[0]),
        "grid/dx": dx,
        "grid/dz": dz,
        "grid/laplacian_smoothing_length": laplacian_smoothing_length,
        "grid/n_valid": int(valid.sum()),
        "grid/n_laplacian": int(displacement_lap.size),
        "grid/displacement_rms": displacement_rms,
        "grid/target_rms": target_rms,
        "grid/residual_rms": residual_rms,
        "grid/error_rms_fraction_of_target": safe_ratio(residual_rms, target_rms),
        "grid/displacement_y_min": float(displacement[..., 1][valid].min()),
        "grid/displacement_y_mean": float(displacement[..., 1][valid].mean()),
        "grid/displacement_y_max": float(displacement[..., 1][valid].max()),
        "grid/displacement_y_std": float(displacement[..., 1][valid].std()),
        "grid/displacement_y_rms": displacement_y_rms,
        "grid/target_y_rms": target_y_rms,
        "grid/residual_y_rms": residual_y_rms,
        "grid/displacement_y_laplacian_rms": displacement_lap_rms,
        "grid/residual_y_laplacian_rms": residual_lap_rms,
        "grid/displacement_y_laplacian_raw_rms": rms(displacement_lap_raw),
        "grid/residual_y_laplacian_raw_rms": rms(residual_lap_raw),
        "grid/displacement_y_highpass_rms": rms(displacement_y_highpass[valid]),
        "grid/residual_y_highpass_rms": rms(residual_y_highpass[valid]),
        "grid/displacement_y_highpass_over_rms": safe_ratio(
            rms(displacement_y_highpass[valid]), displacement_y_rms
        ),
        "grid/residual_y_highpass_over_target_rms": safe_ratio(
            rms(residual_y_highpass[valid]), target_y_rms
        ),
        "grid/displacement_y_laplacian_over_rms": safe_ratio(
            displacement_lap_rms * domain_length**2,
            displacement_y_rms,
        ),
        "grid/residual_y_laplacian_over_target_rms": safe_ratio(
            residual_lap_rms * domain_length**2,
            target_y_rms,
        ),
        "grid/displacement_y_high_frequency_power_fraction": high_frequency_power_fraction(
            displacement[..., 1],
            dx=dx,
            dz=dz,
            cutoff_cycles_per_unit=high_frequency_cutoff_cycles,
        ),
        "grid/residual_y_high_frequency_power_fraction": high_frequency_power_fraction(
            residual[..., 1],
            dx=dx,
            dz=dz,
            cutoff_cycles_per_unit=high_frequency_cutoff_cycles,
        ),
        "grid/surface_area_rest": rest_area,
        "grid/surface_area_deformed": deformed_area,
        "grid/surface_area_target": target_area,
        "grid/surface_area_deformed_over_rest": safe_ratio(deformed_area, rest_area),
        "grid/surface_area_target_over_rest": safe_ratio(target_area, rest_area),
        "grid/surface_area_deformed_over_target": safe_ratio(
            deformed_area, target_area
        ),
        "grid/near_muscle_n": int(muscle_mask.sum()),
        "grid/near_muscle_residual_rms": near_residual_rms,
        "grid/near_muscle_error_rms_fraction_of_target": safe_ratio(
            near_residual_rms, near_target_rms
        ),
    }
    return result


def finite_scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float) and math.isfinite(float(value))
    }
