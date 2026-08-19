from __future__ import annotations

# ruff: noqa: C901, EM101, EM102, FBT003, PLR0912, PLR0915, RUF046, TRY003
import contextlib
import csv
import hashlib
import io
import itertools
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from scipy.ndimage import gaussian_filter

from liblaf import cherries, melon

REPO_ROOT = Path(__file__).resolve().parents[6]
TOY_HELPER_DIR = REPO_ROOT / "exp/2026/06/10/unreachable-toy-skin-tetwild/src"
if str(TOY_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(TOY_HELPER_DIR))

import _toy_skin_tetwild as toy  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
DESIGN = "nested-mesh-isochoric-bumpy-activation-transfer-v2"
AXIS_NAMES = ("x", "y", "z")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_summary: Path = cherries.output(
        "10-bumpy-activation-transfer-summary.json", mkdir=True
    )

    labels: tuple[str, ...] = ("thin", "medium", "thick")
    top_fat_thicknesses: tuple[float, ...] = (0.04, 0.08, 0.12)
    bottom_fat_thickness: float = 0.04
    muscle_thickness: float = 0.02

    nx: int = 48
    nz: int = 48
    vertical_spacing: float = 0.01

    mean_activation_inv_x: float = 0.25
    modulation_rms: float = 0.10
    wave_number: int = 4
    continuation_alphas: tuple[float, ...] = (0.5, 1.0)

    crop_fraction: float = 0.0
    highpass_smoothing_length: float = 0.06
    forward_max_steps: int = 8000
    forward_atol: float = 1.0e-10
    forward_rtol: float = 1.0e-6
    require_convergence: bool = True
    require_branch_agreement: bool = True
    max_branch_difference_over_signal: float = 0.10
    minimum_det_f: float = 0.20
    minimum_det_f_q001: float = 0.40


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def slugify(value: str) -> str:
    slug = "".join(character if character.isalnum() else "-" for character in value)
    slug = "-".join(part for part in slug.lower().split("-") if part)
    if not slug:
        raise ValueError(f"invalid empty label from {value!r}")
    return slug


def layer_count(thickness: float, spacing: float, *, name: str) -> int:
    count = int(round(thickness / spacing))
    if count < 1 or not math.isclose(
        count * spacing, thickness, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError(f"{name}={thickness} must be a positive multiple of {spacing}")
    return count


def validate_config(cfg: Config) -> list[tuple[str, float, int]]:
    labels = tuple(slugify(label) for label in cfg.labels)
    thicknesses = tuple(float(value) for value in cfg.top_fat_thicknesses)
    if len(labels) != len(thicknesses) or not labels:
        raise ValueError(
            "labels and top-fat thicknesses must have equal nonzero length"
        )
    if len(set(labels)) != len(labels) or len(set(thicknesses)) != len(thicknesses):
        raise ValueError("labels and thicknesses must be unique")
    if any(not math.isfinite(value) or value <= 0.0 for value in thicknesses):
        raise ValueError("top-fat thicknesses must be finite and positive")
    if cfg.bottom_fat_thickness <= 0.0 or cfg.muscle_thickness <= 0.0:
        raise ValueError("bottom-fat and muscle thickness must be positive")
    if cfg.nx < 8 or cfg.nz < 8:
        raise ValueError("nx and nz must each be at least 8")
    if not math.isfinite(cfg.vertical_spacing) or cfg.vertical_spacing <= 0.0:
        raise ValueError("vertical spacing must be finite and positive")
    layer_count(
        cfg.bottom_fat_thickness,
        cfg.vertical_spacing,
        name="bottom_fat_thickness",
    )
    layer_count(cfg.muscle_thickness, cfg.vertical_spacing, name="muscle_thickness")
    top_layer_counts = [
        layer_count(value, cfg.vertical_spacing, name=f"top_fat_thickness[{index}]")
        for index, value in enumerate(thicknesses)
    ]
    if cfg.wave_number < 1 or 2 * cfg.wave_number >= min(cfg.nx, cfg.nz):
        raise ValueError(
            "wave number must be positive and below the grid Nyquist limit"
        )
    minimum_samples_per_wave = min(cfg.nx, cfg.nz) / cfg.wave_number
    if minimum_samples_per_wave < 8.0:
        raise ValueError("activation wave needs at least eight cells per wavelength")
    if not 0.0 <= cfg.crop_fraction < 0.4:
        raise ValueError("crop fraction must be in [0, 0.4)")
    if cfg.highpass_smoothing_length <= 0.0:
        raise ValueError("high-pass smoothing length must be positive")
    if not cfg.continuation_alphas or cfg.continuation_alphas[-1] != 1.0:
        raise ValueError("continuation alphas must end at 1.0")
    if any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in cfg.continuation_alphas
    ):
        raise ValueError("continuation alphas must be finite and in (0, 1]")
    if any(b <= a for a, b in itertools.pairwise(cfg.continuation_alphas)):
        raise ValueError("continuation alphas must be strictly increasing")
    if cfg.modulation_rms <= 0.0:
        raise ValueError("activation modulation RMS must be positive")
    if 1.0 + cfg.mean_activation_inv_x <= 0.0:
        raise ValueError("mean I + ActivationInv_x must stay positive definite")
    if cfg.forward_max_steps < 1:
        raise ValueError("forward max steps must be positive")
    if cfg.forward_atol < 0.0 or cfg.forward_rtol < 0.0:
        raise ValueError("forward tolerances must be nonnegative")
    if cfg.max_branch_difference_over_signal <= 0.0:
        raise ValueError("branch agreement threshold must be positive")
    if not 0.0 < cfg.minimum_det_f <= cfg.minimum_det_f_q001:
        raise ValueError("detF gates must be positive and ordered")
    return list(zip(labels, thicknesses, top_layer_counts, strict=True))


def y_coordinates(cfg: Config, top_fat_thickness: float) -> np.ndarray:
    bottom_layers = layer_count(
        cfg.bottom_fat_thickness,
        cfg.vertical_spacing,
        name="bottom_fat_thickness",
    )
    muscle_layers = layer_count(
        cfg.muscle_thickness,
        cfg.vertical_spacing,
        name="muscle_thickness",
    )
    top_layers = layer_count(
        top_fat_thickness,
        cfg.vertical_spacing,
        name="top_fat_thickness",
    )
    bottom = np.linspace(
        0.0,
        cfg.bottom_fat_thickness,
        bottom_layers + 1,
    )
    muscle_top = cfg.bottom_fat_thickness + cfg.muscle_thickness
    muscle = np.linspace(
        cfg.bottom_fat_thickness,
        muscle_top,
        muscle_layers + 1,
    )[1:]
    top = np.linspace(
        muscle_top,
        muscle_top + top_fat_thickness,
        top_layers + 1,
    )[1:]
    return np.concatenate((bottom, muscle, top))


def structured_connectivity(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    point_ids = np.arange((ny + 1) * (nx + 1) * (nz + 1), dtype=np.int64).reshape(
        ny + 1, nx + 1, nz + 1
    )
    permutations = tuple(itertools.permutations(range(3)))
    tets: list[tuple[int, int, int, int]] = []
    for iy in range(ny):
        for ix in range(nx):
            for iz in range(nz):
                local = {
                    (dy, dx, dz): int(point_ids[iy + dy, ix + dx, iz + dz])
                    for dy in (0, 1)
                    for dx in (0, 1)
                    for dz in (0, 1)
                }
                origin = (0, 0, 0)
                destination = (1, 1, 1)
                for permutation in permutations:
                    cursor = [0, 0, 0]
                    first = cursor.copy()
                    first[permutation[0]] = 1
                    second = first.copy()
                    second[permutation[1]] = 1
                    tets.append(
                        (
                            local[origin],
                            local[tuple(first)],
                            local[tuple(second)],
                            local[destination],
                        )
                    )
    return np.asarray(tets, dtype=np.int64), point_ids


def make_mesh(
    cfg: Config,
    *,
    top_fat_thickness: float,
    tets: np.ndarray,
    point_ids: np.ndarray,
) -> pv.UnstructuredGrid:
    xs = np.linspace(0.0, 1.0, cfg.nx + 1)
    zs = np.linspace(0.0, 1.0, cfg.nz + 1)
    ys = y_coordinates(cfg, top_fat_thickness)
    yy, xx, zz = np.meshgrid(ys, xs, zs, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    cells = np.empty((tets.shape[0], 5), dtype=np.int64)
    cells[:, 0] = 4
    cells[:, 1:] = tets
    cell_types = np.full(tets.shape[0], int(pv.CellType.TETRA), dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells.ravel(), cell_types, points)
    mesh = toy.orient_tetra_mesh(mesh)

    muscle_bottom = cfg.bottom_fat_thickness
    muscle_top = muscle_bottom + cfg.muscle_thickness
    mesh.field_data[toy.GEOMETRY_KIND] = np.asarray(["box"])
    mesh.field_data[toy.MUSCLE_BOUNDS_FIELD] = np.asarray(
        [0.0, 1.0, muscle_bottom, muscle_top, 0.0, 1.0], dtype=np.float64
    )
    mesh.field_data["StructuredGridShape"] = np.asarray(
        [cfg.nx, len(ys) - 1, cfg.nz], dtype=np.int64
    )
    mesh.field_data["TopFatThickness"] = np.asarray([top_fat_thickness])
    mesh.field_data["ActivationWaveNumber"] = np.asarray([cfg.wave_number])
    mesh.field_data["VerticalSpacing"] = np.asarray([cfg.vertical_spacing])
    spec = toy.ResolutionSpec(
        name=f"structured-{cfg.nx}x{len(ys) - 1}x{cfg.nz}",
        lr=min(1.0 / cfg.nx, 1.0 / cfg.nz),
    )
    toy.add_material_and_boundary_fields(mesh, spec)

    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    coordinates = np.asarray(mesh.points, dtype=np.float64)
    tolerance = 1.0e-10
    bottom = np.abs(coordinates[:, 1]) <= tolerance
    sides = (
        (np.abs(coordinates[:, 0]) <= tolerance)
        | (np.abs(coordinates[:, 0] - 1.0) <= tolerance)
        | (np.abs(coordinates[:, 2]) <= tolerance)
        | (np.abs(coordinates[:, 2] - 1.0) <= tolerance)
    )
    fixed = bottom
    top_y = muscle_top + top_fat_thickness
    top = np.abs(coordinates[:, 1] - top_y) <= tolerance
    target = top & ~fixed
    mesh.point_data["FixedBottom"] = bottom.astype(np.int8)
    mesh.point_data["FixedSide"] = sides.astype(np.int8)
    mesh.point_data[toy.FIXED_BOUNDARY] = fixed.astype(np.int8)
    mesh.point_data[toy.TOP_SURFACE_MASK] = top.astype(np.int8)
    mesh.point_data[toy.TARGET_SURFACE_MASK] = target.astype(np.int8)
    mesh.point_data[FIXED_MASK.vtk] = np.repeat(fixed[:, None], 3, axis=1)
    mesh.point_data[FIXED_VALUE.vtk] = np.zeros((mesh.n_points, 3), dtype=np.float64)
    if not np.array_equal(np.flatnonzero(top), np.sort(point_ids[-1].ravel())):
        raise RuntimeError("structured top-point identity changed")
    return mesh


def isochoric_activation_inv(strength: np.ndarray) -> np.ndarray:
    """Map x-contraction strength to det(A)=1 uniaxial active strain.

    The constitutive model stores ``A_inv - I``.  Therefore ``strength=0.25``
    means the natural fibre stretch is ``lambda_x=1/1.25=0.8``.  The two
    transverse natural stretches are ``lambda_x**-0.5``.
    """
    one_plus = 1.0 + strength
    if np.any(one_plus <= 0.0):
        raise ValueError("I + ActivationInv_x must stay positive definite")
    transverse_inv = np.sqrt(1.0 / one_plus)
    values = np.zeros((strength.size, 6), dtype=np.float64)
    values[:, 0] = strength
    values[:, 1] = transverse_inv - 1.0
    values[:, 2] = transverse_inv - 1.0
    return values


def active_strain_metrics(activation_inv: np.ndarray) -> dict[str, float]:
    a_inv_eigenvalues = 1.0 + activation_inv[:, :3]
    a_eigenvalues = 1.0 / a_inv_eigenvalues
    determinant = np.prod(a_eigenvalues, axis=1)
    return {
        "active_strain/A_lambda_min": float(a_eigenvalues.min()),
        "active_strain/A_lambda_max": float(a_eigenvalues.max()),
        "active_strain/A_det_min": float(determinant.min()),
        "active_strain/A_det_max": float(determinant.max()),
        "active_strain/A_det_max_abs_error": float(np.max(np.abs(determinant - 1.0))),
        "active_strain/A_inv_lambda_min": float(a_inv_eigenvalues.min()),
        "active_strain/A_inv_lambda_max": float(a_inv_eigenvalues.max()),
    }


def activation_fields(
    cfg: Config, mesh: pv.UnstructuredGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    active_ids = np.flatnonzero(
        np.asarray(mesh.cell_data[toy.ACTIVE_FRACTION], dtype=np.float64)
        > toy.ACTIVE_FRACTION_TOL
    )
    if active_ids.size == 0:
        raise RuntimeError("structured mesh has no active muscle tetrahedra")
    tets = toy.tetra_cells(mesh)
    centers = np.asarray(mesh.points, dtype=np.float64)[tets[active_ids]].mean(axis=1)
    raw_pattern = np.cos(2.0 * np.pi * cfg.wave_number * centers[:, 0]) * np.cos(
        2.0 * np.pi * cfg.wave_number * centers[:, 2]
    )
    tets = toy.tetra_cells(mesh)[active_ids]
    tetra = np.asarray(mesh.points, dtype=np.float64)[tets]
    volumes = (
        np.abs(
            np.linalg.det(
                np.stack(
                    (
                        tetra[:, 1] - tetra[:, 0],
                        tetra[:, 2] - tetra[:, 0],
                        tetra[:, 3] - tetra[:, 0],
                    ),
                    axis=2,
                )
            )
        )
        / 6.0
    )
    pattern_mean = float(np.average(raw_pattern, weights=volumes))
    centered = raw_pattern - pattern_mean
    pattern_rms = float(np.sqrt(np.average(centered**2, weights=volumes)))
    if pattern_rms <= 0.0:
        raise RuntimeError("sampled activation pattern has zero RMS")
    pattern = centered / pattern_rms
    uniform_strength = np.full(active_ids.size, cfg.mean_activation_inv_x)
    bumpy_strength = uniform_strength + cfg.modulation_rms * pattern
    uniform = isochoric_activation_inv(uniform_strength)
    bumpy = isochoric_activation_inv(bumpy_strength)
    diagnostics = {
        "activation/raw_pattern_volume_weighted_mean": pattern_mean,
        "activation/normalized_pattern_volume_weighted_mean": float(
            np.average(pattern, weights=volumes)
        ),
        "activation/normalized_pattern_volume_weighted_rms": float(
            np.sqrt(np.average(pattern**2, weights=volumes))
        ),
        "activation/modulation_inv_x_volume_weighted_mean": float(
            np.average(bumpy_strength - uniform_strength, weights=volumes)
        ),
        "activation/modulation_inv_x_volume_weighted_rms": float(
            np.sqrt(
                np.average(
                    (bumpy_strength - uniform_strength) ** 2,
                    weights=volumes,
                )
            )
        ),
        **{
            f"uniform/{key}": value
            for key, value in active_strain_metrics(uniform).items()
        },
        **{
            f"bumpy/{key}": value for key, value in active_strain_metrics(bumpy).items()
        },
    }
    if abs(diagnostics["activation/normalized_pattern_volume_weighted_mean"]) > 1e-12:
        raise RuntimeError("normalized bumpy activation does not have zero mean")
    if not math.isclose(
        diagnostics["activation/modulation_inv_x_volume_weighted_rms"],
        cfg.modulation_rms,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("sampled bumpy activation does not have requested RMS")
    return active_ids, centers, uniform, bumpy, diagnostics


def set_active_activation(
    forward: Any,
    *,
    active_ids: np.ndarray,
    active_activation_inv: np.ndarray,
    n_cells: int,
) -> None:
    active_ids_t = torch.as_tensor(active_ids, dtype=torch.long, device="cuda")
    active_values_t = torch.as_tensor(
        active_activation_inv,
        dtype=torch.float64,
        device="cuda",
    )
    materials = toy.material_tree(
        forward.model.get_materials(),
        active_values_t,
        active_ids_t,
        n_cells,
    )
    forward.model.set_materials(materials)


def solve(forward: Any) -> tuple[Any, np.ndarray, float]:
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        solution = forward.step()
    elapsed = time.perf_counter() - start
    return solution, toy.to_numpy(forward.state.u).copy(), elapsed


def deformation_metrics(
    mesh: pv.UnstructuredGrid, displacement: np.ndarray
) -> dict[str, float | int]:
    points = np.asarray(mesh.points, dtype=np.float64)
    deformed = points + displacement
    tets = toy.tetra_cells(mesh)

    def edge_matrices(value: np.ndarray) -> np.ndarray:
        tetra = value[tets]
        return np.stack(
            (
                tetra[:, 1] - tetra[:, 0],
                tetra[:, 2] - tetra[:, 0],
                tetra[:, 3] - tetra[:, 0],
            ),
            axis=2,
        )

    deformation_gradient = edge_matrices(deformed) @ np.linalg.inv(
        edge_matrices(points)
    )
    determinant = np.linalg.det(deformation_gradient)
    return {
        "detF/min": float(determinant.min()),
        "detF/q001": float(np.quantile(determinant, 0.001)),
        "detF/inverted": int(np.count_nonzero(determinant <= 0.0)),
        "detF/below_0p2": int(np.count_nonzero(determinant < 0.2)),
    }


def surface_transfer_metrics(
    cfg: Config,
    *,
    point_ids: np.ndarray,
    uniform_u: np.ndarray,
    bumpy_u: np.ndarray,
    direct_u: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    top_ids = point_ids[-1]
    interface_layer = layer_count(
        cfg.bottom_fat_thickness,
        cfg.vertical_spacing,
        name="bottom_fat_thickness",
    ) + layer_count(
        cfg.muscle_thickness,
        cfg.vertical_spacing,
        name="muscle_thickness",
    )
    interface_ids = point_ids[interface_layer]
    xs = np.linspace(0.0, 1.0, cfg.nx + 1)
    zs = np.linspace(0.0, 1.0, cfg.nz + 1)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    pattern = np.cos(2.0 * np.pi * cfg.wave_number * xx) * np.cos(
        2.0 * np.pi * cfg.wave_number * zz
    )

    uniform_y = uniform_u[top_ids, 1]
    bumpy_y = bumpy_u[top_ids, 1]
    direct_y = direct_u[top_ids, 1]
    interface_uniform_y = uniform_u[interface_ids, 1]
    interface_bumpy_y = bumpy_u[interface_ids, 1]
    interface_direct_y = direct_u[interface_ids, 1]
    induced = bumpy_y - uniform_y
    interface_induced = interface_bumpy_y - interface_uniform_y
    direct_induced = direct_y - uniform_y
    direct_interface_induced = interface_direct_y - interface_uniform_y
    direct_difference = direct_y - bumpy_y

    crop_x = int(math.ceil(cfg.crop_fraction * cfg.nx))
    crop_z = int(math.ceil(cfg.crop_fraction * cfg.nz))
    stop_x = cfg.nx + 1 - crop_x if crop_x else None
    stop_z = cfg.nz + 1 - crop_z if crop_z else None
    crop = np.s_[crop_x:stop_x, crop_z:stop_z]
    induced_crop = induced[crop]
    pattern_crop = pattern[crop]
    weights_x = np.ones(cfg.nx + 1, dtype=np.float64)
    weights_z = np.ones(cfg.nz + 1, dtype=np.float64)
    weights_x[[0, -1]] = 0.5
    weights_z[[0, -1]] = 0.5
    weights = np.outer(weights_x, weights_z)[crop]

    def modal_metrics(field: np.ndarray) -> dict[str, float]:
        field_crop = field[crop]
        pattern_mean = float(np.average(pattern_crop, weights=weights))
        pattern_centered = pattern_crop - pattern_mean
        pattern_rms = float(np.sqrt(np.average(pattern_centered**2, weights=weights)))
        normalized_pattern = pattern_centered / pattern_rms
        field_mean = float(np.average(field_crop, weights=weights))
        centered = field_crop - field_mean
        coefficient = float(np.average(centered * normalized_pattern, weights=weights))
        rms = float(np.sqrt(np.average(centered**2, weights=weights)))
        residual = centered - coefficient * normalized_pattern
        residual_rms = float(np.sqrt(np.average(residual**2, weights=weights)))
        correlation = coefficient / max(rms, 1.0e-30)
        return {
            "mean": field_mean,
            "rms": rms,
            "p95_p05": float(
                np.quantile(field_crop, 0.95) - np.quantile(field_crop, 0.05)
            ),
            "modal_coefficient": coefficient,
            "modal_amplitude_abs": abs(coefficient),
            "modal_correlation": correlation,
            "modal_energy_fraction": min(1.0, correlation**2),
            "modal_residual_rms": residual_rms,
            "pattern_weighted_mean": pattern_mean,
            "pattern_weighted_rms": pattern_rms,
        }

    top = modal_metrics(induced)
    interface = modal_metrics(interface_induced)
    direct_top = modal_metrics(direct_induced)
    direct_interface = modal_metrics(direct_interface_induced)
    transmission = top["modal_amplitude_abs"] / max(
        interface["modal_amplitude_abs"], 1.0e-30
    )
    direct_transmission = direct_top["modal_amplitude_abs"] / max(
        direct_interface["modal_amplitude_abs"], 1.0e-30
    )

    dx = 1.0 / cfg.nx
    dz = 1.0 / cfg.nz
    sigma = (
        cfg.highpass_smoothing_length / dx,
        cfg.highpass_smoothing_length / dz,
    )
    highpass = induced - gaussian_filter(induced, sigma=sigma, mode="reflect")
    highpass_crop = highpass[crop]
    highpass_rms = float(np.sqrt(np.average(highpass_crop**2, weights=weights)))

    laplacian = (
        induced[2:, 1:-1] - 2.0 * induced[1:-1, 1:-1] + induced[:-2, 1:-1]
    ) / dx**2 + (
        induced[1:-1, 2:] - 2.0 * induced[1:-1, 1:-1] + induced[1:-1, :-2]
    ) / dz**2
    lap_crop_x = max(0, crop_x - 1)
    lap_crop_z = max(0, crop_z - 1)
    lap_crop = laplacian[
        lap_crop_x : laplacian.shape[0] - lap_crop_x,
        lap_crop_z : laplacian.shape[1] - lap_crop_z,
    ]
    laplacian_rms = float(np.sqrt(np.mean(lap_crop**2)))

    branch_rms = float(
        np.sqrt(np.average(direct_difference[crop] ** 2, weights=weights))
    )
    bumpy_rms = float(np.sqrt(np.mean((bumpy_y[crop] - bumpy_y[crop].mean()) ** 2)))
    uniform_rms = float(
        np.sqrt(np.mean((uniform_y[crop] - uniform_y[crop].mean()) ** 2))
    )
    metrics: dict[str, float | int] = {
        "surface/crop_points": int(induced_crop.size),
        "surface/crop_x_nodes": int(crop_x),
        "surface/crop_z_nodes": int(crop_z),
        "surface/uniform_y_rms": uniform_rms,
        "surface/bumpy_y_rms": bumpy_rms,
        "source/interface_induced_y_rms": interface["rms"],
        "source/interface_modal_coefficient": interface["modal_coefficient"],
        "source/interface_modal_amplitude_abs": interface["modal_amplitude_abs"],
        "source/interface_modal_correlation": interface["modal_correlation"],
        "source/interface_modal_energy_fraction": interface["modal_energy_fraction"],
        "surface/top_induced_y_rms": top["rms"],
        "surface/top_induced_y_p95_p05": top["p95_p05"],
        "surface/top_modal_coefficient": top["modal_coefficient"],
        "surface/top_modal_amplitude_abs": top["modal_amplitude_abs"],
        "surface/top_modal_correlation": top["modal_correlation"],
        "surface/top_modal_energy_fraction": top["modal_energy_fraction"],
        "surface/top_modal_residual_rms": top["modal_residual_rms"],
        "transfer/modal_transmission": transmission,
        "transfer/modal_gain_from_activation": top["modal_amplitude_abs"]
        / cfg.modulation_rms,
        # Backward-compatible aliases used by the plotting/report helpers.
        "transfer/induced_y_rms": top["rms"],
        "transfer/modal_coefficient": top["modal_coefficient"],
        "transfer/modal_amplitude_abs": top["modal_amplitude_abs"],
        "transfer/modal_gain": transmission,
        "transfer/modal_correlation": top["modal_correlation"],
        "transfer/modal_residual_rms": top["modal_residual_rms"],
        "transfer/highpass_rms": highpass_rms,
        "transfer/laplacian_rms": laplacian_rms,
        "direct/source_interface_modal_amplitude_abs": direct_interface[
            "modal_amplitude_abs"
        ],
        "direct/surface_top_modal_amplitude_abs": direct_top["modal_amplitude_abs"],
        "direct/transfer_modal_transmission": direct_transmission,
        "branch/direct_minus_continuation_top_y_rms": branch_rms,
        "branch/difference_over_induced_signal": branch_rms / max(top["rms"], 1.0e-30),
        "branch/transmission_relative_difference": abs(
            direct_transmission - transmission
        )
        / max(transmission, 1.0e-30),
        "analysis/nodal_pattern_weighted_mean": top["pattern_weighted_mean"],
        "analysis/nodal_pattern_weighted_rms": top["pattern_weighted_rms"],
    }
    arrays = {
        "x": xs,
        "z": zs,
        "activation_pattern": pattern,
        "uniform_y": uniform_y,
        "bumpy_y": bumpy_y,
        "direct_y": direct_y,
        "induced_y": induced,
        "interface_uniform_y": interface_uniform_y,
        "interface_bumpy_y": interface_bumpy_y,
        "interface_direct_y": interface_direct_y,
        "interface_induced_y": interface_induced,
        "induced_highpass_y": highpass,
    }
    return metrics, arrays


def full_activation(
    n_cells: int, active_ids: np.ndarray, active_values: np.ndarray
) -> np.ndarray:
    values = np.zeros((n_cells, 6), dtype=np.float64)
    values[active_ids] = active_values
    return values


def solve_case(
    cfg: Config,
    *,
    label: str,
    top_fat_thickness: float,
    top_layers: int,
    tets: np.ndarray,
    point_ids: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    mesh = make_mesh(
        cfg,
        top_fat_thickness=top_fat_thickness,
        tets=tets,
        point_ids=point_ids,
    )
    (
        active_ids,
        active_centers,
        uniform_activation,
        bumpy_activation,
        activation_diagnostics,
    ) = activation_fields(cfg, mesh)
    variant = toy.LossVariant(
        name="l2",
        skin_energy=False,
        skin_prestrain=False,
        activation_mode="per-tet",
    )
    resolution = toy.ResolutionSpec(
        name=f"structured-{cfg.nx}", lr=min(1.0 / cfg.nx, 1.0 / cfg.nz)
    )
    case = toy.ToyCase(
        resolution=resolution,
        mode="squash",
        variant=variant,
        target_y=0.0,
    )

    continuation_forward, _ = toy.build_forward(mesh.copy(deep=True), case)
    continuation_forward.optimizer = continuation_forward.default_optimizer(
        max_steps=cfg.forward_max_steps,
        atol=cfg.forward_atol,
        rtol=cfg.forward_rtol,
    )
    set_active_activation(
        continuation_forward,
        active_ids=active_ids,
        active_activation_inv=uniform_activation,
        n_cells=mesh.n_cells,
    )
    uniform_solution, uniform_u, uniform_elapsed = solve(continuation_forward)
    stage_rows: list[dict[str, Any]] = [
        {
            "stage": "uniform",
            "alpha": 0.0,
            "elapsed_s": uniform_elapsed,
            **toy.forward_solution_metrics(uniform_solution),
            **deformation_metrics(mesh, uniform_u),
        }
    ]
    for alpha in cfg.continuation_alphas:
        activation = uniform_activation + alpha * (
            bumpy_activation - uniform_activation
        )
        set_active_activation(
            continuation_forward,
            active_ids=active_ids,
            active_activation_inv=activation,
            n_cells=mesh.n_cells,
        )
        solution, continuation_u, elapsed = solve(continuation_forward)
        stage_rows.append(
            {
                "stage": "bumpy-continuation",
                "alpha": float(alpha),
                "elapsed_s": elapsed,
                **toy.forward_solution_metrics(solution),
                **deformation_metrics(mesh, continuation_u),
            }
        )
    bumpy_u = continuation_u

    direct_forward, _ = toy.build_forward(mesh.copy(deep=True), case)
    direct_forward.optimizer = direct_forward.default_optimizer(
        max_steps=cfg.forward_max_steps,
        atol=cfg.forward_atol,
        rtol=cfg.forward_rtol,
    )
    set_active_activation(
        direct_forward,
        active_ids=active_ids,
        active_activation_inv=bumpy_activation,
        n_cells=mesh.n_cells,
    )
    direct_solution, direct_u, direct_elapsed = solve(direct_forward)
    stage_rows.append(
        {
            "stage": "bumpy-direct",
            "alpha": 1.0,
            "elapsed_s": direct_elapsed,
            **toy.forward_solution_metrics(direct_solution),
            **deformation_metrics(mesh, direct_u),
        }
    )
    if cfg.require_convergence:
        failed = [row for row in stage_rows if row["forward/success"] is not True]
        if failed:
            raise RuntimeError(f"{label} has failed forward stages: {failed}")
    invalid_geometry = [
        row
        for row in stage_rows
        if row["detF/min"] < cfg.minimum_det_f
        or row["detF/q001"] < cfg.minimum_det_f_q001
        or row["detF/inverted"] != 0
    ]
    if invalid_geometry:
        raise RuntimeError(
            f"{label} failed deformation-quality gates: {invalid_geometry}"
        )

    transfer, top_arrays = surface_transfer_metrics(
        cfg,
        point_ids=point_ids,
        uniform_u=uniform_u,
        bumpy_u=bumpy_u,
        direct_u=direct_u,
    )
    if (
        cfg.require_branch_agreement
        and transfer["branch/difference_over_induced_signal"]
        > cfg.max_branch_difference_over_signal
    ):
        raise RuntimeError(
            f"{label} direct/continuation branch difference exceeds threshold: {transfer}"
        )

    full_uniform = full_activation(mesh.n_cells, active_ids, uniform_activation)
    full_bumpy = full_activation(mesh.n_cells, active_ids, bumpy_activation)
    zero_target = np.zeros_like(bumpy_u)
    row: dict[str, Any] = {
        "label": label,
        "top_fat_thickness": float(top_fat_thickness),
        "total_height": float(
            cfg.bottom_fat_thickness + cfg.muscle_thickness + top_fat_thickness
        ),
        "n_points": int(mesh.n_points),
        "n_tets": int(mesh.n_cells),
        "n_active_tets": int(active_ids.size),
        "top_layers": int(top_layers),
        "vertical_spacing": float(cfg.vertical_spacing),
        "connectivity_sha256": sha256_array(toy.tetra_cells(mesh)),
        "shared_lower_connectivity_sha256": sha256_array(
            toy.tetra_cells(mesh)[
                : (
                    layer_count(
                        cfg.bottom_fat_thickness,
                        cfg.vertical_spacing,
                        name="bottom_fat_thickness",
                    )
                    + layer_count(
                        cfg.muscle_thickness,
                        cfg.vertical_spacing,
                        name="muscle_thickness",
                    )
                )
                * cfg.nx
                * cfg.nz
                * 6
            ]
        ),
        "active_ids_sha256": sha256_array(active_ids),
        "active_centers_xz_sha256": sha256_array(active_centers[:, (0, 2)]),
        "uniform_activation_sha256": sha256_array(uniform_activation),
        "bumpy_activation_sha256": sha256_array(bumpy_activation),
        "activation/pattern_rms": float(
            np.sqrt(np.mean(((bumpy_activation - uniform_activation)[:, 0]) ** 2))
        ),
        "activation/x_min": float(bumpy_activation[:, 0].min()),
        "activation/x_max": float(bumpy_activation[:, 0].max()),
        "activation/fat_nonzero_entries": 0,
        "stages": stage_rows,
        **activation_diagnostics,
        **transfer,
    }

    case_root = output_root / label
    case_root.mkdir(parents=True, exist_ok=False)
    result_path = case_root / f"10-{label}-bumpy-activation-transfer.vtu"
    grid_path = case_root / f"10-{label}-top-transfer-grid.npz"
    summary_path = case_root / f"10-{label}-summary.json"
    result = toy.make_result_mesh(
        mesh,
        zero_target,
        bumpy_u,
        full_bumpy,
        row,
    )
    result.point_data["UniformDisplacement"] = uniform_u
    result.point_data["BumpyDisplacement"] = bumpy_u
    result.point_data["BumpyDirectDisplacement"] = direct_u
    result.point_data["BumpyMinusUniform"] = bumpy_u - uniform_u
    result.point_data["BumpyMinusUniformY"] = (bumpy_u - uniform_u)[:, 1]
    result.point_data["BumpyMinusContinuationY"] = (direct_u - bumpy_u)[:, 1]
    result.cell_data["UniformActivationInv"] = full_uniform
    result.cell_data["BumpyActivationInv"] = full_bumpy
    result.cell_data["ActivationInvXModulation"] = full_bumpy[:, 0] - full_uniform[:, 0]
    result.cell_data["BumpyActiveStrainDeterminant"] = np.prod(
        1.0 / (1.0 + full_bumpy[:, :3]), axis=1
    )
    melon.save(result, result_path)
    np.savez_compressed(grid_path, **top_arrays)
    row["result_path"] = str(result_path.resolve())
    row["top_grid_path"] = str(grid_path.resolve())
    row["summary_path"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for path in (result_path, grid_path, summary_path):
        cherries.log_output(path)
    logger.info(
        "%s h=%.3f: modal gain %.6g, induced RMS %.6g, branch ratio %.3g",
        label,
        top_fat_thickness,
        row["transfer/modal_gain"],
        row["transfer/induced_y_rms"],
        row["branch/difference_over_induced_signal"],
    )
    return row


def dark_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "#0b0f14",
            "axes.facecolor": "#0b0f14",
            "savefig.facecolor": "#0b0f14",
            "text.color": "#f4f6f8",
            "axes.labelcolor": "#f4f6f8",
            "axes.edgecolor": "#d7dce2",
            "xtick.color": "#f4f6f8",
            "ytick.color": "#f4f6f8",
            "grid.color": "#6b7280",
            "grid.alpha": 0.35,
            "font.size": 16,
        }
    )


def plot_metric(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    x = np.asarray([row["top_fat_thickness"] for row in rows], dtype=np.float64)
    y = np.asarray([row[metric] for row in rows], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    axis.plot(x, y, color="#67c1b6", marker="o", linewidth=3.0, markersize=10)
    for row, xi, yi in zip(rows, x, y, strict=True):
        axis.annotate(
            f"{row['label']}\n{yi:.4g}",
            (xi, yi),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=14,
        )
    reduction = 1.0 - y[-1] / y[0]
    axis.text(
        0.04,
        0.06,
        f"thin -> thick: {reduction:.1%} reduction",
        transform=axis.transAxes,
        color="#67c1b6",
        fontsize=18,
        fontweight="bold",
    )
    axis.set_xlabel("top-fat thickness [model length]")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=23, fontweight="bold")
    axis.set_xticks(x)
    axis.grid(True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=200)
    plt.close(figure)
    temporary.replace(path)


def plot_top_field(row: dict[str, Any], path: Path, *, common_limit: float) -> None:
    with np.load(row["top_grid_path"]) as arrays:
        x = arrays["x"]
        z = arrays["z"]
        induced = arrays["induced_y"]
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.pcolormesh(
        z,
        x,
        induced,
        shading="auto",
        cmap="coolwarm",
        vmin=-common_limit,
        vmax=common_limit,
    )
    axis.set_aspect("equal")
    axis.set_xlabel("z [model length]")
    axis.set_ylabel("x [model length]")
    axis.set_title(
        f"{row['label'].capitalize()} fat: induced surface response",
        fontsize=18,
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("vertical displacement [model length]")
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=200)
    plt.close(figure)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "label",
        "top_fat_thickness",
        "n_points",
        "n_tets",
        "n_active_tets",
        "top_layers",
        "activation/pattern_rms",
        "surface/uniform_y_rms",
        "surface/bumpy_y_rms",
        "transfer/induced_y_rms",
        "transfer/modal_amplitude_abs",
        "transfer/modal_gain",
        "source/interface_modal_amplitude_abs",
        "surface/top_modal_amplitude_abs",
        "transfer/modal_transmission",
        "surface/top_modal_energy_fraction",
        "transfer/modal_correlation",
        "transfer/highpass_rms",
        "transfer/laplacian_rms",
        "branch/direct_minus_continuation_top_y_rms",
        "branch/difference_over_induced_signal",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main(cfg: Config) -> None:
    cases = validate_config(cfg)
    toy.configure_runtime()
    cfg.output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_root = cfg.output_summary.parent / "10-bumpy-activation-transfer"
    if cfg.output_summary.exists() or output_root.exists():
        raise FileExistsError("refusing to overwrite existing experiment outputs")
    output_root.mkdir(parents=True)

    bottom_layers = layer_count(
        cfg.bottom_fat_thickness,
        cfg.vertical_spacing,
        name="bottom_fat_thickness",
    )
    muscle_layers = layer_count(
        cfg.muscle_thickness,
        cfg.vertical_spacing,
        name="muscle_thickness",
    )
    rows: list[dict[str, Any]] = []
    for step, (label, thickness, top_layers) in enumerate(cases):
        ny = bottom_layers + muscle_layers + top_layers
        tets, point_ids = structured_connectivity(cfg.nx, ny, cfg.nz)
        row = solve_case(
            cfg,
            label=label,
            top_fat_thickness=thickness,
            top_layers=top_layers,
            tets=tets,
            point_ids=point_ids,
            output_root=output_root,
        )
        rows.append(row)
        cherries.set_step(step)
        cherries.log_metrics(
            {
                f"{label}/top_fat_thickness": thickness,
                f"{label}/modal_gain": row["transfer/modal_gain"],
                f"{label}/modal_transmission": row["transfer/modal_transmission"],
                f"{label}/top_modal_amplitude": row["surface/top_modal_amplitude_abs"],
                f"{label}/induced_y_rms": row["transfer/induced_y_rms"],
                f"{label}/highpass_rms": row["transfer/highpass_rms"],
                f"{label}/branch_ratio": row["branch/difference_over_induced_signal"],
            }
        )

    invariant_keys = (
        "n_active_tets",
        "shared_lower_connectivity_sha256",
        "active_ids_sha256",
        "active_centers_xz_sha256",
        "uniform_activation_sha256",
        "bumpy_activation_sha256",
    )
    for key in invariant_keys:
        if len({row[key] for row in rows}) != 1:
            raise RuntimeError(f"fixed-connectivity invariant changed at {key}")

    thin, thick = rows[0], rows[-1]
    effects = {
        metric: 1.0 - float(thick[metric]) / float(thin[metric])
        for metric in (
            "transfer/induced_y_rms",
            "transfer/modal_amplitude_abs",
            "transfer/modal_gain",
            "source/interface_modal_amplitude_abs",
            "surface/top_modal_amplitude_abs",
            "transfer/modal_transmission",
            "transfer/highpass_rms",
            "transfer/laplacian_rms",
        )
    }
    if effects["transfer/modal_gain"] <= 0.0:
        raise RuntimeError(
            f"thicker fat did not attenuate the imposed activation mode: {effects}"
        )

    dark_style()
    plot_paths = {
        "modal_gain": output_root / "10-modal-gain-vs-thickness.png",
        "induced_rms": output_root / "10-induced-rms-vs-thickness.png",
    }
    plot_metric(
        rows,
        metric="transfer/modal_gain",
        ylabel="surface / muscle-interface modal amplitude",
        title=f"Transmission of k={cfg.wave_number} bumpy muscle activation",
        path=plot_paths["modal_gain"],
    )
    plot_metric(
        rows,
        metric="transfer/induced_y_rms",
        ylabel="RMS of bumpy - uniform surface displacement",
        title="Activation-induced surface bumpiness",
        path=plot_paths["induced_rms"],
    )
    field_paths: dict[str, str] = {}
    common_field_limit = 0.0
    for row in rows:
        with np.load(row["top_grid_path"]) as arrays:
            common_field_limit = max(
                common_field_limit,
                float(np.max(np.abs(arrays["induced_y"]))),
            )
    for row in rows:
        path = output_root / f"10-{row['label']}-top-induced-field.png"
        plot_top_field(row, path, common_limit=common_field_limit)
        field_paths[row["label"]] = str(path.resolve())

    csv_path = output_root / "10-bumpy-activation-transfer.csv"
    write_csv(csv_path, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "status": "ok",
        "question": (
            "Does increasing top-fat thickness attenuate the surface response to "
            "the same spatially bumpy muscle activation field?"
        ),
        "paired_design": {
            "uniform_activation_inv_x": cfg.mean_activation_inv_x,
            "bumpy_activation": (
                "volume-weighted zero-mean, unit-RMS cos(2*pi*k*x) * "
                "cos(2*pi*k*z) modulation of ActivationInv_x; transverse "
                "components enforce det(A)=1"
            ),
            "modulation_rms": cfg.modulation_rms,
            "wave_number": cfg.wave_number,
            "primary_response": "bumpy displacement minus uniform displacement",
            "top_fat_thicknesses": list(cfg.top_fat_thicknesses),
            "paired_mesh_identical_within_thickness": True,
            "deterministic_nested_mesh_across_thickness": True,
            "shared_lower_slab_connectivity": True,
            "active_strain_is_isochoric": True,
            "natural_uniform_fibre_stretch": 1.0 / (1.0 + cfg.mean_activation_inv_x),
            "lateral_sides": "traction-free",
            "bottom": "fully fixed",
            "skin_energy_enabled": False,
            "continuation_alphas": list(cfg.continuation_alphas),
            "direct_branch_check": True,
        },
        "mesh": {
            "nx": cfg.nx,
            "nz": cfg.nz,
            "vertical_spacing": cfg.vertical_spacing,
            "bottom_layers": bottom_layers,
            "muscle_layers": muscle_layers,
            "top_layers_by_case": {label: top_layers for label, _, top_layers in cases},
            "bottom_fat_thickness": cfg.bottom_fat_thickness,
            "muscle_thickness": cfg.muscle_thickness,
            "fat_E_MPa": toy.FAT_E,
            "fat_nu": toy.FAT_NU,
            "muscle_E_MPa": toy.MUSCLE_E,
            "muscle_nu": toy.MUSCLE_NU,
        },
        "analysis": {
            "crop_fraction": cfg.crop_fraction,
            "highpass_smoothing_length": cfg.highpass_smoothing_length,
            "modal_gain_is_primary": True,
            "primary_metric": "transfer/modal_transmission",
            "surface_amplitude_metric": "surface/top_modal_amplitude_abs",
        },
        "solver": {
            "max_steps": cfg.forward_max_steps,
            "atol": cfg.forward_atol,
            "rtol": cfg.forward_rtol,
            "minimum_det_f": cfg.minimum_det_f,
            "minimum_det_f_q001": cfg.minimum_det_f_q001,
            "branch_difference_over_signal_max": cfg.max_branch_difference_over_signal,
        },
        "shared_invariants": {key: rows[0][key] for key in invariant_keys},
        "effects_thick_vs_thin": effects,
        "cases": rows,
        "plots": {key: str(path.resolve()) for key, path in plot_paths.items()},
        "top_field_plots": field_paths,
        "csv_path": str(csv_path.resolve()),
    }
    cfg.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for path in (
        *plot_paths.values(),
        *(Path(value) for value in field_paths.values()),
    ):
        cherries.log_output(path)
    cherries.log_output(csv_path)
    logger.info(
        "Wrote corrected bumpy-activation transfer summary to %s", cfg.output_summary
    )


if __name__ == "__main__":
    cherries.main(main)
