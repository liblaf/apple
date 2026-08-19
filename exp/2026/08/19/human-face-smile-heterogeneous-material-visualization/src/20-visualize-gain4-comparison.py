# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DESIGN = "corrected-isface-gain4-vs-gain8-linear-young-modulus"
GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]

CORRECTED_SKIN = (
    REPO_ROOT / "exp/2026/08/18/human-face-smile-plane-stress-skin/data/"
    "10-corrected-baseline/skin-isface-e0200-p000.vtp"
)
DRIVER_SKIN = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/data/"
    "10-material-candidates/skin-e100-p000.vtp"
)

CORRECTED_SKIN_SIZE_BYTES = 1_138_550
CORRECTED_SKIN_SHA256 = (
    "4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f"
)
DRIVER_SKIN_SIZE_BYTES = 38_742_137
DRIVER_SKIN_SHA256 = "ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f"

EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_SKIN_AREA_M2 = 0.04287998059707303
EXPECTED_SKIN_BOUNDARY_EDGES = 707
EXPECTED_SKIN_INTERIOR_EDGES = 44_495
EXPECTED_EXPANSION_CELLS = 16_770

EXPECTED_RAW_HASHES = {
    "corrected_triangle_keys": (
        "dca8d77662f49b54250657424cfc29a3b438437081f83d65f7e108f312da2310"
    ),
    "mapped_driver_cell_indices": (
        "13458107ceef23ecc144340101574f3fb4f2e157f90a9251434e7b09a86a66c3"
    ),
    "rest_area": "5a7b8eb9861fa509212afd610c60183f894b80db8ded53d22f3f9045bc6889de",
    "expansion_weight": (
        "68de7dd3189c99100e4648845acf67d37e0d8a3a43d0eec3425df4168cd35910"
    ),
}

EXPECTED_GAIN_STATS = {
    "gain4": {
        "gain": 4.0,
        "softening_exponent_sha256_le_f8": (
            "e9578bd43feeb916449bba546c2a737b57087dacab60931623a78776975fe730"
        ),
        "young_sha256_le_f8": (
            "84c15003461f3f69e212be9e60d2e99dc8747072f963a48398ab3fb9d16ecc9c"
        ),
        "lambda_plane_stress_sha256_le_f8": (
            "a81d68e81be3b3d43a46960677e71f0ee86e5fbb99e8900d9951e0a10a10e658"
        ),
        "mu_sha256_le_f8": (
            "7b0bc6ea603c60100050727ca68a8d86faa9a09a8080de33e1c31ad8e9194113"
        ),
        "saturated_cells": 1_946,
        "saturated_rest_area_fraction": 0.03827246051715845,
        "below_fat_E_cells": 2_790,
        "below_fat_E_rest_area_fraction": 0.06004361816654654,
        "area_weighted_mean_MPa": 0.1416524140567568,
    },
    "gain8": {
        "gain": 8.0,
        "softening_exponent_sha256_le_f8": (
            "2a41c846db9859de82332fd09322c7449807d12ae93435428f98534088a1b356"
        ),
        "young_sha256_le_f8": (
            "d42fa66ea54c47890e184fd8b670a7e80cd5621e337ee81b03f658a8b9821a44"
        ),
        "lambda_plane_stress_sha256_le_f8": (
            "47737694649e4753be1bee9c5b3fa86b7c17265ea3e8d85583befedd487ec3d5"
        ),
        "mu_sha256_le_f8": (
            "99d925f0b0e0f749532a440028f6c687a4f66bf90cb40551e58edea2a2dfcfd0"
        ),
        "saturated_cells": 3_643,
        "saturated_rest_area_fraction": 0.0873748559185055,
        "below_fat_E_cells": 6_327,
        "below_fat_E_rest_area_fraction": 0.18095636594650907,
        "area_weighted_mean_MPa": 0.13107545288953745,
    },
}

SKIN_E_MAX_MPA = 0.2
SKIN_E_MIN_MPA = 3.0e-4
FAT_E_MPA = 0.003
SKIN_NU = 0.49
SKIN_THICKNESS_M = 0.001
GAINS = (4.0, 8.0)

OUTPUT_JSON = GROUP_DIR / "data/20-gain4-vs-gain8-stats.json"
OUTPUT_PNG = GROUP_DIR / "data/20-gain4-vs-gain8-linear-young-modulus.png"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_corrected_skin: Path = cherries.input(CORRECTED_SKIN)
    input_driver_skin: Path = cherries.input(DRIVER_SKIN)
    output_json: Path = cherries.output("20-gain4-vs-gain8-stats.json", mkdir=True)
    output_png: Path = cherries.output(
        "20-gain4-vs-gain8-linear-young-modulus.png", mkdir=True
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _require_file_identity(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    name: str,
) -> dict[str, int | str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {name}: {path}")
    actual = _file_identity(path)
    expected = {"size_bytes": expected_size, "sha256": expected_sha256}
    if actual != expected:
        raise ValueError(f"{name} identity mismatch: {actual} != {expected}")
    return actual


def _require_path(actual: Path, expected: Path, *, name: str) -> None:
    if actual.resolve() != expected.resolve():
        raise ValueError(f"{name} must be {expected}, got {actual}")


def _temporary_png(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _validate_config(cfg: Config) -> None:
    for actual, expected, name in (
        (cfg.input_corrected_skin, CORRECTED_SKIN, "input_corrected_skin"),
        (cfg.input_driver_skin, DRIVER_SKIN, "input_driver_skin"),
        (cfg.output_json, OUTPUT_JSON, "output_json"),
        (cfg.output_png, OUTPUT_PNG, "output_png"),
    ):
        _require_path(actual, expected, name=name)
    paths = (cfg.output_json, cfg.output_png, _temporary_png(cfg.output_png))
    stale = [str(path) for path in paths if path.exists()]
    if stale:
        raise FileExistsError(
            f"refusing to overwrite visualization outputs or temporary files: {stale}"
        )


def _raw_sha256(array: np.ndarray, *, dtype: str) -> str:
    values = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _require_raw_hash(
    name: str,
    array: np.ndarray,
    expected: str,
    *,
    dtype: str = "<f8",
) -> str:
    actual = _raw_sha256(array, dtype=dtype)
    if actual != expected:
        raise ValueError(f"{name} raw hash mismatch: {actual} != {expected}")
    return actual


def _require_close(
    actual: float,
    expected: float,
    *,
    name: str,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-15,
) -> None:
    if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
        raise ValueError(f"{name} changed: {actual} != {expected}")


def _triangles(mesh: pv.PolyData, *, name: str) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size != 4 * mesh.n_cells:
        raise ValueError(f"{name} is not triangle-only PolyData")
    encoded = faces.reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        raise ValueError(f"{name} contains a non-triangle face")
    triangles = encoded[:, 1:]
    if np.any((triangles < 0) | (triangles >= mesh.n_points)):
        raise ValueError(f"{name} triangle connectivity is out of range")
    return triangles


def _global_point_ids(mesh: pv.PolyData, *, name: str) -> np.ndarray:
    if "GlobalPointId" not in mesh.point_data:
        raise KeyError(f"{name} is missing point GlobalPointId")
    raw = np.asarray(mesh.point_data["GlobalPointId"])
    ids = np.asarray(raw, dtype=np.int64)
    if raw.shape != (mesh.n_points,) or not np.array_equal(raw, ids):
        raise ValueError(f"{name} GlobalPointId is not an exact integer vector")
    if np.unique(ids).size != ids.size:
        raise ValueError(f"{name} GlobalPointId is not unique")
    return ids


def _canonical_triangle_keys(mesh: pv.PolyData, *, name: str) -> np.ndarray:
    triangles = _triangles(mesh, name=name)
    return np.sort(_global_point_ids(mesh, name=name)[triangles], axis=1)


def _map_driver_cells(
    corrected_skin: pv.PolyData,
    driver_skin: pv.PolyData,
) -> tuple[np.ndarray, dict[str, Any]]:
    corrected_keys = _canonical_triangle_keys(corrected_skin, name="corrected skin")
    driver_keys = _canonical_triangle_keys(driver_skin, name="driver skin")
    lookup: dict[tuple[int, int, int], int] = {}
    for cell_id, key_array in enumerate(driver_keys):
        key = tuple(int(value) for value in key_array)
        if key in lookup:
            raise ValueError(f"driver skin has duplicate triangle key {key}")
        lookup[key] = cell_id
    try:
        mapped = np.asarray(
            [lookup[tuple(int(value) for value in key)] for key in corrected_keys],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            f"corrected triangle key is missing from driver: {error.args[0]}"
        ) from error
    if np.unique(mapped).size != corrected_skin.n_cells:
        raise ValueError("corrected-to-driver triangle mapping is not injective")
    if not np.array_equal(driver_keys[mapped], corrected_keys):
        raise ValueError("corrected-to-driver triangle key readback failed")
    corrected_hash = _require_raw_hash(
        "corrected triangle keys",
        corrected_keys,
        EXPECTED_RAW_HASHES["corrected_triangle_keys"],
        dtype="<i8",
    )
    mapped_hash = _require_raw_hash(
        "mapped driver cell indices",
        mapped,
        EXPECTED_RAW_HASHES["mapped_driver_cell_indices"],
        dtype="<i8",
    )
    return mapped, {
        "method": "sorted GlobalPointId triangle keys",
        "driver_triangles": int(driver_skin.n_cells),
        "corrected_triangles": int(corrected_skin.n_cells),
        "mapped_unique_driver_triangles": int(np.unique(mapped).size),
        "corrected_triangle_keys_sha256_le_i8": corrected_hash,
        "mapped_driver_cell_indices_sha256_le_i8": mapped_hash,
        "exact_readback": True,
    }


def _require_cell_array(mesh: pv.DataSet, name: str) -> np.ndarray:
    if name not in mesh.cell_data:
        raise KeyError(f"mesh is missing cell array {name}")
    values = np.asarray(mesh.cell_data[name], dtype=np.float64)
    if values.shape != (mesh.n_cells,):
        raise ValueError(f"cell array {name} has shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"cell array {name} contains a non-finite value")
    return values


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: tuple[float, ...],
) -> list[float]:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    total = float(cumulative[-1])
    result: list[float] = []
    for quantile in quantiles:
        index = min(int(np.searchsorted(cumulative, quantile * total)), values.size - 1)
        result.append(float(sorted_values[index]))
    return result


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    quantiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "area_weighted_mean": float(np.average(values, weights=weights)),
        "area_weighted_rms": float(
            np.sqrt(np.average(np.square(values), weights=weights))
        ),
        "area_weighted_quantiles": {
            f"q{int(100 * quantile):02d}": value
            for quantile, value in zip(
                quantiles,
                _weighted_quantiles(values, weights, quantiles),
                strict=True,
            )
        },
    }


def _interior_cell_pairs(triangles: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    owners: dict[tuple[int, int], int] = {}
    counts: dict[tuple[int, int], int] = {}
    pairs: list[tuple[int, int]] = []
    for cell_id, triangle in enumerate(triangles):
        for left, right in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(left), int(right))))
            counts[edge] = counts.get(edge, 0) + 1
            if edge in owners:
                pairs.append((owners[edge], cell_id))
            else:
                owners[edge] = cell_id
    nonmanifold = sum(count > 2 for count in counts.values())
    boundary = sum(count == 1 for count in counts.values())
    result = np.asarray(pairs, dtype=np.int64)
    if result.shape != (EXPECTED_SKIN_INTERIOR_EDGES, 2):
        raise ValueError(f"corrected skin interior-edge shape changed: {result.shape}")
    if boundary != EXPECTED_SKIN_BOUNDARY_EDGES or nonmanifold:
        raise ValueError(
            "corrected skin edge topology changed: "
            f"boundary={boundary}, nonmanifold={nonmanifold}"
        )
    return result, {
        "interior_edges": int(result.shape[0]),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
    }


def _jump_stats(values: np.ndarray, pairs: np.ndarray) -> dict[str, float]:
    jumps = np.abs(values[pairs[:, 0]] - values[pairs[:, 1]])
    return {
        "rms": float(np.sqrt(np.mean(np.square(jumps)))),
        "q95": float(np.quantile(jumps, 0.95)),
        "q99": float(np.quantile(jumps, 0.99)),
        "max": float(jumps.max()),
    }


def _gain_fields(
    *,
    gain: float,
    expansion: np.ndarray,
    area: np.ndarray,
    pairs: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    name = f"gain{int(gain)}"
    expected = EXPECTED_GAIN_STATS[name]
    exponent = np.clip(gain * expansion, 0.0, 1.0)
    young = SKIN_E_MAX_MPA * np.exp(
        math.log(SKIN_E_MIN_MPA / SKIN_E_MAX_MPA) * exponent
    )
    lambda_ = young * SKIN_NU / (1.0 - SKIN_NU**2)
    mu = young / (2.0 * (1.0 + SKIN_NU))
    hashes = {
        "softening_exponent_sha256_le_f8": _require_raw_hash(
            f"{name} softening exponent",
            exponent,
            expected["softening_exponent_sha256_le_f8"],
        ),
        "young_sha256_le_f8": _require_raw_hash(
            f"{name} Young's modulus", young, expected["young_sha256_le_f8"]
        ),
        "lambda_plane_stress_sha256_le_f8": _require_raw_hash(
            f"{name} plane-stress lambda",
            lambda_,
            expected["lambda_plane_stress_sha256_le_f8"],
        ),
        "mu_sha256_le_f8": _require_raw_hash(
            f"{name} mu", mu, expected["mu_sha256_le_f8"]
        ),
    }
    saturated = exponent == 1.0
    below_fat = young < FAT_E_MPA
    observed = {
        "saturated_cells": int(np.count_nonzero(saturated)),
        "saturated_rest_area_fraction": float(area[saturated].sum() / area.sum()),
        "below_fat_E_cells": int(np.count_nonzero(below_fat)),
        "below_fat_E_rest_area_fraction": float(area[below_fat].sum() / area.sum()),
        "area_weighted_mean_MPa": float(np.average(young, weights=area)),
    }
    for field in ("saturated_cells", "below_fat_E_cells"):
        if observed[field] != expected[field]:
            raise ValueError(
                f"{name} {field} changed: {observed[field]} != {expected[field]}"
            )
    for field in (
        "saturated_rest_area_fraction",
        "below_fat_E_rest_area_fraction",
        "area_weighted_mean_MPa",
    ):
        _require_close(
            float(observed[field]),
            float(expected[field]),
            name=f"{name} {field}",
        )
    _require_close(float(young.min()), SKIN_E_MIN_MPA, name=f"{name} minimum E")
    _require_close(float(young.max()), SKIN_E_MAX_MPA, name=f"{name} maximum E")
    return young, {
        "gain": gain,
        "formula": (
            f"s = clip({gain:g} * ExpansionWeight, 0, 1); "
            "E = 0.2 * exp(log(0.0003 / 0.2) * s) MPa"
        ),
        "baseline_MPa": SKIN_E_MAX_MPA,
        "floor_MPa": SKIN_E_MIN_MPA,
        "fat_E_MPa_for_parameter_comparison": FAT_E_MPA,
        "nu": SKIN_NU,
        "thickness_m": SKIN_THICKNESS_M,
        "lame_conversion": ("plane stress: lambda = E*nu/(1-nu^2); mu = E/(2*(1+nu))"),
        **observed,
        "young_MPa": _weighted_stats(young, area),
        "lambda_plane_stress_MPa": _weighted_stats(lambda_, area),
        "mu_MPa": _weighted_stats(mu, area),
        "interior_edge_jump_MPa": _jump_stats(young, pairs),
        **hashes,
    }


def _skin_fields(
    corrected_skin: pv.PolyData,
    driver_skin: pv.PolyData,
    mapped: np.ndarray,
) -> tuple[pv.PolyData, dict[str, Any]]:
    area = _require_cell_array(corrected_skin, "RestArea")
    if np.any(area <= 0.0):
        raise ValueError("corrected RestArea must be strictly positive")
    _require_raw_hash("RestArea", area, EXPECTED_RAW_HASHES["rest_area"])
    _require_close(float(area.sum()), EXPECTED_SKIN_AREA_M2, name="skin area")
    expansion = _require_cell_array(driver_skin, "ExpansionWeight")[mapped]
    _require_raw_hash(
        "mapped ExpansionWeight", expansion, EXPECTED_RAW_HASHES["expansion_weight"]
    )
    if np.any((expansion < 0.0) | (expansion > 1.0)):
        raise ValueError("mapped ExpansionWeight escapes [0, 1]")
    if int(np.count_nonzero(expansion > 0.0)) != EXPECTED_EXPANSION_CELLS:
        raise ValueError("mapped expansion support changed")
    triangles = _triangles(corrected_skin, name="corrected skin")
    pairs, topology = _interior_cell_pairs(triangles)
    rendered = corrected_skin.copy(deep=True)
    gain_stats: dict[str, Any] = {}
    for gain in GAINS:
        name = f"gain{int(gain)}"
        young, stats = _gain_fields(
            gain=gain,
            expansion=expansion,
            area=area,
            pairs=pairs,
        )
        rendered.cell_data[f"SkinYoungModulusMPa{name.title()}"] = young
        gain_stats[name] = stats
    return rendered, {
        "topology": {
            "points": int(corrected_skin.n_points),
            "triangles": int(corrected_skin.n_cells),
            "rest_area_m2": float(area.sum()),
            **topology,
        },
        "source_expansion_weight": {
            **_weighted_stats(expansion, area),
            "positive_cells": int(np.count_nonzero(expansion > 0.0)),
            "positive_rest_area_fraction": float(
                area[expansion > 0.0].sum() / area.sum()
            ),
            "sha256_le_f8": EXPECTED_RAW_HASHES["expansion_weight"],
        },
        "cases": gain_stats,
        "comparison": {
            "gain4_minus_gain8_area_weighted_mean_E_MPa": (
                gain_stats["gain4"]["area_weighted_mean_MPa"]
                - gain_stats["gain8"]["area_weighted_mean_MPa"]
            ),
            "gain4_minus_gain8_saturated_rest_area_fraction": (
                gain_stats["gain4"]["saturated_rest_area_fraction"]
                - gain_stats["gain8"]["saturated_rest_area_fraction"]
            ),
            "gain4_minus_gain8_below_fat_E_rest_area_fraction": (
                gain_stats["gain4"]["below_fat_E_rest_area_fraction"]
                - gain_stats["gain8"]["below_fat_E_rest_area_fraction"]
            ),
        },
    }


def _bounds_camera(
    points: np.ndarray,
    *,
    aspect: float = 1.35,
    padding: float = 1.12,
) -> tuple[np.ndarray, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    focus = 0.5 * (minimum + maximum)
    extent = maximum - minimum
    scale = 0.5 * max(float(extent[1]), float(extent[0]) / aspect)
    return focus, padding * scale


def _field_data_strings(mesh: pv.DataSet, name: str) -> tuple[str, ...]:
    if name not in mesh.field_data:
        raise KeyError(f"mesh is missing field-data array {name}")
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(mesh.field_data[name]).reshape(-1)
    )


def _mouth_camera(skin: pv.PolyData) -> tuple[np.ndarray, float]:
    names = _field_data_strings(skin, "GroupName")
    if "GroupId" not in skin.point_data:
        raise KeyError("corrected skin is missing point GroupId")
    group_ids = np.asarray(skin.point_data["GroupId"], dtype=np.int64)
    lip_ids = [index for index, name in enumerate(names) if name.startswith("Lip")]
    lip = np.isin(group_ids, lip_ids)
    if not np.any(lip):
        raise ValueError("corrected skin contains no lip-group points")
    return _bounds_camera(np.asarray(skin.points)[lip], padding=1.25)


def _write_comparison_sheet(
    path: Path,
    *,
    skin: pv.PolyData,
) -> dict[str, Any]:
    face_focus, face_scale = _bounds_camera(np.asarray(skin.points))
    mouth_focus, mouth_scale = _mouth_camera(skin)
    front = np.asarray((0.0, 0.0, 1.0))
    views = (
        ("front", front, face_focus, face_scale),
        (
            "30 degree",
            np.asarray(
                (math.sin(math.radians(30.0)), 0.0, math.cos(math.radians(30.0)))
            ),
            face_focus,
            face_scale,
        ),
        ("mouth", front, mouth_focus, mouth_scale),
    )
    rows = (
        ("gain-4 (primary candidate)", "SkinYoungModulusMPaGain4"),
        ("gain-8 (extreme reference)", "SkinYoungModulusMPaGain8"),
    )
    window_size = (2200, 1450)
    plotter = pv.Plotter(
        shape=(len(rows), len(views)),
        off_screen=True,
        window_size=window_size,
        lighting="light kit",
        border=False,
    )
    temporary = _temporary_png(path)
    image: np.ndarray | None = None
    try:
        plotter.set_background("white")
        for row, (case_label, scalar) in enumerate(rows):
            for column, (view_label, direction, focus, scale) in enumerate(views):
                plotter.subplot(row, column)
                plotter.add_mesh(
                    skin,
                    scalars=scalar,
                    preference="cell",
                    cmap="viridis",
                    clim=(0.0, SKIN_E_MAX_MPA),
                    show_edges=False,
                    smooth_shading=False,
                    show_scalar_bar=column == len(views) - 1,
                    scalar_bar_args={
                        "title": "skin Young's modulus E [MPa]\nlinear: 0 to 0.2",
                        "title_font_size": 10,
                        "label_font_size": 9,
                        "n_labels": 5,
                        "fmt": "%.2f",
                    },
                )
                plotter.add_text(
                    f"{case_label} | {view_label}",
                    position="upper_left",
                    font_size=10,
                    color="black",
                )
                plotter.enable_parallel_projection()
                camera_focus = np.asarray(focus, dtype=np.float64)
                plotter.camera.position = tuple(camera_focus + 0.30 * direction)
                plotter.camera.focal_point = tuple(camera_focus)
                plotter.camera.up = (0.0, 1.0, 0.0)
                plotter.camera.parallel_scale = float(scale)
        image = plotter.screenshot(temporary, return_img=True)
    finally:
        plotter.close()
    if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("gain comparison screenshot returned an invalid image")
    if temporary.stat().st_size <= 10_000:
        raise ValueError("gain comparison screenshot is unexpectedly small")
    temporary.replace(path)
    return {
        "layout": "2 rows (gain-4, gain-8) x 3 columns (front, 30 degree, mouth)",
        "rows": [label for label, _ in rows],
        "views": [name for name, *_ in views],
        "scalar": "SkinYoungModulusMPa",
        "scale": "linear",
        "clim_MPa": [0.0, SKIN_E_MAX_MPA],
        "colormap": "viridis",
        "window_size": list(window_size),
        "image_shape": list(image.shape),
        "face_focus": face_focus.tolist(),
        "face_parallel_scale": float(face_scale),
        "mouth_focus": mouth_focus.tolist(),
        "mouth_parallel_scale": float(mouth_scale),
    }


def _validate_summary(summary: dict[str, Any]) -> None:
    required = {
        "complete",
        "design",
        "execution_contract",
        "input_provenance",
        "mapping",
        "output_contract",
        "render",
        "schema_version",
        "skin",
    }
    if set(summary) != required:
        raise ValueError(
            f"summary top-level schema mismatch: {sorted(set(summary) ^ required)}"
        )
    if summary["schema_version"] != SCHEMA_VERSION or summary["design"] != DESIGN:
        raise ValueError("summary identity fields changed")
    if summary["complete"] is not True:
        raise ValueError("summary is not complete")
    expected_execution = {
        "visualization_only": True,
        "forward_started": False,
        "adjoint_started": False,
        "inverse_started": False,
    }
    if summary["execution_contract"] != expected_execution:
        raise ValueError("execution contract changed")
    render = summary["render"]
    if render["scale"] != "linear" or render["clim_MPa"] != [0.0, 0.2]:
        raise ValueError("primary render is not the required shared linear E scale")


def main(cfg: Config) -> None:
    _validate_config(cfg)
    input_provenance = {
        "corrected_skin": {
            "path": str(cfg.input_corrected_skin.resolve()),
            **_require_file_identity(
                cfg.input_corrected_skin,
                expected_size=CORRECTED_SKIN_SIZE_BYTES,
                expected_sha256=CORRECTED_SKIN_SHA256,
                name="corrected IsFace skin",
            ),
        },
        "driver_skin": {
            "path": str(cfg.input_driver_skin.resolve()),
            "allowed_arrays": ["ExpansionWeight"],
            "prohibited_reuse": [
                "lambda",
                "mu",
                "SkinYoungModulusMPa",
                "ActivationInv",
                "RestArea",
            ],
            **_require_file_identity(
                cfg.input_driver_skin,
                expected_size=DRIVER_SKIN_SIZE_BYTES,
                expected_sha256=DRIVER_SKIN_SHA256,
                name="old full-boundary driver skin",
            ),
        },
    }
    corrected_skin = pv.read(cfg.input_corrected_skin)
    driver_skin = pv.read(cfg.input_driver_skin)
    if not isinstance(corrected_skin, pv.PolyData):
        raise TypeError("corrected skin is not PolyData")
    if not isinstance(driver_skin, pv.PolyData):
        raise TypeError("driver skin is not PolyData")
    if (
        corrected_skin.n_points != EXPECTED_SKIN_POINTS
        or corrected_skin.n_cells != EXPECTED_SKIN_TRIANGLES
    ):
        raise ValueError("corrected skin topology count changed")

    mapped, mapping = _map_driver_cells(corrected_skin, driver_skin)
    rendered_skin, skin_stats = _skin_fields(corrected_skin, driver_skin, mapped)
    render = _write_comparison_sheet(cfg.output_png, skin=rendered_skin)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "execution_contract": {
            "visualization_only": True,
            "forward_started": False,
            "adjoint_started": False,
            "inverse_started": False,
        },
        "input_provenance": input_provenance,
        "mapping": mapping,
        "skin": skin_stats,
        "render": render,
        "output_contract": {
            "json": {"path": str(cfg.output_json.resolve())},
            "png": {
                "path": str(cfg.output_png.resolve()),
                **_file_identity(cfg.output_png),
            },
            "overwrite_policy": "refuse existing live or temporary outputs",
        },
    }
    _validate_summary(summary)
    cfg.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cherries.set_step(0)
    cherries.log_metrics(
        {
            "gain4/E_area_mean_MPa": skin_stats["cases"]["gain4"][
                "area_weighted_mean_MPa"
            ],
            "gain4/E_below_fat_area_fraction": skin_stats["cases"]["gain4"][
                "below_fat_E_rest_area_fraction"
            ],
            "gain8/E_area_mean_MPa": skin_stats["cases"]["gain8"][
                "area_weighted_mean_MPa"
            ],
            "gain8/E_below_fat_area_fraction": skin_stats["cases"]["gain8"][
                "below_fat_E_rest_area_fraction"
            ],
            "execution/forward_started": 0,
            "execution/inverse_started": 0,
        }
    )
    logger.info("Wrote %s", cfg.output_png)
    logger.info("Wrote %s", cfg.output_json)


if __name__ == "__main__":
    cherries.main(main)
