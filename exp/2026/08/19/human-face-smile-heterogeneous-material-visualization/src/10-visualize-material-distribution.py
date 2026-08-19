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
DESIGN = "corrected-isface-heterogeneous-material-visualization"
GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]

PREPARED_MESH = (
    REPO_ROOT / "exp/2026/06/17/human-face-smile-prestrain-v2/data/"
    "10-human-face-prepared.vtu"
)
CORRECTED_SKIN = (
    REPO_ROOT / "exp/2026/08/18/human-face-smile-plane-stress-skin/data/"
    "10-corrected-baseline/skin-isface-e0200-p000.vtp"
)
OLD_DRIVER_SKIN = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/data/"
    "10-material-candidates/skin-e100-p000.vtp"
)

PREPARED_MESH_SIZE_BYTES = 76_792_914
PREPARED_MESH_SHA256 = (
    "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563"
)
CORRECTED_SKIN_SIZE_BYTES = 1_138_550
CORRECTED_SKIN_SHA256 = (
    "4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f"
)
OLD_DRIVER_SKIN_SIZE_BYTES = 38_742_137
OLD_DRIVER_SKIN_SHA256 = (
    "ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f"
)

EXPECTED_MESH_POINTS = 228_660
EXPECTED_MESH_TETS = 1_146_517
EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_SKIN_AREA_M2 = 0.04287998059707303
EXPECTED_SKIN_BOUNDARY_EDGES = 707
EXPECTED_SKIN_INTERIOR_EDGES = 44_495
EXPECTED_EXPANSION_CELLS = 16_770
EXPECTED_CONTRACTION_CELLS = 13_129
EXPECTED_SATURATED_CELLS = 3_643
EXPECTED_SATURATED_AREA_FRACTION = 0.0873748559185055
EXPECTED_BELOW_FAT_AREA_FRACTION = 0.18095636594650907
EXPECTED_AREA_MEAN_E_MPA = 0.13107545288953745

EXPECTED_POSITIVE_FRACTION_CELLS = {
    "FatFraction": 1_042_175,
    "MuscleFraction": 288_235,
    "AponeurosisFraction": 125_174,
    "SMASFraction": 225_347,
}

EXPECTED_RAW_HASHES = {
    "corrected_triangle_keys": (
        "dca8d77662f49b54250657424cfc29a3b438437081f83d65f7e108f312da2310"
    ),
    "mapped_driver_cell_indices": (
        "13458107ceef23ecc144340101574f3fb4f2e157f90a9251434e7b09a86a66c3"
    ),
    "rest_area": ("5a7b8eb9861fa509212afd610c60183f894b80db8ded53d22f3f9045bc6889de"),
    "expansion_weight": (
        "68de7dd3189c99100e4648845acf67d37e0d8a3a43d0eec3425df4168cd35910"
    ),
    "contraction_severity": (
        "a8bbf8d3626a506a5508eafb9c4824e1d694b87c0af5a73523c504f52d80ed41"
    ),
    "softening_exponent": (
        "2a41c846db9859de82332fd09322c7449807d12ae93435428f98534088a1b356"
    ),
    "young_modulus": (
        "d42fa66ea54c47890e184fd8b670a7e80cd5621e337ee81b03f658a8b9821a44"
    ),
    "lambda_plane_stress": (
        "47737694649e4753be1bee9c5b3fa86b7c17265ea3e8d85583befedd487ec3d5"
    ),
    "mu": "99d925f0b0e0f749532a440028f6c687a4f66bf90cb40551e58edea2a2dfcfd0",
    "p200_activation_diag": (
        "649b66d63fe32405cbd0a46c51873ec6fae67693020c517c94948eef91d53a93"
    ),
    "p200_activation_inv": (
        "b6985bf92655086c18a8a832849c5e35088fe648a3eda18188087a347b3e078d"
    ),
    "p200_natural_area_ratio": (
        "d7cd2a58c9b9d3b29b645ca263f42c58fc4f2fc3fde2a6f77014f847ab6787be"
    ),
}

SKIN_E_MPA = 0.2
SKIN_E_FLOOR_MPA = 3.0e-4
FAT_E_MPA = 0.003
SKIN_NU = 0.49
SKIN_THICKNESS_M = 0.001
EXPANSION_WEIGHT_GAIN = 8.0
PRESTRAIN_GAIN = 2.0

OUTPUT_JSON = GROUP_DIR / "data/10-material-distribution-stats.json"
OUTPUT_SKIN_PNG = GROUP_DIR / "data/10-skin-material-fields.png"
OUTPUT_VOLUME_PNG = GROUP_DIR / "data/10-volume-fraction-cross-sections.png"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_corrected_skin: Path = cherries.input(CORRECTED_SKIN)
    input_driver_skin: Path = cherries.input(OLD_DRIVER_SKIN)
    output_json: Path = cherries.output(
        "10-material-distribution-stats.json", mkdir=True
    )
    output_skin_png: Path = cherries.output("10-skin-material-fields.png", mkdir=True)
    output_volume_png: Path = cherries.output(
        "10-volume-fraction-cross-sections.png", mkdir=True
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
        (cfg.input_mesh, PREPARED_MESH, "input_mesh"),
        (cfg.input_corrected_skin, CORRECTED_SKIN, "input_corrected_skin"),
        (cfg.input_driver_skin, OLD_DRIVER_SKIN, "input_driver_skin"),
        (cfg.output_json, OUTPUT_JSON, "output_json"),
        (cfg.output_skin_png, OUTPUT_SKIN_PNG, "output_skin_png"),
        (cfg.output_volume_png, OUTPUT_VOLUME_PNG, "output_volume_png"),
    ):
        _require_path(actual, expected, name=name)
    paths = (
        cfg.output_json,
        cfg.output_skin_png,
        cfg.output_volume_png,
        _temporary_png(cfg.output_skin_png),
        _temporary_png(cfg.output_volume_png),
    )
    stale = [str(path) for path in paths if path.exists()]
    if stale:
        raise FileExistsError(
            f"refusing to overwrite visualization outputs or temporary files: {stale}"
        )


def _raw_sha256(array: np.ndarray, *, dtype: str) -> str:
    values = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _require_hash(name: str, array: np.ndarray, *, dtype: str = "<f8") -> str:
    actual = _raw_sha256(array, dtype=dtype)
    expected = EXPECTED_RAW_HASHES[name]
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


def _global_point_ids(mesh: pv.DataSet, *, name: str) -> np.ndarray:
    if "GlobalPointId" not in mesh.point_data:
        raise KeyError(f"{name} is missing point GlobalPointId")
    raw = np.asarray(mesh.point_data["GlobalPointId"])
    ids = np.asarray(raw, dtype=np.int64)
    if raw.shape != (mesh.n_points,) or not np.array_equal(raw, ids):
        raise ValueError(f"{name} GlobalPointId is not an exact integer vector")
    if np.unique(ids).size != ids.size:
        raise ValueError(f"{name} GlobalPointId is not unique")
    return ids


def _canonical_triangle_keys(
    triangles: np.ndarray, global_ids: np.ndarray
) -> np.ndarray:
    return np.sort(global_ids[triangles], axis=1)


def _map_driver_cells(
    corrected_skin: pv.PolyData,
    driver_skin: pv.PolyData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    corrected_triangles = _triangles(corrected_skin, name="corrected skin")
    driver_triangles = _triangles(driver_skin, name="old driver skin")
    corrected_ids = _global_point_ids(corrected_skin, name="corrected skin")
    driver_ids = _global_point_ids(driver_skin, name="old driver skin")
    corrected_keys = _canonical_triangle_keys(corrected_triangles, corrected_ids)
    driver_keys = _canonical_triangle_keys(driver_triangles, driver_ids)

    lookup: dict[tuple[int, int, int], int] = {}
    for cell_id, key_array in enumerate(driver_keys):
        key = tuple(int(value) for value in key_array)
        if key in lookup:
            raise ValueError(f"old driver skin has duplicate triangle key {key}")
        lookup[key] = cell_id
    mapped_list: list[int] = []
    for key_array in corrected_keys:
        key = tuple(int(value) for value in key_array)
        try:
            mapped_list.append(lookup[key])
        except KeyError as error:
            raise ValueError(
                f"corrected triangle key is missing from old driver: {key}"
            ) from error
    mapped = np.asarray(mapped_list, dtype=np.int64)
    if np.unique(mapped).size != corrected_skin.n_cells:
        raise ValueError("corrected-to-driver triangle mapping is not injective")
    if not np.array_equal(driver_keys[mapped], corrected_keys):
        raise ValueError("corrected-to-driver triangle key readback failed")

    corrected_key_hash = _require_hash(
        "corrected_triangle_keys", corrected_keys, dtype="<i8"
    )
    mapped_hash = _require_hash("mapped_driver_cell_indices", mapped, dtype="<i8")
    return (
        corrected_triangles,
        corrected_ids,
        mapped,
        {
            "method": "sorted GlobalPointId triangle keys",
            "driver_triangles": int(driver_skin.n_cells),
            "corrected_triangles": int(corrected_skin.n_cells),
            "mapped_unique_driver_triangles": int(np.unique(mapped).size),
            "corrected_triangle_keys_sha256_le_i8": corrected_key_hash,
            "mapped_driver_cell_indices_sha256_le_i8": mapped_hash,
            "exact_readback": True,
        },
    )


def _require_cell_array(
    mesh: pv.DataSet,
    name: str,
    *,
    finite: bool = True,
) -> np.ndarray:
    if name not in mesh.cell_data:
        raise KeyError(f"mesh is missing cell array {name}")
    values = np.asarray(mesh.cell_data[name], dtype=np.float64)
    if values.shape != (mesh.n_cells,):
        raise ValueError(f"cell array {name} has shape {values.shape}")
    if finite and not np.isfinite(values).all():
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
    mean = float(np.average(values, weights=weights))
    rms = float(np.sqrt(np.average(np.square(values), weights=weights)))
    quantiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "area_weighted_mean": mean,
        "area_weighted_rms": rms,
        "area_weighted_quantiles": {
            f"q{int(100 * quantile):02d}": value
            for quantile, value in zip(
                quantiles,
                _weighted_quantiles(values, weights, quantiles),
                strict=True,
            )
        },
    }


def _edge_pairs(triangles: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    owners: dict[tuple[int, int], int] = {}
    pairs: list[tuple[int, int]] = []
    boundary = 0
    counts: dict[tuple[int, int], int] = {}
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
    if nonmanifold:
        raise ValueError(f"corrected skin has {nonmanifold} nonmanifold edges")
    boundary = sum(count == 1 for count in counts.values())
    result = np.asarray(pairs, dtype=np.int64)
    if result.shape != (EXPECTED_SKIN_INTERIOR_EDGES, 2):
        raise ValueError(f"corrected skin interior-edge shape changed: {result.shape}")
    if boundary != EXPECTED_SKIN_BOUNDARY_EDGES:
        raise ValueError(f"corrected skin boundary-edge count changed: {boundary}")
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


def _validate_point_alignment(
    corrected_skin: pv.PolyData,
    corrected_ids: np.ndarray,
    driver_skin: pv.PolyData,
    mesh: pv.UnstructuredGrid,
) -> dict[str, float | str]:
    corrected_points = np.asarray(corrected_skin.points, dtype=np.float64)
    driver_ids = _global_point_ids(driver_skin, name="old driver skin")
    # The prepared VTU predates the explicit GlobalPointId array.  The runtime
    # ModelBuilder assigns its canonical global IDs as arange(n_points), which
    # is also the ID convention used by both pinned skin files.
    if "GlobalPointId" in mesh.point_data:
        mesh_ids = _global_point_ids(mesh, name="prepared mesh")
        mesh_id_semantics = "point GlobalPointId array"
    else:
        mesh_ids = np.arange(mesh.n_points, dtype=np.int64)
        mesh_id_semantics = "canonical local point index fallback"
    driver_lookup = {int(gid): index for index, gid in enumerate(driver_ids)}
    mesh_lookup = {int(gid): index for index, gid in enumerate(mesh_ids)}
    try:
        driver_indices = np.asarray(
            [driver_lookup[int(gid)] for gid in corrected_ids], dtype=np.int64
        )
        mesh_indices = np.asarray(
            [mesh_lookup[int(gid)] for gid in corrected_ids], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(
            "a corrected skin GlobalPointId is missing from a pinned input"
        ) from error
    driver_delta = corrected_points - np.asarray(driver_skin.points)[driver_indices]
    mesh_delta = corrected_points - np.asarray(mesh.points)[mesh_indices]
    driver_max = float(np.max(np.abs(driver_delta)))
    mesh_max = float(np.max(np.abs(mesh_delta)))
    if driver_max != 0.0 or mesh_max != 0.0:
        raise ValueError(
            "pinned inputs disagree on corrected skin point coordinates: "
            f"driver={driver_max}, mesh={mesh_max}"
        )
    return {
        "corrected_to_driver_max_abs_m": driver_max,
        "corrected_to_mesh_max_abs_m": mesh_max,
        "prepared_mesh_point_id_semantics": mesh_id_semantics,
    }


def _skin_fields(  # noqa: PLR0915
    corrected_skin: pv.PolyData,
    driver_skin: pv.PolyData,
    mapped: np.ndarray,
    triangles: np.ndarray,
) -> tuple[pv.PolyData, dict[str, Any]]:
    area = _require_cell_array(corrected_skin, "RestArea")
    if np.any(area <= 0.0):
        raise ValueError("corrected RestArea must be strictly positive")
    _require_hash("rest_area", area)
    _require_close(float(area.sum()), EXPECTED_SKIN_AREA_M2, name="skin area")

    expansion = _require_cell_array(driver_skin, "ExpansionWeight")[mapped]
    contraction = _require_cell_array(driver_skin, "ContractionSeverityLogCapped")[
        mapped
    ]
    _require_hash("expansion_weight", expansion)
    _require_hash("contraction_severity", contraction)
    if np.any((expansion < 0.0) | (expansion > 1.0)):
        raise ValueError("mapped ExpansionWeight escapes [0, 1]")
    if np.any(contraction < 0.0):
        raise ValueError("mapped contraction severity is negative")
    expansion_mask = expansion > 0.0
    contraction_mask = contraction > 0.0
    if np.any(expansion_mask & contraction_mask):
        raise ValueError("expansion and contraction driver supports overlap")
    if not np.all(expansion_mask | contraction_mask):
        raise ValueError("expansion and contraction drivers do not partition IsFace")
    if int(expansion_mask.sum()) != EXPECTED_EXPANSION_CELLS:
        raise ValueError("mapped expansion cell count changed")
    if int(contraction_mask.sum()) != EXPECTED_CONTRACTION_CELLS:
        raise ValueError("mapped contraction cell count changed")

    exponent = np.clip(EXPANSION_WEIGHT_GAIN * expansion, 0.0, 1.0)
    young = SKIN_E_MPA * np.exp(math.log(SKIN_E_FLOOR_MPA / SKIN_E_MPA) * exponent)
    lambda_ = young * SKIN_NU / (1.0 - SKIN_NU**2)
    mu = young / (2.0 * (1.0 + SKIN_NU))
    activation_diag = np.exp(contraction) - 1.0
    activation_inv = np.zeros((corrected_skin.n_cells, 3), dtype=np.float64)
    activation_inv[:, 0] = activation_diag
    activation_inv[:, 1] = activation_diag
    natural_area_ratio = np.reciprocal(np.square(1.0 + activation_diag))
    for name, array in (
        ("softening_exponent", exponent),
        ("young_modulus", young),
        ("lambda_plane_stress", lambda_),
        ("mu", mu),
        ("p200_activation_diag", activation_diag),
        ("p200_activation_inv", activation_inv),
        ("p200_natural_area_ratio", natural_area_ratio),
    ):
        _require_hash(name, array)
    saturated = exponent == 1.0
    if int(saturated.sum()) != EXPECTED_SATURATED_CELLS:
        raise ValueError("gain8 saturated-triangle count changed")
    saturated_area_fraction = float(area[saturated].sum() / area.sum())
    below_fat_area_fraction = float(area[young < FAT_E_MPA].sum() / area.sum())
    area_mean_e = float(np.average(young, weights=area))
    _require_close(
        saturated_area_fraction,
        EXPECTED_SATURATED_AREA_FRACTION,
        name="gain8 saturated area fraction",
    )
    _require_close(
        below_fat_area_fraction,
        EXPECTED_BELOW_FAT_AREA_FRACTION,
        name="area fraction below fat E",
    )
    _require_close(area_mean_e, EXPECTED_AREA_MEAN_E_MPA, name="area-mean E")
    _require_close(float(young.min()), SKIN_E_FLOOR_MPA, name="minimum E")
    _require_close(float(young.max()), SKIN_E_MPA, name="maximum E")

    pairs, topology = _edge_pairs(triangles)
    rendered = corrected_skin.copy(deep=True)
    rendered.cell_data["ExpansionWeight"] = expansion
    rendered.cell_data["SofteningExponentGain8"] = exponent
    rendered.cell_data["SkinYoungModulusMPa"] = young
    rendered.cell_data["Log10SkinYoungModulusMPa"] = np.log10(young)
    rendered.cell_data["ContractionSeverityLogCapped"] = contraction
    rendered.cell_data["P200ActivationInvDiag"] = activation_diag
    rendered.cell_data["P200NaturalAreaRatio"] = natural_area_ratio

    stats = {
        "topology": {
            "points": int(corrected_skin.n_points),
            "triangles": int(corrected_skin.n_cells),
            "rest_area_m2": float(area.sum()),
            **topology,
        },
        "domain": {
            "expansion_cells": int(expansion_mask.sum()),
            "expansion_rest_area_fraction": float(
                area[expansion_mask].sum() / area.sum()
            ),
            "contraction_cells": int(contraction_mask.sum()),
            "contraction_rest_area_fraction": float(
                area[contraction_mask].sum() / area.sum()
            ),
            "supports_overlap_cells": int(
                np.count_nonzero(expansion_mask & contraction_mask)
            ),
            "supports_partition_all_triangles": True,
        },
        "source_fields": {
            "ExpansionWeight": {
                **_weighted_stats(expansion, area),
                "sha256_le_f8": EXPECTED_RAW_HASHES["expansion_weight"],
            },
            "ContractionSeverityLogCapped": {
                **_weighted_stats(contraction, area),
                "sha256_le_f8": EXPECTED_RAW_HASHES["contraction_severity"],
            },
        },
        "planned_young_modulus": {
            "formula": (
                "s = clip(8 * ExpansionWeight, 0, 1); "
                "E = 0.2 * exp(log(0.0003 / 0.2) * s) MPa"
            ),
            "baseline_MPa": SKIN_E_MPA,
            "floor_MPa": SKIN_E_FLOOR_MPA,
            "fat_E_MPa_for_parameter_comparison": FAT_E_MPA,
            "gain": EXPANSION_WEIGHT_GAIN,
            "nu": SKIN_NU,
            "thickness_m": SKIN_THICKNESS_M,
            "lame_conversion": (
                "plane stress: lambda = E*nu/(1-nu^2); mu = E/(2*(1+nu))"
            ),
            "saturated_cells": int(saturated.sum()),
            "saturated_rest_area_fraction": saturated_area_fraction,
            "below_fat_E_cells": int(np.count_nonzero(young < FAT_E_MPA)),
            "below_fat_E_rest_area_fraction": below_fat_area_fraction,
            "young": _weighted_stats(young, area),
            "lambda": _weighted_stats(lambda_, area),
            "mu": _weighted_stats(mu, area),
            "interior_edge_jump_MPa": _jump_stats(young, pairs),
            "softening_exponent_sha256_le_f8": EXPECTED_RAW_HASHES[
                "softening_exponent"
            ],
            "young_sha256_le_f8": EXPECTED_RAW_HASHES["young_modulus"],
            "lambda_sha256_le_f8": EXPECTED_RAW_HASHES["lambda_plane_stress"],
            "mu_sha256_le_f8": EXPECTED_RAW_HASHES["mu"],
        },
        "planned_p200": {
            "formula": (
                "a = exp(ContractionSeverityLogCapped) - 1; "
                "ActivationInv = [a, a, 0]; natural area ratio = 1/(1+a)^2"
            ),
            "prestrain_gain": PRESTRAIN_GAIN,
            "energy_weight": "fixed original RestArea; independent of prestrain",
            "activation_diag": _weighted_stats(activation_diag, area),
            "natural_area_ratio": _weighted_stats(natural_area_ratio, area),
            "interior_edge_activation_jump": _jump_stats(activation_diag, pairs),
            "activation_diag_sha256_le_f8": EXPECTED_RAW_HASHES["p200_activation_diag"],
            "activation_inv_sha256_le_f8": EXPECTED_RAW_HASHES["p200_activation_inv"],
            "natural_area_ratio_sha256_le_f8": EXPECTED_RAW_HASHES[
                "p200_natural_area_ratio"
            ],
        },
    }
    return rendered, stats


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


def _mouth_camera(
    skin: pv.PolyData,
    mesh: pv.UnstructuredGrid,
) -> tuple[np.ndarray, float]:
    names = _field_data_strings(mesh, "GroupName")
    if "GroupId" not in skin.point_data:
        raise KeyError("corrected skin is missing point GroupId")
    group_ids = np.asarray(skin.point_data["GroupId"], dtype=np.int64)
    lip_ids = [index for index, name in enumerate(names) if name.startswith("Lip")]
    lip = np.isin(group_ids, lip_ids)
    if not np.any(lip):
        raise ValueError("corrected skin contains no lip-group points")
    return _bounds_camera(np.asarray(skin.points)[lip], padding=1.25)


def _write_skin_sheet(
    path: Path,
    *,
    skin: pv.PolyData,
    mesh: pv.UnstructuredGrid,
) -> dict[str, Any]:
    face_focus, face_scale = _bounds_camera(np.asarray(skin.points))
    mouth_focus, mouth_scale = _mouth_camera(skin, mesh)
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
    fields = (
        (
            "ExpansionWeight",
            "source expansion weight",
            "viridis",
            (0.0, 0.7093692747170706),
        ),
        (
            "Log10SkinYoungModulusMPa",
            "planned log10 E [MPa]",
            "plasma",
            (math.log10(SKIN_E_FLOOR_MPA), math.log10(SKIN_E_MPA)),
        ),
        (
            "ContractionSeverityLogCapped",
            "source contraction severity",
            "cividis",
            (0.0, 0.41746591441203185),
        ),
        (
            "P200NaturalAreaRatio",
            "p200 natural-area ratio",
            "magma_r",
            (0.4339040601747963, 1.0),
        ),
    )
    window_size = (2400, 1650)
    plotter = pv.Plotter(
        shape=(len(views), len(fields)),
        off_screen=True,
        window_size=window_size,
        lighting="light kit",
        border=False,
    )
    temporary = _temporary_png(path)
    image: np.ndarray | None = None
    try:
        plotter.set_background("white")
        for row, (view_name, direction, focus, scale) in enumerate(views):
            for column, (scalar, title, cmap, clim) in enumerate(fields):
                plotter.subplot(row, column)
                plotter.add_mesh(
                    skin,
                    scalars=scalar,
                    preference="cell",
                    cmap=cmap,
                    clim=clim,
                    show_edges=False,
                    smooth_shading=False,
                    show_scalar_bar=row == 0,
                    scalar_bar_args={
                        "title": title,
                        "title_font_size": 10,
                        "label_font_size": 9,
                        "n_labels": 4,
                    },
                )
                plotter.add_text(
                    f"{view_name} | {title}",
                    position="upper_left",
                    font_size=9,
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
        raise ValueError("skin material screenshot returned an invalid image")
    if temporary.stat().st_size <= 10_000:
        raise ValueError("skin material screenshot is unexpectedly small")
    temporary.replace(path)
    return {
        "views": [name for name, *_ in views],
        "fields": [scalar for scalar, *_ in fields],
        "window_size": list(window_size),
        "image_shape": list(image.shape),
        "face_focus": face_focus.tolist(),
        "face_parallel_scale": float(face_scale),
        "mouth_focus": mouth_focus.tolist(),
        "mouth_parallel_scale": float(mouth_scale),
    }


def _volume_fraction_stats(mesh: pv.UnstructuredGrid) -> dict[str, Any]:
    if mesh.n_points != EXPECTED_MESH_POINTS or mesh.n_cells != EXPECTED_MESH_TETS:
        raise ValueError(
            "prepared volume topology changed: "
            f"{mesh.n_points} points, {mesh.n_cells} cells"
        )
    volume = _require_cell_array(mesh, "Volume")
    if np.any(volume <= 0.0):
        raise ValueError("prepared Volume must be strictly positive")
    fields = {
        name: _require_cell_array(mesh, name)
        for name in EXPECTED_POSITIVE_FRACTION_CELLS
    }
    for name, values in fields.items():
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} escapes [0, 1]")
        positive = int(np.count_nonzero(values > 0.0))
        expected = EXPECTED_POSITIVE_FRACTION_CELLS[name]
        if positive != expected:
            raise ValueError(f"{name} positive-cell count changed: {positive}")
    active_sum = (
        fields["FatFraction"] + fields["MuscleFraction"] + fields["AponeurosisFraction"]
    )
    sum_error = float(np.max(np.abs(active_sum - 1.0)))
    if sum_error != 0.0:
        raise ValueError(
            f"active volume fractions no longer sum bit-exactly: {sum_error}"
        )
    total_volume = float(volume.sum())
    result: dict[str, Any] = {
        "points": int(mesh.n_points),
        "tetrahedra": int(mesh.n_cells),
        "total_volume_m3": total_volume,
        "active_fraction_sum_max_abs_error": sum_error,
        "active_fraction_fields": [
            "FatFraction",
            "MuscleFraction",
            "AponeurosisFraction",
        ],
        "diagnostic_only_fields": ["SMASFraction"],
        "SMAS_interpretation": (
            "source diagnostic only; current forward builder does not add a "
            "separate SMAS constitutive energy"
        ),
        "fractions": {},
    }
    for name, values in fields.items():
        result["fractions"][name] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "positive_cells": int(np.count_nonzero(values > 0.0)),
            "positive_tet_volume_fraction": float(
                volume[values > 0.0].sum() / total_volume
            ),
            "tet_volume_weighted_mean": float(np.average(values, weights=volume)),
        }
    return result


def _slice_camera(
    section: pv.PolyData,
    *,
    normal: np.ndarray,
    up: np.ndarray,
    aspect: float = 1.1,
) -> tuple[np.ndarray, float]:
    points = np.asarray(section.points, dtype=np.float64)
    focus = np.asarray(section.center, dtype=np.float64)
    right = np.cross(up, normal)
    right /= np.linalg.norm(right)
    horizontal = points @ right
    vertical = points @ up
    scale = 0.56 * max(float(np.ptp(vertical)), float(np.ptp(horizontal)) / aspect)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("volume cross-section camera scale is invalid")
    return focus, scale


def _write_volume_sheet(
    path: Path,
    *,
    mesh: pv.UnstructuredGrid,
) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(3, 2)
    center = bounds.mean(axis=1)
    planes = (
        (
            "midsagittal",
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((center[0], center[1], center[2])),
        ),
        (
            "coronal",
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((center[0], center[1], center[2])),
        ),
        (
            "axial",
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((center[0], center[1], center[2])),
        ),
    )
    fields = (
        ("FatFraction", "fat", "YlOrBr"),
        ("MuscleFraction", "muscle", "Reds"),
        ("AponeurosisFraction", "aponeurosis", "Blues"),
        ("SMASFraction", "SMAS (diagnostic only)", "Purples"),
    )
    sections: list[pv.PolyData] = []
    section_stats: list[dict[str, Any]] = []
    for name, normal, up, origin in planes:
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0 or section.n_cells == 0:
            raise ValueError(f"{name} volume cross-section is empty")
        missing = [field for field, *_ in fields if field not in section.cell_data]
        if missing:
            raise KeyError(f"{name} cross-section lost cell arrays: {missing}")
        sections.append(section)
        section_stats.append(
            {
                "name": name,
                "normal": normal.tolist(),
                "up": up.tolist(),
                "origin_m": origin.tolist(),
                "points": int(section.n_points),
                "cells": int(section.n_cells),
            }
        )

    window_size = (1950, 2050)
    plotter = pv.Plotter(
        shape=(len(fields), len(planes)),
        off_screen=True,
        window_size=window_size,
        lighting="light kit",
        border=False,
    )
    temporary = _temporary_png(path)
    image: np.ndarray | None = None
    try:
        plotter.set_background("white")
        for row, (field, label, cmap) in enumerate(fields):
            for column, ((plane, normal, up, _), section) in enumerate(
                zip(planes, sections, strict=True)
            ):
                plotter.subplot(row, column)
                plotter.add_mesh(
                    section,
                    scalars=field,
                    preference="cell",
                    cmap=cmap,
                    clim=(0.0, 1.0),
                    show_edges=False,
                    smooth_shading=False,
                    show_scalar_bar=column == len(planes) - 1,
                    scalar_bar_args={
                        "title": f"{label} fraction",
                        "title_font_size": 10,
                        "label_font_size": 9,
                        "n_labels": 5,
                    },
                )
                plotter.add_text(
                    f"{plane} | {label} fraction",
                    position="upper_left",
                    font_size=9,
                    color="black",
                )
                focus, scale = _slice_camera(section, normal=normal, up=up)
                plotter.enable_parallel_projection()
                plotter.camera.position = tuple(focus + 0.35 * normal)
                plotter.camera.focal_point = tuple(focus)
                plotter.camera.up = tuple(up)
                plotter.camera.parallel_scale = scale
        image = plotter.screenshot(temporary, return_img=True)
    finally:
        plotter.close()
    if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("volume-fraction screenshot returned an invalid image")
    if temporary.stat().st_size <= 10_000:
        raise ValueError("volume-fraction screenshot is unexpectedly small")
    temporary.replace(path)
    return {
        "planes": section_stats,
        "fields": [field for field, *_ in fields],
        "window_size": list(window_size),
        "image_shape": list(image.shape),
        "fraction_clim": [0.0, 1.0],
    }


def _validate_summary(summary: dict[str, Any]) -> None:
    required = {
        "complete",
        "design",
        "execution_contract",
        "input_provenance",
        "mapping",
        "output_contract",
        "renders",
        "schema_version",
        "skin",
        "volume",
    }
    if set(summary) != required:
        raise ValueError(
            f"summary top-level schema mismatch: {sorted(set(summary) ^ required)}"
        )
    if summary["schema_version"] != SCHEMA_VERSION or summary["design"] != DESIGN:
        raise ValueError("summary identity fields changed")
    if summary["complete"] is not True:
        raise ValueError("summary is not complete")
    execution = summary["execution_contract"]
    expected_execution = {
        "visualization_only": True,
        "forward_started": False,
        "adjoint_started": False,
        "inverse_started": False,
    }
    if execution != expected_execution:
        raise ValueError(f"execution contract changed: {execution}")
    outputs = summary["output_contract"]
    if outputs["skin_png"]["sha256"] == outputs["volume_png"]["sha256"]:
        raise ValueError("the two visualization PNGs are unexpectedly identical")


def main(cfg: Config) -> None:
    _validate_config(cfg)
    input_provenance = {
        "prepared_mesh": {
            "path": str(cfg.input_mesh.resolve()),
            **_require_file_identity(
                cfg.input_mesh,
                expected_size=PREPARED_MESH_SIZE_BYTES,
                expected_sha256=PREPARED_MESH_SHA256,
                name="prepared mesh",
            ),
        },
        "corrected_skin": {
            "path": str(cfg.input_corrected_skin.resolve()),
            **_require_file_identity(
                cfg.input_corrected_skin,
                expected_size=CORRECTED_SKIN_SIZE_BYTES,
                expected_sha256=CORRECTED_SKIN_SHA256,
                name="corrected IsFace skin",
            ),
        },
        "old_driver_skin": {
            "path": str(cfg.input_driver_skin.resolve()),
            "allowed_arrays": [
                "ExpansionWeight",
                "ContractionSeverityLogCapped",
            ],
            "prohibited_reuse": [
                "lambda",
                "mu",
                "SkinYoungModulusMPa",
                "ActivationInv",
                "RestArea",
            ],
            **_require_file_identity(
                cfg.input_driver_skin,
                expected_size=OLD_DRIVER_SKIN_SIZE_BYTES,
                expected_sha256=OLD_DRIVER_SKIN_SHA256,
                name="old full-boundary driver skin",
            ),
        },
    }

    mesh = pv.read(cfg.input_mesh)
    corrected_skin = pv.read(cfg.input_corrected_skin)
    driver_skin = pv.read(cfg.input_driver_skin)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError("prepared mesh is not an UnstructuredGrid")
    if not isinstance(corrected_skin, pv.PolyData):
        raise TypeError("corrected skin is not PolyData")
    if not isinstance(driver_skin, pv.PolyData):
        raise TypeError("old driver skin is not PolyData")
    if (
        corrected_skin.n_points != EXPECTED_SKIN_POINTS
        or corrected_skin.n_cells != EXPECTED_SKIN_TRIANGLES
    ):
        raise ValueError("corrected skin topology count changed")

    triangles, corrected_ids, mapped, mapping = _map_driver_cells(
        corrected_skin, driver_skin
    )
    mapping["point_alignment"] = _validate_point_alignment(
        corrected_skin, corrected_ids, driver_skin, mesh
    )
    rendered_skin, skin_stats = _skin_fields(
        corrected_skin, driver_skin, mapped, triangles
    )
    volume_stats = _volume_fraction_stats(mesh)
    skin_render = _write_skin_sheet(cfg.output_skin_png, skin=rendered_skin, mesh=mesh)
    volume_render = _write_volume_sheet(cfg.output_volume_png, mesh=mesh)
    output_contract = {
        "json": {"path": str(cfg.output_json.resolve())},
        "skin_png": {
            "path": str(cfg.output_skin_png.resolve()),
            **_file_identity(cfg.output_skin_png),
        },
        "volume_png": {
            "path": str(cfg.output_volume_png.resolve()),
            **_file_identity(cfg.output_volume_png),
        },
        "overwrite_policy": "refuse existing live or temporary outputs",
    }
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
        "volume": volume_stats,
        "renders": {
            "skin_material_fields": skin_render,
            "volume_fraction_cross_sections": volume_render,
        },
        "output_contract": output_contract,
    }
    _validate_summary(summary)
    cfg.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cherries.set_step(0)
    cherries.log_metrics(
        {
            "skin/triangles": EXPECTED_SKIN_TRIANGLES,
            "skin/E_area_mean_MPa": EXPECTED_AREA_MEAN_E_MPA,
            "skin/E_below_fat_area_fraction": EXPECTED_BELOW_FAT_AREA_FRACTION,
            "skin/prestrain_max_activation_inv": float(
                rendered_skin.cell_data["P200ActivationInvDiag"].max()
            ),
            "volume/tetrahedra": EXPECTED_MESH_TETS,
            "execution/forward_started": 0,
            "execution/inverse_started": 0,
        }
    )
    logger.info("Wrote %s", cfg.output_skin_png)
    logger.info("Wrote %s", cfg.output_volume_png)
    logger.info("Wrote %s", cfg.output_json)


if __name__ == "__main__":
    cherries.main(main)
