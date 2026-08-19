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
DESIGN = "corrected-isface-smile-target-rest-area-ratio"
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
EXPECTED_TARGET_AREA_M2 = 0.04270432833950364
EXPECTED_TOTAL_TARGET_REST_AREA_RATIO = 0.9959036301993714
EXPECTED_RATIO_MIN = 0.045415665542901884
EXPECTED_RATIO_MAX = 16.513033660865602
EXPECTED_EXPANSION_CELLS = 16_723
EXPECTED_CONTRACTION_CELLS = 13_159
EXPECTED_UNCHANGED_CELLS = 17
EXPECTED_EXPANSION_AREA_FRACTION = 0.5455308228719783
EXPECTED_CONTRACTION_AREA_FRACTION = 0.45442829010841274
EXPECTED_RAW_DISPLAY_LOW_CELLS = 140
EXPECTED_RAW_DISPLAY_HIGH_CELLS = 607
EXPECTED_RAW_DISPLAY_LOW_AREA_FRACTION = 0.003727836954896607
EXPECTED_RAW_DISPLAY_HIGH_AREA_FRACTION = 0.007750726141812753
EXPECTED_PROCESSED_EXPANSION_CELLS = 16_770
EXPECTED_PROCESSED_CONTRACTION_CELLS = 13_129
EXPECTED_PROCESSED_EXPANSION_AREA_FRACTION = 0.5511754194031411
EXPECTED_PROCESSED_CONTRACTION_AREA_FRACTION = 0.44882458059685887
EXPECTED_DIFFUSED_LOG_MIN = -0.41746591441203185
EXPECTED_DIFFUSED_LOG_MAX = 0.28019188769897707
EXPECTED_PROCESSED_RATIO_MIN = 0.6587139441174722
EXPECTED_PROCESSED_RATIO_MAX = 1.3233837290335404

EXPECTED_RAW_HASHES = {
    "corrected_triangle_keys": (
        "dca8d77662f49b54250657424cfc29a3b438437081f83d65f7e108f312da2310"
    ),
    "mapped_driver_cell_indices": (
        "13458107ceef23ecc144340101574f3fb4f2e157f90a9251434e7b09a86a66c3"
    ),
    "rest_area": "5a7b8eb9861fa509212afd610c60183f894b80db8ded53d22f3f9045bc6889de",
    "target_area": ("b50b815618e75ecd7b99619dc5a11492ea21dcde240dbd3a283030ac36dea580"),
    "target_rest_area_ratio": (
        "da98a6f48694eed30b1683ccf5a1f02fd67c87393b30003d07c52a7fa25bc606"
    ),
    "signed_area_change_percent": (
        "d47fa2fd5d977634131bef237856db1dfd1d0444899559588ac1a5514d778e47"
    ),
    "ln_target_rest_area_ratio": (
        "26329c5975df180df12293335113dd13e0ed868cf4eab5a068f5cefb8770e080"
    ),
    "log2_target_rest_area_ratio": (
        "44b846b3f48be4004d2e813afc453d4727a082cea1f38cfc507f6f27dfb2620a"
    ),
    "log_area_diffused": (
        "df8d57c95f18f63bda06a52eb4abbcd76e86eff9b259a53d6cd15d328bd566df"
    ),
    "processed_area_ratio": (
        "08f1c02973f8798bbbb3950d071e1a3b1316e3ae242899881d52fc72dc1e22b5"
    ),
    "processed_signed_area_change_percent": (
        "7e6201dbb2adab7350fd63f47781cc3b991f3b2dcabed67add96a1ec595a80d4"
    ),
}

RATIO_DISPLAY_CLIM = (0.6, 1.4)

OUTPUT_JSON = GROUP_DIR / "data/30-target-rest-area-ratio-stats.json"
OUTPUT_PNG = GROUP_DIR / "data/30-target-rest-area-ratio.png"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_corrected_skin: Path = cherries.input(CORRECTED_SKIN)
    input_driver_skin: Path = cherries.input(DRIVER_SKIN)
    output_json: Path = cherries.output(
        "30-target-rest-area-ratio-stats.json", mkdir=True
    )
    output_png: Path = cherries.output("30-target-rest-area-ratio.png", mkdir=True)


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


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _validate_config(cfg: Config) -> None:
    for actual, expected, name in (
        (cfg.input_corrected_skin, CORRECTED_SKIN, "input_corrected_skin"),
        (cfg.input_driver_skin, DRIVER_SKIN, "input_driver_skin"),
        (cfg.output_json, OUTPUT_JSON, "output_json"),
        (cfg.output_png, OUTPUT_PNG, "output_png"),
    ):
        _require_path(actual, expected, name=name)
    paths = (
        cfg.output_json,
        cfg.output_png,
        _temporary_path(cfg.output_json),
        _temporary_path(cfg.output_png),
    )
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
    quantiles = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "area_weighted_mean": float(np.average(values, weights=weights)),
        "area_weighted_rms": float(
            np.sqrt(np.average(np.square(values), weights=weights))
        ),
        "area_weighted_quantiles": {
            f"q{100 * quantile:g}": value
            for quantile, value in zip(
                quantiles,
                _weighted_quantiles(values, weights, quantiles),
                strict=True,
            )
        },
    }


def _target_area_fields(  # noqa: PLR0915
    corrected_skin: pv.PolyData,
    driver_skin: pv.PolyData,
    mapped: np.ndarray,
) -> tuple[pv.PolyData, dict[str, Any]]:
    rest_area = _require_cell_array(corrected_skin, "RestArea")
    driver_rest_area = _require_cell_array(driver_skin, "RestArea")[mapped]
    target_area = _require_cell_array(driver_skin, "TargetArea")[mapped]
    ratio = _require_cell_array(driver_skin, "TargetRestAreaRatio")[mapped]
    diffused_log = _require_cell_array(driver_skin, "LogAreaDiffused")[mapped]
    if np.any(rest_area <= 0.0) or np.any(target_area <= 0.0) or np.any(ratio <= 0.0):
        raise ValueError(
            "rest area, target area, and target/rest ratio must be positive"
        )
    _require_raw_hash("corrected RestArea", rest_area, EXPECTED_RAW_HASHES["rest_area"])
    _require_raw_hash(
        "mapped driver RestArea", driver_rest_area, EXPECTED_RAW_HASHES["rest_area"]
    )
    _require_raw_hash(
        "mapped TargetArea", target_area, EXPECTED_RAW_HASHES["target_area"]
    )
    _require_raw_hash(
        "mapped TargetRestAreaRatio",
        ratio,
        EXPECTED_RAW_HASHES["target_rest_area_ratio"],
    )
    _require_raw_hash(
        "mapped LogAreaDiffused",
        diffused_log,
        EXPECTED_RAW_HASHES["log_area_diffused"],
    )
    if not np.array_equal(rest_area, driver_rest_area):
        raise ValueError("corrected and mapped-driver RestArea differ")
    if not np.array_equal(ratio, target_area / rest_area):
        raise ValueError("TargetRestAreaRatio is not exactly TargetArea / RestArea")

    signed_percent = 100.0 * (ratio - 1.0)
    ln_ratio = np.log(ratio)
    log2_ratio = np.log2(ratio)
    processed_ratio = np.exp(diffused_log)
    processed_signed_percent = 100.0 * (processed_ratio - 1.0)
    _require_raw_hash(
        "signed area change percent",
        signed_percent,
        EXPECTED_RAW_HASHES["signed_area_change_percent"],
    )
    _require_raw_hash(
        "natural log target/rest ratio",
        ln_ratio,
        EXPECTED_RAW_HASHES["ln_target_rest_area_ratio"],
    )
    _require_raw_hash(
        "log2 target/rest ratio",
        log2_ratio,
        EXPECTED_RAW_HASHES["log2_target_rest_area_ratio"],
    )
    _require_raw_hash(
        "processed driver-equivalent area ratio",
        processed_ratio,
        EXPECTED_RAW_HASHES["processed_area_ratio"],
    )
    _require_raw_hash(
        "processed signed area change percent",
        processed_signed_percent,
        EXPECTED_RAW_HASHES["processed_signed_area_change_percent"],
    )
    if not np.array_equal(
        ln_ratio, _require_cell_array(driver_skin, "LogAreaRaw")[mapped]
    ):
        raise ValueError("mapped LogAreaRaw is not exactly log(TargetRestAreaRatio)")

    _require_close(float(rest_area.sum()), EXPECTED_SKIN_AREA_M2, name="rest area")
    _require_close(
        float(target_area.sum()), EXPECTED_TARGET_AREA_M2, name="target area"
    )
    total_ratio = float(target_area.sum() / rest_area.sum())
    _require_close(
        total_ratio,
        EXPECTED_TOTAL_TARGET_REST_AREA_RATIO,
        name="total target/rest area ratio",
    )
    _require_close(float(ratio.min()), EXPECTED_RATIO_MIN, name="minimum ratio")
    _require_close(float(ratio.max()), EXPECTED_RATIO_MAX, name="maximum ratio")
    _require_close(
        float(diffused_log.min()),
        EXPECTED_DIFFUSED_LOG_MIN,
        name="minimum diffused log area",
    )
    _require_close(
        float(diffused_log.max()),
        EXPECTED_DIFFUSED_LOG_MAX,
        name="maximum diffused log area",
    )
    _require_close(
        float(processed_ratio.min()),
        EXPECTED_PROCESSED_RATIO_MIN,
        name="minimum processed area ratio",
    )
    _require_close(
        float(processed_ratio.max()),
        EXPECTED_PROCESSED_RATIO_MAX,
        name="maximum processed area ratio",
    )

    expansion = ratio > 1.0
    contraction = ratio < 1.0
    unchanged = ratio == 1.0
    observed_counts = {
        "expansion": int(np.count_nonzero(expansion)),
        "contraction": int(np.count_nonzero(contraction)),
        "unchanged": int(np.count_nonzero(unchanged)),
    }
    expected_counts = {
        "expansion": EXPECTED_EXPANSION_CELLS,
        "contraction": EXPECTED_CONTRACTION_CELLS,
        "unchanged": EXPECTED_UNCHANGED_CELLS,
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"target-area sign counts changed: {observed_counts} != {expected_counts}"
        )
    expansion_fraction = float(rest_area[expansion].sum() / rest_area.sum())
    contraction_fraction = float(rest_area[contraction].sum() / rest_area.sum())
    _require_close(
        expansion_fraction,
        EXPECTED_EXPANSION_AREA_FRACTION,
        name="expansion rest-area fraction",
    )
    _require_close(
        contraction_fraction,
        EXPECTED_CONTRACTION_AREA_FRACTION,
        name="contraction rest-area fraction",
    )

    processed_expansion = processed_ratio > 1.0
    processed_contraction = processed_ratio < 1.0
    processed_unchanged = processed_ratio == 1.0
    processed_counts = {
        "expansion": int(np.count_nonzero(processed_expansion)),
        "contraction": int(np.count_nonzero(processed_contraction)),
        "unchanged": int(np.count_nonzero(processed_unchanged)),
    }
    expected_processed_counts = {
        "expansion": EXPECTED_PROCESSED_EXPANSION_CELLS,
        "contraction": EXPECTED_PROCESSED_CONTRACTION_CELLS,
        "unchanged": 0,
    }
    if processed_counts != expected_processed_counts:
        raise ValueError(
            "processed-area sign counts changed: "
            f"{processed_counts} != {expected_processed_counts}"
        )
    processed_expansion_fraction = float(
        rest_area[processed_expansion].sum() / rest_area.sum()
    )
    processed_contraction_fraction = float(
        rest_area[processed_contraction].sum() / rest_area.sum()
    )
    _require_close(
        processed_expansion_fraction,
        EXPECTED_PROCESSED_EXPANSION_AREA_FRACTION,
        name="processed expansion rest-area fraction",
    )
    _require_close(
        processed_contraction_fraction,
        EXPECTED_PROCESSED_CONTRACTION_AREA_FRACTION,
        name="processed contraction rest-area fraction",
    )

    raw_display_low = ratio < RATIO_DISPLAY_CLIM[0]
    raw_display_high = ratio > RATIO_DISPLAY_CLIM[1]
    processed_display_low = processed_ratio < RATIO_DISPLAY_CLIM[0]
    processed_display_high = processed_ratio > RATIO_DISPLAY_CLIM[1]
    raw_low_fraction = float(rest_area[raw_display_low].sum() / rest_area.sum())
    raw_high_fraction = float(rest_area[raw_display_high].sum() / rest_area.sum())
    if int(np.count_nonzero(raw_display_low)) != EXPECTED_RAW_DISPLAY_LOW_CELLS:
        raise ValueError("raw low display-saturation cell count changed")
    if int(np.count_nonzero(raw_display_high)) != EXPECTED_RAW_DISPLAY_HIGH_CELLS:
        raise ValueError("raw high display-saturation cell count changed")
    if np.any(processed_display_low) or np.any(processed_display_high):
        raise ValueError(
            "processed ratio unexpectedly saturates the shared display scale"
        )
    _require_close(
        raw_low_fraction,
        EXPECTED_RAW_DISPLAY_LOW_AREA_FRACTION,
        name="raw low display-saturation area fraction",
    )
    _require_close(
        raw_high_fraction,
        EXPECTED_RAW_DISPLAY_HIGH_AREA_FRACTION,
        name="raw high display-saturation area fraction",
    )

    rendered = corrected_skin.copy(deep=True)
    rendered.cell_data["TargetRestAreaRatio"] = ratio
    rendered.cell_data["HeuristicProcessedAreaRatio"] = processed_ratio
    stats = {
        "definition": {
            "raw_target_geometry": {
                "ratio": "r_raw = TargetArea / RestArea",
                "signed_percent": "100 * (r_raw - 1)",
            },
            "heuristic_processed_driver_equivalent": {
                "ratio": "r_processed = exp(LogAreaDiffused)",
                "pipeline": (
                    "log(raw target/rest ratio), symmetric soft deadband, "
                    "separate weighted caps, then 5 mm diffusion"
                ),
                "role": (
                    "input from which ExpansionWeight and "
                    "ContractionSeverityLogCapped are decoded"
                ),
                "not": [
                    "target geometry",
                    "deformed geometry",
                    "forward result",
                    "inverse result",
                    "stress-free area ratio",
                ],
            },
            "interpretation": {
                "raw_target_geometry": {
                    "r_below_1": "target triangle contracts in area",
                    "r_equal_1": "target triangle preserves area",
                    "r_above_1": "target triangle expands in area",
                },
                "heuristic_processed_driver_equivalent": {
                    "r_below_1": "processed contraction-side driver",
                    "r_equal_1": "neutral processed driver",
                    "r_above_1": "processed expansion-side driver",
                },
            },
            "log_statistics_note": (
                "Raw log transforms are retained for audit in JSON but are not "
                "rendered in the primary sheet."
            ),
        },
        "topology": {
            "points": int(corrected_skin.n_points),
            "triangles": int(corrected_skin.n_cells),
            "rest_area_m2": float(rest_area.sum()),
            "target_area_m2": float(target_area.sum()),
            "total_target_rest_area_ratio": total_ratio,
            "total_signed_area_change_percent": 100.0 * (total_ratio - 1.0),
        },
        "sign_support": {
            "raw_target_geometry": {
                "cells": observed_counts,
                "expansion_rest_area_fraction": expansion_fraction,
                "contraction_rest_area_fraction": contraction_fraction,
                "unchanged_rest_area_fraction": float(
                    rest_area[unchanged].sum() / rest_area.sum()
                ),
            },
            "heuristic_processed_driver_equivalent": {
                "cells": processed_counts,
                "expansion_rest_area_fraction": processed_expansion_fraction,
                "contraction_rest_area_fraction": processed_contraction_fraction,
                "unchanged_rest_area_fraction": float(
                    rest_area[processed_unchanged].sum() / rest_area.sum()
                ),
            },
        },
        "fields": {
            "target_rest_area_ratio": {
                **_weighted_stats(ratio, rest_area),
                "sha256_le_f8": EXPECTED_RAW_HASHES["target_rest_area_ratio"],
            },
            "signed_area_change_percent": {
                **_weighted_stats(signed_percent, rest_area),
                "sha256_le_f8": EXPECTED_RAW_HASHES["signed_area_change_percent"],
            },
            "ln_target_rest_area_ratio": {
                **_weighted_stats(ln_ratio, rest_area),
                "sha256_le_f8": EXPECTED_RAW_HASHES["ln_target_rest_area_ratio"],
            },
            "log2_target_rest_area_ratio": {
                **_weighted_stats(log2_ratio, rest_area),
                "sha256_le_f8": EXPECTED_RAW_HASHES["log2_target_rest_area_ratio"],
            },
            "log_area_diffused": {
                **_weighted_stats(diffused_log, rest_area),
                "sha256_le_f8": EXPECTED_RAW_HASHES["log_area_diffused"],
            },
            "heuristic_processed_area_ratio": {
                **_weighted_stats(processed_ratio, rest_area),
                "sha256_le_f8": EXPECTED_RAW_HASHES["processed_area_ratio"],
            },
            "heuristic_processed_signed_area_change_percent": {
                **_weighted_stats(processed_signed_percent, rest_area),
                "sha256_le_f8": EXPECTED_RAW_HASHES[
                    "processed_signed_area_change_percent"
                ],
            },
        },
        "display_saturation": {
            "shared_ratio_clim": list(RATIO_DISPLAY_CLIM),
            "raw_target_geometry": {
                "below_clim_cells": int(np.count_nonzero(raw_display_low)),
                "below_clim_rest_area_fraction": raw_low_fraction,
                "above_clim_cells": int(np.count_nonzero(raw_display_high)),
                "above_clim_rest_area_fraction": raw_high_fraction,
                "total_saturated_cells": int(
                    np.count_nonzero(raw_display_low | raw_display_high)
                ),
                "total_saturated_rest_area_fraction": (
                    raw_low_fraction + raw_high_fraction
                ),
            },
            "heuristic_processed_driver_equivalent": {
                "below_clim_cells": int(np.count_nonzero(processed_display_low)),
                "below_clim_rest_area_fraction": 0.0,
                "above_clim_cells": int(np.count_nonzero(processed_display_high)),
                "above_clim_rest_area_fraction": 0.0,
                "total_saturated_cells": int(
                    np.count_nonzero(processed_display_low | processed_display_high)
                ),
                "total_saturated_rest_area_fraction": 0.0,
            },
            "note": (
                "Shared color limits affect display only; JSON statistics retain "
                "every raw and processed triangle value."
            ),
        },
        "identity": {
            "rest_area_sha256_le_f8": EXPECTED_RAW_HASHES["rest_area"],
            "target_area_sha256_le_f8": EXPECTED_RAW_HASHES["target_area"],
            "ratio_exactly_equals_target_div_rest": True,
            "ln_ratio_exactly_equals_mapped_LogAreaRaw": True,
            "mapped_log_area_diffused_sha256_le_f8": EXPECTED_RAW_HASHES[
                "log_area_diffused"
            ],
            "processed_ratio_sha256_le_f8": EXPECTED_RAW_HASHES["processed_area_ratio"],
            "processed_ratio_exactly_equals_exp_mapped_LogAreaDiffused": True,
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


def _write_area_ratio_sheet(
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
        (
            "raw Smile target/rest area ratio",
            "TargetRestAreaRatio",
            RATIO_DISPLAY_CLIM,
            "area ratio (shared linear scale)\n"
            "blue: contraction | white: 1 | red: expansion",
            "%.2f",
        ),
        (
            "heuristic input: deadband + cap + 5 mm diffusion",
            "HeuristicProcessedAreaRatio",
            RATIO_DISPLAY_CLIM,
            "exp(LogAreaDiffused) (shared linear scale)\n"
            "heuristic input; not target geometry or physics result",
            "%.2f",
        ),
    )
    window_size = (2200, 1450)
    plotter = pv.Plotter(
        shape=(len(rows), len(views)),
        off_screen=True,
        window_size=window_size,
        lighting="light kit",
        border=False,
    )
    temporary = _temporary_path(path)
    image: np.ndarray | None = None
    try:
        plotter.set_background("white")
        for row, (row_label, scalar, clim, scalar_title, fmt) in enumerate(rows):
            for column, (view_label, direction, focus, scale) in enumerate(views):
                plotter.subplot(row, column)
                plotter.add_mesh(
                    skin,
                    scalars=scalar,
                    preference="cell",
                    cmap="RdBu_r",
                    clim=clim,
                    show_edges=False,
                    smooth_shading=False,
                    show_scalar_bar=column == len(views) - 1,
                    scalar_bar_args={
                        "title": scalar_title,
                        "title_font_size": 10,
                        "label_font_size": 9,
                        "n_labels": 5,
                        "fmt": fmt,
                    },
                )
                plotter.add_text(
                    f"{row_label} | {view_label}",
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
        raise ValueError("target-area-ratio screenshot returned an invalid image")
    if temporary.stat().st_size <= 10_000:
        raise ValueError("target-area-ratio screenshot is unexpectedly small")
    temporary.replace(path)
    return {
        "layout": (
            "2 rows (raw target/rest ratio, processed heuristic ratio) x 3 columns "
            "(front, 30 degree, mouth)"
        ),
        "rows": [label for label, *_ in rows],
        "views": [name for name, *_ in views],
        "colormap": "RdBu_r",
        "center": {
            "shared_ratio": 1.0,
            "meaning": "area preserving",
        },
        "shared_ratio_clim": list(RATIO_DISPLAY_CLIM),
        "scale": "linear",
        "clip_policy": "display saturation only; source arrays are not clipped",
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
        "target_area_change",
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
    if (
        render["scale"] != "linear"
        or render["shared_ratio_clim"] != [0.6, 1.4]
        or render["center"]["shared_ratio"] != 1.0
    ):
        raise ValueError("primary area-ratio render is not shared linear and centered")
    fields = summary["target_area_change"]["fields"]
    if fields["target_rest_area_ratio"]["min"] <= 0.0:
        raise ValueError("summary contains a nonpositive target/rest area ratio")
    if fields["heuristic_processed_area_ratio"]["min"] <= 0.0:
        raise ValueError("summary contains a nonpositive processed area ratio")
    saturation = summary["target_area_change"]["display_saturation"]
    if (
        saturation["heuristic_processed_driver_equivalent"]["total_saturated_cells"]
        != 0
    ):
        raise ValueError("processed ratio unexpectedly saturates the render")


def main(cfg: Config) -> None:
    _validate_config(cfg)
    input_provenance = {
        "corrected_skin": {
            "path": str(cfg.input_corrected_skin.resolve()),
            "allowed_arrays": ["RestArea", "GlobalPointId", "GroupId", "GroupName"],
            **_require_file_identity(
                cfg.input_corrected_skin,
                expected_size=CORRECTED_SKIN_SIZE_BYTES,
                expected_sha256=CORRECTED_SKIN_SHA256,
                name="corrected IsFace skin",
            ),
        },
        "driver_skin": {
            "path": str(cfg.input_driver_skin.resolve()),
            "allowed_arrays": [
                "RestArea",
                "TargetArea",
                "TargetRestAreaRatio",
                "LogAreaRaw",
                "LogAreaDiffused",
                "GlobalPointId",
            ],
            "prohibited_reuse": [
                "lambda",
                "mu",
                "SkinYoungModulusMPa",
                "ActivationInv",
                "StressFreeAreaRatio",
            ],
            **_require_file_identity(
                cfg.input_driver_skin,
                expected_size=DRIVER_SKIN_SIZE_BYTES,
                expected_sha256=DRIVER_SKIN_SHA256,
                name="old full-boundary target-area driver skin",
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
    rendered_skin, target_stats = _target_area_fields(
        corrected_skin, driver_skin, mapped
    )
    render = _write_area_ratio_sheet(cfg.output_png, skin=rendered_skin)
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
        "target_area_change": target_stats,
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
    temporary_json = _temporary_path(cfg.output_json)
    temporary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(cfg.output_json)
    cherries.set_step(0)
    cherries.log_metrics(
        {
            "target_area/total_target_rest_ratio": target_stats["topology"][
                "total_target_rest_area_ratio"
            ],
            "target_area/total_signed_change_percent": target_stats["topology"][
                "total_signed_area_change_percent"
            ],
            "target_area/ratio_min": target_stats["fields"]["target_rest_area_ratio"][
                "min"
            ],
            "target_area/ratio_max": target_stats["fields"]["target_rest_area_ratio"][
                "max"
            ],
            "target_area/raw_display_saturated_area_fraction": target_stats[
                "display_saturation"
            ]["raw_target_geometry"]["total_saturated_rest_area_fraction"],
            "target_area/processed_ratio_min": target_stats["fields"][
                "heuristic_processed_area_ratio"
            ]["min"],
            "target_area/processed_ratio_max": target_stats["fields"][
                "heuristic_processed_area_ratio"
            ]["max"],
            "target_area/processed_display_saturated_area_fraction": target_stats[
                "display_saturation"
            ]["heuristic_processed_driver_equivalent"][
                "total_saturated_rest_area_fraction"
            ],
            "execution/forward_started": 0,
            "execution/inverse_started": 0,
        }
    )
    logger.info("Wrote %s", cfg.output_png)
    logger.info("Wrote %s", cfg.output_json)


if __name__ == "__main__":
    cherries.main(main)
