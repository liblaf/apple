# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries
from liblaf.apple.common import ACTIVATION_INV, FRACTION, GLOBAL_POINT_ID, LAMBDA, MU

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DESIGN = "corrected-isface-four-case-selective-e000-c020-inverse-materials"
GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
PRODUCER = Path(__file__).resolve()

# This source-level blocker is intentionally checked before inputs or outputs are
# touched.  It has no CLI override.  A separate static review must flip it.
EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True
APPROVAL_BLOCKER = (
    "NO-GO: four-case material preparation awaits static review; do not execute "
    "until this source-level blocker is explicitly changed"
)

BASELINE_DIR = REPO_ROOT / "exp/2026/08/18/human-face-smile-plane-stress-skin/data"
CORRECTED_SKIN = BASELINE_DIR / "10-corrected-baseline/skin-isface-e0200-p000.vtp"
CORRECTED_MANIFEST = BASELINE_DIR / "10-corrected-baseline-manifest.json"
DRIVER_SKIN = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/data/"
    "10-material-candidates/skin-e100-p000.vtp"
)
BASELINE_STEM = (
    "20-human-face-smile-skin-no-prestrain-lr3-corrected-isface-e0200-p000-screen"
)

INPUT_IDENTITIES: dict[str, dict[str, Any]] = {
    "corrected_skin": {
        "path": CORRECTED_SKIN,
        "size_bytes": 1_138_550,
        "sha256": "4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f",
    },
    "corrected_manifest": {
        "path": CORRECTED_MANIFEST,
        "size_bytes": 7_723,
        "sha256": "d999be4fc941253b8daa84dca4a52ab44bd02b3e42ce2af6d151a2e14b64a21a",
    },
    "raw_area_ratio_driver": {
        "path": DRIVER_SKIN,
        "size_bytes": 38_742_137,
        "sha256": "ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f",
    },
}

BASELINE_INVERSE_ARTIFACTS: dict[str, dict[str, Any]] = {
    "aggregate_summary": {
        "path": BASELINE_DIR / "20-corrected-baseline-screen-summary-final.json",
        "size_bytes": 148_046,
        "sha256": "64a030366053b14eed9ad4da322d910146175fe7bb781e2dca8ee976c03c7045",
    },
    "case_summary": {
        "path": BASELINE_DIR / f"{BASELINE_STEM}-summary-final.json",
        "size_bytes": 126_540,
        "sha256": "575ebcbd7152a256917c2a11a9bf9bef9046f00f9831e18adc86d41645be1856",
    },
    "result": {
        "path": BASELINE_DIR / f"{BASELINE_STEM}.vtu",
        "size_bytes": 147_657_021,
        "sha256": "c6a0b183675ffb3ec537c1153544b041acd7aa0fdd5216c0cf9a50022d52b0a4",
    },
    "history": {
        "path": BASELINE_DIR / f"{BASELINE_STEM}-steps.vtkhdf",
        "size_bytes": 2_066_073_161,
        "sha256": "6e29d7b205e7901681942f0d413b091c5e4bce003ec4d789c2d7f69ded430d24",
    },
    "trace": {
        "path": BASELINE_DIR / f"{BASELINE_STEM}-trace.jsonl",
        "size_bytes": 91_767,
        "sha256": "a0f83957c832a119f6f031fb78a46fe52060d3b190a2ba0a1265f000c5d8cde3",
    },
    "target": {
        "path": BASELINE_DIR / f"{BASELINE_STEM}-target.vtu",
        "size_bytes": 84_419_492,
        "sha256": "89ec02dfd87330f7dc1d303639893f7698ef2e6098480c4e39fa2ad94240206c",
    },
}

EXPECTED_POINTS = 15_299
EXPECTED_TRIANGLES = 29_899
EXPECTED_SKIN_AREA_M2 = 0.04287998059707303
EXPECTED_EXPANDING_TRIANGLES = 16_723
EXPECTED_EXPANDING_AREA_FRACTION = 0.5455308228719783
EXPECTED_CONTRACTING_TRIANGLES = 13_159
EXPECTED_UNCHANGED_TRIANGLES = 17
EXPECTED_FLOOR_CLAMPED_TRIANGLES = 31
EXPECTED_FLOOR_CLAMPED_AREA_FRACTION = 0.0008793660414554653

EXPECTED_ARRAY_HASHES = {
    "triangle_keys": "dca8d77662f49b54250657424cfc29a3b438437081f83d65f7e108f312da2310",
    "mapped_driver_cell_indices": "13458107ceef23ecc144340101574f3fb4f2e157f90a9251434e7b09a86a66c3",
    "rest_area": "5a7b8eb9861fa509212afd610c60183f894b80db8ded53d22f3f9045bc6889de",
    "raw_ratio": "da98a6f48694eed30b1683ccf5a1f02fd67c87393b30003d07c52a7fa25bc606",
    "clipped_ratio": "aaf87f8d68485136c0ce09d113ce09de481654613c7d50c80ac2becb40e86e1e",
    "rho_c020": "d631449a1db997e6ce2eac9d3276ff8a23451461350f917026f19e9d50cc89f1",
    "activation_c020": "1366a17e86a2b182dd9b15512b9dc0664c869e416af7b5e591fbfb347fd53d55",
}

SKIN_E_MPA = 0.2
SKIN_NU = 0.49
SKIN_THICKNESS_M = 0.001
LINEAR_TIGHTENING = 0.02
LENGTH_FACTOR = 0.98
AREA_RATIO_FLOOR = 0.5
LAME_CONVERSION = (
    "thin-membrane plane-stress reduction: "
    "lambda = E * nu / (1 - nu**2); "
    "mu = E / (2 * (1 + nu))"
)
ENERGY_MEASURE = "fixed original reference area"

OUTPUT_ROOT = GROUP_DIR / "data/10-prepared-material-cases"
OUTPUT_MANIFEST = GROUP_DIR / "data/10-prepared-material-cases-manifest.json"
OUTPUT_TABLE = GROUP_DIR / "data/10-prepared-material-cases-table.md"
OUTPUT_SKINS = {
    "H0P1": OUTPUT_ROOT / "skin-h0p1-c020.vtp",
    "H1P1": OUTPUT_ROOT / "skin-h1p1-selective-e000-c020.vtp",
    "H1P0": OUTPUT_ROOT / "skin-h1p0-selective-e000-p000.vtp",
}
CASE_ORDER = ("H0P0", "H0P1", "H1P1", "H1P0")

REQUIRED_CELL_ARRAYS = (
    "RestArea",
    "SkinYoungModulusMPa",
    "SkinPoissonRatio",
    LAMBDA.vtk,
    MU.vtk,
    FRACTION.vtk,
    ACTIVATION_INV.vtk,
    "TargetRestAreaRatio",
    "ClippedTargetRestAreaRatio",
    "StressFreeAreaRatio",
    "ExpandingTriangle",
    "SelectiveZeroEnergy",
    "C020PrestrainEnabled",
)
REQUIRED_BASELINE_CELL_ARRAYS = (
    "RestArea",
    "SkinYoungModulusMPa",
    "SkinPoissonRatio",
    LAMBDA.vtk,
    MU.vtk,
    FRACTION.vtk,
    ACTIVATION_INV.vtk,
    "StressFreeAreaRatio",
)
MASK_ARRAYS = {
    "ExpandingTriangle",
    "SelectiveZeroEnergy",
    "C020PrestrainEnabled",
}


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    role: str
    heterogeneous: bool
    c020: bool
    filename: str | None


CASES = (
    CaseSpec(
        case_id="H0P0",
        role="reused corrected baseline",
        heterogeneous=False,
        c020=False,
        filename=None,
    ),
    CaseSpec(
        case_id="H0P1",
        role="prestrain main effect",
        heterogeneous=False,
        c020=True,
        filename=OUTPUT_SKINS["H0P1"].name,
    ),
    CaseSpec(
        case_id="H1P1",
        role="combined candidate",
        heterogeneous=True,
        c020=True,
        filename=OUTPUT_SKINS["H1P1"].name,
    ),
    CaseSpec(
        case_id="H1P0",
        role="selective-softening main effect",
        heterogeneous=True,
        c020=False,
        filename=OUTPUT_SKINS["H1P0"].name,
    ),
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_corrected_skin: Path = cherries.input(CORRECTED_SKIN)
    input_corrected_manifest: Path = cherries.input(CORRECTED_MANIFEST)
    input_raw_area_ratio_driver: Path = cherries.input(DRIVER_SKIN)
    output_manifest: Path = cherries.output(
        "10-prepared-material-cases-manifest.json", mkdir=True
    )
    output_table: Path = cherries.output(
        "10-prepared-material-cases-table.md", mkdir=True
    )
    output_h0p1_skin: Path = cherries.output(
        "10-prepared-material-cases/skin-h0p1-c020.vtp", mkdir=True
    )
    output_h1p1_skin: Path = cherries.output(
        "10-prepared-material-cases/skin-h1p1-selective-e000-c020.vtp",
        mkdir=True,
    )
    output_h1p0_skin: Path = cherries.output(
        "10-prepared-material-cases/skin-h1p0-selective-e000-p000.vtp",
        mkdir=True,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _raw_sha256(array: np.ndarray, *, dtype: str) -> str:
    values = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _require_hash(name: str, array: np.ndarray, expected: str, *, dtype: str) -> str:
    actual = _raw_sha256(array, dtype=dtype)
    if actual != expected:
        raise ValueError(f"{name} hash changed: {actual} != {expected}")
    return actual


def _require_close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError(f"{name} changed: {actual} != {expected}")


def _require_exact_path(actual: Path, expected: Path, *, name: str) -> None:
    if actual.resolve() != expected.resolve():
        raise ValueError(f"{name} must remain {expected}, got {actual}")


def _require_identity(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {name}: {path}")
    actual = _file_identity(path)
    expected = {
        "size_bytes": int(spec["size_bytes"]),
        "sha256": str(spec["sha256"]),
    }
    if actual != expected:
        raise ValueError(f"{name} identity changed: {actual} != {expected}")
    return {"path": str(path), **actual}


def _validate_config(cfg: Config) -> None:
    if not EXECUTION_APPROVED_AFTER_STATIC_REVIEW:
        raise RuntimeError(APPROVAL_BLOCKER)
    exact_paths = (
        (cfg.input_corrected_skin, CORRECTED_SKIN, "input_corrected_skin"),
        (
            cfg.input_corrected_manifest,
            CORRECTED_MANIFEST,
            "input_corrected_manifest",
        ),
        (
            cfg.input_raw_area_ratio_driver,
            DRIVER_SKIN,
            "input_raw_area_ratio_driver",
        ),
        (cfg.output_manifest, OUTPUT_MANIFEST, "output_manifest"),
        (cfg.output_table, OUTPUT_TABLE, "output_table"),
        (cfg.output_h0p1_skin, OUTPUT_SKINS["H0P1"], "output_h0p1_skin"),
        (cfg.output_h1p1_skin, OUTPUT_SKINS["H1P1"], "output_h1p1_skin"),
        (cfg.output_h1p0_skin, OUTPUT_SKINS["H1P0"], "output_h1p0_skin"),
    )
    for actual, expected, name in exact_paths:
        _require_exact_path(actual, expected, name=name)
    paths = [OUTPUT_MANIFEST, OUTPUT_TABLE, *OUTPUT_SKINS.values()]
    paths.extend(_temporary_path(path) for path in tuple(paths))
    stale = [str(path) for path in paths if path.exists()]
    if stale:
        raise FileExistsError(
            "refusing to overwrite material outputs or partial files: " + str(stale)
        )
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {OUTPUT_ROOT}")


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _read_polydata(path: Path, *, name: str) -> pv.PolyData:
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        raise TypeError(f"{name} read as {type(mesh).__name__}, expected PolyData")
    return mesh


def _triangles(mesh: pv.PolyData, *, name: str) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if mesh.n_cells == 0 or faces.size != 4 * mesh.n_cells:
        raise ValueError(f"{name} is not non-empty triangle-only PolyData")
    encoded = faces.reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        raise ValueError(f"{name} contains a non-triangle cell")
    triangles = encoded[:, 1:]
    if np.any((triangles < 0) | (triangles >= mesh.n_points)):
        raise ValueError(f"{name} triangle connectivity is out of range")
    return triangles


def _global_ids(mesh: pv.PolyData, *, name: str) -> np.ndarray:
    if GLOBAL_POINT_ID.vtk not in mesh.point_data:
        raise KeyError(f"{name} is missing {GLOBAL_POINT_ID.vtk}")
    raw = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk])
    ids = np.asarray(raw, dtype=np.int64)
    if raw.shape != (mesh.n_points,) or not np.array_equal(raw, ids):
        raise ValueError(f"{name} GlobalPointId is not an exact integer vector")
    if np.unique(ids).size != ids.size:
        raise ValueError(f"{name} GlobalPointId is not unique")
    return ids


def _triangle_keys(mesh: pv.PolyData, *, name: str) -> np.ndarray:
    return np.sort(_global_ids(mesh, name=name)[_triangles(mesh, name=name)], axis=1)


def _map_driver_cells(
    corrected: pv.PolyData, driver: pv.PolyData
) -> tuple[np.ndarray, dict[str, Any]]:
    corrected_keys = _triangle_keys(corrected, name="corrected skin")
    driver_keys = _triangle_keys(driver, name="raw-area-ratio driver")
    _require_hash(
        "corrected triangle keys",
        corrected_keys,
        EXPECTED_ARRAY_HASHES["triangle_keys"],
        dtype="<i8",
    )
    lookup: dict[tuple[int, int, int], int] = {}
    for cell_id, row in enumerate(driver_keys):
        key = tuple(int(value) for value in row)
        if key in lookup:
            raise ValueError(f"driver has duplicate triangle key {key}")
        lookup[key] = cell_id
    try:
        mapped = np.asarray(
            [lookup[tuple(int(value) for value in row)] for row in corrected_keys],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            f"corrected triangle is absent from driver: {error.args[0]}"
        ) from error
    if np.unique(mapped).size != corrected.n_cells:
        raise ValueError("corrected-to-driver triangle map is not injective")
    if not np.array_equal(driver_keys[mapped], corrected_keys):
        raise ValueError("corrected-to-driver triangle map failed exact readback")
    mapped_hash = _require_hash(
        "mapped driver cell indices",
        mapped,
        EXPECTED_ARRAY_HASHES["mapped_driver_cell_indices"],
        dtype="<i8",
    )
    return mapped, {
        "method": "sorted GlobalPointId triangle keys",
        "corrected_triangles": int(corrected.n_cells),
        "driver_triangles": int(driver.n_cells),
        "mapped_unique_driver_triangles": int(np.unique(mapped).size),
        "corrected_triangle_keys_sha256_le_i8": EXPECTED_ARRAY_HASHES["triangle_keys"],
        "mapped_driver_cell_indices_sha256_le_i8": mapped_hash,
        "exact_readback": True,
    }


def _require_scalar_cell_array(mesh: pv.DataSet, name: str) -> np.ndarray:
    if name not in mesh.cell_data:
        raise KeyError(f"mesh is missing cell array {name}")
    values = np.asarray(mesh.cell_data[name], dtype=np.float64)
    if values.shape != (mesh.n_cells,) or not np.isfinite(values).all():
        raise ValueError(f"cell array {name} is malformed or non-finite")
    return values


def _validate_corrected_skin(skin: pv.PolyData) -> np.ndarray:
    if skin.n_points != EXPECTED_POINTS or skin.n_cells != EXPECTED_TRIANGLES:
        raise ValueError(
            f"corrected skin dimensions changed: {skin.n_points}/{skin.n_cells}"
        )
    triangles = _triangles(skin, name="corrected skin")
    points = np.asarray(skin.points, dtype=np.float64)
    geometric_area = 0.5 * np.linalg.norm(
        np.cross(
            points[triangles[:, 1]] - points[triangles[:, 0]],
            points[triangles[:, 2]] - points[triangles[:, 0]],
        ),
        axis=1,
    )
    rest_area = _require_scalar_cell_array(skin, "RestArea")
    if not np.array_equal(geometric_area, rest_area):
        raise ValueError("corrected RestArea differs from exact original geometry")
    _require_hash(
        "corrected RestArea",
        rest_area,
        EXPECTED_ARRAY_HASHES["rest_area"],
        dtype="<f8",
    )
    _require_close(float(rest_area.sum()), EXPECTED_SKIN_AREA_M2, name="skin area")
    required = {
        "SkinYoungModulusMPa",
        "SkinPoissonRatio",
        LAMBDA.vtk,
        MU.vtk,
        FRACTION.vtk,
        ACTIVATION_INV.vtk,
        "StressFreeAreaRatio",
    }
    missing = sorted(required - set(skin.cell_data))
    if missing:
        raise KeyError(f"corrected skin is missing material arrays: {missing}")
    young = np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64)
    nu = np.asarray(skin.cell_data["SkinPoissonRatio"], dtype=np.float64)
    lam = np.asarray(skin.cell_data[LAMBDA.vtk], dtype=np.float64)
    mu = np.asarray(skin.cell_data[MU.vtk], dtype=np.float64)
    fraction = np.asarray(skin.cell_data[FRACTION.vtk], dtype=np.float64)
    activation = np.asarray(skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    stress_free_area_ratio = np.asarray(
        skin.cell_data["StressFreeAreaRatio"], dtype=np.float64
    )
    expected_lambda = young * nu / (1.0 - np.square(nu))
    expected_mu = young / (2.0 * (1.0 + nu))
    expected_fields = (
        ("E", young, np.full(skin.n_cells, SKIN_E_MPA)),
        ("nu", nu, np.full(skin.n_cells, SKIN_NU)),
        ("Lambda", lam, expected_lambda),
        ("Mu", mu, expected_mu),
        ("Fraction", fraction, np.ones(skin.n_cells)),
        ("ActivationInv", activation, np.zeros((skin.n_cells, 3))),
        (
            "StressFreeAreaRatio",
            stress_free_area_ratio,
            np.ones(skin.n_cells),
        ),
    )
    for name, actual, expected in expected_fields:
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=1.0e-13, atol=1.0e-14
        ):
            raise ValueError(f"corrected skin {name} is not exact H0P0")
    if not np.all(np.asarray(skin.point_data["IsFace"], dtype=bool)):
        raise ValueError("corrected skin contains a non-IsFace point")
    return rest_area


def _derive_fields(
    driver: pv.PolyData,
    mapped: np.ndarray,
    rest_area: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw_ratio = _require_scalar_cell_array(driver, "TargetRestAreaRatio")[mapped]
    if np.any(raw_ratio <= 0.0):
        raise ValueError("mapped raw target/rest area ratio is not positive")
    _require_hash(
        "mapped raw target/rest area ratio",
        raw_ratio,
        EXPECTED_ARRAY_HASHES["raw_ratio"],
        dtype="<f8",
    )
    clipped = np.clip(raw_ratio, AREA_RATIO_FLOOR, 1.0)
    _require_hash(
        "clipped target/rest area ratio",
        clipped,
        EXPECTED_ARRAY_HASHES["clipped_ratio"],
        dtype="<f8",
    )
    rho_c020 = np.square(LENGTH_FACTOR) * clipped
    _require_hash(
        "c020 stress-free area ratio",
        rho_c020,
        EXPECTED_ARRAY_HASHES["rho_c020"],
        dtype="<f8",
    )
    diag = np.reciprocal(np.sqrt(rho_c020)) - 1.0
    activation_c020 = np.stack((diag, diag, np.zeros_like(diag)), axis=1)
    _require_hash(
        "c020 ActivationInv",
        activation_c020,
        EXPECTED_ARRAY_HASHES["activation_c020"],
        dtype="<f8",
    )
    expanding = raw_ratio > 1.0
    contracting = raw_ratio < 1.0
    unchanged = raw_ratio == 1.0
    clamped = raw_ratio < AREA_RATIO_FLOOR
    counts = {
        "expanding": int(expanding.sum()),
        "contracting": int(contracting.sum()),
        "unchanged": int(unchanged.sum()),
        "floor_clamped": int(clamped.sum()),
    }
    expected_counts = {
        "expanding": EXPECTED_EXPANDING_TRIANGLES,
        "contracting": EXPECTED_CONTRACTING_TRIANGLES,
        "unchanged": EXPECTED_UNCHANGED_TRIANGLES,
        "floor_clamped": EXPECTED_FLOOR_CLAMPED_TRIANGLES,
    }
    if counts != expected_counts:
        raise ValueError(
            f"raw-area classification changed: {counts} != {expected_counts}"
        )
    expanding_area_fraction = float(rest_area[expanding].sum() / rest_area.sum())
    clamped_area_fraction = float(rest_area[clamped].sum() / rest_area.sum())
    _require_close(
        expanding_area_fraction,
        EXPECTED_EXPANDING_AREA_FRACTION,
        name="expanding rest-area fraction",
    )
    _require_close(
        clamped_area_fraction,
        EXPECTED_FLOOR_CLAMPED_AREA_FRACTION,
        name="floor-clamped rest-area fraction",
    )
    return (
        {
            "raw_ratio": raw_ratio,
            "clipped_ratio": clipped,
            "rho_c020": rho_c020,
            "activation_c020": activation_c020,
            "expanding": expanding,
        },
        {
            "formula": (
                "R=TargetArea/RestArea; zero_skin_energy=(R>1); "
                "rho_c020=0.98^2*clip(R,0.5,1); "
                "ActivationInv=[rho_c020^(-1/2)-1,"
                "rho_c020^(-1/2)-1,0]"
            ),
            "raw_ratio_sha256_le_f8": EXPECTED_ARRAY_HASHES["raw_ratio"],
            "clipped_ratio_sha256_le_f8": EXPECTED_ARRAY_HASHES["clipped_ratio"],
            "rho_c020_sha256_le_f8": EXPECTED_ARRAY_HASHES["rho_c020"],
            "activation_c020_sha256_le_f8": EXPECTED_ARRAY_HASHES["activation_c020"],
            "expanding_triangles": counts["expanding"],
            "expanding_rest_area_fraction": expanding_area_fraction,
            "contracting_triangles": counts["contracting"],
            "unchanged_triangles": counts["unchanged"],
            "floor_clamped_triangles": counts["floor_clamped"],
            "floor_clamped_rest_area_fraction": clamped_area_fraction,
        },
    )


def _array_record(array: np.ndarray, *, dtype: str) -> dict[str, Any]:
    values = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    finite = bool(np.isfinite(values).all())
    if not finite:
        raise ValueError("cannot record a non-finite material array")
    return {
        "association": "cell",
        "shape": list(values.shape),
        "dtype": dtype,
        "sha256_le_c": hashlib.sha256(values.tobytes()).hexdigest(),
        "min": float(values.min()),
        "max": float(values.max()),
        "finite": True,
    }


def _material_array_records(
    skin: pv.PolyData, *, names: tuple[str, ...] = REQUIRED_CELL_ARRAYS
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in skin.cell_data:
            raise KeyError(f"candidate skin is missing required cell array {name}")
        raw = np.asarray(skin.cell_data[name])
        dtype = "u1" if name in MASK_ARRAYS else "<f8"
        records[name] = _array_record(raw, dtype=dtype)
    return records


def _build_case_skin(
    source: pv.PolyData,
    fields: dict[str, np.ndarray],
    spec: CaseSpec,
) -> pv.PolyData:
    if spec.case_id == "H0P0":
        raise ValueError("H0P0 must be reused, never regenerated")
    skin = source.copy(deep=True)
    expanding = np.asarray(fields["expanding"], dtype=bool)
    young = (
        np.where(expanding, 0.0, SKIN_E_MPA)
        if spec.heterogeneous
        else np.full(skin.n_cells, SKIN_E_MPA)
    )
    nu = np.full(skin.n_cells, SKIN_NU)
    lam = young * nu / (1.0 - np.square(nu))
    mu = young / (2.0 * (1.0 + nu))
    if spec.c020:
        rho = np.asarray(fields["rho_c020"], dtype=np.float64)
        activation = np.asarray(fields["activation_c020"], dtype=np.float64)
    else:
        rho = np.ones(skin.n_cells, dtype=np.float64)
        activation = np.zeros((skin.n_cells, 3), dtype=np.float64)
    skin.cell_data["SkinYoungModulusMPa"] = young
    skin.cell_data["SkinPoissonRatio"] = nu
    skin.cell_data[LAMBDA.vtk] = lam
    skin.cell_data[MU.vtk] = mu
    skin.cell_data[FRACTION.vtk] = np.ones(skin.n_cells, dtype=np.float64)
    skin.cell_data[ACTIVATION_INV.vtk] = activation
    skin.cell_data["TargetRestAreaRatio"] = fields["raw_ratio"]
    skin.cell_data["ClippedTargetRestAreaRatio"] = fields["clipped_ratio"]
    skin.cell_data["StressFreeAreaRatio"] = rho
    skin.cell_data["ExpandingTriangle"] = expanding.astype(np.uint8)
    skin.cell_data["SelectiveZeroEnergy"] = (
        expanding.astype(np.uint8)
        if spec.heterogeneous
        else np.zeros(skin.n_cells, dtype=np.uint8)
    )
    skin.cell_data["C020PrestrainEnabled"] = np.full(
        skin.n_cells, int(spec.c020), dtype=np.uint8
    )
    return skin


def _require_exact_readback(  # noqa: C901
    expected: pv.PolyData, actual: pv.PolyData
) -> None:
    if expected.n_points != actual.n_points or expected.n_cells != actual.n_cells:
        raise ValueError("candidate VTP readback dimensions changed")
    if not np.array_equal(expected.points, actual.points):
        raise ValueError("candidate VTP readback points changed")
    if not np.array_equal(expected.faces, actual.faces):
        raise ValueError("candidate VTP readback connectivity changed")
    if set(expected.point_data) != set(actual.point_data):
        raise ValueError("candidate VTP readback point-array names changed")
    if set(expected.cell_data) != set(actual.cell_data):
        raise ValueError("candidate VTP readback cell-array names changed")
    if set(expected.field_data) != set(actual.field_data):
        raise ValueError("candidate VTP readback field-array names changed")
    for association, expected_data, actual_data in (
        ("point", expected.point_data, actual.point_data),
        ("cell", expected.cell_data, actual.cell_data),
    ):
        for name in expected_data:
            if not np.array_equal(
                np.asarray(expected_data[name]), np.asarray(actual_data[name])
            ):
                raise ValueError(f"candidate VTP {association} array {name} changed")
    for name in expected.field_data:
        if not np.array_equal(
            np.asarray(expected.field_data[name]), np.asarray(actual.field_data[name])
        ):
            raise ValueError(f"candidate VTP field array {name} changed")


def _write_skin_atomic(skin: pv.PolyData, path: Path) -> pv.PolyData:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    # PyVista's XML writer mutates its input by adding private bit-array
    # metadata.  Write a deep copy so the in-memory expected object remains a
    # stable reference for the exact readback comparison below.
    skin.copy(deep=True).save(temporary, binary=True)
    readback = _read_polydata(temporary, name=f"temporary {path.name}")
    _require_exact_readback(skin, readback)
    temporary.replace(path)
    final = _read_polydata(path, name=f"final {path.name}")
    _require_exact_readback(skin, final)
    return final


def _case_definition(spec: CaseSpec) -> dict[str, Any]:
    return {
        "skin_domain": "all-vertex IsFace filtered PolyData",
        "skin_thickness_m": SKIN_THICKNESS_M,
        "young_modulus": (
            "E=0 MPa iff raw R>1, else E=0.2 MPa"
            if spec.heterogeneous
            else "homogeneous E=0.2 MPa"
        ),
        "poisson_ratio": SKIN_NU,
        "lame_conversion": LAME_CONVERSION,
        "prestrain": (
            "rho=0.98^2*clip(raw R,0.5,1); ActivationInv=[rho^-1/2-1,rho^-1/2-1,0]"
            if spec.c020
            else "p000: rho=1 and ActivationInv is exact zero"
        ),
        "energy_measure": ENERGY_MEASURE,
        "important_interaction": (
            "prestrain is stored on all IsFace triangles, but produces no membrane "
            "force on the E=0 expanding triangles"
            if spec.heterogeneous and spec.c020
            else None
        ),
    }


def _case_row(
    spec: CaseSpec,
    *,
    skin_path: Path,
    skin: pv.PolyData,
    arrays: dict[str, dict[str, Any]],
    generated: bool,
    inverse_artifacts: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    relative_path: str | None
    try:
        relative_path = str(skin_path.relative_to(GROUP_DIR / "data"))
    except ValueError:
        relative_path = None
    return {
        "case_id": spec.case_id,
        "role": spec.role,
        "generated": generated,
        "young_modulus_mode": (
            "selective-e000-where-raw-r-gt-1"
            if spec.heterogeneous
            else "homogeneous-e0200"
        ),
        "prestrain_mode": ("c020-raw-area-ratio-floor-050" if spec.c020 else "p000"),
        "definition": _case_definition(spec),
        "skin": {
            "path": str(skin_path),
            "relative_path": relative_path,
            "file_identity": _file_identity(skin_path),
            "points": int(skin.n_points),
            "triangles": int(skin.n_cells),
            "arrays": arrays,
        },
        "inverse_artifacts": inverse_artifacts,
        "validation": {"ok": True, "errors": []},
    }


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _table(cases: list[dict[str, Any]]) -> str:
    lines = [
        "| Case | E mode | Prestrain | Generated | Skin |",
        "|---|---|---|---:|---|",
    ]
    lines.extend(
        "| {case_id} | {young_modulus_mode} | {prestrain_mode} | {generated} | `{path}` |".format(
            case_id=row["case_id"],
            young_modulus_mode=row["young_modulus_mode"],
            prestrain_mode=row["prestrain_mode"],
            generated=row["generated"],
            path=row["skin"]["path"],
        )
        for row in cases
    )
    return "\n".join(lines) + "\n"


def _verify_input_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("complete") is not True:
        raise ValueError("corrected baseline material manifest is incomplete")
    if manifest.get("design") != "isface-plane-stress-corrected-baseline":
        raise ValueError("corrected baseline material manifest design changed")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("corrected baseline material manifest candidate count changed")
    candidate = candidates[0]
    expected = {
        "label": "isface-e0200-p000",
        "validation/ok": True,
        "material/E_MPa": SKIN_E_MPA,
        "material/nu": SKIN_NU,
        "material/lame_conversion": LAME_CONVERSION,
        "material/prestrain": "p000: ActivationInv is exactly zero",
    }
    changed = {
        key: (candidate.get(key), value)
        for key, value in expected.items()
        if candidate.get(key) != value
    }
    if changed:
        raise ValueError(f"corrected baseline manifest contract changed: {changed}")


def _recheck_inputs(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checked: dict[str, dict[str, Any]] = {}
    for name, row in rows.items():
        path = Path(row["path"])
        actual = _file_identity(path)
        expected = {
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }
        if actual != expected:
            raise RuntimeError(f"input {name} changed during preparation")
        checked[name] = {"path": str(path), **actual, "unchanged": True}
    return {"all_unchanged": True, "files": checked}


def main(cfg: Config) -> None:
    _validate_config(cfg)
    producer_identity_before = _file_identity(PRODUCER)
    input_rows = {
        name: _require_identity(name, spec) for name, spec in INPUT_IDENTITIES.items()
    }
    baseline_inverse_rows = {
        name: _require_identity(f"baseline inverse {name}", spec)
        for name, spec in BASELINE_INVERSE_ARTIFACTS.items()
    }
    corrected_manifest = _read_json(cfg.input_corrected_manifest)
    _verify_input_manifest(corrected_manifest)
    corrected = _read_polydata(cfg.input_corrected_skin, name="corrected H0P0 skin")
    driver = _read_polydata(
        cfg.input_raw_area_ratio_driver, name="raw-area-ratio driver skin"
    )
    rest_area = _validate_corrected_skin(corrected)
    mapped, mapping = _map_driver_cells(corrected, driver)
    fields, derivation = _derive_fields(driver, mapped, rest_area)

    cases: list[dict[str, Any]] = []
    for spec in CASES:
        if spec.case_id == "H0P0":
            cases.append(
                _case_row(
                    spec,
                    skin_path=CORRECTED_SKIN,
                    skin=corrected,
                    arrays=_material_array_records(
                        corrected, names=REQUIRED_BASELINE_CELL_ARRAYS
                    ),
                    generated=False,
                    inverse_artifacts=baseline_inverse_rows,
                )
            )
            continue
        path = OUTPUT_SKINS[spec.case_id]
        built = _build_case_skin(corrected, fields, spec)
        expected_arrays = _material_array_records(built)
        readback = _write_skin_atomic(built, path)
        readback_arrays = _material_array_records(readback)
        if readback_arrays != expected_arrays:
            raise ValueError(f"{spec.case_id} material-array readback records changed")
        cases.append(
            _case_row(
                spec,
                skin_path=path,
                skin=readback,
                arrays=readback_arrays,
                generated=True,
                inverse_artifacts=None,
            )
        )

    if tuple(row["case_id"] for row in cases) != CASE_ORDER:
        raise RuntimeError("material case order changed")
    final_recheck = _recheck_inputs({**input_rows, **baseline_inverse_rows})
    table = _table(cases)
    _write_text_atomic(cfg.output_table, table)

    if _file_identity(PRODUCER) != producer_identity_before:
        raise RuntimeError("material producer changed during preparation")
    producer_identity = {
        "path": str(PRODUCER),
        "relative_path": str(PRODUCER.relative_to(GROUP_DIR)),
        "file_identity": producer_identity_before,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "producer": producer_identity,
        "input_provenance": {
            "inputs": input_rows,
            "baseline_inverse_artifacts": baseline_inverse_rows,
            "corrected_manifest_contract_verified": True,
            "final_recheck": final_recheck,
        },
        "mapping": {**mapping, **derivation},
        "constants": {
            "skin_E_MPa": SKIN_E_MPA,
            "skin_nu": SKIN_NU,
            "skin_thickness_m": SKIN_THICKNESS_M,
            "skin_lame_conversion": LAME_CONVERSION,
            "skin_energy_measure": ENERGY_MEASURE,
            "linear_tightening": LINEAR_TIGHTENING,
            "length_factor": LENGTH_FACTOR,
            "uniform_natural_area_ratio": float(np.square(LENGTH_FACTOR)),
            "raw_area_ratio_floor": AREA_RATIO_FLOOR,
            "selective_zero_energy_rule": "E=0 iff raw TargetRestAreaRatio > 1",
        },
        "anatomy_material_contract": {
            "skin": {
                "domain": "all-vertex IsFace filtered PolyData",
                "E_MPa": "case-dependent: 0 or 0.2",
                "nu": SKIN_NU,
                "thickness_m": SKIN_THICKNESS_M,
                "lame_conversion": LAME_CONVERSION,
                "energy_measure": ENERGY_MEASURE,
            },
            "volume": {
                "lame_conversion": (
                    "3d: lambda=E*nu/((1+nu)*(1-2*nu)); mu=E/(2*(1+nu))"
                ),
                "fat": {"E_MPa": 0.003, "nu": 0.49},
                "muscle": {
                    "E_MPa": 0.03,
                    "nu": 0.49,
                    "activation": "fresh exact-zero inverse initialization",
                    "parameterization": "independent 6-DoF per active muscle tet",
                },
                "aponeurosis": {"E_MPa": 0.1, "nu": 0.35},
            },
            "boundary": {
                "policy": "all cross-section incident vertices fixed to zero displacement",
                "expected_cut_vertices": 6_980,
                "expected_model_fixed_vertices": 33_636,
                "expected_model_fixed_dofs": 100_908,
            },
        },
        "case_order": list(CASE_ORDER),
        "cases": cases,
        "output_contract": {
            "manifest_path": str(OUTPUT_MANIFEST),
            "table_path": str(OUTPUT_TABLE),
            "candidate_root": str(OUTPUT_ROOT),
            "generated_candidate_vtps": 3,
            "baseline_reused_by_identity": True,
            "overwrite_policy": "refuse any existing final or temporary output",
        },
        "approval": {
            "material_preparation_static_review": True,
            "inverse_execution_approved": False,
            "forward_or_adjoint_smoke_approved": False,
            "meaning": (
                "this manifest certifies only deterministic material preparation; "
                "downstream smoke and inverse scripts retain independent blockers"
            ),
        },
    }
    _write_text_atomic(
        cfg.output_manifest,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    written = _read_json(cfg.output_manifest)
    if written != manifest or written.get("complete") is not True:
        raise RuntimeError("final material manifest strict readback failed")
    if _file_identity(PRODUCER) != producer_identity_before:
        raise RuntimeError("material producer changed before final readback")
    cherries.log_metrics(
        {
            "material/cases": len(cases),
            "material/generated_vtps": 3,
            "material/zero_energy_triangles": EXPECTED_EXPANDING_TRIANGLES,
            "material/zero_energy_area_fraction": EXPECTED_EXPANDING_AREA_FRACTION,
        }
    )
    logger.info("Wrote reviewed four-case material manifest to %s", cfg.output_manifest)


if __name__ == "__main__":
    cherries.main(main)
