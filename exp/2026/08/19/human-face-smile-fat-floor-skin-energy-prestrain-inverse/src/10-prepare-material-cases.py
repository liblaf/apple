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

SCHEMA_VERSION = 2
DESIGN = "corrected-isface-two-case-selective-efat-c020-inverse-materials-v2"
GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
PRODUCER = Path(__file__).resolve()

# This source-level blocker is intentionally checked before inputs or outputs are
# touched.  It has no CLI override.  A separate static review must flip it.
EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True
APPROVAL_BLOCKER = (
    "NO-GO: fat-floor material preparation awaits static review; do not execute "
    "until this source-level blocker is explicitly changed"
)
APPROVAL_DISABLED_ASSIGNMENT = b"EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False"
APPROVAL_ENABLED_ASSIGNMENT = b"EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True"

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
PRIOR_SELECTIVE_GROUP = (
    REPO_ROOT
    / "exp/2026/08/19/human-face-smile-selective-skin-energy-prestrain-inverse"
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

H0P1_CONTROL_ARTIFACTS: dict[str, dict[str, Any]] = {
    "skin": {
        "path": PRIOR_SELECTIVE_GROUP
        / "data/10-prepared-material-cases/skin-h0p1-c020.vtp",
        "size_bytes": 1_898_983,
        "sha256": "9b69f6cad6cfc7c6deadbd687d2947a44eedf7a295aae3540ac104ea50ebaacf",
    },
    "aggregate_summary": {
        "path": PRIOR_SELECTIVE_GROUP
        / "data/20-selective-skin-prestrain-inverse-summary-final.json",
        "size_bytes": 387_036,
        "sha256": "cf533bb16f481d75587531dfcd5aa21ed1065ed02539ea3ff0290e94d6cd2de6",
    },
    "case_summary": {
        "path": PRIOR_SELECTIVE_GROUP / "data/20-h0p1-summary-final.json",
        "size_bytes": 111_232,
        "sha256": "0ecf17c2a25cc03ebccb42a7ca3bd25bbeaf2fde0b0da3f1dec156efe8d99b2c",
    },
    "result": {
        "path": PRIOR_SELECTIVE_GROUP / "data/20-h0p1.vtu",
        "size_bytes": 147_640_393,
        "sha256": "eabec1d0493f004d066f94f20b5ac6725f8d84245ceccfddc51dc191dd96cde0",
    },
    "history": {
        "path": PRIOR_SELECTIVE_GROUP / "data/20-h0p1-steps.vtkhdf",
        "size_bytes": 2_071_499_339,
        "sha256": "c082c6202218c74a5fa3ef9c01048487a7ba4988621a7628eb1dc6a641debceb",
    },
    "trace": {
        "path": PRIOR_SELECTIVE_GROUP / "data/20-h0p1-trace.jsonl",
        "size_bytes": 79_471,
        "sha256": "657de3a7bc6cbdf361dd805aafa9eedc64bc77a1487b93327e0d82de6257f692",
    },
    "target": {
        "path": PRIOR_SELECTIVE_GROUP / "data/20-h0p1-target.vtu",
        "size_bytes": 84_419_492,
        "sha256": "3d6e3fe1baa48745d8592b109ad513cf7a822f05ece26ccbf2045c57b8f44418",
    },
}

V1_OUTPUT_IDENTITIES: dict[str, dict[str, Any]] = {
    "manifest": {
        "path": GROUP_DIR / "data/10-prepared-material-cases-manifest.json",
        "size_bytes": 34_033,
        "sha256": "843179e074d00ead3469cd9e0e5f69f2f0b521398e86709d1ecb466bda2f26a9",
    },
    "table": {
        "path": GROUP_DIR / "data/10-prepared-material-cases-table.md",
        "size_bytes": 571,
        "sha256": "e3bd37cc1351a42d665dcb21d6eabc01cba95f5fe2a56de3faa47d480e71c0de",
    },
    "hfp1_skin": {
        "path": GROUP_DIR
        / "data/10-prepared-material-cases/skin-hfp1-selective-efat-c020.vtp",
        "size_bytes": 1_933_979,
        "sha256": "2199b33ba7896bfde82a9e1fcf12e7782e9e89daa742b787eb267a824f1ae855",
    },
    "hfp0_skin": {
        "path": GROUP_DIR
        / "data/10-prepared-material-cases/skin-hfp0-selective-efat-p000.vtp",
        "size_bytes": 1_611_312,
        "sha256": "f3c2ebaf95f7b82c15a15743ef2a1be3eea378f1a89f9044df379179732c6bf7",
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
EXPECTED_FAT_FLOOR_AREA_WEIGHTED_E_MPA = 0.09253042789422025

EXPECTED_ARRAY_HASHES = {
    "triangle_keys": "dca8d77662f49b54250657424cfc29a3b438437081f83d65f7e108f312da2310",
    "mapped_driver_cell_indices": "13458107ceef23ecc144340101574f3fb4f2e157f90a9251434e7b09a86a66c3",
    "rest_area": "5a7b8eb9861fa509212afd610c60183f894b80db8ded53d22f3f9045bc6889de",
    "raw_ratio": "da98a6f48694eed30b1683ccf5a1f02fd67c87393b30003d07c52a7fa25bc606",
    "clipped_ratio": "aaf87f8d68485136c0ce09d113ce09de481654613c7d50c80ac2becb40e86e1e",
    "rho_c020": "d631449a1db997e6ce2eac9d3276ff8a23451461350f917026f19e9d50cc89f1",
    "activation_c020": "1366a17e86a2b182dd9b15512b9dc0664c869e416af7b5e591fbfb347fd53d55",
    "expanding_mask": "1da30d0805e41ebb56de39fb26ccd54c2b7a8bd7f4d1257459cbc7b9aa0bc05e",
    "fat_floor_young": "3d16df172d08edfdb52077ac8961aa80bc3363ed9e9d8b0b1f1a0f3695b82c1e",
    "fat_floor_lambda": "d6ef7eed22196a7d8cb2bbe519d565fbeeafbf7c0ad6285e0de7377b374f2bf1",
    "fat_floor_mu": "09a30f9c6a95003ddbc6e10c3880f5f8888c152656d2b49a499286681f4d05fa",
    "p000_stress_free_area_ratio": "aa74d25a4afece1f232101f98e5fda8177935c3e833ae578be636d9ff42294c2",
    "p000_activation": "051fe4599913dc590cb39aa79f7bb51578efc3323cd9c0a337be804d12d8f224",
    "mask_all_zero": "7bd2b07dd38fb3c32897d8931f62e04381be41f4b91df22c102fb4db1f477ce1",
    "mask_all_one": "d2580c50d0997b172975ce03f35909cfcd6660043ca11b84b86a61b4871b367c",
    "activation_diag_c020": "9886106f0c5a60c8007e6a6eda26eaa9c8bc33b159c9f719307bec5c832664d3",
    "activation_diag_p000": "c1a2cfddfccbbbf0c5c331571cb1ccab9fea8d6afb8c7aa920c74abe1d86f7e2",
    "inherited_int8_all_zero": "7bd2b07dd38fb3c32897d8931f62e04381be41f4b91df22c102fb4db1f477ce1",
    "inherited_int8_all_one": "d2580c50d0997b172975ce03f35909cfcd6660043ca11b84b86a61b4871b367c",
    "teeth_proximity": "c8866dd1aecdb6c7c7fee45c009103f48cc5a4b7a3752926cbb45bc88b88a76e",
}

SKIN_E_MPA = 0.2
FAT_E_MPA = 0.003
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
VOLUME_LAME_CONVERSION = (
    "unchanged 3D isotropic volume convention: "
    "lambda = E * nu / ((1 + nu) * (1 - 2 * nu)); "
    "mu = E / (2 * (1 + nu))"
)
ENERGY_MEASURE = "fixed original reference area"

OUTPUT_ROOT = GROUP_DIR / "data/10-prepared-material-cases-v2"
OUTPUT_MANIFEST = GROUP_DIR / "data/10-prepared-material-cases-v2-manifest.json"
OUTPUT_TABLE = GROUP_DIR / "data/10-prepared-material-cases-v2-table.md"
OUTPUT_SKINS = {
    "HFP1": OUTPUT_ROOT / "skin-hfp1-selective-efat-c020.vtp",
    "HFP0": OUTPUT_ROOT / "skin-hfp0-selective-efat-p000.vtp",
}
CASE_ORDER = ("HFP1", "HFP0")
CONTROL_ORDER = ("H0P0", "H0P1")

REQUIRED_CELL_ARRAYS = (
    "RestArea",
    "SkinYoungModulusMPa",
    "SkinPoissonRatio",
    "SkinActivationInvDiag",
    LAMBDA.vtk,
    MU.vtk,
    FRACTION.vtk,
    ACTIVATION_INV.vtk,
    "TargetRestAreaRatio",
    "ClippedTargetRestAreaRatio",
    "StressFreeAreaRatio",
    "ExpandingTriangle",
    "SelectiveFatFloor",
    "C020PrestrainEnabled",
    "ArtificialCutTriangle",
    "DisallowedGroupTriangle",
    "FixedTriangle",
    "GingivaProximityTriangle",
    "IsFaceTriangle",
    "SourceOuterTriangle",
    "TeethProximityTriangle",
)
MASK_ARRAYS = {
    "ExpandingTriangle",
    "SelectiveFatFloor",
    "C020PrestrainEnabled",
}
INHERITED_DOMAIN_ONE_ARRAYS = ("IsFaceTriangle", "SourceOuterTriangle")
INHERITED_DOMAIN_ZERO_ARRAYS = (
    "ArtificialCutTriangle",
    "DisallowedGroupTriangle",
    "FixedTriangle",
)
INHERITED_DIAGNOSTIC_ARRAYS = (
    "GingivaProximityTriangle",
    "TeethProximityTriangle",
)
INHERITED_INT8_ARRAYS = {
    *INHERITED_DOMAIN_ONE_ARRAYS,
    *INHERITED_DOMAIN_ZERO_ARRAYS,
    *INHERITED_DIAGNOSTIC_ARRAYS,
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
        case_id="HFP1",
        role="fat-floor selective softening plus c020",
        heterogeneous=True,
        c020=True,
        filename=OUTPUT_SKINS["HFP1"].name,
    ),
    CaseSpec(
        case_id="HFP0",
        role="fat-floor selective softening without prestrain",
        heterogeneous=True,
        c020=False,
        filename=OUTPUT_SKINS["HFP0"].name,
    ),
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_corrected_skin: Path = cherries.input(CORRECTED_SKIN)
    input_corrected_manifest: Path = cherries.input(CORRECTED_MANIFEST)
    input_raw_area_ratio_driver: Path = cherries.input(DRIVER_SKIN)
    output_manifest: Path = cherries.output(
        "10-prepared-material-cases-v2-manifest.json", mkdir=True
    )
    output_table: Path = cherries.output(
        "10-prepared-material-cases-v2-table.md", mkdir=True
    )
    output_hfp1_skin: Path = cherries.output(
        "10-prepared-material-cases-v2/skin-hfp1-selective-efat-c020.vtp",
        mkdir=True,
    )
    output_hfp0_skin: Path = cherries.output(
        "10-prepared-material-cases-v2/skin-hfp0-selective-efat-p000.vtp",
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


def _bytes_identity(content: bytes) -> dict[str, int | str]:
    return {
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _replace_source_approval_assignment(
    source: bytes, *, expected: bytes, replacement: bytes
) -> tuple[bytes, int]:
    lines = source.splitlines(keepends=True)
    replacements = 0
    for index, line in enumerate(lines):
        body = line.rstrip(b"\r\n")
        ending = line[len(body) :]
        if body == expected:
            lines[index] = replacement + ending
            replacements += 1
    return b"".join(lines), replacements


def _producer_source_provenance(path: Path) -> dict[str, Any]:
    executable = path.read_bytes()
    preapproval, replacement_count = _replace_source_approval_assignment(
        executable,
        expected=APPROVAL_ENABLED_ASSIGNMENT,
        replacement=APPROVAL_DISABLED_ASSIGNMENT,
    )
    if not EXECUTION_APPROVED_AFTER_STATIC_REVIEW or replacement_count != 1:
        raise RuntimeError(
            "material execution requires the single source approval assignment "
            "to be True"
        )
    reconstructed, reconstruction_count = _replace_source_approval_assignment(
        preapproval,
        expected=APPROVAL_DISABLED_ASSIGNMENT,
        replacement=APPROVAL_ENABLED_ASSIGNMENT,
    )
    if reconstruction_count != 1 or reconstructed != executable:
        raise RuntimeError(
            "preapproval source does not reconstruct the executable via only the "
            "approved assignment flip"
        )
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(GROUP_DIR)),
        "statically_reviewed_preapproval_source": {
            "file_identity": _bytes_identity(preapproval),
            "approval_assignment": ("EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False"),
        },
        "executable_source": {
            "file_identity": _bytes_identity(executable),
            "approval_assignment": ("EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True"),
        },
        "approval_only_reconstruction": {
            "verified": True,
            "replacement_count": 1,
            "permitted_edit": (
                "EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False -> True"
            ),
            "reconstructed_executable_identity": _bytes_identity(reconstructed),
        },
    }


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
        (cfg.output_hfp1_skin, OUTPUT_SKINS["HFP1"], "output_hfp1_skin"),
        (cfg.output_hfp0_skin, OUTPUT_SKINS["HFP0"], "output_hfp0_skin"),
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
        "SkinActivationInvDiag",
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


def _audit_convenience_arrays(  # noqa: C901
    skin: pv.PolyData,
    *,
    expected_activation_diag_hash: str,
    inherited_reference: pv.PolyData | None = None,
) -> dict[str, Any]:
    triangles = _triangles(skin, name="convenience-array skin")
    for point_name in ("IsFace", "IsFixed", "IsTeeth", "IsGingiva"):
        if point_name not in skin.point_data:
            raise KeyError(
                f"convenience-array skin is missing point array {point_name}"
            )
    if not np.all(np.asarray(skin.point_data["IsFace"], dtype=bool)):
        raise ValueError("convenience-array skin contains a non-IsFace point")
    if np.any(np.asarray(skin.point_data["IsFixed"], dtype=bool)):
        raise ValueError("convenience-array skin overlaps original fixed points")

    activation = np.asarray(skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    diag = np.asarray(skin.cell_data["SkinActivationInvDiag"], dtype=np.float64)
    expected_diag = activation[:, 0]
    if (
        activation.shape != (skin.n_cells, 3)
        or diag.shape != (skin.n_cells,)
        or not np.array_equal(diag, expected_diag)
    ):
        raise ValueError("SkinActivationInvDiag must exactly equal ActivationInv[:, 0]")
    diag_hash = _require_hash(
        "SkinActivationInvDiag",
        diag,
        expected_activation_diag_hash,
        dtype="<f8",
    )

    expected_values: dict[str, np.ndarray] = {}
    for name in INHERITED_DOMAIN_ONE_ARRAYS:
        expected_values[name] = np.ones(skin.n_cells, dtype=np.int8)
    for name in INHERITED_DOMAIN_ZERO_ARRAYS:
        expected_values[name] = np.zeros(skin.n_cells, dtype=np.int8)
    expected_values["TeethProximityTriangle"] = np.any(
        np.asarray(skin.point_data["IsTeeth"], dtype=bool)[triangles], axis=1
    ).astype(np.int8)
    expected_values["GingivaProximityTriangle"] = np.any(
        np.asarray(skin.point_data["IsGingiva"], dtype=bool)[triangles], axis=1
    ).astype(np.int8)

    expected_hashes = {
        **dict.fromkeys(
            INHERITED_DOMAIN_ONE_ARRAYS, EXPECTED_ARRAY_HASHES["inherited_int8_all_one"]
        ),
        **dict.fromkeys(
            INHERITED_DOMAIN_ZERO_ARRAYS,
            EXPECTED_ARRAY_HASHES["inherited_int8_all_zero"],
        ),
        "TeethProximityTriangle": EXPECTED_ARRAY_HASHES["teeth_proximity"],
        "GingivaProximityTriangle": EXPECTED_ARRAY_HASHES["inherited_int8_all_zero"],
    }
    inherited_records: dict[str, Any] = {}
    for name in sorted(INHERITED_INT8_ARRAYS):
        raw = np.asarray(skin.cell_data[name])
        expected = expected_values[name]
        if raw.dtype != np.dtype("i1") or raw.shape != expected.shape:
            raise ValueError(f"inherited convenience array {name} dtype/shape changed")
        if not np.array_equal(raw, expected):
            raise ValueError(f"inherited convenience array {name} semantics changed")
        if inherited_reference is not None and not np.array_equal(
            raw, np.asarray(inherited_reference.cell_data[name])
        ):
            raise ValueError(f"candidate convenience array {name} changed from source")
        array_hash = _require_hash(
            f"inherited convenience array {name}",
            raw,
            expected_hashes[name],
            dtype="i1",
        )
        inherited_records[name] = {
            "dtype": "i1",
            "shape": [skin.n_cells],
            "sha256_le_c": array_hash,
            "nonzero": int(np.count_nonzero(raw)),
            "exact_semantic_readback": True,
            "unchanged_from_corrected_source": inherited_reference is not None,
        }
    return {
        "complete": True,
        "SkinActivationInvDiag": {
            "dtype": "<f8",
            "shape": [skin.n_cells],
            "sha256_le_c": diag_hash,
            "exactly_equals": "ActivationInv[:, 0]",
        },
        "inherited_cell_arrays": inherited_records,
        "point_domain": {
            "all_IsFace": True,
            "any_IsFixed": False,
            "diagnostic_masks_rederived_from": ["IsTeeth", "IsGingiva"],
        },
    }


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
    _require_hash(
        "expanding triangle mask",
        expanding,
        EXPECTED_ARRAY_HASHES["expanding_mask"],
        dtype="u1",
    )
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
                "R=TargetArea/RestArea; fat_floor_skin=(R>1); "
                "rho_c020=0.98^2*clip(R,0.5,1); "
                "ActivationInv=[rho_c020^(-1/2)-1,"
                "rho_c020^(-1/2)-1,0]"
            ),
            "raw_ratio_sha256_le_f8": EXPECTED_ARRAY_HASHES["raw_ratio"],
            "clipped_ratio_sha256_le_f8": EXPECTED_ARRAY_HASHES["clipped_ratio"],
            "rho_c020_sha256_le_f8": EXPECTED_ARRAY_HASHES["rho_c020"],
            "activation_c020_sha256_le_f8": EXPECTED_ARRAY_HASHES["activation_c020"],
            "expanding_mask_sha256_u1": EXPECTED_ARRAY_HASHES["expanding_mask"],
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
        dtype = (
            "u1"
            if name in MASK_ARRAYS
            else ("i1" if name in INHERITED_INT8_ARRAYS else "<f8")
        )
        records[name] = _array_record(raw, dtype=dtype)
    return records


def _build_case_skin(
    source: pv.PolyData,
    fields: dict[str, np.ndarray],
    spec: CaseSpec,
) -> pv.PolyData:
    skin = source.copy(deep=True)
    expanding = np.asarray(fields["expanding"], dtype=bool)
    young = (
        np.where(expanding, FAT_E_MPA, SKIN_E_MPA)
        if spec.heterogeneous
        else np.full(skin.n_cells, SKIN_E_MPA)
    )
    if (
        not np.all(young > 0.0)
        or float(young.min()) != FAT_E_MPA
        or float(young.max()) != SKIN_E_MPA
        or int(np.count_nonzero(young == FAT_E_MPA)) != EXPECTED_EXPANDING_TRIANGLES
    ):
        raise RuntimeError(f"{spec.case_id} positive fat-modulus floor changed")
    rest_area = np.asarray(skin.cell_data["RestArea"], dtype=np.float64)
    _require_close(
        float(np.dot(young, rest_area) / rest_area.sum()),
        EXPECTED_FAT_FLOOR_AREA_WEIGHTED_E_MPA,
        name=f"{spec.case_id} area-weighted Young's modulus",
    )
    nu = np.full(skin.n_cells, SKIN_NU)
    lam = young * nu / (1.0 - np.square(nu))
    mu = young / (2.0 * (1.0 + nu))
    for name, values, expected_hash_key in (
        ("fat-floor Young's modulus", young, "fat_floor_young"),
        ("fat-floor plane-stress lambda", lam, "fat_floor_lambda"),
        ("fat-floor plane-stress mu", mu, "fat_floor_mu"),
    ):
        _require_hash(
            name,
            values,
            EXPECTED_ARRAY_HASHES[expected_hash_key],
            dtype="<f8",
        )
    if spec.c020:
        rho = np.asarray(fields["rho_c020"], dtype=np.float64)
        activation = np.asarray(fields["activation_c020"], dtype=np.float64)
    else:
        rho = np.ones(skin.n_cells, dtype=np.float64)
        activation = np.zeros((skin.n_cells, 3), dtype=np.float64)
        _require_hash(
            "p000 stress-free area ratio",
            rho,
            EXPECTED_ARRAY_HASHES["p000_stress_free_area_ratio"],
            dtype="<f8",
        )
        _require_hash(
            "p000 ActivationInv",
            activation,
            EXPECTED_ARRAY_HASHES["p000_activation"],
            dtype="<f8",
        )
    skin.cell_data["SkinYoungModulusMPa"] = young
    skin.cell_data["SkinPoissonRatio"] = nu
    skin.cell_data[LAMBDA.vtk] = lam
    skin.cell_data[MU.vtk] = mu
    skin.cell_data[FRACTION.vtk] = np.ones(skin.n_cells, dtype=np.float64)
    skin.cell_data[ACTIVATION_INV.vtk] = activation
    skin.cell_data["SkinActivationInvDiag"] = activation[:, 0].copy()
    skin.cell_data["TargetRestAreaRatio"] = fields["raw_ratio"]
    skin.cell_data["ClippedTargetRestAreaRatio"] = fields["clipped_ratio"]
    skin.cell_data["StressFreeAreaRatio"] = rho
    expanding_marker = expanding.astype(np.uint8)
    fat_floor_marker = expanding_marker.copy()
    c020_marker = np.full(skin.n_cells, int(spec.c020), dtype=np.uint8)
    _require_hash(
        "expanding marker",
        expanding_marker,
        EXPECTED_ARRAY_HASHES["expanding_mask"],
        dtype="u1",
    )
    _require_hash(
        "fat-floor marker",
        fat_floor_marker,
        EXPECTED_ARRAY_HASHES["expanding_mask"],
        dtype="u1",
    )
    _require_hash(
        "c020 marker",
        c020_marker,
        EXPECTED_ARRAY_HASHES["mask_all_one" if spec.c020 else "mask_all_zero"],
        dtype="u1",
    )
    skin.cell_data["ExpandingTriangle"] = expanding_marker
    skin.cell_data["SelectiveFatFloor"] = fat_floor_marker
    skin.cell_data["C020PrestrainEnabled"] = c020_marker
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
            "E=fat modulus (0.003 MPa) iff raw R>1, else E=0.2 MPa"
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
            "prestrain is stored and mechanically active on all IsFace triangles; "
            "expanding triangles retain the positive 0.003 MPa modulus floor"
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
    convenience_array_audit: dict[str, Any],
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
            "selective-efat003-where-raw-r-gt-1"
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
            "convenience_array_audit": convenience_array_audit,
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
    producer_provenance_before = _producer_source_provenance(PRODUCER)
    if (
        producer_provenance_before["executable_source"]["file_identity"]
        != producer_identity_before
    ):
        raise RuntimeError("producer executable identity readback changed")
    input_rows = {
        name: _require_identity(name, spec) for name, spec in INPUT_IDENTITIES.items()
    }
    baseline_inverse_rows = {
        name: _require_identity(f"H0P0 control {name}", spec)
        for name, spec in BASELINE_INVERSE_ARTIFACTS.items()
    }
    h0p1_control_rows = {
        name: _require_identity(f"H0P1 control {name}", spec)
        for name, spec in H0P1_CONTROL_ARTIFACTS.items()
    }
    v1_output_rows = {
        name: _require_identity(f"preserved v1 output {name}", spec)
        for name, spec in V1_OUTPUT_IDENTITIES.items()
    }
    corrected_manifest = _read_json(cfg.input_corrected_manifest)
    _verify_input_manifest(corrected_manifest)
    corrected = _read_polydata(cfg.input_corrected_skin, name="corrected H0P0 skin")
    driver = _read_polydata(
        cfg.input_raw_area_ratio_driver, name="raw-area-ratio driver skin"
    )
    rest_area = _validate_corrected_skin(corrected)
    corrected_convenience_audit = _audit_convenience_arrays(
        corrected,
        expected_activation_diag_hash=EXPECTED_ARRAY_HASHES["activation_diag_p000"],
    )
    mapped, mapping = _map_driver_cells(corrected, driver)
    fields, derivation = _derive_fields(driver, mapped, rest_area)

    cases: list[dict[str, Any]] = []
    for spec in CASES:
        path = OUTPUT_SKINS[spec.case_id]
        built = _build_case_skin(corrected, fields, spec)
        expected_diag_hash = EXPECTED_ARRAY_HASHES[
            "activation_diag_c020" if spec.c020 else "activation_diag_p000"
        ]
        built_convenience_audit = _audit_convenience_arrays(
            built,
            expected_activation_diag_hash=expected_diag_hash,
            inherited_reference=corrected,
        )
        expected_arrays = _material_array_records(built)
        readback = _write_skin_atomic(built, path)
        readback_convenience_audit = _audit_convenience_arrays(
            readback,
            expected_activation_diag_hash=expected_diag_hash,
            inherited_reference=corrected,
        )
        if readback_convenience_audit != built_convenience_audit:
            raise ValueError(
                f"{spec.case_id} convenience-array audit changed during readback"
            )
        readback_arrays = _material_array_records(readback)
        if readback_arrays != expected_arrays:
            raise ValueError(f"{spec.case_id} material-array readback records changed")
        cases.append(
            _case_row(
                spec,
                skin_path=path,
                skin=readback,
                arrays=readback_arrays,
                convenience_array_audit=readback_convenience_audit,
                generated=True,
                inverse_artifacts=None,
            )
        )

    if tuple(row["case_id"] for row in cases) != CASE_ORDER:
        raise RuntimeError("material case order changed")
    final_recheck = _recheck_inputs(
        {
            **input_rows,
            **{f"h0p0_{name}": row for name, row in baseline_inverse_rows.items()},
            **{f"h0p1_{name}": row for name, row in h0p1_control_rows.items()},
            **{f"v1_{name}": row for name, row in v1_output_rows.items()},
        }
    )
    table = _table(cases)
    _write_text_atomic(cfg.output_table, table)

    if _file_identity(PRODUCER) != producer_identity_before:
        raise RuntimeError("material producer changed during preparation")
    producer_provenance_after = _producer_source_provenance(PRODUCER)
    if producer_provenance_after != producer_provenance_before:
        raise RuntimeError("material producer provenance changed during preparation")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "producer": producer_provenance_before,
        "input_provenance": {
            "inputs": input_rows,
            "control_artifacts": {
                "H0P0": baseline_inverse_rows,
                "H0P1": h0p1_control_rows,
            },
            "corrected_manifest_contract_verified": True,
            "preserved_v1_outputs": v1_output_rows,
            "final_recheck": final_recheck,
        },
        "convenience_array_contract": {
            "corrected_source_audit": corrected_convenience_audit,
            "candidate_requirement": (
                "SkinActivationInvDiag exactly equals ActivationInv[:, 0]; inherited "
                "domain and diagnostic arrays retain their corrected-source values "
                "and rederived semantics"
            ),
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
            "fat_E_MPa": FAT_E_MPA,
            "selective_fat_floor_rule": (
                "E=fat modulus (0.003 MPa) iff raw TargetRestAreaRatio > 1; "
                "else E=0.2 MPa"
            ),
            "fat_floor_area_weighted_E_MPa": (EXPECTED_FAT_FLOOR_AREA_WEIGHTED_E_MPA),
        },
        "candidate_array_hashes_le_c": {
            key: EXPECTED_ARRAY_HASHES[key]
            for key in (
                "expanding_mask",
                "fat_floor_young",
                "fat_floor_lambda",
                "fat_floor_mu",
                "p000_stress_free_area_ratio",
                "p000_activation",
                "rho_c020",
                "activation_c020",
                "mask_all_zero",
                "mask_all_one",
                "activation_diag_c020",
                "activation_diag_p000",
                "inherited_int8_all_zero",
                "inherited_int8_all_one",
                "teeth_proximity",
            )
        },
        "anatomy_material_contract": {
            "skin": {
                "domain": "all-vertex IsFace filtered PolyData",
                "E_MPa": "case-dependent: 0.003 or 0.2",
                "nu": SKIN_NU,
                "thickness_m": SKIN_THICKNESS_M,
                "lame_conversion": LAME_CONVERSION,
                "energy_measure": ENERGY_MEASURE,
            },
            "volume": {
                "lame_conversion": VOLUME_LAME_CONVERSION,
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
        "controls": [
            {
                "case_id": "H0P0",
                "role": "exact-pinned homogeneous p000 control; not rerun",
                "young_modulus_mode": "homogeneous-e0200",
                "prestrain_mode": "p000",
                "skin": input_rows["corrected_skin"],
                "inverse_artifacts": baseline_inverse_rows,
                "reused_not_rerun": True,
            },
            {
                "case_id": "H0P1",
                "role": "exact-pinned homogeneous c020 control; not rerun",
                "young_modulus_mode": "homogeneous-e0200",
                "prestrain_mode": "c020-raw-area-ratio-floor-050",
                "skin": h0p1_control_rows["skin"],
                "inverse_artifacts": {
                    name: row
                    for name, row in h0p1_control_rows.items()
                    if name != "skin"
                },
                "reused_not_rerun": True,
            },
        ],
        "control_order": list(CONTROL_ORDER),
        "case_order": list(CASE_ORDER),
        "cases": cases,
        "output_contract": {
            "manifest_path": str(OUTPUT_MANIFEST),
            "table_path": str(OUTPUT_TABLE),
            "candidate_root": str(OUTPUT_ROOT),
            "generated_candidate_vtps": 2,
            "controls_reused_by_identity": True,
            "v1_outputs_preserved_by_identity": True,
            "overwrite_policy": "refuse any existing final or temporary output",
        },
        "approval": {
            "material_preparation_static_review": (
                EXECUTION_APPROVED_AFTER_STATIC_REVIEW
            ),
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
            "material/generated_vtps": 2,
            "material/fat_floor_triangles": EXPECTED_EXPANDING_TRIANGLES,
            "material/fat_floor_area_fraction": EXPECTED_EXPANDING_AREA_FRACTION,
        }
    )
    logger.info("Wrote reviewed fat-floor material manifest to %s", cfg.output_manifest)


if __name__ == "__main__":
    cherries.main(main)
