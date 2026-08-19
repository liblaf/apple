# ruff: noqa: C901, EM101, EM102, PERF401, PLR0912, PLR0915, RUF046, SLF001, TRY003

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries, melon
from liblaf.apple.common import (
    ACTIVATION_INV,
    FIXED_MASK,
    FIXED_VALUE,
    FRACTION,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DESIGN = "corrected-isface-fixed-activation-c020-prestrain-replay"
GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
PRODUCER = Path(__file__).resolve()

PREPARED_MESH = (
    REPO_ROOT
    / "exp/2026/06/17/human-face-smile-prestrain-v2/data/10-human-face-prepared.vtu"
)
CORRECTED_SKIN = (
    REPO_ROOT / "exp/2026/08/18/human-face-smile-plane-stress-skin/data/"
    "10-corrected-baseline/skin-isface-e0200-p000.vtp"
)
DRIVER_SKIN = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/data/"
    "10-material-candidates/skin-e100-p000.vtp"
)
BASELINE_STEM = (
    "20-human-face-smile-skin-no-prestrain-lr3-corrected-isface-e0200-p000-screen"
)
BASELINE_DATA_DIR = REPO_ROOT / "exp/2026/08/18/human-face-smile-plane-stress-skin/data"
BASELINE_RESULT = BASELINE_DATA_DIR / f"{BASELINE_STEM}.vtu"
BASELINE_SUMMARY = BASELINE_DATA_DIR / f"{BASELINE_STEM}-summary-final.json"
BASELINE_TARGET = BASELINE_DATA_DIR / f"{BASELINE_STEM}-target.vtu"
REVIEWED_PROBE = (
    REPO_ROOT / "exp/2026/08/18/human-face-smile-plane-stress-skin/src/"
    "15-forward-domain-conversion-probe.py"
)
REVIEWED_PROBE_SRC = REVIEWED_PROBE.parent
REVIEWED_REFERENCE = REVIEWED_PROBE_SRC / "_reference.py"
RUNTIME_METRICS = (
    REPO_ROOT
    / "exp/2026/06/17/human-face-smile-prestrain-v2/src/_human_face_metrics.py"
)
RUNTIME_COMPAT_CONFIG = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/src/"
    "_human_face_config.py"
)

INPUT_IDENTITIES = {
    "prepared_mesh": {
        "path": PREPARED_MESH,
        "size_bytes": 76_792_914,
        "sha256": "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563",
    },
    "corrected_skin": {
        "path": CORRECTED_SKIN,
        "size_bytes": 1_138_550,
        "sha256": "4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f",
    },
    "driver_skin": {
        "path": DRIVER_SKIN,
        "size_bytes": 38_742_137,
        "sha256": "ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f",
    },
    "baseline_result": {
        "path": BASELINE_RESULT,
        "size_bytes": 147_657_021,
        "sha256": "c6a0b183675ffb3ec537c1153544b041acd7aa0fdd5216c0cf9a50022d52b0a4",
    },
    "baseline_summary": {
        "path": BASELINE_SUMMARY,
        "size_bytes": 126_540,
        "sha256": "575ebcbd7152a256917c2a11a9bf9bef9046f00f9831e18adc86d41645be1856",
    },
    "baseline_target": {
        "path": BASELINE_TARGET,
        "size_bytes": 84_419_492,
        "sha256": "89ec02dfd87330f7dc1d303639893f7698ef2e6098480c4e39fa2ad94240206c",
    },
    "reviewed_probe": {
        "path": REVIEWED_PROBE,
        "size_bytes": 87_717,
        "sha256": "741d3f3db966f8b1e25b389a8734176fb6991a6872e6f8a1a8b875bd3ec5e2f5",
    },
    "reviewed_reference": {
        "path": REVIEWED_REFERENCE,
        "size_bytes": 4_108,
        "sha256": "470db910d6bec9ec81e06b5b46512781a188c252683b44b57b539ddb63295615",
    },
    "runtime_metrics": {
        "path": RUNTIME_METRICS,
        "size_bytes": 3_775,
        "sha256": "1407d2988444b31332f2688c6535eca5db58b5be31d63fae6abd6bf8bf78e0c1",
    },
    "runtime_compat_config": {
        "path": RUNTIME_COMPAT_CONFIG,
        "size_bytes": 2_992,
        "sha256": "fcd7757486c3f0664816a6595e17af27a87ffec1c9c9e24b18908506b444ffeb",
    },
}

EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_ACTIVE_TETS = 288_235
EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_SKIN_AREA_M2 = 0.04287998059707303
EXPECTED_CUT_VERTICES = 6_980
EXPECTED_MODEL_FIXED_VERTICES = 33_636
EXPECTED_MODEL_FIXED_DOFS = 100_908
SKIN_LAME_CONVERSION = (
    "thin-membrane plane-stress reduction: "
    "lambda = E * nu / (1 - nu**2); "
    "mu = E / (2 * (1 + nu))"
)

EXPECTED_ARRAY_HASHES = {
    "mesh_points": "ec9544035eeb2eee2b733f16584a17a1873a0622855905c8d2a98113aab44a74",
    "mesh_cells": "61678752f43b9bbd641602c71fb79ee802d4c6753d1adebc6647b2ff0a9bbab3",
    "mesh_celltypes": "9a7caed190d749ea866232198a7902bb4eacb72690aed702a1e4683d208aa342",
    "target_displacement": "823d503d67916988bad9aba52efc7303ee943bc7c9206112f2b3ee8b5e2ff375",
    "loss_mask": "7f3d956377de1fccb5be08e7c8809ad62ae5f770b61a09a238cdde724a9a4d68",
    "baseline_displacement": "f8ca27d820ff1f4b7afb734d917c9ec1292cd26ab96fc93090277dcc017268fb",
    "fixed_activation": "4494f1eca2ce6f14c2e87a184d2227c080fbfa4594e7d6e96ced0c0c35c981de",
    "rest_area": "5a7b8eb9861fa509212afd610c60183f894b80db8ded53d22f3f9045bc6889de",
    "raw_ratio": "da98a6f48694eed30b1683ccf5a1f02fd67c87393b30003d07c52a7fa25bc606",
    "clipped_ratio": "aaf87f8d68485136c0ce09d113ce09de481654613c7d50c80ac2becb40e86e1e",
    "rho_full": "d631449a1db997e6ce2eac9d3276ff8a23451461350f917026f19e9d50cc89f1",
}

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_ALPHA_HASHES = {
    0.0: (
        "aa74d25a4afece1f232101f98e5fda8177935c3e833ae578be636d9ff42294c2",
        "051fe4599913dc590cb39aa79f7bb51578efc3323cd9c0a337be804d12d8f224",
    ),
    0.25: (
        "91450614bbdd5cb1ee8b8b4588cfc870558886c5ee30eaf35837687318cb8842",
        "14b62c9f14ecd3a553a27583b1c035d8ae4f0fd7c785afdd4095d2c776225e20",
    ),
    0.5: (
        "6c8ec3bc8ed846086c6951478ee913b98dc53c4a3daee0e8e4a5606b9193a4d4",
        "8db4f5203811b445be947a117d748a3a461dc6077b87f5969c435a1464ea147d",
    ),
    0.75: (
        "062d9dfce22c8c7a409fb5b3f2dd32cb9e51b767f36a7aeb32fc40a4d6ed461a",
        "5ca3cefaae0f83c9ff5749d306aea799329f2c8413acee5d13dc861c18f4459e",
    ),
    1.0: (
        "d631449a1db997e6ce2eac9d3276ff8a23451461350f917026f19e9d50cc89f1",
        "1366a17e86a2b182dd9b15512b9dc0664c869e416af7b5e591fbfb347fd53d55",
    ),
}

LINEAR_TIGHTENING = 0.02
LENGTH_FACTOR = 0.98
AREA_RATIO_FLOOR = 0.5
REPLAY_DELTA_FRACTION_OF_TARGET_TOL = 1.0e-3
EXPECTED_LOSS_TARGET_RMS_M = 0.005310139062299789
EXPECTED_ISFACE_TARGET_RMS_M = 0.005310654682438851
OUTPUT_ROOT_NAME = "10-fixed-activation-prestrain-replay"
OUTPUT_SUMMARY = GROUP_DIR / "data/10-fixed-activation-prestrain-replay-summary.json"
OUTPUT_TABLE = GROUP_DIR / "data/10-fixed-activation-prestrain-replay-table.md"
OUTPUT_ROOT = GROUP_DIR / "data" / OUTPUT_ROOT_NAME

# Static-review blocker.  Execution requires an explicit follow-up review that changes
# this source constant; it cannot be bypassed with a CLI flag.
EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True
APPROVAL_BLOCKER = (
    "NO-GO: c020 fixed-activation replay awaits static producer/analyzer review; "
    "do not execute until this source-level blocker is explicitly changed"
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_corrected_skin: Path = cherries.input(CORRECTED_SKIN)
    input_driver_skin: Path = cherries.input(DRIVER_SKIN)
    input_baseline_result: Path = cherries.input(BASELINE_RESULT)
    input_baseline_summary: Path = cherries.input(BASELINE_SUMMARY)
    input_baseline_target: Path = cherries.input(BASELINE_TARGET)
    output_summary: Path = cherries.output(
        "10-fixed-activation-prestrain-replay-summary.json", mkdir=True
    )
    output_table: Path = cherries.output(
        "10-fixed-activation-prestrain-replay-table.md", mkdir=True
    )
    output_root_name: str = OUTPUT_ROOT_NAME
    require_solver_success: bool = True
    replay_delta_fraction_of_target_tol: float = REPLAY_DELTA_FRACTION_OF_TARGET_TOL


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    path_kind: str
    alpha: float
    seed_case_id: str | None


CONTINUATION_CASES = tuple(
    CaseSpec(
        case_id=f"c020-continuation-alpha-{int(round(alpha * 100)):03d}",
        path_kind="continuation",
        alpha=alpha,
        seed_case_id=None
        if alpha == 0.0
        else f"c020-continuation-alpha-{int(round((alpha - 0.25) * 100)):03d}",
    )
    for alpha in ALPHAS
)
DIRECT_CASE = CaseSpec(
    case_id="c020-direct-alpha-100",
    path_kind="direct",
    alpha=1.0,
    seed_case_id=None,
)
CASE_ORDER = (*CONTINUATION_CASES, DIRECT_CASE)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _raw_sha256(array: np.ndarray, *, dtype: str) -> str:
    values = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _require_hash(name: str, array: np.ndarray, expected: str, *, dtype: str) -> str:
    actual = _raw_sha256(array, dtype=dtype)
    if actual != expected:
        raise ValueError(f"{name} hash changed: {actual} != {expected}")
    return actual


def _require_identity(name: str) -> dict[str, int | str]:
    expected = INPUT_IDENTITIES[name]
    path = Path(expected["path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {name}: {path}")
    actual = _file_identity(path)
    wanted = {
        "size_bytes": int(expected["size_bytes"]),
        "sha256": str(expected["sha256"]),
    }
    if actual != wanted:
        raise ValueError(f"{name} identity changed: {actual} != {wanted}")
    return {"path": str(path), **actual}


def _recheck_file_rows(rows: list[dict[str, Any]], *, context: str) -> dict[str, Any]:
    rechecked: list[dict[str, Any]] = []
    for row in rows:
        path = Path(str(row["path"]))
        expected = {
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }
        if not path.is_file():
            raise FileNotFoundError(f"{context} disappeared during execution: {path}")
        actual = _file_identity(path)
        if actual != expected:
            raise RuntimeError(
                f"{context} changed during execution: {path}: {actual} != {expected}"
            )
        rechecked.append({**row, "unchanged_through_all_solves": True})
    return {"all_unchanged": True, "files": rechecked}


def _validate_config(cfg: Config) -> None:
    exact_paths = {
        "input_mesh": (cfg.input_mesh, PREPARED_MESH),
        "input_corrected_skin": (cfg.input_corrected_skin, CORRECTED_SKIN),
        "input_driver_skin": (cfg.input_driver_skin, DRIVER_SKIN),
        "input_baseline_result": (cfg.input_baseline_result, BASELINE_RESULT),
        "input_baseline_summary": (cfg.input_baseline_summary, BASELINE_SUMMARY),
        "input_baseline_target": (cfg.input_baseline_target, BASELINE_TARGET),
        "output_summary": (cfg.output_summary, OUTPUT_SUMMARY),
        "output_table": (cfg.output_table, OUTPUT_TABLE),
    }
    changed = [
        f"{name}: {actual} != {expected}"
        for name, (actual, expected) in exact_paths.items()
        if actual.resolve() != expected.resolve()
    ]
    if changed:
        raise ValueError("reviewed replay paths changed: " + "; ".join(changed))
    if cfg.output_root_name != OUTPUT_ROOT_NAME:
        raise ValueError(f"output_root_name must remain {OUTPUT_ROOT_NAME!r}")
    if cfg.require_solver_success is not True:
        raise ValueError("require_solver_success must remain true")
    if not math.isclose(
        cfg.replay_delta_fraction_of_target_tol,
        REPLAY_DELTA_FRACTION_OF_TARGET_TOL,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("alpha0 replay tolerance is not CLI-adjustable")
    stale = [
        path
        for path in (
            OUTPUT_SUMMARY,
            OUTPUT_TABLE,
            OUTPUT_ROOT,
            _temporary_path(OUTPUT_SUMMARY),
            _temporary_path(OUTPUT_TABLE),
        )
        if path.exists()
    ]
    if stale:
        raise FileExistsError(
            "refusing to overwrite replay outputs: " + ", ".join(map(str, stale))
        )
    if not EXECUTION_APPROVED_AFTER_STATIC_REVIEW:
        raise RuntimeError(APPROVAL_BLOCKER)


def _load_reviewed_probe() -> ModuleType:
    expected = INPUT_IDENTITIES["reviewed_probe"]
    actual = _file_sha256(REVIEWED_PROBE)
    if actual != expected["sha256"]:
        raise ValueError("reviewed forward helper changed before import")
    source_dir = str(REVIEWED_PROBE_SRC)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    spec = importlib.util.spec_from_file_location(
        "_fixed_activation_replay_reviewed_probe", REVIEWED_PROBE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reviewed probe: {REVIEWED_PROBE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for module_name, expected_path in (
        ("_reference", REVIEWED_REFERENCE),
        ("_human_face_metrics", RUNTIME_METRICS),
        ("_human_face_config", RUNTIME_COMPAT_CONFIG),
    ):
        imported = sys.modules.get(module_name)
        imported_file = (
            None if imported is None else getattr(imported, "__file__", None)
        )
        if (
            imported_file is None
            or Path(imported_file).resolve() != expected_path.resolve()
        ):
            raise ImportError(
                f"reviewed probe imported {module_name} from {imported_file}, "
                f"expected {expected_path}"
            )
    return module


def _triangles(mesh: pv.PolyData, *, name: str) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if mesh.n_cells == 0 or faces.size != 4 * mesh.n_cells:
        raise ValueError(f"{name} is not a non-empty triangle-only PolyData")
    encoded = faces.reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        raise ValueError(f"{name} contains a non-triangle face")
    return encoded[:, 1:]


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
    driver_keys = _triangle_keys(driver, name="raw-ratio driver skin")
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
    return mapped, {
        "method": "sorted GlobalPointId triangle keys",
        "corrected_triangles": int(corrected.n_cells),
        "driver_triangles": int(driver.n_cells),
        "mapped_unique_driver_triangles": int(np.unique(mapped).size),
        "corrected_triangle_keys_sha256_le_i8": _raw_sha256(
            corrected_keys, dtype="<i8"
        ),
        "mapped_driver_cell_indices_sha256_le_i8": _raw_sha256(mapped, dtype="<i8"),
        "exact_readback": True,
    }


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _validate_mesh_topology(mesh: pv.UnstructuredGrid) -> None:
    if mesh.n_points != EXPECTED_POINTS or mesh.n_cells != EXPECTED_TETS:
        raise ValueError(
            f"prepared mesh dimensions changed: {mesh.n_points}/{mesh.n_cells}"
        )
    _require_hash(
        "prepared points",
        np.asarray(mesh.points),
        EXPECTED_ARRAY_HASHES["mesh_points"],
        dtype="<f8",
    )
    _require_hash(
        "prepared cells",
        np.asarray(mesh.cells),
        EXPECTED_ARRAY_HASHES["mesh_cells"],
        dtype="<i8",
    )
    _require_hash(
        "prepared cell types",
        np.asarray(mesh.celltypes),
        EXPECTED_ARRAY_HASHES["mesh_celltypes"],
        dtype="u1",
    )
    canonical_ids = np.arange(mesh.n_points, dtype=np.int64)
    if GLOBAL_POINT_ID.vtk in mesh.point_data and not np.array_equal(
        np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64),
        canonical_ids,
    ):
        raise ValueError("prepared mesh has a non-canonical GlobalPointId field")


def _validate_corrected_skin(skin: pv.PolyData) -> None:
    if skin.n_points != EXPECTED_SKIN_POINTS or skin.n_cells != EXPECTED_SKIN_TRIANGLES:
        raise ValueError(
            f"corrected skin dimensions changed: {skin.n_points}/{skin.n_cells}"
        )
    triangles = _triangles(skin, name="corrected skin")
    points = np.asarray(skin.points, dtype=np.float64)
    area = 0.5 * np.linalg.norm(
        np.cross(
            points[triangles[:, 1]] - points[triangles[:, 0]],
            points[triangles[:, 2]] - points[triangles[:, 0]],
        ),
        axis=1,
    )
    rest_area = np.asarray(skin.cell_data["RestArea"], dtype=np.float64)
    if not np.array_equal(area, rest_area):
        raise ValueError("corrected skin RestArea differs from exact live geometry")
    _require_hash(
        "corrected skin RestArea",
        rest_area,
        EXPECTED_ARRAY_HASHES["rest_area"],
        dtype="<f8",
    )
    if not math.isclose(
        float(rest_area.sum()), EXPECTED_SKIN_AREA_M2, rel_tol=1.0e-12, abs_tol=1.0e-15
    ):
        raise ValueError("corrected skin area changed")
    required = {
        "SkinYoungModulusMPa",
        "SkinPoissonRatio",
        LAMBDA.vtk,
        MU.vtk,
        FRACTION.vtk,
        ACTIVATION_INV.vtk,
    }
    missing = sorted(required - set(skin.cell_data))
    if missing:
        raise KeyError(f"corrected skin is missing material fields: {missing}")
    young = np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64)
    nu = np.asarray(skin.cell_data["SkinPoissonRatio"], dtype=np.float64)
    expected_lambda = young * nu / (1.0 - np.square(nu))
    expected_mu = young / (2.0 * (1.0 + nu))
    for name, actual, expected in (
        ("E", young, np.full(skin.n_cells, 0.2)),
        ("nu", nu, np.full(skin.n_cells, 0.49)),
        ("Lambda", np.asarray(skin.cell_data[LAMBDA.vtk]), expected_lambda),
        ("Mu", np.asarray(skin.cell_data[MU.vtk]), expected_mu),
        ("Fraction", np.asarray(skin.cell_data[FRACTION.vtk]), np.ones(skin.n_cells)),
    ):
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=1.0e-13, atol=1.0e-14
        ):
            raise ValueError(f"corrected skin {name} is not homogeneous plane stress")
    activation = np.asarray(skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    if activation.shape != (skin.n_cells, 3) or not np.array_equal(
        activation, np.zeros_like(activation)
    ):
        raise ValueError("corrected source skin is not exact p000")
    if not np.all(np.asarray(skin.point_data["IsFace"], dtype=bool)):
        raise ValueError("corrected skin contains a non-IsFace point")


def _validate_baseline(
    mesh: pv.UnstructuredGrid,
    result: pv.UnstructuredGrid,
    target_mesh: pv.UnstructuredGrid,
    summary: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    for name, candidate in (
        ("baseline result", result),
        ("baseline target", target_mesh),
    ):
        if candidate.n_points != mesh.n_points or candidate.n_cells != mesh.n_cells:
            raise ValueError(f"{name} dimensions differ from prepared mesh")
        if not np.array_equal(candidate.points, mesh.points):
            raise ValueError(f"{name} rest points differ from prepared mesh")
        if not np.array_equal(candidate.cells, mesh.cells) or not np.array_equal(
            candidate.celltypes, mesh.celltypes
        ):
            raise ValueError(f"{name} topology differs from prepared mesh")
    canonical_ids = np.arange(mesh.n_points, dtype=np.int64)
    if not np.array_equal(
        np.asarray(result.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64),
        canonical_ids,
    ):
        raise ValueError("baseline result GlobalPointId is not canonical")
    target = np.asarray(target_mesh.point_data["TargetDisplacement"], dtype=np.float64)
    loss_mask = np.asarray(target_mesh.point_data["LossMask"], dtype=bool)
    if target.shape != (mesh.n_points, 3) or loss_mask.shape != (mesh.n_points,):
        raise ValueError("baseline target arrays are malformed")
    _require_hash(
        "target displacement",
        target,
        EXPECTED_ARRAY_HASHES["target_displacement"],
        dtype="<f8",
    )
    _require_hash(
        "loss mask", loss_mask, EXPECTED_ARRAY_HASHES["loss_mask"], dtype="u1"
    )
    if not np.array_equal(
        np.asarray(result.point_data["TargetDisplacement"], dtype=np.float64), target
    ) or not np.array_equal(
        np.asarray(result.point_data["LossMask"], dtype=bool), loss_mask
    ):
        raise ValueError("baseline result and target artifact disagree")
    displacement = np.asarray(result.point_data["Displacement"], dtype=np.float64)
    activation = np.asarray(result.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    recovered = np.asarray(result.cell_data["RecoveredActivationInv"], dtype=np.float64)
    if displacement.shape != (mesh.n_points, 3) or not np.isfinite(displacement).all():
        raise ValueError("baseline displacement is malformed or non-finite")
    if activation.shape != (mesh.n_cells, 6) or not np.isfinite(activation).all():
        raise ValueError("baseline muscle ActivationInv is malformed or non-finite")
    if not np.array_equal(activation, recovered):
        raise ValueError("baseline ActivationInv and RecoveredActivationInv differ")
    displacement_hash = _require_hash(
        "baseline displacement",
        displacement,
        EXPECTED_ARRAY_HASHES["baseline_displacement"],
        dtype="<f8",
    )
    activation_hash = _require_hash(
        "fixed muscle activation",
        activation,
        EXPECTED_ARRAY_HASHES["fixed_activation"],
        dtype="<f8",
    )
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    if int(active.sum()) != EXPECTED_ACTIVE_TETS or np.any(activation[~active] != 0.0):
        raise ValueError("baseline activation support differs from ActivationMask")
    is_fixed = np.asarray(result.point_data["IsFixed"], dtype=bool)
    fixed_mask = np.asarray(result.point_data[FIXED_MASK.vtk], dtype=bool)
    fixed_value = np.asarray(result.point_data[FIXED_VALUE.vtk], dtype=np.float64)
    cut = np.asarray(result.point_data["ArtificialCutIncident"], dtype=bool)
    if int(cut.sum()) != EXPECTED_CUT_VERTICES:
        raise ValueError("baseline artificial-cut vertex count changed")
    if (
        int(is_fixed.sum()) != EXPECTED_MODEL_FIXED_VERTICES
        or int(fixed_mask.sum()) != EXPECTED_MODEL_FIXED_DOFS
    ):
        raise ValueError("baseline hard-fixed model counts changed")
    if not np.array_equal(fixed_mask, np.repeat(is_fixed[:, None], 3, axis=1)):
        raise ValueError("baseline FixedMask and IsFixed differ")
    if np.any(fixed_value[is_fixed] != 0.0) or np.any(displacement[is_fixed] != 0.0):
        raise ValueError("baseline fixed values/displacements are not exact zero")
    expected_summary = {
        "best/step": 40,
        "final/step": 40.0,
        "history/frames": 41,
        "n_points": EXPECTED_POINTS,
        "n_tets": EXPECTED_TETS,
        "n_active_tets": EXPECTED_ACTIVE_TETS,
        "n_skin_triangles": EXPECTED_SKIN_TRIANGLES,
        "cut_boundary/total_fixed_vertices": EXPECTED_CUT_VERTICES,
        "cut_boundary/model_total_fixed_vertices": EXPECTED_MODEL_FIXED_VERTICES,
        "cut_boundary/model_total_fixed_dofs": EXPECTED_MODEL_FIXED_DOFS,
        "material/skin_lame_conversion": SKIN_LAME_CONVERSION,
        "material/skin_koiter_energy_measure": "fixed original reference area",
    }
    changed = {
        key: (summary.get(key), expected)
        for key, expected in expected_summary.items()
        if summary.get(key) != expected
    }
    if changed:
        raise ValueError(f"baseline step-40 provenance changed: {changed}")
    if (
        summary.get("artifact/result_sha256")
        != INPUT_IDENTITIES["baseline_result"]["sha256"]
    ):
        raise ValueError("baseline summary result hash differs from pinned artifact")
    mesh.point_data[GLOBAL_POINT_ID.vtk] = canonical_ids
    return (
        target,
        loss_mask,
        activation,
        displacement,
        {
            "fixed_activation_sha256_le_f8": activation_hash,
            "baseline_displacement_sha256_le_f8": displacement_hash,
            "best_step": 40,
            "final_step": 40,
            "best_equals_terminal": True,
        },
    )


def _derive_prestrain_basis(
    corrected: pv.PolyData, driver: pv.PolyData
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mapped, mapping = _map_driver_cells(corrected, driver)
    raw_ratio = np.asarray(driver.cell_data["TargetRestAreaRatio"], dtype=np.float64)[
        mapped
    ]
    if raw_ratio.shape != (corrected.n_cells,) or not np.isfinite(raw_ratio).all():
        raise ValueError("mapped raw target/rest area ratio is malformed")
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
        "clipped raw area ratio",
        clipped,
        EXPECTED_ARRAY_HASHES["clipped_ratio"],
        dtype="<f8",
    )
    rho_full = np.square(LENGTH_FACTOR) * clipped
    _require_hash(
        "c020 full natural-area ratio",
        rho_full,
        EXPECTED_ARRAY_HASHES["rho_full"],
        dtype="<f8",
    )
    area = np.asarray(corrected.cell_data["RestArea"], dtype=np.float64)
    clamped = raw_ratio < AREA_RATIO_FLOOR
    contraction = raw_ratio < 1.0
    return (
        raw_ratio,
        rho_full,
        {
            **mapping,
            "formula": (
                "rho_full=(1-c)^2*clip(TargetArea/RestArea,floor,1); "
                "rho_alpha=np.power(rho_full,alpha); "
                "diag=1/sqrt(rho_alpha)-1; ActivationInv=[diag,diag,0]"
            ),
            "alpha_algorithm": (
                "numpy.power(rho_full, alpha), then numpy.reciprocal(numpy.sqrt(rho_alpha))"
            ),
            "linear_tightening": LINEAR_TIGHTENING,
            "length_factor": LENGTH_FACTOR,
            "uniform_natural_area_ratio": float(np.square(LENGTH_FACTOR)),
            "area_ratio_floor": AREA_RATIO_FLOOR,
            "raw_ratio_sha256_le_f8": EXPECTED_ARRAY_HASHES["raw_ratio"],
            "clipped_ratio_sha256_le_f8": EXPECTED_ARRAY_HASHES["clipped_ratio"],
            "rho_full_sha256_le_f8": EXPECTED_ARRAY_HASHES["rho_full"],
            "floor_clamped_triangles": int(clamped.sum()),
            "floor_clamped_rest_area_fraction": float(area[clamped].sum() / area.sum()),
            "contraction_triangles": int(contraction.sum()),
            "contraction_rest_area_fraction": float(
                area[contraction].sum() / area.sum()
            ),
        },
    )


def _skin_for_alpha(
    source: pv.PolyData,
    raw_ratio: np.ndarray,
    rho_full: np.ndarray,
    alpha: float,
) -> tuple[pv.PolyData, dict[str, Any]]:
    if alpha not in EXPECTED_ALPHA_HASHES:
        raise ValueError(f"alpha {alpha} is outside the reviewed grid {ALPHAS}")
    rho_alpha = np.power(rho_full, alpha)
    diag = np.reciprocal(np.sqrt(rho_alpha)) - 1.0
    activation = np.stack((diag, diag, np.zeros_like(diag)), axis=1)
    expected_rho_hash, expected_activation_hash = EXPECTED_ALPHA_HASHES[alpha]
    rho_hash = _require_hash(
        f"alpha={alpha:g} natural-area ratio",
        rho_alpha,
        expected_rho_hash,
        dtype="<f8",
    )
    activation_hash = _require_hash(
        f"alpha={alpha:g} skin ActivationInv",
        activation,
        expected_activation_hash,
        dtype="<f8",
    )
    skin = source.copy(deep=True)
    before = {
        name: np.asarray(skin.cell_data[name]).copy()
        for name in (
            "RestArea",
            "SkinYoungModulusMPa",
            "SkinPoissonRatio",
            LAMBDA.vtk,
            MU.vtk,
            FRACTION.vtk,
        )
    }
    skin.cell_data[ACTIVATION_INV.vtk] = activation
    skin.cell_data["SkinActivationInvDiag"] = diag
    skin.cell_data["TargetRestAreaRatio"] = raw_ratio
    skin.cell_data["PrestrainNaturalAreaRatioFull"] = rho_full
    skin.cell_data["PrestrainNaturalAreaRatio"] = rho_alpha
    skin.cell_data["StressFreeAreaRatio"] = rho_alpha
    skin.cell_data["PrestrainAlpha"] = np.full(skin.n_cells, alpha)
    for name, expected in before.items():
        if not np.array_equal(np.asarray(skin.cell_data[name]), expected):
            raise ValueError(f"alpha={alpha:g} unexpectedly changed skin {name}")
    if not np.array_equal(
        np.asarray(skin.points), np.asarray(source.points)
    ) or not np.array_equal(np.asarray(skin.faces), np.asarray(source.faces)):
        raise ValueError(f"alpha={alpha:g} changed corrected skin geometry/topology")
    area = np.asarray(skin.cell_data["RestArea"], dtype=np.float64)
    return skin, {
        "skin/alpha": alpha,
        "skin/rho_min": float(rho_alpha.min()),
        "skin/rho_max": float(rho_alpha.max()),
        "skin/rho_area_weighted_mean": float(np.average(rho_alpha, weights=area)),
        "skin/rho_sha256_le_f8": rho_hash,
        "skin/activation_inv_min": float(activation.min()),
        "skin/activation_inv_max": float(activation.max()),
        "skin/activation_diag_area_weighted_mean": float(
            np.average(diag, weights=area)
        ),
        "skin/activation_diag_area_weighted_rms": float(
            np.sqrt(np.average(np.square(diag), weights=area))
        ),
        "skin/activation_inv_sha256_le_f8": activation_hash,
        "skin/E_MPa": 0.2,
        "skin/nu": 0.49,
        "skin/lame_conversion": SKIN_LAME_CONVERSION,
        "skin/domain": "all-vertex IsFace filtered PolyData",
        "skin/triangles": int(skin.n_cells),
        "skin/energy_measure": "fixed original reference area",
    }


def _case_dir(spec: CaseSpec) -> Path:
    alpha_dir = f"alpha-{int(round(spec.alpha * 100)):03d}"
    return OUTPUT_ROOT / "c020" / spec.path_kind / alpha_dir


def _case_paths(spec: CaseSpec) -> dict[str, Path]:
    root = _case_dir(spec)
    return {
        "root": root,
        "result": root / "result.vtu",
        "skin": root / "skin.vtp",
        "summary": root / "forward-summary.json",
    }


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _write_mesh_atomic(mesh: pv.DataSet, path: Path) -> None:
    temporary = _temporary_path(path)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing stale mesh output: {path} or {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    melon.save(mesh, temporary)
    temporary.replace(path)


def _write_json_atomic(value: dict[str, Any], path: Path) -> None:
    temporary = _temporary_path(path)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing stale JSON output: {path} or {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _rms(values: np.ndarray) -> float:
    return float(np.linalg.norm(values) / math.sqrt(values.shape[0]))


def _target_metrics(
    *,
    displacement: np.ndarray,
    target: np.ndarray,
    loss_mask: np.ndarray,
    face_ids: np.ndarray,
) -> dict[str, float | int]:
    residual = displacement - target
    return {
        "target/loss_points": int(loss_mask.sum()),
        "target/isface_points": int(face_ids.size),
        "target/loss_mask_error_rms_m": _rms(residual[loss_mask]),
        "target/loss_mask_error_rms_mm": 1000.0 * _rms(residual[loss_mask]),
        "target/isface_error_rms_m": _rms(residual[face_ids]),
        "target/isface_error_rms_mm": 1000.0 * _rms(residual[face_ids]),
    }


def _validate_live_materials(
    *,
    probe: ModuleType,
    forward: Any,
    mesh: pv.UnstructuredGrid,
    skin: pv.PolyData,
    fixed_activation: np.ndarray,
) -> dict[str, Any]:
    materials = forward.model.get_materials()
    if set(materials) != {"aponeurosis", "fat", "muscle", "skin"}:
        raise ValueError(f"forward potentials changed: {sorted(materials)}")
    volume_metrics: dict[str, Any] = {}
    volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    for name, young, nu, fraction_name in (
        (
            "aponeurosis",
            probe.APONEUROSIS_E,
            probe.APONEUROSIS_NU,
            probe.APONEUROSIS_FRACTION,
        ),
        ("fat", probe.FAT_E, probe.FAT_NU, probe.FAT_FRACTION),
        ("muscle", probe.MUSCLE_E, probe.MUSCLE_NU, probe.MUSCLE_FRACTION),
    ):
        expected_lambda, expected_mu = probe._volume_lambda_mu(young, nu)
        live_lambda = probe.to_numpy(materials[name][LAMBDA.value])
        live_mu = probe.to_numpy(materials[name][MU.value])
        live_dv = probe.to_numpy(materials[name]["dV"])
        fraction = np.asarray(mesh.cell_data[fraction_name], dtype=np.float64)
        integrated_dv = (
            np.asarray(live_dv, dtype=np.float64).reshape(fraction.size, -1).sum(axis=1)
        )
        if not np.allclose(
            live_lambda, expected_lambda, rtol=1.0e-13, atol=1.0e-14
        ) or not np.allclose(live_mu, expected_mu, rtol=1.0e-13, atol=1.0e-14):
            raise ValueError(f"live {name} Lamé fields are not the pinned 3D values")
        if not np.allclose(
            integrated_dv,
            volume * fraction,
            rtol=1.0e-10,
            atol=1.0e-18,
        ):
            raise ValueError(f"live {name} volume-fraction integration changed")
        volume_metrics.update(
            {
                f"material/{name}_E_MPa": float(young),
                f"material/{name}_nu": float(nu),
                f"material/{name}_lambda_MPa": float(expected_lambda),
                f"material/{name}_mu_MPa": float(expected_mu),
                f"material/{name}_volume_conversion": "3d",
            }
        )
    live_skin_activation = probe.to_numpy(
        materials["skin"][ACTIVATION_INV.value]
    ).astype(np.float64, copy=True)
    expected_skin_activation = np.asarray(
        skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64
    )
    if not np.array_equal(live_skin_activation, expected_skin_activation):
        raise ValueError("live Koiter ActivationInv differs from derived skin field")
    for name, key in (("Lambda", LAMBDA), ("Mu", MU), ("Fraction", FRACTION)):
        live = probe.to_numpy(materials["skin"][key.value]).astype(
            np.float64, copy=True
        )
        expected = np.asarray(skin.cell_data[key.vtk], dtype=np.float64)
        if not np.array_equal(live, expected):
            raise ValueError(f"live Koiter {name} differs from derived skin field")
    live_fixed_activation = probe.to_numpy(
        materials["muscle"][ACTIVATION_INV.value]
    ).astype(np.float64, copy=True)
    if not np.array_equal(live_fixed_activation, fixed_activation):
        raise ValueError(
            "live muscle activation differs from frozen step-40 activation"
        )
    return {
        **volume_metrics,
        "material/live_skin_activation_exact": True,
        "material/live_skin_activation_sha256_le_f8": _raw_sha256(
            live_skin_activation, dtype="<f8"
        ),
        "material/live_fixed_activation_exact": True,
        "material/live_fixed_activation_sha256_le_f8": _raw_sha256(
            live_fixed_activation, dtype="<f8"
        ),
    }


def _prune_result(result: pv.UnstructuredGrid) -> None:
    point_keep = {
        GLOBAL_POINT_ID.vtk,
        "GroupId",
        "IsFace",
        "IsFixed",
        "IsTeeth",
        "IsGingiva",
        "IsLip",
        "HistoricalIsFixed",
        "ArtificialCutIncident",
        "CutBoundaryPreexistingFixed",
        "CutBoundaryAddedFixed",
        FIXED_MASK.vtk,
        FIXED_VALUE.vtk,
        "Displacement",
        "TargetDisplacement",
        "LossMask",
        "DisplacementError",
        "DisplacementErrorNorm",
        "DeformedPoint",
        "TargetPoint",
    }
    cell_keep = {
        "ActivationMask",
        "MuscleFraction",
        "FatFraction",
        "AponeurosisFraction",
        "SMASFraction",
        ACTIVATION_INV.vtk,
        "RecoveredActivationInv",
        "RecoveredActivationInvNorm",
    }
    for name in list(result.point_data):
        if name not in point_keep:
            del result.point_data[name]
    for name in list(result.cell_data):
        if name not in cell_keep:
            del result.cell_data[name]


def _validate_result_readback(
    *,
    path: Path,
    mesh: pv.UnstructuredGrid,
    expected_displacement: np.ndarray,
    fixed_activation: np.ndarray,
) -> dict[str, Any]:
    result = pv.read(path)
    if not isinstance(result, pv.UnstructuredGrid):
        raise TypeError(f"{path} read as {type(result).__name__}, expected grid")
    if result.n_points != mesh.n_points or result.n_cells != mesh.n_cells:
        raise ValueError(f"{path} dimensions changed during readback")
    if not np.array_equal(result.points, mesh.points):
        raise ValueError(f"{path} rest points changed during readback")
    if not np.array_equal(result.cells, mesh.cells) or not np.array_equal(
        result.celltypes, mesh.celltypes
    ):
        raise ValueError(f"{path} topology changed during readback")
    displacement = np.asarray(result.point_data["Displacement"], dtype=np.float64)
    activation = np.asarray(result.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    recovered = np.asarray(result.cell_data["RecoveredActivationInv"], dtype=np.float64)
    if not np.array_equal(displacement, expected_displacement):
        raise ValueError(f"{path} displacement changed during readback")
    if not np.array_equal(activation, fixed_activation) or not np.array_equal(
        recovered, fixed_activation
    ):
        raise ValueError(f"{path} fixed activation changed during readback")
    arrays = [
        np.asarray(result.points),
        displacement,
        np.asarray(result.point_data["TargetDisplacement"]),
        activation,
    ]
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError(f"{path} contains a non-finite required array")
    is_fixed = np.asarray(result.point_data["IsFixed"], dtype=bool)
    fixed_mask = np.asarray(result.point_data[FIXED_MASK.vtk], dtype=bool)
    fixed_value = np.asarray(result.point_data[FIXED_VALUE.vtk], dtype=np.float64)
    cut = np.asarray(result.point_data["ArtificialCutIncident"], dtype=bool)
    if int(cut.sum()) != EXPECTED_CUT_VERTICES:
        raise ValueError(f"{path} artificial-cut vertex count changed")
    if (
        int(is_fixed.sum()) != EXPECTED_MODEL_FIXED_VERTICES
        or int(fixed_mask.sum()) != EXPECTED_MODEL_FIXED_DOFS
    ):
        raise ValueError(f"{path} hard-fixed model counts changed")
    if np.any(fixed_value[is_fixed] != 0.0) or np.any(displacement[is_fixed] != 0.0):
        raise ValueError(f"{path} fixed value/displacement is not exact zero")
    return {
        "readback/result_ok": True,
        "readback/displacement_sha256_le_f8": _raw_sha256(displacement, dtype="<f8"),
        "readback/fixed_activation_sha256_le_f8": _raw_sha256(activation, dtype="<f8"),
        "readback/cut_displacement_exact_zero": bool(np.all(displacement[cut] == 0.0)),
        "readback/all_fixed_displacement_exact_zero": True,
    }


def _validate_skin_readback(
    *, path: Path, source: pv.PolyData, expected: pv.PolyData
) -> dict[str, Any]:
    skin = pv.read(path)
    if not isinstance(skin, pv.PolyData):
        raise TypeError(f"{path} read as {type(skin).__name__}, expected PolyData")
    if skin.n_points != source.n_points or skin.n_cells != source.n_cells:
        raise ValueError(f"{path} skin dimensions changed during readback")
    if not np.array_equal(skin.points, source.points) or not np.array_equal(
        skin.faces, source.faces
    ):
        raise ValueError(f"{path} skin geometry/topology changed during readback")
    for name in (
        "RestArea",
        "SkinYoungModulusMPa",
        "SkinPoissonRatio",
        LAMBDA.vtk,
        MU.vtk,
        FRACTION.vtk,
        ACTIVATION_INV.vtk,
        "SkinActivationInvDiag",
        "TargetRestAreaRatio",
        "PrestrainNaturalAreaRatioFull",
        "PrestrainNaturalAreaRatio",
        "StressFreeAreaRatio",
        "PrestrainAlpha",
    ):
        actual = np.asarray(skin.cell_data[name])
        wanted = np.asarray(expected.cell_data[name])
        if not np.array_equal(actual, wanted):
            raise ValueError(f"{path} skin {name} changed during readback")
        if not np.isfinite(np.asarray(actual, dtype=np.float64)).all():
            raise ValueError(f"{path} skin {name} contains a non-finite value")
    return {
        "readback/skin_ok": True,
        "readback/skin_activation_inv_sha256_le_f8": _raw_sha256(
            np.asarray(skin.cell_data[ACTIVATION_INV.vtk]), dtype="<f8"
        ),
        "readback/skin_rest_area_sha256_le_f8": _raw_sha256(
            np.asarray(skin.cell_data["RestArea"]), dtype="<f8"
        ),
    }


def _replay_gate(
    *,
    displacement: np.ndarray,
    baseline_displacement: np.ndarray,
    target: np.ndarray,
    loss_mask: np.ndarray,
    face_ids: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    delta = displacement - baseline_displacement
    checks: dict[str, Any] = {}
    passed = True
    for name, selection in (("smile_loss_mask", loss_mask), ("isface", face_ids)):
        target_rms = _rms(target[selection])
        delta_rms = _rms(delta[selection])
        fraction = delta_rms / target_rms
        gate = fraction <= tolerance
        checks[f"replay/{name}_target_rms_m"] = target_rms
        checks[f"replay/{name}_delta_rms_m"] = delta_rms
        checks[f"replay/{name}_delta_fraction_of_target"] = fraction
        checks[f"replay/{name}_gate"] = gate
        passed &= gate
    checks["replay/tolerance_fraction_of_target"] = tolerance
    checks["replay/gate"] = passed
    if not passed:
        raise RuntimeError(f"alpha0 baseline replay gate failed: {checks}")
    return checks


def _solve_case(
    *,
    cfg: Config,
    probe: ModuleType,
    base_mesh: pv.UnstructuredGrid,
    source_skin: pv.PolyData,
    driver_skin: pv.PolyData,
    skin: pv.PolyData,
    skin_metrics: dict[str, Any],
    spec: CaseSpec,
    seed_displacement: np.ndarray,
    seed_source: str,
    fixed_activation: np.ndarray,
    baseline_displacement: np.ndarray,
    target: np.ndarray,
    loss_mask: np.ndarray,
    face_ids: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    import torch

    paths = _case_paths(spec)
    stale = [
        path
        for path in (
            paths["root"],
            *(_temporary_path(paths[key]) for key in ("result", "skin", "summary")),
        )
        if path.exists()
    ]
    if stale:
        raise FileExistsError(f"refusing stale case outputs: {stale}")
    case_mesh = base_mesh.copy(deep=True)
    cut_ids, cut_metrics = probe._configure_cut_boundary(
        case_mesh, driver_skin, "hard-fixed"
    )
    if cut_ids.size != EXPECTED_CUT_VERTICES:
        raise ValueError("reviewed hard-fixed helper returned the wrong cut set")
    if int(np.asarray(case_mesh.point_data["IsFixed"], dtype=bool).sum()) != (
        EXPECTED_MODEL_FIXED_VERTICES
    ):
        raise ValueError("hard-fixed replay model vertex count changed")
    seed = np.asarray(seed_displacement, dtype=np.float64).copy()
    if seed.shape != (case_mesh.n_points, 3) or not np.isfinite(seed).all():
        raise ValueError(f"{spec.case_id} seed displacement is malformed")
    fixed = np.asarray(case_mesh.point_data["IsFixed"], dtype=bool)
    if np.any(seed[fixed] != 0.0):
        raise ValueError(f"{spec.case_id} seed violates an exact-zero fixed vertex")

    forward, materials = probe._build_forward(case_mesh, skin)
    if int(forward.model.n_fixed) != EXPECTED_MODEL_FIXED_DOFS:
        raise ValueError(f"{spec.case_id} forward fixed DoF count changed")
    activation_t = torch.as_tensor(
        fixed_activation,
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    materials["muscle"][ACTIVATION_INV.value] = activation_t
    forward.model.set_materials(materials)
    pre_solve_materials = _validate_live_materials(
        probe=probe,
        forward=forward,
        mesh=case_mesh,
        skin=skin,
        fixed_activation=fixed_activation,
    )
    forward.model.update(
        forward.state,
        torch.as_tensor(
            seed,
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        ),
    )
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        solution = forward.step()
    elapsed_s = time.perf_counter() - start
    displacement = probe.to_numpy(forward.state.u).astype(np.float64, copy=True)
    if displacement.shape != seed.shape or not np.isfinite(displacement).all():
        raise RuntimeError(f"{spec.case_id} forward displacement is malformed")
    if np.any(displacement[fixed] != 0.0):
        raise RuntimeError(f"{spec.case_id} changed an exact-zero fixed displacement")
    post_solve_materials = _validate_live_materials(
        probe=probe,
        forward=forward,
        mesh=case_mesh,
        skin=skin,
        fixed_activation=fixed_activation,
    )
    solver = probe.forward_solution_metrics(solution)
    if cfg.require_solver_success and not bool(solver["forward/success"]):
        raise RuntimeError(f"{spec.case_id} forward solve failed: {solver}")

    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "status": "ok",
        "case_id": spec.case_id,
        "dose_id": "c020",
        "path_kind": spec.path_kind,
        "alpha": spec.alpha,
        "seed/source": seed_source,
        "seed/case_id": spec.seed_case_id,
        "seed/displacement_sha256_le_f8": _raw_sha256(seed, dtype="<f8"),
        "seed/rms_m": _rms(seed),
        "fixed_activation/source": str(BASELINE_RESULT),
        "fixed_activation/sha256_le_f8": EXPECTED_ARRAY_HASHES["fixed_activation"],
        "fixed_activation/exact_before_solve": bool(
            pre_solve_materials["material/live_fixed_activation_exact"]
        ),
        "fixed_activation/exact_after_solve": bool(
            post_solve_materials["material/live_fixed_activation_exact"]
        ),
        "execution/forward_started": True,
        "execution/forward_only": True,
        "execution/inverse_started": False,
        "execution/adjoint_started": False,
        "execution/backward_started": False,
        "forward/elapsed_s": elapsed_s,
        **solver,
        **skin_metrics,
        **cut_metrics,
        **post_solve_materials,
        **_target_metrics(
            displacement=displacement,
            target=target,
            loss_mask=loss_mask,
            face_ids=face_ids,
        ),
    }
    if spec.path_kind == "continuation" and spec.alpha == 0.0:
        metrics.update(
            _replay_gate(
                displacement=displacement,
                baseline_displacement=baseline_displacement,
                target=target,
                loss_mask=loss_mask,
                face_ids=face_ids,
                tolerance=cfg.replay_delta_fraction_of_target_tol,
            )
        )

    result_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, int | float | bool)
    }
    result = probe.make_result_mesh(
        case_mesh,
        target,
        loss_mask,
        displacement,
        fixed_activation,
        result_metrics,
    )
    _prune_result(result)
    _write_mesh_atomic(result, paths["result"])
    _write_mesh_atomic(skin, paths["skin"])
    result_identity = _file_identity(paths["result"])
    skin_identity = _file_identity(paths["skin"])
    metrics.update(
        {
            "artifact/result_path": str(paths["result"]),
            "artifact/result_size_bytes": result_identity["size_bytes"],
            "artifact/result_sha256": result_identity["sha256"],
            "artifact/skin_path": str(paths["skin"]),
            "artifact/skin_size_bytes": skin_identity["size_bytes"],
            "artifact/skin_sha256": skin_identity["sha256"],
            "artifact/summary_path": str(paths["summary"]),
            **_validate_result_readback(
                path=paths["result"],
                mesh=case_mesh,
                expected_displacement=displacement,
                fixed_activation=fixed_activation,
            ),
            **_validate_skin_readback(
                path=paths["skin"], source=source_skin, expected=skin
            ),
            "validation/ok": True,
            "validation/errors": [],
        }
    )
    _write_json_atomic(metrics, paths["summary"])
    for key in ("result", "skin", "summary"):
        cherries.log_output(paths[key])
    return metrics, displacement


def _branch_comparison(
    *,
    continuation: np.ndarray,
    direct: np.ndarray,
    target: np.ndarray,
    loss_mask: np.ndarray,
    face_ids: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    delta = continuation - direct
    result: dict[str, Any] = {
        "comparison": "c020 alpha1 continuation minus c020 alpha1 direct",
        "role": "branch/path sensitivity diagnostic; not an inverse comparison",
        "tolerance_fraction_of_target": tolerance,
    }
    agreement = True
    for name, selection in (("smile_loss_mask", loss_mask), ("isface", face_ids)):
        target_rms = _rms(target[selection])
        delta_rms = _rms(delta[selection])
        fraction = delta_rms / target_rms
        within = fraction <= tolerance
        result[f"{name}/target_rms_m"] = target_rms
        result[f"{name}/delta_rms_m"] = delta_rms
        result[f"{name}/delta_fraction_of_target"] = fraction
        result[f"{name}/within_tolerance"] = within
        agreement &= within
    result["within_tolerance"] = agreement
    result["status"] = "same-branch-within-tolerance" if agreement else "path-sensitive"
    return result


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| case | path | alpha | seed | forward | steps | loss RMS mm | IsFace RMS mm |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {path_kind} | {alpha:.2f} | {seed} | {success} | "
            "{steps} | {loss:.6g} | {face:.6g} |".format(
                case=row["case_id"],
                path_kind=row["path_kind"],
                alpha=float(row["alpha"]),
                seed=row["seed/source"],
                success=row["forward/success"],
                steps=row["forward/steps"],
                loss=row["target/loss_mask_error_rms_mm"],
                face=row["target/isface_error_rms_mm"],
            )
        )
    temporary = _temporary_path(path)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing stale table output: {path} or {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_runtime_dependencies(probe: ModuleType) -> dict[str, Any]:
    dependencies = (
        (probe.KOITER_IMPLEMENTATION, probe.KOITER_IMPLEMENTATION_SHA256, "Koiter"),
        (
            probe.VOLUME_LAME_IMPLEMENTATION,
            probe.VOLUME_LAME_IMPLEMENTATION_SHA256,
            "volume 3D Lame implementation",
        ),
        (
            probe.VOLUME_FORWARD_IMPLEMENTATION,
            probe.VOLUME_FORWARD_IMPLEMENTATION_SHA256,
            "volume forward builder",
        ),
        (probe.TARGET_IMPLEMENTATION, probe.TARGET_IMPLEMENTATION_SHA256, "target"),
        (probe.OUTPUT_IMPLEMENTATION, probe.OUTPUT_IMPLEMENTATION_SHA256, "output"),
        (
            RUNTIME_COMPAT_CONFIG,
            INPUT_IDENTITIES["runtime_compat_config"]["sha256"],
            "08/17 compatibility config imported as _human_face_config",
        ),
        (
            probe.CORE_MODULI_IMPLEMENTATION,
            probe.CORE_MODULI_IMPLEMENTATION_SHA256,
            "core moduli",
        ),
    )
    rows: list[dict[str, Any]] = []
    for path, expected, name in dependencies:
        actual = probe.require_file_sha256(path, expected, name=name)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return {"files": rows, "all_exact": True}


def run(cfg: Config) -> None:
    # This producer has no inverse, adjoint, backward, or optimizer mode.  The
    # source-level approval blocker is checked before input reads or CUDA/Warp setup.
    _validate_config(cfg)
    producer_identity_start = {"path": str(PRODUCER), **_file_identity(PRODUCER)}
    input_provenance = {name: _require_identity(name) for name in INPUT_IDENTITIES}
    probe = _load_reviewed_probe()
    runtime_dependencies = _require_runtime_dependencies(probe)

    mesh = pv.read(cfg.input_mesh)
    corrected_skin = pv.read(cfg.input_corrected_skin)
    driver_skin = pv.read(cfg.input_driver_skin)
    baseline_result = pv.read(cfg.input_baseline_result)
    baseline_target = pv.read(cfg.input_baseline_target)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError(f"prepared mesh read as {type(mesh).__name__}")
    if not isinstance(corrected_skin, pv.PolyData) or not isinstance(
        driver_skin, pv.PolyData
    ):
        raise TypeError("corrected and driver skins must read as PolyData")
    if not isinstance(baseline_result, pv.UnstructuredGrid) or not isinstance(
        baseline_target, pv.UnstructuredGrid
    ):
        raise TypeError("baseline result and target must read as UnstructuredGrid")
    baseline_summary = _read_json(cfg.input_baseline_summary)
    _validate_mesh_topology(mesh)
    _validate_corrected_skin(corrected_skin)
    target, loss_mask, fixed_activation, baseline_displacement, baseline = (
        _validate_baseline(
            mesh,
            baseline_result,
            baseline_target,
            baseline_summary,
        )
    )
    raw_ratio, rho_full, mapping = _derive_prestrain_basis(corrected_skin, driver_skin)
    face_ids = _global_ids(corrected_skin, name="corrected skin")
    if np.any(face_ids < 0) or np.any(face_ids >= mesh.n_points):
        raise ValueError("corrected IsFace GlobalPointId is outside prepared mesh")
    if not np.array_equal(
        np.asarray(corrected_skin.points, dtype=np.float64),
        np.asarray(mesh.points, dtype=np.float64)[face_ids],
    ):
        raise ValueError("corrected IsFace points do not match prepared mesh")
    loss_target_rms = _rms(target[loss_mask])
    face_target_rms = _rms(target[face_ids])
    if not math.isclose(
        loss_target_rms, EXPECTED_LOSS_TARGET_RMS_M, rel_tol=1.0e-13, abs_tol=1.0e-15
    ) or not math.isclose(
        face_target_rms,
        EXPECTED_ISFACE_TARGET_RMS_M,
        rel_tol=1.0e-13,
        abs_tol=1.0e-15,
    ):
        raise ValueError("alpha0 replay normalization changed")

    # Derive and hash every reviewed material state before runtime initialization.
    skins: dict[float, tuple[pv.PolyData, dict[str, Any]]] = {
        alpha: _skin_for_alpha(corrected_skin, raw_ratio, rho_full, alpha)
        for alpha in ALPHAS
    }
    probe.configure_runtime()

    rows: list[dict[str, Any]] = []
    displacements: dict[str, np.ndarray] = {}
    continuation_seed = baseline_displacement
    continuation_seed_source = str(BASELINE_RESULT)
    for step, spec in enumerate(CONTINUATION_CASES):
        cherries.set_step(step)
        skin, skin_metrics = skins[spec.alpha]
        logger.info(
            "Solving fixed-activation c020 continuation alpha %.2f from %s",
            spec.alpha,
            continuation_seed_source,
        )
        row, displacement = _solve_case(
            cfg=cfg,
            probe=probe,
            base_mesh=mesh,
            source_skin=corrected_skin,
            driver_skin=driver_skin,
            skin=skin,
            skin_metrics=skin_metrics,
            spec=spec,
            seed_displacement=continuation_seed,
            seed_source=continuation_seed_source,
            fixed_activation=fixed_activation,
            baseline_displacement=baseline_displacement,
            target=target,
            loss_mask=loss_mask,
            face_ids=face_ids,
        )
        rows.append(row)
        displacements[spec.case_id] = displacement
        continuation_seed = displacement
        continuation_seed_source = str(_case_paths(spec)["result"])
        cherries.log_metrics(
            {
                "forward/success": float(bool(row["forward/success"])),
                "target/loss_mask_error_rms_mm": row["target/loss_mask_error_rms_mm"],
                "target/isface_error_rms_mm": row["target/isface_error_rms_mm"],
            }
        )

    cherries.set_step(len(CONTINUATION_CASES))
    direct_skin, direct_skin_metrics = skins[DIRECT_CASE.alpha]
    direct_row, direct_displacement = _solve_case(
        cfg=cfg,
        probe=probe,
        base_mesh=mesh,
        source_skin=corrected_skin,
        driver_skin=driver_skin,
        skin=direct_skin,
        skin_metrics=direct_skin_metrics,
        spec=DIRECT_CASE,
        seed_displacement=baseline_displacement,
        seed_source=str(BASELINE_RESULT),
        fixed_activation=fixed_activation,
        baseline_displacement=baseline_displacement,
        target=target,
        loss_mask=loss_mask,
        face_ids=face_ids,
    )
    rows.append(direct_row)
    displacements[DIRECT_CASE.case_id] = direct_displacement
    cherries.log_metrics(
        {
            "forward/success": float(bool(direct_row["forward/success"])),
            "target/loss_mask_error_rms_mm": direct_row[
                "target/loss_mask_error_rms_mm"
            ],
            "target/isface_error_rms_mm": direct_row["target/isface_error_rms_mm"],
        }
    )

    continuation_alpha1 = CONTINUATION_CASES[-1].case_id
    branch = _branch_comparison(
        continuation=displacements[continuation_alpha1],
        direct=displacements[DIRECT_CASE.case_id],
        target=target,
        loss_mask=loss_mask,
        face_ids=face_ids,
        tolerance=cfg.replay_delta_fraction_of_target_tol,
    )
    final_dependency_recheck = {
        "input_provenance": _recheck_file_rows(
            [{"name": name, **identity} for name, identity in input_provenance.items()],
            context="pinned input",
        ),
        "runtime_dependencies": _recheck_file_rows(
            runtime_dependencies["files"],
            context="runtime dependency",
        ),
    }
    producer_identity_end = {"path": str(PRODUCER), **_file_identity(PRODUCER)}
    if producer_identity_end != producer_identity_start:
        raise RuntimeError("producer source changed during replay execution")
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "execution_contract": {
            "forward_only": True,
            "forward_solves": len(CASE_ORDER),
            "fixed_muscle_activation": True,
            "inverse_started": False,
            "adjoint_started": False,
            "backward_started": False,
            "activation_optimized": False,
        },
        "approval": {
            "static_source_blocker_was_explicitly_cleared": True,
            "c020_only": True,
            "c050_started": False,
            "c050_policy": (
                "conditional second-stage only after c020 analysis and a new isolated "
                "reviewed producer/run; this executable exposes no c050 option"
            ),
        },
        "protocol": {
            "dose_id": "c020",
            "linear_tightening": LINEAR_TIGHTENING,
            "length_factor": LENGTH_FACTOR,
            "uniform_natural_area_ratio": float(np.square(LENGTH_FACTOR)),
            "area_ratio_floor": AREA_RATIO_FLOOR,
            "continuation_alphas": list(ALPHAS),
            "direct_alphas": [1.0],
            "alpha_interpolation": "rho_alpha=np.power(rho_full,alpha)",
            "alpha0_replay_tolerance_fraction_of_target": (
                cfg.replay_delta_fraction_of_target_tol
            ),
            "alpha0_replay_domains": ["SmileLossMask", "corrected IsFace"],
            "continuation_seed_rule": "previous solved equilibrium displacement",
            "direct_seed_rule": "exact pinned corrected p000 step-40 displacement",
        },
        "input_provenance": input_provenance,
        "producer_identity": {
            **producer_identity_start,
            "unchanged_through_all_solves": True,
        },
        "runtime_dependencies": runtime_dependencies,
        "final_dependency_recheck": final_dependency_recheck,
        "baseline": baseline,
        "mapping": mapping,
        "material_contract": {
            "skin": {
                "domain": "all-vertex IsFace filtered PolyData",
                "E_MPa": float(probe.SKIN_E),
                "nu": float(probe.SKIN_NU),
                "thickness_m": float(probe.SKIN_THICKNESS),
                "lame_conversion": SKIN_LAME_CONVERSION,
                "energy_measure": "fixed original reference area",
            },
            "volume": {
                "lame_conversion": (
                    "3d: lambda=E*nu/((1+nu)*(1-2*nu)); mu=E/(2*(1+nu))"
                ),
                "fat": {"E_MPa": float(probe.FAT_E), "nu": float(probe.FAT_NU)},
                "muscle": {
                    "E_MPa": float(probe.MUSCLE_E),
                    "nu": float(probe.MUSCLE_NU),
                    "activation": "fixed corrected p000 best/terminal step-40 tensor",
                },
                "aponeurosis": {
                    "E_MPa": float(probe.APONEUROSIS_E),
                    "nu": float(probe.APONEUROSIS_NU),
                },
            },
        },
        "case_order": [spec.case_id for spec in CASE_ORDER],
        "cases": rows,
        "branch_comparison": branch,
        "output_contract": {
            "root": str(OUTPUT_ROOT),
            "summary_path": str(cfg.output_summary),
            "table_path": str(cfg.output_table),
            "case_layout": (
                "<root>/c020/{continuation,direct}/alpha-NNN/"
                "{result.vtu,skin.vtp,forward-summary.json}"
            ),
            "expected_result_vtus": len(CASE_ORDER),
            "expected_skin_vtps": len(CASE_ORDER),
            "expected_forward_sidecars": len(CASE_ORDER),
            "overwrite_policy": "refuse any existing aggregate or result root",
        },
    }
    _write_json_atomic(aggregate, cfg.output_summary)
    _write_table(cfg.output_table, rows)
    logger.info("Wrote %s", cfg.output_summary)


if __name__ == "__main__":
    cherries.main(run)
