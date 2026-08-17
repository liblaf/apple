from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from _human_face_config import (
    APONEUROSIS_E,
    APONEUROSIS_FRACTION,
    APONEUROSIS_NU,
    FAT_E,
    FAT_FRACTION,
    FAT_NU,
    FORWARD_ATOL,
    FORWARD_MAX_STEPS,
    FORWARD_RTOL,
    MUSCLE_E,
    MUSCLE_FRACTION,
    MUSCLE_NU,
    SETUP_SKIN_ESTIMATED_PRESTRAIN,
    SKIN_THICKNESS,
    InverseCase,
    configure_runtime,
)
from _material_heuristics import (
    MaterialCandidate,
    file_sha256,
    make_candidate_skin,
    make_signed_heat_field,
    prepare_surface_geometry,
    skin_material_content_hash,
    skin_solver_content_hash,
    skin_topology_content_hash,
)
from _reference import PREPARED_MESH, enable_reference_modules

from liblaf import cherries, melon

mpl.use("Agg", force=True)

enable_reference_modules()

from _human_face_forward import material_tree, set_volume_material  # noqa: E402
from _human_face_metrics import forward_solution_metrics, to_numpy  # noqa: E402
from _human_face_output import make_result_mesh  # noqa: E402
from _human_face_targets import target_displacement_and_mask  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 2
WORKER_FLAG = "--worker-request"
WORKER_MODE = WORKER_FLAG in sys.argv
DEBUG_MODE = os.environ.get("DEBUG") == "1"

FORMAL_YOUNG_SCALES = (0.25, 0.5, 1.0)
FORMAL_PRESTRAIN_GAINS = (0.5, 0.75, 1.0)
FORMAL_CANDIDATES = tuple(
    MaterialCandidate(young_min_scale=young, prestrain_gain=gain)
    for young in FORMAL_YOUNG_SCALES
    for gain in FORMAL_PRESTRAIN_GAINS
)
FORMAL_LABELS = tuple(candidate.label for candidate in FORMAL_CANDIDATES)
FORMAL_ORDERS = {
    "r0": FORMAL_LABELS,
    "r1": tuple(reversed(FORMAL_LABELS)),
}
ANCHOR_LABELS = ("e025-p000", "e100-p000")
MANIFEST_LABELS = (
    "e100-p000",
    "e100-p050",
    "e100-p100",
    "e025-p000",
    "e025-p050",
    "e025-p100",
)
MANIFEST_GRID = {
    "young_min_scales": [1.0, 0.25],
    "prestrain_gains": [0.0, 0.5, 1.0],
}
NEW_LABELS = tuple(label for label in FORMAL_LABELS if label not in MANIFEST_LABELS)

EXPECTED_HEURISTIC = {
    "area_deadband": 0.01,
    "cap_quantile": 0.99,
    "diffusion_sigma_m": 0.005,
    "lame_conversion": (
        "existing 3D isotropic convention: "
        "lambda = E * nu / ((1 + nu) * (1 - 2 * nu)); "
        "mu = E / (2 * (1 + nu))"
    ),
}
EXPECTED_MATERIAL_GATES = {
    "max_normalized_interior_jump_q99": 0.08,
    "max_normalized_interior_jump": 0.20,
    "max_normalized_boundary_jump_q99": 0.08,
    "max_normalized_boundary_jump": 0.20,
    "max_e_edge_jump_mpa": 0.06,
    "max_activation_edge_jump": 0.08,
    "max_e_edge_rms_mpa": 0.012,
    "max_activation_edge_rms": 0.02,
    "max_singleton_components": 20,
    "max_small_component_area_fraction": 0.005,
}
EXPECTED_PHYSICAL_GATES = {
    "min_det_f_q001": 0.20,
    "min_skin_area_ratio_q001": 0.10,
    "max_skin_area_ratio_q999": 10.0,
    "min_muscle_activation_eigenvalue": 1.0e-6,
}
EXPECTED_REPRODUCIBILITY_GATES = {
    "max_fidelity_difference": 0.001,
    "max_displacement_difference_fraction_of_target": 0.01,
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = (
        Path(PREPARED_MESH) if WORKER_MODE else cherries.input(PREPARED_MESH)
    )
    input_candidates: Path = (
        Path("10-material-candidates-manifest.json")
        if WORKER_MODE
        else cherries.input("10-material-candidates-manifest.json")
    )
    output_summary: Path = (
        Path("40-static-material-admissibility.json")
        if WORKER_MODE
        else cherries.output("40-static-material-admissibility.json", mkdir=True)
    )
    output_csv: Path = (
        Path("40-static-material-admissibility.csv")
        if WORKER_MODE
        else cherries.output("40-static-material-admissibility.csv", mkdir=True)
    )
    output_table: Path = (
        Path("40-static-material-admissibility.md")
        if WORKER_MODE
        else cherries.output("40-static-material-admissibility.md", mkdir=True)
    )
    artifact_dir_name: str = "40-static-material-admissibility-artifacts"

    stage: str = "formal"
    debug_label: str = "e100-p050"
    young_min_scales: str = "0.25,0.5,1.0"
    prestrain_gains: str = "0.5,0.75,1.0"
    replicate_count: int = 2
    order_contract: str = "lexicographic-low-to-high-and-exact-reverse-v1"
    worker_timeout_s: float = 28_800.0

    min_det_f_q001: float = 0.20
    min_skin_area_ratio_q001: float = 0.10
    max_skin_area_ratio_q999: float = 10.0
    min_muscle_activation_eigenvalue: float = 1.0e-6
    max_fidelity_difference: float = 0.001
    max_displacement_difference_fraction_of_target: float = 0.01


def _reject_json_constant(value: str) -> None:
    msg = f"non-standard JSON constant {value!r}"
    raise ValueError(msg)


def _validate_finite_json(value: Any, *, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{path} contains non-finite number {value}"
        raise ValueError(msg)
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_finite_json(item, path=f"{path}[{index}]")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
    )
    _validate_finite_json(payload)
    if not isinstance(payload, dict):
        msg = f"{path} must contain a JSON object"
        raise TypeError(msg)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _validate_finite_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = sorted(set(keys) - set(mapping))
    if missing:
        msg = f"{context} is missing required keys: {missing}"
        raise KeyError(msg)


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        msg = f"missing file: {resolved}"
        raise FileNotFoundError(msg)
    return {
        "size_bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
    }


def _verify_file_identity(path: Path, expected: Any, context: str) -> dict[str, Any]:
    if not isinstance(expected, dict):
        msg = f"{context} identity must be an object"
        raise TypeError(msg)
    _require_keys(expected, ("size_bytes", "sha256"), f"{context} identity")
    actual = _file_identity(path)
    canonical = {
        "size_bytes": int(expected["size_bytes"]),
        "sha256": str(expected["sha256"]),
    }
    if actual != canonical:
        msg = f"{context} identity mismatch: expected {canonical}, got {actual}"
        raise ValueError(msg)
    return actual


def _candidate_parameters(label: str) -> tuple[float, float]:
    match = next(
        (candidate for candidate in FORMAL_CANDIDATES if candidate.label == label), None
    )
    if match is None:
        msg = f"unknown formal candidate {label!r}"
        raise ValueError(msg)
    return match.young_min_scale, match.prestrain_gain


def _parse_exact_float_list(value: str, *, name: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        msg = f"{name} must be a comma-separated float list, got {value!r}"
        raise ValueError(msg) from error
    if not parsed or not np.isfinite(parsed).all() or len(set(parsed)) != len(parsed):
        msg = f"{name} must contain unique finite values, got {parsed}"
        raise ValueError(msg)
    return parsed


def _skin_hashes(skin: pv.PolyData) -> dict[str, str]:
    return {
        "topology_sha256": skin_topology_content_hash(skin),
        "material_sha256": skin_material_content_hash(skin),
        "solver_sha256": skin_solver_content_hash(skin),
    }


def _candidate_path(manifest_path: Path, row: dict[str, Any]) -> Path:
    path = Path(str(row["skin/path"]))
    resolved = (
        (manifest_path.parent / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )
    root = manifest_path.parent.resolve()
    if not resolved.is_relative_to(root):
        msg = f"candidate skin escapes manifest directory: {path}"
        raise ValueError(msg)
    return resolved


def _validate_manifest_candidate(  # noqa: C901
    manifest_path: Path, row: dict[str, Any]
) -> tuple[Path, pv.PolyData, dict[str, Any]]:
    required = (
        "schema_version",
        "label",
        "young_min_scale",
        "prestrain_gain",
        "skin/path",
        "skin/file_identity",
        "content/n_points",
        "content/n_triangles",
        "content/topology_sha256",
        "content/material_sha256",
        "content/solver_sha256",
        "validation/ok",
        "validation/errors",
        "readback/ok",
        "readback/errors",
        "readback/content/topology_sha256",
        "readback/content/material_sha256",
        "readback/content/solver_sha256",
    )
    label = str(row.get("label", "<missing>"))
    _require_keys(row, required, f"manifest candidate {label}")
    if int(row["schema_version"]) != MANIFEST_SCHEMA_VERSION:
        msg = f"candidate {label} schema changed: {row['schema_version']}"
        raise ValueError(msg)
    if label not in MANIFEST_LABELS:
        msg = f"unexpected manifest candidate {label!r}"
        raise ValueError(msg)
    expected_parameters = {
        "e100-p000": (1.0, 0.0),
        "e100-p050": (1.0, 0.5),
        "e100-p100": (1.0, 1.0),
        "e025-p000": (0.25, 0.0),
        "e025-p050": (0.25, 0.5),
        "e025-p100": (0.25, 1.0),
    }[label]
    actual_parameters = (
        float(row["young_min_scale"]),
        float(row["prestrain_gain"]),
    )
    if actual_parameters != expected_parameters:
        msg = (
            f"candidate {label} parameters {actual_parameters} differ from "
            f"{expected_parameters}"
        )
        raise ValueError(msg)
    if not bool(row["validation/ok"]) or list(row["validation/errors"]):
        msg = f"candidate {label} failed material validation"
        raise ValueError(msg)
    if not bool(row["readback/ok"]) or list(row["readback/errors"]):
        msg = f"candidate {label} failed readback validation"
        raise ValueError(msg)
    for name in ("topology", "material", "solver"):
        if str(row[f"content/{name}_sha256"]) != str(
            row[f"readback/content/{name}_sha256"]
        ):
            msg = f"candidate {label} {name} and readback hashes differ"
            raise ValueError(msg)

    path = _candidate_path(manifest_path, row)
    file_identity = _verify_file_identity(
        path, row["skin/file_identity"], f"candidate {label} skin"
    )
    skin = pv.read(path)
    if not isinstance(skin, pv.PolyData):
        msg = f"candidate {label} read as {type(skin).__name__}, expected PolyData"
        raise TypeError(msg)
    if skin.n_points != int(row["content/n_points"]) or skin.n_cells != int(
        row["content/n_triangles"]
    ):
        msg = f"candidate {label} live dimensions changed"
        raise ValueError(msg)
    hashes = _skin_hashes(skin)
    for name, digest in hashes.items():
        if digest != str(row[f"content/{name}"]):
            msg = f"candidate {label} live {name} mismatch"
            raise ValueError(msg)
    return path, skin, {"file_identity": file_identity, **hashes}


def load_manifest(  # noqa: C901, PLR0912, PLR0915
    input_mesh: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _load_json(manifest_path)
    _require_keys(
        manifest,
        (
            "schema_version",
            "complete",
            "input_mesh",
            "input_mesh_identity",
            "input_mesh_identity_verified_stable",
            "target",
            "selection",
            "grid",
            "heuristic",
            "material_gates",
            "surface_geometry",
            "primary_signed_heat_field",
            "validation_errors",
            "candidate_validation_errors",
            "n_candidates",
            "candidates",
        ),
        "material manifest",
    )
    if int(manifest["schema_version"]) != MANIFEST_SCHEMA_VERSION:
        msg = f"unsupported material manifest schema {manifest['schema_version']}"
        raise ValueError(msg)
    if not bool(manifest["complete"]):
        msg = "material manifest is incomplete"
        raise ValueError(msg)
    if list(manifest["validation_errors"]) or dict(
        manifest["candidate_validation_errors"]
    ):
        msg = "material manifest contains validation errors"
        raise ValueError(msg)
    if not bool(manifest["input_mesh_identity_verified_stable"]):
        msg = "material manifest did not verify stable input identity"
        raise ValueError(msg)
    if Path(str(manifest["input_mesh"])).resolve() != input_mesh.resolve():
        msg = "material manifest input mesh differs from static-sweep input"
        raise ValueError(msg)
    _verify_file_identity(
        input_mesh, manifest["input_mesh_identity"], "prepared input mesh"
    )
    if manifest["target"] != "Smile":
        msg = f"manifest target changed: {manifest['target']!r}"
        raise ValueError(msg)
    if manifest["selection"] != (
        "all surface-triangle vertices are finite IsFace points"
    ):
        msg = f"manifest selection changed: {manifest['selection']!r}"
        raise ValueError(msg)
    if manifest["grid"] != MANIFEST_GRID:
        msg = f"manifest source grid changed: {manifest['grid']}"
        raise ValueError(msg)
    heuristic = manifest["heuristic"]
    if not isinstance(heuristic, dict):
        msg = "manifest heuristic must be an object"
        raise TypeError(msg)
    actual_heuristic = {key: heuristic.get(key) for key in EXPECTED_HEURISTIC}
    if actual_heuristic != EXPECTED_HEURISTIC:
        msg = f"manifest fixed heuristic changed: {actual_heuristic}"
        raise ValueError(msg)
    if manifest["material_gates"] != EXPECTED_MATERIAL_GATES:
        msg = f"manifest material gates changed: {manifest['material_gates']}"
        raise ValueError(msg)
    heat = manifest["primary_signed_heat_field"]
    if not isinstance(heat, dict):
        msg = "primary signed heat field must be an object"
        raise TypeError(msg)
    if not bool(heat.get("validation/ok", False)) or list(
        heat.get("validation/errors", [])
    ):
        msg = "primary signed heat field failed validation"
        raise ValueError(msg)
    heat_metrics = heat.get("metrics")
    if not isinstance(heat_metrics, dict):
        msg = "primary signed heat metrics must be an object"
        raise TypeError(msg)
    expected_heat = {
        "cap_quantile": 0.99,
        "diffusion_length": 0.005,
        "soft_deadband_log": math.log1p(0.01),
    }
    if {key: heat_metrics.get(key) for key in expected_heat} != expected_heat:
        msg = "primary signed heat parameters changed"
        raise ValueError(msg)

    rows = manifest["candidates"]
    if not isinstance(rows, list) or int(manifest["n_candidates"]) != 6:
        msg = "schema-2 manifest must contain exactly six candidates"
        raise ValueError(msg)
    if tuple(str(row.get("label")) for row in rows) != MANIFEST_LABELS:
        msg = "schema-2 candidate labels/order changed"
        raise ValueError(msg)
    catalog: dict[str, dict[str, Any]] = {}
    for row_any in rows:
        if not isinstance(row_any, dict):
            msg = "manifest candidate rows must be objects"
            raise TypeError(msg)
        row = dict(row_any)
        path, _skin, provenance = _validate_manifest_candidate(manifest_path, row)
        catalog[str(row["label"])] = {
            "label": str(row["label"]),
            "young_min_scale": float(row["young_min_scale"]),
            "prestrain_gain": float(row["prestrain_gain"]),
            "skin/path": str(path),
            "skin/source": "schema2-manifest",
            "n_points": int(row["content/n_points"]),
            "n_triangles": int(row["content/n_triangles"]),
            **provenance,
        }
    return manifest, catalog


def _make_candidate(
    geometry: Any, signed_field: Any, candidate: MaterialCandidate
) -> tuple[pv.PolyData, dict[str, Any]]:
    return make_candidate_skin(
        geometry,
        signed_field,
        candidate,
        max_e_edge_jump=EXPECTED_MATERIAL_GATES["max_e_edge_jump_mpa"],
        max_activation_edge_jump=EXPECTED_MATERIAL_GATES["max_activation_edge_jump"],
        max_e_edge_rms=EXPECTED_MATERIAL_GATES["max_e_edge_rms_mpa"],
        max_activation_edge_rms=EXPECTED_MATERIAL_GATES["max_activation_edge_rms"],
        max_singleton_components=EXPECTED_MATERIAL_GATES["max_singleton_components"],
        max_small_component_area_fraction=EXPECTED_MATERIAL_GATES[
            "max_small_component_area_fraction"
        ],
    )


def prepare_formal_catalog(  # noqa: C901
    *,
    cfg: Config,
    base_mesh: pv.UnstructuredGrid,
    manifest: dict[str, Any],
    manifest_catalog: dict[str, dict[str, Any]],
    selected_labels: tuple[str, ...],
    artifact_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    needs_generated = any(label in NEW_LABELS for label in selected_labels)
    if not needs_generated:
        return (
            {label: manifest_catalog[label] for label in selected_labels},
            [],
        )

    geometry = prepare_surface_geometry(base_mesh)
    signed_field = make_signed_heat_field(
        geometry,
        area_deadband=EXPECTED_HEURISTIC["area_deadband"],
        cap_quantile=EXPECTED_HEURISTIC["cap_quantile"],
        diffusion_sigma=EXPECTED_HEURISTIC["diffusion_sigma_m"],
        max_normalized_interior_jump_q99=EXPECTED_MATERIAL_GATES[
            "max_normalized_interior_jump_q99"
        ],
        max_normalized_interior_jump=EXPECTED_MATERIAL_GATES[
            "max_normalized_interior_jump"
        ],
        max_normalized_boundary_jump_q99=EXPECTED_MATERIAL_GATES[
            "max_normalized_boundary_jump_q99"
        ],
        max_normalized_boundary_jump=EXPECTED_MATERIAL_GATES[
            "max_normalized_boundary_jump"
        ],
    )
    if signed_field.validation_errors:
        msg = (
            f"reconstructed signed heat field failed: {signed_field.validation_errors}"
        )
        raise ValueError(msg)

    # Reproduce all schema-2 candidates before extending its field to new points.
    manifest_by_label = {str(row["label"]): row for row in manifest["candidates"]}
    for label in MANIFEST_LABELS:
        row = manifest_by_label[label]
        candidate = MaterialCandidate(
            young_min_scale=float(row["young_min_scale"]),
            prestrain_gain=float(row["prestrain_gain"]),
        )
        reconstructed, metrics = _make_candidate(geometry, signed_field, candidate)
        if not bool(metrics["validation/ok"]) or list(metrics["validation/errors"]):
            msg = f"reconstructed manifest candidate {label} failed validation"
            raise ValueError(msg)
        hashes = _skin_hashes(reconstructed)
        expected_hashes = {
            "topology_sha256": str(row["content/topology_sha256"]),
            "material_sha256": str(row["content/material_sha256"]),
            "solver_sha256": str(row["content/solver_sha256"]),
        }
        if hashes != expected_hashes:
            msg = (
                f"reconstructed candidate {label} differs from schema-2 content: "
                f"expected {expected_hashes}, got {hashes}"
            )
            raise ValueError(msg)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    catalog = {
        label: dict(manifest_catalog[label])
        for label in selected_labels
        if label in manifest_catalog
    }
    generated_rows: list[dict[str, Any]] = []
    for candidate in FORMAL_CANDIDATES:
        if candidate.label not in selected_labels or candidate.label not in NEW_LABELS:
            continue
        skin, metrics = _make_candidate(geometry, signed_field, candidate)
        if not bool(metrics["validation/ok"]) or list(metrics["validation/errors"]):
            msg = (
                f"new candidate {candidate.label} failed validation: "
                f"{metrics['validation/errors']}"
            )
            raise ValueError(msg)
        path = artifact_dir / f"skin-{candidate.label}.vtp"
        melon.save(skin, path)
        readback = pv.read(path)
        if not isinstance(readback, pv.PolyData):
            msg = f"{path} read back as {type(readback).__name__}"
            raise TypeError(msg)
        expected_hashes = _skin_hashes(skin)
        actual_hashes = _skin_hashes(readback)
        if actual_hashes != expected_hashes:
            msg = f"new candidate {candidate.label} changed during VTP readback"
            raise ValueError(msg)
        row = {
            "label": candidate.label,
            "young_min_scale": float(candidate.young_min_scale),
            "prestrain_gain": float(candidate.prestrain_gain),
            "skin/path": str(path.resolve()),
            "skin/source": "40-generated-from-schema2-field",
            "n_points": int(readback.n_points),
            "n_triangles": int(readback.n_cells),
            "file_identity": _file_identity(path),
            **actual_hashes,
            "skin/E_MPa_min": float(metrics["skin/E_MPa_min"]),
            "skin/E_MPa_area_weighted_mean": float(
                metrics["skin/E_MPa_area_weighted_mean"]
            ),
            "skin/activation_inv_diag_max": float(
                metrics["skin/activation_inv_diag_max"]
            ),
            "validation/ok": True,
            "validation/errors": [],
        }
        catalog[candidate.label] = row
        generated_rows.append(row)
        cherries.log_output(path)
        logger.info("Generated %s", path)
    if tuple(sorted(catalog)) != tuple(sorted(selected_labels)):
        msg = f"candidate catalog is incomplete: {sorted(catalog)}"
        raise ValueError(msg)
    if (
        cfg.stage == "formal"
        and tuple(row["label"] for row in generated_rows) != NEW_LABELS
    ):
        msg = "formal run did not generate the exact five pre-registered skins"
        raise ValueError(msg)
    return catalog, generated_rows


def build_static_forward(
    mesh: pv.UnstructuredGrid, skin: pv.PolyData
) -> tuple[Any, pv.PolyData]:
    from liblaf.apple.common import GLOBAL_POINT_ID
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import Koiter, StableNeoHookean, StableNeoHookeanActive

    candidate_skin = skin.copy(deep=True)
    global_ids = np.asarray(
        candidate_skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
    )
    if (
        global_ids.shape != (candidate_skin.n_points,)
        or np.unique(global_ids).size != candidate_skin.n_points
    ):
        msg = "candidate skin has missing or duplicate GlobalPointId values"
        raise ValueError(msg)
    if global_ids.min() < 0 or global_ids.max() >= mesh.n_points:
        msg = "candidate skin GlobalPointId values escape the volume mesh"
        raise ValueError(msg)

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    set_volume_material(
        mesh,
        E=APONEUROSIS_E,
        nu=APONEUROSIS_NU,
        fraction=np.asarray(mesh.cell_data[APONEUROSIS_FRACTION], dtype=np.float64),
    )
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="aponeurosis"))
    set_volume_material(
        mesh,
        E=FAT_E,
        nu=FAT_NU,
        fraction=np.asarray(mesh.cell_data[FAT_FRACTION], dtype=np.float64),
    )
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="fat"))
    set_volume_material(
        mesh,
        E=MUSCLE_E,
        nu=MUSCLE_NU,
        fraction=np.asarray(mesh.cell_data[MUSCLE_FRACTION], dtype=np.float64),
    )
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))
    builder.add_potential(
        Koiter.from_pyvista(candidate_skin, name="skin", thickness=SKIN_THICKNESS)
    )
    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=FORWARD_MAX_STEPS,
        atol=FORWARD_ATOL,
        rtol=FORWARD_RTOL,
    )
    return forward, candidate_skin


def tetra_det_f_metrics(result: pv.UnstructuredGrid) -> dict[str, Any]:
    encoded = np.asarray(result.cells, dtype=np.int64).reshape(-1, 5)
    if not np.all(encoded[:, 0] == 4):
        msg = "deformation quality expects tetrahedral cells"
        raise ValueError(msg)
    tets = encoded[:, 1:]
    rest = np.asarray(result.points, dtype=np.float64)
    displacement = np.asarray(result.point_data["Displacement"], dtype=np.float64)
    deformed = rest + displacement

    def six_volume(points: np.ndarray) -> np.ndarray:
        edge_1 = points[tets[:, 1]] - points[tets[:, 0]]
        edge_2 = points[tets[:, 2]] - points[tets[:, 0]]
        edge_3 = points[tets[:, 3]] - points[tets[:, 0]]
        return np.einsum("ij,ij->i", edge_1, np.cross(edge_2, edge_3))

    rest_six = six_volume(rest)
    if np.any(np.abs(rest_six) <= np.finfo(np.float64).eps):
        msg = "rest mesh contains zero-volume tetrahedra"
        raise ValueError(msg)
    det_f = six_volume(deformed) / rest_six
    if not np.isfinite(det_f).all():
        msg = "deformed determinant contains non-finite values"
        raise ValueError(msg)
    return {
        "quality/detF_min": float(det_f.min()),
        "quality/detF_q001": float(np.quantile(det_f, 0.001)),
        "quality/detF_median": float(np.median(det_f)),
        "quality/detF_max": float(det_f.max()),
        "quality/inverted_tets": int((det_f <= 0.0).sum()),
        "quality/inverted_tet_fraction": float(np.mean(det_f <= 0.0)),
        "quality/detF_lt_0p2_tets": int((det_f < 0.2).sum()),
        "quality/detF_lt_0p5_tets": int((det_f < 0.5).sum()),
    }


def skin_deformation_quality(
    result: pv.UnstructuredGrid, skin: pv.PolyData
) -> dict[str, Any]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    encoded = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    if encoded.size == 0 or not np.all(encoded[:, 0] == 3):
        msg = "skin quality expects a non-empty triangle mesh"
        raise ValueError(msg)
    triangles = encoded[:, 1:]
    skin_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    result_ids = np.asarray(result.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if np.unique(result_ids).size != result.n_points:
        msg = "result GlobalPointId values are not unique"
        raise ValueError(msg)
    order = np.argsort(result_ids)
    positions = np.searchsorted(result_ids[order], skin_ids)
    if np.any(positions >= result_ids.size) or not np.array_equal(
        result_ids[order[positions]], skin_ids
    ):
        msg = "skin GlobalPointId values do not map to the result mesh"
        raise ValueError(msg)
    result_point_ids = order[positions]
    displacement = np.asarray(result.point_data["Displacement"], dtype=np.float64)[
        result_point_ids
    ]
    rest = np.asarray(skin.points, dtype=np.float64)
    deformed = rest + displacement

    def double_area_vectors(points: np.ndarray) -> np.ndarray:
        edge_1 = points[triangles[:, 1]] - points[triangles[:, 0]]
        edge_2 = points[triangles[:, 2]] - points[triangles[:, 0]]
        return np.cross(edge_1, edge_2)

    rest_vector = double_area_vectors(rest)
    deformed_vector = double_area_vectors(deformed)
    rest_norm = np.linalg.norm(rest_vector, axis=1)
    deformed_norm = np.linalg.norm(deformed_vector, axis=1)
    if np.any(rest_norm <= np.finfo(np.float64).eps):
        msg = "skin quality reference contains degenerate triangles"
        raise ValueError(msg)
    area_ratio = deformed_norm / rest_norm
    signed_normal_ratio = np.einsum(
        "ij,ij->i", deformed_vector, rest_vector
    ) / np.square(rest_norm)
    normal_cosine = np.einsum("ij,ij->i", deformed_vector, rest_vector) / np.maximum(
        rest_norm * deformed_norm, np.finfo(np.float64).tiny
    )
    if not (
        np.isfinite(area_ratio).all()
        and np.isfinite(signed_normal_ratio).all()
        and np.isfinite(normal_cosine).all()
    ):
        msg = "skin deformation quality contains non-finite values"
        raise ValueError(msg)
    folded = signed_normal_ratio <= 0.0
    return {
        "quality/skin_triangles": int(triangles.shape[0]),
        "quality/skin_folded_triangles": int(folded.sum()),
        "quality/skin_folded_triangle_fraction": float(folded.mean()),
        "quality/skin_area_ratio_min": float(area_ratio.min()),
        "quality/skin_area_ratio_q001": float(np.quantile(area_ratio, 0.001)),
        "quality/skin_area_ratio_median": float(np.median(area_ratio)),
        "quality/skin_area_ratio_q999": float(np.quantile(area_ratio, 0.999)),
        "quality/skin_area_ratio_max": float(area_ratio.max()),
        "quality/skin_signed_normal_ratio_min": float(signed_normal_ratio.min()),
        "quality/skin_signed_normal_ratio_q001": float(
            np.quantile(signed_normal_ratio, 0.001)
        ),
        "quality/skin_normal_cosine_min": float(normal_cosine.min()),
        "quality/skin_normal_cosine_q001": float(np.quantile(normal_cosine, 0.001)),
    }


def muscle_activation_spd_metrics(result: pv.UnstructuredGrid) -> dict[str, Any]:
    activation = np.asarray(
        result.cell_data["RecoveredActivationInv"], dtype=np.float64
    )
    active = np.asarray(result.cell_data["ActivationMask"], dtype=bool)
    if activation.shape != (result.n_cells, 6):
        msg = (
            f"RecoveredActivationInv shape {activation.shape} differs from "
            f"{(result.n_cells, 6)}"
        )
        raise ValueError(msg)
    if not np.isfinite(activation).all() or not np.any(active):
        msg = "active muscle ActivationInv is empty or non-finite"
        raise ValueError(msg)
    values = activation[active]
    matrices = np.zeros((values.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 + values[:, 0]
    matrices[:, 1, 1] = 1.0 + values[:, 1]
    matrices[:, 2, 2] = 1.0 + values[:, 2]
    matrices[:, 0, 1] = matrices[:, 1, 0] = values[:, 3]
    matrices[:, 1, 2] = matrices[:, 2, 1] = values[:, 4]
    matrices[:, 0, 2] = matrices[:, 2, 0] = values[:, 5]
    eigenvalues = np.linalg.eigvalsh(matrices)
    minimum = eigenvalues[:, 0]
    maximum = eigenvalues[:, 2]
    determinant = np.linalg.det(matrices)
    positive = minimum > 0.0
    positive_condition = maximum[positive] / minimum[positive]
    if not np.isfinite(positive_condition).all():
        msg = "positive-SPD muscle activation condition numbers are non-finite"
        raise ValueError(msg)
    return {
        "quality/muscle_activation_tets": int(values.shape[0]),
        "quality/muscle_activation_non_spd_tets": int((~positive).sum()),
        "quality/muscle_activation_min_eigenvalue": float(minimum.min()),
        "quality/muscle_activation_min_eigenvalue_q001": float(
            np.quantile(minimum, 0.001)
        ),
        "quality/muscle_activation_min_eigenvalue_median": float(np.median(minimum)),
        "quality/muscle_activation_max_eigenvalue_q999": float(
            np.quantile(maximum, 0.999)
        ),
        "quality/muscle_activation_max_eigenvalue": float(maximum.max()),
        "quality/muscle_activation_determinant_min": float(determinant.min()),
        "quality/muscle_activation_condition_q999": float(
            np.quantile(positive_condition, 0.999)
        ),
        "quality/muscle_activation_condition_max": float(positive_condition.max()),
    }


def physical_quality_gates(
    quality: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, bool]:
    return {
        "scientific/gate_detF_no_inversions": int(quality["quality/inverted_tets"])
        == 0,
        "scientific/gate_detF_min_positive": float(quality["quality/detF_min"]) > 0.0,
        "scientific/gate_detF_q001": float(quality["quality/detF_q001"])
        >= float(protocol["min_det_f_q001"]),
        "scientific/gate_skin_no_folds": int(quality["quality/skin_folded_triangles"])
        == 0,
        "scientific/gate_skin_area_q001": float(quality["quality/skin_area_ratio_q001"])
        >= float(protocol["min_skin_area_ratio_q001"]),
        "scientific/gate_skin_area_q999": float(quality["quality/skin_area_ratio_q999"])
        <= float(protocol["max_skin_area_ratio_q999"]),
        "scientific/gate_muscle_activation_spd": int(
            quality["quality/muscle_activation_non_spd_tets"]
        )
        == 0
        and float(quality["quality/muscle_activation_min_eigenvalue"])
        >= float(protocol["min_muscle_activation_eigenvalue"]),
    }


def _optional_finite(value: Any) -> float | int | bool | str | None:
    if isinstance(value, bool | int | str):
        return value
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _displacement_hash(displacement: np.ndarray) -> str:
    import hashlib

    canonical = np.ascontiguousarray(displacement, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _read_worker_skin(candidate: dict[str, Any]) -> pv.PolyData:
    path = Path(str(candidate["skin/path"])).resolve()
    _verify_file_identity(
        path, candidate["file_identity"], f"worker candidate {candidate['label']}"
    )
    skin = pv.read(path)
    if not isinstance(skin, pv.PolyData):
        msg = f"{path} read as {type(skin).__name__}, expected PolyData"
        raise TypeError(msg)
    if skin.n_points != int(candidate["n_points"]) or skin.n_cells != int(
        candidate["n_triangles"]
    ):
        msg = f"worker candidate {candidate['label']} dimensions changed"
        raise ValueError(msg)
    actual = _skin_hashes(skin)
    expected = {
        name: str(candidate[name])
        for name in (
            "topology_sha256",
            "material_sha256",
            "solver_sha256",
        )
    }
    if actual != expected:
        msg = (
            f"worker candidate {candidate['label']} content mismatch: "
            f"expected {expected}, got {actual}"
        )
        raise ValueError(msg)
    return skin


def run_static_case(  # noqa: C901, PLR0912, PLR0915
    *,
    base_mesh: pv.UnstructuredGrid,
    candidate: dict[str, Any],
    order_position: int,
    replicate_id: str,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    label = str(candidate["label"])
    case_mesh = base_mesh.copy(deep=True)
    case = InverseCase(
        target="smile",
        lr=0.0,
        setup=SETUP_SKIN_ESTIMATED_PRESTRAIN,
        label=f"static-{replicate_id}-{label}",
    )
    target, mask, target_metrics = target_displacement_and_mask(case_mesh, case, None)
    if not np.isfinite(target).all() or not np.any(mask):
        msg = f"{label} target or loss mask is invalid"
        raise ValueError(msg)
    target_rms = float(np.linalg.norm(target[mask]) / math.sqrt(int(mask.sum())))
    if not math.isfinite(target_rms) or target_rms <= 0.0:
        msg = f"{label} target RMS must be finite and positive, got {target_rms}"
        raise ValueError(msg)

    skin = _read_worker_skin(candidate)
    builder_calls = 0
    builder_calls += 1
    forward, quality_skin = build_static_forward(case_mesh, skin)
    if builder_calls != 1:
        msg = f"{label} constructed {builder_calls} forward models"
        raise RuntimeError(msg)

    initial_displacement = to_numpy(forward.state.u).astype(np.float64, copy=True)
    if initial_displacement.shape != (case_mesh.n_points, 3):
        msg = f"{label} initial displacement has shape {initial_displacement.shape}"
        raise ValueError(msg)
    if not np.allclose(initial_displacement, 0.0, rtol=0.0, atol=0.0):
        msg = f"{label} fresh forward state is not exact zero displacement"
        raise ValueError(msg)

    active_ids = np.flatnonzero(
        np.asarray(case_mesh.cell_data["ActivationMask"], dtype=bool)
    ).astype(np.int64)
    if active_ids.size == 0:
        msg = f"{label} has no active muscle tetrahedra"
        raise ValueError(msg)
    active_ids_t = torch.as_tensor(
        active_ids, dtype=torch.long, device=torch.get_default_device()
    )
    active_zero = torch.zeros(
        (active_ids.size, 6),
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    materials = material_tree(
        forward.model.get_materials(), active_zero, active_ids_t, case_mesh.n_cells
    )
    full_activation = to_numpy(materials["muscle"]["activation_inv"])
    if full_activation.shape != (case_mesh.n_cells, 6) or not np.allclose(
        full_activation, 0.0, rtol=0.0, atol=0.0
    ):
        msg = f"{label} explicit muscle ActivationInv is not exact zero"
        raise ValueError(msg)
    forward.model.set_materials(materials)

    stdout = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(stdout):
        solution = forward.step()
    elapsed_s = time.perf_counter() - start
    displacement = to_numpy(forward.state.u).astype(np.float64, copy=True)
    if displacement.shape != (case_mesh.n_points, 3):
        msg = f"{label} displacement has shape {displacement.shape}"
        raise ValueError(msg)
    if not np.isfinite(displacement).all():
        msg = f"{label} displacement contains non-finite values"
        raise ValueError(msg)
    live_activation = to_numpy(
        forward.model.get_materials()["muscle"]["activation_inv"]
    ).astype(np.float64, copy=True)
    if live_activation.shape != (case_mesh.n_cells, 6):
        msg = (
            f"{label} live post-forward ActivationInv has shape {live_activation.shape}"
        )
        raise ValueError(msg)
    if not np.isfinite(live_activation).all() or not np.allclose(
        live_activation, 0.0, rtol=0.0, atol=0.0
    ):
        msg = f"{label} live post-forward muscle ActivationInv is not exact zero"
        raise ValueError(msg)

    solver_raw = forward_solution_metrics(solution)
    solver_metrics = {key: _optional_finite(value) for key, value in solver_raw.items()}
    solver_nonfinite = sorted(
        key
        for key, value in solver_raw.items()
        if isinstance(value, float) and not math.isfinite(value)
    )
    residual = displacement[mask] - target[mask]
    error_rms = float(np.linalg.norm(residual) / math.sqrt(int(mask.sum())))
    fidelity = error_rms / target_rms
    displacement_rms = float(
        np.linalg.norm(displacement[mask]) / math.sqrt(int(mask.sum()))
    )
    displacement_max = float(np.linalg.norm(displacement, axis=1).max())
    if not np.isfinite(
        (error_rms, fidelity, displacement_rms, displacement_max, elapsed_s)
    ).all():
        msg = f"{label} static forward scalar metrics are non-finite"
        raise ValueError(msg)

    result = make_result_mesh(
        case_mesh,
        target,
        mask,
        displacement,
        live_activation,
        {},
    )
    quality = {
        **tetra_det_f_metrics(result),
        **skin_deformation_quality(result, quality_skin),
        **muscle_activation_spd_metrics(result),
    }
    gates = physical_quality_gates(quality, protocol)
    execution_finite = not solver_nonfinite
    forward_success = bool(solver_metrics["forward/success"])
    physical_pass = all(gates.values())
    admissible = forward_success and execution_finite and physical_pass
    if admissible:
        classification = "admissible"
    elif forward_success and execution_finite:
        classification = "physical-inadmissible"
    else:
        classification = "execution-invalid"

    global_ids = np.asarray(case_mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if global_ids.shape != (case_mesh.n_points,) or np.unique(global_ids).size != (
        case_mesh.n_points
    ):
        msg = f"{label} result mesh has invalid GlobalPointId values"
        raise ValueError(msg)
    row: dict[str, Any] = {
        "candidate": label,
        "candidate/young_min_scale": float(candidate["young_min_scale"]),
        "candidate/prestrain_gain": float(candidate["prestrain_gain"]),
        "replicate/id": replicate_id,
        "replicate/order_position": int(order_position),
        "builder/fresh_independent": True,
        "builder/calls": builder_calls,
        "initial_displacement/enabled": False,
        "initial_displacement/rms": 0.0,
        "initial_displacement/max_abs": 0.0,
        "activation/mode": "per-muscle-tet-6dof-static-zero",
        "activation_inv/rms": 0.0,
        "activation_inv/max_abs": 0.0,
        "activation_inv/live_post_forward_exact_zero": True,
        "activation_inv/live_post_forward_rms": float(
            np.linalg.norm(live_activation) / math.sqrt(live_activation.size)
        ),
        "activation_inv/live_post_forward_max_abs": float(
            np.abs(live_activation).max()
        ),
        "n_points": int(case_mesh.n_points),
        "n_tets": int(case_mesh.n_cells),
        "n_active_tets": int(active_ids.size),
        "n_activation_parameter_dofs": int(6 * active_ids.size),
        "inverse/optimizer_used": False,
        "adjoint/used": False,
        "forward/evaluations": 1,
        "forward/elapsed_s": float(elapsed_s),
        "forward/nonlinear_solver": "PNCG",
        "forward/rtol": float(FORWARD_RTOL),
        "forward/atol": float(FORWARD_ATOL),
        "forward/max_steps": int(FORWARD_MAX_STEPS),
        "forward/captured_stdout_bytes": len(stdout.getvalue().encode("utf-8")),
        "forward/nonfinite_solver_fields": solver_nonfinite,
        **solver_metrics,
        **target_metrics,
        "target/error_rms": error_rms,
        "target/error_rms_fraction_of_target": fidelity,
        "displacement/loss_mask_rms": displacement_rms,
        "displacement/all_max": displacement_max,
        "displacement/sha256": _displacement_hash(displacement),
        **quality,
        **gates,
        "scientific/execution_finite": execution_finite,
        "scientific/physical_pass": physical_pass,
        "scientific/classification": classification,
        "scientific/admissible": admissible,
        "status": "ok",
        "validation/errors": [],
    }
    _validate_finite_json(row)
    return row, displacement


def _validate_worker_request(  # noqa: C901
    request: dict[str, Any], request_path: Path
) -> None:
    _require_keys(
        request,
        (
            "schema_version",
            "formal",
            "source",
            "input_mesh",
            "manifest",
            "replicate",
            "protocol",
            "candidates",
            "output_npz",
            "output_result",
        ),
        "worker request",
    )
    if int(request["schema_version"]) != SCHEMA_VERSION:
        msg = f"worker request schema changed: {request['schema_version']}"
        raise ValueError(msg)
    source = request["source"]
    if not isinstance(source, dict):
        msg = "worker source identity must be an object"
        raise TypeError(msg)
    _verify_file_identity(Path(__file__), source["file_identity"], "worker source")
    if Path(str(source["path"])).resolve() != Path(__file__).resolve():
        msg = "worker source path differs from the running script"
        raise ValueError(msg)
    replicate = request["replicate"]
    if not isinstance(replicate, dict):
        msg = "worker replicate must be an object"
        raise TypeError(msg)
    replicate_id = str(replicate.get("id"))
    order = tuple(str(label) for label in replicate.get("order", []))
    candidates = request["candidates"]
    if not isinstance(candidates, list):
        msg = "worker candidates must be a list"
        raise TypeError(msg)
    if tuple(str(row.get("label")) for row in candidates) != order:
        msg = "worker candidate rows do not follow the requested order"
        raise ValueError(msg)
    if len(order) != len(set(order)) or not order:
        msg = f"worker order is empty or duplicated: {order}"
        raise ValueError(msg)
    if not set(order).issubset(FORMAL_LABELS):
        msg = f"worker order contains non-formal candidates: {order}"
        raise ValueError(msg)
    if bool(request["formal"]) and (
        replicate_id not in FORMAL_ORDERS or order != FORMAL_ORDERS[replicate_id]
    ):
        msg = f"formal worker {replicate_id} order changed: {order}"
        raise ValueError(msg)
    if request_path.resolve() == Path(str(request["output_result"])).resolve():
        msg = "worker request and result paths must differ"
        raise ValueError(msg)


def run_worker(  # noqa: C901, PLR0912, PLR0915
    request_path: Path,
) -> tuple[int, dict[str, Any]]:
    request = _load_json(request_path)
    _validate_worker_request(request, request_path)
    input_mesh = Path(str(request["input_mesh"]["path"])).resolve()
    manifest_path = Path(str(request["manifest"]["path"])).resolve()
    _verify_file_identity(
        input_mesh, request["input_mesh"]["file_identity"], "worker input mesh"
    )
    _verify_file_identity(
        manifest_path, request["manifest"]["file_identity"], "worker manifest"
    )
    _manifest, manifest_catalog = load_manifest(input_mesh, manifest_path)
    candidates = [dict(row) for row in request["candidates"]]
    for candidate in candidates:
        label = str(candidate["label"])
        if label in manifest_catalog:
            live = manifest_catalog[label]
            for key in (
                "young_min_scale",
                "prestrain_gain",
                "skin/path",
                "file_identity",
                "topology_sha256",
                "material_sha256",
                "solver_sha256",
            ):
                if candidate[key] != live[key]:
                    msg = f"worker candidate {label} differs from schema-2 at {key}"
                    raise ValueError(msg)

    configure_runtime()
    mesh = pv.read(input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    from liblaf.apple.common import GLOBAL_POINT_ID

    if GLOBAL_POINT_ID.vtk not in mesh.point_data:
        # ModelBuilder.add_vertices establishes this same canonical indexing.
        mesh.point_data[GLOBAL_POINT_ID.vtk] = np.arange(mesh.n_points, dtype=np.int64)
    global_ids = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if (
        global_ids.shape != (mesh.n_points,)
        or np.unique(global_ids).size != mesh.n_points
    ):
        msg = "worker input mesh has invalid GlobalPointId values"
        raise ValueError(msg)
    case = InverseCase(
        target="smile",
        lr=0.0,
        setup=SETUP_SKIN_ESTIMATED_PRESTRAIN,
        label="static-target",
    )
    target, mask, _target_metrics = target_displacement_and_mask(mesh, case, None)
    target_rms = float(np.linalg.norm(target[mask]) / math.sqrt(int(mask.sum())))

    rows: list[dict[str, Any]] = []
    displacements: list[np.ndarray] = []
    errors: list[str] = []
    for position, candidate in enumerate(candidates):
        label = str(candidate["label"])
        try:
            row, displacement = run_static_case(
                base_mesh=mesh,
                candidate=candidate,
                order_position=position,
                replicate_id=str(request["replicate"]["id"]),
                protocol=dict(request["protocol"]),
            )
            rows.append(row)
            displacements.append(displacement)
        except Exception as error:
            message = f"{label}: {type(error).__name__}: {error}"
            errors.append(message)
            rows.append(
                {
                    "candidate": label,
                    "replicate/id": str(request["replicate"]["id"]),
                    "replicate/order_position": position,
                    "status": "error",
                    "scientific/classification": "execution-error",
                    "scientific/admissible": False,
                    "validation/errors": [message],
                }
            )
            logger.exception("Static case %s failed", label)

    npz_path = Path(str(request["output_npz"])).resolve()
    npz_identity: dict[str, Any] | None = None
    if not errors and len(displacements) == len(candidates):
        stacked = np.stack(displacements, axis=0)
        if not np.isfinite(stacked).all():
            errors.append("stacked worker displacements contain non-finite values")
        else:
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            partial = npz_path.with_suffix(".partial.npz")
            np.savez_compressed(
                partial,
                labels=np.asarray(
                    [str(row["candidate"]) for row in rows], dtype=np.str_
                ),
                global_point_id=global_ids,
                loss_mask=mask.astype(np.uint8),
                target_rms=np.asarray(target_rms, dtype=np.float64),
                displacement=stacked,
                forward_success=np.asarray(
                    [bool(row["forward/success"]) for row in rows], dtype=np.uint8
                ),
                fidelity=np.asarray(
                    [float(row["target/error_rms_fraction_of_target"]) for row in rows],
                    dtype=np.float64,
                ),
                det_f_min=np.asarray(
                    [float(row["quality/detF_min"]) for row in rows],
                    dtype=np.float64,
                ),
                det_f_q001=np.asarray(
                    [float(row["quality/detF_q001"]) for row in rows],
                    dtype=np.float64,
                ),
                inverted_tets=np.asarray(
                    [int(row["quality/inverted_tets"]) for row in rows],
                    dtype=np.int64,
                ),
                skin_folded_triangles=np.asarray(
                    [int(row["quality/skin_folded_triangles"]) for row in rows],
                    dtype=np.int64,
                ),
                skin_area_ratio_q001=np.asarray(
                    [float(row["quality/skin_area_ratio_q001"]) for row in rows],
                    dtype=np.float64,
                ),
                skin_area_ratio_q999=np.asarray(
                    [float(row["quality/skin_area_ratio_q999"]) for row in rows],
                    dtype=np.float64,
                ),
                scientific_admissible=np.asarray(
                    [bool(row["scientific/admissible"]) for row in rows],
                    dtype=np.uint8,
                ),
            )
            partial.replace(npz_path)
            npz_identity = _file_identity(npz_path)

    source_after = _file_identity(Path(__file__))
    if source_after != request["source"]["file_identity"]:
        errors.append("worker source changed during execution")
    result = {
        "schema_version": SCHEMA_VERSION,
        "complete": not errors and len(rows) == len(candidates),
        "process": {
            "pid": os.getpid(),
            "uuid": str(uuid.uuid4()),
            "hostname": socket.gethostname(),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "source": {
            "path": str(Path(__file__).resolve()),
            "file_identity_before": request["source"]["file_identity"],
            "file_identity_after": source_after,
        },
        "replicate": request["replicate"],
        "protocol": request["protocol"],
        "rows": rows,
        "npz": {
            "path": str(npz_path),
            "file_identity": npz_identity,
        },
        "execution/errors": errors,
    }
    _write_json(Path(str(request["output_result"])), result)
    return (0 if result["complete"] else 2), result


def validate_config(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915
    if cfg.stage not in {"formal", "debug"}:
        msg = f"stage must be formal or debug, got {cfg.stage!r}"
        raise ValueError(msg)
    if (cfg.stage == "debug") != DEBUG_MODE:
        msg = "debug stage requires DEBUG=1, and DEBUG=1 cannot run formal stage"
        raise ValueError(msg)
    if str(mpl.get_backend()).lower() != "agg":
        msg = f"static sweep requires Agg, got {mpl.get_backend()}"
        raise RuntimeError(msg)
    if cfg.input_mesh.resolve() != Path(PREPARED_MESH).resolve():
        msg = (
            "static sweep only accepts the prepared Smile mesh "
            f"{Path(PREPARED_MESH).resolve()}"
        )
        raise ValueError(msg)
    if (
        len(
            {
                cfg.output_summary.resolve(),
                cfg.output_csv.resolve(),
                cfg.output_table.resolve(),
            }
        )
        != 3
    ):
        msg = "JSON, CSV, and Markdown outputs must be distinct"
        raise ValueError(msg)
    if not (
        cfg.output_summary.parent.resolve()
        == cfg.output_csv.parent.resolve()
        == cfg.output_table.parent.resolve()
    ):
        msg = "aggregate JSON, CSV, and Markdown must share one output directory"
        raise ValueError(msg)
    artifact_name = Path(cfg.artifact_dir_name)
    if artifact_name.name != cfg.artifact_dir_name or artifact_name.is_absolute():
        msg = "artifact_dir_name must be a single relative directory name"
        raise ValueError(msg)
    young = _parse_exact_float_list(cfg.young_min_scales, name="young_min_scales")
    gains = _parse_exact_float_list(cfg.prestrain_gains, name="prestrain_gains")
    if young != FORMAL_YOUNG_SCALES or gains != FORMAL_PRESTRAIN_GAINS:
        msg = (
            "static sweep requires the fixed 3x3 grid "
            f"{FORMAL_YOUNG_SCALES} x {FORMAL_PRESTRAIN_GAINS}, got "
            f"{young} x {gains}"
        )
        raise ValueError(msg)
    if cfg.order_contract != "lexicographic-low-to-high-and-exact-reverse-v1":
        msg = f"static worker order contract changed: {cfg.order_contract!r}"
        raise ValueError(msg)
    physical = {key: getattr(cfg, key) for key in EXPECTED_PHYSICAL_GATES}
    reproducibility = {key: getattr(cfg, key) for key in EXPECTED_REPRODUCIBILITY_GATES}
    if physical != EXPECTED_PHYSICAL_GATES:
        msg = f"physical gates changed: {physical}"
        raise ValueError(msg)
    if reproducibility != EXPECTED_REPRODUCIBILITY_GATES:
        msg = f"reproducibility gates changed: {reproducibility}"
        raise ValueError(msg)
    if not math.isfinite(cfg.worker_timeout_s) or cfg.worker_timeout_s <= 0.0:
        msg = "worker_timeout_s must be finite and positive"
        raise ValueError(msg)
    if cfg.stage == "formal" and cfg.replicate_count != 2:
        msg = "formal static sweep requires exactly two worker processes"
        raise ValueError(msg)
    if cfg.stage == "debug":
        if cfg.replicate_count != 1:
            msg = "debug static smoke requires exactly one worker process"
            raise ValueError(msg)
        if cfg.debug_label not in FORMAL_LABELS:
            msg = f"debug_label must be one of {FORMAL_LABELS}"
            raise ValueError(msg)


def _protocol() -> dict[str, Any]:
    return {
        "target": "Smile",
        "setup": "skin-material-static-zero-activation",
        "inverse_optimizer_used": False,
        "adjoint_used": False,
        "initial_displacement_used": False,
        "activation_mode": "per-muscle-tet-6dof-static-zero",
        "forward_evaluations_per_case": 1,
        "forward_solver": "PNCG",
        "forward_rtol": float(FORWARD_RTOL),
        "forward_atol": float(FORWARD_ATOL),
        "forward_max_steps": int(FORWARD_MAX_STEPS),
        **EXPECTED_PHYSICAL_GATES,
        **EXPECTED_REPRODUCIBILITY_GATES,
    }


def _selected_orders(cfg: Config) -> dict[str, tuple[str, ...]]:
    if cfg.stage == "formal":
        return FORMAL_ORDERS
    return {"r0": (cfg.debug_label,)}


def _worker_request(
    *,
    cfg: Config,
    replicate_id: str,
    order: tuple[str, ...],
    catalog: dict[str, dict[str, Any]],
    source_identity: dict[str, Any],
    manifest_identity: dict[str, Any],
    input_identity: dict[str, Any],
    npz_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "formal": cfg.stage == "formal",
        "source": {
            "path": str(Path(__file__).resolve()),
            "file_identity": source_identity,
        },
        "input_mesh": {
            "path": str(cfg.input_mesh.resolve()),
            "file_identity": input_identity,
        },
        "manifest": {
            "path": str(cfg.input_candidates.resolve()),
            "file_identity": manifest_identity,
        },
        "replicate": {
            "id": replicate_id,
            "index": int(replicate_id.removeprefix("r")),
            "order": list(order),
            "order_contract": cfg.order_contract,
        },
        "protocol": _protocol(),
        "candidates": [dict(catalog[label]) for label in order],
        "output_npz": str(npz_path.resolve()),
        "output_result": str(result_path.resolve()),
    }


def launch_workers(
    *,
    cfg: Config,
    catalog: dict[str, dict[str, Any]],
    artifact_dir: Path,
    source_identity: dict[str, Any],
    manifest_identity: dict[str, Any],
    input_identity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    orders = _selected_orders(cfg)
    worker_records: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    cfg.output_summary.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="40-static-worker-protocol-", dir=cfg.output_summary.parent
    ) as temporary:
        temp_root = Path(temporary)
        for replicate_id, order in orders.items():
            request_path = temp_root / f"request-{replicate_id}.json"
            result_path = temp_root / f"result-{replicate_id}.json"
            npz_path = artifact_dir / f"replicate-{replicate_id}.npz"
            request = _worker_request(
                cfg=cfg,
                replicate_id=replicate_id,
                order=order,
                catalog=catalog,
                source_identity=source_identity,
                manifest_identity=manifest_identity,
                input_identity=input_identity,
                npz_path=npz_path,
                result_path=result_path,
            )
            _write_json(request_path, request)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                WORKER_FLAG,
                str(request_path),
            ]
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = "0"
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=cfg.worker_timeout_s,
                )
                returncode: int | None = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
                timeout = False
            except subprocess.TimeoutExpired as error:
                returncode = None
                stdout = "" if error.stdout is None else str(error.stdout)
                stderr = "" if error.stderr is None else str(error.stderr)
                timeout = True
            elapsed_s = time.perf_counter() - started
            result = _load_json(result_path) if result_path.is_file() else None
            record: dict[str, Any] = {
                "replicate/id": replicate_id,
                "replicate/order": list(order),
                "process/returncode": returncode,
                "process/timeout": timeout,
                "process/elapsed_s": float(elapsed_s),
                "process/command": command,
                "process/stdout_tail": stdout[-4000:],
                "process/stderr_tail": stderr[-4000:],
                "result": result,
            }
            worker_records.append(record)
            if timeout:
                hard_failures.append(f"{replicate_id}: worker timed out")
            elif returncode != 0:
                hard_failures.append(
                    f"{replicate_id}: worker exited with status {returncode}"
                )
            if result is None:
                hard_failures.append(f"{replicate_id}: worker result JSON is missing")
            elif not bool(result.get("complete", False)):
                hard_failures.append(f"{replicate_id}: worker result is incomplete")
            if npz_path.is_file():
                cherries.log_output(npz_path)
                logger.info("Worker %s wrote %s", replicate_id, npz_path)
            else:
                hard_failures.append(f"{replicate_id}: compressed NPZ is missing")
    process_uuids = [
        str(record["result"]["process"]["uuid"])
        for record in worker_records
        if isinstance(record.get("result"), dict)
        and isinstance(record["result"].get("process"), dict)
        and record["result"]["process"].get("uuid") is not None
    ]
    if len(process_uuids) != len(set(process_uuids)):
        hard_failures.append("worker process UUIDs are not independent")
    return worker_records, hard_failures


def load_worker_npz(  # noqa: C901, PLR0912, PLR0915
    worker_record: dict[str, Any], expected_order: tuple[str, ...]
) -> dict[str, Any]:
    result = worker_record["result"]
    if not isinstance(result, dict):
        msg = "worker result is unavailable"
        raise TypeError(msg)
    npz_record = result["npz"]
    path = Path(str(npz_record["path"])).resolve()
    _verify_file_identity(path, npz_record["file_identity"], "worker NPZ")
    required = {
        "labels",
        "global_point_id",
        "loss_mask",
        "target_rms",
        "displacement",
        "forward_success",
        "fidelity",
        "det_f_min",
        "det_f_q001",
        "inverted_tets",
        "skin_folded_triangles",
        "skin_area_ratio_q001",
        "skin_area_ratio_q999",
        "scientific_admissible",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            msg = f"worker NPZ is missing arrays: {missing}"
            raise KeyError(msg)
        labels = tuple(str(label) for label in archive["labels"].tolist())
        global_ids = np.asarray(archive["global_point_id"], dtype=np.int64).copy()
        loss_mask = np.asarray(archive["loss_mask"], dtype=bool).copy()
        target_rms = float(np.asarray(archive["target_rms"]).item())
        displacement = np.asarray(archive["displacement"], dtype=np.float64).copy()
        fidelity = np.asarray(archive["fidelity"], dtype=np.float64).copy()
        forward_success = np.asarray(archive["forward_success"], dtype=bool).copy()
        admissible = np.asarray(archive["scientific_admissible"], dtype=bool).copy()
    if labels != expected_order:
        msg = f"worker NPZ labels/order changed: {labels}"
        raise ValueError(msg)
    if (
        displacement.ndim != 3
        or displacement.shape[0] != len(labels)
        or displacement.shape[2] != 3
    ):
        msg = f"worker displacement has invalid shape {displacement.shape}"
        raise ValueError(msg)
    if (
        global_ids.shape != (displacement.shape[1],)
        or np.unique(global_ids).size != global_ids.size
    ):
        msg = "worker NPZ GlobalPointId is invalid"
        raise ValueError(msg)
    if loss_mask.shape != (displacement.shape[1],) or not np.any(loss_mask):
        msg = "worker NPZ loss mask is invalid"
        raise ValueError(msg)
    if not (
        np.isfinite(displacement).all()
        and np.isfinite(fidelity).all()
        and math.isfinite(target_rms)
        and target_rms > 0.0
    ):
        msg = "worker NPZ contains non-finite numerical evidence"
        raise ValueError(msg)
    rows = list(result["rows"])
    if len(rows) != len(labels):
        msg = "worker result rows and NPZ labels differ in length"
        raise ValueError(msg)
    for index, row in enumerate(rows):
        if str(row["candidate"]) != labels[index]:
            msg = "worker JSON row order differs from NPZ"
            raise ValueError(msg)
        if _displacement_hash(displacement[index]) != str(row["displacement/sha256"]):
            msg = f"worker displacement hash mismatch for {labels[index]}"
            raise ValueError(msg)
        if fidelity[index] != float(row["target/error_rms_fraction_of_target"]):
            msg = f"worker fidelity mismatch for {labels[index]}"
            raise ValueError(msg)
        if forward_success[index] != bool(row["forward/success"]):
            msg = f"worker forward-success mismatch for {labels[index]}"
            raise ValueError(msg)
        if admissible[index] != bool(row["scientific/admissible"]):
            msg = f"worker admissibility mismatch for {labels[index]}"
            raise ValueError(msg)
    return {
        "path": str(path),
        "labels": labels,
        "global_ids": global_ids,
        "loss_mask": loss_mask,
        "target_rms": target_rms,
        "displacement": displacement,
        "rows": rows,
    }


def _worst_physical_metrics(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    minimum_keys = (
        "quality/detF_min",
        "quality/detF_q001",
        "quality/skin_area_ratio_min",
        "quality/skin_area_ratio_q001",
        "quality/skin_signed_normal_ratio_min",
        "quality/skin_signed_normal_ratio_q001",
        "quality/skin_normal_cosine_min",
        "quality/skin_normal_cosine_q001",
        "quality/muscle_activation_min_eigenvalue",
        "quality/muscle_activation_min_eigenvalue_q001",
        "quality/muscle_activation_determinant_min",
    )
    maximum_keys = (
        "quality/inverted_tets",
        "quality/inverted_tet_fraction",
        "quality/detF_lt_0p2_tets",
        "quality/detF_lt_0p5_tets",
        "quality/skin_folded_triangles",
        "quality/skin_folded_triangle_fraction",
        "quality/skin_area_ratio_q999",
        "quality/skin_area_ratio_max",
        "quality/muscle_activation_non_spd_tets",
        "quality/muscle_activation_max_eigenvalue_q999",
        "quality/muscle_activation_max_eigenvalue",
        "quality/muscle_activation_condition_q999",
        "quality/muscle_activation_condition_max",
    )
    worst: dict[str, Any] = {}
    for key in minimum_keys:
        worst[f"worst/{key}"] = min(float(left[key]), float(right[key]))
    for key in maximum_keys:
        values = (left[key], right[key])
        worst[f"worst/{key}"] = (
            max(int(value) for value in values)
            if all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in values
            )
            else max(float(value) for value in values)
        )
    return worst


def aggregate_formal_cases(  # noqa: C901, PLR0915
    archives: dict[str, dict[str, Any]], cfg: Config
) -> list[dict[str, Any]]:
    r0 = archives["r0"]
    r1 = archives["r1"]
    if not np.array_equal(r0["global_ids"], r1["global_ids"]):
        msg = "replicate GlobalPointId arrays differ"
        raise ValueError(msg)
    if not np.array_equal(r0["loss_mask"], r1["loss_mask"]):
        msg = "replicate loss masks differ"
        raise ValueError(msg)
    if float(r0["target_rms"]) != float(r1["target_rms"]):
        msg = "replicate target RMS values differ"
        raise ValueError(msg)
    target_rms = float(r0["target_rms"])
    left_rows = {str(row["candidate"]): row for row in r0["rows"]}
    right_rows = {str(row["candidate"]): row for row in r1["rows"]}
    left_index = {label: index for index, label in enumerate(r0["labels"])}
    right_index = {label: index for index, label in enumerate(r1["labels"])}
    rows: list[dict[str, Any]] = []
    for label in FORMAL_LABELS:
        left = left_rows[label]
        right = right_rows[label]
        difference = (
            r0["displacement"][left_index[label]][r0["loss_mask"]]
            - r1["displacement"][right_index[label]][r1["loss_mask"]]
        )
        difference_rms = float(
            np.linalg.norm(difference) / math.sqrt(int(r0["loss_mask"].sum()))
        )
        difference_fraction = difference_rms / target_rms
        fidelity_left = float(left["target/error_rms_fraction_of_target"])
        fidelity_right = float(right["target/error_rms_fraction_of_target"])
        fidelity_difference = abs(fidelity_left - fidelity_right)
        classification_match = str(left["scientific/classification"]) == str(
            right["scientific/classification"]
        )
        both_success = bool(left["forward/success"]) and bool(right["forward/success"])
        both_finite = bool(left["scientific/execution_finite"]) and bool(
            right["scientific/execution_finite"]
        )
        both_physical = bool(left["scientific/physical_pass"]) and bool(
            right["scientific/physical_pass"]
        )
        fidelity_gate = fidelity_difference <= cfg.max_fidelity_difference
        displacement_gate = (
            difference_fraction <= cfg.max_displacement_difference_fraction_of_target
        )
        has_inversion_or_fold = (
            max(
                int(left["quality/inverted_tets"]),
                int(right["quality/inverted_tets"]),
            )
            > 0
            or max(
                int(left["quality/skin_folded_triangles"]),
                int(right["quality/skin_folded_triangles"]),
            )
            > 0
        )
        reasons: list[str] = []
        if not both_success:
            reasons.append("both replicate forward solves must succeed")
        if not both_finite:
            reasons.append("both replicate outputs must be finite")
        if not both_physical:
            reasons.append("both replicate physical-gate sets must pass")
        if has_inversion_or_fold:
            reasons.append("any replicate inversion or skin fold is branch-unstable")
        if not classification_match:
            reasons.append("replicate scientific classifications disagree")
        if not fidelity_gate:
            reasons.append("replicate fidelity difference exceeds 0.001")
        if not displacement_gate:
            reasons.append("replicate displacement difference exceeds 1% target RMS")
        robust_pass = not reasons
        branch_unstable = (
            has_inversion_or_fold
            or not classification_match
            or not fidelity_gate
            or not displacement_gate
        )
        young, gain = _candidate_parameters(label)
        row = {
            "candidate": label,
            "candidate/young_min_scale": young,
            "candidate/prestrain_gain": gain,
            "r0/classification": str(left["scientific/classification"]),
            "r1/classification": str(right["scientific/classification"]),
            "r0/forward_success": bool(left["forward/success"]),
            "r1/forward_success": bool(right["forward/success"]),
            "r0/physical_pass": bool(left["scientific/physical_pass"]),
            "r1/physical_pass": bool(right["scientific/physical_pass"]),
            "r0/fidelity": fidelity_left,
            "r1/fidelity": fidelity_right,
            "replicate/fidelity_difference": fidelity_difference,
            "replicate/displacement_difference_rms": difference_rms,
            "replicate/displacement_difference_fraction_of_target": (
                difference_fraction
            ),
            "replicate/classification_match": classification_match,
            "replicate/gate_fidelity_difference": fidelity_gate,
            "replicate/gate_displacement_difference": displacement_gate,
            "replicate/has_inversion_or_fold": has_inversion_or_fold,
            "scientific/both_forward_success": both_success,
            "scientific/both_finite": both_finite,
            "scientific/both_physical_pass": both_physical,
            "scientific/branch_unstable": branch_unstable,
            "scientific/robust_pass": robust_pass,
            "scientific/ineligible_reasons": reasons,
            **_worst_physical_metrics(left, right),
            "replicates": {"r0": left, "r1": right},
        }
        _validate_finite_json(row)
        rows.append(row)
    return rows


def safe_low_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scale: dict[float, list[dict[str, Any]]] = {
        young: [row for row in rows if float(row["candidate/young_min_scale"]) == young]
        for young in FORMAL_YOUNG_SCALES
    }
    row_pass = {
        f"e{round(100 * young):03d}": len(by_scale[young]) == 3
        and all(bool(row["scientific/robust_pass"]) for row in by_scale[young])
        for young in FORMAL_YOUNG_SCALES
    }
    if not row_pass["e100"]:
        safe_low: float | None = None
        status = "A1-fail"
        reason = (
            "the required E=1 high-endpoint row is not robustly admissible; "
            "the high endpoint must be redefined before Stage B"
        )
    elif row_pass["e025"]:
        safe_low: float | None = 0.25
        status = "pass"
        reason = (
            "the required E=1 high-endpoint row and all three E=0.25 "
            "prestrain branches robustly pass"
        )
    elif row_pass["e050"]:
        safe_low = 0.5
        status = "pass"
        reason = (
            "the required E=1 high-endpoint row and all three E=0.5 branches pass; "
            "the E=0.25 row did not robustly pass"
        )
    else:
        safe_low = None
        status = "A1-fail"
        reason = "neither the E=0.25 nor E=0.5 row robustly passes"
    return {
        "status": status,
        "safe_low": safe_low,
        "rule": (
            "the whole E=1 p={0.5,0.75,1} row must robustly pass; then choose "
            "0.25 iff its whole row passes, otherwise choose 0.5 iff its whole "
            "row passes; otherwise A1 fails"
        ),
        "row_pass": row_pass,
        "reason": reason,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "candidate",
        "young_min_scale",
        "prestrain_gain",
        "r0_classification",
        "r1_classification",
        "r0_fidelity",
        "r1_fidelity",
        "fidelity_difference",
        "displacement_difference_fraction_of_target",
        "worst_inverted_tets",
        "worst_det_f_q001",
        "worst_det_f_min",
        "worst_skin_folds",
        "worst_skin_area_q001",
        "worst_skin_area_q999",
        "classification_match",
        "branch_unstable",
        "robust_pass",
        "ineligible_reasons",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate": row["candidate"],
                    "young_min_scale": row["candidate/young_min_scale"],
                    "prestrain_gain": row["candidate/prestrain_gain"],
                    "r0_classification": row["r0/classification"],
                    "r1_classification": row["r1/classification"],
                    "r0_fidelity": row["r0/fidelity"],
                    "r1_fidelity": row["r1/fidelity"],
                    "fidelity_difference": row["replicate/fidelity_difference"],
                    "displacement_difference_fraction_of_target": row[
                        "replicate/displacement_difference_fraction_of_target"
                    ],
                    "worst_inverted_tets": row["worst/quality/inverted_tets"],
                    "worst_det_f_q001": row["worst/quality/detF_q001"],
                    "worst_det_f_min": row["worst/quality/detF_min"],
                    "worst_skin_folds": row["worst/quality/skin_folded_triangles"],
                    "worst_skin_area_q001": row["worst/quality/skin_area_ratio_q001"],
                    "worst_skin_area_q999": row["worst/quality/skin_area_ratio_q999"],
                    "classification_match": row["replicate/classification_match"],
                    "branch_unstable": row["scientific/branch_unstable"],
                    "robust_pass": row["scientific/robust_pass"],
                    "ineligible_reasons": "; ".join(
                        row["scientific/ineligible_reasons"]
                    ),
                }
            )


def write_table(
    path: Path,
    rows: list[dict[str, Any]],
    decision: dict[str, Any] | None,
    *,
    stage: str,
) -> None:
    lines = [
        "# Static material admissibility sweep",
        "",
        f"Stage: `{stage}`.",
        "",
    ]
    if decision is not None:
        lines.extend(
            [
                f"A1 decision: **{decision['status']}**; "
                f"safe_low = `{decision['safe_low']}`. {decision['reason']}",
                "",
            ]
        )
    if not rows:
        lines.extend(["No complete matched replicate rows are available.", ""])
    else:
        lines.extend(
            [
                "| candidate | R0 / R1 class | fidelity R0 / R1 | Δ fidelity | Δu / target | worst inv | worst detF q001 / min | worst folds | worst area q001 / q999 | stable | robust pass |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        lines.extend(
            (
                f"| {row['candidate']} | {row['r0/classification']} / "
                f"{row['r1/classification']} | {row['r0/fidelity']:.6g} / "
                f"{row['r1/fidelity']:.6g} | "
                f"{row['replicate/fidelity_difference']:.6g} | "
                f"{row['replicate/displacement_difference_fraction_of_target']:.6g} | "
                f"{row['worst/quality/inverted_tets']} | "
                f"{row['worst/quality/detF_q001']:.5g} / "
                f"{row['worst/quality/detF_min']:.5g} | "
                f"{row['worst/quality/skin_folded_triangles']} | "
                f"{row['worst/quality/skin_area_ratio_q001']:.5g} / "
                f"{row['worst/quality/skin_area_ratio_q999']:.5g} | "
                f"{'yes' if not row['scientific/branch_unstable'] else 'no'} | "
                f"{'yes' if row['scientific/robust_pass'] else 'no'} |"
            )
            for row in rows
        )
        lines.append("")
    lines.extend(
        [
            "Robust pass requires two successful finite forwards, all physical gates "
            "in both branches, matching classifications, absolute fidelity difference "
            "at most 0.001, and loss-ROI displacement disagreement at most 1% of "
            "target RMS.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915
    validate_config(cfg)
    source_identity_before = _file_identity(Path(__file__))
    input_identity_before = _file_identity(cfg.input_mesh)
    manifest_identity_before = _file_identity(cfg.input_candidates)
    manifest, manifest_catalog = load_manifest(cfg.input_mesh, cfg.input_candidates)
    base_mesh = pv.read(cfg.input_mesh)
    if not isinstance(base_mesh, pv.UnstructuredGrid):
        base_mesh = base_mesh.cast_to_unstructured_grid()

    orders = _selected_orders(cfg)
    selected_labels = tuple(
        label
        for label in FORMAL_LABELS
        if any(label in order for order in orders.values())
    )
    artifact_dir = (
        cfg.output_summary.parent.resolve() / cfg.artifact_dir_name
    ).resolve()
    catalog, generated_rows = prepare_formal_catalog(
        cfg=cfg,
        base_mesh=base_mesh,
        manifest=manifest,
        manifest_catalog=manifest_catalog,
        selected_labels=selected_labels,
        artifact_dir=artifact_dir,
    )
    anchors = [
        {
            **manifest_catalog[label],
            "executed": False,
            "role": "hash-bound-p0-anchor-only",
            "reason": "p=0 is outside the pre-registered 3x3 static forward grid",
        }
        for label in ANCHOR_LABELS
    ]
    worker_records, hard_failures = launch_workers(
        cfg=cfg,
        catalog=catalog,
        artifact_dir=artifact_dir,
        source_identity=source_identity_before,
        manifest_identity=manifest_identity_before,
        input_identity=input_identity_before,
    )

    archives: dict[str, dict[str, Any]] = {}
    for record in worker_records:
        replicate_id = str(record["replicate/id"])
        try:
            archives[replicate_id] = load_worker_npz(
                record, tuple(record["replicate/order"])
            )
        except Exception as error:  # noqa: BLE001
            hard_failures.append(
                f"{replicate_id}: NPZ validation failed: {type(error).__name__}: {error}"
            )

    robust_rows: list[dict[str, Any]] = []
    provisional_decision: dict[str, Any] | None = None
    if cfg.stage == "formal" and set(archives) == {"r0", "r1"}:
        try:
            robust_rows = aggregate_formal_cases(archives, cfg)
            provisional_decision = safe_low_decision(robust_rows)
        except Exception as error:  # noqa: BLE001
            hard_failures.append(
                f"aggregate validation failed: {type(error).__name__}: {error}"
            )

    source_identity_after = _file_identity(Path(__file__))
    input_identity_after = _file_identity(cfg.input_mesh)
    manifest_identity_after = _file_identity(cfg.input_candidates)
    if source_identity_after != source_identity_before:
        hard_failures.append("40 source changed during the root/worker execution")
    if input_identity_after != input_identity_before:
        hard_failures.append("prepared input mesh changed during static sweep")
    if manifest_identity_after != manifest_identity_before:
        hard_failures.append("schema-2 manifest changed during static sweep")
    for label, candidate in catalog.items():
        try:
            _verify_file_identity(
                Path(str(candidate["skin/path"])),
                candidate["file_identity"],
                f"post-worker candidate {label}",
            )
        except Exception as error:  # noqa: BLE001
            hard_failures.append(
                f"{label}: candidate artifact changed: {type(error).__name__}: {error}"
            )

    expected_forward_count = 18 if cfg.stage == "formal" else 1
    completed_forward_count = sum(
        sum(
            row.get("status") == "ok" and int(row.get("forward/evaluations", 0)) == 1
            for row in record["result"]["rows"]
        )
        for record in worker_records
        if isinstance(record.get("result"), dict)
    )
    if completed_forward_count != expected_forward_count:
        hard_failures.append(
            f"expected {expected_forward_count} case rows, got {completed_forward_count}"
        )
    if cfg.stage == "formal" and len(generated_rows) != 5:
        hard_failures.append(
            f"formal run must persist five new skins, got {len(generated_rows)}"
        )

    hard_failures = sorted(set(hard_failures))
    decision: dict[str, Any] | None = provisional_decision
    if cfg.stage == "formal" and hard_failures:
        decision = {
            "status": "execution-invalid",
            "safe_low": None,
            "rule": (
                "no safe_low may be selected unless all provenance, worker, artifact, "
                "and expected-forward-count checks complete"
            ),
            "row_pass": None,
            "reason": "formal execution/provenance validation failed",
            "diagnostic_only_provisional_decision": provisional_decision,
        }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "complete": not hard_failures,
        "stage": cfg.stage,
        "experiment": "zero-activation-static-material-admissibility",
        "protocol": {
            **_protocol(),
            "grid": {
                "young_min_scales": list(FORMAL_YOUNG_SCALES),
                "prestrain_gains": list(FORMAL_PRESTRAIN_GAINS),
                "labels": list(FORMAL_LABELS),
            },
            "replicate_count": cfg.replicate_count,
            "independent_processes": True,
            "workers_run_sequentially": True,
            "fresh_builder_per_case": True,
            "orders": {key: list(value) for key, value in orders.items()},
            "order_contract": cfg.order_contract,
            "robust_pass_rule": (
                "both forward success and finite; both physical gate sets pass; "
                "classification matches; abs fidelity difference <= 0.001; "
                "loss-ROI RMS displacement difference / target RMS <= 0.01"
            ),
            "safe_low_rule": (
                "the full E=1 row must robustly pass; safe_low=0.25 additionally "
                "requires the full E=0.25 row, otherwise safe_low=0.5 requires "
                "the full E=0.5 row; any E=1 failure is A1-fail and requires a "
                "redefined high endpoint"
            ),
            "branch_disagreement_rule": (
                "any replicate inversion/fold, classification disagreement, or "
                "reproducibility disagreement marks the branch unstable and ineligible; "
                "a later single-case diagnostic cannot rescue it"
            ),
        },
        "provenance": {
            "source_path": str(Path(__file__).resolve()),
            "source_identity_before": source_identity_before,
            "source_identity_after": source_identity_after,
            "input_mesh": str(cfg.input_mesh.resolve()),
            "input_mesh_identity_before": input_identity_before,
            "input_mesh_identity_after": input_identity_after,
            "manifest": str(cfg.input_candidates.resolve()),
            "manifest_schema_version": int(manifest["schema_version"]),
            "manifest_identity_before": manifest_identity_before,
            "manifest_identity_after": manifest_identity_after,
        },
        "anchors": anchors,
        "candidate_catalog": [catalog[label] for label in selected_labels],
        "generated_candidates": generated_rows,
        "workers": worker_records,
        "expected_forward_count": expected_forward_count,
        "completed_forward_count": completed_forward_count,
        "cases": robust_rows,
        "safe_low": decision,
        "scientific": {
            "execution_complete": not hard_failures,
            "robust_pass_candidates": [
                row["candidate"]
                for row in robust_rows
                if bool(row["scientific/robust_pass"])
            ],
            "branch_unstable_candidates": [
                row["candidate"]
                for row in robust_rows
                if bool(row["scientific/branch_unstable"])
            ],
            "A1_status": None if decision is None else decision["status"],
            "safe_low": None if decision is None else decision["safe_low"],
        },
        "hard_failures": hard_failures,
    }
    _write_json(cfg.output_summary, payload)
    write_csv(cfg.output_csv, robust_rows)
    write_table(cfg.output_table, robust_rows, decision, stage=cfg.stage)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_csv)
    logger.info("Wrote %s", cfg.output_table)
    if hard_failures:
        msg = "static material sweep execution failed: " + "; ".join(hard_failures)
        raise RuntimeError(msg)


def worker_entry(request_path: Path) -> int:
    try:
        return run_worker(request_path)[0]
    except Exception as error:  # noqa: BLE001
        message = f"{type(error).__name__}: {error}"
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            result_path = Path(str(request["output_result"]))
            failure = {
                "schema_version": SCHEMA_VERSION,
                "complete": False,
                "process": {
                    "pid": os.getpid(),
                    "uuid": str(uuid.uuid4()),
                    "hostname": socket.gethostname(),
                    "python": sys.version,
                    "python_executable": sys.executable,
                    "platform": platform.platform(),
                },
                "source": {
                    "path": str(Path(__file__).resolve()),
                    "file_identity_after": _file_identity(Path(__file__)),
                },
                "replicate": request.get("replicate"),
                "protocol": request.get("protocol"),
                "rows": [],
                "npz": {"path": request.get("output_npz"), "file_identity": None},
                "execution/errors": [message],
            }
            _write_json(result_path, failure)
        except Exception:  # noqa: BLE001, S110
            pass
        print(message, file=sys.stderr)
        return 3


if __name__ == "__main__":
    if WORKER_MODE:
        if len(sys.argv) != 3 or sys.argv[1] != WORKER_FLAG:
            print(
                f"worker usage: {Path(__file__).name} {WORKER_FLAG} REQUEST.json",
                file=sys.stderr,
            )
            raise SystemExit(2)
        raise SystemExit(worker_entry(Path(sys.argv[2])))
    cherries.main(run)
