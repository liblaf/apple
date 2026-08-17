from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from _human_face_config import (
    ADAM_EPS,
    APONEUROSIS_E,
    APONEUROSIS_FRACTION,
    APONEUROSIS_NU,
    FAT_E,
    FAT_FRACTION,
    FAT_NU,
    FORWARD_ATOL,
    FORWARD_MAX_STEPS,
    FORWARD_RTOL,
    LOSS_SCALE,
    MUSCLE_E,
    MUSCLE_FRACTION,
    MUSCLE_NU,
    SETUP_NO_SKIN,
    SETUP_SKIN_ESTIMATED_PRESTRAIN,
    SETUP_SKIN_NO_PRESTRAIN,
    SKIN_THICKNESS,
    InverseCase,
    configure_runtime,
)
from _material_heuristics import (
    file_sha256,
    skin_material_content_hash,
    skin_solver_content_hash,
    skin_topology_content_hash,
)
from _reference import PREPARED_MESH, enable_reference_modules

from liblaf import cherries

mpl.use("Agg", force=True)

enable_reference_modules()

import _human_face_runtime as reference_runtime  # noqa: E402
from _human_face_case import solve_case  # noqa: E402
from _human_face_forward import set_volume_material  # noqa: E402
from _human_face_runtime import CasePaths  # noqa: E402

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 2
BASELINE_LABEL = "e100-p000"
NO_SKIN_LABEL = "no-skin"
EXPECTED_LABELS = (
    "e100-p000",
    "e100-p050",
    "e100-p100",
    "e025-p000",
    "e025-p050",
    "e025-p100",
)
EXPECTED_CANDIDATES = {
    "e100-p000": (1.0, 0.0),
    "e100-p050": (1.0, 0.5),
    "e100-p100": (1.0, 1.0),
    "e025-p000": (0.25, 0.0),
    "e025-p050": (0.25, 0.5),
    "e025-p100": (0.25, 1.0),
}
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
EXPECTED_SCIENTIFIC_GATES = {
    "max_relative_inverted_tets": 1.10,
    "max_relative_error_rms": 1.05,
    "min_det_f_q001": 0.20,
    "min_skin_area_ratio_q001": 0.10,
    "max_skin_area_ratio_q999": 10.0,
    "min_muscle_activation_eigenvalue": 1.0e-6,
}
EXPECTED_INVERSE_PROTOCOL = {
    "inverse_lr": 0.3,
    "loss_scale": LOSS_SCALE,
    "adam_eps": ADAM_EPS,
    "segment_steps": 8,
    "live_snapshot_interval": 0,
    "area_ratio_floor": 0.1,
    "diagnostic_min_delta_rel": 1.0e-3,
    "flat_log_slope_tol": 5.0e-3,
    "aggressive_lr_factor": 2.0,
    "slow_lr_factor": 1.5,
    "lr_shrink_factor": 0.5,
    "max_lr": 1.0,
    "min_lr": 0.00375,
    "loss_deterioration_rel": 1.0e-2,
    "time_budget_hours": 10.0,
    "reserve_minutes": 5.0,
    "step_time_budget_s": 180.0,
    "require_convergence": False,
    "require_solver_success": True,
    "max_solver_failure_fraction": 0.05,
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_candidates: Path = cherries.input("10-material-candidates-manifest.json")
    output_summary: Path = cherries.output(
        "20-material-screen-summary.json", mkdir=True
    )
    output_table: Path = cherries.output("20-material-screen-table.md", mkdir=True)
    live_plot_dir: Path = Path("figs/live-material-screen")

    stage: str = "screen"
    candidate_set: str = "all-with-no-skin"
    initial_activation_mesh: Path | None = None
    use_initial_displacement: bool = False
    inverse_lr: float = 0.3
    loss_scale: float = LOSS_SCALE
    adam_eps: float = ADAM_EPS
    inverse_max_steps: int = 40
    mandatory_baseline_steps: int = 40
    segment_steps: int = 8
    live_snapshot_interval: int = 0
    area_ratio_floor: float = 0.1
    diagnostic_min_delta_rel: float = 1.0e-3
    flat_log_slope_tol: float = 5.0e-3
    aggressive_lr_factor: float = 2.0
    slow_lr_factor: float = 1.5
    lr_shrink_factor: float = 0.5
    max_lr: float = 1.0
    min_lr: float = 0.00375
    loss_deterioration_rel: float = 1.0e-2
    time_budget_hours: float = 10.0
    reserve_minutes: float = 5.0
    step_time_budget_s: float = 180.0
    require_convergence: bool = False
    require_solver_success: bool = True
    max_solver_failure_fraction: float = 0.05
    max_relative_inverted_tets: float = 1.10
    max_relative_error_rms: float = 1.05
    min_det_f_q001: float = 0.20
    min_skin_area_ratio_q001: float = 0.10
    max_skin_area_ratio_q999: float = 10.0
    min_muscle_activation_eigenvalue: float = 1.0e-6


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = sorted(set(keys) - set(mapping))
    if missing:
        msg = f"{context} is missing required keys: {missing}"
        raise KeyError(msg)


def _reject_json_constant(value: str) -> None:
    msg = f"material manifest contains non-standard JSON constant {value!r}"
    raise ValueError(msg)


def _validate_json_numbers(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{path} contains non-finite number {value}"
        raise ValueError(msg)
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json_numbers(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_numbers(item, path=f"{path}[{index}]")


def normalize_absent_initial_displacement(
    summary: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(summary)
    if normalized.get("initial_displacement/enabled") is not False:
        msg = "material inverse unexpectedly enabled an initial displacement"
        raise ValueError(msg)
    for key in ("initial_displacement/rms", "initial_displacement/max"):
        if key not in normalized:
            msg = f"material inverse summary is missing {key}"
            raise KeyError(msg)
        value = normalized.get(key)
        if value is not None and math.isfinite(float(value)):
            msg = (
                f"{key} must be absent/non-finite when initial displacement is disabled"
            )
            raise ValueError(msg)
        normalized[key] = None
    return normalized


def _verify_file_identity(path: Path, identity: Any, context: str) -> dict[str, Any]:
    if not isinstance(identity, dict):
        msg = f"{context} identity must be an object"
        raise TypeError(msg)
    _require_keys(identity, ("size_bytes", "sha256"), f"{context} identity")
    if not path.is_file():
        msg = f"{context} does not exist: {path}"
        raise FileNotFoundError(msg)
    actual = {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
    expected = {
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }
    if actual != expected:
        msg = f"{context} identity mismatch: expected {expected}, got {actual}"
        raise ValueError(msg)
    return actual


def _validate_candidate_record(row: dict[str, Any]) -> None:
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
        "readback/ok",
        "readback/errors",
        "readback/content/topology_sha256",
        "readback/content/material_sha256",
        "readback/content/solver_sha256",
        "validation/ok",
        "validation/errors",
    )
    label = str(row.get("label", "<missing>"))
    _require_keys(row, required, f"candidate {label}")
    if int(row["schema_version"]) != MANIFEST_SCHEMA_VERSION:
        msg = f"candidate {label} schema version changed: {row['schema_version']}"
        raise ValueError(msg)
    if label not in EXPECTED_CANDIDATES:
        msg = f"unexpected fixed-grid material candidate {label!r}"
        raise ValueError(msg)
    actual_parameters = (
        float(row["young_min_scale"]),
        float(row["prestrain_gain"]),
    )
    if actual_parameters != EXPECTED_CANDIDATES[label]:
        msg = (
            f"candidate {label} metadata {actual_parameters} does not match fixed "
            f"design {EXPECTED_CANDIDATES[label]}"
        )
        raise ValueError(msg)
    if not bool(row["validation/ok"]) or list(row["validation/errors"]):
        msg = f"candidate {label} failed material validation"
        raise ValueError(msg)
    if not bool(row["readback/ok"]) or list(row["readback/errors"]):
        msg = f"candidate {label} failed VTP readback validation"
        raise ValueError(msg)
    for name in ("topology", "material", "solver"):
        content = str(row[f"content/{name}_sha256"])
        readback = str(row[f"readback/content/{name}_sha256"])
        if content != readback:
            msg = f"candidate {label} {name} and readback hashes differ"
            raise ValueError(msg)


def load_manifest(cfg: Config) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    manifest = json.loads(
        cfg.input_candidates.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    _validate_json_numbers(manifest)
    if not isinstance(manifest, dict):
        msg = f"material manifest must contain an object: {cfg.input_candidates}"
        raise TypeError(msg)
    _require_keys(
        manifest,
        (
            "schema_version",
            "complete",
            "input_mesh",
            "input_mesh_identity",
            "input_mesh_identity_verified_stable",
            "grid",
            "target",
            "selection",
            "heuristic",
            "material_gates",
            "surface_geometry",
            "primary_signed_heat_field",
            "field_sensitivity",
            "validation_errors",
            "candidate_validation_errors",
            "n_candidates",
            "candidates",
        ),
        "material manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        msg = f"unsupported material manifest schema: {manifest['schema_version']}"
        raise ValueError(msg)
    if not bool(manifest["complete"]):
        msg = f"material manifest is incomplete: {cfg.input_candidates}"
        raise ValueError(msg)
    if list(manifest["validation_errors"]) or dict(
        manifest["candidate_validation_errors"]
    ):
        msg = "material manifest contains validation errors"
        raise ValueError(msg)
    if not bool(manifest["input_mesh_identity_verified_stable"]):
        msg = "material manifest did not verify stable input-mesh identity"
        raise ValueError(msg)
    if Path(str(manifest["input_mesh"])).resolve() != cfg.input_mesh.resolve():
        msg = "material manifest input mesh differs from inverse input mesh"
        raise ValueError(msg)
    _verify_file_identity(
        cfg.input_mesh, manifest["input_mesh_identity"], "inverse input mesh"
    )
    if manifest["grid"] != {
        "young_min_scales": [1.0, 0.25],
        "prestrain_gains": [0.0, 0.5, 1.0],
    }:
        msg = f"material manifest grid is not the fixed 2x3 grid: {manifest['grid']}"
        raise ValueError(msg)
    if manifest["target"] != "Smile":
        msg = f"material manifest target is not Smile: {manifest['target']!r}"
        raise ValueError(msg)
    if manifest["selection"] != (
        "all surface-triangle vertices are finite IsFace points"
    ):
        msg = f"material manifest selection rule changed: {manifest['selection']!r}"
        raise ValueError(msg)
    heuristic = manifest["heuristic"]
    if not isinstance(heuristic, dict):
        msg = "material manifest heuristic must be an object"
        raise TypeError(msg)
    actual_heuristic = {key: heuristic.get(key) for key in EXPECTED_HEURISTIC}
    if actual_heuristic != EXPECTED_HEURISTIC:
        msg = (
            "material manifest does not use the fixed 1%/99%/5mm heuristic and "
            f"existing Lame convention: {actual_heuristic}"
        )
        raise ValueError(msg)
    if manifest["material_gates"] != EXPECTED_MATERIAL_GATES:
        msg = (
            f"material manifest calibrated gates changed: {manifest['material_gates']}"
        )
        raise ValueError(msg)
    heat = manifest["primary_signed_heat_field"]
    if not isinstance(heat, dict) or not bool(heat.get("validation/ok", False)):
        msg = "primary signed heat field did not pass validation"
        raise ValueError(msg)
    if list(heat.get("validation/errors", [])):
        msg = "primary signed heat field contains validation errors"
        raise ValueError(msg)
    heat_metrics = heat.get("metrics")
    if not isinstance(heat_metrics, dict):
        msg = "primary signed heat field metrics must be an object"
        raise TypeError(msg)
    expected_heat = {
        "cap_quantile": 0.99,
        "diffusion_length": 0.005,
        "soft_deadband_log": math.log1p(0.01),
    }
    actual_heat = {key: heat_metrics.get(key) for key in expected_heat}
    if actual_heat != expected_heat:
        msg = f"primary signed heat field parameters changed: {actual_heat}"
        raise ValueError(msg)
    candidates = manifest["candidates"]
    if not isinstance(candidates, list):
        msg = "material manifest candidates must be a list"
        raise TypeError(msg)
    if int(manifest["n_candidates"]) != len(candidates) or len(candidates) != 6:
        msg = "material manifest must contain exactly six candidates"
        raise ValueError(msg)
    for row in candidates:
        if not isinstance(row, dict):
            msg = "material manifest candidate rows must be objects"
            raise TypeError(msg)
        _validate_candidate_record(row)
    labels = tuple(str(row["label"]) for row in candidates)
    if labels != EXPECTED_LABELS:
        msg = f"material manifest labels/order changed: {labels}"
        raise ValueError(msg)
    return manifest


def select_candidates(
    manifest: dict[str, Any], candidate_set: str
) -> list[dict[str, Any]]:
    candidates = [dict(row) for row in manifest["candidates"]]
    by_label = {str(row["label"]): row for row in candidates}
    corners = ["e100-p000", "e100-p100", "e025-p000", "e025-p100"]
    aliases = {
        "all": list(by_label),
        "all-with-no-skin": [*by_label, NO_SKIN_LABEL],
        "baseline": [BASELINE_LABEL],
        "combined": [BASELINE_LABEL, "e025-p100"],
        "corners": corners,
        "corners-with-no-skin": [*corners, NO_SKIN_LABEL],
        NO_SKIN_LABEL: [NO_SKIN_LABEL],
    }
    labels = aliases.get(
        candidate_set,
        [item.strip() for item in candidate_set.split(",") if item.strip()],
    )
    if not labels:
        msg = "candidate_set selected no candidates"
        raise ValueError(msg)
    if len(set(labels)) != len(labels):
        msg = f"candidate_set contains duplicate labels: {labels}"
        raise ValueError(msg)
    unknown = sorted(set(labels) - {*by_label, NO_SKIN_LABEL})
    if unknown:
        msg = f"unknown material candidates {unknown}; available={sorted(by_label)}"
        raise ValueError(msg)
    material_labels = [label for label in labels if label != NO_SKIN_LABEL]
    if any(label != BASELINE_LABEL for label in material_labels) and (
        BASELINE_LABEL not in material_labels
    ):
        msg = "non-baseline material candidates require e100-p000 in the same run"
        raise ValueError(msg)
    return [
        {"label": NO_SKIN_LABEL, "control/type": NO_SKIN_LABEL}
        if label == NO_SKIN_LABEL
        else by_label[label]
        for label in labels
    ]


def candidate_skin_path(cfg: Config, candidate: dict[str, Any]) -> Path:
    path = Path(str(candidate["skin/path"]))
    path = path if path.is_absolute() else cfg.input_candidates.parent / path
    resolved = path.resolve()
    root = cfg.input_candidates.parent.resolve()
    if not resolved.is_relative_to(root):
        msg = f"candidate skin escapes manifest data directory: {path}"
        raise ValueError(msg)
    return resolved


def verified_candidate_skin(
    cfg: Config, candidate: dict[str, Any]
) -> tuple[Path, pv.PolyData, dict[str, Any]]:
    path = candidate_skin_path(cfg, candidate)
    file_identity = _verify_file_identity(
        path, candidate["skin/file_identity"], f"candidate {candidate['label']} skin"
    )
    skin = pv.read(path)
    if not isinstance(skin, pv.PolyData):
        msg = f"{path} read as {type(skin).__name__}, expected PolyData"
        raise TypeError(msg)
    actual = {
        "topology": skin_topology_content_hash(skin),
        "material": skin_material_content_hash(skin),
        "solver": skin_solver_content_hash(skin),
    }
    for name, digest in actual.items():
        if digest != str(candidate[f"content/{name}_sha256"]):
            msg = f"candidate {candidate['label']} live {name} hash mismatch"
            raise ValueError(msg)
    if skin.n_points != int(candidate["content/n_points"]) or skin.n_cells != int(
        candidate["content/n_triangles"]
    ):
        msg = f"candidate {candidate['label']} live VTP dimensions changed"
        raise ValueError(msg)
    provenance = {
        "provenance/skin_file_size_bytes": int(file_identity["size_bytes"]),
        "provenance/skin_file_sha256": str(file_identity["sha256"]),
        "provenance/skin_topology_sha256": actual["topology"],
        "provenance/skin_material_sha256": actual["material"],
        "provenance/skin_solver_sha256": actual["solver"],
    }
    return path, skin, provenance


def build_candidate_forward(
    mesh: pv.UnstructuredGrid,
    _case: InverseCase,
    *,
    area_ratio_floor: float,
    skin_path: Path,
    verified_skin: pv.PolyData,
    candidate: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[Any, pv.PolyData, dict[str, Any]]:
    del area_ratio_floor
    from liblaf.apple.common import GLOBAL_POINT_ID
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import Koiter, StableNeoHookean, StableNeoHookeanActive

    skin = verified_skin.copy(deep=True)
    global_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if global_ids.size != skin.n_points or np.unique(global_ids).size != skin.n_points:
        msg = f"{skin_path} has invalid or duplicate GlobalPointId values"
        raise ValueError(msg)
    if global_ids.min() < 0 or global_ids.max() >= mesh.n_points:
        msg = f"{skin_path} GlobalPointId values escape mesh point range"
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
        Koiter.from_pyvista(skin, name="skin", thickness=SKIN_THICKNESS)
    )
    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=FORWARD_MAX_STEPS,
        atol=FORWARD_ATOL,
        rtol=FORWARD_RTOL,
    )
    skin_metrics = {
        **candidate,
        "material/candidate": str(candidate["label"]),
        "material/skin_path": str(skin_path),
        "skin/enabled": True,
        "skin/prestrain_enabled": float(candidate["prestrain_gain"]) > 0.0,
        "skin/young_spatially_varying": float(candidate["young_min_scale"]) < 1.0,
        **provenance,
    }
    return forward, skin, skin_metrics


def tetra_det_f_metrics(result: pv.UnstructuredGrid) -> dict[str, Any]:
    encoded = np.asarray(result.cells, dtype=np.int64).reshape(-1, 5)
    if not np.all(encoded[:, 0] == 4):
        msg = "deformation quality check expects tetrahedral cells"
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
        msg = "skin quality check expects a non-empty triangle mesh"
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
            "RecoveredActivationInv has shape "
            f"{activation.shape}, expected {(result.n_cells, 6)}"
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
        "quality/muscle_activation_condition_q999": (
            float(np.quantile(positive_condition, 0.999))
            if positive_condition.size
            else None
        ),
        "quality/muscle_activation_condition_max": (
            float(positive_condition.max()) if positive_condition.size else None
        ),
    }


def physical_quality_gates(quality: dict[str, Any], cfg: Config) -> dict[str, bool]:
    return {
        "scientific/gate_detF_no_inversions": int(quality["quality/inverted_tets"])
        == 0,
        "scientific/gate_detF_min_positive": float(quality["quality/detF_min"]) > 0.0,
        "scientific/gate_detF_q001": float(quality["quality/detF_q001"])
        >= cfg.min_det_f_q001,
        "scientific/gate_skin_no_folds": int(quality["quality/skin_folded_triangles"])
        == 0,
        "scientific/gate_skin_area_q001": float(quality["quality/skin_area_ratio_q001"])
        >= cfg.min_skin_area_ratio_q001,
        "scientific/gate_skin_area_q999": float(quality["quality/skin_area_ratio_q999"])
        <= cfg.max_skin_area_ratio_q999,
        "scientific/gate_muscle_activation_spd": int(
            quality["quality/muscle_activation_non_spd_tets"]
        )
        == 0
        and float(quality["quality/muscle_activation_min_eigenvalue"])
        >= cfg.min_muscle_activation_eigenvalue,
    }


def validate_case(  # noqa: C901, PLR0912, PLR0915
    summary: dict[str, Any],
    result_path: Path,
    quality_skin: pv.PolyData,
    cfg: Config,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    if float(summary["activation_inv/initial_rms"]) != 0.0:
        errors.append("initial activation RMS is not zero")
    if float(summary["activation_inv/initial_max_abs"]) != 0.0:
        errors.append("initial activation max is not zero")
    if summary.get("initial_displacement/enabled", True):
        errors.append("initial forward displacement was reused")
    if summary.get("activation/mode") != "per-muscle-tet-6dof":
        errors.append("activation mode is not per-muscle-tet-6dof")
    if int(summary["n_activation_parameter_dofs"]) != 6 * int(summary["n_active_tets"]):
        errors.append("activation DoF count is not six per active muscle tet")
    expected_evaluations = cfg.inverse_max_steps + 1
    if summary.get("baseline/completed") is not True:
        errors.append("fixed-budget baseline did not complete")
    evaluation_keys = (
        "baseline/evaluations",
        "baseline/evaluations_expected",
        "inverse/evaluations",
    )
    errors.extend(
        f"{key}={summary.get(key)} does not equal {expected_evaluations}"
        for key in evaluation_keys
        if int(summary.get(key, -1)) != expected_evaluations
    )
    if int(summary.get("baseline/mandatory_optimizer_steps", -1)) != (
        cfg.inverse_max_steps
    ):
        errors.append("mandatory optimizer-step budget changed")
    if float(summary.get("baseline/fixed_lr", math.nan)) != cfg.inverse_lr:
        errors.append("fixed-budget learning rate changed")
    if int(summary.get("baseline/lr_deviation_count", -1)) != 0:
        errors.append("fixed-budget trajectory changed learning rate")

    trace = list(summary.get("trace", []))
    if not trace:
        errors.append("inverse trace is empty")
    else:
        actual_steps = [int(row["step"]) for row in trace]
        if actual_steps != list(range(expected_evaluations)):
            errors.append(
                "inverse trace does not contain the complete ordered fixed-budget steps"
            )
        step_zero_rows = [row for row in trace if int(row["step"]) == 0]
        if len(step_zero_rows) != 1:
            errors.append("inverse trace does not contain exactly one step-0 row")
        else:
            step_zero = step_zero_rows[0]
            if float(step_zero["activation_inv/rms"]) != 0.0:
                errors.append("step-0 activation RMS is not exactly zero")
            if float(step_zero["activation_inv/max_abs"]) != 0.0:
                errors.append("step-0 activation max is not exactly zero")
            if not bool(step_zero.get("forward/success", False)):
                errors.append("step-0 forward solve did not succeed")
            if not bool(step_zero.get("adjoint/success", False)):
                errors.append("step-0 adjoint solve did not succeed")
        best_step = int(summary["best/step"])
        best_rows = [row for row in trace if int(row["step"]) == best_step]
        if len(best_rows) != 1:
            errors.append(f"best step {best_step} does not identify one trace row")
        elif not bool(best_rows[0].get("best/accepted", False)):
            errors.append("best inverse state was not accepted by the forward gate")
        elif not bool(best_rows[0].get("forward/success", False)) or not bool(
            best_rows[0].get("adjoint/success", False)
        ):
            errors.append("best inverse state lacks successful forward/adjoint solves")
        finite_keys = (
            "loss/total",
            "target/error_rms",
            "grad/norm",
            "forward/relative_grad_norm",
            "adjoint/relative_residual",
        )
        for row in trace:
            for key in finite_keys:
                if not math.isfinite(float(row[key])):
                    errors.append(f"step {int(row['step'])} has non-finite {key}")
                    break
        failure_fraction = (
            int(summary["inverse/forward_fail_count"])
            + int(summary["inverse/adjoint_fail_count"])
        ) / (2.0 * len(trace))
        if failure_fraction > cfg.max_solver_failure_fraction:
            errors.append(
                "solver failure fraction "
                f"{failure_fraction:.3%} exceeds {cfg.max_solver_failure_fraction:.3%}"
            )
        elif failure_fraction > 0.0:
            warnings.append(f"isolated solver failure fraction {failure_fraction:.3%}")
            errors.append("fixed-budget trajectory contains a solver failure")
    if not bool(summary.get("last/forward/success", False)):
        errors.append("last forward solve did not succeed")
    if not bool(summary.get("last/adjoint/success", False)):
        errors.append("last adjoint solve did not succeed")

    result = pv.read(result_path)
    if not isinstance(result, pv.UnstructuredGrid):
        errors.append(f"result read back as {type(result).__name__}")
        return errors, warnings, {}
    if result.n_points != int(summary["n_points"]):
        errors.append("result point count changed during readback")
    if result.n_cells != int(summary["n_tets"]):
        errors.append("result tet count changed during readback")
    quality = {
        **tetra_det_f_metrics(result),
        **skin_deformation_quality(result, quality_skin),
        **muscle_activation_spd_metrics(result),
    }
    gates = physical_quality_gates(quality, cfg)
    failed_gates = sorted(name for name, passed in gates.items() if not passed)
    if failed_gates:
        warnings.append(
            "best state failed scientific gates: " + ", ".join(failed_gates)
        )
    quality.update(gates)
    return errors, warnings, quality


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| candidate | status | Pareto | evals | best step | error/target | disp lap RMS | activation RMS | inv tets | detF q001/min | skin folds | skin area q001/q999 | muscle eig min | relative gate | stop |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"| {row['candidate']} | failed | no | - | - | - | - | - | - | - | - | - | - | - | `{row.get('error', '')}` |"
            )
            continue
        lines.append(
            "| {candidate} | ok | {pareto} | {inverse/evaluations} | {best/step} | "
            "{best/error_rms_fraction_of_target:.6g} | "
            "{bumpiness/displacement_laplacian_rms:.6g} | "
            "{activation_inv/rms:.6g} | {quality/inverted_tets} | "
            "{quality/detF_q001:.4g}/{quality/detF_min:.4g} | "
            "{quality/skin_folded_triangles} | "
            "{quality/skin_area_ratio_q001:.4g}/{quality/skin_area_ratio_q999:.4g} | "
            "{quality/muscle_activation_min_eigenvalue:.4g} | "
            "{scientific/quality_gate_relative_to_baseline} | "
            "{inverse/stop_reason} |".format(
                **row,
                pareto="yes" if row["scientific/eligible_for_pareto"] else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_relative_quality_gate(rows: list[dict[str, Any]], cfg: Config) -> None:
    valid_rows = [row for row in rows if row.get("status") == "ok"]
    baseline = next(
        (row for row in valid_rows if row["candidate"] == BASELINE_LABEL),
        None,
    )
    if baseline is None:
        for row in valid_rows:
            row["scientific/quality_gate_relative_to_baseline"] = None
            row["scientific/gate_relative_target_error"] = False
            row["scientific/gate_relative_inverted_tets"] = False
            row["scientific/baseline_available"] = False
        _update_pareto_eligibility(rows)
        return
    baseline_inverted = int(baseline["quality/inverted_tets"])
    baseline_error = float(baseline["best/error_rms"])
    allowed = max(
        baseline_inverted + 10,
        math.ceil(cfg.max_relative_inverted_tets * baseline_inverted),
    )
    for row in valid_rows:
        error_ratio = float(row["best/error_rms"]) / baseline_error
        inverted_gate = int(row["quality/inverted_tets"]) <= allowed
        target_gate = error_ratio <= cfg.max_relative_error_rms
        row["scientific/baseline_available"] = True
        row["scientific/baseline_inverted_tets"] = baseline_inverted
        row["scientific/max_allowed_inverted_tets"] = allowed
        row["scientific/error_rms_ratio_to_baseline"] = error_ratio
        row["scientific/max_error_rms_ratio_to_baseline"] = cfg.max_relative_error_rms
        row["scientific/gate_relative_inverted_tets"] = inverted_gate
        row["scientific/gate_relative_target_error"] = target_gate
        row["scientific/quality_gate_relative_to_baseline"] = (
            inverted_gate and target_gate
        )
    _update_pareto_eligibility(rows)


def _update_pareto_eligibility(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        reasons: list[str] = []
        if row.get("status") != "ok":
            reasons.append("case status is not ok")
        if bool(row.get("scientific/is_control", False)):
            reasons.append("control cases are not material candidates")
        for key, value in sorted(row.items()):
            if key.startswith("scientific/gate_") and value is not True:
                reasons.append(f"{key}={value}")
        relative = row.get("scientific/quality_gate_relative_to_baseline")
        if relative is not True:
            reasons.append(f"scientific/quality_gate_relative_to_baseline={relative}")
        row["scientific/ineligible_reasons"] = sorted(set(reasons))
        row["scientific/eligible_for_pareto"] = not reasons


def validate_stage_outputs(cfg: Config) -> None:  # noqa: C901
    if cfg.stage not in {"smoke", "screen", "long"}:
        msg = f"stage must be smoke, screen, or long, got {cfg.stage!r}"
        raise ValueError(msg)
    if cfg.output_summary.resolve() == cfg.output_table.resolve():
        msg = "aggregate summary and table paths must differ"
        raise ValueError(msg)
    stage_paths = (cfg.output_summary, cfg.output_table, cfg.live_plot_dir)
    missing_stage = [str(path) for path in stage_paths if cfg.stage not in path.name]
    if missing_stage:
        msg = (
            f"stage {cfg.stage!r} must appear in every aggregate/live output path; "
            f"unsafe paths={missing_stage}"
        )
        raise ValueError(msg)
    if str(mpl.get_backend()).lower() != "agg":
        msg = f"material inverse requires non-interactive Agg backend, got {mpl.get_backend()}"
        raise RuntimeError(msg)
    scientific_gates = {key: getattr(cfg, key) for key in EXPECTED_SCIENTIFIC_GATES}
    if not all(math.isfinite(float(value)) for value in scientific_gates.values()):
        msg = "inverse scientific-gate thresholds must all be finite"
        raise ValueError(msg)
    if scientific_gates != EXPECTED_SCIENTIFIC_GATES:
        msg = (
            "inverse requires the fixed scientific gates "
            f"{EXPECTED_SCIENTIFIC_GATES}, got {scientific_gates}"
        )
        raise ValueError(msg)
    inverse_protocol = {key: getattr(cfg, key) for key in EXPECTED_INVERSE_PROTOCOL}
    numeric_protocol = (
        value for value in inverse_protocol.values() if not isinstance(value, bool)
    )
    if not all(math.isfinite(float(value)) for value in numeric_protocol):
        msg = "inverse protocol parameters must all be finite"
        raise ValueError(msg)
    if inverse_protocol != EXPECTED_INVERSE_PROTOCOL:
        msg = (
            "inverse requires the fixed optimizer/solver protocol "
            f"{EXPECTED_INVERSE_PROTOCOL}, got {inverse_protocol}"
        )
        raise ValueError(msg)
    if cfg.stage == "smoke" and (
        cfg.candidate_set != "corners-with-no-skin"
        or cfg.inverse_max_steps != 0
        or cfg.mandatory_baseline_steps != 0
    ):
        msg = (
            "smoke stage requires corners-with-no-skin and "
            "max_steps=mandatory_baseline_steps=0"
        )
        raise ValueError(msg)
    if cfg.stage == "screen" and (
        cfg.candidate_set != "all-with-no-skin"
        or cfg.inverse_max_steps != 40
        or cfg.mandatory_baseline_steps != 40
    ):
        msg = "screen stage requires all-with-no-skin and a fixed 40-step budget"
        raise ValueError(msg)
    if cfg.stage == "long" and (
        cfg.inverse_max_steps != 200 or cfg.mandatory_baseline_steps != 200
    ):
        msg = "long stage requires a fixed 200-step budget"
        raise ValueError(msg)


def rewrite_individual_summaries(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        summary_path = row.get("artifact/summary_path")
        if summary_path is None:
            continue
        Path(str(summary_path)).write_text(
            json.dumps(row, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )


def run(cfg: Config) -> None:  # noqa: C901, PLR0915
    if cfg.initial_activation_mesh is not None or cfg.use_initial_displacement:
        msg = "material candidates must start from fresh zero activation/displacement"
        raise ValueError(msg)
    if cfg.mandatory_baseline_steps > cfg.inverse_max_steps:
        msg = "mandatory_baseline_steps cannot exceed inverse_max_steps"
        raise ValueError(msg)
    validate_stage_outputs(cfg)
    manifest = load_manifest(cfg)
    selected = select_candidates(manifest, cfg.candidate_set)
    manifest_by_label = {
        str(candidate["label"]): candidate for candidate in manifest["candidates"]
    }
    selected_material = [
        candidate for candidate in selected if str(candidate["label"]) != NO_SKIN_LABEL
    ]
    verification_labels = {
        BASELINE_LABEL,
        *(str(candidate["label"]) for candidate in selected_material),
    }
    verified: dict[str, tuple[Path, pv.PolyData, dict[str, Any]]] = {
        label: verified_candidate_skin(cfg, manifest_by_label[label])
        for label in verification_labels
    }
    quality_control_skin = verified[BASELINE_LABEL][1]
    configure_runtime()
    base_mesh = pv.read(cfg.input_mesh)
    if not isinstance(base_mesh, pv.UnstructuredGrid):
        base_mesh = base_mesh.cast_to_unstructured_grid()

    original_builder: Callable[..., Any] = reference_runtime.build_forward
    rows: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    for candidate in selected:
        label = str(candidate["label"])
        is_control = label == NO_SKIN_LABEL
        prestrain_enabled = (
            False if is_control else float(candidate["prestrain_gain"]) > 0.0
        )
        if is_control:
            setup = SETUP_NO_SKIN
            skin_path = None
            verified_skin = None
            provenance: dict[str, Any] = {}
            quality_skin = quality_control_skin
        else:
            setup = (
                SETUP_SKIN_ESTIMATED_PRESTRAIN
                if prestrain_enabled
                else SETUP_SKIN_NO_PRESTRAIN
            )
            skin_path, verified_skin, provenance = verified[label]
            quality_skin = verified_skin
        case = InverseCase(
            target="smile",
            lr=cfg.inverse_lr,
            setup=setup,
            label=f"material-{label}-{cfg.stage}",
        )
        paths = CasePaths.from_case(cfg.output_summary.parent, case)
        builder_calls = 0

        def independent_builder(
            mesh: pv.UnstructuredGrid,
            inverse_case: InverseCase,
            *,
            area_ratio_floor: float,
            _is_control: bool = is_control,
            _skin_path: Path | None = skin_path,
            _verified_skin: pv.PolyData | None = verified_skin,
            _candidate: dict[str, Any] = candidate,
            _provenance: dict[str, Any] = provenance,
            _label: str = label,
        ) -> tuple[Any, pv.PolyData | None, dict[str, Any]]:
            nonlocal builder_calls
            builder_calls += 1
            if builder_calls != 1:
                msg = f"{_label} requested more than one forward builder instance"
                raise RuntimeError(msg)
            if _is_control:
                built = original_builder(
                    mesh, inverse_case, area_ratio_floor=area_ratio_floor
                )
                if built[1] is not None or bool(built[2].get("skin/enabled", True)):
                    msg = "no-skin control unexpectedly constructed a skin"
                    raise RuntimeError(msg)
            else:
                if _skin_path is None or _verified_skin is None:
                    msg = f"{_label} lacks a verified candidate skin"
                    raise RuntimeError(msg)
                built = build_candidate_forward(
                    mesh,
                    inverse_case,
                    area_ratio_floor=area_ratio_floor,
                    skin_path=_skin_path,
                    verified_skin=_verified_skin,
                    candidate=_candidate,
                    provenance=_provenance,
                )
            return built

        reference_runtime.build_forward = independent_builder
        try:
            summary = normalize_absent_initial_displacement(
                solve_case(case, base_mesh.copy(deep=True), cfg)
            )
            errors, warnings, quality = validate_case(
                summary, paths.result, quality_skin, cfg
            )
            if builder_calls != 1:
                errors.append(f"forward builder was called {builder_calls} times")
            row = {
                **summary,
                **quality,
                "candidate": label,
                "candidate/young_min_scale": (
                    None if is_control else float(candidate["young_min_scale"])
                ),
                "candidate/prestrain_gain": (
                    None if is_control else float(candidate["prestrain_gain"])
                ),
                "candidate/skin_path": None if skin_path is None else str(skin_path),
                "provenance/manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "provenance/manifest_path": str(cfg.input_candidates),
                "provenance/input_mesh_sha256": str(
                    manifest["input_mesh_identity"]["sha256"]
                ),
                "provenance/input_mesh_size_bytes": int(
                    manifest["input_mesh_identity"]["size_bytes"]
                ),
                **provenance,
                "builder/fresh_independent": builder_calls == 1,
                "builder/calls": builder_calls,
                "scientific/is_control": is_control,
                "scientific/control_type": NO_SKIN_LABEL if is_control else None,
                "scientific/quality_surface_candidate": (
                    BASELINE_LABEL if is_control else label
                ),
                "artifact/summary_path": str(paths.summary),
                "stage": cfg.stage,
                "status": "ok" if not errors else "invalid",
                "validation/errors": errors,
                "validation/warnings": warnings,
            }
            paths.summary.write_text(
                json.dumps(row, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            rows.append(row)
            if errors:
                hard_failures.append(f"{label}: " + "; ".join(errors))
        except Exception as error:
            logger.exception("material candidate %s failed", label)
            rows.append(
                {
                    "candidate": label,
                    "stage": cfg.stage,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "artifact/summary_path": str(paths.summary),
                    "scientific/is_control": is_control,
                    "scientific/control_type": NO_SKIN_LABEL if is_control else None,
                }
            )
            hard_failures.append(f"{label}: {type(error).__name__}: {error}")
        finally:
            reference_runtime.build_forward = original_builder

    apply_relative_quality_gate(rows, cfg)
    rewrite_individual_summaries(rows)
    convergence_failures = [
        f"{row['candidate']}: {row['inverse/stop_reason']}"
        for row in rows
        if row.get("status") == "ok" and not bool(row["inverse/converged"])
    ]
    aggregate = {
        "schema_version": 2,
        "complete": not hard_failures and len(rows) == len(selected),
        "stage": cfg.stage,
        "input_mesh": str(cfg.input_mesh),
        "input_candidates": str(cfg.input_candidates),
        "candidate_set": cfg.candidate_set,
        "fresh_zero_activation": True,
        "activation_mode": "per-muscle-tet-6dof-unconstrained",
        "activation_shared": False,
        "activation_transferred_between_candidates": False,
        "forward_builder_shared_between_candidates": False,
        "plot_backend": str(mpl.get_backend()),
        "inverse_lr": cfg.inverse_lr,
        "inverse_max_steps": cfg.inverse_max_steps,
        "mandatory_baseline_steps": cfg.mandatory_baseline_steps,
        "hard_failures": hard_failures,
        "convergence_failures": convergence_failures,
        "baseline_available": any(
            row.get("status") == "ok" and row["candidate"] == BASELINE_LABEL
            for row in rows
        ),
        "pareto_candidates": [
            row["candidate"]
            for row in rows
            if bool(row.get("scientific/eligible_for_pareto", False))
        ],
        "cases": rows,
    }
    cfg.output_summary.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_table(cfg.output_table, rows)
    cherries.log_output(cfg.output_summary)
    cherries.log_output(cfg.output_table)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_table)
    if hard_failures:
        msg = "material inverse sweep failed validation: " + " | ".join(hard_failures)
        raise RuntimeError(msg)
    if convergence_failures and cfg.require_convergence:
        msg = "material inverse cases did not converge: " + ", ".join(
            convergence_failures
        )
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(run)
