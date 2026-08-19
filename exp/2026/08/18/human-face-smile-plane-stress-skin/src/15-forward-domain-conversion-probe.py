from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from _reference import (
    GROUP_DIR,
    KOITER_IMPLEMENTATION,
    KOITER_IMPLEMENTATION_SHA256,
    PREPARED_MESH,
    REPO_ROOT,
    SOURCE_SKIN,
    SOURCE_SKIN_SHA256,
    SOURCE_SKIN_SIZE_BYTES,
    VOLUME_FORWARD_IMPLEMENTATION,
    VOLUME_FORWARD_IMPLEMENTATION_SHA256,
    VOLUME_LAME_IMPLEMENTATION,
    VOLUME_LAME_IMPLEMENTATION_SHA256,
    enable_reference_modules,
    file_sha256,
    require_file_sha256,
)

from liblaf import cherries, melon

logger = logging.getLogger(__name__)

enable_reference_modules()

from _human_face_config import (  # noqa: E402
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
    SETUP_SKIN_NO_PRESTRAIN,
    SKIN_E,
    SKIN_NU,
    SKIN_THICKNESS,
    InverseCase,
    configure_runtime,
)
from _human_face_forward import set_volume_material  # noqa: E402
from _human_face_metrics import forward_solution_metrics, to_numpy  # noqa: E402
from _human_face_output import (  # noqa: E402
    bumpiness_metrics,
    make_result_mesh,
    surface_edges_for_mask,
)
from _human_face_targets import target_displacement_and_mask  # noqa: E402

from liblaf.apple.common import (  # noqa: E402
    ACTIVATION_INV,
    FIXED_MASK,
    FIXED_VALUE,
    FRACTION,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
    lame_converter_plane_stress,
)

DESIGN = "fixed-activation-domain-conversion-plus-cut-boundary-bracket"
SCHEMA_VERSION = 2
INTERPRETATION = (
    "causal forward probe with a transferred historical activation; it is not "
    "an inverse solution or recovered activation for any changed setup; the "
    "hard-fixed artificial-cut case is a sensitivity bracket, not ground truth"
)

MATERIAL_DATA_DIR = SOURCE_SKIN.parents[1]
HISTORICAL_STEM = "20-human-face-smile-skin-no-prestrain-lr3-material-e100-p000-screen"
HISTORICAL_RESULT = MATERIAL_DATA_DIR / f"{HISTORICAL_STEM}.vtu"
HISTORICAL_SUMMARY = MATERIAL_DATA_DIR / f"{HISTORICAL_STEM}-summary.json"
HISTORICAL_TARGET = MATERIAL_DATA_DIR / f"{HISTORICAL_STEM}-target.vtu"

PREPARED_MESH_SIZE_BYTES = 76_792_914
PREPARED_MESH_SHA256 = (
    "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563"
)
HISTORICAL_RESULT_SIZE_BYTES = 148_064_384
HISTORICAL_RESULT_SHA256 = (
    "0596f3dcf378f745d80533ac6bd7c0c3f289846e6320e761ef5e10d899e556d5"
)
HISTORICAL_SUMMARY_SIZE_BYTES = 123_434
HISTORICAL_SUMMARY_SHA256 = (
    "cba0574628ddef2f41fa79af14e9f84577e3d1fea9a1dec2ec6796822e621d65"
)
HISTORICAL_TARGET_SIZE_BYTES = 84_419_492
HISTORICAL_TARGET_SHA256 = (
    "58a2f997dec6e9b3d39e02ab122b9dfc5f0689815e4bbd613a786d21a41a4075"
)

TARGET_IMPLEMENTATION = VOLUME_FORWARD_IMPLEMENTATION.parent / "_human_face_targets.py"
TARGET_IMPLEMENTATION_SHA256 = (
    "34a1583fcb8f90f357647dd4574e2e7ef27f8049f2b3ba1e2fa7dc838fcbb696"
)
OUTPUT_IMPLEMENTATION = VOLUME_FORWARD_IMPLEMENTATION.parent / "_human_face_output.py"
OUTPUT_IMPLEMENTATION_SHA256 = (
    "29bae977a4b31e82276aca15fdaae3bdda37e6a3e71493876b6fd973db1a1c61"
)
CONFIG_IMPLEMENTATION = VOLUME_FORWARD_IMPLEMENTATION.parent / "_human_face_config.py"
CONFIG_IMPLEMENTATION_SHA256 = (
    "eca100cc6bdd4e2a1ac689c6e2e7e02cf80a9bea8fa9ac12e9590eca5f23ffb6"
)
CORE_MODULI_IMPLEMENTATION = REPO_ROOT / "src/liblaf/apple/common/_moduli.py"
CORE_MODULI_IMPLEMENTATION_SHA256 = (
    "9d5c14f27b9a08a8a4f9cd3ce4e3076f2375ed1108e84e94d307c9439e1a303d"
)

EXPECTED_FULL_TRIANGLES = 128_172
EXPECTED_FULL_AREA_M2 = 0.14204698861747428
EXPECTED_FULL_TOPOLOGY_SHA256 = (
    "5cc5e84531e2eb27fd62d8435b31959be4e1a9e60dcc519bcc4f3df506c430b1"
)
EXPECTED_FULL_UNASSIGNED_GROUP_POINTS = 6_000
EXPECTED_FULL_CUT_TRIANGLES = 13_165
EXPECTED_CUT_INCIDENT_VERTICES = 6_980
EXPECTED_CUT_PREEXISTING_FIXED_VERTICES = 380
EXPECTED_CUT_NEWLY_FIXED_VERTICES = 6_600
EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256 = (
    "8207cda8f9e11dbb4406f683e5ad818a6950e3515ac373719514094fb5b7fe5d"
)
EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256 = (
    "ca39cdc839855be34e75222964a1e5c129dd210e8800c684d7e6d1ce6424f138"
)
EXPECTED_ISFACE_TRIANGLES = 29_899
EXPECTED_ISFACE_AREA_M2 = 0.042879980597073028
EXPECTED_ISFACE_COMPONENTS = 1
EXPECTED_ISFACE_TOPOLOGY_SHA256 = (
    "1cbfa9a27bc26d4bd937d8fae0ab98bf8b07d977f923bcf25681155523cd82c7"
)
AREA_RTOL = 1.0e-12
FORMULA_RTOL = 1.0e-13
FORMULA_ATOL = 1.0e-14

Domain = Literal["full", "isface"]
Conversion = Literal["3d", "plane-stress"]
Seed = Literal["zero", "old"]
CutBoundary = Literal["current", "hard-fixed"]

CURRENT_CUT_BOUNDARY_POLICY = "historical-isfixed"
HARD_FIXED_CUT_BOUNDARY_POLICY = "all-artificial-cut-incident-vertices-hard-fixed"


@dataclass(frozen=True)
class ProbeSpec:
    label: str
    domain: Domain
    conversion: Conversion
    cut_boundary: CutBoundary = "current"


ALL_PROBES = (
    ProbeSpec("full-3d-replay", "full", "3d"),
    ProbeSpec("full-plane-stress", "full", "plane-stress"),
    ProbeSpec("isface-3d", "isface", "3d"),
    ProbeSpec("isface-plane-stress", "isface", "plane-stress"),
    ProbeSpec("isface-plane-stress-cut-fixed", "isface", "plane-stress", "hard-fixed"),
)
CHANGED_PROBES = ALL_PROBES[1:]
EXPECTED_PROBE_SET = ",".join(spec.label for spec in ALL_PROBES)
EXPECTED_SEED_SET = "zero,old"
EXPECTED_OUTPUT_SUMMARY_NAME = "15-forward-domain-conversion-probe-summary.json"
EXPECTED_OUTPUT_TABLE_NAME = "15-forward-domain-conversion-probe-table.md"
EXPECTED_OUTPUT_DIR_NAME = "15-forward-domain-conversion-probe"

EXPECTED_ISFACE_GROUP_NAMES = (
    "EyelidTop",
    "EyelidBottom",
    "EyelidOuterBottom",
    "EyelidOuterTop",
    "LipBottom",
    "LipTop",
    "LipOuterTop",
    "LipOuterBottom",
    "Chin",
    "Face",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_skin: Path = cherries.input(SOURCE_SKIN)
    input_historical_result: Path = cherries.input(HISTORICAL_RESULT)
    input_historical_summary: Path = cherries.input(HISTORICAL_SUMMARY)
    input_historical_target: Path = cherries.input(HISTORICAL_TARGET)
    output_summary: Path = cherries.output(
        "15-forward-domain-conversion-probe-summary.json", mkdir=True
    )
    output_table: Path = cherries.output(
        "15-forward-domain-conversion-probe-table.md", mkdir=True
    )
    output_dir_name: str = EXPECTED_OUTPUT_DIR_NAME

    probe_set: str = EXPECTED_PROBE_SET
    seed_set: str = EXPECTED_SEED_SET
    require_solver_success: bool = True
    branch_delta_fraction_of_target_tol: float = 1.0e-3


@dataclass(frozen=True)
class MetricBasis:
    target: np.ndarray
    loss_mask: np.ndarray
    target_rms: float
    bump_edges: np.ndarray
    metric_skin: pv.PolyData
    skin_mesh_ids: np.ndarray
    triangles: np.ndarray
    rest_area: np.ndarray
    target_area: np.ndarray
    target_vertex_normals: np.ndarray
    face_triangle_mask: np.ndarray
    face_vertex_ids: np.ndarray
    face_edges: np.ndarray
    contraction_tri_0: np.ndarray
    contraction_tri_1: np.ndarray
    contraction_target_dihedral: np.ndarray
    contraction_edge_weight: np.ndarray
    tets: np.ndarray
    rest_six_volume: np.ndarray
    rest_area_vectors: np.ndarray
    rest_area_vector_norm: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {token!r} in {path}")
        ),
    )
    if not isinstance(value, dict):
        msg = f"expected a JSON object in {path}"
        raise TypeError(msg)
    return value


def _require_identity(
    path: Path, *, expected_size: int, expected_sha256: str, name: str
) -> dict[str, int | str]:
    if not path.is_file():
        msg = f"missing pinned {name}: {path}"
        raise FileNotFoundError(msg)
    actual = {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
    expected = {"size_bytes": expected_size, "sha256": expected_sha256}
    if actual != expected:
        msg = f"{name} identity mismatch: expected {expected}, got {actual}"
        raise ValueError(msg)
    return actual


def _triangle_faces(surface: pv.PolyData) -> np.ndarray:
    encoded = np.asarray(surface.faces, dtype=np.int64)
    if encoded.size != 4 * surface.n_cells:
        msg = "skin connectivity is not packed triangles"
        raise ValueError(msg)
    encoded = encoded.reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        msg = "skin contains a non-triangle cell"
        raise ValueError(msg)
    return encoded[:, 1:].copy()


def _triangle_geometry(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float64).eps):
        msg = "skin contains a non-finite or degenerate triangle"
        raise ValueError(msg)
    return vectors, 0.5 * norms, vectors / norms[:, None]


def _canonical_topology_sha256(surface: pv.PolyData) -> str:
    triangles = _triangle_faces(surface)
    global_ids = np.asarray(surface.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    canonical = np.sort(global_ids[triangles], axis=1).astype("<i8", copy=False)
    order = np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0]))
    return hashlib.sha256(np.ascontiguousarray(canonical[order]).tobytes()).hexdigest()


def _unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _interior_edge_adjacency(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    triangle_ids = np.tile(np.arange(triangles.shape[0], dtype=np.int64), 3)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    triangle_ids = triangle_ids[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(np.diff(edges, axis=0), axis=1))]
    ends = np.r_[starts[1:], edges.shape[0]]
    interior = ends - starts == 2
    unique_edges = edges[starts[interior]]
    tri_0 = triangle_ids[starts[interior]]
    tri_1 = triangle_ids[starts[interior] + 1]
    length = np.linalg.norm(
        points[unique_edges[:, 1]] - points[unique_edges[:, 0]], axis=1
    )
    return unique_edges, tri_0, tri_1, length


def _triangle_component_count(triangles: np.ndarray) -> int:
    _, tri_0, tri_1, _ = _interior_edge_adjacency(
        np.zeros((int(triangles.max()) + 1, 3), dtype=np.float64), triangles
    )
    parent = np.arange(triangles.shape[0], dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for left, right in zip(tri_0, tri_1, strict=True):
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left != root_right:
            parent[root_right] = root_left
    return len({find(index) for index in range(triangles.shape[0])})


def _copy_selected_skin(source: pv.PolyData, mask: np.ndarray) -> pv.PolyData:
    source_triangles = _triangle_faces(source)
    selected_ids = np.flatnonzero(mask).astype(np.int64)
    selected = source_triangles[selected_ids]
    used_points, inverse = np.unique(selected.reshape(-1), return_inverse=True)
    local_triangles = inverse.reshape(-1, 3)
    faces = np.column_stack(
        (np.full(local_triangles.shape[0], 3, dtype=np.int64), local_triangles)
    )
    result = pv.PolyData(np.asarray(source.points)[used_points], faces)
    for name, values in source.point_data.items():
        result.point_data[name] = np.asarray(values)[used_points]
    for name, values in source.cell_data.items():
        result.cell_data[name] = np.asarray(values)[selected_ids]
    result.cell_data["ProbeSourceCellId"] = selected_ids
    return result


def _map_global_ids(mesh: pv.UnstructuredGrid, surface: pv.PolyData) -> np.ndarray:
    mesh_ids = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    requested = np.asarray(surface.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if np.unique(mesh_ids).size != mesh_ids.size:
        msg = "prepared mesh GlobalPointId values are not unique"
        raise ValueError(msg)
    order = np.argsort(mesh_ids)
    positions = np.searchsorted(mesh_ids[order], requested)
    if np.any(positions >= mesh_ids.size) or not np.array_equal(
        mesh_ids[order[positions]], requested
    ):
        msg = "skin GlobalPointId values do not map exactly to the prepared mesh"
        raise ValueError(msg)
    mapped = order[positions]
    if not np.array_equal(
        np.asarray(surface.points, dtype=np.float64),
        np.asarray(mesh.points, dtype=np.float64)[mapped],
    ):
        msg = "skin coordinates do not match the prepared mesh at GlobalPointId"
        raise ValueError(msg)
    return mapped


def _configure_cut_boundary(  # noqa: C901, PLR0912, PLR0915
    mesh: pv.UnstructuredGrid,
    source: pv.PolyData,
    policy: CutBoundary,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate the pinned artificial cut and apply one explicit BC policy."""
    triangles = _triangle_faces(source)
    source_group_ids = np.asarray(source.point_data["GroupId"], dtype=np.int64)
    if source_group_ids.shape != (source.n_points,):
        msg = "source skin GroupId field is malformed"
        raise ValueError(msg)
    unassigned = source_group_ids == -1
    if int(unassigned.sum()) != EXPECTED_FULL_UNASSIGNED_GROUP_POINTS:
        msg = (
            "artificial-cut marker point count changed: "
            f"{int(unassigned.sum())} != {EXPECTED_FULL_UNASSIGNED_GROUP_POINTS}"
        )
        raise ValueError(msg)
    cut_triangles = np.any(unassigned[triangles], axis=1)
    if int(cut_triangles.sum()) != EXPECTED_FULL_CUT_TRIANGLES:
        msg = (
            "artificial-cut triangle count changed: "
            f"{int(cut_triangles.sum())} != {EXPECTED_FULL_CUT_TRIANGLES}"
        )
        raise ValueError(msg)

    source_global_ids = np.asarray(
        source.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
    )
    canonical_cut = np.sort(source_global_ids[triangles[cut_triangles]], axis=1).astype(
        "<i8", copy=False
    )
    order = np.lexsort((canonical_cut[:, 2], canonical_cut[:, 1], canonical_cut[:, 0]))
    cut_topology_sha256 = hashlib.sha256(
        np.ascontiguousarray(canonical_cut[order]).tobytes()
    ).hexdigest()
    if cut_topology_sha256 != EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256:
        msg = f"artificial-cut topology digest changed: {cut_topology_sha256}"
        raise ValueError(msg)

    mapped = _map_global_ids(mesh, source)
    cut_source_point_ids = np.unique(triangles[cut_triangles])
    cut_mesh_ids = np.sort(mapped[cut_source_point_ids]).astype(np.int64, copy=False)
    if cut_mesh_ids.size != EXPECTED_CUT_INCIDENT_VERTICES:
        msg = (
            "artificial-cut incident vertex count changed: "
            f"{cut_mesh_ids.size} != {EXPECTED_CUT_INCIDENT_VERTICES}"
        )
        raise ValueError(msg)
    mesh_global_ids = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    cut_global_ids = np.sort(mesh_global_ids[cut_mesh_ids]).astype("<i8", copy=False)
    cut_global_ids_sha256 = hashlib.sha256(
        np.ascontiguousarray(cut_global_ids).tobytes()
    ).hexdigest()
    if cut_global_ids_sha256 != EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256:
        msg = f"artificial-cut incident GlobalPointId digest changed: {cut_global_ids_sha256}"
        raise ValueError(msg)

    required = {"IsFace", "IsFixed", FIXED_MASK.vtk, FIXED_VALUE.vtk}
    missing = sorted(required - set(mesh.point_data))
    if missing:
        msg = f"prepared mesh is missing cut-boundary fields: {missing}"
        raise KeyError(msg)
    is_face = np.asarray(mesh.point_data["IsFace"], dtype=bool)
    is_fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool).copy()
    fixed_mask = np.asarray(mesh.point_data[FIXED_MASK.vtk], dtype=bool).copy()
    fixed_value = np.asarray(mesh.point_data[FIXED_VALUE.vtk], dtype=np.float64).copy()
    if is_face.shape != (mesh.n_points,) or is_fixed.shape != (mesh.n_points,):
        msg = "prepared IsFace/IsFixed point fields are malformed"
        raise ValueError(msg)
    if fixed_mask.shape != (mesh.n_points, 3) or fixed_value.shape != (
        mesh.n_points,
        3,
    ):
        msg = "prepared FixedMask/FixedValue point fields are malformed"
        raise ValueError(msg)
    expected_fixed_mask = np.repeat(is_fixed[:, None], 3, axis=1)
    if not np.array_equal(fixed_mask, expected_fixed_mask):
        msg = "prepared FixedMask is inconsistent with IsFixed"
        raise ValueError(msg)
    if not np.array_equal(fixed_value, np.zeros_like(fixed_value)):
        msg = "formal probe requires exact-zero FixedValue"
        raise ValueError(msg)
    if np.any(is_face[cut_mesh_ids]):
        msg = "artificial-cut incident vertices overlap IsFace"
        raise ValueError(msg)

    preexisting = is_fixed[cut_mesh_ids]
    if int(preexisting.sum()) != EXPECTED_CUT_PREEXISTING_FIXED_VERTICES:
        msg = (
            "artificial-cut preexisting fixed overlap changed: "
            f"{int(preexisting.sum())} != "
            f"{EXPECTED_CUT_PREEXISTING_FIXED_VERTICES}"
        )
        raise ValueError(msg)
    added_ids = cut_mesh_ids[~preexisting]
    if added_ids.size != EXPECTED_CUT_NEWLY_FIXED_VERTICES:
        msg = (
            "artificial-cut newly fixed candidate count changed: "
            f"{added_ids.size} != {EXPECTED_CUT_NEWLY_FIXED_VERTICES}"
        )
        raise ValueError(msg)

    hard_fixed = policy == "hard-fixed"
    if hard_fixed:
        is_fixed[cut_mesh_ids] = True
        fixed_mask[cut_mesh_ids] = True
        fixed_value[cut_mesh_ids] = 0.0
    elif policy != "current":
        msg = f"unsupported cut-boundary policy: {policy}"
        raise ValueError(msg)

    if not np.array_equal(fixed_mask, np.repeat(is_fixed[:, None], 3, axis=1)):
        msg = f"{policy} FixedMask is inconsistent with IsFixed"
        raise ValueError(msg)
    if not np.array_equal(fixed_value[is_fixed], np.zeros_like(fixed_value[is_fixed])):
        msg = f"{policy} fixed values are not exact zero"
        raise ValueError(msg)

    historical_is_fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    incident = np.zeros(mesh.n_points, dtype=np.int8)
    incident[cut_mesh_ids] = 1
    preexisting_field = np.zeros(mesh.n_points, dtype=np.int8)
    preexisting_field[cut_mesh_ids[preexisting]] = 1
    added_field = np.zeros(mesh.n_points, dtype=np.int8)
    if hard_fixed:
        added_field[added_ids] = 1
    mesh.point_data["HistoricalIsFixed"] = historical_is_fixed.astype(np.int8)
    mesh.point_data["ArtificialCutIncident"] = incident
    mesh.point_data["CutBoundaryPreexistingFixed"] = preexisting_field
    mesh.point_data["CutBoundaryAddedFixed"] = added_field
    mesh.point_data["IsFixed"] = is_fixed
    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value
    persisted_is_fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    persisted_fixed_mask = np.asarray(mesh.point_data[FIXED_MASK.vtk], dtype=bool)
    persisted_fixed_value = np.asarray(
        mesh.point_data[FIXED_VALUE.vtk], dtype=np.float64
    )
    if not np.array_equal(persisted_is_fixed, is_fixed):
        msg = f"{policy} IsFixed changed while installing the boundary policy"
        raise ValueError(msg)
    if not np.array_equal(
        persisted_fixed_mask,
        np.repeat(persisted_is_fixed[:, None], 3, axis=1),
    ):
        msg = f"{policy} persisted FixedMask is inconsistent with IsFixed"
        raise ValueError(msg)
    if not np.array_equal(
        persisted_fixed_value[persisted_is_fixed],
        np.zeros_like(persisted_fixed_value[persisted_is_fixed]),
    ):
        msg = f"{policy} persisted fixed values are not exact zero"
        raise ValueError(msg)

    total_fixed = int(persisted_is_fixed[cut_mesh_ids].sum())
    expected_total_fixed = (
        EXPECTED_CUT_INCIDENT_VERTICES
        if hard_fixed
        else EXPECTED_CUT_PREEXISTING_FIXED_VERTICES
    )
    if total_fixed != expected_total_fixed:
        msg = f"{policy} artificial-cut total fixed count changed"
        raise ValueError(msg)
    return cut_mesh_ids, {
        "cut_boundary/policy": (
            HARD_FIXED_CUT_BOUNDARY_POLICY
            if hard_fixed
            else CURRENT_CUT_BOUNDARY_POLICY
        ),
        "cut_boundary/marker": "source skin triangle touches mapped GroupId=-1 vertex",
        "cut_boundary/triangles": int(cut_triangles.sum()),
        "cut_boundary/triangle_topology_sha256": cut_topology_sha256,
        "cut_boundary/incident_vertices": int(cut_mesh_ids.size),
        "cut_boundary/incident_global_ids_sha256": cut_global_ids_sha256,
        "cut_boundary/preexisting_fixed_vertices": int(preexisting.sum()),
        "cut_boundary/newly_fixed_vertices": int(added_ids.size) if hard_fixed else 0,
        "cut_boundary/total_fixed_vertices": total_fixed,
        "cut_boundary/hard_fixed_is_ground_truth": False,
    }


def _validate_domain(  # noqa: C901, PLR0912, PLR0915
    mesh: pv.UnstructuredGrid, skin: pv.PolyData, domain: Domain
) -> dict[str, Any]:
    triangles = _triangle_faces(skin)
    _, area, _ = _triangle_geometry(
        np.asarray(skin.points, dtype=np.float64), triangles
    )
    topology = _canonical_topology_sha256(skin)
    mapped = _map_global_ids(mesh, skin)
    is_face_points = np.asarray(mesh.point_data["IsFace"], dtype=bool)[mapped]
    is_fixed_points = np.asarray(mesh.point_data["IsFixed"], dtype=bool)[mapped]
    is_teeth_points = np.asarray(mesh.point_data["IsTeeth"], dtype=bool)[mapped]
    is_gingiva_points = np.asarray(mesh.point_data["IsGingiva"], dtype=bool)[mapped]
    group_ids = np.asarray(mesh.point_data["GroupId"], dtype=np.int64)[mapped]
    group_names = tuple(str(name) for name in mesh.field_data["GroupName"])
    if np.any(group_ids < -1) or np.any(group_ids >= len(group_names)):
        msg = f"{domain} domain contains an invalid GroupId"
        raise ValueError(msg)
    valid_group_ids = group_ids[group_ids >= 0]
    selected_group_names = tuple(
        sorted({group_names[index] for index in valid_group_ids})
    )
    unassigned_group_points = group_ids == -1
    any_unassigned_group = np.any(unassigned_group_points[triangles], axis=1)
    all_face = np.all(is_face_points[triangles], axis=1)
    any_fixed = np.any(is_fixed_points[triangles], axis=1)
    any_teeth = np.any(is_teeth_points[triangles], axis=1)
    any_gingiva = np.any(is_gingiva_points[triangles], axis=1)
    if domain == "full":
        expected_count = EXPECTED_FULL_TRIANGLES
        expected_area = EXPECTED_FULL_AREA_M2
        expected_topology = EXPECTED_FULL_TOPOLOGY_SHA256
        if int(unassigned_group_points.sum()) != EXPECTED_FULL_UNASSIGNED_GROUP_POINTS:
            msg = (
                "full domain unassigned cut-point count changed: "
                f"{int(unassigned_group_points.sum())} != "
                f"{EXPECTED_FULL_UNASSIGNED_GROUP_POINTS}"
            )
            raise ValueError(msg)
        if int(any_unassigned_group.sum()) != EXPECTED_FULL_CUT_TRIANGLES:
            msg = (
                "full domain artificial-cut triangle count changed: "
                f"{int(any_unassigned_group.sum())} != {EXPECTED_FULL_CUT_TRIANGLES}"
            )
            raise ValueError(msg)
    else:
        expected_count = EXPECTED_ISFACE_TRIANGLES
        expected_area = EXPECTED_ISFACE_AREA_M2
        expected_topology = EXPECTED_ISFACE_TOPOLOGY_SHA256
        if not np.all(all_face):
            msg = "IsFace domain contains a triangle with a non-IsFace vertex"
            raise ValueError(msg)
        if np.any(unassigned_group_points):
            msg = "IsFace domain contains an unassigned GroupId=-1 point"
            raise ValueError(msg)
        if selected_group_names != tuple(sorted(EXPECTED_ISFACE_GROUP_NAMES)):
            msg = (
                "IsFace domain anatomical groups changed: "
                f"{selected_group_names} != {tuple(sorted(EXPECTED_ISFACE_GROUP_NAMES))}"
            )
            raise ValueError(msg)
        if np.any(any_fixed | any_gingiva):
            msg = "IsFace domain overlaps fixed or gingiva vertices"
            raise ValueError(msg)
    actual_area = float(area.sum())
    if skin.n_cells != expected_count:
        msg = f"{domain} domain cell count changed: {skin.n_cells} != {expected_count}"
        raise ValueError(msg)
    if not math.isclose(actual_area, expected_area, rel_tol=AREA_RTOL, abs_tol=1.0e-15):
        msg = f"{domain} domain area changed: {actual_area} != {expected_area}"
        raise ValueError(msg)
    if topology != expected_topology:
        msg = f"{domain} domain topology digest changed: {topology}"
        raise ValueError(msg)
    components = _triangle_component_count(triangles)
    if domain == "isface" and components != EXPECTED_ISFACE_COMPONENTS:
        msg = f"IsFace domain component count changed: {components}"
        raise ValueError(msg)
    return {
        "domain/name": domain,
        "domain/n_points": int(skin.n_points),
        "domain/n_triangles": int(skin.n_cells),
        "domain/rest_area_m2": actual_area,
        "domain/topology_sha256": topology,
        "domain/components": components,
        "domain/all_vertex_isface_triangles": int(all_face.sum()),
        "domain/triangles_touching_fixed": int(any_fixed.sum()),
        "domain/anatomical_group_names": list(selected_group_names),
        "domain/unassigned_group_points": int(unassigned_group_points.sum()),
        "domain/triangles_touching_unassigned_group": int(any_unassigned_group.sum()),
        "domain/unassigned_group_policy": (
            "allowed only for the pinned full boundary where GroupId=-1 marks "
            "the artificial InFaceConvex cut; forbidden in the IsFace ROI"
        ),
        "domain/triangles_touching_teeth_proximity_mask": int(any_teeth.sum()),
        "domain/teeth_proximity_policy": (
            "diagnostic only: IsTeeth is a 2 mm proximity mask and includes valid "
            "LipTop/LipBottom triangles; exact GroupName membership is the gate"
        ),
        "domain/triangles_touching_gingiva": int(any_gingiva.sum()),
    }


def _lame_parameters(conversion: Conversion) -> tuple[float, float]:
    if conversion == "3d":
        lambda_ = SKIN_E * SKIN_NU / ((1.0 + SKIN_NU) * (1.0 - 2.0 * SKIN_NU))
        mu = SKIN_E / (2.0 * (1.0 + SKIN_NU))
        return float(lambda_), float(mu)
    young = torch.tensor(SKIN_E, dtype=torch.float64, device="cpu")
    poisson = torch.tensor(SKIN_NU, dtype=torch.float64, device="cpu")
    lambda_, mu = lame_converter_plane_stress(young, poisson)
    return float(lambda_.item()), float(mu.item())


def _make_probe_skin(
    mesh: pv.UnstructuredGrid,
    source: pv.PolyData,
    *,
    domain: Domain,
    conversion: Conversion,
) -> tuple[pv.PolyData, dict[str, Any]]:
    source_face_mask = np.asarray(source.cell_data["IsFaceTriangle"], dtype=bool)
    skin = (
        source.copy(deep=True)
        if domain == "full"
        else _copy_selected_skin(source, source_face_mask)
    )
    lambda_, mu = _lame_parameters(conversion)
    skin.cell_data[LAMBDA.vtk] = np.full(skin.n_cells, lambda_, dtype=np.float64)
    skin.cell_data[MU.vtk] = np.full(skin.n_cells, mu, dtype=np.float64)
    skin.cell_data[FRACTION.vtk] = np.ones(skin.n_cells, dtype=np.float64)
    skin.cell_data[ACTIVATION_INV.vtk] = np.zeros((skin.n_cells, 3), dtype=np.float64)
    metrics = {
        **_validate_domain(mesh, skin, domain),
        "skin/conversion": conversion,
        "skin/E_MPa": float(SKIN_E),
        "skin/nu_3d_input": float(SKIN_NU),
        "skin/lambda_MPa": lambda_,
        "skin/mu_MPa": mu,
        "skin/thickness_m": float(SKIN_THICKNESS),
        "skin/prestrain": "exact-zero",
    }
    for name, actual, expected in (
        ("Lambda", np.asarray(skin.cell_data[LAMBDA.vtk]), lambda_),
        ("Mu", np.asarray(skin.cell_data[MU.vtk]), mu),
        ("Fraction", np.asarray(skin.cell_data[FRACTION.vtk]), 1.0),
    ):
        if not np.allclose(actual, expected, rtol=FORMULA_RTOL, atol=FORMULA_ATOL):
            msg = f"{domain}+{conversion} live skin {name} field changed"
            raise ValueError(msg)
    if not np.allclose(
        np.asarray(skin.cell_data[ACTIVATION_INV.vtk]),
        0.0,
        rtol=0.0,
        atol=0.0,
    ):
        msg = f"{domain}+{conversion} skin prestrain is not exact zero"
        raise ValueError(msg)
    return skin, metrics


def _volume_lambda_mu(E: float, nu: float) -> tuple[float, float]:
    return (
        E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        E / (2.0 * (1.0 + nu)),
    )


def _build_forward(
    mesh: pv.UnstructuredGrid, skin: pv.PolyData
) -> tuple[Any, dict[str, dict[str, torch.Tensor]]]:
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import (
        Koiter,
        StableNeoHookean,
        StableNeoHookeanActive,
    )

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    for name, E, nu, fraction_name, potential_type in (
        (
            "aponeurosis",
            APONEUROSIS_E,
            APONEUROSIS_NU,
            APONEUROSIS_FRACTION,
            StableNeoHookean,
        ),
        ("fat", FAT_E, FAT_NU, FAT_FRACTION, StableNeoHookean),
        ("muscle", MUSCLE_E, MUSCLE_NU, MUSCLE_FRACTION, StableNeoHookeanActive),
    ):
        fraction = np.asarray(mesh.cell_data[fraction_name], dtype=np.float64)
        set_volume_material(mesh, E=E, nu=nu, fraction=fraction)
        builder.add_potential(potential_type.from_pyvista(mesh, name=name))
    builder.add_potential(
        Koiter.from_pyvista(skin, name="skin", thickness=SKIN_THICKNESS)
    )
    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=FORWARD_MAX_STEPS,
        atol=FORWARD_ATOL,
        rtol=FORWARD_RTOL,
    )
    materials = forward.model.get_materials()
    return forward, materials


def _validate_live_materials(
    mesh: pv.UnstructuredGrid,
    skin: pv.PolyData,
    materials: dict[str, dict[str, torch.Tensor]],
    *,
    conversion: Conversion,
) -> dict[str, Any]:
    expected_potentials = {"aponeurosis", "fat", "muscle", "skin"}
    if set(materials) != expected_potentials:
        msg = f"forward potentials changed: {sorted(materials)}"
        raise ValueError(msg)
    volume_metrics: dict[str, Any] = {}
    cell_volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    for name, E, nu, fraction_name in (
        ("aponeurosis", APONEUROSIS_E, APONEUROSIS_NU, APONEUROSIS_FRACTION),
        ("fat", FAT_E, FAT_NU, FAT_FRACTION),
        ("muscle", MUSCLE_E, MUSCLE_NU, MUSCLE_FRACTION),
    ):
        lambda_, mu = _volume_lambda_mu(E, nu)
        live_lambda = to_numpy(materials[name][LAMBDA.value])
        live_mu = to_numpy(materials[name][MU.value])
        live_dv = to_numpy(materials[name]["dV"])
        fraction = np.asarray(mesh.cell_data[fraction_name], dtype=np.float64)
        integrated_dv = (
            np.asarray(live_dv, dtype=np.float64).reshape(mesh.n_cells, -1).sum(axis=1)
        )
        if not np.allclose(live_lambda, lambda_, rtol=FORMULA_RTOL, atol=FORMULA_ATOL):
            msg = f"live {name} Lambda is not the pinned 3D conversion"
            raise ValueError(msg)
        if not np.allclose(live_mu, mu, rtol=FORMULA_RTOL, atol=FORMULA_ATOL):
            msg = f"live {name} Mu is not the pinned 3D conversion"
            raise ValueError(msg)
        if not np.allclose(
            integrated_dv,
            cell_volume * fraction,
            rtol=1.0e-10,
            atol=1.0e-18,
        ):
            msg = f"live {name} integration weights do not encode its volume fraction"
            raise ValueError(msg)
        volume_metrics.update(
            {
                f"volume/{name}/E_MPa": float(E),
                f"volume/{name}/nu": float(nu),
                f"volume/{name}/lambda_MPa": float(lambda_),
                f"volume/{name}/mu_MPa": float(mu),
                f"volume/{name}/fraction_sum": float(fraction.sum()),
                f"volume/{name}/weighted_volume_m3": float(integrated_dv.sum()),
                f"volume/{name}/conversion": "3d",
            }
        )
    skin_lambda, skin_mu = _lame_parameters(conversion)
    live_skin_lambda = to_numpy(materials["skin"][LAMBDA.value])
    live_skin_mu = to_numpy(materials["skin"][MU.value])
    live_fraction = to_numpy(materials["skin"][FRACTION.value])
    live_prestrain = to_numpy(materials["skin"][ACTIVATION_INV.value])
    if live_skin_lambda.shape != (skin.n_cells,) or not np.allclose(
        live_skin_lambda, skin_lambda, rtol=FORMULA_RTOL, atol=FORMULA_ATOL
    ):
        msg = "live Koiter Lambda or cell count changed"
        raise ValueError(msg)
    if live_skin_mu.shape != (skin.n_cells,) or not np.allclose(
        live_skin_mu, skin_mu, rtol=FORMULA_RTOL, atol=FORMULA_ATOL
    ):
        msg = "live Koiter Mu or cell count changed"
        raise ValueError(msg)
    if not np.allclose(live_fraction, 1.0, rtol=0.0, atol=0.0):
        msg = "live Koiter Fraction is not exact one"
        raise ValueError(msg)
    if not np.allclose(live_prestrain, 0.0, rtol=0.0, atol=0.0):
        msg = "live Koiter prestrain is not exact zero"
        raise ValueError(msg)
    return {
        **volume_metrics,
        "koiter/live_n_triangles": int(live_skin_lambda.shape[0]),
        "koiter/live_lambda_MPa": float(live_skin_lambda[0]),
        "koiter/live_mu_MPa": float(live_skin_mu[0]),
        "koiter/live_fraction_min": float(live_fraction.min()),
        "koiter/live_fraction_max": float(live_fraction.max()),
        "koiter/live_prestrain_max_abs": float(np.abs(live_prestrain).max()),
    }


def _encoded_tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    encoded = np.asarray(mesh.cells, dtype=np.int64)
    if encoded.size != 5 * mesh.n_cells:
        msg = "prepared mesh is not pure tetrahedral"
        raise ValueError(msg)
    encoded = encoded.reshape(-1, 5)
    if not np.all(encoded[:, 0] == 4):
        msg = "prepared mesh contains a non-tetrahedral cell"
        raise ValueError(msg)
    return encoded[:, 1:].copy()


def _six_volume(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ij,ij->i",
        points[tets[:, 1]] - points[tets[:, 0]],
        np.cross(
            points[tets[:, 2]] - points[tets[:, 0]],
            points[tets[:, 3]] - points[tets[:, 0]],
        ),
    )


def _vertex_normals(
    points: np.ndarray,
    triangles: np.ndarray,
    area_vectors: np.ndarray,
    triangle_mask: np.ndarray,
) -> np.ndarray:
    normals = np.zeros_like(points)
    for local in range(3):
        np.add.at(
            normals,
            triangles[triangle_mask, local],
            area_vectors[triangle_mask],
        )
    norm = np.linalg.norm(normals, axis=1)
    used = np.unique(triangles[triangle_mask])
    if np.any(norm[used] <= np.finfo(np.float64).eps):
        msg = "metric IsFace surface contains a vertex with undefined normal"
        raise ValueError(msg)
    normals[used] /= norm[used, None]
    return normals


def _build_metric_basis(
    mesh: pv.UnstructuredGrid,
    metric_skin: pv.PolyData,
    target: np.ndarray,
    loss_mask: np.ndarray,
) -> MetricBasis:
    skin_mesh_ids = _map_global_ids(mesh, metric_skin)
    triangles = _triangle_faces(metric_skin)
    rest_points = np.asarray(metric_skin.points, dtype=np.float64)
    rest_vectors, rest_area, _ = _triangle_geometry(rest_points, triangles)
    target_points = rest_points + target[skin_mesh_ids]
    target_vectors, target_area, target_normals = _triangle_geometry(
        target_points, triangles
    )
    face_triangle_mask = np.asarray(metric_skin.cell_data["IsFaceTriangle"], dtype=bool)
    target_vertex_normals = _vertex_normals(
        target_points, triangles, target_vectors, face_triangle_mask
    )
    face_vertex_ids = np.unique(triangles[face_triangle_mask])
    face_edges = _unique_edges(triangles[face_triangle_mask])
    _, tri_0, tri_1, edge_length = _interior_edge_adjacency(rest_points, triangles)
    contraction = np.asarray(
        metric_skin.cell_data["ContractionPrestrainMask"], dtype=bool
    )
    contraction_edges = contraction[tri_0] & contraction[tri_1]
    if not np.any(contraction_edges):
        msg = "target-defined contraction ROI contains no interior edge"
        raise ValueError(msg)
    contraction_tri_0 = tri_0[contraction_edges]
    contraction_tri_1 = tri_1[contraction_edges]
    contraction_target_dihedral = np.arccos(
        np.clip(
            np.einsum(
                "ij,ij->i",
                target_normals[contraction_tri_0],
                target_normals[contraction_tri_1],
            ),
            -1.0,
            1.0,
        )
    )
    target_rms = float(
        np.linalg.norm(target[loss_mask]) / math.sqrt(int(loss_mask.sum()))
    )
    if not math.isfinite(target_rms) or target_rms <= 0.0:
        msg = f"invalid target RMS: {target_rms}"
        raise ValueError(msg)
    tets = _encoded_tetrahedra(mesh)
    rest_six_volume = _six_volume(np.asarray(mesh.points), tets)
    if np.any(np.abs(rest_six_volume) <= np.finfo(np.float64).eps):
        msg = "prepared mesh contains a zero-volume tetrahedron"
        raise ValueError(msg)
    return MetricBasis(
        target=target,
        loss_mask=loss_mask,
        target_rms=target_rms,
        bump_edges=surface_edges_for_mask(mesh, loss_mask),
        metric_skin=metric_skin,
        skin_mesh_ids=skin_mesh_ids,
        triangles=triangles,
        rest_area=rest_area,
        target_area=target_area,
        target_vertex_normals=target_vertex_normals,
        face_triangle_mask=face_triangle_mask,
        face_vertex_ids=face_vertex_ids,
        face_edges=face_edges,
        contraction_tri_0=contraction_tri_0,
        contraction_tri_1=contraction_tri_1,
        contraction_target_dihedral=contraction_target_dihedral,
        contraction_edge_weight=edge_length[contraction_edges],
        tets=tets,
        rest_six_volume=rest_six_volume,
        rest_area_vectors=rest_vectors,
        rest_area_vector_norm=np.linalg.norm(rest_vectors, axis=1),
    )


def _area_weighted_point_rms(
    point_vectors: np.ndarray,
    triangles: np.ndarray,
    triangle_mask: np.ndarray,
    weights: np.ndarray,
) -> float:
    squared = np.sum(np.square(point_vectors), axis=1)
    triangle_squared = np.mean(squared[triangles], axis=1)
    active = triangle_mask & (weights > 0.0)
    return float(
        np.sqrt(
            np.dot(weights[active], triangle_squared[active]) / weights[active].sum()
        )
    )


def _scalar_graph_laplacian(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    neighbor_sum = np.zeros_like(values)
    neighbor_count = np.zeros(values.shape[0], dtype=np.int64)
    np.add.at(neighbor_sum, edges[:, 0], values[edges[:, 1]])
    np.add.at(neighbor_sum, edges[:, 1], values[edges[:, 0]])
    np.add.at(neighbor_count, edges[:, 0], 1)
    np.add.at(neighbor_count, edges[:, 1], 1)
    active = neighbor_count > 0
    result = np.zeros_like(values)
    result[active] = values[active] - neighbor_sum[active] / neighbor_count[active]
    return result


def _fixed_activation_metrics(
    mesh: pv.UnstructuredGrid, basis: MetricBasis, displacement: np.ndarray
) -> dict[str, Any]:
    if displacement.shape != basis.target.shape or not np.isfinite(displacement).all():
        msg = "forward displacement is malformed or non-finite"
        raise ValueError(msg)
    residual = displacement - basis.target
    error_rms = float(
        np.linalg.norm(residual[basis.loss_mask])
        / math.sqrt(int(basis.loss_mask.sum()))
    )
    skin_displacement = displacement[basis.skin_mesh_ids]
    skin_residual = residual[basis.skin_mesh_ids]
    deformed = np.asarray(basis.metric_skin.points) + skin_displacement
    deformed_vectors, _, deformed_normals = _triangle_geometry(
        deformed, basis.triangles
    )
    deformed_dihedral = np.arccos(
        np.clip(
            np.einsum(
                "ij,ij->i",
                deformed_normals[basis.contraction_tri_0],
                deformed_normals[basis.contraction_tri_1],
            ),
            -1.0,
            1.0,
        )
    )
    dihedral_delta = deformed_dihedral - basis.contraction_target_dihedral
    dihedral_rms = float(
        np.sqrt(
            np.dot(basis.contraction_edge_weight, np.square(dihedral_delta))
            / basis.contraction_edge_weight.sum()
        )
    )
    residual_normal = np.einsum("ij,ij->i", skin_residual, basis.target_vertex_normals)
    residual_normal_laplacian = _scalar_graph_laplacian(
        residual_normal, basis.face_edges
    )
    deformed_volume = _six_volume(np.asarray(mesh.points) + displacement, basis.tets)
    det_f = deformed_volume / basis.rest_six_volume
    signed_normal_ratio = np.einsum(
        "ij,ij->i", deformed_vectors, basis.rest_area_vectors
    ) / np.square(basis.rest_area_vector_norm)
    if not np.isfinite(det_f).all() or not np.isfinite(signed_normal_ratio).all():
        msg = "deformation diagnostics contain non-finite values"
        raise ValueError(msg)
    legacy = bumpiness_metrics(
        mask=basis.loss_mask,
        edges=basis.bump_edges,
        displacement=displacement,
        target=basis.target,
    )
    return {
        "target/error_rms_m": error_rms,
        "target/error_rms_mm": 1.0e3 * error_rms,
        "target/error_rms_fraction_of_target": error_rms / basis.target_rms,
        "target/face_rest_area_weighted_error_rms_m": _area_weighted_point_rms(
            skin_residual,
            basis.triangles,
            basis.face_triangle_mask,
            basis.rest_area,
        ),
        "target/face_target_area_weighted_error_rms_m": _area_weighted_point_rms(
            skin_residual,
            basis.triangles,
            basis.face_triangle_mask,
            basis.target_area,
        ),
        "bumpiness/contraction_target_relative_dihedral_rms_rad": dihedral_rms,
        "bumpiness/contraction_target_relative_dihedral_rms_deg": math.degrees(
            dihedral_rms
        ),
        "bumpiness/residual_normal_laplacian_rms_m": float(
            np.linalg.norm(residual_normal_laplacian[basis.face_vertex_ids])
            / math.sqrt(basis.face_vertex_ids.size)
        ),
        **legacy,
        "warning/inverted_tets": int(np.sum(det_f <= 0.0)),
        "warning/inverted_tet_fraction": float(np.mean(det_f <= 0.0)),
        "warning/detF_min": float(det_f.min()),
        "warning/isface_folded_triangles": int(
            np.sum(basis.face_triangle_mask & (signed_normal_ratio <= 0.0))
        ),
        "warning/isface_folded_triangle_fraction": float(
            np.mean(signed_normal_ratio[basis.face_triangle_mask] <= 0.0)
        ),
        "warning/policy": "visual-review-only; small inversions/folds are not a veto",
    }


def _validate_historical_inputs(  # noqa: C901, PLR0912, PLR0915
    cfg: Config,
    mesh: pv.UnstructuredGrid,
    source_skin: pv.PolyData,
) -> tuple[
    pv.UnstructuredGrid,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
]:
    identities = {
        "mesh": _require_identity(
            cfg.input_mesh,
            expected_size=PREPARED_MESH_SIZE_BYTES,
            expected_sha256=PREPARED_MESH_SHA256,
            name="prepared mesh",
        ),
        "skin": _require_identity(
            cfg.input_skin,
            expected_size=SOURCE_SKIN_SIZE_BYTES,
            expected_sha256=SOURCE_SKIN_SHA256,
            name="historical homogeneous skin",
        ),
        "historical_result": _require_identity(
            cfg.input_historical_result,
            expected_size=HISTORICAL_RESULT_SIZE_BYTES,
            expected_sha256=HISTORICAL_RESULT_SHA256,
            name="historical e100-p000 result",
        ),
        "historical_summary": _require_identity(
            cfg.input_historical_summary,
            expected_size=HISTORICAL_SUMMARY_SIZE_BYTES,
            expected_sha256=HISTORICAL_SUMMARY_SHA256,
            name="historical e100-p000 summary",
        ),
        "historical_target": _require_identity(
            cfg.input_historical_target,
            expected_size=HISTORICAL_TARGET_SIZE_BYTES,
            expected_sha256=HISTORICAL_TARGET_SHA256,
            name="historical e100-p000 target",
        ),
    }
    historical = pv.read(cfg.input_historical_result)
    historical_target = pv.read(cfg.input_historical_target)
    if not isinstance(historical, pv.UnstructuredGrid) or not isinstance(
        historical_target, pv.UnstructuredGrid
    ):
        msg = "historical result or target is not an UnstructuredGrid"
        raise TypeError(msg)
    for name, candidate in (
        ("historical result", historical),
        ("historical target", historical_target),
    ):
        if candidate.n_points != mesh.n_points or candidate.n_cells != mesh.n_cells:
            msg = f"{name} dimensions differ from the prepared mesh"
            raise ValueError(msg)
        if not np.array_equal(candidate.points, mesh.points):
            msg = f"{name} rest points differ from the prepared mesh"
            raise ValueError(msg)
        if not np.array_equal(candidate.cells, mesh.cells) or not np.array_equal(
            candidate.celltypes, mesh.celltypes
        ):
            msg = f"{name} topology differs from the prepared mesh"
            raise ValueError(msg)
    local_global_ids = np.arange(mesh.n_points, dtype=np.int64)
    if GLOBAL_POINT_ID.vtk in mesh.point_data and not np.array_equal(
        np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64),
        local_global_ids,
    ):
        msg = "prepared mesh contains a non-canonical GlobalPointId field"
        raise ValueError(msg)
    if GLOBAL_POINT_ID.vtk not in historical.point_data or not np.array_equal(
        np.asarray(historical.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64),
        local_global_ids,
    ):
        msg = "historical result does not bind GlobalPointId to prepared local indices"
        raise ValueError(msg)
    # ModelBuilder.add_vertices uses this same local-index fallback. Install it
    # explicitly only after the prepared artifact's bytes/topology are pinned,
    # so all skin-domain checks and every forward use an identical mapping.
    mesh.point_data[GLOBAL_POINT_ID.vtk] = local_global_ids
    case = InverseCase(
        target="smile", lr=0.3, setup=SETUP_SKIN_NO_PRESTRAIN, label="probe"
    )
    target, loss_mask, target_metrics = target_displacement_and_mask(mesh, case, None)
    stored_target = np.asarray(
        historical_target.point_data["TargetDisplacement"], dtype=np.float64
    )
    stored_mask = np.asarray(historical_target.point_data["LossMask"], dtype=bool)
    if not np.array_equal(stored_target, target) or not np.array_equal(
        stored_mask, loss_mask
    ):
        msg = "historical target artifact differs from the live pinned target rule"
        raise ValueError(msg)
    for name in ("TargetDisplacement", "LossMask"):
        if not np.array_equal(
            np.asarray(historical.point_data[name]),
            np.asarray(historical_target.point_data[name]),
        ):
            msg = f"historical result {name} differs from the target artifact"
            raise ValueError(msg)
    activation = np.asarray(
        historical.cell_data["RecoveredActivationInv"], dtype=np.float64
    )
    displacement = np.asarray(historical.point_data["Displacement"], dtype=np.float64)
    if activation.shape != (mesh.n_cells, 6) or not np.isfinite(activation).all():
        msg = "historical recovered activation is malformed or non-finite"
        raise ValueError(msg)
    if displacement.shape != (mesh.n_points, 3) or not np.isfinite(displacement).all():
        msg = "historical displacement is malformed or non-finite"
        raise ValueError(msg)
    if not np.array_equal(
        activation, np.asarray(historical.cell_data[ACTIVATION_INV.vtk])
    ):
        msg = "historical RecoveredActivationInv and ActivationInv differ"
        raise ValueError(msg)
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    if not np.allclose(activation[~active], 0.0, rtol=0.0, atol=0.0):
        msg = "historical activation is nonzero outside ActivationMask"
        raise ValueError(msg)
    fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    if not np.allclose(displacement[fixed], 0.0, rtol=0.0, atol=1.0e-15):
        msg = "historical displacement violates the pinned fixed boundary"
        raise ValueError(msg)
    summary = _read_json(cfg.input_historical_summary)
    expected_summary = {
        "best/step": 40,
        "final/step": 40.0,
        "history/frames": 41,
        "n_points": mesh.n_points,
        "n_tets": mesh.n_cells,
        "n_skin_triangles": EXPECTED_FULL_TRIANGLES,
        "target/name": "smile",
        "case/setup": SETUP_SKIN_NO_PRESTRAIN,
        "activation/mode": "per-muscle-tet-6dof",
    }
    changed = {
        key: (summary.get(key), expected)
        for key, expected in expected_summary.items()
        if summary.get(key) != expected
    }
    if changed:
        msg = f"historical terminal/best provenance changed: {changed}"
        raise ValueError(msg)
    old_lambda, old_mu = _lame_parameters("3d")
    if not np.allclose(
        np.asarray(source_skin.cell_data[LAMBDA.vtk]),
        old_lambda,
        rtol=FORMULA_RTOL,
        atol=FORMULA_ATOL,
    ) or not np.allclose(
        np.asarray(source_skin.cell_data[MU.vtk]),
        old_mu,
        rtol=FORMULA_RTOL,
        atol=FORMULA_ATOL,
    ):
        msg = "historical control skin is no longer homogeneous full-domain 3D Lame"
        raise ValueError(msg)
    return (
        historical,
        target,
        loss_mask,
        activation,
        displacement,
        summary,
        {
            "identities": identities,
            "target_metrics": target_metrics,
            "mesh/global_point_id_source": (
                "explicit arange(n_points), matching ModelBuilder fallback and "
                "the pinned historical result"
            ),
        },
    )


def _solve_probe(  # noqa: PLR0915
    *,
    cfg: Config,
    base_mesh: pv.UnstructuredGrid,
    source_skin: pv.PolyData,
    basis: MetricBasis,
    activation: np.ndarray,
    old_displacement: np.ndarray,
    spec: ProbeSpec,
    seed: Seed,
) -> dict[str, Any]:
    case_mesh = base_mesh.copy(deep=True)
    cut_mesh_ids, cut_boundary_metrics = _configure_cut_boundary(
        case_mesh, source_skin, spec.cut_boundary
    )
    skin, skin_metrics = _make_probe_skin(
        case_mesh,
        source_skin,
        domain=spec.domain,
        conversion=spec.conversion,
    )
    forward, materials = _build_forward(case_mesh, skin)
    live_material_metrics = _validate_live_materials(
        case_mesh, skin, materials, conversion=spec.conversion
    )
    expected_fixed_dofs = int(
        np.asarray(case_mesh.point_data[FIXED_MASK.vtk], dtype=bool).sum()
    )
    if forward.model.n_fixed != expected_fixed_dofs:
        msg = (
            f"{spec.label} model fixed DoFs changed: "
            f"{forward.model.n_fixed} != {expected_fixed_dofs}"
        )
        raise ValueError(msg)
    activation_t = torch.as_tensor(
        activation,
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    materials["muscle"][ACTIVATION_INV.value] = activation_t
    forward.model.set_materials(materials)
    seed_displacement = (
        np.zeros_like(old_displacement) if seed == "zero" else old_displacement.copy()
    )
    seed_projection_vertices = 0
    seed_projection_enforced_zero_vertices = 0
    seed_projection_rms_m = 0.0
    if spec.cut_boundary == "hard-fixed":
        added_ids = np.flatnonzero(
            np.asarray(case_mesh.point_data["CutBoundaryAddedFixed"], dtype=bool)
        )
        if added_ids.size != EXPECTED_CUT_NEWLY_FIXED_VERTICES:
            msg = f"{spec.label} newly hard-fixed vertex count changed"
            raise ValueError(msg)
        if seed == "old":
            removed = seed_displacement[added_ids].copy()
            seed_projection_vertices = int(added_ids.size)
            seed_projection_enforced_zero_vertices = int(cut_mesh_ids.size)
            seed_projection_rms_m = float(
                np.linalg.norm(removed) / math.sqrt(added_ids.size)
            )
        # Enforce every hard-fixed value exactly, including the 380 vertices
        # already fixed by the historical boundary condition.  The reported
        # projection norm is restricted to the 6,600 newly constrained vertices.
        seed_displacement[cut_mesh_ids] = 0.0
        if not np.array_equal(
            seed_displacement[cut_mesh_ids],
            np.zeros_like(seed_displacement[cut_mesh_ids]),
        ):
            msg = f"{spec.label}/{seed} cut-fixed seed projection is not exact zero"
            raise ValueError(msg)
    forward.model.update(
        forward.state,
        torch.as_tensor(
            seed_displacement,
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        ),
    )
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        solution = forward.step()
    elapsed_s = time.perf_counter() - start
    displacement = to_numpy(forward.state.u).astype(np.float64, copy=True)
    cut_displacement = displacement[cut_mesh_ids]
    cut_final_rms_m = float(
        np.linalg.norm(cut_displacement) / math.sqrt(cut_mesh_ids.size)
    )
    cut_final_max_abs_m = float(np.abs(cut_displacement).max())
    if spec.cut_boundary == "hard-fixed" and not np.array_equal(
        cut_displacement, np.zeros_like(cut_displacement)
    ):
        msg = f"{spec.label}/{seed} final cut displacement is not exact zero"
        raise ValueError(msg)
    live_activation = to_numpy(
        forward.model.get_materials()["muscle"][ACTIVATION_INV.value]
    ).astype(np.float64, copy=True)
    if not np.array_equal(live_activation, activation):
        msg = f"{spec.label}/{seed} changed the fixed historical activation"
        raise ValueError(msg)
    solver_metrics = forward_solution_metrics(solution)
    if cfg.require_solver_success and not bool(solver_metrics["forward/success"]):
        msg = f"{spec.label}/{seed} forward solve failed: {solver_metrics}"
        raise RuntimeError(msg)
    metrics = {
        "case": spec.label,
        "domain": spec.domain,
        "conversion": spec.conversion,
        "cut_boundary": spec.cut_boundary,
        "seed": seed,
        "status": "ok",
        "interpretation": INTERPRETATION,
        "activation/source": str(cfg.input_historical_result),
        "activation/transferred": True,
        "activation/new_inverse_solution": False,
        "activation/fixed_during_forward": True,
        "activation/rms": float(
            np.linalg.norm(activation) / math.sqrt(activation.size)
        ),
        "activation/max_abs": float(np.abs(activation).max()),
        "initial_displacement/source": "exact-zero"
        if seed == "zero"
        else str(cfg.input_historical_result),
        "initial_displacement/projection": (
            "zero-on-newly-hard-fixed-artificial-cut-vertices"
            if spec.cut_boundary == "hard-fixed" and seed == "old"
            else "not-required-seed-is-exact-zero"
            if spec.cut_boundary == "hard-fixed"
            else "none"
        ),
        "initial_displacement/rms_m": float(
            np.linalg.norm(seed_displacement) / math.sqrt(seed_displacement.shape[0])
        ),
        **cut_boundary_metrics,
        "cut_boundary/model_total_fixed_dofs": expected_fixed_dofs,
        "cut_boundary/seed_projection_vertices": seed_projection_vertices,
        "cut_boundary/seed_projection_enforced_zero_vertices": (
            seed_projection_enforced_zero_vertices
        ),
        "cut_boundary/seed_projection_rms_m": seed_projection_rms_m,
        "cut_boundary/final_displacement_rms_m": cut_final_rms_m,
        "cut_boundary/final_displacement_max_abs_m": cut_final_max_abs_m,
        "forward/elapsed_s": float(elapsed_s),
        **skin_metrics,
        **live_material_metrics,
        **solver_metrics,
        **_fixed_activation_metrics(base_mesh, basis, displacement),
    }
    result = make_result_mesh(
        case_mesh,
        basis.target,
        basis.loss_mask,
        displacement,
        activation,
        {
            key: value
            for key, value in metrics.items()
            if isinstance(value, int | float | bool)
        },
    )
    case_dir = cfg.output_summary.parent / cfg.output_dir_name / spec.label / seed
    result_path = case_dir / "result.vtu"
    summary_path = case_dir / "forward-summary.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    melon.save(result, result_path)
    metrics.update(
        {
            "artifact/result_path": str(result_path),
            "artifact/result_size_bytes": result_path.stat().st_size,
            "artifact/result_sha256": file_sha256(result_path),
            "artifact/summary_path": str(summary_path),
        }
    )
    summary_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    cherries.log_output(result_path)
    cherries.log_output(summary_path)
    return {**metrics, "_displacement": displacement}


def _branch_summary(
    rows: list[dict[str, Any]], *, basis: MetricBasis, tolerance: float
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    face_mesh_ids = basis.skin_mesh_ids[basis.face_vertex_ids]
    face_target_rms = float(
        np.linalg.norm(basis.target[face_mesh_ids]) / math.sqrt(face_mesh_ids.size)
    )
    if not math.isfinite(face_target_rms) or face_target_rms <= 0.0:
        msg = f"invalid IsFace target RMS for branch normalization: {face_target_rms}"
        raise ValueError(msg)
    for spec in ALL_PROBES:
        by_seed = {str(row["seed"]): row for row in rows if row["case"] == spec.label}
        if set(by_seed) != {"zero", "old"}:
            msg = f"{spec.label} is missing a zero/old branch pair"
            raise ValueError(msg)
        delta = np.asarray(by_seed["zero"]["_displacement"]) - np.asarray(
            by_seed["old"]["_displacement"]
        )
        full_delta_rms = float(np.linalg.norm(delta) / math.sqrt(delta.shape[0]))
        loss_delta_rms = float(
            np.linalg.norm(delta[basis.loss_mask])
            / math.sqrt(int(basis.loss_mask.sum()))
        )
        face_delta_rms = float(
            np.linalg.norm(delta[face_mesh_ids]) / math.sqrt(face_mesh_ids.size)
        )
        loss_delta_fraction = loss_delta_rms / basis.target_rms
        face_delta_fraction = face_delta_rms / face_target_rms
        summaries.append(
            {
                "case": spec.label,
                "domain": spec.domain,
                "conversion": spec.conversion,
                "cut_boundary": spec.cut_boundary,
                "zero_old/full_displacement_delta_rms_m": full_delta_rms,
                "zero_old/loss_mask_displacement_delta_rms_m": loss_delta_rms,
                "zero_old/loss_mask_delta_fraction_of_target_rms": (
                    loss_delta_fraction
                ),
                "zero_old/isface_displacement_delta_rms_m": face_delta_rms,
                "zero_old/isface_delta_fraction_of_isface_target_rms": (
                    face_delta_fraction
                ),
                "zero_old_target_error_fraction_delta": float(
                    by_seed["zero"]["target/error_rms_fraction_of_target"]
                    - by_seed["old"]["target/error_rms_fraction_of_target"]
                ),
                "stable_within_declared_tolerance": (
                    loss_delta_fraction <= tolerance
                    and face_delta_fraction <= tolerance
                ),
                "gate_domains": ["SmileLossMask", "all-vertex IsFace ROI"],
                "tolerance_fraction_of_target_rms": tolerance,
                "interpretation_if_false": (
                    "branch-sensitive causal probe; do not rank the material/domain "
                    "choice from a single seed"
                ),
            }
        )
    return summaries


def _boundary_sensitivity_summary(
    rows: list[dict[str, Any]], *, basis: MetricBasis
) -> list[dict[str, Any]]:
    """Compare current and hard-fixed artificial-cut policies seed by seed."""
    face_mesh_ids = basis.skin_mesh_ids[basis.face_vertex_ids]
    face_target_rms = float(
        np.linalg.norm(basis.target[face_mesh_ids]) / math.sqrt(face_mesh_ids.size)
    )
    if not math.isfinite(face_target_rms) or face_target_rms <= 0.0:
        msg = (
            "invalid IsFace target RMS for artificial-cut boundary normalization: "
            f"{face_target_rms}"
        )
        raise ValueError(msg)
    summaries: list[dict[str, Any]] = []
    for seed in ("zero", "old"):
        reference = [
            row
            for row in rows
            if row["case"] == "isface-plane-stress" and row["seed"] == seed
        ]
        bracket = [
            row
            for row in rows
            if row["case"] == "isface-plane-stress-cut-fixed" and row["seed"] == seed
        ]
        if len(reference) != 1 or len(bracket) != 1:
            msg = f"missing unique current/hard-fixed boundary pair for seed {seed}"
            raise ValueError(msg)
        current = reference[0]
        hard_fixed = bracket[0]
        delta = np.asarray(hard_fixed["_displacement"]) - np.asarray(
            current["_displacement"]
        )
        full_delta_rms = float(np.linalg.norm(delta) / math.sqrt(delta.shape[0]))
        loss_delta_rms = float(
            np.linalg.norm(delta[basis.loss_mask])
            / math.sqrt(int(basis.loss_mask.sum()))
        )
        face_delta_rms = float(
            np.linalg.norm(delta[face_mesh_ids]) / math.sqrt(face_mesh_ids.size)
        )
        summaries.append(
            {
                "seed": seed,
                "reference_case": "isface-plane-stress",
                "bracket_case": "isface-plane-stress-cut-fixed",
                "full_displacement_delta_rms_m": full_delta_rms,
                "loss_mask_displacement_delta_rms_m": loss_delta_rms,
                "loss_mask_delta_fraction_of_target_rms": (
                    loss_delta_rms / basis.target_rms
                ),
                "isface_displacement_delta_rms_m": face_delta_rms,
                "isface_delta_fraction_of_isface_target_rms": (
                    face_delta_rms / face_target_rms
                ),
                "target_error_fraction_delta": float(
                    hard_fixed["target/error_rms_fraction_of_target"]
                    - current["target/error_rms_fraction_of_target"]
                ),
                "contraction_target_relative_dihedral_rms_deg_delta": float(
                    hard_fixed["bumpiness/contraction_target_relative_dihedral_rms_deg"]
                    - current["bumpiness/contraction_target_relative_dihedral_rms_deg"]
                ),
                "residual_normal_laplacian_rms_m_delta": float(
                    hard_fixed["bumpiness/residual_normal_laplacian_rms_m"]
                    - current["bumpiness/residual_normal_laplacian_rms_m"]
                ),
                "hard_fixed_is_ground_truth": False,
                "interpretation": (
                    "sensitivity bracket only; hard-fixed is not an anatomical "
                    "ground truth"
                ),
            }
        )
    return summaries


def _historical_replay_summary(
    rows: list[dict[str, Any]],
    *,
    basis: MetricBasis,
    historical_displacement: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    matches = [
        row for row in rows if row["case"] == "full-3d-replay" and row["seed"] == "old"
    ]
    if len(matches) != 1:
        msg = "expected one full+3D replay from the historical displacement"
        raise ValueError(msg)
    delta = np.asarray(matches[0]["_displacement"]) - historical_displacement
    face_mesh_ids = basis.skin_mesh_ids[basis.face_vertex_ids]
    loss_delta_rms = float(
        np.linalg.norm(delta[basis.loss_mask]) / math.sqrt(int(basis.loss_mask.sum()))
    )
    face_delta_rms = float(
        np.linalg.norm(delta[face_mesh_ids]) / math.sqrt(face_mesh_ids.size)
    )
    face_target_rms = float(
        np.linalg.norm(basis.target[face_mesh_ids]) / math.sqrt(face_mesh_ids.size)
    )
    loss_fraction = loss_delta_rms / basis.target_rms
    face_fraction = face_delta_rms / face_target_rms
    return {
        "case": "full-3d-replay/old",
        "purpose": (
            "verify that the current pinned forward path reproduces the old-model "
            "equilibrium before attributing changes to domain or conversion"
        ),
        "loss_mask_delta_rms_m": loss_delta_rms,
        "loss_mask_delta_fraction_of_target_rms": loss_fraction,
        "isface_delta_rms_m": face_delta_rms,
        "isface_delta_fraction_of_isface_target_rms": face_fraction,
        "reproduces_historical_control_within_tolerance": (
            loss_fraction <= tolerance and face_fraction <= tolerance
        ),
        "tolerance_fraction_of_corresponding_target_rms": tolerance,
        "eligibility_if_false": (
            "none of the domain/conversion contrasts may be interpreted causally"
        ),
    }


def _write_table(
    path: Path, control: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    lines = [
        "| case | domain | conversion | cut boundary | seed | status | Koiter tris | error/target | area-weighted error mm | dihedral deg | residual-normal Lap mm | disp Lap mm | forward |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        (
            "| {case} | {domain} | {conversion} | {cut_boundary} | {seed} | {status} | "
            "{koiter} | {error:.6g} | {area_error:.6g} | {dihedral:.6g} | "
            "{normal_lap:.6g} | {disp_lap:.6g} | {forward} |".format(
                case=row["case"],
                domain=row["domain"],
                conversion=row["conversion"],
                cut_boundary=row["cut_boundary"],
                seed=row["seed"],
                status=row["status"],
                koiter=row["koiter/live_n_triangles"],
                error=row["target/error_rms_fraction_of_target"],
                area_error=1.0e3 * row["target/face_rest_area_weighted_error_rms_m"],
                dihedral=row["bumpiness/contraction_target_relative_dihedral_rms_deg"],
                normal_lap=1.0e3 * row["bumpiness/residual_normal_laplacian_rms_m"],
                disp_lap=1.0e3 * row["bumpiness/displacement_laplacian_rms"],
                forward=row["forward/result"],
            )
        )
        for row in (control, *rows)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_config(cfg: Config) -> None:
    if cfg.probe_set != EXPECTED_PROBE_SET:
        msg = f"formal probe_set must be {EXPECTED_PROBE_SET!r}"
        raise ValueError(msg)
    if cfg.seed_set != EXPECTED_SEED_SET:
        msg = f"formal seed_set must be {EXPECTED_SEED_SET!r}"
        raise ValueError(msg)
    if cfg.require_solver_success is not True:
        msg = "formal fixed-activation probe requires every forward solve to succeed"
        raise ValueError(msg)
    if cfg.branch_delta_fraction_of_target_tol != 1.0e-3:
        msg = "formal branch tolerance must remain exactly 1e-3 of target RMS"
        raise ValueError(msg)
    if cfg.output_summary.resolve() == cfg.output_table.resolve():
        msg = "summary and table outputs must differ"
        raise ValueError(msg)
    if (
        cfg.output_summary.name != EXPECTED_OUTPUT_SUMMARY_NAME
        or cfg.output_table.name != EXPECTED_OUTPUT_TABLE_NAME
        or cfg.output_summary.parent.resolve() != cfg.output_table.parent.resolve()
        or cfg.output_summary.parent.resolve() != (GROUP_DIR / "data").resolve()
    ):
        msg = "formal summary/table output names or directories changed"
        raise ValueError(msg)
    if cfg.output_dir_name != EXPECTED_OUTPUT_DIR_NAME:
        msg = f"formal output_dir_name must be {EXPECTED_OUTPUT_DIR_NAME!r}"
        raise ValueError(msg)
    stale_outputs = [
        path
        for path in (
            cfg.output_summary,
            cfg.output_table,
            cfg.output_summary.parent / cfg.output_dir_name,
        )
        if path.exists()
    ]
    if stale_outputs:
        msg = (
            "refusing to overwrite an earlier/partial probe; review and remove or "
            f"archive explicitly: {[str(path) for path in stale_outputs]}"
        )
        raise FileExistsError(msg)
    if (
        Path(cfg.output_dir_name).is_absolute()
        or ".." in Path(cfg.output_dir_name).parts
    ):
        msg = "output_dir_name must be a safe relative path"
        raise ValueError(msg)


def run(cfg: Config) -> None:
    # Explicitly approved for this fixed-activation forward probe only. There is
    # no inverse or optimizer mode hidden behind a CLI flag.
    _validate_config(cfg)
    for path, expected, name in (
        (KOITER_IMPLEMENTATION, KOITER_IMPLEMENTATION_SHA256, "Koiter implementation"),
        (
            VOLUME_LAME_IMPLEMENTATION,
            VOLUME_LAME_IMPLEMENTATION_SHA256,
            "volume 3D Lame implementation",
        ),
        (
            VOLUME_FORWARD_IMPLEMENTATION,
            VOLUME_FORWARD_IMPLEMENTATION_SHA256,
            "volume forward builder",
        ),
        (TARGET_IMPLEMENTATION, TARGET_IMPLEMENTATION_SHA256, "target implementation"),
        (
            OUTPUT_IMPLEMENTATION,
            OUTPUT_IMPLEMENTATION_SHA256,
            "metric/output implementation",
        ),
        (
            CONFIG_IMPLEMENTATION,
            CONFIG_IMPLEMENTATION_SHA256,
            "experiment configuration",
        ),
        (
            CORE_MODULI_IMPLEMENTATION,
            CORE_MODULI_IMPLEMENTATION_SHA256,
            "core Lame converters",
        ),
    ):
        require_file_sha256(path, expected, name=name)
    mesh = pv.read(cfg.input_mesh)
    source_skin = pv.read(cfg.input_skin)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    if not isinstance(source_skin, pv.PolyData):
        msg = f"source skin read as {type(source_skin).__name__}"
        raise TypeError(msg)
    (
        historical,
        target,
        loss_mask,
        activation,
        old_displacement,
        historical_summary,
        input_provenance,
    ) = _validate_historical_inputs(cfg, mesh, source_skin)
    basis = _build_metric_basis(mesh, source_skin, target, loss_mask)
    control_metrics = _fixed_activation_metrics(
        mesh, basis, np.asarray(historical.point_data["Displacement"])
    )
    stored_error_fraction = float(
        historical_summary["best/error_rms_fraction_of_target"]
    )
    if not math.isclose(
        float(control_metrics["target/error_rms_fraction_of_target"]),
        stored_error_fraction,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        msg = "recomputed historical target metric differs from its pinned summary"
        raise ValueError(msg)
    control_domain = _validate_domain(mesh, source_skin, "full")
    control_lambda, control_mu = _lame_parameters("3d")
    control_cut_mesh = mesh.copy(deep=True)
    control_cut_ids, control_cut_boundary = _configure_cut_boundary(
        control_cut_mesh, source_skin, "current"
    )
    control_cut_displacement = np.asarray(historical.point_data["Displacement"])[
        control_cut_ids
    ]
    control = {
        "case": "historical-full-3d",
        "domain": "full",
        "conversion": "3d",
        "cut_boundary": "current",
        "seed": "historical-inverse-result",
        "status": "reused-pinned-control",
        "interpretation": (
            "historical full-boundary, 3D-Lame Koiter result; valid only as the "
            "old-model control and not as thin anatomical skin"
        ),
        "activation/transferred": False,
        "activation/new_inverse_solution": False,
        "koiter/live_n_triangles": EXPECTED_FULL_TRIANGLES,
        "koiter/live_lambda_MPa": control_lambda,
        "koiter/live_mu_MPa": control_mu,
        "forward/result": historical_summary["last/forward/result"],
        "forward/success": bool(historical_summary["last/forward/success"]),
        **control_cut_boundary,
        "cut_boundary/model_total_fixed_dofs": int(
            np.asarray(control_cut_mesh.point_data[FIXED_MASK.vtk], dtype=bool).sum()
        ),
        "cut_boundary/seed_projection_vertices": 0,
        "cut_boundary/seed_projection_enforced_zero_vertices": 0,
        "cut_boundary/seed_projection_rms_m": 0.0,
        "cut_boundary/final_displacement_rms_m": float(
            np.linalg.norm(control_cut_displacement) / math.sqrt(control_cut_ids.size)
        ),
        "cut_boundary/final_displacement_max_abs_m": float(
            np.abs(control_cut_displacement).max()
        ),
        **control_domain,
        **control_metrics,
        "artifact/result_path": str(cfg.input_historical_result),
        "artifact/result_sha256": HISTORICAL_RESULT_SHA256,
    }
    domain_material_preflight: dict[str, dict[str, Any]] = {}
    for spec in ALL_PROBES:
        preflight_mesh = mesh.copy(deep=True)
        _, boundary_preflight = _configure_cut_boundary(
            preflight_mesh, source_skin, spec.cut_boundary
        )
        _, skin_preflight = _make_probe_skin(
            preflight_mesh,
            source_skin,
            domain=spec.domain,
            conversion=spec.conversion,
        )
        domain_material_preflight[spec.label] = {
            **skin_preflight,
            **boundary_preflight,
        }
    # Delay CUDA/Warp initialization until every byte identity, topology,
    # target, activation, domain, and historical-control metric gate has passed.
    configure_runtime()
    rows: list[dict[str, Any]] = []
    for spec in ALL_PROBES:
        for seed in ("zero", "old"):
            logger.info(
                "Solving causal probe %s from %s displacement", spec.label, seed
            )
            rows.append(
                _solve_probe(
                    cfg=cfg,
                    base_mesh=mesh,
                    source_skin=source_skin,
                    basis=basis,
                    activation=activation,
                    old_displacement=old_displacement,
                    spec=spec,
                    seed=seed,
                )
            )
    branches = _branch_summary(
        rows,
        basis=basis,
        tolerance=cfg.branch_delta_fraction_of_target_tol,
    )
    boundary_sensitivity = _boundary_sensitivity_summary(rows, basis=basis)
    historical_replay = _historical_replay_summary(
        rows,
        basis=basis,
        historical_displacement=old_displacement,
        tolerance=cfg.branch_delta_fraction_of_target_tol,
    )
    branch_stable_all = len(branches) == len(ALL_PROBES) and all(
        bool(row["stable_within_declared_tolerance"]) for row in branches
    )
    historical_replay_stable = bool(
        historical_replay["reproduces_historical_control_within_tolerance"]
    )
    causal_contrasts_eligible = branch_stable_all and historical_replay_stable
    visual_review_status = "pending"
    serializable_rows = [
        {key: value for key, value in row.items() if key != "_displacement"}
        for row in rows
    ]
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "design": DESIGN,
        "interpretation": INTERPRETATION,
        "expensive_inverse_started": False,
        "historical_control": control,
        "new_forward_cases": serializable_rows,
        "branch_checks": branches,
        "boundary_sensitivity_checks": boundary_sensitivity,
        "comparison/branch_stable_all": branch_stable_all,
        "historical_replay_check": historical_replay,
        "comparison/causal_contrasts_eligible": causal_contrasts_eligible,
        "inverse/eligible_to_start": False,
        "inverse/eligibility_status": (
            "pending-visual-review"
            if causal_contrasts_eligible
            else "not-eligible-numeric-gates-failed"
        ),
        "inverse/required_gates": {
            "historical_full_3d_replay_stable": historical_replay_stable,
            "all_five_setup_seed_pairs_branch_stable": branch_stable_all,
            "matched_view_visual_review": visual_review_status,
            "policy": (
                "inverse requires both numeric gates and an accepted visual review; "
                "visual review is pending, so inverse remains ineligible"
            ),
        },
        "domain_material_preflight": domain_material_preflight,
        "input_provenance": input_provenance,
        "topology_provenance": {
            "prepared_volume": {
                "path": str(cfg.input_mesh),
                **input_provenance["identities"]["mesh"],
                "role": "pinned rest points, tetrahedral connectivity, and point order",
            },
            "historical_full_skin": {
                "path": str(cfg.input_skin),
                **input_provenance["identities"]["skin"],
                "role": (
                    "pinned historical full extracted-boundary topology for the "
                    "domain-by-conversion causal control"
                ),
                "n_triangles": EXPECTED_FULL_TRIANGLES,
                "canonical_global_triangle_sha256": EXPECTED_FULL_TOPOLOGY_SHA256,
                "artificial_cut": {
                    "marker": "mapped point GroupId == -1",
                    "unassigned_points": EXPECTED_FULL_UNASSIGNED_GROUP_POINTS,
                    "triangles_touching_unassigned_points": (
                        EXPECTED_FULL_CUT_TRIANGLES
                    ),
                    "canonical_global_triangle_sha256": (
                        EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256
                    ),
                    "incident_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
                    "incident_global_ids_sha256": (
                        EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256
                    ),
                    "preexisting_fixed_vertices": (
                        EXPECTED_CUT_PREEXISTING_FIXED_VERTICES
                    ),
                    "newly_fixed_vertices_in_bracket": (
                        EXPECTED_CUT_NEWLY_FIXED_VERTICES
                    ),
                    "policy": (
                        "intentional full-boundary diagnostic; never admitted to "
                        "the IsFace membrane ROI"
                    ),
                },
            },
            "artificial_cut_boundary_bracket": {
                "reference_case": "isface-plane-stress",
                "bracket_case": "isface-plane-stress-cut-fixed",
                "reference_policy": CURRENT_CUT_BOUNDARY_POLICY,
                "bracket_policy": HARD_FIXED_CUT_BOUNDARY_POLICY,
                "bracket_fixed_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
                "bracket_newly_fixed_vertices": (EXPECTED_CUT_NEWLY_FIXED_VERTICES),
                "fixed_value_m": [0.0, 0.0, 0.0],
                "historical_seed_projection": (
                    "measure and zero the 6,600 newly constrained vertices; "
                    "enforce exact zero on all 6,980 cut-incident vertices"
                ),
                "hard_fixed_is_ground_truth": False,
                "interpretation": (
                    "boundary-condition sensitivity bracket only; it is not an "
                    "anatomical boundary claim or an inverse-eligibility gate"
                ),
            },
            "isface_roi_derivation": {
                "source": "pinned historical_full_skin",
                "selection": (
                    "exact source cell_data IsFaceTriangle, then compact only the "
                    "referenced points without changing triangle orientation"
                ),
                "global_point_id_mapping": input_provenance[
                    "mesh/global_point_id_source"
                ],
                "n_triangles": EXPECTED_ISFACE_TRIANGLES,
                "canonical_global_triangle_sha256": (EXPECTED_ISFACE_TOPOLOGY_SHA256),
                "expected_components": EXPECTED_ISFACE_COMPONENTS,
                "allowed_group_names": list(EXPECTED_ISFACE_GROUP_NAMES),
            },
            "runtime_gate": (
                "both domains recompute canonical sorted GlobalPointId triangle "
                "digests, exact area, component count, and mesh-coordinate mapping"
            ),
        },
        "implementation": {
            "koiter/path": str(KOITER_IMPLEMENTATION),
            "koiter/sha256": KOITER_IMPLEMENTATION_SHA256,
            "volume_lame/path": str(VOLUME_LAME_IMPLEMENTATION),
            "volume_lame/sha256": VOLUME_LAME_IMPLEMENTATION_SHA256,
            "volume_forward/path": str(VOLUME_FORWARD_IMPLEMENTATION),
            "volume_forward/sha256": VOLUME_FORWARD_IMPLEMENTATION_SHA256,
            "target/path": str(TARGET_IMPLEMENTATION),
            "target/sha256": TARGET_IMPLEMENTATION_SHA256,
            "output_metrics/path": str(OUTPUT_IMPLEMENTATION),
            "output_metrics/sha256": OUTPUT_IMPLEMENTATION_SHA256,
            "core_moduli/path": str(CORE_MODULI_IMPLEMENTATION),
            "core_moduli/sha256": CORE_MODULI_IMPLEMENTATION_SHA256,
            "skin_plane_stress_converter": (
                "liblaf.apple.common.lame_converter_plane_stress"
            ),
        },
        "fixed_design": {
            "factorial_cells": [
                "historical full+3D result (reused as pinned control)",
                "full+3D fixed-activation replay (two new forward solves)",
                "full+plane-stress",
                "IsFace+3D",
                "IsFace+plane-stress",
                "IsFace+plane-stress+artificial-cut hard-fixed sensitivity bracket",
            ],
            "new_forward_solves": 10,
            "new_inverse_solves": 0,
            "new_setup_count": len(CHANGED_PROBES),
            "changed_probe_labels": [spec.label for spec in CHANGED_PROBES],
            "replayed_reference_label": ALL_PROBES[0].label,
            "seeds": ["zero", "old historical displacement"],
            "activation": (
                "pinned best=terminal step-40 e100-p000 full+3D activation, "
                "held fixed exactly"
            ),
            "skin_E_MPa": float(SKIN_E),
            "skin_nu_3d_input": float(SKIN_NU),
            "skin_thickness_m": float(SKIN_THICKNESS),
            "skin_prestrain": "none",
            "volume_conversion": "3d and unchanged",
            "target": "pinned Smile displacement and SmileLossMask",
            "branch_policy": (
                "compare the equilibria reached from exact-zero and historical "
                "displacement; a failed tolerance makes the probe branch-sensitive, "
                "not a single-equilibrium material ranking"
            ),
            "cut_boundary_policy": (
                "compare current IsFixed and hard-zero all 6,980 vertices incident "
                "to the 13,165 artificial-cut triangles, seed matched; report as "
                "sensitivity only, never ground truth"
            ),
            "fold_inversion_policy": "warning-only; matched-view visual review decides",
            "rng_used": False,
            "numeric_reproducibility": (
                "no stochastic sampling is used; CUDA reductions and nonlinear "
                "solves are not assumed bitwise reproducible"
            ),
        },
        "visual_review": {
            "status": visual_review_status,
            "generated_by_this_script": False,
            "available_inputs": (
                "each forward writes a full result.vtu with rest points, target, "
                "displacement, and the exact transferred activation"
            ),
            "required_follow_up": (
                "run a separately reviewed static matched-view analyzer before "
                "approving an inverse; this entrypoint does not silently import "
                "the inverse-screen analyzer or claim visual acceptance"
            ),
        },
        "output_contract": {
            "root": str((cfg.output_summary.parent / cfg.output_dir_name).resolve()),
            "summary_path": str(cfg.output_summary.resolve()),
            "table_path": str(cfg.output_table.resolve()),
            "case_order": [spec.label for spec in ALL_PROBES],
            "seed_order": ["zero", "old"],
            "case_layout": "<root>/<case>/<seed>/{result.vtu,forward-summary.json}",
            "expected_result_vtus": 10,
            "expected_forward_sidecars": 10,
            "overwrite_policy": (
                "refuse before input reads or runtime initialization if summary, "
                "table, or result root already exists"
            ),
        },
        "decision_rule_before_inverse": (
            "first require the full+3D old-seed replay to reproduce the historical "
            "control and all five setup seed pairs to pass the branch gate; then "
            "review target fidelity, area-weighted face error, target-relative "
            "contraction dihedral, residual-normal high-frequency roughness, the "
            "seed-matched current-vs-hard-fixed boundary sensitivity, and matched "
            "visual views. Inverse remains ineligible until visual review is "
            "explicitly accepted."
        ),
    }
    cfg.output_summary.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_table(cfg.output_table, control, serializable_rows)
    cherries.log_output(cfg.output_summary)
    cherries.log_output(cfg.output_table)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_table)


if __name__ == "__main__":
    cherries.main(run)
