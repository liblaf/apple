from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from _reference import (
    GROUP_DIR,
    KOITER_IMPLEMENTATION,
    KOITER_IMPLEMENTATION_SHA256,
    LEGACY_INVERSE,
    LEGACY_INVERSE_SHA256,
    MATERIAL_REFERENCE_SRC,
    PREPARED_MESH,
    REPO_ROOT,
    RUNTIME_REFERENCE_SRC,
    SOURCE_SKIN,
    SOURCE_SKIN_SHA256,
    SOURCE_SKIN_SIZE_BYTES,
    VOLUME_FORWARD_IMPLEMENTATION,
    VOLUME_FORWARD_IMPLEMENTATION_SHA256,
    VOLUME_LAME_IMPLEMENTATION,
    VOLUME_LAME_IMPLEMENTATION_SHA256,
    enable_reference_modules,
    file_sha256,
    load_pinned_module,
    require_file_sha256,
)

from liblaf import cherries
from liblaf.apple.common import (
    ACTIVATION_INV,
    FIXED_MASK,
    FIXED_VALUE,
    FRACTION,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
)

mpl.use("Agg", force=True)
logger = logging.getLogger(__name__)

enable_reference_modules()
legacy, LEGACY_IDENTITY = load_pinned_module(
    LEGACY_INVERSE,
    LEGACY_INVERSE_SHA256,
    module_name="_corrected_baseline_inverse_reference",
)

MANIFEST_DESIGN = "isface-plane-stress-corrected-baseline"
DESIGN = "isface-plane-stress-hard-fixed-corrected-baseline-inverse"
MANIFEST_SCHEMA_VERSION = 3
AGGREGATE_SCHEMA_VERSION = 4
CANDIDATE_LABEL = "isface-e0200-p000"
EXPECTED_CANDIDATES = {CANDIDATE_LABEL: (1.0, 0.0)}
EXPECTED_LABELS = (CANDIDATE_LABEL,)
EXPECTED_PROTOCOL = dict(legacy.EXPECTED_PROTOCOL)
EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_FULL_BOUNDARY_TRIANGLES = 128_172
EXPECTED_SKIN_AREA_M2 = 0.04287998059707303
EXPECTED_SKIN_COMPONENTS = 1
EXPECTED_FULL_UNASSIGNED_GROUP_POINTS = 6_000
EXPECTED_ARTIFICIAL_CUT_TRIANGLES = 13_165
EXPECTED_CUT_INCIDENT_VERTICES = 6_980
EXPECTED_CUT_PREEXISTING_FIXED_VERTICES = 380
EXPECTED_CUT_NEWLY_FIXED_VERTICES = 6_600
EXPECTED_MODEL_FIXED_VERTICES = 33_636
EXPECTED_MODEL_FIXED_DOFS = 100_908
EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256 = (
    "8207cda8f9e11dbb4406f683e5ad818a6950e3515ac373719514094fb5b7fe5d"
)
EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256 = (
    "ca39cdc839855be34e75222964a1e5c129dd210e8800c684d7e6d1ce6424f138"
)
INVERSE_RUNTIME_DEPENDENCIES = (
    (
        "runtime/_human_face_case.py",
        RUNTIME_REFERENCE_SRC / "_human_face_case.py",
        "5e9e67be5246ecc9cf38c3a5c53fe4e2041c3b6af272dceab1ed8c94a9bf5d68",
    ),
    (
        "runtime/_human_face_config.py",
        RUNTIME_REFERENCE_SRC / "_human_face_config.py",
        "eca100cc6bdd4e2a1ac689c6e2e7e02cf80a9bea8fa9ac12e9590eca5f23ffb6",
    ),
    (
        "runtime/_human_face_forward.py",
        RUNTIME_REFERENCE_SRC / "_human_face_forward.py",
        "2d0ff39b13555300c000e6dd43e16c274752263b703746ad8174072033819e03",
    ),
    (
        "runtime/_human_face_loop.py",
        RUNTIME_REFERENCE_SRC / "_human_face_loop.py",
        "97a47be4f6140a0213a84b460c8585f92e38657b01db4ae21f67177048f915c5",
    ),
    (
        "runtime/_human_face_mesh.py",
        RUNTIME_REFERENCE_SRC / "_human_face_mesh.py",
        "f1e1cdc806273c4ce5a37e52e3032d357b44bfd201de3fc58c35d793d11454bc",
    ),
    (
        "runtime/_human_face_metrics.py",
        RUNTIME_REFERENCE_SRC / "_human_face_metrics.py",
        "1407d2988444b31332f2688c6535eca5db58b5be31d63fae6abd6bf8bf78e0c1",
    ),
    (
        "runtime/_human_face_output.py",
        RUNTIME_REFERENCE_SRC / "_human_face_output.py",
        "29bae977a4b31e82276aca15fdaae3bdda37e6a3e71493876b6fd973db1a1c61",
    ),
    (
        "runtime/_human_face_runtime.py",
        RUNTIME_REFERENCE_SRC / "_human_face_runtime.py",
        "b2aefe4b5cd702c837d08442f7b588fafb80f1e6c8a745eed874ce18fdce1f45",
    ),
    (
        "runtime/_human_face_skin.py",
        RUNTIME_REFERENCE_SRC / "_human_face_skin.py",
        "a3bded895ff949dab274707e068d323d1277284a46680fe513b069e207a119a9",
    ),
    (
        "runtime/_human_face_targets.py",
        RUNTIME_REFERENCE_SRC / "_human_face_targets.py",
        "34a1583fcb8f90f357647dd4574e2e7ef27f8049f2b3ba1e2fa7dc838fcbb696",
    ),
    (
        "material/_material_heuristics.py",
        MATERIAL_REFERENCE_SRC / "_material_heuristics.py",
        "d21091bb931ed2d218d65f72305792e1a48ced5e703d6b344388d2d1d803c84f",
    ),
    (
        "core/src/liblaf/apple/inverse/_diff_forward.py",
        REPO_ROOT / "src/liblaf/apple/inverse/_diff_forward.py",
        "72de3eeb2a1cfe9addc29aea812f13c077a4f2e098ab65bbe564837d04a5fe30",
    ),
)
EXPECTED_INVERSE_RUNTIME_BUNDLE_SHA256 = (
    "3086071201576008047a0b86394e4282c8dc2d37bc0c21a8c8bd4edc73932426"
)
HARD_FIXED_CUT_BOUNDARY_POLICY = "all-artificial-cut-incident-vertices-hard-fixed"
AREA_ATOL_M2 = 5.0e-13
FORMULA_RTOL = 1.0e-13
FORMULA_ATOL = 1.0e-14
FACE_GROUPS = (
    "Chin",
    "EyelidBottom",
    "EyelidOuterBottom",
    "EyelidOuterTop",
    "EyelidTop",
    "Face",
    "LipBottom",
    "LipOuterBottom",
    "LipOuterTop",
    "LipTop",
)

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
FORMAL_APPROVAL_BLOCKER = (
    "NO-GO: the 40-update corrected-baseline inverse remains approval-blocked; "
    "the zero-step smoke authorization must not unlock the formal screen."
)
SMOKE_APPROVAL_BLOCKER = (
    "NO-GO: the isolated zero-step hard-fixed smoke awaits root static review."
)
FORMAL_APPROVED_AFTER_USER_REVIEW = True
SMOKE_APPROVED_AFTER_ROOT_REVIEW = True

PREPARE_IMPLEMENTATION = Path(__file__).with_name("10-prepare-plane-stress-skin.py")
PREPARE_IMPLEMENTATION_SHA256 = (
    "b0a547389dbb192e46732e84bd649d27ee4e89246bf6823d7dcc587322d4bed9"
)
EXPECTED_INPUT_CANDIDATES = GROUP_DIR / "data/10-corrected-baseline-manifest.json"
EXPECTED_OUTPUT_SUMMARY = GROUP_DIR / "data/20-corrected-baseline-screen-summary.json"
EXPECTED_OUTPUT_TABLE = GROUP_DIR / "data/20-corrected-baseline-screen-table.md"
EXPECTED_LIVE_PLOT_DIR = GROUP_DIR / "figs/live-corrected-baseline-screen"
EXPECTED_SMOKE_ROOT = GROUP_DIR / "tmp/hard-fixed-smoke-v3"
EXPECTED_SMOKE_OUTPUT_SUMMARY = (
    EXPECTED_SMOKE_ROOT / "20-corrected-baseline-hard-fixed-smoke-summary.json"
)
EXPECTED_SMOKE_OUTPUT_TABLE = (
    EXPECTED_SMOKE_ROOT / "20-corrected-baseline-hard-fixed-smoke-table.md"
)
EXPECTED_SMOKE_LIVE_PLOT_DIR = (
    EXPECTED_SMOKE_ROOT / "figs/live-corrected-baseline-hard-fixed-smoke"
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_cut_reference: Path = cherries.input(SOURCE_SKIN)
    input_candidates: Path = cherries.input("10-corrected-baseline-manifest.json")
    output_summary: Path = cherries.output(
        "20-corrected-baseline-screen-summary.json", mkdir=True
    )
    output_table: Path = cherries.output(
        "20-corrected-baseline-screen-table.md", mkdir=True
    )
    live_plot_dir: Path = Path("figs/live-corrected-baseline-screen")

    stage: str = "screen"
    candidate_set: str = CANDIDATE_LABEL
    initial_activation_mesh: Path | None = None
    use_initial_displacement: bool = False
    inverse_lr: float = 0.3
    loss_scale: float = legacy.LOSS_SCALE
    adam_eps: float = legacy.ADAM_EPS
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
    time_budget_hours: float = 6.0
    reserve_minutes: float = 5.0
    step_time_budget_s: float = 180.0
    require_convergence: bool = False
    require_solver_success: bool = True
    max_solver_failure_fraction: float = 0.0


def _file_identity(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        msg = f"missing required file: {path}"
        raise FileNotFoundError(msg)
    return {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _require_inverse_runtime_identity(*, context: str) -> dict[str, Any]:
    files: list[dict[str, int | str]] = []
    bundle_payload = bytearray()
    for label, path, expected_sha256 in INVERSE_RUNTIME_DEPENDENCIES:
        actual_sha256 = require_file_sha256(
            path,
            expected_sha256,
            name=f"{context} inverse runtime dependency {label}",
        )
        files.append(
            {
                "label": label,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": actual_sha256,
            }
        )
        bundle_payload.extend(f"{label}\0{actual_sha256}\n".encode())
    bundle_sha256 = hashlib.sha256(bundle_payload).hexdigest()
    if bundle_sha256 != EXPECTED_INVERSE_RUNTIME_BUNDLE_SHA256:
        msg = (
            f"{context} inverse runtime bundle digest changed: "
            f"{bundle_sha256} != {EXPECTED_INVERSE_RUNTIME_BUNDLE_SHA256}"
        )
        raise ValueError(msg)
    return {
        "algorithm": "sha256(label + NUL + file_sha256 + newline), ordered",
        "bundle_sha256": bundle_sha256,
        "files": files,
    }


def _validate_corrected_paths(cfg: Config) -> None:  # noqa: C901
    if cfg.stage == "screen":
        stage_exact = {
            "output_summary": (cfg.output_summary, EXPECTED_OUTPUT_SUMMARY),
            "output_table": (cfg.output_table, EXPECTED_OUTPUT_TABLE),
            "live_plot_dir": (cfg.live_plot_dir, EXPECTED_LIVE_PLOT_DIR),
        }
    elif cfg.stage == "smoke":
        stage_exact = {
            "output_summary": (
                cfg.output_summary,
                EXPECTED_SMOKE_OUTPUT_SUMMARY,
            ),
            "output_table": (cfg.output_table, EXPECTED_SMOKE_OUTPUT_TABLE),
            "live_plot_dir": (
                cfg.live_plot_dir,
                EXPECTED_SMOKE_LIVE_PLOT_DIR,
            ),
        }
    else:
        msg = f"stage must be screen or smoke, got {cfg.stage!r}"
        raise ValueError(msg)
    exact = {
        "input_mesh": (cfg.input_mesh, PREPARED_MESH),
        "input_cut_reference": (cfg.input_cut_reference, SOURCE_SKIN),
        "input_candidates": (cfg.input_candidates, EXPECTED_INPUT_CANDIDATES),
        **stage_exact,
    }
    mismatches = [
        f"{name}: {actual} != {expected}"
        for name, (actual, expected) in exact.items()
        if actual.resolve() != expected.resolve()
    ]
    if mismatches:
        msg = "corrected inverse paths differ from the reviewed contract: " + "; ".join(
            mismatches
        )
        raise ValueError(msg)
    if cfg.stage == "screen":
        if cfg.inverse_max_steps != 40 or cfg.mandatory_baseline_steps != 40:
            msg = "formal corrected baseline requires exactly 40 optimizer updates"
            raise ValueError(msg)
    elif cfg.inverse_max_steps != 0 or cfg.mandatory_baseline_steps != 0:
        msg = "smoke requires zero optimizer updates and one evaluation"
        raise ValueError(msg)
    if cfg.candidate_set != CANDIDATE_LABEL:
        msg = f"corrected baseline requires only {CANDIDATE_LABEL}"
        raise ValueError(msg)
    if cfg.initial_activation_mesh is not None or cfg.use_initial_displacement:
        msg = "corrected baseline must start from exact-zero activation/displacement"
        raise ValueError(msg)
    if cfg.stage == "smoke":
        smoke_root = EXPECTED_SMOKE_ROOT.resolve()
        smoke_outputs = (
            cfg.output_summary.resolve(),
            cfg.output_table.resolve(),
            cfg.live_plot_dir.resolve(),
            _expected_case_paths(cfg).target.resolve(),
            _expected_case_paths(cfg).result.resolve(),
            _expected_case_paths(cfg).summary.resolve(),
            _expected_case_paths(cfg).history.resolve(),
            _expected_case_paths(cfg).trace.resolve(),
            _canonical_archive_paths(cfg)[0].resolve(),
            _canonical_archive_paths(cfg)[1].resolve(),
        )
        escaped = [
            str(path) for path in smoke_outputs if not path.is_relative_to(smoke_root)
        ]
        if escaped:
            msg = f"smoke output escaped the isolated tmp root: {escaped}"
            raise ValueError(msg)


def _triangles(surface: pv.PolyData) -> np.ndarray:
    faces = np.asarray(surface.faces, dtype=np.int64)
    if surface.n_cells == 0 or faces.size != 4 * surface.n_cells:
        msg = "Koiter input must be a non-empty triangle-only PolyData"
        raise ValueError(msg)
    encoded = faces.reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        msg = "Koiter input contains a non-triangle face"
        raise ValueError(msg)
    return encoded[:, 1:]


def _triangle_area(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def _map_global_ids(mesh: pv.UnstructuredGrid, surface: pv.PolyData) -> np.ndarray:
    mesh_ids = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    requested = np.asarray(surface.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if mesh_ids.shape != (mesh.n_points,) or np.unique(mesh_ids).size != mesh.n_points:
        msg = "prepared mesh GlobalPointId values are malformed or non-unique"
        raise ValueError(msg)
    if (
        requested.shape != (surface.n_points,)
        or np.unique(requested).size != surface.n_points
    ):
        msg = "cut-reference GlobalPointId values are malformed or non-unique"
        raise ValueError(msg)
    order = np.argsort(mesh_ids)
    positions = np.searchsorted(mesh_ids[order], requested)
    if np.any(positions >= mesh_ids.size) or not np.array_equal(
        mesh_ids[order[positions]], requested
    ):
        msg = "cut-reference GlobalPointId values do not map to the prepared mesh"
        raise ValueError(msg)
    mapped = order[positions]
    if not np.array_equal(
        np.asarray(surface.points, dtype=np.float64),
        np.asarray(mesh.points, dtype=np.float64)[mapped],
    ):
        msg = "cut-reference points differ from the mapped prepared-mesh points"
        raise ValueError(msg)
    return mapped


def _configure_hard_fixed_cut_boundary(  # noqa: C901, PLR0912, PLR0915
    mesh: pv.UnstructuredGrid,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive and install the pinned conservative artificial-cut constraint."""
    source_identity = _file_identity(SOURCE_SKIN)
    if source_identity != {
        "size_bytes": SOURCE_SKIN_SIZE_BYTES,
        "sha256": SOURCE_SKIN_SHA256,
    }:
        msg = f"artificial-cut reference identity changed: {source_identity}"
        raise ValueError(msg)
    source = pv.read(SOURCE_SKIN)
    if not isinstance(source, pv.PolyData):
        msg = f"artificial-cut reference read as {type(source).__name__}"
        raise TypeError(msg)
    if source.n_cells != EXPECTED_FULL_BOUNDARY_TRIANGLES:
        msg = (
            "artificial-cut reference triangle count changed: "
            f"{source.n_cells} != {EXPECTED_FULL_BOUNDARY_TRIANGLES}"
        )
        raise ValueError(msg)

    local_global_ids = np.arange(mesh.n_points, dtype=np.int64)
    if GLOBAL_POINT_ID.vtk in mesh.point_data and not np.array_equal(
        np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64),
        local_global_ids,
    ):
        msg = "prepared mesh contains a non-canonical GlobalPointId field"
        raise ValueError(msg)
    mesh.point_data[GLOBAL_POINT_ID.vtk] = local_global_ids

    triangles = _triangles(source)
    source_group_ids = np.asarray(source.point_data["GroupId"], dtype=np.int64)
    if source_group_ids.shape != (source.n_points,):
        msg = "artificial-cut reference GroupId field is malformed"
        raise ValueError(msg)
    unassigned = source_group_ids == -1
    if int(unassigned.sum()) != EXPECTED_FULL_UNASSIGNED_GROUP_POINTS:
        msg = (
            "artificial-cut marker point count changed: "
            f"{int(unassigned.sum())} != {EXPECTED_FULL_UNASSIGNED_GROUP_POINTS}"
        )
        raise ValueError(msg)
    cut_triangles = np.any(unassigned[triangles], axis=1)
    if int(cut_triangles.sum()) != EXPECTED_ARTIFICIAL_CUT_TRIANGLES:
        msg = (
            "artificial-cut triangle count changed: "
            f"{int(cut_triangles.sum())} != {EXPECTED_ARTIFICIAL_CUT_TRIANGLES}"
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
    cut_global_ids = np.sort(local_global_ids[cut_mesh_ids]).astype("<i8", copy=False)
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
    historical_is_fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    fixed_mask = np.asarray(mesh.point_data[FIXED_MASK.vtk], dtype=bool)
    fixed_value = np.asarray(mesh.point_data[FIXED_VALUE.vtk], dtype=np.float64)
    if is_face.shape != (mesh.n_points,) or historical_is_fixed.shape != (
        mesh.n_points,
    ):
        msg = "prepared IsFace/IsFixed fields are malformed"
        raise ValueError(msg)
    if fixed_mask.shape != (mesh.n_points, 3) or fixed_value.shape != (
        mesh.n_points,
        3,
    ):
        msg = "prepared FixedMask/FixedValue fields are malformed"
        raise ValueError(msg)
    if not np.array_equal(
        fixed_mask, np.repeat(historical_is_fixed[:, None], 3, axis=1)
    ):
        msg = "prepared FixedMask is inconsistent with historical IsFixed"
        raise ValueError(msg)
    if not np.array_equal(fixed_value, np.zeros_like(fixed_value)):
        msg = "corrected baseline requires exact-zero prepared FixedValue"
        raise ValueError(msg)
    if np.any(is_face[cut_mesh_ids]):
        msg = "artificial-cut incident vertices unexpectedly overlap IsFace"
        raise ValueError(msg)

    preexisting = historical_is_fixed[cut_mesh_ids]
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
            "artificial-cut newly fixed vertex count changed: "
            f"{added_ids.size} != {EXPECTED_CUT_NEWLY_FIXED_VERTICES}"
        )
        raise ValueError(msg)

    is_fixed = historical_is_fixed.copy()
    installed_mask = fixed_mask.copy()
    installed_value = fixed_value.copy()
    is_fixed[cut_mesh_ids] = True
    installed_mask[cut_mesh_ids] = True
    installed_value[cut_mesh_ids] = 0.0
    incident = np.zeros(mesh.n_points, dtype=np.int8)
    incident[cut_mesh_ids] = 1
    preexisting_field = np.zeros(mesh.n_points, dtype=np.int8)
    preexisting_field[cut_mesh_ids[preexisting]] = 1
    added_field = np.zeros(mesh.n_points, dtype=np.int8)
    added_field[added_ids] = 1
    mesh.point_data["HistoricalIsFixed"] = historical_is_fixed.astype(np.int8)
    mesh.point_data["ArtificialCutIncident"] = incident
    mesh.point_data["CutBoundaryPreexistingFixed"] = preexisting_field
    mesh.point_data["CutBoundaryAddedFixed"] = added_field
    mesh.point_data["IsFixed"] = is_fixed
    mesh.point_data[FIXED_MASK.vtk] = installed_mask
    mesh.point_data[FIXED_VALUE.vtk] = installed_value

    persisted_is_fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    persisted_fixed_mask = np.asarray(mesh.point_data[FIXED_MASK.vtk], dtype=bool)
    persisted_fixed_value = np.asarray(
        mesh.point_data[FIXED_VALUE.vtk], dtype=np.float64
    )
    if not np.array_equal(
        persisted_fixed_mask,
        np.repeat(persisted_is_fixed[:, None], 3, axis=1),
    ):
        msg = "hard-fixed persisted FixedMask is inconsistent with IsFixed"
        raise ValueError(msg)
    if not np.array_equal(
        persisted_fixed_value[persisted_is_fixed],
        np.zeros_like(persisted_fixed_value[persisted_is_fixed]),
    ):
        msg = "hard-fixed persisted FixedValue is not exact zero"
        raise ValueError(msg)
    if int(persisted_is_fixed[cut_mesh_ids].sum()) != EXPECTED_CUT_INCIDENT_VERTICES:
        msg = "not every artificial-cut incident vertex is hard-fixed"
        raise ValueError(msg)
    model_fixed_vertices = int(persisted_is_fixed.sum())
    model_fixed_dofs = int(persisted_fixed_mask.sum())
    if model_fixed_vertices != EXPECTED_MODEL_FIXED_VERTICES:
        msg = (
            f"model fixed vertex count changed: {model_fixed_vertices} != "
            f"{EXPECTED_MODEL_FIXED_VERTICES}"
        )
        raise ValueError(msg)
    if model_fixed_dofs != EXPECTED_MODEL_FIXED_DOFS:
        msg = (
            f"model fixed DoF count changed: {model_fixed_dofs} != "
            f"{EXPECTED_MODEL_FIXED_DOFS}"
        )
        raise ValueError(msg)

    return cut_mesh_ids, {
        "cut_boundary/policy": HARD_FIXED_CUT_BOUNDARY_POLICY,
        "cut_boundary/marker": (
            "source skin triangle touches mapped GroupId=-1 vertex"
        ),
        "cut_boundary/reference_path": str(SOURCE_SKIN),
        "cut_boundary/reference_size_bytes": SOURCE_SKIN_SIZE_BYTES,
        "cut_boundary/reference_sha256": SOURCE_SKIN_SHA256,
        "cut_boundary/triangles": int(cut_triangles.sum()),
        "cut_boundary/triangle_topology_sha256": cut_topology_sha256,
        "cut_boundary/incident_vertices": int(cut_mesh_ids.size),
        "cut_boundary/incident_global_ids_sha256": cut_global_ids_sha256,
        "cut_boundary/preexisting_fixed_vertices": int(preexisting.sum()),
        "cut_boundary/newly_fixed_vertices": int(added_ids.size),
        "cut_boundary/total_fixed_vertices": int(
            persisted_is_fixed[cut_mesh_ids].sum()
        ),
        "cut_boundary/model_total_fixed_vertices": model_fixed_vertices,
        "cut_boundary/model_total_fixed_dofs": model_fixed_dofs,
        "cut_boundary/fixed_values_max_abs_m": float(
            np.abs(persisted_fixed_value[persisted_is_fixed]).max()
        ),
        "cut_boundary/configured_exact_zero": True,
        "cut_boundary/hard_fixed_is_ground_truth": False,
        "protocol/forward_initial_displacement_exact_zero": True,
        "protocol/forward_initial_displacement_max_abs_m": 0.0,
    }


def _component_count(triangles: np.ndarray) -> tuple[int, int]:
    n_faces = triangles.shape[0]
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    owners = np.tile(np.arange(n_faces, dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owners = owners[order]
    starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1]
    stops = np.r_[starts[1:], edges.shape[0]]

    parent = np.arange(n_faces, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for start, stop in zip(starts, stops, strict=True):
        first = int(owners[start])
        for owner in owners[start + 1 : stop]:
            union(first, int(owner))
    components = len({find(face) for face in range(n_faces)})
    nonmanifold = sum(
        stop - start > 2 for start, stop in zip(starts, stops, strict=True)
    )
    return components, nonmanifold


def _require_manifest_domain(manifest: dict[str, Any]) -> None:  # noqa: C901
    domain = manifest.get("domain_contract")
    if not isinstance(domain, dict):
        msg = "manifest is missing the audited domain contract"
        raise TypeError(msg)
    exact = {
        "full_boundary_triangles": EXPECTED_FULL_BOUNDARY_TRIANGLES,
        "source_outer_triangles": 115_007,
        "artificial_cut_triangles": 13_165,
        "full_boundary_unassigned_group_points": (
            EXPECTED_FULL_UNASSIGNED_GROUP_POINTS
        ),
        "skin_triangles": EXPECTED_SKIN_TRIANGLES,
        "skin_artificial_cut_overlap_triangles": 0,
        "skin_fixed_overlap_triangles": 0,
        "skin_disallowed_group_overlap_triangles": 0,
        "skin_nonfinite_target_triangles": 0,
    }
    for key, expected in exact.items():
        if int(domain.get(key, -1)) != expected:
            msg = f"domain contract {key} changed: {domain.get(key)!r} != {expected}"
            raise ValueError(msg)
    if domain.get("validation_ok") is not True or domain.get("validation_errors") != []:
        msg = "manifest domain audit did not pass"
        raise ValueError(msg)
    if domain.get("selection") != "all three triangle vertices have IsFace=true":
        msg = "manifest skin-domain selection changed"
        raise ValueError(msg)
    if domain.get("face_group_allowlist") != list(FACE_GROUPS):
        msg = "manifest facial anatomy allowlist differs from Melon FACE_GROUPS"
        raise ValueError(msg)
    if set(domain.get("observed_skin_group_names", [])) != set(FACE_GROUPS):
        msg = "manifest membrane groups do not cover the exact FACE_GROUPS set"
        raise ValueError(msg)
    for key in (
        "skin_teeth_proximity_overlap_triangles",
        "skin_gingiva_proximity_overlap_triangles",
    ):
        if int(domain.get(key, -1)) < 0:
            msg = f"manifest is missing proximity diagnostic {key}"
            raise ValueError(msg)
    if domain.get("isface_global_face_key_sha256") != domain.get(
        "topology_reference_global_face_key_sha256"
    ):
        msg = "manifest IsFace and topology-reference identities differ"
        raise ValueError(msg)


def load_manifest(cfg: Config) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    manifest = json.loads(
        cfg.input_candidates.read_text(encoding="utf-8"),
        parse_constant=legacy.reject_json_constant,
    )
    if not isinstance(manifest, dict):
        msg = "corrected-baseline manifest must contain a JSON object"
        raise TypeError(msg)
    legacy.require_finite_json(manifest)
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        msg = f"unexpected manifest schema: {manifest.get('schema_version')}"
        raise ValueError(msg)
    if manifest.get("complete") is not True:
        msg = "corrected-baseline manifest is incomplete"
        raise ValueError(msg)
    if manifest.get("design") != MANIFEST_DESIGN:
        msg = f"unexpected experiment design: {manifest.get('design')!r}"
        raise ValueError(msg)
    if cfg.input_mesh.resolve() != PREPARED_MESH.resolve():
        msg = f"input mesh must be pinned to {PREPARED_MESH}"
        raise ValueError(msg)
    legacy.verify_file_identity(
        cfg.input_mesh,
        manifest["input_mesh_identity"],
        "prepared input mesh",
    )
    _require_manifest_domain(manifest)

    fixed = manifest.get("fixed_design")
    if not isinstance(fixed, dict):
        msg = "manifest is missing fixed_design"
        raise TypeError(msg)
    exact_fixed = {
        "candidate_labels": [CANDIDATE_LABEL],
        "skin_domain": "all-vertex IsFace physically filtered PolyData",
        "skin_triangles": EXPECTED_SKIN_TRIANGLES,
        "skin_components": EXPECTED_SKIN_COMPONENTS,
        "skin_E_MPa": 0.2,
        "skin_nu": 0.49,
        "skin_prestrain": "p000",
        "skin_lame_conversion": LAME_CONVERSION,
        "volume_lame_conversion": VOLUME_LAME_CONVERSION,
        "inverse_activation_initialization": "fresh exact zero",
        "inverse_optimizer": "Adam",
        "inverse_lr": 0.3,
        "inverse_optimizer_steps": 40,
        "inverse_evaluations": 41,
    }
    for key, expected in exact_fixed.items():
        if fixed.get(key) != expected:
            msg = f"fixed design {key} changed: {fixed.get(key)!r} != {expected!r}"
            raise ValueError(msg)
    if not math.isclose(
        float(fixed.get("skin_area_m2", math.nan)),
        EXPECTED_SKIN_AREA_M2,
        rel_tol=0.0,
        abs_tol=AREA_ATOL_M2,
    ):
        msg = "fixed skin area changed"
        raise ValueError(msg)

    constitutive = manifest.get("constitutive_contract")
    if not isinstance(constitutive, dict):
        msg = "manifest is missing constitutive_contract"
        raise TypeError(msg)
    if constitutive.get("skin") != LAME_CONVERSION:
        msg = "manifest skin conversion is not plane stress"
        raise ValueError(msg)
    if constitutive.get("volume") != VOLUME_LAME_CONVERSION:
        msg = "manifest volume conversion is not the unchanged 3D convention"
        raise ValueError(msg)
    if constitutive.get("heterogeneous_material_fields") is not False:
        msg = "corrected baseline must use homogeneous skin material"
        raise ValueError(msg)

    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        msg = "manifest must contain exactly one corrected baseline"
        raise ValueError(msg)
    row = candidates[0]
    if not isinstance(row, dict) or row.get("label") != CANDIDATE_LABEL:
        msg = "manifest corrected-baseline label changed"
        raise ValueError(msg)
    if (float(row["young_min_scale"]), float(row["prestrain_gain"])) != (
        1.0,
        0.0,
    ):
        msg = "corrected baseline is not homogeneous e100-p000"
        raise ValueError(msg)
    if row.get("validation/ok") is not True or row.get("validation/errors") != []:
        msg = "corrected-baseline preparation validation failed"
        raise ValueError(msg)
    if row.get("readback/ok") is not True or row.get("readback/errors") != []:
        msg = "corrected-baseline VTP readback validation failed"
        raise ValueError(msg)
    if int(row.get("content/n_triangles", -1)) != EXPECTED_SKIN_TRIANGLES:
        msg = "manifest candidate is not the 29,899-triangle filtered PolyData"
        raise ValueError(msg)
    if int(row.get("content/n_points", -1)) != EXPECTED_SKIN_POINTS:
        msg = "manifest candidate point count changed"
        raise ValueError(msg)
    if int(row.get("topology/components", -1)) != EXPECTED_SKIN_COMPONENTS:
        msg = "manifest candidate is not one connected component"
        raise ValueError(msg)
    if row.get("skin/lame_conversion") != LAME_CONVERSION:
        msg = "candidate does not declare plane-stress skin Lamé parameters"
        raise ValueError(msg)
    if row.get("skin/domain") != "all-vertex IsFace filtered PolyData":
        msg = "candidate is not the physically filtered IsFace PolyData"
        raise ValueError(msg)
    return manifest


_verified_skin = legacy.verified_skin


def verified_skin(  # noqa: C901, PLR0912
    cfg: Config, candidate: dict[str, Any]
) -> tuple[Path, pv.PolyData, dict[str, Any]]:
    path, skin, provenance = _verified_skin(cfg, candidate)
    triangles = _triangles(skin)
    errors: list[str] = []
    if skin.n_points != EXPECTED_SKIN_POINTS:
        errors.append(f"live Koiter point count {skin.n_points} != 15299")
    if skin.n_cells != EXPECTED_SKIN_TRIANGLES:
        errors.append(f"live Koiter triangle count {skin.n_cells} != 29899")
    if skin.n_cells >= EXPECTED_FULL_BOUNDARY_TRIANGLES:
        errors.append("live Koiter input was not physically filtered")
    area = _triangle_area(np.asarray(skin.points, dtype=np.float64), triangles)
    total_area = float(math.fsum(float(value) for value in area))
    if not math.isclose(
        total_area, EXPECTED_SKIN_AREA_M2, rel_tol=0.0, abs_tol=AREA_ATOL_M2
    ):
        errors.append("live Koiter area differs from the audited IsFace area")
    components, nonmanifold = _component_count(triangles)
    if components != EXPECTED_SKIN_COMPONENTS:
        errors.append("live Koiter input is not one connected component")
    if nonmanifold:
        errors.append(f"live Koiter input has {nonmanifold} nonmanifold edges")

    E = np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64)
    nu = np.asarray(skin.cell_data["SkinPoissonRatio"], dtype=np.float64)
    lambda_ = np.asarray(skin.cell_data[LAMBDA.vtk], dtype=np.float64)
    mu = np.asarray(skin.cell_data[MU.vtk], dtype=np.float64)
    fraction = np.asarray(skin.cell_data[FRACTION.vtk], dtype=np.float64)
    activation = np.asarray(skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    expected_E = np.full(skin.n_cells, 0.2, dtype=np.float64)
    expected_nu = np.full(skin.n_cells, 0.49, dtype=np.float64)
    expected_lambda = expected_E * expected_nu / (1.0 - np.square(expected_nu))
    expected_mu = expected_E / (2.0 * (1.0 + expected_nu))
    formulas = {
        "E": (E, expected_E),
        "nu": (nu, expected_nu),
        "Lambda": (lambda_, expected_lambda),
        "Mu": (mu, expected_mu),
        "Fraction": (fraction, np.ones(skin.n_cells, dtype=np.float64)),
    }
    for name, (actual, expected) in formulas.items():
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=FORMULA_RTOL, atol=FORMULA_ATOL
        ):
            errors.append(f"live {name} differs from homogeneous plane-stress p000")
    if activation.shape != (skin.n_cells, 3) or not np.array_equal(
        activation, np.zeros_like(activation)
    ):
        errors.append("live skin ActivationInv is not exact p000")
    for name in ("IsFaceTriangle", "SourceOuterTriangle"):
        values = np.asarray(skin.cell_data[name], dtype=np.int8)
        if values.shape != (skin.n_cells,) or not np.all(values == 1):
            errors.append(f"live {name} is not one on every Koiter triangle")
    for name in (
        "ArtificialCutTriangle",
        "FixedTriangle",
        "DisallowedGroupTriangle",
    ):
        values = np.asarray(skin.cell_data[name], dtype=np.int8)
        if values.shape != (skin.n_cells,) or np.any(values != 0):
            errors.append(f"live {name} overlaps the Koiter input")
    group_names = tuple(
        raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        for raw in np.asarray(skin.field_data["GroupName"]).reshape(-1)
    )
    group_ids = np.asarray(skin.point_data["GroupId"], dtype=np.int64)
    observed_group_names = {group_names[index] for index in np.unique(group_ids)}
    if not observed_group_names <= set(FACE_GROUPS):
        errors.append("live Koiter input contains a non-FACE_GROUPS anatomy label")
    if errors:
        msg = "live corrected-baseline skin gates failed: " + "; ".join(errors)
        raise RuntimeError(msg)

    provenance.update(
        {
            "provenance/skin_domain": "all-vertex IsFace filtered PolyData",
            "provenance/koiter_input_points": int(skin.n_points),
            "provenance/koiter_input_triangles": int(skin.n_cells),
            "provenance/koiter_input_area_m2": total_area,
            "provenance/koiter_input_components": components,
            "provenance/skin_lame_conversion": "plane-stress",
            "provenance/skin_lambda_formula_max_abs_error": float(
                np.max(np.abs(lambda_ - expected_lambda))
            ),
            "provenance/skin_mu_formula_max_abs_error": float(
                np.max(np.abs(mu - expected_mu))
            ),
            "provenance/skin_activation_inv_max_abs": float(np.max(np.abs(activation))),
            "provenance/skin_group_names": sorted(observed_group_names),
            "provenance/teeth_proximity_triangles": int(
                np.count_nonzero(skin.cell_data["TeethProximityTriangle"])
            ),
            "provenance/gingiva_proximity_triangles": int(
                np.count_nonzero(skin.cell_data["GingivaProximityTriangle"])
            ),
        }
    )
    return path, skin, provenance


_build_candidate_forward = legacy.build_candidate_forward


def build_corrected_forward(  # noqa: C901
    mesh: pv.UnstructuredGrid,
    case: Any,
    *,
    area_ratio_floor: float,
    skin_path: Path,
    skin: pv.PolyData,
    candidate: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[Any, pv.PolyData, dict[str, Any]]:
    if candidate.get("label") != CANDIDATE_LABEL:
        msg = f"refusing unexpected corrected-baseline candidate {candidate.get('label')!r}"
        raise ValueError(msg)
    if skin.n_cells != EXPECTED_SKIN_TRIANGLES:
        msg = (
            "refusing Koiter construction: input must be the physically filtered "
            f"29,899-triangle IsFace PolyData, got {skin.n_cells}"
        )
        raise RuntimeError(msg)
    if skin.n_cells >= EXPECTED_FULL_BOUNDARY_TRIANGLES:
        msg = "refusing Koiter construction on the complete extracted boundary"
        raise RuntimeError(msg)
    if not np.all(np.asarray(skin.cell_data[FRACTION.vtk]) == 1.0):
        msg = "filtered Koiter input must use Fraction=1 on every retained triangle"
        raise RuntimeError(msg)

    cut_mesh_ids, cut_boundary = _configure_hard_fixed_cut_boundary(mesh)
    if cut_mesh_ids.size != EXPECTED_CUT_INCIDENT_VERTICES:
        msg = "hard-fixed cut configuration returned an unexpected vertex set"
        raise RuntimeError(msg)
    global_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if (
        global_ids.shape != (skin.n_points,)
        or np.unique(global_ids).size != skin.n_points
        or global_ids.min() < 0
        or global_ids.max() >= mesh.n_points
    ):
        msg = "Koiter GlobalPointId values do not index the prepared volume"
        raise ValueError(msg)
    if not np.array_equal(
        np.asarray(skin.points, dtype=np.float64),
        np.asarray(mesh.points, dtype=np.float64)[global_ids],
    ):
        msg = "Koiter points do not exactly match prepared-volume GlobalPointId"
        raise ValueError(msg)

    forward, koiter_skin, metrics = _build_candidate_forward(
        mesh,
        case,
        area_ratio_floor=area_ratio_floor,
        skin_path=skin_path,
        skin=skin,
        candidate=candidate,
        provenance=provenance,
    )
    if koiter_skin.n_cells != EXPECTED_SKIN_TRIANGLES:
        msg = "forward builder changed the Koiter triangle count"
        raise RuntimeError(msg)
    if int(forward.model.n_fixed) != EXPECTED_MODEL_FIXED_DOFS:
        msg = (
            f"forward model fixed DoFs changed: {int(forward.model.n_fixed)} != "
            f"{EXPECTED_MODEL_FIXED_DOFS}"
        )
        raise RuntimeError(msg)
    initial_displacement = forward.state.u.detach()
    if tuple(initial_displacement.shape) != (mesh.n_points, 3):
        msg = (
            "newly built forward state has an unexpected displacement shape: "
            f"{tuple(initial_displacement.shape)}"
        )
        raise RuntimeError(msg)
    if int(torch.count_nonzero(initial_displacement).item()) != 0:
        msg = "newly built corrected forward state is not exact-zero displacement"
        raise RuntimeError(msg)
    metrics.update(
        {
            "skin/domain": "all-vertex IsFace filtered PolyData",
            "skin/koiter_input_triangles": int(koiter_skin.n_cells),
            "skin/koiter_input_points": int(koiter_skin.n_points),
            "skin/E_MPa": 0.2,
            "skin/nu": 0.49,
            "skin/prestrain": "p000",
            "skin/lame_conversion": LAME_CONVERSION,
            "skin/koiter_energy_measure": "fixed original reference area",
            "volume/lame_conversion": VOLUME_LAME_CONVERSION,
            "protocol/forward_initial_displacement_exact_zero": True,
            "protocol/forward_initial_displacement_max_abs_m": 0.0,
            **cut_boundary,
        }
    )
    return forward, koiter_skin, metrics


_inverse_case = legacy.InverseCase
_validate_case = legacy.validate_case


def _cut_boundary_readback(  # noqa: C901, PLR0912, PLR0915
    result: pv.UnstructuredGrid,
) -> dict[str, Any]:
    required = {
        GLOBAL_POINT_ID.vtk,
        "IsFixed",
        "HistoricalIsFixed",
        "ArtificialCutIncident",
        "CutBoundaryPreexistingFixed",
        "CutBoundaryAddedFixed",
        FIXED_MASK.vtk,
        FIXED_VALUE.vtk,
        "Displacement",
    }
    missing = sorted(required - set(result.point_data))
    if missing:
        msg = f"result is missing hard-fixed cut readback fields: {missing}"
        raise KeyError(msg)
    global_ids = np.asarray(result.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    is_fixed = np.asarray(result.point_data["IsFixed"], dtype=bool)
    historical_is_fixed = np.asarray(result.point_data["HistoricalIsFixed"], dtype=bool)
    incident = np.asarray(result.point_data["ArtificialCutIncident"], dtype=bool)
    preexisting = np.asarray(
        result.point_data["CutBoundaryPreexistingFixed"], dtype=bool
    )
    added = np.asarray(result.point_data["CutBoundaryAddedFixed"], dtype=bool)
    fixed_mask = np.asarray(result.point_data[FIXED_MASK.vtk], dtype=bool)
    fixed_value = np.asarray(result.point_data[FIXED_VALUE.vtk], dtype=np.float64)
    displacement = np.asarray(result.point_data["Displacement"], dtype=np.float64)
    scalar_fields = (
        global_ids,
        is_fixed,
        historical_is_fixed,
        incident,
        preexisting,
        added,
    )
    if any(values.shape != (result.n_points,) for values in scalar_fields):
        msg = "result hard-fixed cut scalar fields are malformed"
        raise ValueError(msg)
    if fixed_mask.shape != (result.n_points, 3) or fixed_value.shape != (
        result.n_points,
        3,
    ):
        msg = "result hard-fixed FixedMask/FixedValue fields are malformed"
        raise ValueError(msg)
    if displacement.shape != (result.n_points, 3):
        msg = "result Displacement field is malformed"
        raise ValueError(msg)
    cut_ids = np.flatnonzero(incident)
    if cut_ids.size != EXPECTED_CUT_INCIDENT_VERTICES:
        msg = (
            f"result artificial-cut vertex count changed: {cut_ids.size} != "
            f"{EXPECTED_CUT_INCIDENT_VERTICES}"
        )
        raise ValueError(msg)
    if int(preexisting.sum()) != EXPECTED_CUT_PREEXISTING_FIXED_VERTICES:
        msg = "result preexisting artificial-cut fixed marker count changed"
        raise ValueError(msg)
    if int(added.sum()) != EXPECTED_CUT_NEWLY_FIXED_VERTICES:
        msg = "result newly fixed artificial-cut marker count changed"
        raise ValueError(msg)
    if np.any(preexisting & added) or not np.array_equal(incident, preexisting | added):
        msg = "result artificial-cut marker partition is inconsistent"
        raise ValueError(msg)
    if not np.array_equal(preexisting, incident & historical_is_fixed):
        msg = "result preexisting marker differs from historical IsFixed"
        raise ValueError(msg)
    if not np.array_equal(added, incident & ~historical_is_fixed):
        msg = "result added marker differs from historical IsFixed complement"
        raise ValueError(msg)
    if not np.array_equal(fixed_mask, np.repeat(is_fixed[:, None], 3, axis=1)):
        msg = "result FixedMask is inconsistent with IsFixed"
        raise ValueError(msg)
    if not np.all(is_fixed[cut_ids]):
        msg = "result does not retain all artificial-cut vertices as fixed"
        raise ValueError(msg)
    if int(is_fixed.sum()) != EXPECTED_MODEL_FIXED_VERTICES:
        msg = "result model fixed vertex count changed"
        raise ValueError(msg)
    if int(fixed_mask.sum()) != EXPECTED_MODEL_FIXED_DOFS:
        msg = "result model fixed DoF count changed"
        raise ValueError(msg)
    if not np.array_equal(fixed_value[is_fixed], np.zeros_like(fixed_value[is_fixed])):
        msg = "result fixed values are not exact zero"
        raise ValueError(msg)
    cut_displacement = displacement[cut_ids]
    if not np.array_equal(cut_displacement, np.zeros_like(cut_displacement)):
        msg = "result artificial-cut displacement is not exact zero"
        raise ValueError(msg)
    cut_global_ids = np.sort(global_ids[cut_ids]).astype("<i8", copy=False)
    cut_global_ids_sha256 = hashlib.sha256(
        np.ascontiguousarray(cut_global_ids).tobytes()
    ).hexdigest()
    if cut_global_ids_sha256 != EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256:
        msg = f"result artificial-cut GlobalPointId digest changed: {cut_global_ids_sha256}"
        raise ValueError(msg)
    return {
        "cut_boundary/readback_incident_vertices": int(cut_ids.size),
        "cut_boundary/readback_incident_global_ids_sha256": cut_global_ids_sha256,
        "cut_boundary/readback_total_fixed_vertices": int(is_fixed[cut_ids].sum()),
        "cut_boundary/readback_model_total_fixed_vertices": int(is_fixed.sum()),
        "cut_boundary/readback_model_total_fixed_dofs": int(fixed_mask.sum()),
        "cut_boundary/readback_fixed_values_max_abs_m": float(
            np.abs(fixed_value[is_fixed]).max()
        ),
        "cut_boundary/readback_displacement_rms_m": float(
            np.linalg.norm(cut_displacement) / math.sqrt(cut_ids.size)
        ),
        "cut_boundary/readback_displacement_max_abs_m": float(
            np.abs(cut_displacement).max()
        ),
        "cut_boundary/readback_exact_zero": True,
    }


def corrected_inverse_case(*args: Any, **kwargs: Any) -> Any:
    label = str(kwargs.get("label", ""))
    expected_prefix = "exaggerated-"
    if not label.startswith(expected_prefix):
        msg = f"unexpected reference case label: {label!r}"
        raise ValueError(msg)
    kwargs["label"] = "corrected-" + label.removeprefix(expected_prefix)
    return _inverse_case(*args, **kwargs)


def validate_case(
    summary: dict[str, Any], paths: Any, skin: pv.PolyData, cfg: Config
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors, warnings, diagnostics = _validate_case(summary, paths, skin, cfg)
    exact = {
        "skin/domain": "all-vertex IsFace filtered PolyData",
        "skin/koiter_input_triangles": EXPECTED_SKIN_TRIANGLES,
        "skin/koiter_input_points": EXPECTED_SKIN_POINTS,
        "skin/E_MPa": 0.2,
        "skin/nu": 0.49,
        "skin/prestrain": "p000",
        "skin/lame_conversion": LAME_CONVERSION,
        "skin/koiter_energy_measure": "fixed original reference area",
        "volume/lame_conversion": VOLUME_LAME_CONVERSION,
        "cut_boundary/policy": HARD_FIXED_CUT_BOUNDARY_POLICY,
        "cut_boundary/triangles": EXPECTED_ARTIFICIAL_CUT_TRIANGLES,
        "cut_boundary/triangle_topology_sha256": (
            EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256
        ),
        "cut_boundary/incident_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
        "cut_boundary/incident_global_ids_sha256": (
            EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256
        ),
        "cut_boundary/preexisting_fixed_vertices": (
            EXPECTED_CUT_PREEXISTING_FIXED_VERTICES
        ),
        "cut_boundary/newly_fixed_vertices": EXPECTED_CUT_NEWLY_FIXED_VERTICES,
        "cut_boundary/total_fixed_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
        "cut_boundary/model_total_fixed_vertices": EXPECTED_MODEL_FIXED_VERTICES,
        "cut_boundary/model_total_fixed_dofs": EXPECTED_MODEL_FIXED_DOFS,
        "cut_boundary/configured_exact_zero": True,
        "cut_boundary/hard_fixed_is_ground_truth": False,
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            errors.append(f"{key} differs from the corrected-baseline contract")
    if float(summary.get("skin/thickness", math.nan)) != float(legacy.SKIN_THICKNESS):
        errors.append("skin thickness differs from the fixed 1 mm convention")
    expected_evaluations = cfg.inverse_max_steps + 1
    if int(summary.get("inverse/evaluations", -1)) != expected_evaluations:
        errors.append(
            "corrected baseline evaluation count differs from the stage contract"
        )
    if paths.result.is_file():
        result = pv.read(paths.result)
        if not isinstance(result, pv.UnstructuredGrid):
            errors.append("corrected-baseline result did not read as UnstructuredGrid")
        else:
            diagnostics.update(_cut_boundary_readback(result))
    return sorted(set(errors)), warnings, diagnostics


def _expected_case_paths(cfg: Config) -> Any:
    case = _inverse_case(
        target="smile",
        lr=cfg.inverse_lr,
        setup=legacy.SETUP_SKIN_NO_PRESTRAIN,
        label=f"corrected-{CANDIDATE_LABEL}-{cfg.stage}",
    )
    return legacy.CasePaths.from_case(cfg.output_summary.parent, case)


def _canonical_archive_paths(cfg: Config) -> tuple[Path, Path]:
    """Return unique post-rewrite metadata copies for the Cherries snapshot."""
    case_summary = _expected_case_paths(cfg).summary
    final_case_summary = case_summary.with_name(f"{case_summary.stem}-final.json")
    final_aggregate = cfg.output_summary.with_name(
        f"{cfg.output_summary.stem}-final.json"
    )
    return final_case_summary, final_aggregate


def _refuse_stale_output_targets(cfg: Config) -> None:
    paths = _expected_case_paths(cfg)
    final_case_summary, final_aggregate = _canonical_archive_paths(cfg)
    expected = {
        "aggregate summary": cfg.output_summary,
        "aggregate table": cfg.output_table,
        "live-plot directory": cfg.live_plot_dir,
        "case target": paths.target,
        "case result": paths.result,
        "case summary": paths.summary,
        "case history": paths.history,
        "case history temporary": paths.history.with_name(f"{paths.history.name}.tmp"),
        "case trace": paths.trace,
        "canonical case summary archive": final_case_summary,
        "canonical aggregate archive": final_aggregate,
    }
    stale = [f"{name}: {path}" for name, path in expected.items() if path.exists()]
    if stale:
        msg = (
            "refusing corrected-baseline inverse because expected output targets "
            "already exist; archive or remove them explicitly before approval: "
            + "; ".join(stale)
        )
        raise FileExistsError(msg)


def _artifact_identity_fields(paths: Any) -> dict[str, int | str]:
    fields: dict[str, int | str] = {}
    for name in ("target", "result", "history", "trace"):
        path = Path(getattr(paths, name))
        identity = _file_identity(path)
        fields[f"artifact/{name}_path"] = str(path)
        fields[f"artifact/{name}_size_bytes"] = int(identity["size_bytes"])
        fields[f"artifact/{name}_sha256"] = str(identity["sha256"])
    return fields


def _rewrite_outputs(
    cfg: Config,
    *,
    input_candidates_identity: dict[str, int | str],
    inverse_runtime_identity: dict[str, Any],
    producer_identity: dict[str, int | str],
) -> None:
    if not cfg.output_summary.is_file():
        msg = "successful corrected-baseline run did not write its aggregate summary"
        raise FileNotFoundError(msg)
    aggregate = json.loads(
        cfg.output_summary.read_text(encoding="utf-8"),
        parse_constant=legacy.reject_json_constant,
    )
    legacy.require_finite_json(aggregate)
    cases = aggregate.get("cases")
    if (
        aggregate.get("complete") is not True
        or aggregate.get("candidate_set") != CANDIDATE_LABEL
        or not isinstance(cases, list)
        or len(cases) != 1
        or cases[0].get("candidate") != CANDIDATE_LABEL
        or cases[0].get("status") != "ok"
    ):
        msg = "refusing to relabel an incomplete or non-current inverse aggregate"
        raise RuntimeError(msg)
    expected_summary = _expected_case_paths(cfg).summary.resolve()
    final_case_summary, final_aggregate = _canonical_archive_paths(cfg)
    archive_fields = {
        "archive/canonical_case_summary_path": str(final_case_summary),
        "archive/canonical_aggregate_path": str(final_aggregate),
        "archive/metadata_snapshot_policy": (
            "unique post-rewrite copies avoid the Local plugin same-name overwrite bug"
        ),
    }
    for row in cases:
        paths = _expected_case_paths(cfg)
        artifact_identities = _artifact_identity_fields(paths)
        row.update(
            {
                "design": DESIGN,
                "stage": cfg.stage,
                "material/skin_domain": "all-vertex IsFace filtered PolyData",
                "material/skin_lame_conversion": LAME_CONVERSION,
                "material/skin_koiter_energy_measure": (
                    "fixed original reference area"
                ),
                "material/volume_lame_conversion": VOLUME_LAME_CONVERSION,
                "material/skin_E_MPa": 0.2,
                "material/skin_nu": 0.49,
                "material/skin_prestrain": "p000",
                "protocol/fresh_zero_activation": True,
                "protocol/fresh_zero_displacement": True,
                "protocol/optimizer_steps": cfg.inverse_max_steps,
                "protocol/evaluations": cfg.inverse_max_steps + 1,
                **archive_fields,
                **artifact_identities,
            }
        )
        summary_path = Path(str(row.get("artifact/summary_path", "")))
        if summary_path.resolve() != expected_summary or not summary_path.is_file():
            msg = (
                f"aggregate does not reference the current case summary: {summary_path}"
            )
            raise RuntimeError(msg)
        summary = json.loads(
            summary_path.read_text(encoding="utf-8"),
            parse_constant=legacy.reject_json_constant,
        )
        legacy.require_finite_json(summary)
        if summary.get("candidate") != CANDIDATE_LABEL or summary.get("status") != "ok":
            msg = "refusing to relabel an incomplete or non-current case summary"
            raise RuntimeError(msg)
        summary.update(
            {
                "design": DESIGN,
                "stage": cfg.stage,
                "material/skin_domain": ("all-vertex IsFace filtered PolyData"),
                "material/skin_lame_conversion": LAME_CONVERSION,
                "material/skin_koiter_energy_measure": (
                    "fixed original reference area"
                ),
                "material/volume_lame_conversion": VOLUME_LAME_CONVERSION,
                "material/skin_E_MPa": 0.2,
                "material/skin_nu": 0.49,
                "material/skin_prestrain": "p000",
                "protocol/fresh_zero_activation": True,
                "protocol/fresh_zero_displacement": True,
                "protocol/optimizer_steps": cfg.inverse_max_steps,
                "protocol/evaluations": cfg.inverse_max_steps + 1,
                **archive_fields,
                **artifact_identities,
            }
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    aggregate.update(
        {
            "schema_version": AGGREGATE_SCHEMA_VERSION,
            "design": DESIGN,
            "experiment": (
                "human-face Smile IsFace plane-stress hard-fixed-cut "
                "corrected-baseline inverse"
            ),
            "stage": cfg.stage,
            "candidate_set": CANDIDATE_LABEL,
            "n_candidates": 1,
            "fresh_zero_activation": True,
            "fresh_zero_displacement": True,
            "activation_mode": "per-muscle-tet-6dof-unconstrained",
            "activation_shared": False,
            "activation_transferred": False,
            "inverse_lr": 0.3,
            "inverse_optimizer_steps": cfg.inverse_max_steps,
            "inverse_evaluations": cfg.inverse_max_steps + 1,
            "input_candidates": str(cfg.input_candidates),
            "input_candidates_identity": input_candidates_identity,
            "input_cut_reference": str(cfg.input_cut_reference),
            "input_cut_reference_identity": {
                "size_bytes": SOURCE_SKIN_SIZE_BYTES,
                "sha256": SOURCE_SKIN_SHA256,
            },
            "archive_policy": {
                "canonical_case_summary_path": str(final_case_summary),
                "canonical_aggregate_path": str(final_aggregate),
                "canonical_copies_are_byte_identical_to_live": True,
                "reason": (
                    "unique post-rewrite copies avoid the Local plugin "
                    "same-name overwrite bug"
                ),
            },
            "constitutive_policy": {
                "skin": LAME_CONVERSION,
                "volume": VOLUME_LAME_CONVERSION,
                "skin_E_MPa": 0.2,
                "skin_nu": 0.49,
                "skin_prestrain": "p000",
                "skin_koiter_energy_measure": "fixed original reference area",
            },
            "domain_policy": {
                "skin": "all-vertex IsFace filtered PolyData",
                "koiter_input_triangles": EXPECTED_SKIN_TRIANGLES,
                "koiter_input_area_m2": EXPECTED_SKIN_AREA_M2,
                "components": EXPECTED_SKIN_COMPONENTS,
                "artificial_cut_overlap_triangles": 0,
                "fixed_overlap_triangles": 0,
                "disallowed_group_overlap_triangles": 0,
                "face_group_allowlist": list(FACE_GROUPS),
                "teeth_and_gingiva_proximity": "diagnostic only",
            },
            "boundary_policy": {
                "policy": HARD_FIXED_CUT_BOUNDARY_POLICY,
                "marker": ("source skin triangle touches mapped GroupId=-1 vertex"),
                "reference_path": str(SOURCE_SKIN),
                "reference_size_bytes": SOURCE_SKIN_SIZE_BYTES,
                "reference_sha256": SOURCE_SKIN_SHA256,
                "triangles": EXPECTED_ARTIFICIAL_CUT_TRIANGLES,
                "triangle_topology_sha256": (EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256),
                "incident_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
                "incident_global_ids_sha256": (EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256),
                "preexisting_fixed_vertices": (EXPECTED_CUT_PREEXISTING_FIXED_VERTICES),
                "newly_fixed_vertices": EXPECTED_CUT_NEWLY_FIXED_VERTICES,
                "total_fixed_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
                "model_total_fixed_vertices": EXPECTED_MODEL_FIXED_VERTICES,
                "model_total_fixed_dofs": EXPECTED_MODEL_FIXED_DOFS,
                "fixed_values_max_abs_m": 0.0,
                "configured_exact_zero": True,
                "readback_displacement_max_abs_m": 0.0,
                "readback_exact_zero": True,
                "hard_fixed_is_ground_truth": False,
                "interpretation": "user-approved conservative approximation",
            },
            "implementation": {
                "producer/path": str(Path(__file__).resolve()),
                "producer/size_bytes": int(producer_identity["size_bytes"]),
                "producer/sha256": str(producer_identity["sha256"]),
                "prepare/path": str(PREPARE_IMPLEMENTATION),
                "prepare/sha256": PREPARE_IMPLEMENTATION_SHA256,
                "prepare/identity_verified_stable": (
                    require_file_sha256(
                        PREPARE_IMPLEMENTATION,
                        PREPARE_IMPLEMENTATION_SHA256,
                        name="corrected-baseline prepare implementation after inverse",
                    )
                    == PREPARE_IMPLEMENTATION_SHA256
                ),
                "reference_inverse/path": str(LEGACY_INVERSE),
                "reference_inverse/sha256": LEGACY_IDENTITY,
                "reference_inverse/identity_verified_stable": (
                    require_file_sha256(
                        LEGACY_INVERSE,
                        LEGACY_INVERSE_SHA256,
                        name="reference inverse module after corrected baseline",
                    )
                    == LEGACY_IDENTITY
                ),
                "inverse_runtime_bundle": inverse_runtime_identity,
                "intentional_changes": [
                    "one corrected homogeneous p000 candidate only",
                    "physically filtered 29,899-triangle IsFace Koiter input",
                    "plane-stress skin lambda with unchanged skin mu",
                    "fixed original-reference-area Koiter energy weight",
                    "unchanged 3D volume material conversion",
                    "all 6,980 artificial-cut incident vertices fixed to exact zero",
                ],
                "koiter/path": str(KOITER_IMPLEMENTATION),
                "koiter/sha256": KOITER_IMPLEMENTATION_SHA256,
                "volume_lame/path": str(VOLUME_LAME_IMPLEMENTATION),
                "volume_lame/sha256": VOLUME_LAME_IMPLEMENTATION_SHA256,
                "volume_forward/path": str(VOLUME_FORWARD_IMPLEMENTATION),
                "volume_forward/sha256": VOLUME_FORWARD_IMPLEMENTATION_SHA256,
            },
            "visual_review": {
                "status": "not-applicable-to-smoke"
                if cfg.stage == "smoke"
                else "pending",
                "policy": (
                    "small inverted tetrahedra and folded skin triangles are "
                    "warning-only; acceptance depends on matched visual views"
                ),
            },
            "execution_scope": (
                "single zero-update forward-plus-adjoint integration smoke"
                if cfg.stage == "smoke"
                else "single 40-update scientific corrected baseline"
            ),
        }
    )
    legacy.require_finite_json(aggregate)
    cfg.output_summary.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def run(cfg: Config) -> None:
    if cfg.stage == "screen" and not FORMAL_APPROVED_AFTER_USER_REVIEW:
        raise RuntimeError(FORMAL_APPROVAL_BLOCKER)
    if cfg.stage == "smoke" and not SMOKE_APPROVED_AFTER_ROOT_REVIEW:
        raise RuntimeError(SMOKE_APPROVAL_BLOCKER)

    # Only an explicitly approved stage contract may become reachable. Exact
    # paths and budgets below prevent the formal approval from broadening scope.
    _validate_corrected_paths(cfg)
    producer_identity = _file_identity(Path(__file__).resolve())
    input_candidates_identity = _file_identity(cfg.input_candidates)
    cherries.log_input(cfg.input_candidates)
    inverse_runtime_identity = _require_inverse_runtime_identity(context="pre-solve")
    legacy.MANIFEST_SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION
    legacy.EXPECTED_CANDIDATES = EXPECTED_CANDIDATES
    legacy.EXPECTED_LABELS = EXPECTED_LABELS
    legacy.EXPECTED_PROTOCOL = EXPECTED_PROTOCOL
    legacy.load_manifest = load_manifest
    legacy.verified_skin = verified_skin
    legacy.build_candidate_forward = build_corrected_forward
    legacy.InverseCase = corrected_inverse_case
    legacy.validate_case = validate_case
    for path, expected, name in (
        (
            PREPARE_IMPLEMENTATION,
            PREPARE_IMPLEMENTATION_SHA256,
            "corrected-baseline prepare implementation",
        ),
        (SOURCE_SKIN, SOURCE_SKIN_SHA256, "artificial-cut topology reference"),
        (KOITER_IMPLEMENTATION, KOITER_IMPLEMENTATION_SHA256, "Koiter implementation"),
        (
            VOLUME_LAME_IMPLEMENTATION,
            VOLUME_LAME_IMPLEMENTATION_SHA256,
            "volume Lamé implementation",
        ),
        (
            VOLUME_FORWARD_IMPLEMENTATION,
            VOLUME_FORWARD_IMPLEMENTATION_SHA256,
            "volume forward implementation",
        ),
    ):
        require_file_sha256(path, expected, name=name)
    cherries.log_input(cfg.input_cut_reference)
    for _, path, _ in INVERSE_RUNTIME_DEPENDENCIES:
        cherries.log_input(path)
    _refuse_stale_output_targets(cfg)
    legacy.run(cfg)
    post_runtime_identity = _require_inverse_runtime_identity(context="post-solve")
    if post_runtime_identity != inverse_runtime_identity:
        msg = "inverse runtime dependency identities changed during the run"
        raise RuntimeError(msg)
    if _file_identity(cfg.input_candidates) != input_candidates_identity:
        msg = "corrected-baseline candidate manifest changed during inverse"
        raise RuntimeError(msg)
    if _file_identity(Path(__file__).resolve()) != producer_identity:
        msg = "corrected-baseline producer implementation changed during inverse"
        raise RuntimeError(msg)
    _rewrite_outputs(
        cfg,
        input_candidates_identity=input_candidates_identity,
        inverse_runtime_identity=inverse_runtime_identity,
        producer_identity=producer_identity,
    )
    final_case_summary, final_aggregate = _canonical_archive_paths(cfg)
    case_summary = _expected_case_paths(cfg).summary
    final_case_summary.write_bytes(case_summary.read_bytes())
    final_aggregate.write_bytes(cfg.output_summary.read_bytes())
    if (
        final_case_summary.read_bytes() != case_summary.read_bytes()
        or final_aggregate.read_bytes() != cfg.output_summary.read_bytes()
    ):
        msg = "canonical Cherries metadata copies differ from live final summaries"
        raise RuntimeError(msg)
    # The legacy runner already logged its raw summaries. Unique post-rewrite
    # names avoid the Local plugin's broken same-name overwrite path when its
    # optional log file is absent.
    for path in (final_case_summary, final_aggregate):
        cherries.log_output(path)
    logger.info(
        "Wrote corrected-baseline %s outputs under %s",
        cfg.stage,
        cfg.output_summary.parent,
    )


if __name__ == "__main__":
    cherries.main(run)
