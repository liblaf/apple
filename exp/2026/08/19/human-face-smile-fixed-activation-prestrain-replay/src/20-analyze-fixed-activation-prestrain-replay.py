from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import pyvista as pv

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

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
INPUT_SCHEMA_VERSION = 1
INPUT_DESIGN = "corrected-isface-fixed-activation-c020-prestrain-replay"
INPUT_SUMMARY_NAME = "10-fixed-activation-prestrain-replay-summary.json"
INPUT_RESULT_DIR = "10-fixed-activation-prestrain-replay"
INPUT_PRODUCER = Path(__file__).with_name("10-fixed-activation-prestrain-replay.py")
INPUT_PRODUCER_SHA256 = (
    "231668ac7963bd7eff14705a94125ad8396d8886e0d34a152a4f51253f32c7f6"
)

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
PRODUCER_SUMMARY = GROUP_DIR / f"data/{INPUT_SUMMARY_NAME}"
PRODUCER_TABLE = GROUP_DIR / "data/10-fixed-activation-prestrain-replay-table.md"
PRODUCER_OUTPUT_ROOT = GROUP_DIR / f"data/{INPUT_RESULT_DIR}"
BASELINE_GROUP = REPO_ROOT / "exp/2026/08/18/human-face-smile-plane-stress-skin"
BASELINE_STEM = (
    "20-human-face-smile-skin-no-prestrain-lr3-corrected-isface-e0200-p000-screen"
)
BASELINE_RESULT = BASELINE_GROUP / f"data/{BASELINE_STEM}.vtu"
BASELINE_SUMMARY = BASELINE_GROUP / f"data/{BASELINE_STEM}-summary-final.json"
BASELINE_SKIN = BASELINE_GROUP / "data/10-corrected-baseline/skin-isface-e0200-p000.vtp"
METRIC_SKIN = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/"
    "data/10-material-candidates/skin-e100-p000.vtp"
)
PREPARED_MESH = (
    REPO_ROOT
    / "exp/2026/06/17/human-face-smile-prestrain-v2/data/10-human-face-prepared.vtu"
)
BASELINE_TARGET = BASELINE_GROUP / f"data/{BASELINE_STEM}-target.vtu"
REVIEWED_PROBE = BASELINE_GROUP / "src/15-forward-domain-conversion-probe.py"
REVIEWED_REFERENCE = BASELINE_GROUP / "src/_reference.py"
RUNTIME_METRICS = (
    REPO_ROOT
    / "exp/2026/06/17/human-face-smile-prestrain-v2/src/_human_face_metrics.py"
)
RUNTIME_COMPAT_CONFIG = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/src/"
    "_human_face_config.py"
)

OUTPUT_JSON = GROUP_DIR / "data/20-fixed-activation-prestrain-replay-analysis.json"
OUTPUT_CSV = GROUP_DIR / "data/20-fixed-activation-prestrain-replay-trajectories.csv"
OUTPUT_TRAJECTORY_PLOT = (
    GROUP_DIR / "data/20-fixed-activation-prestrain-alpha-trajectories.png"
)
OUTPUT_QUALITY_PLOT = (
    GROUP_DIR / "data/20-fixed-activation-prestrain-quality-trajectories.png"
)
OUTPUT_GEOMETRY_VIEWS = (
    GROUP_DIR / "data/20-fixed-activation-prestrain-terminal-geometry.png"
)
OUTPUT_RESIDUAL_VIEWS = (
    GROUP_DIR / "data/20-fixed-activation-prestrain-terminal-normal-residual.png"
)
OUTPUT_REPORT = GROUP_DIR / "data/20-fixed-activation-prestrain-replay-analysis.md"

# Static-review blocker. It cannot be bypassed with a CLI flag. A later, isolated
# approval edit may change only this constant after both producer and analyzer pins
# are final; this implementation task must leave it false and must not run.
ANALYZER_EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True
ANALYZER_APPROVAL_BLOCKER = (
    "NO-GO: fixed-activation replay analyzer awaits final producer/analyzer audit; "
    "do not execute until this source-level blocker is explicitly changed"
)

BASELINE_RESULT_SIZE_BYTES = 147_657_021
BASELINE_RESULT_SHA256 = (
    "c6a0b183675ffb3ec537c1153544b041acd7aa0fdd5216c0cf9a50022d52b0a4"
)
BASELINE_SUMMARY_SIZE_BYTES = 126_540
BASELINE_SUMMARY_SHA256 = (
    "575ebcbd7152a256917c2a11a9bf9bef9046f00f9831e18adc86d41645be1856"
)
BASELINE_SKIN_SIZE_BYTES = 1_138_550
BASELINE_SKIN_SHA256 = (
    "4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f"
)
METRIC_SKIN_SIZE_BYTES = 38_742_137
METRIC_SKIN_SHA256 = "ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f"
EXPECTED_INPUT_IDENTITIES = {
    "prepared_mesh": {
        "path": PREPARED_MESH,
        "size_bytes": 76_792_914,
        "sha256": "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563",
    },
    "corrected_skin": {
        "path": BASELINE_SKIN,
        "size_bytes": BASELINE_SKIN_SIZE_BYTES,
        "sha256": BASELINE_SKIN_SHA256,
    },
    "driver_skin": {
        "path": METRIC_SKIN,
        "size_bytes": METRIC_SKIN_SIZE_BYTES,
        "sha256": METRIC_SKIN_SHA256,
    },
    "baseline_result": {
        "path": BASELINE_RESULT,
        "size_bytes": BASELINE_RESULT_SIZE_BYTES,
        "sha256": BASELINE_RESULT_SHA256,
    },
    "baseline_summary": {
        "path": BASELINE_SUMMARY,
        "size_bytes": BASELINE_SUMMARY_SIZE_BYTES,
        "sha256": BASELINE_SUMMARY_SHA256,
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
EXPECTED_RUNTIME_DEPENDENCIES = (
    {
        "name": "Koiter",
        "path": REPO_ROOT / "src/liblaf/apple/warp/fem/_koiter.py",
        "size_bytes": 17_329,
        "sha256": "f7b7c9547c82976a130a88faf8df5172312309238c2b0cf8c8e762e1ec463e8c",
    },
    {
        "name": "volume 3D Lame implementation",
        "path": (
            REPO_ROOT
            / "exp/2026/06/17/human-face-smile-prestrain-v2/src/_human_face_mesh.py"
        ),
        "size_bytes": 7_816,
        "sha256": "f1e1cdc806273c4ce5a37e52e3032d357b44bfd201de3fc58c35d793d11454bc",
    },
    {
        "name": "volume forward builder",
        "path": (
            REPO_ROOT
            / "exp/2026/06/17/human-face-smile-prestrain-v2/src/_human_face_forward.py"
        ),
        "size_bytes": 8_205,
        "sha256": "2d0ff39b13555300c000e6dd43e16c274752263b703746ad8174072033819e03",
    },
    {
        "name": "target",
        "path": (
            REPO_ROOT
            / "exp/2026/06/17/human-face-smile-prestrain-v2/src/_human_face_targets.py"
        ),
        "size_bytes": 1_863,
        "sha256": "34a1583fcb8f90f357647dd4574e2e7ef27f8049f2b3ba1e2fa7dc838fcbb696",
    },
    {
        "name": "output",
        "path": (
            REPO_ROOT
            / "exp/2026/06/17/human-face-smile-prestrain-v2/src/_human_face_output.py"
        ),
        "size_bytes": 8_395,
        "sha256": "29bae977a4b31e82276aca15fdaae3bdda37e6a3e71493876b6fd973db1a1c61",
    },
    {
        "name": "08/17 compatibility config imported as _human_face_config",
        "path": RUNTIME_COMPAT_CONFIG,
        "size_bytes": 2_992,
        "sha256": "fcd7757486c3f0664816a6595e17af27a87ffec1c9c9e24b18908506b444ffeb",
    },
    {
        "name": "core moduli",
        "path": REPO_ROOT / "src/liblaf/apple/common/_moduli.py",
        "size_bytes": 1_210,
        "sha256": "9d5c14f27b9a08a8a4f9cd3ce4e3076f2375ed1108e84e94d307c9439e1a303d",
    },
)
EXPECTED_CORRECTED_TRIANGLE_KEYS_SHA256 = (
    "dca8d77662f49b54250657424cfc29a3b438437081f83d65f7e108f312da2310"
)
EXPECTED_MAPPED_DRIVER_CELLS_SHA256 = (
    "13458107ceef23ecc144340101574f3fb4f2e157f90a9251434e7b09a86a66c3"
)
EXPECTED_RAW_RATIO_SHA256 = (
    "da98a6f48694eed30b1683ccf5a1f02fd67c87393b30003d07c52a7fa25bc606"
)
EXPECTED_RHO_FULL_SHA256 = (
    "d631449a1db997e6ce2eac9d3276ff8a23451461350f917026f19e9d50cc89f1"
)

CASE_ID = "c020"
EXPECTED_DOSE = 0.02
EXPECTED_FLOOR = 0.5
CONTINUATION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_PATHS = (
    *(("continuation", alpha) for alpha in CONTINUATION_ALPHAS),
    ("direct", 1.0),
)
EXPECTED_CASE_IDS = (
    "c020-continuation-alpha-000",
    "c020-continuation-alpha-025",
    "c020-continuation-alpha-050",
    "c020-continuation-alpha-075",
    "c020-continuation-alpha-100",
    "c020-direct-alpha-100",
)
EXPECTED_LOSS_TARGET_RMS_M = 0.005310139062299789
EXPECTED_ISFACE_TARGET_RMS_M = 0.005310654682438851
EXPECTED_BASELINE_TARGET_ERROR_RMS_M = 0.0027209482247538275
EXPECTED_BASELINE_DIHEDRAL_RMS_DEG = 13.328980970609415
EXPECTED_BASELINE_NORMAL_LAPLACIAN_RMS_M = 0.0002170610984016332
EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_ACTIVE_TETS = 288_235
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_CUT_VERTICES = 6_980
EXPECTED_MODEL_FIXED_VERTICES = 33_636
EXPECTED_MODEL_FIXED_DOFS = 100_908
EXPECTED_FIXED_ACTIVATION_SHA256 = (
    "4494f1eca2ce6f14c2e87a184d2227c080fbfa4594e7d6e96ced0c0c35c981de"
)
SKIN_LAME_CONVERSION = (
    "thin-membrane plane-stress reduction: "
    "lambda = E * nu / (1 - nu**2); "
    "mu = E / (2 * (1 + nu))"
)
VOLUME_LAME_CONVERSION = "3d: lambda=E*nu/((1+nu)*(1-2*nu)); mu=E/(2*(1+nu))"
EXPECTED_SKIN_E_MPA = 0.2
EXPECTED_SKIN_NU = 0.49
EXPECTED_SKIN_THICKNESS_M = 0.001
EXPECTED_VOLUME_MATERIALS = {
    "fat": {"E_MPa": 0.003, "nu": 0.49},
    "muscle": {"E_MPa": 0.03, "nu": 0.49},
    "aponeurosis": {"E_MPa": 0.1, "nu": 0.35},
}
BOTH_BUMPINESS_IMPROVEMENT_FRACTION = 0.05
SINGLE_BUMPINESS_IMPROVEMENT_FRACTION = 0.10
PAIRED_BUMPINESS_MAX_REGRESSION_FRACTION = 0.01
ALPHA0_ROUGHNESS_RELATIVE_TOLERANCE = 0.01
MAX_TARGET_ERROR_RELATIVE_INCREASE = 0.05
BRANCH_DELTA_FRACTION_OF_TARGET_TOLERANCE = 1.0e-3
ALPHA0_DELTA_FRACTION_OF_TARGET_TOLERANCE = 1.0e-3

PRIMARY_BUMPINESS = (
    "bumpiness/contraction_target_relative_dihedral_rms_deg",
    "bumpiness/residual_normal_laplacian_rms_m",
)
FIDELITY_METRICS = (
    "target/error_rms_fraction_of_target",
    "target/face_target_area_weighted_error_rms_m",
)
REPORT_METRICS = (
    "target/error_rms_m",
    "target/error_rms_mm",
    "target/error_rms_fraction_of_target",
    "target/face_rest_area_weighted_error_rms_m",
    "target/face_target_area_weighted_error_rms_m",
    "bumpiness/contraction_target_relative_dihedral_rms_rad",
    "bumpiness/contraction_target_relative_dihedral_rms_deg",
    "bumpiness/residual_normal_laplacian_rms_m",
    "bumpiness/displacement_laplacian_rms_m",
    "bumpiness/residual_laplacian_rms_m",
    "quality/inverted_tets",
    "quality/inverted_tet_fraction",
    "quality/detF_min",
    "quality/detF_q001",
    "quality/skin_folded_triangles",
    "quality/skin_folded_triangle_fraction",
    "quality/skin_folded_rest_area_fraction",
    "quality/skin_signed_normal_ratio_q001",
    "quality/skin_signed_normal_ratio_q999",
    "quality/skin_area_ratio_q001",
    "quality/skin_area_ratio_q999",
    "fixed/displacement_exact_zero",
    "fixed/displacement_max_abs_m",
)
PINNED_FIXED_POINT_ARRAYS = (
    "IsFixed",
    "HistoricalIsFixed",
    "ArtificialCutIncident",
    "CutBoundaryPreexistingFixed",
    "CutBoundaryAddedFixed",
    FIXED_MASK.vtk,
    FIXED_VALUE.vtk,
)


class Config(cherries.BaseConfig):
    """Readback-only analysis; it never invokes a physics or inverse solver."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_summary: Path = cherries.input(PRODUCER_SUMMARY)
    input_baseline_result: Path = cherries.input(BASELINE_RESULT)
    input_baseline_summary: Path = cherries.input(BASELINE_SUMMARY)
    input_baseline_skin: Path = cherries.input(BASELINE_SKIN)
    input_metric_skin: Path = cherries.input(METRIC_SKIN)
    output_json: Path = cherries.output(
        "20-fixed-activation-prestrain-replay-analysis.json", mkdir=True
    )
    output_csv: Path = cherries.output(
        "20-fixed-activation-prestrain-replay-trajectories.csv", mkdir=True
    )
    output_trajectory_plot: Path = cherries.output(
        "20-fixed-activation-prestrain-alpha-trajectories.png", mkdir=True
    )
    output_quality_plot: Path = cherries.output(
        "20-fixed-activation-prestrain-quality-trajectories.png", mkdir=True
    )
    output_geometry_views: Path = cherries.output(
        "20-fixed-activation-prestrain-terminal-geometry.png", mkdir=True
    )
    output_residual_views: Path = cherries.output(
        "20-fixed-activation-prestrain-terminal-normal-residual.png", mkdir=True
    )
    output_report: Path = cherries.output(
        "20-fixed-activation-prestrain-replay-analysis.md", mkdir=True
    )


@dataclass(frozen=True)
class MetricBasis:
    mesh: pv.UnstructuredGrid
    skin: pv.PolyData
    target: np.ndarray
    baseline_displacement: np.ndarray
    baseline_activation: np.ndarray
    loss_mask: np.ndarray
    fixed_mask: np.ndarray
    fixed_point_arrays: dict[str, np.ndarray]
    target_rms: float
    face_target_rms: float
    skin_mesh_ids: np.ndarray
    triangles: np.ndarray
    edges: np.ndarray
    rest_area: np.ndarray
    target_area: np.ndarray
    target_vertex_normals: np.ndarray
    contraction_tri_0: np.ndarray
    contraction_tri_1: np.ndarray
    contraction_target_dihedral: np.ndarray
    contraction_edge_weight: np.ndarray
    tets: np.ndarray
    rest_six_volume: np.ndarray
    rest_area_vectors: np.ndarray
    rest_area_vector_norm: np.ndarray
    contraction_mask_sha256: str


@dataclass(frozen=True)
class ReplayArtifact:
    case_id: str
    path_kind: str
    alpha: float
    row: dict[str, Any]
    result_path: Path
    skin_path: Path
    displacement: np.ndarray
    metrics: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.case_id}/{self.path_kind}/alpha-{round(100 * self.alpha):03d}"


@dataclass(frozen=True)
class RenderBasis:
    face_focus: np.ndarray
    face_scale: float
    mouth_focus: np.ndarray
    mouth_scale: float
    eye_focus: np.ndarray
    eye_scale: float


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.dtype.kind == "f":
        array = array.astype("<f8", copy=False)
    elif array.dtype.kind in {"i", "u"}:
        array = array.astype("<i8", copy=False)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {token!r} in {path}")
        ),
    )
    if not isinstance(value, dict):
        msg = f"expected JSON object in {path}"
        raise TypeError(msg)
    return value


def _validate_config(cfg: Config) -> None:
    exact_paths = {
        "input_summary": (cfg.input_summary, PRODUCER_SUMMARY),
        "input_baseline_result": (cfg.input_baseline_result, BASELINE_RESULT),
        "input_baseline_summary": (cfg.input_baseline_summary, BASELINE_SUMMARY),
        "input_baseline_skin": (cfg.input_baseline_skin, BASELINE_SKIN),
        "input_metric_skin": (cfg.input_metric_skin, METRIC_SKIN),
        "output_json": (cfg.output_json, OUTPUT_JSON),
        "output_csv": (cfg.output_csv, OUTPUT_CSV),
        "output_trajectory_plot": (
            cfg.output_trajectory_plot,
            OUTPUT_TRAJECTORY_PLOT,
        ),
        "output_quality_plot": (cfg.output_quality_plot, OUTPUT_QUALITY_PLOT),
        "output_geometry_views": (
            cfg.output_geometry_views,
            OUTPUT_GEOMETRY_VIEWS,
        ),
        "output_residual_views": (
            cfg.output_residual_views,
            OUTPUT_RESIDUAL_VIEWS,
        ),
        "output_report": (cfg.output_report, OUTPUT_REPORT),
    }
    changed = [
        f"{name}: {actual} != {expected}"
        for name, (actual, expected) in exact_paths.items()
        if actual.resolve() != expected.resolve()
    ]
    if changed:
        msg = "reviewed analyzer paths changed: " + "; ".join(changed)
        raise ValueError(msg)
    if not ANALYZER_EXECUTION_APPROVED_AFTER_STATIC_REVIEW:
        raise RuntimeError(ANALYZER_APPROVAL_BLOCKER)


def _require_identity(
    path: Path, *, expected_size: int, expected_sha256: str, name: str
) -> dict[str, int | str]:
    if not path.is_file():
        msg = f"missing pinned {name}: {path}"
        raise FileNotFoundError(msg)
    identity = {"size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}
    expected = {"size_bytes": expected_size, "sha256": expected_sha256}
    if identity != expected:
        msg = f"{name} identity mismatch: expected {expected}, got {identity}"
        raise ValueError(msg)
    return identity


def _require_artifact_identity(
    path: Path, row: dict[str, Any], *, prefix: str
) -> dict[str, int | str]:
    if not path.is_file():
        msg = f"missing replay artifact: {path}"
        raise FileNotFoundError(msg)
    actual = {"size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}
    expected = {
        "size_bytes": int(row[f"artifact/{prefix}_size_bytes"]),
        "sha256": str(row[f"artifact/{prefix}_sha256"]),
    }
    if actual != expected:
        msg = (
            f"{prefix} identity mismatch for {path}: expected {expected}, got {actual}"
        )
        raise ValueError(msg)
    return actual


def _resolve_artifact_path(value: Any, *, expected_parent: Path) -> Path:
    path = Path(str(value)).resolve()
    parent = expected_parent.resolve()
    if parent not in path.parents:
        msg = f"artifact escapes the producer output directory: {path}"
        raise ValueError(msg)
    return path


def _triangle_faces(surface: pv.PolyData) -> np.ndarray:
    encoded = np.asarray(surface.faces, dtype=np.int64)
    if encoded.size != 4 * surface.n_cells:
        msg = "skin connectivity is not packed triangles"
        raise ValueError(msg)
    encoded = encoded.reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        msg = "skin contains non-triangle cells"
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
        msg = "surface contains non-finite or degenerate triangles"
        raise ValueError(msg)
    return vectors, 0.5 * norms, vectors / norms[:, None]


def _unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.vstack(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _interior_edge_adjacency(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.vstack(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
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
    lengths = np.linalg.norm(
        points[unique_edges[:, 1]] - points[unique_edges[:, 0]], axis=1
    )
    return tri_0, tri_1, lengths


def _map_global_ids(mesh: pv.UnstructuredGrid, surface: pv.PolyData) -> np.ndarray:
    mesh_ids = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    requested = np.asarray(surface.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if np.unique(mesh_ids).size != mesh_ids.size:
        msg = "baseline mesh GlobalPointId is not unique"
        raise ValueError(msg)
    order = np.argsort(mesh_ids)
    positions = np.searchsorted(mesh_ids[order], requested)
    if np.any(positions >= mesh_ids.size) or not np.array_equal(
        mesh_ids[order[positions]], requested
    ):
        msg = "skin GlobalPointId does not map exactly to the baseline mesh"
        raise ValueError(msg)
    mapped = order[positions]
    if not np.array_equal(np.asarray(surface.points), np.asarray(mesh.points)[mapped]):
        msg = "skin coordinates differ from mapped baseline coordinates"
        raise ValueError(msg)
    return mapped


def _canonical_triangle_keys(surface: pv.PolyData) -> np.ndarray:
    triangles = _triangle_faces(surface)
    global_ids = np.asarray(surface.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    return np.sort(global_ids[triangles], axis=1)


def _map_canonical_contraction_mask(
    metric_skin: pv.PolyData, corrected_skin: pv.PolyData
) -> tuple[np.ndarray, dict[str, Any]]:
    required = {"IsFaceTriangle", "ContractionPrestrainMask"}
    missing = required - set(metric_skin.cell_data)
    if missing:
        msg = f"canonical metric skin misses arrays: {sorted(missing)}"
        raise ValueError(msg)
    source_keys = _canonical_triangle_keys(metric_skin)
    corrected_keys = _canonical_triangle_keys(corrected_skin)
    corrected_hash = _array_sha256(corrected_keys)
    if corrected_hash != EXPECTED_CORRECTED_TRIANGLE_KEYS_SHA256:
        msg = (
            "corrected triangle-key identity changed: "
            f"expected {EXPECTED_CORRECTED_TRIANGLE_KEYS_SHA256}, got {corrected_hash}"
        )
        raise ValueError(msg)
    source_lookup = {
        tuple(int(value) for value in key): index
        for index, key in enumerate(source_keys)
    }
    if len(source_lookup) != source_keys.shape[0]:
        msg = "canonical metric skin has duplicate triangle keys"
        raise ValueError(msg)
    try:
        mapped_source = np.asarray(
            [
                source_lookup[tuple(int(value) for value in key)]
                for key in corrected_keys
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        msg = "corrected skin does not map exactly into canonical metric skin"
        raise ValueError(msg) from error
    face = np.asarray(metric_skin.cell_data["IsFaceTriangle"], dtype=bool)
    if not np.all(face[mapped_source]):
        msg = "corrected skin mapped outside canonical IsFace metric ROI"
        raise ValueError(msg)
    mapped_hash = _array_sha256(mapped_source)
    if mapped_hash != EXPECTED_MAPPED_DRIVER_CELLS_SHA256:
        msg = f"canonical metric mapping changed: {mapped_hash}"
        raise ValueError(msg)
    contraction = np.asarray(
        metric_skin.cell_data["ContractionPrestrainMask"], dtype=bool
    )[mapped_source]
    return contraction, {
        "corrected_triangle_keys_sha256_le_i8": corrected_hash,
        "mapped_source_indices_sha256_le_i8": mapped_hash,
        "contraction_mask_sha256": _array_sha256(contraction.astype(np.int8)),
        "contraction_triangles": int(contraction.sum()),
    }


def _encoded_tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    encoded = np.asarray(mesh.cells, dtype=np.int64)
    if encoded.size != 5 * mesh.n_cells:
        msg = "baseline mesh is not pure tetrahedral"
        raise ValueError(msg)
    encoded = encoded.reshape(-1, 5)
    if not np.all(encoded[:, 0] == 4):
        msg = "baseline mesh contains non-tetrahedral cells"
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
    points: np.ndarray, triangles: np.ndarray, area_vectors: np.ndarray
) -> np.ndarray:
    normals = np.zeros_like(points)
    for local in range(3):
        np.add.at(normals, triangles[:, local], area_vectors)
    norms = np.linalg.norm(normals, axis=1)
    used = np.unique(triangles)
    if np.any(norms[used] <= np.finfo(np.float64).eps):
        msg = "skin has a vertex with undefined target normal"
        raise ValueError(msg)
    normals[used] /= norms[used, None]
    return normals


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    if not 0.0 <= quantile <= 1.0:
        msg = f"invalid quantile {quantile}"
        raise ValueError(msg)
    order = np.argsort(values)
    ordered_values = np.asarray(values)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(ordered_weights)
    if cumulative[-1] <= 0.0:
        msg = "weighted quantile has zero total weight"
        raise ValueError(msg)
    return float(ordered_values[np.searchsorted(cumulative, quantile * cumulative[-1])])


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


def _vector_graph_laplacian(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    neighbor_sum = np.zeros_like(values)
    neighbor_count = np.zeros(values.shape[0], dtype=np.int64)
    np.add.at(neighbor_sum, edges[:, 0], values[edges[:, 1]])
    np.add.at(neighbor_sum, edges[:, 1], values[edges[:, 0]])
    np.add.at(neighbor_count, edges[:, 0], 1)
    np.add.at(neighbor_count, edges[:, 1], 1)
    active = neighbor_count > 0
    result = np.zeros_like(values)
    result[active] = (
        values[active] - neighbor_sum[active] / neighbor_count[active, None]
    )
    return result


def _area_weighted_point_rms(
    values: np.ndarray, triangles: np.ndarray, weights: np.ndarray
) -> float:
    point_squared = np.sum(np.square(values), axis=1)
    triangle_squared = np.mean(point_squared[triangles], axis=1)
    return float(np.sqrt(np.dot(weights, triangle_squared) / weights.sum()))


def _build_metric_basis(  # noqa: C901, PLR0912, PLR0915
    mesh: pv.UnstructuredGrid, skin: pv.PolyData, metric_skin: pv.PolyData
) -> MetricBasis:
    required_point = {
        "Displacement",
        "TargetDisplacement",
        "LossMask",
        GLOBAL_POINT_ID.vtk,
        *PINNED_FIXED_POINT_ARRAYS,
    }
    missing_point = required_point - set(mesh.point_data)
    if missing_point:
        msg = f"baseline result misses point arrays: {sorted(missing_point)}"
        raise ValueError(msg)
    missing_activation = {
        ACTIVATION_INV.vtk,
        "RecoveredActivationInv",
    } - set(mesh.cell_data)
    if missing_activation:
        msg = f"baseline result misses activation arrays: {sorted(missing_activation)}"
        raise ValueError(msg)
    target = np.asarray(mesh.point_data["TargetDisplacement"], dtype=np.float64)
    baseline_displacement = np.asarray(
        mesh.point_data["Displacement"], dtype=np.float64
    )
    baseline_activation = np.asarray(
        mesh.cell_data["RecoveredActivationInv"], dtype=np.float64
    )
    baseline_activation_saved = np.asarray(
        mesh.cell_data[ACTIVATION_INV.vtk], dtype=np.float64
    )
    if not np.array_equal(baseline_activation, baseline_activation_saved):
        msg = "baseline ActivationInv and RecoveredActivationInv differ"
        raise ValueError(msg)
    if _array_sha256(baseline_activation) != EXPECTED_FIXED_ACTIVATION_SHA256:
        msg = "baseline fixed muscle activation identity changed"
        raise ValueError(msg)
    loss_mask = np.asarray(mesh.point_data["LossMask"], dtype=bool)
    fixed_point_arrays = {
        name: np.asarray(mesh.point_data[name]).copy()
        for name in PINNED_FIXED_POINT_ARRAYS
    }
    is_fixed = np.asarray(fixed_point_arrays["IsFixed"], dtype=bool)
    cut = np.asarray(fixed_point_arrays["ArtificialCutIncident"], dtype=bool)
    cut_preexisting = np.asarray(
        fixed_point_arrays["CutBoundaryPreexistingFixed"], dtype=bool
    )
    cut_added = np.asarray(fixed_point_arrays["CutBoundaryAddedFixed"], dtype=bool)
    fixed_mask = np.asarray(fixed_point_arrays[FIXED_MASK.vtk], dtype=bool)
    fixed_value = np.asarray(fixed_point_arrays[FIXED_VALUE.vtk], dtype=np.float64)
    if (
        is_fixed.shape != (mesh.n_points,)
        or cut.shape != (mesh.n_points,)
        or cut_preexisting.shape != (mesh.n_points,)
        or cut_added.shape != (mesh.n_points,)
        or fixed_mask.shape != (mesh.n_points, 3)
        or fixed_value.shape != (mesh.n_points, 3)
    ):
        msg = "baseline fixed/cut arrays are malformed"
        raise ValueError(msg)
    if (
        int(is_fixed.sum()) != EXPECTED_MODEL_FIXED_VERTICES
        or int(fixed_mask.sum()) != EXPECTED_MODEL_FIXED_DOFS
        or int(cut.sum()) != EXPECTED_CUT_VERTICES
    ):
        msg = "baseline hard-fixed or cut counts changed"
        raise ValueError(msg)
    if not (
        np.array_equal(fixed_mask, np.repeat(is_fixed[:, None], 3, axis=1))
        and np.array_equal(cut_preexisting | cut_added, cut)
        and not np.any(cut_preexisting & cut_added)
        and np.all(is_fixed[cut])
        and np.all(fixed_value[is_fixed] == 0.0)
        and np.all(baseline_displacement[is_fixed] == 0.0)
    ):
        msg = "baseline hard-fixed/cut exact-zero contract changed"
        raise ValueError(msg)
    for name, values in (
        ("target", target),
        ("baseline displacement", baseline_displacement),
        ("baseline activation", baseline_activation),
    ):
        if not np.isfinite(values).all():
            msg = f"{name} is non-finite"
            raise ValueError(msg)
    target_rms = float(np.linalg.norm(target[loss_mask]) / math.sqrt(loss_mask.sum()))
    if target_rms <= 0.0:
        msg = "baseline target RMS is not positive"
        raise ValueError(msg)
    if not math.isclose(
        target_rms,
        EXPECTED_LOSS_TARGET_RMS_M,
        rel_tol=1.0e-13,
        abs_tol=1.0e-15,
    ):
        msg = f"SmileLossMask target RMS changed: {target_rms}"
        raise ValueError(msg)

    skin_mesh_ids = _map_global_ids(mesh, skin)
    face_target_rms = float(
        np.linalg.norm(target[skin_mesh_ids]) / math.sqrt(skin_mesh_ids.size)
    )
    if face_target_rms <= 0.0:
        msg = "baseline IsFace target RMS is not positive"
        raise ValueError(msg)
    if not math.isclose(
        face_target_rms,
        EXPECTED_ISFACE_TARGET_RMS_M,
        rel_tol=1.0e-13,
        abs_tol=1.0e-15,
    ):
        msg = f"IsFace target RMS changed: {face_target_rms}"
        raise ValueError(msg)
    triangles = _triangle_faces(skin)
    rest_points = np.asarray(skin.points, dtype=np.float64)
    rest_vectors, rest_area, _ = _triangle_geometry(rest_points, triangles)
    target_points = rest_points + target[skin_mesh_ids]
    target_vectors, target_area, target_normals = _triangle_geometry(
        target_points, triangles
    )
    target_vertex_normals = _vertex_normals(target_points, triangles, target_vectors)
    edges = _unique_edges(triangles)
    tri_0, tri_1, edge_length = _interior_edge_adjacency(rest_points, triangles)
    contraction, contraction_mapping = _map_canonical_contraction_mask(
        metric_skin, skin
    )
    contraction_edges = contraction[tri_0] & contraction[tri_1]
    if not np.any(contraction_edges):
        msg = "raw target contraction ROI contains no interior edges"
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
    tets = _encoded_tetrahedra(mesh)
    rest_six_volume = _six_volume(np.asarray(mesh.points), tets)
    if np.any(np.abs(rest_six_volume) <= np.finfo(np.float64).eps):
        msg = "baseline contains a degenerate tetrahedron"
        raise ValueError(msg)
    return MetricBasis(
        mesh=mesh,
        skin=skin,
        target=target,
        baseline_displacement=baseline_displacement,
        baseline_activation=baseline_activation,
        loss_mask=loss_mask,
        fixed_mask=fixed_mask,
        fixed_point_arrays=fixed_point_arrays,
        target_rms=target_rms,
        face_target_rms=face_target_rms,
        skin_mesh_ids=skin_mesh_ids,
        triangles=triangles,
        edges=edges,
        rest_area=rest_area,
        target_area=target_area,
        target_vertex_normals=target_vertex_normals,
        contraction_tri_0=contraction_tri_0,
        contraction_tri_1=contraction_tri_1,
        contraction_target_dihedral=contraction_target_dihedral,
        contraction_edge_weight=edge_length[contraction_edges],
        tets=tets,
        rest_six_volume=rest_six_volume,
        rest_area_vectors=rest_vectors,
        rest_area_vector_norm=np.linalg.norm(rest_vectors, axis=1),
        contraction_mask_sha256=str(contraction_mapping["contraction_mask_sha256"]),
    )


def _recompute_metrics(basis: MetricBasis, displacement: np.ndarray) -> dict[str, Any]:
    if displacement.shape != basis.target.shape or not np.isfinite(displacement).all():
        msg = "replay displacement is malformed or non-finite"
        raise ValueError(msg)
    residual = displacement - basis.target
    error_rms = float(
        np.linalg.norm(residual[basis.loss_mask]) / math.sqrt(basis.loss_mask.sum())
    )
    skin_displacement = displacement[basis.skin_mesh_ids]
    skin_residual = residual[basis.skin_mesh_ids]
    deformed = np.asarray(basis.skin.points) + skin_displacement
    deformed_vectors, deformed_area, deformed_normals = _triangle_geometry(
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
    normal_laplacian = _scalar_graph_laplacian(residual_normal, basis.edges)
    displacement_laplacian = _vector_graph_laplacian(skin_displacement, basis.edges)
    residual_laplacian = _vector_graph_laplacian(skin_residual, basis.edges)

    deformed_six_volume = _six_volume(
        np.asarray(basis.mesh.points) + displacement, basis.tets
    )
    det_f = deformed_six_volume / basis.rest_six_volume
    signed_normal_ratio = np.einsum(
        "ij,ij->i", deformed_vectors, basis.rest_area_vectors
    ) / np.square(basis.rest_area_vector_norm)
    area_ratio = deformed_area / basis.rest_area
    if not all(
        np.isfinite(values).all() for values in (det_f, signed_normal_ratio, area_ratio)
    ):
        msg = "recomputed quality diagnostics are non-finite"
        raise ValueError(msg)
    folded = signed_normal_ratio <= 0.0
    fixed_values = displacement[basis.fixed_mask]
    return {
        "target/error_rms_m": error_rms,
        "target/error_rms_mm": 1.0e3 * error_rms,
        "target/error_rms_fraction_of_target": error_rms / basis.target_rms,
        "target/face_rest_area_weighted_error_rms_m": _area_weighted_point_rms(
            skin_residual, basis.triangles, basis.rest_area
        ),
        "target/face_target_area_weighted_error_rms_m": _area_weighted_point_rms(
            skin_residual, basis.triangles, basis.target_area
        ),
        "bumpiness/contraction_target_relative_dihedral_rms_rad": dihedral_rms,
        "bumpiness/contraction_target_relative_dihedral_rms_deg": math.degrees(
            dihedral_rms
        ),
        "bumpiness/residual_normal_laplacian_rms_m": float(
            np.linalg.norm(normal_laplacian) / math.sqrt(normal_laplacian.size)
        ),
        "bumpiness/displacement_laplacian_rms_m": float(
            np.linalg.norm(displacement_laplacian)
            / math.sqrt(displacement_laplacian.shape[0])
        ),
        "bumpiness/residual_laplacian_rms_m": float(
            np.linalg.norm(residual_laplacian) / math.sqrt(residual_laplacian.shape[0])
        ),
        "quality/inverted_tets": int(np.sum(det_f <= 0.0)),
        "quality/inverted_tet_fraction": float(np.mean(det_f <= 0.0)),
        "quality/detF_min": float(det_f.min()),
        "quality/detF_q001": float(np.quantile(det_f, 0.001)),
        "quality/skin_folded_triangles": int(folded.sum()),
        "quality/skin_folded_triangle_fraction": float(np.mean(folded)),
        "quality/skin_folded_rest_area_fraction": float(
            basis.rest_area[folded].sum() / basis.rest_area.sum()
        ),
        "quality/skin_signed_normal_ratio_q001": _weighted_quantile(
            signed_normal_ratio, basis.rest_area, 0.001
        ),
        "quality/skin_signed_normal_ratio_q999": _weighted_quantile(
            signed_normal_ratio, basis.rest_area, 0.999
        ),
        "quality/skin_area_ratio_q001": _weighted_quantile(
            area_ratio, basis.rest_area, 0.001
        ),
        "quality/skin_area_ratio_q999": _weighted_quantile(
            area_ratio, basis.rest_area, 0.999
        ),
        "fixed/displacement_exact_zero": bool(np.all(fixed_values == 0.0)),
        "fixed/displacement_max_abs_m": float(np.max(np.abs(fixed_values))),
        "field/residual_normal_m": residual_normal,
    }


def _displacement_delta(
    basis: MetricBasis, left: np.ndarray, right: np.ndarray
) -> dict[str, float]:
    delta = left - right
    skin_delta = delta[basis.skin_mesh_ids]
    return {
        "full_rms_m": float(np.linalg.norm(delta) / math.sqrt(delta.shape[0])),
        "loss_mask_rms_m": float(
            np.linalg.norm(delta[basis.loss_mask]) / math.sqrt(basis.loss_mask.sum())
        ),
        "loss_mask_fraction_of_target": float(
            np.linalg.norm(delta[basis.loss_mask])
            / math.sqrt(basis.loss_mask.sum())
            / basis.target_rms
        ),
        "isface_rms_m": float(
            np.linalg.norm(skin_delta) / math.sqrt(skin_delta.shape[0])
        ),
        "isface_fraction_of_target": float(
            np.linalg.norm(skin_delta)
            / math.sqrt(skin_delta.shape[0])
            / basis.face_target_rms
        ),
    }


def _require_close(left: Any, right: Any, *, context: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1.0e-9, abs_tol=1.0e-12):
        msg = f"{context} mismatch: recomputed {left!r}, producer {right!r}"
        raise ValueError(msg)


def _volume_lambda_mu(young: float, poisson: float) -> tuple[float, float]:
    lambda_ = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return lambda_, mu


def _expected_material_contract() -> dict[str, Any]:
    return {
        "skin": {
            "domain": "all-vertex IsFace filtered PolyData",
            "E_MPa": EXPECTED_SKIN_E_MPA,
            "nu": EXPECTED_SKIN_NU,
            "thickness_m": EXPECTED_SKIN_THICKNESS_M,
            "lame_conversion": SKIN_LAME_CONVERSION,
            "energy_measure": "fixed original reference area",
        },
        "volume": {
            "lame_conversion": VOLUME_LAME_CONVERSION,
            "fat": EXPECTED_VOLUME_MATERIALS["fat"],
            "muscle": {
                **EXPECTED_VOLUME_MATERIALS["muscle"],
                "activation": "fixed corrected p000 best/terminal step-40 tensor",
            },
            "aponeurosis": EXPECTED_VOLUME_MATERIALS["aponeurosis"],
        },
    }


def _expected_input_provenance() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(values["path"]),
            "size_bytes": values["size_bytes"],
            "sha256": values["sha256"],
        }
        for name, values in EXPECTED_INPUT_IDENTITIES.items()
    }


def _expected_runtime_dependency_rows() -> list[dict[str, Any]]:
    return [
        {
            "name": values["name"],
            "path": str(values["path"]),
            "size_bytes": values["size_bytes"],
            "sha256": values["sha256"],
        }
        for values in EXPECTED_RUNTIME_DEPENDENCIES
    ]


def _validate_aggregate_provenance(summary: dict[str, Any]) -> None:
    input_provenance = _expected_input_provenance()
    runtime_rows = _expected_runtime_dependency_rows()
    expected_runtime = {"files": runtime_rows, "all_exact": True}
    expected_recheck = {
        "input_provenance": {
            "all_unchanged": True,
            "files": [
                {"name": name, **identity, "unchanged_through_all_solves": True}
                for name, identity in input_provenance.items()
            ],
        },
        "runtime_dependencies": {
            "all_unchanged": True,
            "files": [
                {**identity, "unchanged_through_all_solves": True}
                for identity in runtime_rows
            ],
        },
    }
    expected_baseline = {
        "fixed_activation_sha256_le_f8": EXPECTED_FIXED_ACTIVATION_SHA256,
        "baseline_displacement_sha256_le_f8": (
            "f8ca27d820ff1f4b7afb734d917c9ec1292cd26ab96fc93090277dcc017268fb"
        ),
        "best_step": 40,
        "final_step": 40,
        "best_equals_terminal": True,
    }
    expected_output = {
        "root": str(PRODUCER_OUTPUT_ROOT),
        "summary_path": str(PRODUCER_SUMMARY),
        "table_path": str(PRODUCER_TABLE),
        "case_layout": (
            "<root>/c020/{continuation,direct}/alpha-NNN/"
            "{result.vtu,skin.vtp,forward-summary.json}"
        ),
        "expected_result_vtus": len(EXPECTED_PATHS),
        "expected_skin_vtps": len(EXPECTED_PATHS),
        "expected_forward_sidecars": len(EXPECTED_PATHS),
        "overwrite_policy": "refuse any existing aggregate or result root",
    }
    expected_sections = {
        "input_provenance": input_provenance,
        "runtime_dependencies": expected_runtime,
        "final_dependency_recheck": expected_recheck,
        "baseline": expected_baseline,
        "output_contract": expected_output,
    }
    changed = [
        name
        for name, expected in expected_sections.items()
        if summary[name] != expected
    ]
    if changed:
        msg = f"producer provenance/output sections changed: {changed}"
        raise ValueError(msg)


def _validate_aggregate_identity_and_material(
    summary: dict[str, Any], *, producer_sha256: str
) -> None:
    expected_identity = {
        "path": str(INPUT_PRODUCER.resolve()),
        "size_bytes": INPUT_PRODUCER.stat().st_size,
        "sha256": producer_sha256,
        "unchanged_through_all_solves": True,
    }
    if summary["producer_identity"] != expected_identity:
        msg = "aggregate producer identity is not the audited source identity"
        raise ValueError(msg)
    if summary["material_contract"] != _expected_material_contract():
        msg = "aggregate anatomy material contract changed"
        raise ValueError(msg)


def _reviewed_thresholds() -> dict[str, float]:
    return {
        "both_bumpiness_improvement_fraction": (BOTH_BUMPINESS_IMPROVEMENT_FRACTION),
        "single_bumpiness_improvement_fraction": (
            SINGLE_BUMPINESS_IMPROVEMENT_FRACTION
        ),
        "paired_bumpiness_max_regression_fraction": (
            PAIRED_BUMPINESS_MAX_REGRESSION_FRACTION
        ),
        "alpha0_roughness_relative_tolerance": ALPHA0_ROUGHNESS_RELATIVE_TOLERANCE,
        "max_target_error_relative_increase": MAX_TARGET_ERROR_RELATIVE_INCREASE,
        "branch_delta_fraction_of_target_tolerance": (
            BRANCH_DELTA_FRACTION_OF_TARGET_TOLERANCE
        ),
        "alpha0_delta_fraction_of_target_tolerance": (
            ALPHA0_DELTA_FRACTION_OF_TARGET_TOLERANCE
        ),
    }


def _validate_case_material_row(
    row: dict[str, Any], *, case_id: str, alpha: float, material: dict[str, Any]
) -> None:
    exact_expected: dict[str, Any] = {
        "skin/lame_conversion": SKIN_LAME_CONVERSION,
        "skin/domain": "all-vertex IsFace filtered PolyData",
        "skin/triangles": 29_899,
        "skin/energy_measure": "fixed original reference area",
        "skin/rho_sha256_le_f8": material["rho_sha256_le_f8"],
        "skin/activation_inv_sha256_le_f8": material["activation_sha256_le_f8"],
        "material/live_skin_activation_exact": True,
        "material/live_skin_activation_sha256_le_f8": material[
            "activation_sha256_le_f8"
        ],
        "material/live_fixed_activation_exact": True,
        "material/live_fixed_activation_sha256_le_f8": (
            EXPECTED_FIXED_ACTIVATION_SHA256
        ),
    }
    numeric_expected: dict[str, float] = {
        "skin/alpha": alpha,
        "skin/E_MPa": EXPECTED_SKIN_E_MPA,
        "skin/nu": EXPECTED_SKIN_NU,
        "skin/rho_min": material["rho_min"],
        "skin/rho_area_weighted_mean": material["rho_area_weighted_mean"],
        "skin/activation_inv_max": material["activation_diag_max"],
    }
    for name, values in EXPECTED_VOLUME_MATERIALS.items():
        young = values["E_MPa"]
        poisson = values["nu"]
        lambda_, mu = _volume_lambda_mu(young, poisson)
        exact_expected[f"material/{name}_volume_conversion"] = "3d"
        numeric_expected.update(
            {
                f"material/{name}_E_MPa": young,
                f"material/{name}_nu": poisson,
                f"material/{name}_lambda_MPa": lambda_,
                f"material/{name}_mu_MPa": mu,
            }
        )
    missing = (set(exact_expected) | set(numeric_expected)) - set(row)
    if missing:
        msg = f"{case_id} material row misses fields: {sorted(missing)}"
        raise ValueError(msg)
    changed_exact = {
        key: (row[key], expected)
        for key, expected in exact_expected.items()
        if row[key] != expected
    }
    if changed_exact:
        msg = f"{case_id} exact material contract changed: {changed_exact}"
        raise ValueError(msg)
    for key, expected in numeric_expected.items():
        _require_close(row[key], expected, context=f"{case_id} {key}")


def _validate_skin_material(
    skin: pv.PolyData, *, alpha: float, case_id: str
) -> dict[str, Any]:
    required = {
        "RestArea",
        "SkinYoungModulusMPa",
        "SkinPoissonRatio",
        LAMBDA.vtk,
        MU.vtk,
        FRACTION.vtk,
        "TargetRestAreaRatio",
        "PrestrainNaturalAreaRatioFull",
        "PrestrainNaturalAreaRatio",
        "StressFreeAreaRatio",
        "SkinActivationInvDiag",
        "PrestrainAlpha",
        ACTIVATION_INV.vtk,
    }
    missing = required - set(skin.cell_data)
    if missing:
        msg = f"{case_id} alpha={alpha} skin misses arrays {sorted(missing)}"
        raise ValueError(msg)
    triangles = _triangle_faces(skin)
    points = np.asarray(skin.points, dtype=np.float64)
    geometric_area = 0.5 * np.linalg.norm(
        np.cross(
            points[triangles[:, 1]] - points[triangles[:, 0]],
            points[triangles[:, 2]] - points[triangles[:, 0]],
        ),
        axis=1,
    )
    rest_area = np.asarray(skin.cell_data["RestArea"], dtype=np.float64)
    young = np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64)
    nu = np.asarray(skin.cell_data["SkinPoissonRatio"], dtype=np.float64)
    lambda_ = np.asarray(skin.cell_data[LAMBDA.vtk], dtype=np.float64)
    mu = np.asarray(skin.cell_data[MU.vtk], dtype=np.float64)
    fraction = np.asarray(skin.cell_data[FRACTION.vtk], dtype=np.float64)
    ratio = np.asarray(skin.cell_data["TargetRestAreaRatio"], dtype=np.float64)
    rho_full_saved = np.asarray(
        skin.cell_data["PrestrainNaturalAreaRatioFull"], dtype=np.float64
    )
    rho = np.asarray(skin.cell_data["PrestrainNaturalAreaRatio"], dtype=np.float64)
    stress_free = np.asarray(skin.cell_data["StressFreeAreaRatio"], dtype=np.float64)
    activation = np.asarray(skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    saved_diag = np.asarray(skin.cell_data["SkinActivationInvDiag"], dtype=np.float64)
    saved_alpha = np.asarray(skin.cell_data["PrestrainAlpha"], dtype=np.float64)
    arrays = (
        rest_area,
        young,
        nu,
        lambda_,
        mu,
        fraction,
        ratio,
        rho_full_saved,
        rho,
        stress_free,
        activation,
        saved_diag,
        saved_alpha,
    )
    if not all(
        values.ndim >= 1 and values.shape[0] == skin.n_cells for values in arrays
    ) or not all(np.isfinite(values).all() for values in arrays):
        msg = f"{case_id} alpha={alpha} has malformed or non-finite skin fields"
        raise ValueError(msg)
    expected_lambda = young * nu / (1.0 - np.square(nu))
    expected_mu = young / (2.0 * (1.0 + nu))
    homogeneous_plane_stress = (
        np.array_equal(rest_area, geometric_area)
        and np.allclose(young, EXPECTED_SKIN_E_MPA, rtol=1.0e-13, atol=1.0e-14)
        and np.allclose(nu, EXPECTED_SKIN_NU, rtol=1.0e-13, atol=1.0e-14)
        and np.allclose(lambda_, expected_lambda, rtol=1.0e-13, atol=1.0e-14)
        and np.allclose(mu, expected_mu, rtol=1.0e-13, atol=1.0e-14)
        and np.array_equal(fraction, np.ones_like(fraction))
    )
    if not homogeneous_plane_stress:
        msg = f"{case_id} alpha={alpha} is not the pinned homogeneous plane-stress skin"
        raise ValueError(msg)
    if _array_sha256(ratio) != EXPECTED_RAW_RATIO_SHA256:
        msg = f"{case_id} alpha={alpha} target-area driver changed"
        raise ValueError(msg)
    rho_full = (1.0 - EXPECTED_DOSE) ** 2 * np.clip(ratio, EXPECTED_FLOOR, 1.0)
    expected_rho = np.power(rho_full, alpha)
    expected_diag = np.power(expected_rho, -0.5) - 1.0
    if activation.shape != (skin.n_cells, 3):
        msg = f"malformed skin ActivationInv shape {activation.shape}"
        raise ValueError(msg)
    if not np.allclose(rho_full_saved, rho_full, rtol=2.0e-13, atol=2.0e-15):
        msg = f"{case_id} alpha={alpha} full natural-area formula mismatch"
        raise ValueError(msg)
    if not (
        np.allclose(rho, expected_rho, rtol=2.0e-13, atol=2.0e-15)
        and np.array_equal(stress_free, rho)
    ):
        msg = f"{case_id} alpha={alpha} stress-free area formula mismatch"
        raise ValueError(msg)
    if not (
        np.allclose(activation[:, 0], expected_diag, rtol=2.0e-13, atol=2.0e-15)
        and np.array_equal(activation[:, 0], activation[:, 1])
        and np.all(activation[:, 2] == 0.0)
        and np.array_equal(saved_diag, activation[:, 0])
        and np.array_equal(saved_alpha, np.full_like(saved_alpha, alpha))
    ):
        msg = f"{case_id} alpha={alpha} ActivationInv formula mismatch"
        raise ValueError(msg)
    return {
        "rho_sha256_le_f8": _array_sha256(rho),
        "activation_sha256_le_f8": _array_sha256(activation),
        "rho_min": float(rho.min()),
        "rho_area_weighted_mean": float(
            np.dot(np.asarray(skin.cell_data["RestArea"]), rho)
            / np.asarray(skin.cell_data["RestArea"]).sum()
        ),
        "activation_diag_max": float(expected_diag.max()),
    }


def _load_artifact(  # noqa: C901, PLR0912, PLR0915
    row: dict[str, Any], *, basis: MetricBasis, output_root: Path
) -> ReplayArtifact:
    required = {
        "case_id",
        "dose_id",
        "path_kind",
        "alpha",
        "status",
        "artifact/result_path",
        "artifact/result_sha256",
        "artifact/result_size_bytes",
        "artifact/skin_path",
        "artifact/skin_sha256",
        "artifact/skin_size_bytes",
        "artifact/summary_path",
        "forward/success",
        "fixed_activation/exact_before_solve",
        "fixed_activation/exact_after_solve",
        "execution/forward_only",
        "execution/inverse_started",
        "execution/adjoint_started",
        "execution/backward_started",
        "readback/result_ok",
        "readback/skin_ok",
        "readback/all_fixed_displacement_exact_zero",
        "readback/cut_displacement_exact_zero",
        "validation/ok",
    }
    missing = required - set(row)
    if missing:
        msg = f"producer row misses keys: {sorted(missing)}"
        raise ValueError(msg)
    case_id = str(row["case_id"])
    path_kind = str(row["path_kind"])
    alpha = float(row["alpha"])
    result_path = _resolve_artifact_path(
        row["artifact/result_path"], expected_parent=output_root
    )
    skin_path = _resolve_artifact_path(
        row["artifact/skin_path"], expected_parent=output_root
    )
    summary_path = _resolve_artifact_path(
        row["artifact/summary_path"], expected_parent=output_root
    )
    alpha_dir = f"alpha-{round(100 * alpha):03d}"
    expected_root = output_root / "c020" / path_kind / alpha_dir
    if (
        result_path != expected_root / "result.vtu"
        or skin_path != expected_root / "skin.vtp"
    ):
        msg = f"unexpected replay artifact layout for {case_id}"
        raise ValueError(msg)
    if summary_path != expected_root / "forward-summary.json":
        msg = f"unexpected replay sidecar layout for {case_id}"
        raise ValueError(msg)
    if not summary_path.is_file() or _read_json(summary_path) != row:
        msg = f"aggregate row and per-case sidecar differ for {case_id}"
        raise ValueError(msg)
    _require_artifact_identity(result_path, row, prefix="result")
    _require_artifact_identity(skin_path, row, prefix="skin")
    result = pv.read(result_path)
    if not isinstance(result, pv.UnstructuredGrid):
        msg = f"result is not UnstructuredGrid: {result_path}"
        raise TypeError(msg)
    if result.n_points != basis.mesh.n_points or result.n_cells != basis.mesh.n_cells:
        msg = f"result topology size changed: {result_path}"
        raise ValueError(msg)
    if not (
        np.array_equal(np.asarray(result.points), np.asarray(basis.mesh.points))
        and np.array_equal(np.asarray(result.cells), np.asarray(basis.mesh.cells))
        and np.array_equal(
            np.asarray(result.celltypes), np.asarray(basis.mesh.celltypes)
        )
    ):
        msg = f"result rest geometry or topology changed: {result_path}"
        raise ValueError(msg)
    for name, expected in (
        ("TargetDisplacement", basis.target),
        ("LossMask", basis.loss_mask.astype(np.int8)),
        (
            GLOBAL_POINT_ID.vtk,
            np.asarray(basis.mesh.point_data[GLOBAL_POINT_ID.vtk]),
        ),
    ):
        if name not in result.point_data or not np.array_equal(
            np.asarray(result.point_data[name]), expected
        ):
            msg = f"result {name} changed: {result_path}"
            raise ValueError(msg)
    for name, expected in basis.fixed_point_arrays.items():
        if name not in result.point_data or not np.array_equal(
            np.asarray(result.point_data[name]), expected
        ):
            msg = f"result fixed/cut array {name} changed: {result_path}"
            raise ValueError(msg)
    required_activation_fields = {ACTIVATION_INV.vtk, "RecoveredActivationInv"}
    missing_activation = required_activation_fields - set(result.cell_data)
    if missing_activation:
        msg = f"result misses fixed activation fields: {sorted(missing_activation)}"
        raise ValueError(msg)
    activation = np.asarray(result.cell_data[ACTIVATION_INV.vtk])
    recovered_activation = np.asarray(result.cell_data["RecoveredActivationInv"])
    if not (
        np.array_equal(activation, basis.baseline_activation)
        and np.array_equal(recovered_activation, basis.baseline_activation)
    ):
        msg = f"fixed muscle activation changed: {result_path}"
        raise ValueError(msg)
    displacement = np.asarray(result.point_data["Displacement"], dtype=np.float64)
    is_fixed = np.asarray(basis.fixed_point_arrays["IsFixed"], dtype=bool)
    cut = np.asarray(basis.fixed_point_arrays["ArtificialCutIncident"], dtype=bool)
    if np.any(displacement[is_fixed] != 0.0) or np.any(displacement[cut] != 0.0):
        msg = f"result violates exact-zero hard-fixed/cut displacement: {result_path}"
        raise ValueError(msg)
    metrics = _recompute_metrics(basis, displacement)
    for key in (*FIDELITY_METRICS, *PRIMARY_BUMPINESS):
        if key in row:
            _require_close(
                metrics[key], row[key], context=f"{case_id}/{path_kind}/{alpha} {key}"
            )
    for producer_key, metric_key in (
        ("target/loss_mask_error_rms_m", "target/error_rms_m"),
        ("target/loss_mask_error_rms_mm", "target/error_rms_mm"),
    ):
        _require_close(
            metrics[metric_key],
            row[producer_key],
            context=f"{case_id}/{path_kind}/{alpha} {producer_key}",
        )
    skin = pv.read(skin_path)
    if not isinstance(skin, pv.PolyData):
        msg = f"skin is not PolyData: {skin_path}"
        raise TypeError(msg)
    if not (
        np.array_equal(np.asarray(skin.points), np.asarray(basis.skin.points))
        and np.array_equal(_triangle_faces(skin), basis.triangles)
    ):
        msg = f"skin geometry or topology changed: {skin_path}"
        raise ValueError(msg)
    material = _validate_skin_material(skin, alpha=alpha, case_id=case_id)
    _validate_case_material_row(row, case_id=case_id, alpha=alpha, material=material)
    metrics = {
        **metrics,
        **{f"material/{key}": value for key, value in material.items()},
    }
    if str(row["status"]) != "ok" or not bool(row["forward/success"]):
        msg = f"producer row is not a successful forward solve: {case_id}/{path_kind}/{alpha}"
        raise ValueError(msg)
    required_true = (
        "fixed_activation/exact_before_solve",
        "fixed_activation/exact_after_solve",
        "execution/forward_only",
        "readback/result_ok",
        "readback/skin_ok",
        "readback/all_fixed_displacement_exact_zero",
        "readback/cut_displacement_exact_zero",
        "validation/ok",
    )
    if not all(bool(row[key]) for key in required_true) or any(
        bool(row[key])
        for key in (
            "execution/inverse_started",
            "execution/adjoint_started",
            "execution/backward_started",
        )
    ):
        msg = f"case execution/readback contract failed for {case_id}"
        raise ValueError(msg)
    if str(row["dose_id"]) != CASE_ID:
        msg = f"unexpected dose_id in {case_id}: {row['dose_id']}"
        raise ValueError(msg)
    return ReplayArtifact(
        case_id=case_id,
        path_kind=path_kind,
        alpha=alpha,
        row=row,
        result_path=result_path,
        skin_path=skin_path,
        displacement=displacement,
        metrics=metrics,
    )


def _producer_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("cases")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        msg = "producer cases must be a flat list of row objects"
        raise TypeError(msg)
    return rows


def _validate_producer_contract(summary: dict[str, Any]) -> dict[str, Any]:
    thresholds = _reviewed_thresholds()
    actual_producer_sha = _file_sha256(INPUT_PRODUCER)
    if actual_producer_sha != INPUT_PRODUCER_SHA256:
        msg = (
            "producer changed after audit: "
            f"expected {INPUT_PRODUCER_SHA256}, got {actual_producer_sha}"
        )
        raise ValueError(msg)
    expected_top = {
        "schema_version",
        "design",
        "complete",
        "execution_contract",
        "approval",
        "protocol",
        "input_provenance",
        "producer_identity",
        "runtime_dependencies",
        "final_dependency_recheck",
        "baseline",
        "mapping",
        "material_contract",
        "case_order",
        "cases",
        "branch_comparison",
        "output_contract",
    }
    if set(summary) != expected_top:
        msg = (
            f"producer top-level schema changed: {sorted(set(summary) ^ expected_top)}"
        )
        raise ValueError(msg)
    if int(summary["schema_version"]) != INPUT_SCHEMA_VERSION:
        msg = "producer schema version changed"
        raise ValueError(msg)
    if str(summary["design"]) != INPUT_DESIGN or not bool(summary["complete"]):
        msg = "producer design is wrong or incomplete"
        raise ValueError(msg)
    _validate_aggregate_identity_and_material(
        summary, producer_sha256=actual_producer_sha
    )
    _validate_aggregate_provenance(summary)
    execution = summary["execution_contract"]
    expected_execution = {
        "forward_only": True,
        "forward_solves": len(EXPECTED_PATHS),
        "fixed_muscle_activation": True,
        "inverse_started": False,
        "adjoint_started": False,
        "backward_started": False,
        "activation_optimized": False,
    }
    if execution != expected_execution:
        msg = "producer violated the reviewed six-solve forward-only contract"
        raise ValueError(msg)
    approval = summary["approval"]
    expected_approval = {
        "static_source_blocker_was_explicitly_cleared": True,
        "c020_only": True,
        "c050_started": False,
        "c050_policy": (
            "conditional second-stage only after c020 analysis and a new isolated "
            "reviewed producer/run; this executable exposes no c050 option"
        ),
    }
    if approval != expected_approval:
        msg = "producer approval record is inconsistent with the reviewed c020-only run"
        raise ValueError(msg)
    if summary["case_order"] != list(EXPECTED_CASE_IDS):
        msg = "producer case_order changed"
        raise ValueError(msg)
    protocol = summary["protocol"]
    expected_protocol = {
        "dose_id": CASE_ID,
        "linear_tightening": EXPECTED_DOSE,
        "length_factor": 1.0 - EXPECTED_DOSE,
        "uniform_natural_area_ratio": (1.0 - EXPECTED_DOSE) ** 2,
        "area_ratio_floor": EXPECTED_FLOOR,
        "continuation_alphas": list(CONTINUATION_ALPHAS),
        "direct_alphas": [1.0],
        "alpha_interpolation": "rho_alpha=np.power(rho_full,alpha)",
        "alpha0_replay_tolerance_fraction_of_target": 1.0e-3,
        "alpha0_replay_domains": ["SmileLossMask", "corrected IsFace"],
        "continuation_seed_rule": "previous solved equilibrium displacement",
        "direct_seed_rule": "exact pinned corrected p000 step-40 displacement",
    }
    if protocol != expected_protocol:
        msg = "producer protocol differs from the reviewed c020 alpha replay"
        raise ValueError(msg)
    mapping = summary["mapping"]
    expected_mapping_fields = {
        "corrected_triangle_keys_sha256_le_i8": (
            EXPECTED_CORRECTED_TRIANGLE_KEYS_SHA256
        ),
        "mapped_driver_cell_indices_sha256_le_i8": (
            EXPECTED_MAPPED_DRIVER_CELLS_SHA256
        ),
        "raw_ratio_sha256_le_f8": EXPECTED_RAW_RATIO_SHA256,
        "rho_full_sha256_le_f8": EXPECTED_RHO_FULL_SHA256,
        "linear_tightening": EXPECTED_DOSE,
        "length_factor": 1.0 - EXPECTED_DOSE,
        "area_ratio_floor": EXPECTED_FLOOR,
        "floor_clamped_triangles": 31,
        "contraction_triangles": 13_159,
        "exact_readback": True,
    }
    changed_mapping = {
        key: (mapping.get(key), expected)
        for key, expected in expected_mapping_fields.items()
        if mapping.get(key) != expected
    }
    if changed_mapping:
        msg = f"producer prestrain mapping changed: {changed_mapping}"
        raise ValueError(msg)
    return {
        "producer_sha256": actual_producer_sha,
        "execution_contract": execution,
        "advisory_thresholds": thresholds,
    }


def _quality_warnings(baseline: dict[str, Any], terminal: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for key, noun in (
        ("quality/inverted_tets", "inverted tetrahedra"),
        ("quality/skin_folded_triangles", "folded IsFace triangles"),
    ):
        delta = int(terminal[key]) - int(baseline[key])
        if delta > 0:
            warnings.append(f"{delta} additional {noun} versus the exact baseline")
    if float(terminal["quality/detF_min"]) < float(baseline["quality/detF_min"]):
        warnings.append("minimum det(F) is lower than the exact baseline")
    if float(terminal["quality/skin_signed_normal_ratio_q001"]) < float(
        baseline["quality/skin_signed_normal_ratio_q001"]
    ):
        warnings.append("area-weighted q0.1% signed skin ratio is lower than baseline")
    if not bool(terminal["fixed/displacement_exact_zero"]):
        warnings.append("a fixed point has nonzero displacement")
    return warnings


def _smoothing_response(
    *,
    alpha0: dict[str, Any],
    terminal: dict[str, Any],
) -> dict[str, Any]:
    ratios = {
        key: float(terminal[key]) / float(alpha0[key]) for key in PRIMARY_BUMPINESS
    }
    values = tuple(ratios.values())
    both_improve = all(
        ratio <= 1.0 - BOTH_BUMPINESS_IMPROVEMENT_FRACTION for ratio in values
    )
    one_strong_other_safe = bool(
        min(values) <= 1.0 - SINGLE_BUMPINESS_IMPROVEMENT_FRACTION
        and max(values) <= 1.0 + PAIRED_BUMPINESS_MAX_REGRESSION_FRACTION
    )
    return {
        "ratios_to_alpha0": ratios,
        "both_improve_at_least_5_percent": both_improve,
        "one_improves_at_least_10_percent_other_worsens_at_most_1_percent": (
            one_strong_other_safe
        ),
        "passes": bool(both_improve or one_strong_other_safe),
    }


def _decision(
    *,
    basis: MetricBasis,
    baseline: dict[str, Any],
    continuation: ReplayArtifact,
    direct: ReplayArtifact,
    alpha0: ReplayArtifact,
) -> dict[str, Any]:
    alpha0_metrics = alpha0.metrics
    alpha0_roughness_ratios = {
        key: float(alpha0_metrics[key]) / float(baseline[key])
        for key in PRIMARY_BUMPINESS
    }
    alpha0_roughness_stable = all(
        abs(ratio - 1.0) <= ALPHA0_ROUGHNESS_RELATIVE_TOLERANCE
        for ratio in alpha0_roughness_ratios.values()
    )
    smoothing = {
        "continuation": _smoothing_response(
            alpha0=alpha0_metrics, terminal=continuation.metrics
        ),
        "direct": _smoothing_response(alpha0=alpha0_metrics, terminal=direct.metrics),
    }
    meaningful_smoothing = all(value["passes"] for value in smoothing.values())
    target_error_ratios = {
        "continuation": float(continuation.metrics["target/error_rms_m"])
        / float(alpha0_metrics["target/error_rms_m"]),
        "direct": float(direct.metrics["target/error_rms_m"])
        / float(alpha0_metrics["target/error_rms_m"]),
    }
    fidelity_acceptable = all(
        ratio <= 1.0 + MAX_TARGET_ERROR_RELATIVE_INCREASE
        for ratio in target_error_ratios.values()
    )
    alpha0_delta = _displacement_delta(
        basis, alpha0.displacement, basis.baseline_displacement
    )
    branch_delta = _displacement_delta(
        basis, continuation.displacement, direct.displacement
    )
    alpha0_stable = bool(
        alpha0_delta["loss_mask_fraction_of_target"]
        <= ALPHA0_DELTA_FRACTION_OF_TARGET_TOLERANCE
        and alpha0_delta["isface_fraction_of_target"]
        <= ALPHA0_DELTA_FRACTION_OF_TARGET_TOLERANCE
    )
    branch_stable = bool(
        branch_delta["loss_mask_fraction_of_target"]
        <= BRANCH_DELTA_FRACTION_OF_TARGET_TOLERANCE
        and branch_delta["isface_fraction_of_target"]
        <= BRANCH_DELTA_FRACTION_OF_TARGET_TOLERANCE
    )
    quality_warnings = {
        "continuation": _quality_warnings(baseline, continuation.metrics),
        "direct": _quality_warnings(baseline, direct.metrics),
    }
    numerical_eligible = bool(
        alpha0_stable
        and alpha0_roughness_stable
        and branch_stable
        and bool(continuation.metrics["fixed/displacement_exact_zero"])
        and bool(direct.metrics["fixed/displacement_exact_zero"])
    )
    if not numerical_eligible:
        outcome = "stop-c020-replay-or-branch-failure-do-not-escalate"
        reason = (
            "c=.02 did not reproduce the canonical alpha-0 roughness/branch within "
            "the declared tolerances; c=.05 must not be run until this is resolved"
        )
    elif not fidelity_acceptable:
        outcome = "stop-c020-target-fit-failure-do-not-escalate"
        reason = (
            "at least one alpha-1 branch worsens target RMS by more than 5% relative "
            "to the matched numerical alpha-0 replay; increasing dose is unsafe"
        )
    elif meaningful_smoothing:
        outcome = "pending-visual-c020-quantitatively-sufficient"
        reason = (
            "both alpha-1 branches clear the independent smoothing and fit rules; "
            "c=.02 is sufficient only if at least three of four matched views improve "
            "and no new visible artifact is found"
        )
    else:
        outcome = "pending-visual-c050-forward-probe-conditionally-required"
        reason = (
            "c=.02 is stable and keeps target RMS within 5%, but one or both alpha-1 "
            "branches do not clear the independent smoothing effect-size rule; if "
            "the four-view review finds no new artifact, request separate human "
            "approval for the c=.05 forward probe"
        )
    return {
        "outcome": outcome,
        "reason": reason,
        "advisory_only": True,
        "not_inverse_authorization": True,
        "visual_review_required": True,
        "final_c020_sufficiency_visual_rule": (
            "smoothing must be visible in at least three of the four fixed views "
            "(front, 30 degree, mouth, eye-cheek) with no new visible artifact"
        ),
        "visual_review_status": "pending",
        "visual_review_fields": {
            "fixed_views": ["front", "30 degree", "mouth", "eye-cheek (+x)"],
            "record_smoothing_visible_per_view": True,
            "record_any_new_visible_artifact": True,
        },
        "visual_review_resolution_rules": {
            "c020_sufficient": (
                "quantitative smoothing and fit pass, smoothing is visible in at "
                "least 3/4 fixed views, and no new visible artifact is present"
            ),
            "c050_conditionally_required": (
                "replay, branch, and fit pass and no new visible artifact is present, "
                "but either the quantitative smoothing rule fails or smoothing is "
                "visible in fewer than 3/4 fixed views; obtain separate approval and "
                "use a separate c=.05 forward-only producer"
            ),
            "stop_do_not_escalate": (
                "any replay, branch, solver, fixed-boundary, or fit failure, or any "
                "new visible artifact"
            ),
        },
        "threshold_basis": (
            "deterministic engineering effect sizes, not statistical significance: "
            "on both continuation and direct alpha-1 branches, either both primary "
            "Bumpy metrics improve by at least 5%, or one improves by at least 10% "
            "while the other worsens by at most 1%; target RMS may worsen by at most "
            "5% relative to alpha-0; alpha-0 canonical roughness tolerance is 1%; "
            "SmileLossMask and IsFace replay/branch tolerances are 1e-3 of their "
            "corresponding target RMS"
        ),
        "alpha0_roughness_ratios_to_canonical": alpha0_roughness_ratios,
        "alpha0_roughness_stable": alpha0_roughness_stable,
        "smoothing_response": smoothing,
        "meaningful_smoothing": meaningful_smoothing,
        "target_error_ratios_to_alpha0": target_error_ratios,
        "fidelity_acceptable": fidelity_acceptable,
        "alpha0_delta": alpha0_delta,
        "alpha0_stable": alpha0_stable,
        "branch_delta_continuation_vs_direct": branch_delta,
        "branch_stable": branch_stable,
        "quality_warnings": quality_warnings,
        "quality_policy": (
            "reported for matched visual review; small visually imperceptible folds "
            "or inversions are not automatic vetoes"
        ),
    }


def _validate_branch_summary(
    summary: dict[str, Any], *, basis: MetricBasis, left: np.ndarray, right: np.ndarray
) -> dict[str, float]:
    delta = _displacement_delta(basis, left, right)
    branch = summary["branch_comparison"]
    if not isinstance(branch, dict):
        msg = "producer branch_comparison is not an object"
        raise TypeError(msg)
    for producer_key, delta_key in (
        (
            "smile_loss_mask/delta_fraction_of_target",
            "loss_mask_fraction_of_target",
        ),
        ("isface/delta_fraction_of_target", "isface_fraction_of_target"),
        ("smile_loss_mask/delta_rms_m", "loss_mask_rms_m"),
        ("isface/delta_rms_m", "isface_rms_m"),
    ):
        _require_close(
            delta[delta_key], branch[producer_key], context=f"branch {producer_key}"
        )
    within = bool(
        delta["loss_mask_fraction_of_target"]
        <= BRANCH_DELTA_FRACTION_OF_TARGET_TOLERANCE
        and delta["isface_fraction_of_target"]
        <= BRANCH_DELTA_FRACTION_OF_TARGET_TOLERANCE
    )
    if bool(branch["within_tolerance"]) != within:
        msg = "producer branch gate differs from independent readback"
        raise ValueError(msg)
    return delta


def _bounds_camera(
    points: np.ndarray, *, aspect: float = 1.35, padding: float = 1.12
) -> tuple[np.ndarray, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    focus = 0.5 * (minimum + maximum)
    extent = maximum - minimum
    scale = 0.5 * max(float(extent[1]), float(extent[0]) / aspect)
    return focus, padding * scale


def _build_render_basis(basis: MetricBasis) -> RenderBasis:
    points = np.asarray(basis.skin.points)
    face_focus, face_scale = _bounds_camera(points)
    mesh_ids = basis.skin_mesh_ids
    lip = np.asarray(basis.mesh.point_data["IsLip"], dtype=bool)[mesh_ids]
    if not np.any(lip):
        msg = "IsFace skin has no lip points"
        raise ValueError(msg)
    mouth_focus, mouth_scale = _bounds_camera(points[lip], padding=1.25)
    group_names = tuple(str(value) for value in basis.skin.field_data["GroupName"])
    group_ids = np.asarray(basis.skin.point_data["GroupId"], dtype=np.int64)
    eyelid_names = {
        "EyelidTop",
        "EyelidBottom",
        "EyelidOuterTop",
        "EyelidOuterBottom",
    }
    eyelid_ids = [
        index for index, name in enumerate(group_names) if name in eyelid_names
    ]
    eyelid = np.isin(group_ids, eyelid_ids)
    if not np.any(eyelid):
        msg = "IsFace skin has no eyelid group points"
        raise ValueError(msg)
    one_eye = eyelid & (points[:, 0] >= np.median(points[eyelid, 0]))
    eye_focus, _ = _bounds_camera(points[one_eye])
    eye_focus = eye_focus.copy()
    eye_focus[1] -= 0.08 * float(np.ptp(points[:, 1]))
    eye_scale = 0.24 * float(np.ptp(points[:, 1]))
    return RenderBasis(
        face_focus=face_focus,
        face_scale=face_scale,
        mouth_focus=mouth_focus,
        mouth_scale=mouth_scale,
        eye_focus=eye_focus,
        eye_scale=eye_scale,
    )


def _deformed_skin(
    basis: MetricBasis, displacement: np.ndarray, *, scalar: np.ndarray | None = None
) -> pv.PolyData:
    surface = basis.skin.copy(deep=True)
    surface.points = np.asarray(surface.points) + displacement[basis.skin_mesh_ids]
    if scalar is not None:
        surface.point_data["TargetNormalResidualMM"] = 1.0e3 * scalar
    return surface


def _render_terminal_views(
    path: Path,
    *,
    basis: MetricBasis,
    render: RenderBasis,
    continuation: ReplayArtifact,
    direct: ReplayArtifact,
    residual: bool,
) -> None:
    target_disp = basis.target
    columns = (
        ("target", target_disp, _recompute_metrics(basis, target_disp)),
        (
            "exact baseline",
            basis.baseline_displacement,
            _recompute_metrics(basis, basis.baseline_displacement),
        ),
        ("c020 continuation", continuation.displacement, continuation.metrics),
        ("c020 direct", direct.displacement, direct.metrics),
    )
    scalar_values = [
        np.asarray(metrics["field/residual_normal_m"]) for _, _, metrics in columns
    ]
    residual_limit_mm = max(
        0.25,
        1.0e3 * float(np.quantile(np.abs(np.concatenate(scalar_values[1:])), 0.99)),
    )
    views = (
        ("front", np.asarray((0.0, 0.0, 1.0)), render.face_focus, render.face_scale),
        (
            "30 degree",
            np.asarray((math.sin(math.radians(30)), 0.0, math.cos(math.radians(30)))),
            render.face_focus,
            render.face_scale,
        ),
        ("mouth", np.asarray((0.0, 0.0, 1.0)), render.mouth_focus, render.mouth_scale),
        (
            "eye-cheek (+x)",
            np.asarray((0.0, 0.0, 1.0)),
            render.eye_focus,
            render.eye_scale,
        ),
    )
    plotter = pv.Plotter(
        shape=(len(views), len(columns)),
        off_screen=True,
        window_size=(560 * len(columns), 1600),
        lighting="light kit",
        border=False,
    )
    plotter.set_background("white")
    for row_id, (view_name, direction, focus, scale) in enumerate(views):
        for column_id, (label, displacement, metrics) in enumerate(columns):
            plotter.subplot(row_id, column_id)
            scalar = np.asarray(metrics["field/residual_normal_m"])
            surface = _deformed_skin(
                basis, displacement, scalar=scalar if residual else None
            )
            if residual:
                plotter.add_mesh(
                    surface,
                    scalars="TargetNormalResidualMM",
                    cmap="RdBu_r",
                    clim=(-residual_limit_mm, residual_limit_mm),
                    smooth_shading=False,
                    show_edges=False,
                    show_scalar_bar=False,
                )
                suffix = f"normal residual, shared +/-{residual_limit_mm:.2f} mm"
            else:
                plotter.add_mesh(
                    surface,
                    color="#d8b49c",
                    smooth_shading=True,
                    specular=0.15,
                    show_edges=False,
                )
                suffix = (
                    f"err={metrics['target/error_rms_mm']:.3f} mm | "
                    f"dih={metrics['bumpiness/contraction_target_relative_dihedral_rms_deg']:.2f} deg | "
                    f"nLap={1e3 * metrics['bumpiness/residual_normal_laplacian_rms_m']:.3f} mm"
                )
            plotter.add_text(
                f"{view_name} | {label}\n{suffix}",
                position="upper_left",
                font_size=8,
                color="black",
            )
            plotter.enable_parallel_projection()
            camera_focus = np.asarray(focus)
            plotter.camera.position = tuple(camera_focus + 0.30 * direction)
            plotter.camera.focal_point = tuple(camera_focus)
            plotter.camera.up = (0.0, 1.0, 0.0)
            plotter.camera.parallel_scale = float(scale)
    plotter.screenshot(path)
    plotter.close()


def _plot_trajectories(
    path: Path,
    *,
    rows: list[ReplayArtifact],
    baseline: dict[str, Any],
    direct: ReplayArtifact,
) -> None:
    continuation = sorted(
        (row for row in rows if row.path_kind == "continuation"),
        key=lambda row: row.alpha,
    )
    alpha = np.asarray([row.alpha for row in continuation])
    specs = (
        ("target/error_rms_fraction_of_target", "target error / target RMS", 1.0),
        (
            "target/face_target_area_weighted_error_rms_m",
            "target-area-weighted face error (mm)",
            1.0e3,
        ),
        (
            "bumpiness/contraction_target_relative_dihedral_rms_deg",
            "contraction target-relative dihedral RMS (deg)",
            1.0,
        ),
        (
            "bumpiness/residual_normal_laplacian_rms_m",
            "normal-residual Laplacian RMS (mm)",
            1.0e3,
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for axis, (key, label, scale) in zip(axes.flat, specs, strict=True):
        values = scale * np.asarray([row.metrics[key] for row in continuation])
        axis.plot(alpha, values, "o-", color="#2b6cb0", label="c=.02 continuation")
        axis.scatter(
            [1.0],
            [scale * direct.metrics[key]],
            marker="x",
            s=80,
            color="#c53030",
            label="c=.02 direct",
        )
        axis.axhline(
            scale * baseline[key],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="exact baseline",
        )
        axis.set_xlabel("prestrain continuation alpha")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Fixed muscle activation: raw-area + 2% tightening replay\n"
        "alpha scales log natural-area prestrain; no inverse optimization"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_quality(
    path: Path, rows: list[ReplayArtifact], baseline: dict[str, Any]
) -> None:
    continuation = sorted(
        (row for row in rows if row.path_kind == "continuation"),
        key=lambda row: row.alpha,
    )
    alpha = np.asarray([row.alpha for row in continuation])
    specs = (
        ("quality/inverted_tets", "inverted tetrahedra"),
        ("quality/skin_folded_triangles", "folded IsFace triangles"),
        ("quality/detF_q001", "det(F) q0.1%"),
        ("quality/skin_signed_normal_ratio_q001", "signed skin ratio q0.1%"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, specs, strict=True):
        values = np.asarray([row.metrics[key] for row in continuation])
        axis.plot(alpha, values, "o-", color="#805ad5")
        axis.axhline(baseline[key], color="black", linestyle="--", linewidth=1.0)
        axis.set_xlabel("prestrain continuation alpha")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    fig.suptitle("Quality diagnostics are advisory and require matched visual review")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _json_ready(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }


def _write_csv(path: Path, artifacts: list[ReplayArtifact], basis: MetricBasis) -> None:
    rows = [
        {
            "case_id": artifact.case_id,
            "path_kind": artifact.path_kind,
            "alpha": artifact.alpha,
            **_json_ready(artifact.metrics),
            **{
                f"delta_vs_baseline/{key}": value
                for key, value in _displacement_delta(
                    basis, artifact.displacement, basis.baseline_displacement
                ).items()
            },
        }
        for artifact in artifacts
    ]
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    *,
    baseline: dict[str, Any],
    continuation: ReplayArtifact,
    direct: ReplayArtifact,
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Fixed-activation prestrain replay analysis",
        "",
        "This is a fixed-muscle-activation forward diagnostic. It is not an inverse",
        "result and does not authorize an inverse experiment.",
        "",
        "## Terminal comparison",
        "",
        "| checkpoint | target RMS (mm) | error/target | dihedral RMS (deg) | normal Laplacian (mm) | inverted tets | folded skin triangles |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in (
        ("exact baseline", baseline),
        ("c020 continuation", continuation.metrics),
        ("c020 direct", direct.metrics),
    ):
        lines.append(
            f"| {label} | {metrics['target/error_rms_mm']:.6g} | "
            f"{metrics['target/error_rms_fraction_of_target']:.6g} | "
            f"{metrics['bumpiness/contraction_target_relative_dihedral_rms_deg']:.6g} | "
            f"{1e3 * metrics['bumpiness/residual_normal_laplacian_rms_m']:.6g} | "
            f"{metrics['quality/inverted_tets']} | "
            f"{metrics['quality/skin_folded_triangles']} |"
        )
    lines.extend(
        [
            "",
            "## Advisory decision",
            "",
            f"- Outcome: `{decision['outcome']}`",
            f"- Reason: {decision['reason']}",
            f"- Meaningful smoothing: {decision['meaningful_smoothing']}",
            f"- Fidelity acceptable: {decision['fidelity_acceptable']}",
            f"- Alpha-0 replay stable: {decision['alpha0_stable']}",
            f"- Alpha-0 roughness stable: {decision['alpha0_roughness_stable']}",
            f"- Continuation/direct branch stable: {decision['branch_stable']}",
            "- The effect-size thresholds are advisory deterministic engineering",
            "  thresholds, not statistical claims.",
            "- Quality counts and matched fixed views require human visual review;",
            "  small imperceptible inversions or folds are not automatic vetoes.",
            "- A c=.05 forward probe requires separate explicit approval. This analyzer",
            "  cannot launch it.",
            "",
            "## Quality warnings",
            "",
        ]
    )
    warnings_by_branch = decision["quality_warnings"]
    warnings = [
        f"{branch}: {warning}"
        for branch, branch_warnings in warnings_by_branch.items()
        for warning in branch_warnings
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append(
            "- No new numerical quality warning relative to the exact baseline."
        )
    lines.extend(
        [
            "",
            "## Required fixed-view review",
            "",
            "Record whether smoothing is visible in each matched geometry and normal-",
            "residual view, and separately record any new visible artifact:",
            "",
            "- [ ] front smoothing visible",
            "- [ ] 30 degree smoothing visible",
            "- [ ] mouth smoothing visible",
            "- [ ] eye-cheek (+x) smoothing visible",
            "- [ ] no new visible artifact in any fixed view",
            "",
            "Final resolution rules:",
            "",
            "- c=.02 is sufficient only when the quantitative rule passes, smoothing",
            "  is visible in at least three of four views, and no new artifact appears.",
            "- A stable, fit-safe but quantitatively or visibly weak c=.02 result",
            "  conditionally requires a separately approved c=.05 forward probe.",
            "- Any replay, branch, solver, fixed-boundary, fit, or visible-artifact",
            "  failure stops escalation; do not run c=.05.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _require_outputs_absent(cfg: Config) -> None:
    paths = (
        cfg.output_json,
        cfg.output_csv,
        cfg.output_trajectory_plot,
        cfg.output_quality_plot,
        cfg.output_geometry_views,
        cfg.output_residual_views,
        cfg.output_report,
    )
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        msg = f"refusing to overwrite existing analysis outputs: {existing}"
        raise FileExistsError(msg)


def main(cfg: Config) -> None:
    _validate_config(cfg)
    _require_outputs_absent(cfg)
    baseline_identities = {
        "result": _require_identity(
            cfg.input_baseline_result,
            expected_size=BASELINE_RESULT_SIZE_BYTES,
            expected_sha256=BASELINE_RESULT_SHA256,
            name="corrected exact baseline result",
        ),
        "summary": _require_identity(
            cfg.input_baseline_summary,
            expected_size=BASELINE_SUMMARY_SIZE_BYTES,
            expected_sha256=BASELINE_SUMMARY_SHA256,
            name="corrected exact baseline summary",
        ),
        "skin": _require_identity(
            cfg.input_baseline_skin,
            expected_size=BASELINE_SKIN_SIZE_BYTES,
            expected_sha256=BASELINE_SKIN_SHA256,
            name="corrected IsFace skin",
        ),
        "canonical_metric_skin": _require_identity(
            cfg.input_metric_skin,
            expected_size=METRIC_SKIN_SIZE_BYTES,
            expected_sha256=METRIC_SKIN_SHA256,
            name="canonical primary-Bumpy metric skin",
        ),
    }
    producer_summary = _read_json(cfg.input_summary)
    producer_contract = _validate_producer_contract(producer_summary)

    baseline_mesh = pv.read(cfg.input_baseline_result)
    baseline_skin = pv.read(cfg.input_baseline_skin)
    metric_skin = pv.read(cfg.input_metric_skin)
    if (
        not isinstance(baseline_mesh, pv.UnstructuredGrid)
        or not isinstance(baseline_skin, pv.PolyData)
        or not isinstance(metric_skin, pv.PolyData)
    ):
        msg = "baseline result/skin dataset types changed"
        raise TypeError(msg)
    basis = _build_metric_basis(baseline_mesh, baseline_skin, metric_skin)
    baseline_metrics = _recompute_metrics(basis, basis.baseline_displacement)
    for key, expected in (
        ("target/error_rms_m", EXPECTED_BASELINE_TARGET_ERROR_RMS_M),
        (
            "bumpiness/contraction_target_relative_dihedral_rms_deg",
            EXPECTED_BASELINE_DIHEDRAL_RMS_DEG,
        ),
        (
            "bumpiness/residual_normal_laplacian_rms_m",
            EXPECTED_BASELINE_NORMAL_LAPLACIAN_RMS_M,
        ),
    ):
        if not math.isclose(
            float(baseline_metrics[key]), expected, rel_tol=1.0e-10, abs_tol=1.0e-12
        ):
            msg = f"canonical primary metric changed for {key}: {baseline_metrics[key]}"
            raise ValueError(msg)

    output_root = PRODUCER_OUTPUT_ROOT
    rows = _producer_rows(producer_summary)
    artifacts = [
        _load_artifact(row, basis=basis, output_root=output_root) for row in rows
    ]
    observed = [(row.path_kind, row.alpha) for row in artifacts]
    if len(observed) != len(EXPECTED_PATHS) or any(
        left[0] != right[0] or not math.isclose(left[1], right[1], abs_tol=1.0e-15)
        for left, right in zip(observed, EXPECTED_PATHS, strict=True)
    ):
        msg = f"replay order changed: expected {EXPECTED_PATHS}, got {observed}"
        raise ValueError(msg)
    observed_case_ids = tuple(row.case_id for row in artifacts)
    if observed_case_ids != EXPECTED_CASE_IDS:
        msg = f"producer case IDs changed: {observed_case_ids}"
        raise ValueError(msg)

    alpha0 = artifacts[0]
    alpha0_readback = _displacement_delta(
        basis, alpha0.displacement, basis.baseline_displacement
    )
    for producer_key, delta_key in (
        (
            "replay/smile_loss_mask_delta_fraction_of_target",
            "loss_mask_fraction_of_target",
        ),
        ("replay/isface_delta_fraction_of_target", "isface_fraction_of_target"),
        ("replay/smile_loss_mask_delta_rms_m", "loss_mask_rms_m"),
        ("replay/isface_delta_rms_m", "isface_rms_m"),
    ):
        _require_close(
            alpha0_readback[delta_key],
            alpha0.row[producer_key],
            context=f"alpha0 {producer_key}",
        )
    if not bool(alpha0.row["replay/gate"]):
        msg = "producer alpha0 replay gate is false"
        raise ValueError(msg)
    continuation = next(
        row
        for row in artifacts
        if row.path_kind == "continuation" and math.isclose(row.alpha, 1.0)
    )
    direct = next(row for row in artifacts if row.path_kind == "direct")
    branch_readback = _validate_branch_summary(
        producer_summary,
        basis=basis,
        left=continuation.displacement,
        right=direct.displacement,
    )
    decision = _decision(
        basis=basis,
        baseline=baseline_metrics,
        continuation=continuation,
        direct=direct,
        alpha0=alpha0,
    )

    _write_csv(cfg.output_csv, artifacts, basis)
    _plot_trajectories(
        cfg.output_trajectory_plot,
        rows=artifacts,
        baseline=baseline_metrics,
        direct=direct,
    )
    _plot_quality(cfg.output_quality_plot, artifacts, baseline_metrics)
    render = _build_render_basis(basis)
    _render_terminal_views(
        cfg.output_geometry_views,
        basis=basis,
        render=render,
        continuation=continuation,
        direct=direct,
        residual=False,
    )
    _render_terminal_views(
        cfg.output_residual_views,
        basis=basis,
        render=render,
        continuation=continuation,
        direct=direct,
        residual=True,
    )
    _write_report(
        cfg.output_report,
        baseline=baseline_metrics,
        continuation=continuation,
        direct=direct,
        decision=decision,
    )

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fixed-activation-prestrain-replay-static-readback-analysis",
        "design": INPUT_DESIGN,
        "complete": True,
        "execution_contract": {
            "readback_and_render_only": True,
            "forward_started": False,
            "inverse_started": False,
            "adjoint_started": False,
            "muscle_activation_optimized": False,
            "c050_started": False,
        },
        "interpretation": (
            "same fixed optimized muscle activation in every checkpoint; alpha only "
            "scales the prescribed skin prestrain in log natural-area space; this "
            "is a forward mechanism replay, not an inverse result"
        ),
        "producer_contract": producer_contract,
        "input_provenance": baseline_identities,
        "metric_contract": {
            "primary_bumpiness": list(PRIMARY_BUMPINESS),
            "target_fidelity": list(FIDELITY_METRICS),
            "contraction_roi": (
                "canonical ContractionPrestrainMask from the pinned 2026-08-17 "
                "metric skin, mapped by sorted GlobalPointId triangle keys"
            ),
            "contraction_mask_sha256": basis.contraction_mask_sha256,
            "all_metrics_recomputed_from_saved_result_meshes": True,
            "producer_scalar_fields_trusted_without_readback": False,
        },
        "protocol": {
            "case_id": CASE_ID,
            "constant_linear_tightening": EXPECTED_DOSE,
            "raw_area_ratio_floor": EXPECTED_FLOOR,
            "continuation_alphas": list(CONTINUATION_ALPHAS),
            "alpha_formula": (
                "rho_full=(1-c)^2*clip(R,0.5,1); "
                "rho_alpha=rho_full**alpha; Ainv_diag=rho_alpha**(-1/2)-1"
            ),
            "alpha0_reference": "numerical replay compared to exact corrected baseline",
            "alpha1_direct": "seeded from the exact corrected baseline displacement",
        },
        "baseline": _json_ready(baseline_metrics),
        "trajectory": [
            {
                "case_id": row.case_id,
                "path_kind": row.path_kind,
                "alpha": row.alpha,
                "result_path": str(row.result_path),
                "skin_path": str(row.skin_path),
                "metrics": _json_ready(row.metrics),
                "delta_vs_exact_baseline": _displacement_delta(
                    basis, row.displacement, basis.baseline_displacement
                ),
            }
            for row in artifacts
        ],
        "advisory_decision": decision,
        "branch_readback": branch_readback,
        "inverse_eligibility": {
            "eligible": False,
            "status": "not-assessed-by-this-forward-only-analysis",
            "required_next_action": (
                "human review of both fixed-view sheets; if and only if the advisory "
                "decision requests c=.05, obtain explicit approval for a separate "
                "c=.05 fixed-activation forward producer/run"
            ),
        },
        "outputs": {
            "csv": str(cfg.output_csv),
            "trajectory_plot": str(cfg.output_trajectory_plot),
            "quality_plot": str(cfg.output_quality_plot),
            "geometry_views": str(cfg.output_geometry_views),
            "residual_views": str(cfg.output_residual_views),
            "report": str(cfg.output_report),
        },
    }
    cfg.output_json.write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Advisory decision: %s", decision["outcome"])
    logger.info("Wrote %s", cfg.output_json)


if __name__ == "__main__":
    cherries.main(main)
