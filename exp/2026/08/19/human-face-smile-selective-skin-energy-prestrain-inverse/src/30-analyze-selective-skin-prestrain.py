from __future__ import annotations

# This audit script intentionally raises rich, contextual contract failures inline.
# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonExecutionModel import (
    vtkStreamingDemandDrivenPipeline as StreamingPipeline,
)

from liblaf import cherries

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
INPUT_SCHEMA_VERSION = 1
INPUT_DESIGN = "corrected-isface-selective-e000-c020-three-case-inverse"
EXPECTED_FRAMES = 41
TERMINAL_STEP = 40
CASE_ORDER = ("H0P0", "H0P1", "H1P1", "H1P0")
FORMAL_CASE_ORDER = CASE_ORDER[1:]
DISPLAY_NAMES = {
    "H0P0": "homogeneous E, no prestrain",
    "H0P1": "homogeneous E + c020 prestrain",
    "H1P1": "expansion E=0 + c020 prestrain",
    "H1P0": "expansion E=0, no prestrain",
}
STEMS = {
    "H0P0": (
        "20-human-face-smile-skin-no-prestrain-lr3-corrected-isface-e0200-p000-screen"
    ),
    "H0P1": "20-h0p1",
    "H1P1": "20-h1p1",
    "H1P0": "20-h1p0",
}

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_DIR = GROUP_DIR / "data"
BASELINE_DIR = REPO_ROOT / "exp/2026/08/18/human-face-smile-plane-stress-skin/data"
BASELINE_STEM = STEMS["H0P0"]
BASELINE_SKIN = BASELINE_DIR / "10-corrected-baseline/skin-isface-e0200-p000.vtp"
METRIC_SKIN = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/"
    "data/10-material-candidates/skin-e100-p000.vtp"
)
PRODUCER = Path(__file__).with_name("20-inverse-selective-skin-prestrain.py")
AGGREGATE = DATA_DIR / "20-selective-skin-prestrain-inverse-summary-final.json"

OUTPUT_JSON = DATA_DIR / "30-selective-skin-prestrain-analysis.json"
OUTPUT_CSV = DATA_DIR / "30-selective-skin-prestrain-trajectories.csv"
OUTPUT_CHECKPOINTS = DATA_DIR / "30-selective-skin-prestrain-checkpoints.md"
OUTPUT_TRAJECTORIES = DATA_DIR / "30-selective-skin-prestrain-trajectories.png"
OUTPUT_PARETO = DATA_DIR / "30-selective-skin-prestrain-pareto.png"
PARAVIEW_INPUT_DIR = DATA_DIR / "30-paraview-inputs"

# Execution is deliberately impossible until the completed inverse producer and this
# analyzer have both passed static review. Formal execution must leave this blocker
# in place until the producer and canonical final aggregate placeholders below have
# been replaced with identities reviewed after the inverse batch completes.
ANALYZER_EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True
EXPECTED_PRODUCER_SHA256 = (
    "deece64950f8bf21984fa0ba970d2e1f0e0f71e23db483919bd59de47052456b"
)
EXPECTED_FORMAL_AGGREGATE_SIZE_BYTES: int | None = 387_036
EXPECTED_FORMAL_AGGREGATE_SHA256: str | None = (
    "cf533bb16f481d75587531dfcd5aa21ed1065ed02539ea3ff0290e94d6cd2de6"
)

EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_ACTIVE_TETS = 288_235
EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_FIXED_VERTICES = 33_636
EXPECTED_CUT_VERTICES = 6_980
EXPECTED_TARGET_RMS_M = 0.005310139062299789
EXPECTED_BASELINE_TERMINAL_RMS_M = 0.0027209482247538275
BASELINE_FIDELITY_GAP_TOLERANCE = 0.01

EXPECTED_TRIANGLE_KEYS_SHA256 = (
    "dca8d77662f49b54250657424cfc29a3b438437081f83d65f7e108f312da2310"
)
EXPECTED_MAPPED_DRIVER_SHA256 = (
    "13458107ceef23ecc144340101574f3fb4f2e157f90a9251434e7b09a86a66c3"
)
EXPECTED_RAW_AREA_RATIO_SHA256 = (
    "da98a6f48694eed30b1683ccf5a1f02fd67c87393b30003d07c52a7fa25bc606"
)
EXPECTED_RAW_CONTRACTION_TRIANGLES = 13_159
EXPECTED_RAW_CONTRACTION_MASK_SHA256 = (
    "276296bf0dab911ded6d6609f5288c8f4560cb4d92211188aba11d30222ddeab"
)

BASELINE_IDENTITIES = {
    "summary": (
        126_540,
        "575ebcbd7152a256917c2a11a9bf9bef9046f00f9831e18adc86d41645be1856",
    ),
    "trace": (
        91_767,
        "a0f83957c832a119f6f031fb78a46fe52060d3b190a2ba0a1265f000c5d8cde3",
    ),
    "history": (
        2_066_073_161,
        "6e29d7b205e7901681942f0d413b091c5e4bce003ec4d789c2d7f69ded430d24",
    ),
    "result": (
        147_657_021,
        "c6a0b183675ffb3ec537c1153544b041acd7aa0fdd5216c0cf9a50022d52b0a4",
    ),
    "target": (
        84_419_492,
        "89ec02dfd87330f7dc1d303639893f7698ef2e6098480c4e39fa2ad94240206c",
    ),
    "skin": (
        1_138_550,
        "4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f",
    ),
}
METRIC_SKIN_IDENTITY = (
    38_742_137,
    "ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f",
)

REQUIRED_POINT_ARRAYS = {
    "GlobalPointId",
    "IsFixed",
    "ArtificialCutIncident",
    "FixedMask",
    "FixedValue",
    "Displacement",
    "TargetDisplacement",
    "LossMask",
}
REQUIRED_CELL_ARRAYS = {
    "ActivationMask",
    "ActivationInv",
    "RecoveredActivationInv",
}


class Config(cherries.BaseConfig):
    input_aggregate: Path = cherries.input(AGGREGATE)
    input_baseline_history: Path = cherries.input(
        BASELINE_DIR / f"{BASELINE_STEM}-steps.vtkhdf"
    )
    input_baseline_skin: Path = cherries.input(BASELINE_SKIN)
    output_json: Path = cherries.output(OUTPUT_JSON, mkdir=True)
    output_csv: Path = cherries.output(OUTPUT_CSV, mkdir=True)
    output_checkpoints: Path = cherries.output(OUTPUT_CHECKPOINTS, mkdir=True)
    output_trajectories: Path = cherries.output(OUTPUT_TRAJECTORIES, mkdir=True)
    output_pareto: Path = cherries.output(OUTPUT_PARETO, mkdir=True)


@dataclass(frozen=True)
class FileIdentity:
    size_bytes: int
    sha256: str


@dataclass
class TemporalHistory:
    case_id: str
    path: Path
    reader: Any
    times: np.ndarray

    @classmethod
    def open(cls, case_id: str, path: Path) -> TemporalHistory:
        reader = pv.get_reader(path)
        vtk_reader = reader.reader
        vtk_reader.UpdateInformation()
        info = vtk_reader.GetOutputInformation(0)
        key = StreamingPipeline.TIME_STEPS()
        if not info.Has(key):
            raise ValueError(f"{case_id} history has no TIME_STEPS: {path}")
        times = np.asarray(
            [info.Get(key, i) for i in range(info.Length(key))], dtype=np.float64
        )
        expected = np.arange(EXPECTED_FRAMES, dtype=np.float64)
        if not np.array_equal(times, expected):
            raise ValueError(f"{case_id} history TIME_STEPS are not exactly 0..40")
        return cls(case_id=case_id, path=path, reader=reader, times=times)

    def frame(self, step: int, *, deep: bool = False) -> pv.UnstructuredGrid:
        if not 0 <= step < EXPECTED_FRAMES:
            raise IndexError(f"invalid {self.case_id} history step {step}")
        vtk_reader = self.reader.reader
        vtk_reader.UpdateTimeStep(float(self.times[step]))
        mesh = pv.wrap(vtk_reader.GetOutputDataObject(0))
        if not isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.cast_to_unstructured_grid()
        return mesh.copy(deep=True) if deep else mesh


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    summary_path: Path
    canonical_summary_path: Path
    trace_path: Path
    history_path: Path
    result_path: Path
    target_path: Path
    skin_path: Path
    identities: dict[str, FileIdentity]
    summary: dict[str, Any]
    trace: list[dict[str, Any]]
    history: TemporalHistory


@dataclass(frozen=True)
class MetricBasis:
    base_points: np.ndarray
    cells: np.ndarray
    celltypes: np.ndarray
    global_ids: np.ndarray
    target: np.ndarray
    loss_mask: np.ndarray
    target_rms: float
    activation_mask: np.ndarray
    is_fixed: np.ndarray
    fixed_mask: np.ndarray
    fixed_value: np.ndarray
    cut_mask: np.ndarray
    skin: pv.PolyData
    skin_mesh_ids: np.ndarray
    triangles: np.ndarray
    edges: np.ndarray
    rest_area: np.ndarray
    target_area: np.ndarray
    target_normals: np.ndarray
    contraction_tri_0: np.ndarray
    contraction_tri_1: np.ndarray
    contraction_target_dihedral: np.ndarray
    contraction_edge_weight: np.ndarray
    raw_contraction_domain: dict[str, Any]
    tets: np.ndarray
    rest_six_volume: np.ndarray
    rest_area_vectors: np.ndarray
    rest_area_vector_norm: np.ndarray
    render_views: dict[str, dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    if array.dtype.kind == "f":
        array = array.astype("<f8", copy=False)
    elif array.dtype.kind in {"i", "u"}:
        array = array.astype("<i8", copy=False)
    elif array.dtype.kind == "b":
        array = array.astype("u1", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _identity(path: Path) -> FileIdentity:
    if not path.is_file():
        raise FileNotFoundError(path)
    return FileIdentity(path.stat().st_size, _sha256(path))


def _require_identity(path: Path, expected: FileIdentity, *, label: str) -> None:
    actual = _identity(path)
    if actual != expected:
        raise ValueError(f"{label} identity changed: {actual} != {expected}")


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    rows = [
        json.loads(line, parse_constant=reject_constant)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != EXPECTED_FRAMES or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} is not a complete 41-row JSONL trace")
    if [int(row["step"]) for row in rows] != list(range(EXPECTED_FRAMES)):
        raise ValueError(f"{path} trace steps are not exactly 0..40")
    return rows


def _resolve_under_data(value: Any) -> Path:
    path = Path(str(value)).resolve()
    if DATA_DIR.resolve() not in path.parents:
        raise ValueError(f"formal artifact escapes experiment data directory: {path}")
    return path


def _baseline_paths() -> dict[str, Path]:
    return {
        "summary": BASELINE_DIR / f"{BASELINE_STEM}-summary-final.json",
        "trace": BASELINE_DIR / f"{BASELINE_STEM}-trace.jsonl",
        "history": BASELINE_DIR / f"{BASELINE_STEM}-steps.vtkhdf",
        "result": BASELINE_DIR / f"{BASELINE_STEM}.vtu",
        "target": BASELINE_DIR / f"{BASELINE_STEM}-target.vtu",
        "skin": BASELINE_SKIN,
    }


def _validate_config(cfg: Config) -> None:
    expected = {
        "input_aggregate": AGGREGATE,
        "input_baseline_history": _baseline_paths()["history"],
        "input_baseline_skin": BASELINE_SKIN,
        "output_json": OUTPUT_JSON,
        "output_csv": OUTPUT_CSV,
        "output_checkpoints": OUTPUT_CHECKPOINTS,
        "output_trajectories": OUTPUT_TRAJECTORIES,
        "output_pareto": OUTPUT_PARETO,
    }
    changed = [
        name
        for name, path in expected.items()
        if Path(getattr(cfg, name)).resolve() != path.resolve()
    ]
    if changed:
        raise ValueError(f"analyzer paths cannot be overridden: {changed}")
    outputs = (*expected.values(), PARAVIEW_INPUT_DIR)
    stale = [
        str(path)
        for path in outputs
        if path.exists()
        and path not in {AGGREGATE, _baseline_paths()["history"], BASELINE_SKIN}
    ]
    if stale:
        raise FileExistsError(f"refusing stale analyzer outputs: {stale}")
    if not ANALYZER_EXECUTION_APPROVED_AFTER_STATIC_REVIEW:
        raise RuntimeError(
            "NO-GO: analyzer awaits completed inverse artifacts and isolated static approval"
        )
    if (
        EXPECTED_FORMAL_AGGREGATE_SIZE_BYTES is None
        or EXPECTED_FORMAL_AGGREGATE_SHA256 is None
    ):
        raise RuntimeError(
            "NO-GO: fill the reviewed canonical final aggregate size/SHA256 first"
        )
    if _sha256(PRODUCER) != EXPECTED_PRODUCER_SHA256:
        raise ValueError(
            "inverse producer source is not the reviewed executable identity"
        )
    _require_identity(
        AGGREGATE,
        FileIdentity(
            EXPECTED_FORMAL_AGGREGATE_SIZE_BYTES,
            EXPECTED_FORMAL_AGGREGATE_SHA256,
        ),
        label="reviewed canonical final formal aggregate",
    )


def _load_cases(aggregate: dict[str, Any]) -> list[CaseInput]:
    if int(aggregate.get("schema_version", -1)) != INPUT_SCHEMA_VERSION:
        raise ValueError("formal inverse aggregate schema changed")
    if aggregate.get("design") != INPUT_DESIGN:
        raise ValueError("formal inverse aggregate design changed")
    if not bool(aggregate.get("complete")):
        raise ValueError("formal inverse aggregate is not complete")
    if aggregate.get("case_order") != list(FORMAL_CASE_ORDER):
        raise ValueError("formal inverse case order changed")
    expected_execution = {
        "stage": "formal",
        "case_order": list(FORMAL_CASE_ORDER),
        "sequential": True,
        "optimizer": "Adam",
        "learning_rate": 0.3,
        "optimizer_updates_per_case": TERMINAL_STEP,
        "evaluations_per_case": EXPECTED_FRAMES,
        "fresh_zero_activation_per_case": True,
        "fresh_zero_displacement_per_case": True,
        "independent_forward_and_optimizer_per_case": True,
        "smoke_step0_grad_norm_must_exceed": 1.0e-12,
    }
    if aggregate.get("execution_contract") != expected_execution:
        raise ValueError("formal inverse execution contract changed")
    expected_artifact = {
        "case_stems": {case: STEMS[case] for case in FORMAL_CASE_ORDER},
        "history_format": "VTKHDFTemporalUnstructuredGrid",
        "history_time_steps": list(range(EXPECTED_FRAMES)),
        "trace_steps": list(range(EXPECTED_FRAMES)),
        "result_state": "best saved inverse state",
        "history_state": "every evaluated inverse state",
        "case_row_identity_keys": [
            f"artifact/{name}_{suffix}"
            for name in ("summary", "trace", "history", "result", "target", "skin")
            for suffix in ("path", "size_bytes", "sha256")
        ],
    }
    if aggregate.get("artifact_contract") != expected_artifact:
        raise ValueError("formal inverse artifact contract changed")
    if aggregate.get("hard_failures") != []:
        raise ValueError("formal inverse aggregate contains hard failures")
    post_checks = aggregate.get("post_run_identity_checks")
    if (
        not isinstance(post_checks, dict)
        or not post_checks
        or not all(value is True for value in post_checks.values())
    ):
        raise ValueError("formal inverse post-run identity checks are not all exact")
    rows = aggregate.get("cases")
    if not isinstance(rows, list) or len(rows) != len(FORMAL_CASE_ORDER):
        raise ValueError("aggregate cases must be the three formal cases")
    by_case = {str(row.get("case_id")): row for row in rows}
    if set(by_case) != set(FORMAL_CASE_ORDER):
        raise ValueError(f"unexpected formal cases: {sorted(by_case)}")
    producer_identity = aggregate.get("producer_identity")
    expected_producer_identity = {
        "path": str(PRODUCER.resolve()),
        "size_bytes": PRODUCER.stat().st_size,
        "sha256": EXPECTED_PRODUCER_SHA256,
        "unchanged_through_all_runs": True,
    }
    if producer_identity != expected_producer_identity:
        raise ValueError("aggregate producer identity is not the reviewed source")

    cases: list[CaseInput] = []
    baseline_paths = _baseline_paths()
    baseline_ids = {
        name: FileIdentity(*BASELINE_IDENTITIES[name]) for name in BASELINE_IDENTITIES
    }
    for name, path in baseline_paths.items():
        _require_identity(path, baseline_ids[name], label=f"H0P0 {name}")
    _require_identity(
        METRIC_SKIN, FileIdentity(*METRIC_SKIN_IDENTITY), label="metric skin"
    )
    baseline_summary = _read_json(baseline_paths["summary"])
    cases.append(
        CaseInput(
            case_id="H0P0",
            **{f"{name}_path": path for name, path in baseline_paths.items()},
            canonical_summary_path=baseline_paths["summary"],
            identities={
                **baseline_ids,
                "canonical_summary": baseline_ids["summary"],
            },
            summary=baseline_summary,
            trace=_read_jsonl(baseline_paths["trace"]),
            history=TemporalHistory.open("H0P0", baseline_paths["history"]),
        )
    )

    for case_id in FORMAL_CASE_ORDER:
        row = by_case[case_id]
        paths: dict[str, Path] = {}
        identities: dict[str, FileIdentity] = {}
        for name in ("summary", "trace", "history", "result", "target", "skin"):
            for key in (
                f"artifact/{name}_path",
                f"artifact/{name}_size_bytes",
                f"artifact/{name}_sha256",
            ):
                if key not in row:
                    raise KeyError(f"{case_id} aggregate row misses {key}")
            path = _resolve_under_data(row[f"artifact/{name}_path"])
            expected_name = {
                "summary": f"{STEMS[case_id]}-summary.json",
                "trace": f"{STEMS[case_id]}-trace.jsonl",
                "history": f"{STEMS[case_id]}-steps.vtkhdf",
                "result": f"{STEMS[case_id]}.vtu",
                "target": f"{STEMS[case_id]}-target.vtu",
            }.get(name)
            if expected_name is not None and path.name != expected_name:
                raise ValueError(f"{case_id} {name} basename changed: {path.name}")
            identity = FileIdentity(
                int(row[f"artifact/{name}_size_bytes"]),
                str(row[f"artifact/{name}_sha256"]),
            )
            _require_identity(path, identity, label=f"{case_id} {name}")
            paths[name] = path
            identities[name] = identity
        canonical_keys = (
            "artifact/canonical_summary_path",
            "artifact/canonical_summary_size_bytes",
            "artifact/canonical_summary_sha256",
        )
        if any(key not in row for key in canonical_keys):
            raise KeyError(f"{case_id} aggregate row misses canonical summary identity")
        canonical_path = _resolve_under_data(row[canonical_keys[0]])
        if canonical_path.name != f"{STEMS[case_id]}-summary-final.json":
            raise ValueError(f"{case_id} canonical summary basename changed")
        canonical_identity = FileIdentity(
            int(row[canonical_keys[1]]), str(row[canonical_keys[2]])
        )
        _require_identity(
            canonical_path,
            canonical_identity,
            label=f"{case_id} canonical summary",
        )
        if canonical_identity != identities["summary"]:
            raise ValueError(f"{case_id} live and canonical summaries differ")
        identities["canonical_summary"] = canonical_identity
        summary = _read_json(paths["summary"])
        if str(summary.get("case_id", summary.get("case"))) != case_id:
            raise ValueError(f"{case_id} summary case identity changed")
        expected_summary = {
            "status": "ok",
            "baseline/completed": True,
            "baseline/evaluations": EXPECTED_FRAMES,
            "baseline/evaluations_expected": EXPECTED_FRAMES,
            "history/frames": EXPECTED_FRAMES,
            "history_frames": EXPECTED_FRAMES,
            "protocol/evaluations": EXPECTED_FRAMES,
            "protocol/optimizer_steps": TERMINAL_STEP,
            "protocol/fresh_zero_activation": True,
            "protocol/fresh_zero_displacement": True,
            "protocol/forward_initial_displacement_exact_zero": True,
            "cut_boundary/hard_fixed_is_ground_truth": False,
            "cut_boundary/configured_exact_zero": True,
            "cut_boundary/readback_exact_zero": True,
            "skin/domain": "all-vertex IsFace filtered PolyData",
            "skin/lame_conversion": (
                "thin-membrane plane-stress reduction: lambda = E * nu / "
                "(1 - nu**2); mu = E / (2 * (1 + nu))"
            ),
            "skin/koiter_energy_measure": "fixed original reference area",
        }
        changed = [
            key
            for key, expected in expected_summary.items()
            if summary.get(key) != expected
        ]
        if changed:
            raise ValueError(
                f"{case_id} summary execution/material contract changed: {changed}"
            )
        cases.append(
            CaseInput(
                case_id=case_id,
                **{f"{name}_path": path for name, path in paths.items()},
                canonical_summary_path=canonical_path,
                identities=identities,
                summary=summary,
                trace=_read_jsonl(paths["trace"]),
                history=TemporalHistory.open(case_id, paths["history"]),
            )
        )
    return cases


def _triangles(surface: pv.PolyData) -> np.ndarray:
    faces = np.asarray(surface.faces, dtype=np.int64)
    if faces.size != 4 * surface.n_cells:
        raise ValueError("skin is not packed triangles")
    faces = faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("skin contains a non-triangle cell")
    return faces[:, 1:].copy()


def _triangle_geometry(
    points: np.ndarray, tri: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.cross(
        points[tri[:, 1]] - points[tri[:, 0]], points[tri[:, 2]] - points[tri[:, 0]]
    )
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("surface has a non-finite or degenerate triangle")
    return vectors, 0.5 * norms, vectors / norms[:, None]


def _unique_edges(tri: np.ndarray) -> np.ndarray:
    edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _interior_edges(
    points: np.ndarray, tri: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    tri_ids = np.tile(np.arange(tri.shape[0], dtype=np.int64), 3)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, tri_ids = edges[order], tri_ids[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(np.diff(edges, axis=0), axis=1))]
    ends = np.r_[starts[1:], edges.shape[0]]
    keep = ends - starts == 2
    unique = edges[starts[keep]]
    lengths = np.linalg.norm(points[unique[:, 1]] - points[unique[:, 0]], axis=1)
    return tri_ids[starts[keep]], tri_ids[starts[keep] + 1], lengths


def _canonical_keys(surface: pv.PolyData) -> np.ndarray:
    ids = np.asarray(surface.point_data["GlobalPointId"], dtype=np.int64)
    return np.sort(ids[_triangles(surface)], axis=1)


def _raw_contraction_mask(
    metric_skin: pv.PolyData, skin: pv.PolyData
) -> tuple[np.ndarray, dict[str, Any]]:
    source_keys = _canonical_keys(metric_skin)
    keys = _canonical_keys(skin)
    if _array_sha256(keys) != EXPECTED_TRIANGLE_KEYS_SHA256:
        raise ValueError("canonical IsFace triangle identity changed")
    lookup = {tuple(row): index for index, row in enumerate(source_keys.tolist())}
    if len(lookup) != len(source_keys):
        raise ValueError("metric skin contains duplicate canonical triangles")
    try:
        mapped = np.asarray(
            [lookup[tuple(row)] for row in keys.tolist()], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("IsFace skin does not map into metric skin") from error
    if _array_sha256(mapped) != EXPECTED_MAPPED_DRIVER_SHA256:
        raise ValueError("canonical driver-cell mapping changed")
    if not np.all(
        np.asarray(metric_skin.cell_data["IsFaceTriangle"], dtype=bool)[mapped]
    ):
        raise ValueError("mapped metric triangles leave IsFace")
    raw_ratio = np.asarray(
        metric_skin.cell_data["TargetRestAreaRatio"], dtype=np.float64
    )[mapped]
    raw_ratio_hash = _array_sha256(raw_ratio)
    if raw_ratio_hash != EXPECTED_RAW_AREA_RATIO_SHA256:
        raise ValueError("mapped raw target/rest area-ratio field changed")
    if not np.isfinite(raw_ratio).all() or np.any(raw_ratio <= 0.0):
        raise ValueError("raw target/rest area ratio is non-finite or non-positive")
    contraction = raw_ratio < 1.0
    contraction_count = int(contraction.sum())
    contraction_hash = _array_sha256(contraction)
    if contraction_count != EXPECTED_RAW_CONTRACTION_TRIANGLES:
        raise ValueError(
            f"raw R<1 triangle count changed: {contraction_count} != "
            f"{EXPECTED_RAW_CONTRACTION_TRIANGLES}"
        )
    if contraction_hash != EXPECTED_RAW_CONTRACTION_MASK_SHA256:
        raise ValueError("raw R<1 triangle mask changed")
    return contraction, {
        "definition": "strict raw TargetRestAreaRatio < 1.0 on canonical IsFace triangles",
        "source_array": "TargetRestAreaRatio",
        "threshold": 1.0,
        "strict_less_than": True,
        "deadband": None,
        "cap": None,
        "diffusion": None,
        "triangle_count": contraction_count,
        "triangle_mask_sha256": contraction_hash,
        "raw_area_ratio_sha256": raw_ratio_hash,
    }


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
    points: np.ndarray, tri: np.ndarray, vectors: np.ndarray
) -> np.ndarray:
    normals = np.zeros_like(points)
    for local in range(3):
        np.add.at(normals, tri[:, local], vectors)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= np.finfo(np.float64).eps):
        raise ValueError("target skin has an undefined vertex normal")
    return normals / lengths[:, None]


def _bounds_camera(
    points: np.ndarray, *, padding: float = 1.12, aspect: float = 1.35
) -> tuple[list[float], float]:
    low, high = points.min(axis=0), points.max(axis=0)
    focus = 0.5 * (low + high)
    extent = high - low
    scale = padding * 0.5 * max(float(extent[1]), float(extent[0]) / aspect)
    return focus.tolist(), scale


def _render_views(
    mesh: pv.UnstructuredGrid, skin: pv.PolyData, skin_ids: np.ndarray
) -> dict[str, dict[str, Any]]:
    points = np.asarray(skin.points, dtype=np.float64)
    face_focus, face_scale = _bounds_camera(points)
    is_lip = np.asarray(mesh.point_data["IsLip"], dtype=bool)[skin_ids]
    mouth_focus, mouth_scale = _bounds_camera(points[is_lip], padding=1.25)
    names = tuple(str(value) for value in skin.field_data["GroupName"])
    group_ids = np.asarray(skin.point_data["GroupId"], dtype=np.int64)
    eyelid_names = {"EyelidTop", "EyelidBottom", "EyelidOuterTop", "EyelidOuterBottom"}
    eyelid_ids = [i for i, name in enumerate(names) if name in eyelid_names]
    eyelid = np.isin(group_ids, eyelid_ids)
    one_eye = eyelid & (points[:, 0] >= np.median(points[eyelid, 0]))
    eye_focus, _ = _bounds_camera(points[one_eye])
    eye_focus[1] -= 0.08 * float(np.ptp(points[:, 1]))
    return {
        "front": {
            "direction": [0.0, 0.0, 1.0],
            "focus": face_focus,
            "parallel_scale": face_scale,
        },
        "30-degree": {
            "direction": [0.5, 0.0, math.sqrt(3.0) / 2.0],
            "focus": face_focus,
            "parallel_scale": face_scale,
        },
        "mouth": {
            "direction": [0.0, 0.0, 1.0],
            "focus": mouth_focus,
            "parallel_scale": mouth_scale,
        },
        "eye-cheek+x": {
            "direction": [0.0, 0.0, 1.0],
            "focus": eye_focus,
            "parallel_scale": 0.24 * float(np.ptp(points[:, 1])),
        },
    }


def _build_basis(
    mesh: pv.UnstructuredGrid, skin: pv.PolyData, metric_skin: pv.PolyData
) -> MetricBasis:
    if mesh.n_points != EXPECTED_POINTS or mesh.n_cells != EXPECTED_TETS:
        raise ValueError("baseline anatomy dimensions changed")
    if skin.n_points != EXPECTED_SKIN_POINTS or skin.n_cells != EXPECTED_SKIN_TRIANGLES:
        raise ValueError("baseline IsFace skin dimensions changed")
    missing = REQUIRED_POINT_ARRAYS - set(mesh.point_data)
    if missing:
        raise KeyError(f"baseline result misses point arrays: {sorted(missing)}")
    missing = REQUIRED_CELL_ARRAYS - set(mesh.cell_data)
    if missing:
        raise KeyError(f"baseline result misses cell arrays: {sorted(missing)}")
    base_points = np.asarray(mesh.points, dtype=np.float64)
    target = np.asarray(mesh.point_data["TargetDisplacement"], dtype=np.float64)
    loss_mask = np.asarray(mesh.point_data["LossMask"], dtype=bool)
    target_rms = float(np.linalg.norm(target[loss_mask]) / math.sqrt(loss_mask.sum()))
    if not math.isclose(
        target_rms, EXPECTED_TARGET_RMS_M, rel_tol=1e-13, abs_tol=1e-15
    ):
        raise ValueError(f"target RMS changed: {target_rms}")
    global_ids = np.asarray(mesh.point_data["GlobalPointId"], dtype=np.int64)
    if np.unique(global_ids).size != global_ids.size:
        raise ValueError("mesh GlobalPointId is not unique")
    requested = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    order = np.argsort(global_ids)
    pos = np.searchsorted(global_ids[order], requested)
    if np.any(pos >= len(order)) or not np.array_equal(
        global_ids[order[pos]], requested
    ):
        raise ValueError("skin GlobalPointId mapping failed")
    skin_ids = order[pos]
    if not np.array_equal(np.asarray(skin.points), base_points[skin_ids]):
        raise ValueError("skin rest coordinates differ from anatomy mesh")
    tri = _triangles(skin)
    rest_vectors, rest_area, _ = _triangle_geometry(np.asarray(skin.points), tri)
    target_points = np.asarray(skin.points) + target[skin_ids]
    target_vectors, target_area, target_tri_normals = _triangle_geometry(
        target_points, tri
    )
    target_normals = _vertex_normals(target_points, tri, target_vectors)
    tri0, tri1, lengths = _interior_edges(np.asarray(skin.points), tri)
    contraction, raw_contraction_domain = _raw_contraction_mask(metric_skin, skin)
    selected = contraction[tri0] & contraction[tri1]
    raw_contraction_domain = {
        **raw_contraction_domain,
        "interior_edge_rule": "both incident triangles satisfy raw TargetRestAreaRatio < 1.0",
        "interior_edge_count": int(selected.sum()),
        "interior_edge_pair_sha256": _array_sha256(
            np.column_stack((tri0[selected], tri1[selected]))
        ),
    }
    target_dihedral = np.arccos(
        np.clip(
            np.einsum(
                "ij,ij->i",
                target_tri_normals[tri0[selected]],
                target_tri_normals[tri1[selected]],
            ),
            -1.0,
            1.0,
        )
    )
    encoded = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)
    if not np.all(encoded[:, 0] == 4):
        raise ValueError("anatomy mesh is not pure tetrahedra")
    tets = encoded[:, 1:].copy()
    rest_six_volume = _six_volume(base_points, tets)
    if np.any(np.abs(rest_six_volume) <= np.finfo(np.float64).eps):
        raise ValueError("anatomy mesh contains a degenerate tetrahedron")
    is_fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    fixed_mask = np.asarray(mesh.point_data["FixedMask"], dtype=bool)
    fixed_value = np.asarray(mesh.point_data["FixedValue"], dtype=np.float64)
    cut_mask = np.asarray(mesh.point_data["ArtificialCutIncident"], dtype=bool)
    activation_mask = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    if (
        int(is_fixed.sum()) != EXPECTED_FIXED_VERTICES
        or int(cut_mask.sum()) != EXPECTED_CUT_VERTICES
    ):
        raise ValueError("hard-fixed/cut counts changed")
    if int(activation_mask.sum()) != EXPECTED_ACTIVE_TETS:
        raise ValueError("active muscle tetrahedron count changed")
    if (
        not np.array_equal(fixed_mask, np.repeat(is_fixed[:, None], 3, axis=1))
        or np.any(fixed_value != 0.0)
        or not np.all(is_fixed[cut_mask])
    ):
        raise ValueError("hard-fixed zero-displacement contract changed")
    return MetricBasis(
        base_points=base_points,
        cells=np.asarray(mesh.cells).copy(),
        celltypes=np.asarray(mesh.celltypes).copy(),
        global_ids=global_ids,
        target=target,
        loss_mask=loss_mask,
        target_rms=target_rms,
        activation_mask=activation_mask,
        is_fixed=is_fixed,
        fixed_mask=fixed_mask,
        fixed_value=fixed_value,
        cut_mask=cut_mask,
        skin=skin,
        skin_mesh_ids=skin_ids,
        triangles=tri,
        edges=_unique_edges(tri),
        rest_area=rest_area,
        target_area=target_area,
        target_normals=target_normals,
        contraction_tri_0=tri0[selected],
        contraction_tri_1=tri1[selected],
        contraction_target_dihedral=target_dihedral,
        contraction_edge_weight=lengths[selected],
        raw_contraction_domain=raw_contraction_domain,
        tets=tets,
        rest_six_volume=rest_six_volume,
        rest_area_vectors=rest_vectors,
        rest_area_vector_norm=np.linalg.norm(rest_vectors, axis=1),
        render_views=_render_views(mesh, skin, skin_ids),
    )


def _scalar_laplacian(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    total = np.zeros_like(values)
    count = np.zeros(values.shape[0], dtype=np.int64)
    np.add.at(total, edges[:, 0], values[edges[:, 1]])
    np.add.at(total, edges[:, 1], values[edges[:, 0]])
    np.add.at(count, edges[:, 0], 1)
    np.add.at(count, edges[:, 1], 1)
    result = np.zeros_like(values)
    active = count > 0
    result[active] = values[active] - total[active] / count[active]
    return result


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.dot(weights, np.square(values)) / weights.sum()))


def _activation_quality(active: np.ndarray) -> dict[str, Any]:
    matrices = np.zeros((active.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 + active[:, 0]
    matrices[:, 1, 1] = 1.0 + active[:, 1]
    matrices[:, 2, 2] = 1.0 + active[:, 2]
    matrices[:, 0, 1] = matrices[:, 1, 0] = active[:, 3]
    matrices[:, 1, 2] = matrices[:, 2, 1] = active[:, 4]
    matrices[:, 0, 2] = matrices[:, 2, 0] = active[:, 5]
    eigen = np.linalg.eigvalsh(matrices)
    determinant = np.prod(eigen, axis=1)
    singular = np.abs(eigen)
    condition = singular.max(axis=1) / np.maximum(
        singular.min(axis=1), np.finfo(np.float64).tiny
    )
    if not all(np.isfinite(v).all() for v in (eigen, determinant, condition)):
        raise ValueError("activation conditioning is non-finite")
    return {
        "activation/rms": float(np.linalg.norm(active) / math.sqrt(active.size)),
        "activation/max_abs": float(np.abs(active).max()),
        "activation/I_plus_Ainv_min_eigenvalue": float(eigen.min()),
        "activation/I_plus_Ainv_min_determinant": float(determinant.min()),
        "activation/I_plus_Ainv_max_condition_number": float(condition.max()),
        "quality/I_plus_Ainv_non_spd_active_tets": int(
            np.sum(eigen.min(axis=1) <= 0.0)
        ),
        "quality/I_plus_Ainv_nonpositive_det_active_tets": int(
            np.sum(determinant <= 0.0)
        ),
    }


def _validate_frame(
    case: CaseInput, basis: MetricBasis, frame: pv.UnstructuredGrid, step: int
) -> tuple[np.ndarray, np.ndarray]:
    if frame.n_points != EXPECTED_POINTS or frame.n_cells != EXPECTED_TETS:
        raise ValueError(f"{case.case_id} step {step} dimensions changed")
    for label, actual, expected in (
        ("points", frame.points, basis.base_points),
        ("cells", frame.cells, basis.cells),
        ("celltypes", frame.celltypes, basis.celltypes),
        ("GlobalPointId", frame.point_data["GlobalPointId"], basis.global_ids),
        ("TargetDisplacement", frame.point_data["TargetDisplacement"], basis.target),
        ("LossMask", frame.point_data["LossMask"], basis.loss_mask),
        ("IsFixed", frame.point_data["IsFixed"], basis.is_fixed),
        ("FixedMask", frame.point_data["FixedMask"], basis.fixed_mask),
        ("FixedValue", frame.point_data["FixedValue"], basis.fixed_value),
        (
            "ArtificialCutIncident",
            frame.point_data["ArtificialCutIncident"],
            basis.cut_mask,
        ),
        ("ActivationMask", frame.cell_data["ActivationMask"], basis.activation_mask),
    ):
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            raise ValueError(f"{case.case_id} step {step} {label} changed")
    displacement = np.asarray(frame.point_data["Displacement"], dtype=np.float64)
    activation = np.asarray(frame.cell_data["RecoveredActivationInv"], dtype=np.float64)
    live_activation = np.asarray(frame.cell_data["ActivationInv"], dtype=np.float64)
    if not np.isfinite(displacement).all() or not np.isfinite(activation).all():
        raise ValueError(f"{case.case_id} step {step} contains non-finite state")
    if not np.array_equal(activation, live_activation):
        raise ValueError(f"{case.case_id} step {step} activation readbacks differ")
    if np.any(displacement[basis.is_fixed] != 0.0) or np.any(
        displacement[basis.cut_mask] != 0.0
    ):
        raise ValueError(f"{case.case_id} step {step} violates fixed/cut exact zero")
    if np.any(activation[~basis.activation_mask] != 0.0):
        raise ValueError(f"{case.case_id} step {step} activates non-muscle tetrahedra")
    if step == 0 and np.any(activation[basis.activation_mask] != 0.0):
        raise ValueError(f"{case.case_id} step 0 activation is not exact zero")
    trace = case.trace[step]
    if not bool(trace["forward/success"]) or not bool(trace["adjoint/success"]):
        raise ValueError(f"{case.case_id} step {step} forward/adjoint failed")
    error = float(
        np.linalg.norm((displacement - basis.target)[basis.loss_mask])
        / math.sqrt(basis.loss_mask.sum())
    )
    active = activation[basis.activation_mask]
    activation_rms = float(np.linalg.norm(active) / math.sqrt(active.size))
    if not math.isclose(
        error, float(trace["target/error_rms"]), rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(f"{case.case_id} step {step} trace error differs from state")
    if not math.isclose(
        activation_rms, float(trace["activation_inv/rms"]), rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            f"{case.case_id} step {step} trace activation differs from state"
        )
    return displacement, activation


def _frame_metrics(
    case: CaseInput, basis: MetricBasis, frame: pv.UnstructuredGrid, step: int
) -> dict[str, Any]:
    displacement, activation = _validate_frame(case, basis, frame, step)
    residual = displacement - basis.target
    skin_displacement = displacement[basis.skin_mesh_ids]
    skin_residual = residual[basis.skin_mesh_ids]
    deformed = np.asarray(basis.skin.points) + skin_displacement
    vectors, area, normals = _triangle_geometry(deformed, basis.triangles)
    dihedral = np.arccos(
        np.clip(
            np.einsum(
                "ij,ij->i",
                normals[basis.contraction_tri_0],
                normals[basis.contraction_tri_1],
            ),
            -1.0,
            1.0,
        )
    )
    dihedral_rms = _weighted_rms(
        dihedral - basis.contraction_target_dihedral, basis.contraction_edge_weight
    )
    residual_normal = np.einsum("ij,ij->i", skin_residual, basis.target_normals)
    normal_lap = _scalar_laplacian(residual_normal, basis.edges)
    area_ratio_to_target = area / basis.target_area
    log_area_error = np.log(area_ratio_to_target)
    det_f = (
        _six_volume(basis.base_points + displacement, basis.tets)
        / basis.rest_six_volume
    )
    signed_skin_ratio = np.einsum(
        "ij,ij->i", vectors, basis.rest_area_vectors
    ) / np.square(basis.rest_area_vector_norm)
    if not all(
        np.isfinite(v).all() for v in (area_ratio_to_target, det_f, signed_skin_ratio)
    ):
        raise ValueError(
            f"{case.case_id} step {step} quality diagnostics are non-finite"
        )
    active = activation[basis.activation_mask]
    return {
        "case_id": case.case_id,
        "display_name": DISPLAY_NAMES[case.case_id],
        "step": step,
        "target/error_rms_m": float(
            np.linalg.norm(residual[basis.loss_mask]) / math.sqrt(basis.loss_mask.sum())
        ),
        "target/error_rms_mm": float(
            1e3
            * np.linalg.norm(residual[basis.loss_mask])
            / math.sqrt(basis.loss_mask.sum())
        ),
        "target/error_rms_fraction_of_target": float(
            np.linalg.norm(residual[basis.loss_mask])
            / math.sqrt(basis.loss_mask.sum())
            / basis.target_rms
        ),
        "bumpiness/contraction_target_relative_dihedral_rms_deg": math.degrees(
            dihedral_rms
        ),
        "bumpiness/residual_normal_laplacian_rms_m": float(
            np.linalg.norm(normal_lap) / math.sqrt(normal_lap.size)
        ),
        "area/deformed_to_target_ratio_rms_error": _weighted_rms(
            area_ratio_to_target - 1.0, basis.rest_area
        ),
        "area/deformed_to_target_log_ratio_rms": _weighted_rms(
            log_area_error, basis.rest_area
        ),
        "area/deformed_to_target_ratio_mean": float(
            np.dot(basis.rest_area, area_ratio_to_target) / basis.rest_area.sum()
        ),
        **_activation_quality(active),
        "quality/inverted_tets": int(np.sum(det_f <= 0.0)),
        "quality/detF_min": float(det_f.min()),
        "quality/detF_q001": float(np.quantile(det_f, 0.001)),
        "quality/skin_folded_triangles": int(np.sum(signed_skin_ratio <= 0.0)),
        "quality/skin_signed_ratio_q001": float(np.quantile(signed_skin_ratio, 0.001)),
        "fixed/displacement_exact_zero": True,
        "fixed/displacement_max_abs_m": 0.0,
        "field/residual_normal_m": residual_normal,
    }


def _validate_case_skin(case: CaseInput, basis: MetricBasis) -> None:
    skin = pv.read(case.skin_path)
    if not isinstance(skin, pv.PolyData):
        raise TypeError(f"{case.case_id} skin is not PolyData")
    if skin.n_points != EXPECTED_SKIN_POINTS or skin.n_cells != EXPECTED_SKIN_TRIANGLES:
        raise ValueError(f"{case.case_id} skin dimensions changed")
    if not np.array_equal(np.asarray(skin.points), np.asarray(basis.skin.points)):
        raise ValueError(f"{case.case_id} skin coordinates changed")
    if not np.array_equal(np.asarray(skin.faces), np.asarray(basis.skin.faces)):
        raise ValueError(f"{case.case_id} skin topology changed")
    if not np.array_equal(
        np.asarray(skin.point_data["GlobalPointId"]),
        np.asarray(basis.skin.point_data["GlobalPointId"]),
    ):
        raise ValueError(f"{case.case_id} skin GlobalPointId changed")
    required = {
        "lambda",
        "mu",
        "Fraction",
        "ActivationInv",
        "SkinYoungModulusMPa",
        "SkinPoissonRatio",
        "StressFreeAreaRatio",
        "RestArea",
        "IsFaceTriangle",
    }
    missing = required - set(skin.cell_data)
    if missing:
        raise KeyError(f"{case.case_id} skin misses arrays: {sorted(missing)}")

    raw_ratio = basis.target_area / basis.rest_area
    heterogeneous = case.case_id.startswith("H1")
    prestrained = case.case_id.endswith("P1")
    expected_e = (
        np.where(raw_ratio > 1.0, 0.0, 0.2)
        if heterogeneous
        else np.full_like(raw_ratio, 0.2)
    )
    expected_rho = (
        0.98**2 * np.clip(raw_ratio, 0.5, 1.0)
        if prestrained
        else np.ones_like(raw_ratio)
    )
    expected_diag = np.power(expected_rho, -0.5) - 1.0
    expected_activation = np.column_stack(
        (expected_diag, expected_diag, np.zeros_like(expected_diag))
    )
    numeric = {
        "SkinYoungModulusMPa": expected_e,
        "SkinPoissonRatio": np.full_like(expected_e, 0.49),
        "StressFreeAreaRatio": expected_rho,
        "ActivationInv": expected_activation,
        "lambda": expected_e * 0.49 / (1.0 - 0.49**2),
        "mu": expected_e / (2.0 * (1.0 + 0.49)),
        "Fraction": np.ones_like(expected_e),
        "RestArea": basis.rest_area,
    }
    for name, expected in numeric.items():
        actual = np.asarray(skin.cell_data[name], dtype=np.float64)
        if not np.allclose(actual, expected, rtol=2e-13, atol=2e-15):
            raise ValueError(
                f"{case.case_id} skin {name} violates the locked material field"
            )
    if not np.all(np.asarray(skin.cell_data["IsFaceTriangle"], dtype=bool)):
        raise ValueError(f"{case.case_id} applies membrane outside IsFace")


def _validate_static_case(case: CaseInput, basis: MetricBasis) -> None:
    _validate_case_skin(case, basis)
    target = pv.read(case.target_path)
    result = pv.read(case.result_path)
    for label, mesh in (("target", target), ("result", result)):
        if not isinstance(mesh, pv.UnstructuredGrid):
            raise TypeError(f"{case.case_id} {label} is not UnstructuredGrid")
        if not np.array_equal(
            np.asarray(mesh.points), basis.base_points
        ) or not np.array_equal(np.asarray(mesh.cells), basis.cells):
            raise ValueError(f"{case.case_id} {label} topology changed")
        if not np.array_equal(
            np.asarray(mesh.point_data["TargetDisplacement"]), basis.target
        ):
            raise ValueError(f"{case.case_id} {label} target changed")
    best_step = int(case.summary.get("best/step", case.summary.get("best_step", -1)))
    if not 0 <= best_step <= TERMINAL_STEP:
        raise ValueError(f"{case.case_id} best step is invalid")
    best = case.history.frame(best_step, deep=True)
    if not np.array_equal(
        np.asarray(result.point_data["Displacement"]),
        np.asarray(best.point_data["Displacement"]),
    ):
        raise ValueError(f"{case.case_id} result is not its declared best frame")
    if not np.array_equal(
        np.asarray(result.cell_data["RecoveredActivationInv"]),
        np.asarray(best.cell_data["RecoveredActivationInv"]),
    ):
        raise ValueError(f"{case.case_id} result activation is not its best frame")


def _scan(
    cases: list[CaseInput], basis: MetricBasis
) -> dict[str, list[dict[str, Any]]]:
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        logger.info(
            "Auditing all %d history frames for %s", EXPECTED_FRAMES, case.case_id
        )
        _validate_static_case(case, basis)
        trajectories[case.case_id] = [
            _frame_metrics(case, basis, case.history.frame(step), step)
            for step in range(EXPECTED_FRAMES)
        ]
    return trajectories


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in row.items() if not isinstance(value, np.ndarray)
    }


def _select_cohorts(
    trajectories: dict[str, list[dict[str, Any]]], basis: MetricBasis
) -> dict[str, Any]:
    baseline_fraction = EXPECTED_BASELINE_TERMINAL_RMS_M / basis.target_rms
    terminal = {case: trajectories[case][TERMINAL_STEP] for case in CASE_ORDER}
    fidelity: dict[str, dict[str, Any]] = {}
    for case in CASE_ORDER:
        row = min(
            trajectories[case],
            key=lambda item: (
                abs(
                    float(item["target/error_rms_fraction_of_target"])
                    - baseline_fraction
                ),
                int(item["step"]),
            ),
        )
        gap = abs(float(row["target/error_rms_fraction_of_target"]) - baseline_fraction)
        fidelity[case] = {
            **row,
            "selection/reference_fraction": baseline_fraction,
            "selection/normalized_gap": gap,
            "selection/reached": gap <= BASELINE_FIDELITY_GAP_TOLERANCE,
            "selection/status": "reached"
            if gap <= BASELINE_FIDELITY_GAP_TOLERANCE
            else "did-not-reach",
        }
    minima = {
        case: min(
            float(row["target/error_rms_fraction_of_target"])
            for row in trajectories[case]
        )
        for case in CASE_ORDER
    }
    tau = max(minima.values())
    common: dict[str, dict[str, Any]] = {}
    for case in CASE_ORDER:
        row = min(
            trajectories[case],
            key=lambda item: (
                abs(float(item["target/error_rms_fraction_of_target"]) - tau),
                int(item["step"]),
            ),
        )
        common[case] = {
            **row,
            "selection/tau": tau,
            "selection/normalized_gap": abs(
                float(row["target/error_rms_fraction_of_target"]) - tau
            ),
        }
    return {
        "terminal": {case: _public(row) for case, row in terminal.items()},
        "baseline-fidelity": {case: _public(row) for case, row in fidelity.items()},
        "common-tau": {case: _public(row) for case, row in common.items()},
        "baseline_fidelity_rule": {
            "reference_error_rms_m": EXPECTED_BASELINE_TERMINAL_RMS_M,
            "reference_fraction_of_target": baseline_fraction,
            "nearest_saved_frame_only": True,
            "normalized_gap_tolerance": BASELINE_FIDELITY_GAP_TOLERANCE,
            "did_not_reach_rule": "absolute normalized gap > 0.01",
        },
        "common_tau_rule": {
            "tau": tau,
            "definition": "max across four cases of each 41-frame trajectory minimum target-error fraction; nearest saved frame only, no interpolation",
            "per_case_minimum": minima,
            "secondary_comparison_only": True,
        },
    }


def _write_paraview_inputs(
    cases: list[CaseInput],
    basis: MetricBasis,
    trajectories: dict[str, list[dict[str, Any]]],
    cohorts: dict[str, Any],
) -> dict[str, Any]:
    by_case = {case.case_id: case for case in cases}
    PARAVIEW_INPUT_DIR.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, Any] = {}
    residual_values: list[np.ndarray] = []
    for cohort in ("terminal", "baseline-fidelity", "common-tau"):
        cohort_dir = PARAVIEW_INPUT_DIR / cohort
        cohort_dir.mkdir()
        outputs[cohort] = {}
        for case_id in CASE_ORDER:
            selected = cohorts[cohort][case_id]
            step = int(selected["step"])
            frame = by_case[case_id].history.frame(step, deep=True)
            displacement = np.asarray(
                frame.point_data["Displacement"], dtype=np.float64
            )
            row = trajectories[case_id][step]
            residual = np.asarray(row["field/residual_normal_m"], dtype=np.float64)
            residual_values.append(residual)
            surface = basis.skin.copy(deep=True)
            surface.points = (
                np.asarray(surface.points) + displacement[basis.skin_mesh_ids]
            )
            surface.point_data["TargetNormalResidualMM"] = 1e3 * residual
            surface.point_data["DisplacementMM"] = (
                1e3 * displacement[basis.skin_mesh_ids]
            )
            surface.field_data["CaseId"] = np.asarray([case_id])
            surface.field_data["CheckpointStep"] = np.asarray([step], dtype=np.int32)
            output = cohort_dir / f"{case_id.lower()}.vtp"
            surface.save(output)
            identity = _identity(output)
            outputs[cohort][case_id] = {
                "path": str(output.resolve()),
                "size_bytes": identity.size_bytes,
                "sha256": identity.sha256,
                "step": step,
                "label": DISPLAY_NAMES[case_id],
                "target_error_rms_mm": selected["target/error_rms_mm"],
                "dihedral_rms_deg": selected[
                    "bumpiness/contraction_target_relative_dihedral_rms_deg"
                ],
                "normal_laplacian_rms_mm": 1e3
                * selected["bumpiness/residual_normal_laplacian_rms_m"],
                "area_ratio_rms_error": selected[
                    "area/deformed_to_target_ratio_rms_error"
                ],
                "selection_status": selected.get("selection/status", "selected"),
            }
            cherries.log_output(output)
    residual_limit = max(
        0.25, 1e3 * float(np.quantile(np.abs(np.concatenate(residual_values)), 0.99))
    )
    return {
        "inputs": outputs,
        "case_order": list(CASE_ORDER),
        "cohort_order": ["terminal", "baseline-fidelity", "common-tau"],
        "view_order": list(basis.render_views),
        "views": basis.render_views,
        "normal_residual_shared_limit_mm": residual_limit,
        "renderer": "ParaView 6.1.1 only; Matplotlib geometry render is prohibited",
    }


CSV_FIELDS = (
    "case_id",
    "display_name",
    "step",
    "target/error_rms_m",
    "target/error_rms_mm",
    "target/error_rms_fraction_of_target",
    "bumpiness/contraction_target_relative_dihedral_rms_deg",
    "bumpiness/residual_normal_laplacian_rms_m",
    "area/deformed_to_target_ratio_rms_error",
    "area/deformed_to_target_log_ratio_rms",
    "area/deformed_to_target_ratio_mean",
    "activation/rms",
    "activation/max_abs",
    "activation/I_plus_Ainv_min_eigenvalue",
    "activation/I_plus_Ainv_min_determinant",
    "activation/I_plus_Ainv_max_condition_number",
    "quality/I_plus_Ainv_non_spd_active_tets",
    "quality/I_plus_Ainv_nonpositive_det_active_tets",
    "quality/inverted_tets",
    "quality/detF_min",
    "quality/detF_q001",
    "quality/skin_folded_triangles",
    "quality/skin_signed_ratio_q001",
    "fixed/displacement_exact_zero",
    "fixed/displacement_max_abs_m",
)


def _write_csv(path: Path, trajectories: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for case in CASE_ORDER:
            for row in trajectories[case]:
                writer.writerow({key: row[key] for key in CSV_FIELDS})


def _plot_trajectories(
    path: Path, trajectories: dict[str, list[dict[str, Any]]]
) -> None:
    specs = (
        ("target/error_rms_mm", "target RMS error [mm]", 1.0),
        (
            "bumpiness/contraction_target_relative_dihedral_rms_deg",
            "D: target-relative dihedral RMS [deg]",
            1.0,
        ),
        (
            "bumpiness/residual_normal_laplacian_rms_m",
            "L: normal-residual Laplacian RMS [mm]",
            1e3,
        ),
        ("area/deformed_to_target_ratio_rms_error", "target-area ratio RMS error", 1.0),
        ("activation/rms", "active-muscle ActivationInv RMS", 1.0),
        ("activation/I_plus_Ainv_min_eigenvalue", "min eig(I + ActivationInv)", 1.0),
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    colors = ("#111111", "#2b6cb0", "#c53030", "#2f855a")
    for axis, (key, ylabel, scale) in zip(axes.flat, specs, strict=True):
        for case, color in zip(CASE_ORDER, colors, strict=True):
            rows = trajectories[case]
            axis.plot(
                [r["step"] for r in rows],
                [scale * r[key] for r in rows],
                label=case,
                color=color,
            )
        axis.set_xlabel("inverse evaluation (0..40)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(
        "Equal-budget inverse trajectories; geometry is rendered only in ParaView"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pareto(
    path: Path, trajectories: dict[str, list[dict[str, Any]]], cohorts: dict[str, Any]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    specs = (
        ("bumpiness/contraction_target_relative_dihedral_rms_deg", "D [deg]"),
        ("bumpiness/residual_normal_laplacian_rms_m", "L [m]"),
        ("area/deformed_to_target_ratio_rms_error", "area-ratio RMS error"),
    )
    colors = ("#111111", "#2b6cb0", "#c53030", "#2f855a")
    for axis, (key, ylabel) in zip(axes, specs, strict=True):
        for case, color in zip(CASE_ORDER, colors, strict=True):
            rows = trajectories[case]
            axis.plot(
                [r["target/error_rms_mm"] for r in rows],
                [r[key] for r in rows],
                color=color,
                alpha=0.7,
                label=case,
            )
            for marker, cohort in (
                ("s", "terminal"),
                ("o", "baseline-fidelity"),
                ("^", "common-tau"),
            ):
                selected = cohorts[cohort][case]
                axis.scatter(
                    selected["target/error_rms_mm"],
                    selected[key],
                    color=color,
                    marker=marker,
                    s=40,
                )
        axis.set_xlabel("target RMS error [mm]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(
        "Target-fit / smoothness Pareto paths (saved frames; no interpolation)"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_checkpoint_table(path: Path, cohorts: dict[str, Any]) -> None:
    lines = [
        "# Selective skin energy + prestrain inverse checkpoints",
        "",
        "All rows are actual saved frames. Terminal is the equal-budget step-40 comparison.",
        "Baseline-fidelity rows farther than 0.01 target-RMS units are explicitly marked `did-not-reach`.",
        "",
        "| cohort | case | step | status | error mm | error/target | D deg | L mm | area ratio RMS | activation RMS | non-SPD | inverted | folded |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cohort in ("terminal", "baseline-fidelity", "common-tau"):
        for case in CASE_ORDER:
            row = cohorts[cohort][case]
            lines.append(
                f"| {cohort} | {case} | {row['step']} | {row.get('selection/status', 'selected')} | "
                f"{row['target/error_rms_mm']:.6g} | {row['target/error_rms_fraction_of_target']:.6g} | "
                f"{row['bumpiness/contraction_target_relative_dihedral_rms_deg']:.6g} | "
                f"{1e3 * row['bumpiness/residual_normal_laplacian_rms_m']:.6g} | "
                f"{row['area/deformed_to_target_ratio_rms_error']:.6g} | {row['activation/rms']:.6g} | "
                f"{row['quality/I_plus_Ainv_non_spd_active_tets']} | {row['quality/inverted_tets']} | "
                f"{row['quality/skin_folded_triangles']} |"
            )
    lines.extend(
        [
            "",
            f"Common tau: `{cohorts['common_tau_rule']['tau']:.12g}`.",
            "",
            "The common-tau cohort is secondary. Tau is the worst of the four per-case minima,",
            "then each case uses its nearest saved checkpoint without interpolation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_finite(value: Any, *, context: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite(child, context=f"{context}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite(child, context=f"{context}/{index}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {context}")


def _identity_record(
    path: Path, expected: FileIdentity, *, label: str
) -> dict[str, Any]:
    _require_identity(path, expected, label=label)
    return {
        "path": str(path.resolve()),
        "size_bytes": expected.size_bytes,
        "sha256": expected.sha256,
        "unchanged": True,
    }


def _recheck_analysis_inputs(
    cases: list[CaseInput],
    *,
    aggregate_identity: FileIdentity,
    producer_identity: FileIdentity,
) -> dict[str, Any]:
    case_checks: dict[str, Any] = {}
    for case in cases:
        case_checks[case.case_id] = {
            name: _identity_record(
                getattr(case, f"{name}_path"),
                identity,
                label=f"post-analysis {case.case_id} {name}",
            )
            for name, identity in case.identities.items()
        }
    return {
        "all_unchanged": True,
        "canonical_final_aggregate": _identity_record(
            AGGREGATE,
            aggregate_identity,
            label="post-analysis canonical final formal aggregate",
        ),
        "producer": _identity_record(
            PRODUCER, producer_identity, label="post-analysis inverse producer"
        ),
        "metric_skin": _identity_record(
            METRIC_SKIN,
            FileIdentity(*METRIC_SKIN_IDENTITY),
            label="post-analysis raw-area-ratio metric skin",
        ),
        "cases": case_checks,
    }


def main(cfg: Config) -> None:
    _validate_config(cfg)
    aggregate_identity = _identity(cfg.input_aggregate)
    producer_identity = _identity(PRODUCER)
    aggregate = _read_json(cfg.input_aggregate)
    cases = _load_cases(aggregate)
    baseline_result = pv.read(cases[0].result_path)
    skin = pv.read(cases[0].skin_path)
    metric_skin = pv.read(METRIC_SKIN)
    if (
        not isinstance(baseline_result, pv.UnstructuredGrid)
        or not isinstance(skin, pv.PolyData)
        or not isinstance(metric_skin, pv.PolyData)
    ):
        raise TypeError("canonical metric inputs have unexpected VTK types")
    basis = _build_basis(baseline_result, skin, metric_skin)
    trajectories = _scan(cases, basis)
    baseline_terminal = trajectories["H0P0"][TERMINAL_STEP]
    if not math.isclose(
        float(baseline_terminal["target/error_rms_m"]),
        EXPECTED_BASELINE_TERMINAL_RMS_M,
        rel_tol=1e-12,
        abs_tol=1e-14,
    ):
        raise ValueError("authoritative H0P0 terminal fidelity changed")
    cohorts = _select_cohorts(trajectories, basis)
    paraview = _write_paraview_inputs(cases, basis, trajectories, cohorts)
    _write_csv(cfg.output_csv, trajectories)
    _write_checkpoint_table(cfg.output_checkpoints, cohorts)
    _plot_trajectories(cfg.output_trajectories, trajectories)
    _plot_pareto(cfg.output_pareto, trajectories, cohorts)
    post_analysis_checks = _recheck_analysis_inputs(
        cases,
        aggregate_identity=aggregate_identity,
        producer_identity=producer_identity,
    )

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "design": "2x2-selective-skin-energy-c020-prestrain-inverse-analysis",
        "complete": True,
        "analysis_scope": "four independent 41-evaluation inverse histories; H0P0 is the pinned authoritative baseline",
        "case_order": list(CASE_ORDER),
        "case_labels": DISPLAY_NAMES,
        "metric_definitions": {
            "D": "rest-edge-length-weighted RMS of deformed-minus-target dihedral angle on interior edges whose two incident canonical IsFace triangles both satisfy strict raw TargetRestAreaRatio < 1.0",
            "L": "RMS graph Laplacian of target-normal displacement residual on all IsFace vertices",
            "area_ratio_error": "rest-area-weighted RMS of deformed triangle area / target triangle area - 1",
            "activation": "six-component symmetric ActivationInv on active muscle tetrahedra; I+ActivationInv eigen diagnostics",
            "quality": "det(F), signed skin orientation, folds, inversions, and exact fixed/cut displacement",
        },
        "metric_domains": {"D/raw_contraction": basis.raw_contraction_domain},
        "formal_aggregate": {
            "path": str(cfg.input_aggregate.resolve()),
            "size_bytes": aggregate_identity.size_bytes,
            "sha256": aggregate_identity.sha256,
            "canonical_final": True,
        },
        "inputs": {
            case.case_id: {
                name: {
                    "path": str(getattr(case, f"{name}_path")),
                    "size_bytes": identity.size_bytes,
                    "sha256": identity.sha256,
                }
                for name, identity in case.identities.items()
            }
            for case in cases
        },
        "producer": {
            "path": str(PRODUCER.resolve()),
            "size_bytes": producer_identity.size_bytes,
            "sha256": EXPECTED_PRODUCER_SHA256,
        },
        "post_analysis_input_identity_checks": post_analysis_checks,
        "trajectories": {
            case: [_public(row) for row in trajectories[case]] for case in CASE_ORDER
        },
        "cohorts": cohorts,
        "paraview": paraview,
        "outputs": {
            "csv": str(cfg.output_csv.resolve()),
            "checkpoint_table": str(cfg.output_checkpoints.resolve()),
            "trajectory_plot": str(cfg.output_trajectories.resolve()),
            "pareto_plot": str(cfg.output_pareto.resolve()),
        },
        "limitations": [
            "H1 lowers mean membrane stiffness; this is an extreme selective membrane-removal ablation, not physiological skin identification.",
            "The common-tau cohort is secondary and uses nearest saved frames without interpolation.",
            "Fold and inversion counts are warnings requiring ParaView review, not automatic vetoes.",
        ],
    }
    _validate_finite(analysis)
    cfg.output_json.write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote audited inverse analysis to %s", cfg.output_json)


if __name__ == "__main__":
    # Never allow the default Cherries profile to stage or commit this dirty
    # research worktree.  The debug profile retains Local + Logging provenance
    # while its Git plugin is explicitly non-committing.
    cherries.main(main, profile="debug")
