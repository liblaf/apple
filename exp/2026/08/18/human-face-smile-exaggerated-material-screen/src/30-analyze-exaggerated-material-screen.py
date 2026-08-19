from __future__ import annotations

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
import pydantic_settings as ps
import pyvista as pv
from _reference import (
    GROUP_DIR,
    MATERIAL_REFERENCE_GROUP,
    PREPARED_MESH,
    SOURCE_SKIN,
    SOURCE_SKIN_SHA256,
    SOURCE_SKIN_SIZE_BYTES,
    enable_reference_modules,
)
from vtkmodules.vtkCommonExecutionModel import (
    vtkStreamingDemandDrivenPipeline as StreamingPipeline,
)

from liblaf import cherries
from liblaf.apple.common import GLOBAL_POINT_ID

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt

enable_reference_modules()

from _human_face_output import (  # noqa: E402
    bumpiness_metrics,
    surface_edges_for_mask,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
EXPECTED_EVALUATIONS = 41
TERMINAL_STEP = EXPECTED_EVALUATIONS - 1
EXPECTED_LR = 0.3
EXPECTED_NEW_LABELS = ("e100-p200", "e005-p000", "e005-p200")
EXPECTED_CASE_ORDER = (
    "e100-p000",
    "e100-p200",
    "e005-p000",
    "e005-p200",
    "e025-p100",
    "no-skin",
)
EXPECTED_PARAMETERS: dict[str, tuple[float | None, float | None]] = {
    "e100-p000": (1.0, 0.0),
    "e100-p200": (1.0, 2.0),
    "e005-p000": (0.05, 0.0),
    "e005-p200": (0.05, 2.0),
    "e025-p100": (0.25, 1.0),
    "no-skin": (None, None),
}
DISPLAY_NAMES = {
    "e100-p000": "baseline",
    "e100-p200": "prestrain only",
    "e005-p000": "softening only",
    "e005-p200": "combined extreme",
    "e025-p100": "current moderate",
    "no-skin": "no skin",
}
NEW_SUMMARY_NAME = "20-exaggerated-material-screen-summary.json"
OLD_DATA_DIR = MATERIAL_REFERENCE_GROUP / "data"
JSON_RTOL = 1.0e-10
JSON_ATOL = 1.0e-12


@dataclass(frozen=True)
class FileIdentity:
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, int | str]:
        return {"size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class OldCaseSpec:
    label: str
    stem: str
    summary_identity: FileIdentity
    trace_identity: FileIdentity
    history_identity: FileIdentity


OLD_CASE_SPECS = (
    OldCaseSpec(
        label="e100-p000",
        stem="20-human-face-smile-skin-no-prestrain-lr3-material-e100-p000-screen",
        summary_identity=FileIdentity(
            123_434,
            "cba0574628ddef2f41fa79af14e9f84577e3d1fea9a1dec2ec6796822e621d65",
        ),
        trace_identity=FileIdentity(
            87_435,
            "9afef41e0a7553666fe87fb8c464624af51c1ed2e421e33a09af22689007fae5",
        ),
        history_identity=FileIdentity(
            2_073_226_098,
            "05550fa7559c2f78aad6f34460edf58a6fe3a18b3dd4c7527231d366dfabb80d",
        ),
    ),
    OldCaseSpec(
        label="e025-p100",
        stem=(
            "20-human-face-smile-skin-estimated-prestrain-lr3-material-e025-p100-screen"
        ),
        summary_identity=FileIdentity(
            124_562,
            "946c204bd8c8160d26d9c959446bb59f994ceb599cf858032719c0b1fe05b9cb",
        ),
        trace_identity=FileIdentity(
            88_132,
            "afb355d1babd2c36f8b6d8cdc63ce982d6aa79f2c37bdd1dc59edd655edbe17a",
        ),
        history_identity=FileIdentity(
            2_077_943_165,
            "1255640c9738ba829bc89b9a3f8a643eb52f7f5edadd4d83ee79c70627cd19e5",
        ),
    ),
    OldCaseSpec(
        label="no-skin",
        stem="20-human-face-smile-no-skin-lr3-material-no-skin-screen",
        summary_identity=FileIdentity(
            114_796,
            "4f3fdb590df48377453a7df4b990cd99df3d8e03ee274da1ae376c2bd04fd1da",
        ),
        trace_identity=FileIdentity(
            86_560,
            "ab8167401cc3de9c4c58f284d665824b80ed56e673b768f4deea43a8d0f43a95",
        ),
        history_identity=FileIdentity(
            2_077_120_296,
            "45e3aef89f62e0ac8f88ea0f08d4c1deaef57ae336ecd856cd23a99f26305642",
        ),
    ),
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_skin: Path = cherries.input(SOURCE_SKIN)
    input_candidates: Path = cherries.input("10-exaggerated-materials-manifest.json")
    input_new_summary: Path = cherries.input(NEW_SUMMARY_NAME)
    output_json: Path = cherries.output(
        "30-exaggerated-material-screen-analysis.json", mkdir=True
    )
    output_csv: Path = cherries.output(
        "30-exaggerated-material-screen-trajectories.csv", mkdir=True
    )
    output_table: Path = cherries.output(
        "30-exaggerated-material-screen-checkpoints.md", mkdir=True
    )
    output_plot: Path = cherries.output(
        "30-exaggerated-material-screen-trajectories.png", mkdir=True
    )
    output_terminal_views: Path = cherries.output(
        "30-exaggerated-material-screen-terminal-views.png", mkdir=True
    )
    output_matched_views: Path = cherries.output(
        "30-exaggerated-material-screen-matched-views.png", mkdir=True
    )


@dataclass
class TemporalHistory:
    label: str
    path: Path
    pyvista_reader: Any
    times: np.ndarray

    @classmethod
    def open(cls, label: str, path: Path) -> TemporalHistory:
        reader = pv.get_reader(path)
        vtk_reader = reader.reader
        vtk_reader.UpdateInformation()
        information = vtk_reader.GetOutputInformation(0)
        key = StreamingPipeline.TIME_STEPS()
        if not information.Has(key):
            msg = f"case {label!r} history exposes no TIME_STEPS: {path}"
            raise ValueError(msg)
        times = np.asarray(
            [information.Get(key, index) for index in range(information.Length(key))],
            dtype=np.float64,
        )
        expected = np.arange(EXPECTED_EVALUATIONS, dtype=np.float64)
        if not np.array_equal(times, expected):
            msg = f"case {label!r} history times are not exact steps 0..40"
            raise ValueError(msg)
        return cls(label=label, path=path, pyvista_reader=reader, times=times)

    def frame(self, step: int, *, deep_copy: bool = False) -> pv.UnstructuredGrid:
        if not 0 <= step < self.times.size:
            msg = f"case {self.label!r} has no temporal step {step}"
            raise IndexError(msg)
        vtk_reader = self.pyvista_reader.reader
        vtk_reader.UpdateTimeStep(float(self.times[step]))
        result = pv.wrap(vtk_reader.GetOutputDataObject(0))
        if not isinstance(result, pv.UnstructuredGrid):
            result = result.cast_to_unstructured_grid()
        return result.copy(deep=True) if deep_copy else result


@dataclass(frozen=True)
class CaseInput:
    label: str
    origin: str
    young_min_scale: float | None
    prestrain_gain: float | None
    summary_path: Path
    trace_path: Path
    history_path: Path
    summary: dict[str, Any]
    trace: list[dict[str, Any]]
    identities: dict[str, dict[str, int | str]]
    history: TemporalHistory


@dataclass(frozen=True)
class SurfaceBasis:
    base_points: np.ndarray
    base_cells: np.ndarray
    base_celltypes: np.ndarray
    base_global_ids: np.ndarray
    tets: np.ndarray
    rest_six_volume: np.ndarray
    target: np.ndarray
    loss_mask: np.ndarray
    target_rms: float
    legacy_edges: np.ndarray
    skin: pv.PolyData
    skin_points: np.ndarray
    skin_mesh_ids: np.ndarray
    triangles: np.ndarray
    rest_area_vectors: np.ndarray
    rest_area_vector_norm: np.ndarray
    contraction_edge_tri_0: np.ndarray
    contraction_edge_tri_1: np.ndarray
    contraction_target_dihedral: np.ndarray
    contraction_edge_weight: np.ndarray
    face_triangle_mask: np.ndarray
    face_focus: np.ndarray
    face_parallel_scale: float
    mouth_focus: np.ndarray
    mouth_parallel_scale: float


def reject_json_constant(value: str) -> None:
    msg = f"non-standard JSON constant {value!r}"
    raise ValueError(msg)


def validate_finite_json(value: Any, *, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{context} contains non-finite number {value}"
        raise ValueError(msg)
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite_json(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite_json(item, context=f"{context}[{index}]")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_json_constant
    )
    if not isinstance(value, dict):
        msg = f"expected a JSON object in {path}"
        raise TypeError(msg)
    validate_finite_json(value, context=str(path))
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                msg = f"blank JSONL row at {path}:{line_number}"
                raise ValueError(msg)
            row = json.loads(line, parse_constant=reject_json_constant)
            if not isinstance(row, dict):
                msg = f"expected a JSON object at {path}:{line_number}"
                raise TypeError(msg)
            validate_finite_json(row, context=f"{path}:{line_number}")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, hash_content: bool = True) -> dict[str, int | str]:
    if not path.is_file():
        msg = f"missing input artifact: {path}"
        raise FileNotFoundError(msg)
    identity: dict[str, int | str] = {"size_bytes": path.stat().st_size}
    if hash_content:
        identity["sha256"] = sha256_file(path)
    return identity


def require_identity(
    path: Path,
    expected: FileIdentity,
    *,
    context: str,
    verify_digest: bool,
) -> dict[str, int | str]:
    if not path.is_file():
        msg = f"{context} is missing: {path}"
        raise FileNotFoundError(msg)
    if path.stat().st_size != expected.size_bytes:
        msg = (
            f"{context} size mismatch: expected {expected.size_bytes}, "
            f"got {path.stat().st_size}"
        )
        raise ValueError(msg)
    if verify_digest:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected.sha256:
            msg = (
                f"{context} SHA-256 mismatch: expected {expected.sha256}, "
                f"got {actual_sha256}"
            )
            raise ValueError(msg)
    return {
        "size_bytes": expected.size_bytes,
        "sha256": expected.sha256,
        "sha256_verified": verify_digest,
    }


def require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        msg = f"{context}: expected {expected!r}, got {actual!r}"
        raise ValueError(msg)


def require_true(actual: Any, context: str) -> None:
    if actual is not True:
        msg = f"{context}: expected True, got {actual!r}"
        raise ValueError(msg)


def require_false(actual: Any, context: str) -> None:
    if actual is not False:
        msg = f"{context}: expected False, got {actual!r}"
        raise ValueError(msg)


def require_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=JSON_RTOL, abs_tol=JSON_ATOL):
        msg = f"{context}: expected {expected:.17g}, got {actual:.17g}"
        raise ValueError(msg)


def safe_sibling(directory: Path, basename: Any, *, context: str) -> Path:
    name = Path(str(basename))
    if name.name != str(name) or name.is_absolute():
        msg = f"{context} is not a safe basename: {basename!r}"
        raise ValueError(msg)
    result = (directory / name).resolve()
    if result.parent != directory.resolve():
        msg = f"{context} escapes {directory}: {result}"
        raise ValueError(msg)
    return result


def compare_trace_rows(
    label: str,
    embedded: list[dict[str, Any]],
    recorded: list[dict[str, Any]],
) -> None:
    require_equal(len(recorded), len(embedded), f"{label} live trace length")
    for step, (source, live) in enumerate(zip(embedded, recorded, strict=True)):
        expected = {
            key: value for key, value in source.items() if key != "time/live_plot_s"
        }
        require_equal(set(live), set(expected), f"{label} trace step {step} keys")
        for key, expected_value in expected.items():
            actual = live[key]
            if isinstance(expected_value, float):
                require_close(
                    float(actual), expected_value, f"{label} trace step {step} {key}"
                )
            else:
                require_equal(
                    actual, expected_value, f"{label} trace step {step} {key}"
                )


def validate_case_summary(
    label: str,
    summary: dict[str, Any],
    recorded_trace: list[dict[str, Any]],
    *,
    history_path: Path,
    trace_path: Path,
) -> list[dict[str, Any]]:
    require_equal(summary.get("candidate"), label, f"{label} candidate")
    require_equal(summary.get("stage"), "screen", f"{label} stage")
    require_equal(summary.get("status"), "ok", f"{label} status")
    require_equal(summary.get("validation/errors"), [], f"{label} validation")
    require_true(summary.get("baseline/completed"), f"{label} budget")
    for key in (
        "inverse/evaluations",
        "baseline/evaluations",
        "baseline/evaluations_expected",
        "history/frames",
        "history_frames",
    ):
        require_equal(summary.get(key), EXPECTED_EVALUATIONS, f"{label} {key}")
    require_equal(
        summary.get("baseline/mandatory_optimizer_steps"),
        TERMINAL_STEP,
        f"{label} optimizer steps",
    )
    require_close(
        float(summary.get("baseline/fixed_lr", math.nan)),
        EXPECTED_LR,
        f"{label} fixed LR",
    )
    require_equal(
        summary.get("baseline/lr_deviation_count"), 0, f"{label} LR deviations"
    )
    require_equal(
        summary.get("activation/mode"),
        "per-muscle-tet-6dof",
        f"{label} activation mode",
    )
    require_equal(
        summary.get("activation_inv/initial_rms"), 0.0, f"{label} initial RMS"
    )
    require_equal(
        summary.get("activation_inv/initial_max_abs"),
        0.0,
        f"{label} initial max",
    )
    require_false(
        summary.get("initial_displacement/enabled"), f"{label} initial displacement"
    )
    require_equal(
        Path(str(summary.get("history/path"))).name,
        history_path.name,
        f"{label} history basename",
    )
    require_equal(
        Path(str(summary.get("trace/path"))).name,
        trace_path.name,
        f"{label} trace basename",
    )
    expected_parameters = EXPECTED_PARAMETERS[label]
    actual_parameters = (
        summary.get("candidate/young_min_scale"),
        summary.get("candidate/prestrain_gain"),
    )
    require_equal(actual_parameters, expected_parameters, f"{label} parameters")
    require_equal(
        summary.get("skin/enabled"), label != "no-skin", f"{label} skin enabled"
    )
    if not math.isfinite(float(summary.get("target/displacement_rms", math.nan))):
        msg = f"{label} target RMS is not finite"
        raise ValueError(msg)

    trace = summary.get("trace")
    if not isinstance(trace, list) or not all(isinstance(row, dict) for row in trace):
        msg = f"{label} embedded trace must be a list of objects"
        raise TypeError(msg)
    require_equal(len(trace), EXPECTED_EVALUATIONS, f"{label} trace length")
    finite_keys = (
        "loss/total",
        "loss/mm2",
        "loss/m2",
        "target/error_rms",
        "target/error_rms_mm",
        "activation_inv/rms",
        "activation_inv/max_abs",
        "grad/norm",
        "forward/relative_grad_norm",
        "adjoint/relative_residual",
    )
    for step, row in enumerate(trace):
        require_equal(int(row.get("step", -1)), step, f"{label} trace step")
        require_true(row.get("forward/success"), f"{label} forward {step}")
        require_true(row.get("adjoint/success"), f"{label} adjoint {step}")
        require_close(
            float(row.get("inverse/lr", math.nan)),
            EXPECTED_LR,
            f"{label} LR step {step}",
        )
        for key in finite_keys:
            if not math.isfinite(float(row.get(key, math.nan))):
                msg = f"{label} step {step} has non-finite {key}"
                raise ValueError(msg)
    require_equal(trace[0]["activation_inv/rms"], 0.0, f"{label} step-0 RMS")
    require_equal(trace[0]["activation_inv/max_abs"], 0.0, f"{label} step-0 max")
    compare_trace_rows(label, list(trace), recorded_trace)
    return list(trace)


def old_case_paths(spec: OldCaseSpec) -> tuple[Path, Path, Path]:
    return (
        OLD_DATA_DIR / f"{spec.stem}-summary.json",
        OLD_DATA_DIR / f"{spec.stem}-trace.jsonl",
        OLD_DATA_DIR / f"{spec.stem}-steps.vtkhdf",
    )


def load_old_case(spec: OldCaseSpec, *, verify_history_digest: bool) -> CaseInput:
    summary_path, trace_path, history_path = old_case_paths(spec)
    identities = {
        "summary": require_identity(
            summary_path,
            spec.summary_identity,
            context=f"reused {spec.label} summary",
            verify_digest=True,
        ),
        "trace": require_identity(
            trace_path,
            spec.trace_identity,
            context=f"reused {spec.label} trace",
            verify_digest=True,
        ),
        "history": require_identity(
            history_path,
            spec.history_identity,
            context=f"reused {spec.label} history",
            verify_digest=verify_history_digest,
        ),
    }
    summary = read_json(summary_path)
    recorded_trace = read_jsonl(trace_path)
    trace = validate_case_summary(
        spec.label,
        summary,
        recorded_trace,
        history_path=history_path,
        trace_path=trace_path,
    )
    history = TemporalHistory.open(spec.label, history_path)
    parameters = EXPECTED_PARAMETERS[spec.label]
    return CaseInput(
        label=spec.label,
        origin="reused-2026-08-17",
        young_min_scale=parameters[0],
        prestrain_gain=parameters[1],
        summary_path=summary_path,
        trace_path=trace_path,
        history_path=history_path,
        summary=summary,
        trace=trace,
        identities=identities,
        history=history,
    )


def load_manifest(cfg: Config) -> dict[str, Any]:
    manifest = read_json(cfg.input_candidates)
    require_equal(manifest.get("schema_version"), 1, "candidate manifest schema")
    require_true(manifest.get("complete"), "candidate manifest complete")
    require_equal(
        manifest.get("design"),
        "exaggerated-heterogeneous-mechanism-screen",
        "candidate manifest design",
    )
    require_equal(
        manifest.get("validation_errors"), [], "candidate manifest validation"
    )
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not all(
        isinstance(row, dict) for row in candidates
    ):
        msg = "candidate manifest candidates must be a list of objects"
        raise TypeError(msg)
    require_equal(
        tuple(row.get("label") for row in candidates),
        EXPECTED_NEW_LABELS,
        "candidate manifest labels",
    )
    for row in candidates:
        label = str(row["label"])
        require_equal(
            (row.get("young_min_scale"), row.get("prestrain_gain")),
            EXPECTED_PARAMETERS[label],
            f"manifest {label} parameters",
        )
        require_true(row.get("validation/ok"), f"manifest {label} valid")
        require_equal(row.get("validation/errors"), [], f"manifest {label} errors")
    return manifest


def load_new_cases(cfg: Config, manifest: dict[str, Any]) -> list[CaseInput]:
    aggregate = read_json(cfg.input_new_summary)
    require_equal(aggregate.get("schema_version"), 1, "new summary schema")
    require_true(aggregate.get("complete"), "new summary complete")
    require_equal(
        aggregate.get("design"),
        "exaggerated-heterogeneous-mechanism-screen",
        "new summary design",
    )
    require_equal(aggregate.get("stage"), "screen", "new summary stage")
    require_equal(
        aggregate.get("candidate_set"),
        ",".join(EXPECTED_NEW_LABELS),
        "new candidate set",
    )
    require_equal(aggregate.get("inverse_lr"), EXPECTED_LR, "new summary LR")
    require_equal(
        aggregate.get("inverse_max_steps"), TERMINAL_STEP, "new summary steps"
    )
    require_equal(aggregate.get("hard_failures"), [], "new hard failures")
    require_equal(
        Path(str(aggregate.get("input_mesh"))).resolve(),
        cfg.input_mesh.resolve(),
        "new summary input mesh",
    )
    require_equal(
        Path(str(aggregate.get("input_candidates"))).resolve(),
        cfg.input_candidates.resolve(),
        "new summary input candidates",
    )
    raw_cases = aggregate.get("cases")
    if not isinstance(raw_cases, list) or not all(
        isinstance(row, dict) for row in raw_cases
    ):
        msg = "new summary cases must be a list of objects"
        raise TypeError(msg)
    require_equal(
        tuple(row.get("candidate") for row in raw_cases),
        EXPECTED_NEW_LABELS,
        "new summary labels",
    )
    manifest_by_label = {str(row["label"]): row for row in manifest["candidates"]}
    data_dir = cfg.input_new_summary.resolve().parent
    cases: list[CaseInput] = []
    for row in raw_cases:
        label = str(row["candidate"])
        require_true(
            row.get("comparison/numerically_eligible_pending_visual_review"),
            f"{label} numerical eligibility",
        )
        stem = str(row.get("case"))
        summary_path = safe_sibling(
            data_dir, f"{stem}-summary.json", context=f"{label} summary"
        )
        trace_path = safe_sibling(
            data_dir, row.get("trace/path"), context=f"{label} trace"
        )
        history_path = safe_sibling(
            data_dir, row.get("history/path"), context=f"{label} history"
        )
        require_equal(
            Path(str(row.get("artifact/summary_path"))).resolve(),
            summary_path,
            f"{label} declared summary path",
        )
        individual = read_json(summary_path)
        require_equal(individual, row, f"{label} aggregate/individual summary")
        recorded_trace = read_jsonl(trace_path)
        trace = validate_case_summary(
            label,
            individual,
            recorded_trace,
            history_path=history_path,
            trace_path=trace_path,
        )
        candidate = manifest_by_label[label]
        require_equal(
            individual.get("provenance/skin_file_sha256"),
            candidate["skin/file_identity"]["sha256"],
            f"{label} skin SHA-256",
        )
        identities = {
            "summary": file_identity(summary_path),
            "trace": file_identity(trace_path),
            "history": file_identity(history_path),
        }
        parameters = EXPECTED_PARAMETERS[label]
        cases.append(
            CaseInput(
                label=label,
                origin="new-2026-08-18",
                young_min_scale=parameters[0],
                prestrain_gain=parameters[1],
                summary_path=summary_path,
                trace_path=trace_path,
                history_path=history_path,
                summary=individual,
                trace=trace,
                identities=identities,
                history=TemporalHistory.open(label, history_path),
            )
        )
    return cases


def triangle_geometry(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    vector_norm = np.linalg.norm(vectors, axis=1)
    normals = vectors / np.maximum(vector_norm[:, None], np.finfo(np.float64).tiny)
    return vectors, 0.5 * vector_norm, normals


def interior_edge_adjacency(
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
    sorted_edges = edges[order]
    sorted_triangles = triangle_ids[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(np.diff(sorted_edges, axis=0), axis=1))]
    ends = np.r_[starts[1:], sorted_edges.shape[0]]
    counts = ends - starts
    interior = counts == 2
    unique_edges = sorted_edges[starts[interior]]
    tri_0 = sorted_triangles[starts[interior]]
    tri_1 = sorted_triangles[starts[interior] + 1]
    lengths = np.linalg.norm(
        points[unique_edges[:, 1]] - points[unique_edges[:, 0]], axis=1
    )
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        msg = "canonical skin contains invalid interior edge lengths"
        raise ValueError(msg)
    return unique_edges, tri_0, tri_1, lengths


def encoded_tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    encoded = np.asarray(mesh.cells, dtype=np.int64)
    if encoded.size != 5 * mesh.n_cells:
        msg = "prepared mesh connectivity is not pure tetrahedral"
        raise ValueError(msg)
    encoded = encoded.reshape(-1, 5)
    if not np.all(encoded[:, 0] == 4):
        msg = "prepared mesh contains non-tetrahedral cells"
        raise ValueError(msg)
    return encoded[:, 1:].copy()


def six_volume(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ij,ij->i",
        points[tets[:, 1]] - points[tets[:, 0]],
        np.cross(
            points[tets[:, 2]] - points[tets[:, 0]],
            points[tets[:, 3]] - points[tets[:, 0]],
        ),
    )


def map_global_ids(mesh_ids: np.ndarray, requested: np.ndarray) -> np.ndarray:
    if np.unique(mesh_ids).size != mesh_ids.size:
        msg = "mesh GlobalPointId values are not unique"
        raise ValueError(msg)
    order = np.argsort(mesh_ids)
    positions = np.searchsorted(mesh_ids[order], requested)
    if np.any(positions >= mesh_ids.size) or not np.array_equal(
        mesh_ids[order[positions]], requested
    ):
        msg = "canonical skin GlobalPointId values do not map to the volume mesh"
        raise ValueError(msg)
    return order[positions]


def bounds_camera(
    points: np.ndarray, *, aspect: float = 1.0, padding: float = 1.12
) -> tuple[np.ndarray, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    focus = 0.5 * (minimum + maximum)
    extent = maximum - minimum
    parallel_scale = 0.5 * max(float(extent[1]), float(extent[0]) / aspect)
    return focus, padding * parallel_scale


def build_surface_basis(  # noqa: PLR0915
    base_mesh: pv.UnstructuredGrid, skin: pv.PolyData
) -> SurfaceBasis:
    base_points = np.asarray(base_mesh.points, dtype=np.float64).copy()
    base_global_ids = (
        np.asarray(base_mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
        if GLOBAL_POINT_ID.vtk in base_mesh.point_data
        else np.arange(base_mesh.n_points, dtype=np.int64)
    )
    skin_global_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    skin_mesh_ids = map_global_ids(base_global_ids, skin_global_ids)
    skin_points = np.asarray(skin.points, dtype=np.float64).copy()
    if not np.array_equal(skin_points, base_points[skin_mesh_ids]):
        msg = "canonical skin rest points differ from the prepared mesh"
        raise ValueError(msg)
    encoded = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        msg = "canonical skin is not triangular"
        raise ValueError(msg)
    triangles = encoded[:, 1:].copy()
    rest_vectors, rest_area, _ = triangle_geometry(skin_points, triangles)
    if np.any(rest_area <= np.finfo(np.float64).eps):
        msg = "canonical skin contains a degenerate triangle"
        raise ValueError(msg)

    target = np.nan_to_num(
        np.asarray(base_mesh.point_data["Smile"], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    loss_mask = np.asarray(base_mesh.point_data["SmileLossMask"], dtype=bool)
    target_rms = float(
        np.linalg.norm(target[loss_mask]) / math.sqrt(int(loss_mask.sum()))
    )
    if not math.isfinite(target_rms) or target_rms <= 0.0:
        msg = "prepared Smile target RMS is invalid"
        raise ValueError(msg)
    target_skin = skin_points + target[skin_mesh_ids]
    _, target_area, target_normals = triangle_geometry(target_skin, triangles)
    if np.any(target_area <= np.finfo(np.float64).eps):
        msg = "Smile target skin contains a degenerate triangle"
        raise ValueError(msg)

    _, edge_tri_0, edge_tri_1, edge_length = interior_edge_adjacency(
        skin_points, triangles
    )
    contraction = np.asarray(skin.cell_data["ContractionPrestrainMask"], dtype=bool)
    contraction_edges = contraction[edge_tri_0] & contraction[edge_tri_1]
    contraction_tri_0 = edge_tri_0[contraction_edges]
    contraction_tri_1 = edge_tri_1[contraction_edges]
    contraction_weight = edge_length[contraction_edges]
    if contraction_weight.size == 0:
        msg = "canonical contraction ROI contains no interior edge"
        raise ValueError(msg)
    target_dihedral = np.arccos(
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

    face_triangle_mask = np.asarray(skin.cell_data["IsFaceTriangle"], dtype=bool)
    face_local_ids = np.unique(triangles[face_triangle_mask])
    face_focus, face_scale = bounds_camera(skin_points[face_local_ids])
    lip = np.asarray(base_mesh.point_data["IsLip"], dtype=bool)[skin_mesh_ids]
    lip &= np.isin(np.arange(skin.n_points), face_local_ids)
    if not np.any(lip):
        msg = "canonical surface has no mapped lip vertices"
        raise ValueError(msg)
    mouth_focus, mouth_scale = bounds_camera(skin_points[lip], padding=1.20)

    tets = encoded_tetrahedra(base_mesh)
    rest_six = six_volume(base_points, tets)
    if np.any(np.abs(rest_six) <= np.finfo(np.float64).eps):
        msg = "prepared volume mesh contains a zero-volume tetrahedron"
        raise ValueError(msg)
    return SurfaceBasis(
        base_points=base_points,
        base_cells=np.asarray(base_mesh.cells).copy(),
        base_celltypes=np.asarray(base_mesh.celltypes).copy(),
        base_global_ids=base_global_ids,
        tets=tets,
        rest_six_volume=rest_six,
        target=target,
        loss_mask=loss_mask,
        target_rms=target_rms,
        legacy_edges=surface_edges_for_mask(base_mesh, loss_mask),
        skin=skin,
        skin_points=skin_points,
        skin_mesh_ids=skin_mesh_ids,
        triangles=triangles,
        rest_area_vectors=rest_vectors,
        rest_area_vector_norm=np.linalg.norm(rest_vectors, axis=1),
        contraction_edge_tri_0=contraction_tri_0,
        contraction_edge_tri_1=contraction_tri_1,
        contraction_target_dihedral=target_dihedral,
        contraction_edge_weight=contraction_weight,
        face_triangle_mask=face_triangle_mask,
        face_focus=face_focus,
        face_parallel_scale=face_scale,
        mouth_focus=mouth_focus,
        mouth_parallel_scale=mouth_scale,
    )


def field_scalar(frame: pv.UnstructuredGrid, name: str) -> float:
    if name not in frame.field_data:
        msg = f"history frame has no field_data[{name!r}]"
        raise KeyError(msg)
    values = np.asarray(frame.field_data[name]).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        msg = f"history frame field {name!r} is not one finite scalar"
        raise ValueError(msg)
    return float(values[0])


def validate_frame(
    frame: pv.UnstructuredGrid,
    basis: SurfaceBasis,
    *,
    label: str,
    step: int,
    trace: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if (
        frame.n_points != basis.base_points.shape[0]
        or frame.n_cells != basis.tets.shape[0]
    ):
        msg = f"{label} step {step} mesh dimensions changed"
        raise ValueError(msg)
    if not np.array_equal(np.asarray(frame.points), basis.base_points):
        msg = f"{label} step {step} rest points changed"
        raise ValueError(msg)
    if not np.array_equal(
        np.asarray(frame.cells), basis.base_cells
    ) or not np.array_equal(np.asarray(frame.celltypes), basis.base_celltypes):
        msg = f"{label} step {step} volume topology changed"
        raise ValueError(msg)
    frame_ids = np.asarray(frame.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if not np.array_equal(frame_ids, basis.base_global_ids):
        msg = f"{label} step {step} GlobalPointId changed"
        raise ValueError(msg)
    require_close(field_scalar(frame, "inverse_step"), float(step), "history step")
    displacement = np.asarray(frame.point_data["Displacement"], dtype=np.float64)
    target = np.asarray(frame.point_data["TargetDisplacement"], dtype=np.float64)
    mask = np.asarray(frame.point_data["LossMask"], dtype=bool)
    if displacement.shape != basis.target.shape or not np.isfinite(displacement).all():
        msg = f"{label} step {step} displacement is malformed or non-finite"
        raise ValueError(msg)
    if not np.array_equal(target, basis.target) or not np.array_equal(
        mask, basis.loss_mask
    ):
        msg = f"{label} step {step} target or loss mask changed"
        raise ValueError(msg)
    error_rms = float(
        np.linalg.norm((displacement - target)[mask]) / math.sqrt(int(mask.sum()))
    )
    require_close(
        error_rms,
        float(trace["target/error_rms"]),
        f"{label} step {step} reconstructed error RMS",
    )
    require_close(
        field_scalar(frame, "inverse_error_rms"),
        error_rms,
        f"{label} step {step} stored error RMS",
    )
    return displacement, target


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or values.shape != weights.shape or not np.any(weights > 0.0):
        msg = "cannot compute a weighted RMS from empty or invalid data"
        raise ValueError(msg)
    return float(np.sqrt(np.sum(weights * np.square(values)) / np.sum(weights)))


def contraction_dihedral_rms(basis: SurfaceBasis, displacement: np.ndarray) -> float:
    deformed = basis.skin_points + displacement[basis.skin_mesh_ids]
    _, _, normals = triangle_geometry(deformed, basis.triangles)
    deformed_dihedral = np.arccos(
        np.clip(
            np.einsum(
                "ij,ij->i",
                normals[basis.contraction_edge_tri_0],
                normals[basis.contraction_edge_tri_1],
            ),
            -1.0,
            1.0,
        )
    )
    return weighted_rms(
        deformed_dihedral - basis.contraction_target_dihedral,
        basis.contraction_edge_weight,
    )


def deformation_warnings(
    basis: SurfaceBasis, displacement: np.ndarray
) -> dict[str, int | float | bool | str]:
    deformed = basis.base_points + displacement
    det_f = six_volume(deformed, basis.tets) / basis.rest_six_volume
    skin_deformed = basis.skin_points + displacement[basis.skin_mesh_ids]
    deformed_vectors, _, _ = triangle_geometry(skin_deformed, basis.triangles)
    signed_normal_ratio = np.einsum(
        "ij,ij->i", deformed_vectors, basis.rest_area_vectors
    ) / np.square(basis.rest_area_vector_norm)
    if not np.isfinite(det_f).all() or not np.isfinite(signed_normal_ratio).all():
        msg = "deformation warning diagnostics contain non-finite values"
        raise ValueError(msg)
    inverted = int(np.sum(det_f <= 0.0))
    folded = int(np.sum(signed_normal_ratio <= 0.0))
    warning_text = []
    if inverted:
        warning_text.append(f"{inverted} inverted tets")
    if folded:
        warning_text.append(f"{folded} folded triangles")
    return {
        "warning/inverted_tets": inverted,
        "warning/inverted_tet_fraction": float(np.mean(det_f <= 0.0)),
        "warning/detF_min": float(det_f.min()),
        "warning/skin_folded_triangles": folded,
        "warning/skin_folded_triangle_fraction": float(
            np.mean(signed_normal_ratio <= 0.0)
        ),
        "warning/skin_signed_normal_ratio_min": float(signed_normal_ratio.min()),
        "warning/has_inversion_or_fold": bool(inverted or folded),
        "warning/text": "; ".join(warning_text) if warning_text else "none",
        "warning/policy": "visual-review-only; not an eligibility gate",
    }


def frame_metrics(
    case: CaseInput,
    basis: SurfaceBasis,
    frame: pv.UnstructuredGrid,
    step: int,
) -> dict[str, Any]:
    displacement, target = validate_frame(
        frame,
        basis,
        label=case.label,
        step=step,
        trace=case.trace[step],
    )
    residual = displacement - target
    error_rms = float(
        np.linalg.norm(residual[basis.loss_mask])
        / math.sqrt(int(basis.loss_mask.sum()))
    )
    fidelity = error_rms / basis.target_rms
    legacy = bumpiness_metrics(
        mask=basis.loss_mask,
        edges=basis.legacy_edges,
        displacement=displacement,
        target=target,
    )
    laplacian = float(legacy["bumpiness/displacement_laplacian_rms"])
    dihedral = contraction_dihedral_rms(basis, displacement)
    metrics = {
        "candidate": case.label,
        "display_name": DISPLAY_NAMES[case.label],
        "origin": case.origin,
        "young_min_scale": case.young_min_scale,
        "prestrain_gain": case.prestrain_gain,
        "step": step,
        "target/error_rms_fraction_of_target": fidelity,
        "target/error_rms_m": error_rms,
        "target/error_rms_mm": 1.0e3 * error_rms,
        "bumpiness/contraction_target_relative_dihedral_rms_rad": dihedral,
        "bumpiness/contraction_target_relative_dihedral_rms_deg": math.degrees(
            dihedral
        ),
        "bumpiness/displacement_laplacian_rms_m": laplacian,
        "bumpiness/displacement_laplacian_rms_mm": 1.0e3 * laplacian,
        **deformation_warnings(basis, displacement),
    }
    validate_finite_json(metrics, context=f"{case.label} step {step} metrics")
    return metrics


def scan_cases(
    cases: list[CaseInput], basis: SurfaceBasis
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for case_index, case in enumerate(cases, start=1):
        logger.info("Scanning %s history (%d/%d)", case.label, case_index, len(cases))
        rows: list[dict[str, Any]] = []
        for step in range(EXPECTED_EVALUATIONS):
            rows.append(frame_metrics(case, basis, case.history.frame(step), step))
            if step % 10 == 0 or step == TERMINAL_STEP:
                logger.info(
                    "%s step %d: target %.6g, dihedral %.6g deg, lap %.6g mm",
                    case.label,
                    step,
                    rows[-1]["target/error_rms_fraction_of_target"],
                    rows[-1]["bumpiness/contraction_target_relative_dihedral_rms_deg"],
                    rows[-1]["bumpiness/displacement_laplacian_rms_mm"],
                )
        result[case.label] = rows
    return result


def select_checkpoints(
    trajectories: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    terminal = [trajectories[label][TERMINAL_STEP] for label in EXPECTED_CASE_ORDER]
    tau = max(float(row["target/error_rms_fraction_of_target"]) for row in terminal)
    matched: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    for label in EXPECTED_CASE_ORDER:
        nearest = min(
            trajectories[label],
            key=lambda row: (
                abs(float(row["target/error_rms_fraction_of_target"]) - tau),
                int(row["step"]),
            ),
        )
        selected = dict(nearest)
        selected["matching/tau"] = tau
        selected["matching/signed_error"] = (
            float(nearest["target/error_rms_fraction_of_target"]) - tau
        )
        selected["matching/absolute_error"] = abs(
            float(selected["matching/signed_error"])
        )
        matched.append(selected)
        selections[label] = {
            "step": int(selected["step"]),
            "fidelity": selected["target/error_rms_fraction_of_target"],
            "signed_error": selected["matching/signed_error"],
            "absolute_error": selected["matching/absolute_error"],
        }
    selected_fidelity = np.asarray(
        [row["target/error_rms_fraction_of_target"] for row in matched],
        dtype=np.float64,
    )
    matching = {
        "tau": tau,
        "tau_rule": "maximum terminal target-error fraction across all six cases",
        "selection_rule": (
            "nearest actual discrete checkpoint by absolute fidelity error; "
            "ties choose the earlier step; geometry is never interpolated"
        ),
        "selected_fidelity_min": float(selected_fidelity.min()),
        "selected_fidelity_max": float(selected_fidelity.max()),
        "selected_fidelity_spread": float(np.ptp(selected_fidelity)),
        "selections": selections,
        "physical_prefix_required": False,
        "inversion_fold_policy": "warning only",
    }
    return terminal, matched, matching


def relative_change_percent(new: float, reference: float) -> float:
    if not math.isfinite(new) or not math.isfinite(reference) or reference == 0.0:
        msg = "relative effect requires finite values and a nonzero reference"
        raise ValueError(msg)
    return 100.0 * (new / reference - 1.0)


def checkpoint_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label = {str(row["candidate"]): row for row in rows}
    comparisons = (
        ("prestrain-only vs baseline", "e100-p200", "e100-p000"),
        ("softening-only vs baseline", "e005-p000", "e100-p000"),
        ("combined vs baseline", "e005-p200", "e100-p000"),
        ("combined vs prestrain-only", "e005-p200", "e100-p200"),
        ("combined vs softening-only", "e005-p200", "e005-p000"),
        ("combined vs moderate", "e005-p200", "e025-p100"),
        ("no-skin vs baseline", "no-skin", "e100-p000"),
    )
    metrics = (
        "target/error_rms_fraction_of_target",
        "bumpiness/contraction_target_relative_dihedral_rms_deg",
        "bumpiness/displacement_laplacian_rms_mm",
    )
    effects: list[dict[str, Any]] = []
    for name, candidate, reference in comparisons:
        effects.append(
            {
                "comparison": name,
                "candidate": candidate,
                "reference": reference,
                **{
                    f"relative_change_percent/{metric}": relative_change_percent(
                        float(by_label[candidate][metric]),
                        float(by_label[reference][metric]),
                    )
                    for metric in metrics
                },
            }
        )
    return effects


CSV_FIELDS = (
    "candidate",
    "display_name",
    "origin",
    "young_min_scale",
    "prestrain_gain",
    "step",
    "target/error_rms_fraction_of_target",
    "target/error_rms_m",
    "target/error_rms_mm",
    "bumpiness/contraction_target_relative_dihedral_rms_rad",
    "bumpiness/contraction_target_relative_dihedral_rms_deg",
    "bumpiness/displacement_laplacian_rms_m",
    "bumpiness/displacement_laplacian_rms_mm",
    "warning/inverted_tets",
    "warning/inverted_tet_fraction",
    "warning/detF_min",
    "warning/skin_folded_triangles",
    "warning/skin_folded_triangle_fraction",
    "warning/skin_signed_normal_ratio_min",
    "warning/has_inversion_or_fold",
    "warning/text",
    "warning/policy",
)


def write_csv(path: Path, trajectories: dict[str, list[dict[str, Any]]]) -> None:
    rows = [row for label in EXPECTED_CASE_ORDER for row in trajectories[label]]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: Any, spec: str = ".5g") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def checkpoint_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| candidate | origin | step | error/target | error RMS mm | contraction dihedral deg | displacement Laplacian mm | inv tets | folds | warning only |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        (
            "| {candidate} | {origin} | {step} | {fidelity} | {error_mm} | "
            "{dihedral} | {laplacian} | {inverted} | {folded} | {warning} |".format(
                candidate=row["candidate"],
                origin=row["origin"],
                step=row["step"],
                fidelity=format_number(
                    row["target/error_rms_fraction_of_target"], ".6g"
                ),
                error_mm=format_number(row["target/error_rms_mm"], ".6g"),
                dihedral=format_number(
                    row["bumpiness/contraction_target_relative_dihedral_rms_deg"],
                    ".6g",
                ),
                laplacian=format_number(
                    row["bumpiness/displacement_laplacian_rms_mm"], ".6g"
                ),
                inverted=row["warning/inverted_tets"],
                folded=row["warning/skin_folded_triangles"],
                warning=row["warning/text"],
            )
        )
        for row in rows
    )
    return lines


def effect_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "Negative percentages are improvements because all three quantities are minimized.",
        "",
        "| comparison | target error change | dihedral change | Laplacian change |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            "| {comparison} | {target:.3f}% | {dihedral:.3f}% | {laplacian:.3f}% |".format(
                comparison=row["comparison"],
                target=row[
                    "relative_change_percent/target/error_rms_fraction_of_target"
                ],
                dihedral=row[
                    "relative_change_percent/bumpiness/"
                    "contraction_target_relative_dihedral_rms_deg"
                ],
                laplacian=row[
                    "relative_change_percent/bumpiness/displacement_laplacian_rms_mm"
                ],
            )
        )
        for row in rows
    )
    return lines


def write_table(
    path: Path,
    terminal: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    matching: dict[str, Any],
    terminal_effects: list[dict[str, Any]],
    matched_effects: list[dict[str, Any]],
) -> None:
    lines = [
        "# Exaggerated material screen checkpoints",
        "",
        (
            "Inverted tetrahedra and folded triangles are recorded as visual-review "
            "warnings only. They do not remove a trajectory or checkpoint."
        ),
        "",
        *checkpoint_table("Terminal fixed-budget checkpoint (step 40)", terminal),
        "",
        (
            "The common-fidelity target is "
            f"`{matching['tau']:.9g}`. Each row is the closest actual saved "
            "checkpoint; no geometry is interpolated."
        ),
        "",
        *checkpoint_table("Nearest discrete common-fidelity checkpoint", matched),
        "",
        *effect_table("Terminal relative effects", terminal_effects),
        "",
        *effect_table("Common-fidelity relative effects", matched_effects),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_trajectories(
    path: Path,
    trajectories: dict[str, list[dict[str, Any]]],
    terminal: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    matching: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(EXPECTED_CASE_ORDER)))
    terminal_by_label = {str(row["candidate"]): row for row in terminal}
    matched_by_label = {str(row["candidate"]): row for row in matched}
    for color, label in zip(colors, EXPECTED_CASE_ORDER, strict=True):
        rows = trajectories[label]
        step = np.asarray([row["step"] for row in rows])
        fidelity = np.asarray(
            [row["target/error_rms_fraction_of_target"] for row in rows]
        )
        dihedral = np.asarray(
            [
                row["bumpiness/contraction_target_relative_dihedral_rms_deg"]
                for row in rows
            ]
        )
        laplacian = np.asarray(
            [row["bumpiness/displacement_laplacian_rms_mm"] for row in rows]
        )
        axes[0].plot(step, fidelity, color=color, label=label)
        axes[1].plot(fidelity, dihedral, color=color, label=label)
        axes[2].plot(fidelity, laplacian, color=color, label=label)
        for marker, lookup in (("s", terminal_by_label), ("o", matched_by_label)):
            point = lookup[label]
            x_fidelity = point["target/error_rms_fraction_of_target"]
            axes[0].scatter(
                [point["step"]], [x_fidelity], color=[color], marker=marker, s=34
            )
            axes[1].scatter(
                [x_fidelity],
                [point["bumpiness/contraction_target_relative_dihedral_rms_deg"]],
                color=[color],
                marker=marker,
                s=34,
            )
            axes[2].scatter(
                [x_fidelity],
                [point["bumpiness/displacement_laplacian_rms_mm"]],
                color=[color],
                marker=marker,
                s=34,
            )
    axes[0].axhline(float(matching["tau"]), color="black", linestyle=":")
    axes[0].set_xlabel("inverse evaluation step")
    axes[0].set_ylabel("target error RMS / target RMS")
    axes[0].set_title("Target fit (square=terminal, circle=matched)")
    axes[1].set_xlabel("target error RMS / target RMS")
    axes[1].set_ylabel("contraction target-relative dihedral RMS [deg]")
    axes[1].set_title("Target-fit / surface-roughness trajectory")
    axes[2].set_xlabel("target error RMS / target RMS")
    axes[2].set_ylabel("displacement umbrella-Laplacian RMS [mm]")
    axes[2].set_title("Target-fit / displacement-roughness trajectory")
    for axis in axes:
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize="small", ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def deformed_face(basis: SurfaceBasis, displacement: np.ndarray) -> pv.PolyData:
    surface = basis.skin.copy(deep=True)
    surface.points = basis.skin_points + displacement[basis.skin_mesh_ids]
    face = surface.extract_cells(np.flatnonzero(basis.face_triangle_mask))
    return face.extract_surface(algorithm="dataset_surface")


def frame_displacement(case: CaseInput, step: int) -> np.ndarray:
    frame = case.history.frame(step, deep_copy=True)
    displacement = np.asarray(frame.point_data["Displacement"], dtype=np.float64)
    if not np.isfinite(displacement).all():
        msg = f"{case.label} render frame {step} contains non-finite displacement"
        raise ValueError(msg)
    return displacement


def render_contact_sheet(
    path: Path,
    *,
    basis: SurfaceBasis,
    cases: list[CaseInput],
    checkpoints: list[dict[str, Any]],
    title: str,
) -> None:
    checkpoint_by_label = {str(row["candidate"]): row for row in checkpoints}
    target_face = deformed_face(basis, basis.target)
    surfaces: list[tuple[str, pv.PolyData, str]] = [
        ("Target", target_face, "reference target")
    ]
    for case in cases:
        checkpoint = checkpoint_by_label[case.label]
        step = int(checkpoint["step"])
        face = deformed_face(basis, frame_displacement(case, step))
        annotation = (
            f"{case.label} | step {step}\n"
            f"error/target={checkpoint['target/error_rms_fraction_of_target']:.4f}"
        )
        surfaces.append((case.label, face, annotation))

    views = (
        (
            "front",
            np.asarray((0.0, 0.0, 1.0)),
            basis.face_focus,
            basis.face_parallel_scale,
        ),
        (
            "30 degree",
            np.asarray(
                (math.sin(math.radians(30.0)), 0.0, math.cos(math.radians(30.0)))
            ),
            basis.face_focus,
            basis.face_parallel_scale,
        ),
        (
            "mouth closeup",
            np.asarray((0.0, 0.0, 1.0)),
            basis.mouth_focus,
            basis.mouth_parallel_scale,
        ),
    )
    plotter = pv.Plotter(
        shape=(len(views), len(surfaces)),
        off_screen=True,
        window_size=(2800, 1200),
        lighting="light kit",
        border=False,
    )
    plotter.set_background("white")
    for row, (view_name, direction, focus, parallel_scale) in enumerate(views):
        for column, (label, surface, annotation) in enumerate(surfaces):
            plotter.subplot(row, column)
            plotter.add_mesh(
                surface,
                color="#d8b49c",
                smooth_shading=True,
                specular=0.15,
                show_edges=False,
            )
            plotter.add_text(
                f"{view_name} | {annotation}",
                position="upper_left",
                font_size=8,
                color="black",
            )
            plotter.enable_parallel_projection()
            camera_focus = np.asarray(focus, dtype=np.float64)
            plotter.camera.position = tuple(camera_focus + 0.30 * direction)
            plotter.camera.focal_point = tuple(camera_focus)
            plotter.camera.up = (0.0, 1.0, 0.0)
            plotter.camera.parallel_scale = float(parallel_scale)
            if label == "Target" and row == 0:
                plotter.add_text(
                    title,
                    position="lower_left",
                    font_size=9,
                    color="black",
                )
    plotter.screenshot(path)
    plotter.close()


def validate_config(cfg: Config) -> None:
    require_equal(cfg.input_mesh.resolve(), PREPARED_MESH.resolve(), "input mesh")
    require_equal(cfg.input_skin.resolve(), SOURCE_SKIN.resolve(), "canonical skin")
    require_equal(
        cfg.input_new_summary.resolve(),
        (GROUP_DIR / "data" / NEW_SUMMARY_NAME).resolve(),
        "new summary path",
    )
    require_equal(
        cfg.input_candidates.resolve(),
        (GROUP_DIR / "data/10-exaggerated-materials-manifest.json").resolve(),
        "candidate manifest path",
    )
    if str(mpl.get_backend()).lower() != "agg":
        msg = f"analysis requires the Agg backend, got {mpl.get_backend()}"
        raise RuntimeError(msg)
    outputs = (
        cfg.output_json,
        cfg.output_csv,
        cfg.output_table,
        cfg.output_plot,
        cfg.output_terminal_views,
        cfg.output_matched_views,
    )
    if len({path.resolve() for path in outputs}) != len(outputs):
        msg = "analysis output paths must be distinct"
        raise ValueError(msg)


def lightweight_contract_smoke() -> dict[str, Any]:
    cases = [
        load_old_case(spec, verify_history_digest=False) for spec in OLD_CASE_SPECS
    ]
    synthetic = {
        label: [
            {
                "candidate": label,
                "step": step,
                "target/error_rms_fraction_of_target": 1.0 - 0.01 * step,
            }
            for step in range(EXPECTED_EVALUATIONS)
        ]
        for label in EXPECTED_CASE_ORDER
    }
    terminal, matched, matching = select_checkpoints(synthetic)
    require_equal(len(terminal), len(EXPECTED_CASE_ORDER), "smoke terminal count")
    require_equal(len(matched), len(EXPECTED_CASE_ORDER), "smoke matched count")
    return {
        "old_cases": [case.label for case in cases],
        "history_times_verified": True,
        "synthetic_tau": matching["tau"],
    }


def run(cfg: Config) -> None:
    validate_config(cfg)
    manifest = load_manifest(cfg)
    require_equal(
        file_identity(cfg.input_skin),
        {
            "size_bytes": SOURCE_SKIN_SIZE_BYTES,
            "sha256": SOURCE_SKIN_SHA256,
        },
        "canonical skin identity",
    )
    mesh_identity = file_identity(cfg.input_mesh)
    require_equal(
        mesh_identity,
        manifest["input_mesh_identity"],
        "prepared mesh identity",
    )
    old_cases = [
        load_old_case(spec, verify_history_digest=True) for spec in OLD_CASE_SPECS
    ]
    new_cases = load_new_cases(cfg, manifest)
    by_label = {case.label: case for case in (*old_cases, *new_cases)}
    require_equal(set(by_label), set(EXPECTED_CASE_ORDER), "aggregate case labels")
    cases = [by_label[label] for label in EXPECTED_CASE_ORDER]

    base_mesh = pv.read(cfg.input_mesh)
    if not isinstance(base_mesh, pv.UnstructuredGrid):
        base_mesh = base_mesh.cast_to_unstructured_grid()
    skin = pv.read(cfg.input_skin)
    if not isinstance(skin, pv.PolyData):
        msg = f"canonical skin read as {type(skin).__name__}, expected PolyData"
        raise TypeError(msg)
    basis = build_surface_basis(base_mesh, skin)
    trajectories = scan_cases(cases, basis)
    terminal, matched, matching = select_checkpoints(trajectories)
    terminal_effects = checkpoint_effects(terminal)
    matched_effects = checkpoint_effects(matched)

    source = {
        "prepared_mesh": {"path": str(cfg.input_mesh), **mesh_identity},
        "canonical_skin": {
            "path": str(cfg.input_skin),
            **file_identity(cfg.input_skin),
        },
        "candidate_manifest": {
            "path": str(cfg.input_candidates),
            **file_identity(cfg.input_candidates),
        },
        "new_aggregate_summary": {
            "path": str(cfg.input_new_summary),
            **file_identity(cfg.input_new_summary),
        },
        "cases": {
            case.label: {
                "origin": case.origin,
                "summary_path": str(case.summary_path),
                "trace_path": str(case.trace_path),
                "history_path": str(case.history_path),
                "identities": case.identities,
            }
            for case in cases
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "kind": "human-face-smile-exaggerated-material-target-bumpiness-screen",
        "design": "exaggerated-heterogeneous-mechanism-screen",
        "purpose": (
            "deliberately exaggerated mechanism contrast; not a physiological "
            "material calibration"
        ),
        "case_order": list(EXPECTED_CASE_ORDER),
        "protocol": {
            "evaluations_per_case": EXPECTED_EVALUATIONS,
            "terminal_step": TERMINAL_STEP,
            "fixed_learning_rate": EXPECTED_LR,
            "fresh_zero_activation": True,
            "activation_mode": "per-muscle-tet-6dof-unconstrained",
            "reused_cases": [spec.label for spec in OLD_CASE_SPECS],
            "new_cases": list(EXPECTED_NEW_LABELS),
        },
        "acceptance_policy": {
            "hard_failures": [
                "failed forward or adjoint solve",
                "non-finite trajectory or metric",
                "missing, corrupt, identity-mismatched, or topology-mismatched artifact",
            ],
            "visual_review_warnings_only": [
                "inverted tetrahedra",
                "folded skin triangles",
            ],
            "warning_effect": (
                "record and render the state; never truncate the trajectory or "
                "exclude a checkpoint"
            ),
        },
        "metric_definitions": {
            "target/error_rms_fraction_of_target": (
                "RMS(Displacement-TargetDisplacement) / RMS(TargetDisplacement) "
                "on SmileLossMask"
            ),
            "bumpiness/contraction_target_relative_dihedral_rms": (
                "rest-edge-length-weighted RMS difference between deformed and "
                "target dihedral angles on contraction-ROI interior edges"
            ),
            "bumpiness/displacement_laplacian_rms": (
                "legacy fixed-topology umbrella-Laplacian RMS of displacement on "
                "the SmileLossMask surface"
            ),
        },
        "matching": matching,
        "terminal_checkpoints": terminal,
        "matched_checkpoints": matched,
        "terminal_effects": terminal_effects,
        "matched_effects": matched_effects,
        "trajectories": trajectories,
        "source": source,
        "render_policy": {
            "surface": "canonical e100-p000 skin mapped by GlobalPointId for all cases",
            "views": ["front", "30 degree", "mouth closeup"],
            "projection": "parallel",
            "appearance": "plain fixed color, white background, smooth shading",
            "terminal_contact_sheet": str(cfg.output_terminal_views),
            "matched_contact_sheet": str(cfg.output_matched_views),
        },
        "limitations": [
            "single deterministic target and no replicate",
            "exaggerated material amplitudes are not physiological estimates",
            "umbrella-Laplacian values are comparable only on this fixed topology",
            "nearest-fidelity comparison uses actual saved states and may retain a reported fidelity spread",
            "inversion and fold counts are not proxies for visible artifact severity",
        ],
    }
    validate_finite_json(payload, context="analysis payload")
    cfg.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(cfg.output_csv, trajectories)
    write_table(
        cfg.output_table,
        terminal,
        matched,
        matching,
        terminal_effects,
        matched_effects,
    )
    plot_trajectories(cfg.output_plot, trajectories, terminal, matched, matching)
    render_contact_sheet(
        cfg.output_terminal_views,
        basis=basis,
        cases=cases,
        checkpoints=terminal,
        title="terminal fixed-budget states",
    )
    render_contact_sheet(
        cfg.output_matched_views,
        basis=basis,
        cases=cases,
        checkpoints=matched,
        title="nearest discrete common-fidelity states",
    )
    for index, row in enumerate(terminal):
        cherries.set_step(index)
        cherries.log_metrics(
            {
                f"{row['candidate']}/terminal_target_error_fraction": row[
                    "target/error_rms_fraction_of_target"
                ],
                f"{row['candidate']}/terminal_dihedral_rms_deg": row[
                    "bumpiness/contraction_target_relative_dihedral_rms_deg"
                ],
                f"{row['candidate']}/terminal_laplacian_rms_mm": row[
                    "bumpiness/displacement_laplacian_rms_mm"
                ],
            }
        )
    for path in (
        cfg.output_json,
        cfg.output_csv,
        cfg.output_table,
        cfg.output_plot,
        cfg.output_terminal_views,
        cfg.output_matched_views,
    ):
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    cherries.main(run)
