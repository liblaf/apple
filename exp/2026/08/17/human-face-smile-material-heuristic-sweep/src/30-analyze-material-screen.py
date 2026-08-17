from __future__ import annotations

import csv
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
from _human_face_config import SMILE_LOSS_MASK, SMILE_TARGET
from _material_heuristics import (
    file_sha256,
    skin_material_content_hash,
    skin_solver_content_hash,
    skin_topology_content_hash,
)
from _reference import PREPARED_MESH, enable_reference_modules
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
SOURCE_SCHEMA_VERSION = 2
BASELINE_LABEL = "e100-p000"
NO_SKIN_LABEL = "no-skin"
MATERIAL_LABELS = (
    "e100-p000",
    "e100-p050",
    "e100-p100",
    "e025-p000",
    "e025-p050",
    "e025-p100",
)
EXPECTED_LABELS = (*MATERIAL_LABELS, NO_SKIN_LABEL)
EXPECTED_PARAMETERS = {
    "e100-p000": (1.0, 0.0),
    "e100-p050": (1.0, 0.5),
    "e100-p100": (1.0, 1.0),
    "e025-p000": (0.25, 0.0),
    "e025-p050": (0.25, 0.5),
    "e025-p100": (0.25, 1.0),
}
EXPECTED_SETUPS = {
    "e100-p000": "skin-no-prestrain",
    "e100-p050": "skin-estimated-prestrain",
    "e100-p100": "skin-estimated-prestrain",
    "e025-p000": "skin-no-prestrain",
    "e025-p050": "skin-estimated-prestrain",
    "e025-p100": "skin-estimated-prestrain",
    NO_SKIN_LABEL: "no-skin",
}
EXPECTED_TRACE_STEPS = 41
EXPECTED_LR = 0.3
EXPECTED_LOSS_SCALE = 1.0e6
JSON_RTOL = 1.0e-10
JSON_ATOL = 1.0e-12
ROI_ARRAYS = (
    "RestArea",
    "TargetArea",
    "TargetRestAreaRatio",
    "LogAreaRaw",
    "LogAreaDeadbanded",
    "LogAreaCapped",
    "LogAreaDiffused",
    "IsFaceTriangle",
    "FiniteTargetTriangle",
    "EligibleMaterialTriangle",
    "ExpansionMaterialMask",
    "ContractionPrestrainMask",
    "ExpansionSeverityLogSoftThreshold",
    "ContractionSeverityLogSoftThreshold",
    "ContractionSeverityLogCapped",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_summary: Path = cherries.input("20-material-screen-summary.json")
    input_candidates: Path = cherries.input("10-material-candidates-manifest.json")
    input_mesh: Path = cherries.input(PREPARED_MESH)
    output_json: Path = cherries.output("30-material-screen-analysis.json", mkdir=True)
    output_csv: Path = cherries.output("30-material-screen-analysis.csv", mkdir=True)
    output_table: Path = cherries.output("30-material-screen-table.md", mkdir=True)
    output_trajectory_plot: Path = cherries.output(
        "30-material-screen-trajectories.png", mkdir=True
    )
    output_screen_plot: Path = cherries.output(
        "30-material-screen-matched.png", mkdir=True
    )

    max_fidelity_spread: float = 0.001
    min_det_f_q001: float = 0.20
    min_skin_area_ratio_q001: float = 0.10
    max_skin_area_ratio_q999: float = 10.0
    min_muscle_activation_eigenvalue: float = 1.0e-6
    pareto_rtol: float = 1.0e-6
    pareto_atol: float = 1.0e-12


@dataclass(frozen=True)
class CandidateArtifacts:
    manifest: dict[str, Any]
    skins: dict[str, pv.PolyData]
    skin_paths: dict[str, Path]
    identities: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SurfaceBasis:
    points: np.ndarray
    triangles: np.ndarray
    mesh_point_ids: np.ndarray
    rest_area: np.ndarray
    target_area: np.ndarray
    target_points: np.ndarray
    target_displacement: np.ndarray
    target_normals: np.ndarray
    target_vertex_normals: np.ndarray
    all_edges: np.ndarray
    edge_tri_0: np.ndarray
    edge_tri_1: np.ndarray
    edge_length: np.ndarray
    incident_triangle_count: np.ndarray
    roi_masks: dict[str, np.ndarray]


@dataclass
class TemporalHistory:
    label: str
    case: dict[str, Any]
    path: Path
    pyvista_reader: Any
    times: np.ndarray

    @classmethod
    def open(
        cls, label: str, case: dict[str, Any], *, source_dir: Path
    ) -> TemporalHistory:
        history_name = case.get("history/path")
        if not isinstance(history_name, str) or Path(history_name).name != history_name:
            msg = (
                f"case {label!r} history/path is not a safe basename: {history_name!r}"
            )
            raise ValueError(msg)
        path = (source_dir / history_name).resolve()
        if path.parent != source_dir.resolve() or not path.is_file():
            msg = f"case {label!r} history is missing or escapes source data: {path}"
            raise FileNotFoundError(msg)
        pyvista_reader = pv.get_reader(path)
        vtk_reader = pyvista_reader.reader
        vtk_reader.UpdateInformation()
        information = vtk_reader.GetOutputInformation(0)
        key = StreamingPipeline.TIME_STEPS()
        if not information.Has(key):
            msg = f"case {label!r} VTKHDF history exposes no TIME_STEPS"
            raise ValueError(msg)
        times = np.asarray(
            [information.Get(key, index) for index in range(information.Length(key))],
            dtype=np.float64,
        )
        expected = np.arange(EXPECTED_TRACE_STEPS, dtype=np.float64)
        if times.shape != expected.shape or not np.allclose(
            times, expected, rtol=0.0, atol=1.0e-12
        ):
            msg = f"case {label!r} history times are not exact steps 0..40"
            raise ValueError(msg)
        return cls(
            label=label,
            case=case,
            path=path,
            pyvista_reader=pyvista_reader,
            times=times,
        )

    def frame(self, step: int, *, deep_copy: bool = False) -> pv.UnstructuredGrid:
        if not 0 <= step < self.times.size:
            msg = f"case {self.label!r} has no temporal step {step}"
            raise IndexError(msg)
        vtk_reader = self.pyvista_reader.reader
        vtk_reader.UpdateTimeStep(float(self.times[step]))
        output = pv.wrap(vtk_reader.GetOutputDataObject(0))
        if deep_copy:
            output = output.copy(deep=True)
        if not isinstance(output, pv.UnstructuredGrid):
            output = output.cast_to_unstructured_grid()
        return output


def reject_json_constant(value: str) -> None:
    msg = f"non-standard JSON constant {value!r}"
    raise ValueError(msg)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_json_constant
    )
    if not isinstance(payload, dict):
        msg = f"expected a JSON object in {path}"
        raise TypeError(msg)
    validate_finite_json(payload, context=str(path))
    return payload


def validate_finite_json(value: Any, *, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{context} contains non-finite value {value}"
        raise ValueError(msg)
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite_json(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite_json(item, context=f"{context}[{index}]")


def require_keys(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = sorted(set(keys) - set(mapping))
    if missing:
        msg = f"{context} is missing keys {missing}"
        raise KeyError(msg)


def require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        msg = f"{context}: expected {expected!r}, got {actual!r}"
        raise ValueError(msg)


def require_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=JSON_RTOL, abs_tol=JSON_ATOL):
        msg = f"{context}: expected {expected:.17g}, got {actual:.17g}"
        raise ValueError(msg)


def require_true(actual: Any, context: str) -> None:
    if actual is not True:
        msg = f"{context}: expected True, got {actual!r}"
        raise ValueError(msg)


def file_identity(path: Path, *, hash_content: bool = True) -> dict[str, Any]:
    if not path.is_file():
        msg = f"artifact does not exist: {path}"
        raise FileNotFoundError(msg)
    identity: dict[str, Any] = {"size_bytes": path.stat().st_size}
    if hash_content:
        identity["sha256"] = file_sha256(path)
    return identity


def resolve_candidate_skin(manifest_path: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        msg = f"candidate skin/path must be relative, got {path}"
        raise ValueError(msg)
    resolved = (manifest_path.parent / path).resolve()
    if not resolved.is_relative_to(manifest_path.parent.resolve()):
        msg = f"candidate skin escapes manifest data directory: {path}"
        raise ValueError(msg)
    return resolved


def load_candidate_artifacts(cfg: Config) -> CandidateArtifacts:  # noqa: C901, PLR0915
    manifest = read_json(cfg.input_candidates)
    require_keys(
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
            "validation_errors",
            "candidate_validation_errors",
            "n_candidates",
            "candidates",
        ),
        "candidate manifest",
    )
    require_equal(manifest["schema_version"], 2, "candidate manifest schema")
    require_true(manifest["complete"], "candidate manifest complete")
    require_true(
        manifest["input_mesh_identity_verified_stable"],
        "candidate manifest stable input identity",
    )
    require_equal(manifest["validation_errors"], [], "candidate validation errors")
    require_equal(
        manifest["candidate_validation_errors"],
        {},
        "per-candidate validation errors",
    )
    require_equal(
        Path(str(manifest["input_mesh"])).resolve(),
        cfg.input_mesh.resolve(),
        "candidate manifest input mesh",
    )
    expected_mesh_identity = manifest["input_mesh_identity"]
    require_equal(
        file_identity(cfg.input_mesh), expected_mesh_identity, "input mesh identity"
    )
    require_equal(
        manifest["grid"],
        {
            "young_min_scales": [1.0, 0.25],
            "prestrain_gains": [0.0, 0.5, 1.0],
        },
        "fixed material grid",
    )
    require_equal(manifest["target"], "Smile", "candidate target")
    require_equal(
        manifest["selection"],
        "all surface-triangle vertices are finite IsFace points",
        "candidate selection",
    )
    candidates = manifest["candidates"]
    if not isinstance(candidates, list) or not all(
        isinstance(row, dict) for row in candidates
    ):
        msg = "candidate manifest candidates must be a list of objects"
        raise TypeError(msg)
    require_equal(len(candidates), 6, "candidate count")
    require_equal(manifest["n_candidates"], 6, "manifest n_candidates")
    labels = tuple(str(row.get("label")) for row in candidates)
    require_equal(labels, MATERIAL_LABELS, "candidate labels/order")

    skins: dict[str, pv.PolyData] = {}
    skin_paths: dict[str, Path] = {}
    identities: dict[str, dict[str, Any]] = {}
    for row in candidates:
        label = str(row["label"])
        require_keys(
            row,
            (
                "schema_version",
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
                "validation/ok",
                "validation/errors",
            ),
            f"candidate {label}",
        )
        require_equal(row["schema_version"], 2, f"candidate {label} schema")
        require_equal(
            (float(row["young_min_scale"]), float(row["prestrain_gain"])),
            EXPECTED_PARAMETERS[label],
            f"candidate {label} parameters",
        )
        require_true(row["validation/ok"], f"candidate {label} validation")
        require_equal(
            row["validation/errors"], [], f"candidate {label} validation errors"
        )
        require_true(row["readback/ok"], f"candidate {label} readback")
        require_equal(row["readback/errors"], [], f"candidate {label} readback errors")
        path = resolve_candidate_skin(cfg.input_candidates, row["skin/path"])
        identity = file_identity(path)
        require_equal(identity, row["skin/file_identity"], f"candidate {label} VTP")
        skin = pv.read(path)
        if not isinstance(skin, pv.PolyData):
            msg = f"candidate {label} read as {type(skin).__name__}, expected PolyData"
            raise TypeError(msg)
        live_hashes = {
            "topology": skin_topology_content_hash(skin),
            "material": skin_material_content_hash(skin),
            "solver": skin_solver_content_hash(skin),
        }
        for name, digest in live_hashes.items():
            require_equal(
                digest,
                str(row[f"content/{name}_sha256"]),
                f"candidate {label} live {name} hash",
            )
        require_equal(skin.n_points, row["content/n_points"], f"{label} points")
        require_equal(skin.n_cells, row["content/n_triangles"], f"{label} triangles")
        skins[label] = skin
        skin_paths[label] = path
        identities[label] = {
            **identity,
            **{f"{k}_sha256": v for k, v in live_hashes.items()},
        }

    baseline = skins[BASELINE_LABEL]
    baseline_ids = np.asarray(baseline.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    for label in MATERIAL_LABELS[1:]:
        skin = skins[label]
        if not np.array_equal(np.asarray(skin.faces), np.asarray(baseline.faces)):
            msg = f"candidate {label} topology differs from baseline"
            raise ValueError(msg)
        if not np.array_equal(np.asarray(skin.points), np.asarray(baseline.points)):
            msg = f"candidate {label} rest points differ from baseline"
            raise ValueError(msg)
        if not np.array_equal(
            np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64),
            baseline_ids,
        ):
            msg = f"candidate {label} GlobalPointId differs from baseline"
            raise ValueError(msg)
        for name in ROI_ARRAYS:
            if not np.array_equal(
                np.asarray(skin.cell_data[name]),
                np.asarray(baseline.cell_data[name]),
                equal_nan=True,
            ):
                msg = f"candidate {label} canonical ROI field {name!r} changed"
                raise ValueError(msg)
    return CandidateArtifacts(
        manifest=manifest,
        skins=skins,
        skin_paths=skin_paths,
        identities=identities,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                msg = f"blank JSONL row at {path}:{line_number}"
                raise ValueError(msg)
            row = json.loads(line, parse_constant=reject_json_constant)
            if not isinstance(row, dict):
                msg = f"JSONL row is not an object at {path}:{line_number}"
                raise TypeError(msg)
            validate_finite_json(row, context=f"{path}:{line_number}")
            rows.append(row)
    return rows


def expected_case_stem(label: str) -> str:
    setup = EXPECTED_SETUPS[label]
    return f"20-human-face-smile-{setup}-lr3-material-{label}-screen"


def validate_trace(label: str, case: dict[str, Any]) -> list[dict[str, Any]]:
    trace = case.get("trace")
    if not isinstance(trace, list) or not all(isinstance(row, dict) for row in trace):
        msg = f"case {label!r} trace must be a list of objects"
        raise TypeError(msg)
    require_equal(len(trace), EXPECTED_TRACE_STEPS, f"case {label} trace length")
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
        require_equal(int(float(row.get("step", -1))), step, f"{label} trace step")
        require_true(row.get("forward/success"), f"{label} step {step} forward")
        require_true(row.get("adjoint/success"), f"{label} step {step} adjoint")
        require_equal(
            float(row.get("inverse/lr", math.nan)),
            EXPECTED_LR,
            f"{label} step {step} fixed learning rate",
        )
        for key in finite_keys:
            value = float(row.get(key, math.nan))
            if not math.isfinite(value):
                msg = f"case {label!r} step {step} has non-finite {key}"
                raise ValueError(msg)
        require_close(
            float(row["loss/total"]),
            float(row["loss/mm2"]),
            f"{label} step {step} loss total/mm2",
        )
        require_close(
            float(row["loss/mm2"]),
            EXPECTED_LOSS_SCALE * float(row["loss/m2"]),
            f"{label} step {step} loss scale",
        )
    require_equal(float(trace[0]["activation_inv/rms"]), 0.0, f"{label} step0 RMS")
    require_equal(float(trace[0]["activation_inv/max_abs"]), 0.0, f"{label} step0 max")
    best_step = int(case["best/step"])
    if not 0 <= best_step < len(trace):
        msg = f"case {label!r} best step {best_step} is outside trace"
        raise ValueError(msg)
    require_true(trace[best_step].get("best/accepted"), f"{label} best accepted")
    return list(trace)


def validate_live_trace(
    label: str, embedded: list[dict[str, Any]], live: list[dict[str, Any]]
) -> None:
    require_equal(len(live), len(embedded), f"case {label} live trace length")
    for step, (source, recorded) in enumerate(zip(embedded, live, strict=True)):
        expected = {
            key: value for key, value in source.items() if key != "time/live_plot_s"
        }
        require_equal(
            set(recorded), set(expected), f"case {label} step {step} JSONL keys"
        )
        for key, expected_value in expected.items():
            actual = recorded[key]
            if isinstance(expected_value, float):
                require_close(
                    float(actual),
                    expected_value,
                    f"case {label} step {step} JSONL {key}",
                )
            else:
                require_equal(
                    actual, expected_value, f"case {label} step {step} JSONL {key}"
                )


def load_source_summary(
    cfg: Config,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, TemporalHistory]]:
    summary = read_json(cfg.input_summary)
    expected_top = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "complete": True,
        "stage": "screen",
        "candidate_set": "all-with-no-skin",
        "fresh_zero_activation": True,
        "activation_mode": "per-muscle-tet-6dof-unconstrained",
        "activation_shared": False,
        "activation_transferred_between_candidates": False,
        "forward_builder_shared_between_candidates": False,
        "inverse_lr": EXPECTED_LR,
        "inverse_max_steps": 40,
        "mandatory_baseline_steps": 40,
        "hard_failures": [],
        "baseline_available": True,
    }
    for key, expected in expected_top.items():
        require_equal(summary.get(key), expected, f"source summary {key}")
    require_equal(
        Path(str(summary.get("input_mesh"))).resolve(),
        cfg.input_mesh.resolve(),
        "source summary input mesh",
    )
    require_equal(
        Path(str(summary.get("input_candidates"))).resolve(),
        cfg.input_candidates.resolve(),
        "source summary candidate manifest",
    )
    if str(summary.get("plot_backend", "")).lower() != "agg":
        msg = f"source plot backend is not Agg: {summary.get('plot_backend')!r}"
        raise ValueError(msg)
    raw_cases = summary.get("cases")
    if not isinstance(raw_cases, list) or not all(
        isinstance(case, dict) for case in raw_cases
    ):
        msg = "source cases must be a list of objects"
        raise TypeError(msg)
    cases = list(raw_cases)
    labels = tuple(str(case.get("candidate")) for case in cases)
    require_equal(labels, EXPECTED_LABELS, "source case labels/order")
    source_dir = cfg.input_summary.resolve().parent
    histories: dict[str, TemporalHistory] = {}
    expected_convergence_failures: list[str] = []
    for label, case in zip(labels, cases, strict=True):
        stem = expected_case_stem(label)
        expected_parameters = EXPECTED_PARAMETERS.get(label, (None, None))
        expected_case = {
            "candidate": label,
            "stage": "screen",
            "status": "ok",
            "validation/errors": [],
            "case": stem,
            "case/label": f"material-{label}-screen",
            "case/setup": EXPECTED_SETUPS[label],
            "target/name": "smile",
            "activation/mode": "per-muscle-tet-6dof",
            "activation/shared": False,
            "activation/range_clamping": False,
            "activation_inv/initial_rms": 0.0,
            "activation_inv/initial_max_abs": 0.0,
            "initial_displacement/enabled": False,
            "baseline/completed": True,
            "baseline/evaluations": EXPECTED_TRACE_STEPS,
            "baseline/evaluations_expected": EXPECTED_TRACE_STEPS,
            "baseline/mandatory_optimizer_steps": 40,
            "baseline/fixed_lr": EXPECTED_LR,
            "baseline/lr_deviation_count": 0,
            "inverse/evaluations": EXPECTED_TRACE_STEPS,
            "best/step": EXPECTED_TRACE_STEPS - 1,
            "history/format": "VTKHDFTemporalUnstructuredGrid",
            "history/frames": EXPECTED_TRACE_STEPS,
            "history/path": f"{stem}-steps.vtkhdf",
            "trace/path": f"{stem}-trace.jsonl",
            "candidate/young_min_scale": expected_parameters[0],
            "candidate/prestrain_gain": expected_parameters[1],
            "scientific/is_control": label == NO_SKIN_LABEL,
            "scientific/control_type": NO_SKIN_LABEL
            if label == NO_SKIN_LABEL
            else None,
            "scientific/quality_surface_candidate": BASELINE_LABEL
            if label == NO_SKIN_LABEL
            else label,
        }
        for key, expected in expected_case.items():
            require_equal(case.get(key), expected, f"case {label} {key}")
        require_equal(
            int(case["n_activation_parameter_dofs"]),
            6 * int(case["n_active_tets"]),
            f"case {label} activation DoFs",
        )
        require_equal(
            case.get("skin/enabled"), label != NO_SKIN_LABEL, f"case {label} skin"
        )
        if label != NO_SKIN_LABEL:
            require_equal(case.get("label"), label, f"case {label} embedded label")
        trace = validate_trace(label, case)
        summary_path = (source_dir / f"{stem}-summary.json").resolve()
        require_equal(
            Path(str(case["artifact/summary_path"])).resolve(),
            summary_path,
            f"case {label} individual summary path",
        )
        require_equal(read_json(summary_path), case, f"case {label} individual summary")
        trace_path = (source_dir / f"{stem}-trace.jsonl").resolve()
        validate_live_trace(label, trace, read_jsonl(trace_path))
        histories[label] = TemporalHistory.open(label, case, source_dir=source_dir)
        if bool(case.get("inverse/converged", False)):
            msg = f"screen case {label!r} unexpectedly reports convergence"
            raise ValueError(msg)
        stop_reason = str(case.get("inverse/stop_reason"))
        if stop_reason not in {"step_limit_smooth_decrease", "ambiguous_plateau"}:
            msg = f"screen case {label!r} has unexpected stop {stop_reason!r}"
            raise ValueError(msg)
        expected_convergence_failures.append(f"{label}: {stop_reason}")
    require_equal(
        summary.get("convergence_failures"),
        expected_convergence_failures,
        "source convergence failures",
    )
    source_pareto = [
        str(case["candidate"])
        for case in cases
        if bool(case.get("scientific/eligible_for_pareto", False))
    ]
    require_equal(summary.get("pareto_candidates"), source_pareto, "source Pareto list")
    return summary, cases, histories


def encoded_tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    cells = np.asarray(mesh.cells, dtype=np.int64)
    if cells.size != 5 * mesh.n_cells:
        msg = "volume mesh connectivity is not pure tetrahedral"
        raise ValueError(msg)
    encoded = cells.reshape(-1, 5)
    if not np.all(encoded[:, 0] == 4):
        msg = "volume mesh contains a non-tetrahedral cell"
        raise ValueError(msg)
    return encoded[:, 1:].copy()


def triangle_geometry(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    double_area = np.linalg.norm(vectors, axis=1)
    normals = vectors / np.maximum(double_area[:, None], np.finfo(np.float64).tiny)
    return vectors, 0.5 * double_area, normals


def surface_adjacency(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if np.any(counts > 2):
        logger.info(
            "Ignoring %d non-manifold canonical edges outside the validated ROI",
            int((counts > 2).sum()),
        )
    interior = counts == 2
    unique_edges = sorted_edges[starts[interior]]
    tri_0 = sorted_triangles[starts[interior]]
    tri_1 = sorted_triangles[starts[interior] + 1]
    length = np.linalg.norm(
        points[unique_edges[:, 1]] - points[unique_edges[:, 0]], axis=1
    )
    if not np.all(np.isfinite(length)) or np.any(length <= 0.0):
        msg = "canonical skin contains invalid interior edge lengths"
        raise ValueError(msg)
    return unique_edges, tri_0, tri_1, length, edges


def map_global_ids(mesh: pv.UnstructuredGrid, global_ids: np.ndarray) -> np.ndarray:
    mesh_ids = (
        np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
        if GLOBAL_POINT_ID.vtk in mesh.point_data
        else np.arange(mesh.n_points, dtype=np.int64)
    )
    if mesh_ids.shape != (mesh.n_points,) or np.unique(mesh_ids).size != mesh.n_points:
        msg = "input mesh GlobalPointId must be unique"
        raise ValueError(msg)
    order = np.argsort(mesh_ids)
    positions = np.searchsorted(mesh_ids[order], global_ids)
    if np.any(positions >= mesh_ids.size) or not np.array_equal(
        mesh_ids[order[positions]], global_ids
    ):
        msg = "canonical skin GlobalPointId does not map to input mesh"
        raise ValueError(msg)
    return order[positions]


def build_surface_basis(
    base_mesh: pv.UnstructuredGrid, skin: pv.PolyData
) -> SurfaceBasis:
    encoded = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    if encoded.size == 0 or not np.all(encoded[:, 0] == 3):
        msg = "canonical skin is not a non-empty triangle mesh"
        raise ValueError(msg)
    triangles = encoded[:, 1:].copy()
    points = np.asarray(skin.points, dtype=np.float64).copy()
    global_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    mesh_point_ids = map_global_ids(base_mesh, global_ids)
    if not np.array_equal(points, np.asarray(base_mesh.points)[mesh_point_ids]):
        msg = "canonical skin rest points differ from input volume mesh"
        raise ValueError(msg)
    target = np.nan_to_num(
        np.asarray(base_mesh.point_data[SMILE_TARGET], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )[mesh_point_ids]
    target_points = points + target
    _, rest_area, _ = triangle_geometry(points, triangles)
    _, target_area, target_normals = triangle_geometry(target_points, triangles)
    if np.any(rest_area <= np.finfo(np.float64).eps) or np.any(
        target_area <= np.finfo(np.float64).eps
    ):
        msg = "canonical rest/target surface contains degenerate triangles"
        raise ValueError(msg)
    require_array_close(
        rest_area,
        np.asarray(skin.cell_data["RestArea"], dtype=np.float64),
        "canonical RestArea",
    )
    require_array_close(
        target_area,
        np.asarray(skin.cell_data["TargetArea"], dtype=np.float64),
        "canonical TargetArea",
    )
    edges, tri_0, tri_1, edge_length, all_oriented_edges = surface_adjacency(
        points, triangles
    )
    incident = np.bincount(triangles.ravel(), minlength=skin.n_points)
    target_vectors, _, _ = triangle_geometry(target_points, triangles)
    vertex_vector = np.zeros((skin.n_points, 3), dtype=np.float64)
    for local in range(3):
        np.add.at(vertex_vector, triangles[:, local], target_vectors)
    target_vertex_normals = vertex_vector / np.maximum(
        np.linalg.norm(vertex_vector, axis=1)[:, None], np.finfo(np.float64).tiny
    )
    roi_masks = {
        "eligible": np.asarray(skin.cell_data["EligibleMaterialTriangle"], dtype=bool),
        "contraction": np.asarray(
            skin.cell_data["ContractionPrestrainMask"], dtype=bool
        ),
        "expansion": np.asarray(skin.cell_data["ExpansionMaterialMask"], dtype=bool),
    }
    for name, mask in roi_masks.items():
        if mask.shape != (skin.n_cells,) or not np.any(mask):
            msg = f"canonical {name} ROI is empty or malformed"
            raise ValueError(msg)
    eligible_edges = np.vstack(
        (
            triangles[roi_masks["eligible"]][:, [0, 1]],
            triangles[roi_masks["eligible"]][:, [1, 2]],
            triangles[roi_masks["eligible"]][:, [2, 0]],
        )
    )
    eligible_edges.sort(axis=1)
    _, eligible_edge_counts = np.unique(eligible_edges, axis=0, return_counts=True)
    if np.any(eligible_edge_counts > 2):
        msg = "eligible material ROI contains a non-manifold edge"
        raise ValueError(msg)
    if not np.array_equal(
        roi_masks["contraction"] | roi_masks["expansion"], roi_masks["eligible"]
    ) or np.any(roi_masks["contraction"] & roi_masks["expansion"]):
        msg = "contraction/expansion ROIs do not partition eligible material surface"
        raise ValueError(msg)
    del all_oriented_edges
    return SurfaceBasis(
        points=points,
        triangles=triangles,
        mesh_point_ids=mesh_point_ids,
        rest_area=rest_area,
        target_area=target_area,
        target_points=target_points,
        target_displacement=target,
        target_normals=target_normals,
        target_vertex_normals=target_vertex_normals,
        all_edges=edges,
        edge_tri_0=tri_0,
        edge_tri_1=tri_1,
        edge_length=edge_length,
        incident_triangle_count=incident,
        roi_masks=roi_masks,
    )


def require_array_close(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=JSON_RTOL, atol=JSON_ATOL
    ):
        msg = f"{context} array mismatch: {actual.shape} vs {expected.shape}"
        raise ValueError(msg)


def field_scalar(mesh: pv.DataSet, name: str) -> float:
    if name not in mesh.field_data:
        msg = f"history frame has no field_data[{name!r}]"
        raise KeyError(msg)
    values = np.asarray(mesh.field_data[name]).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        msg = f"history field {name!r} is not one finite scalar"
        raise ValueError(msg)
    return float(values[0])


def six_volume(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    edge_1 = points[tets[:, 1]] - points[tets[:, 0]]
    edge_2 = points[tets[:, 2]] - points[tets[:, 0]]
    edge_3 = points[tets[:, 3]] - points[tets[:, 0]]
    return np.einsum("ij,ij->i", edge_1, np.cross(edge_2, edge_3))


def det_f_metrics(
    deformed: np.ndarray, tets: np.ndarray, rest_six_volume: np.ndarray
) -> dict[str, Any]:
    det_f = six_volume(deformed, tets) / rest_six_volume
    if not np.isfinite(det_f).all():
        msg = "frame deformation determinant contains non-finite values"
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


def skin_quality_metrics(
    basis: SurfaceBasis, displacement: np.ndarray
) -> dict[str, Any]:
    deformed = basis.points + displacement[basis.mesh_point_ids]
    deformed_vectors, deformed_area, deformed_normals = triangle_geometry(
        deformed, basis.triangles
    )
    rest_vectors, rest_area, rest_normals = triangle_geometry(
        basis.points, basis.triangles
    )
    area_ratio = deformed_area / rest_area
    signed_normal_ratio = np.einsum(
        "ij,ij->i", deformed_vectors, rest_vectors
    ) / np.maximum(
        np.einsum("ij,ij->i", rest_vectors, rest_vectors),
        np.finfo(np.float64).tiny,
    )
    normal_cosine = np.einsum("ij,ij->i", deformed_normals, rest_normals)
    if not (
        np.isfinite(area_ratio).all()
        and np.isfinite(signed_normal_ratio).all()
        and np.isfinite(normal_cosine).all()
    ):
        msg = "frame skin quality contains non-finite values"
        raise ValueError(msg)
    folded = signed_normal_ratio <= 0.0
    return {
        "quality/skin_triangles": int(basis.triangles.shape[0]),
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


def activation_spd_metrics(
    activation: np.ndarray, active: np.ndarray
) -> dict[str, Any]:
    if activation.shape != (active.size, 6) or not np.isfinite(activation).all():
        msg = "frame RecoveredActivationInv is malformed or non-finite"
        raise ValueError(msg)
    values = activation[active]
    if values.size == 0:
        msg = "frame contains no active muscle tetrahedra"
        raise ValueError(msg)
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
    condition = maximum[positive] / minimum[positive]
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
        "quality/muscle_activation_condition_q999": float(np.quantile(condition, 0.999))
        if condition.size
        else None,
        "quality/muscle_activation_condition_max": float(condition.max())
        if condition.size
        else None,
    }


def physical_gates(metrics: dict[str, Any], cfg: Config) -> dict[str, bool]:
    return {
        "physical/gate_detF_no_inversions": int(metrics["quality/inverted_tets"]) == 0,
        "physical/gate_detF_min_positive": float(metrics["quality/detF_min"]) > 0.0,
        "physical/gate_detF_q001": float(metrics["quality/detF_q001"])
        >= cfg.min_det_f_q001,
        "physical/gate_skin_no_folds": int(metrics["quality/skin_folded_triangles"])
        == 0,
        "physical/gate_skin_area_q001": float(metrics["quality/skin_area_ratio_q001"])
        >= cfg.min_skin_area_ratio_q001,
        "physical/gate_skin_area_q999": float(metrics["quality/skin_area_ratio_q999"])
        <= cfg.max_skin_area_ratio_q999,
        "physical/gate_muscle_activation_spd": int(
            metrics["quality/muscle_activation_non_spd_tets"]
        )
        == 0
        and float(metrics["quality/muscle_activation_min_eigenvalue"])
        >= cfg.min_muscle_activation_eigenvalue,
    }


def validate_frame_arrays(  # noqa: C901
    frame: pv.UnstructuredGrid,
    *,
    base_mesh: pv.UnstructuredGrid,
    base_cells: np.ndarray,
    base_celltypes: np.ndarray,
    base_global_ids: np.ndarray,
    target: np.ndarray,
    loss_mask: np.ndarray,
    active: np.ndarray,
    step: int,
    label: str,
    trace: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if frame.n_points != base_mesh.n_points or frame.n_cells != base_mesh.n_cells:
        msg = f"case {label!r} step {step} topology dimensions changed"
        raise ValueError(msg)
    if not np.array_equal(np.asarray(frame.points), np.asarray(base_mesh.points)):
        msg = f"case {label!r} step {step} rest points changed"
        raise ValueError(msg)
    if not np.array_equal(np.asarray(frame.cells), base_cells) or not np.array_equal(
        np.asarray(frame.celltypes), base_celltypes
    ):
        msg = f"case {label!r} step {step} volume connectivity changed"
        raise ValueError(msg)
    if not np.array_equal(
        np.asarray(frame.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64),
        base_global_ids,
    ):
        msg = f"case {label!r} step {step} GlobalPointId changed"
        raise ValueError(msg)
    require_equal(field_scalar(frame, "inverse_step"), float(step), "history step")
    for field, key in (
        ("inverse_error_rms", "target/error_rms"),
        ("inverse_error_rms_mm", "target/error_rms_mm"),
        ("inverse_loss", "loss/total"),
        ("inverse_loss_mm2", "loss/mm2"),
        ("inverse_loss_m2", "loss/m2"),
    ):
        require_close(
            field_scalar(frame, field),
            float(trace[key]),
            f"case {label} step {step} {field}",
        )
    require_close(
        field_scalar(frame, "inverse_loss_scale"),
        EXPECTED_LOSS_SCALE,
        f"case {label} step {step} loss scale field",
    )
    displacement = np.asarray(frame.point_data["Displacement"], dtype=np.float64)
    frame_target = np.asarray(frame.point_data["TargetDisplacement"], dtype=np.float64)
    frame_mask = np.asarray(frame.point_data["LossMask"], dtype=bool)
    if displacement.shape != target.shape or not np.isfinite(displacement).all():
        msg = f"case {label!r} step {step} Displacement is malformed/non-finite"
        raise ValueError(msg)
    if not np.array_equal(frame_target, target) or not np.array_equal(
        frame_mask, loss_mask
    ):
        msg = f"case {label!r} step {step} target or loss mask changed"
        raise ValueError(msg)
    residual = displacement - target
    require_array_close(
        np.asarray(frame.point_data["DisplacementError"], dtype=np.float64),
        residual,
        f"case {label} step {step} DisplacementError",
    )
    require_array_close(
        np.asarray(frame.point_data["DeformedPoint"], dtype=np.float64),
        np.asarray(base_mesh.points) + displacement,
        f"case {label} step {step} DeformedPoint",
    )
    require_array_close(
        np.asarray(frame.point_data["TargetPoint"], dtype=np.float64),
        np.asarray(base_mesh.points) + target,
        f"case {label} step {step} TargetPoint",
    )
    activation = np.asarray(frame.cell_data["RecoveredActivationInv"], dtype=np.float64)
    frame_active = np.asarray(frame.cell_data["ActivationMask"], dtype=bool)
    if not np.array_equal(frame_active, active):
        msg = f"case {label!r} step {step} ActivationMask changed"
        raise ValueError(msg)
    if activation.shape != (base_mesh.n_cells, 6) or not np.isfinite(activation).all():
        msg = f"case {label!r} step {step} activation is malformed/non-finite"
        raise ValueError(msg)
    if not np.array_equal(activation[~active], np.zeros((int((~active).sum()), 6))):
        msg = f"case {label!r} step {step} inactive tetrahedra have activation"
        raise ValueError(msg)
    if step == 0 and not np.array_equal(
        activation[active], np.zeros_like(activation[active])
    ):
        msg = f"case {label!r} step0 activation is not exactly zero"
        raise ValueError(msg)
    return displacement, activation


def validate_source_best(
    case: dict[str, Any], frame_metrics: dict[str, Any], label: str
) -> None:
    mappings = (
        "quality/detF_min",
        "quality/detF_q001",
        "quality/detF_median",
        "quality/detF_max",
        "quality/inverted_tet_fraction",
        "quality/skin_folded_triangle_fraction",
        "quality/skin_area_ratio_min",
        "quality/skin_area_ratio_q001",
        "quality/skin_area_ratio_median",
        "quality/skin_area_ratio_q999",
        "quality/skin_area_ratio_max",
        "quality/muscle_activation_min_eigenvalue",
        "quality/muscle_activation_min_eigenvalue_q001",
        "quality/muscle_activation_min_eigenvalue_median",
        "quality/muscle_activation_max_eigenvalue_q999",
        "quality/muscle_activation_max_eigenvalue",
        "quality/muscle_activation_determinant_min",
    )
    integers = (
        "quality/inverted_tets",
        "quality/detF_lt_0p2_tets",
        "quality/detF_lt_0p5_tets",
        "quality/skin_triangles",
        "quality/skin_folded_triangles",
        "quality/muscle_activation_tets",
        "quality/muscle_activation_non_spd_tets",
    )
    for key in mappings:
        require_close(
            float(frame_metrics[key]), float(case[key]), f"{label} best {key}"
        )
    for key in integers:
        require_equal(frame_metrics[key], case[key], f"{label} best {key}")
    gate_map = {
        "physical/gate_detF_no_inversions": "scientific/gate_detF_no_inversions",
        "physical/gate_detF_min_positive": "scientific/gate_detF_min_positive",
        "physical/gate_detF_q001": "scientific/gate_detF_q001",
        "physical/gate_skin_no_folds": "scientific/gate_skin_no_folds",
        "physical/gate_skin_area_q001": "scientific/gate_skin_area_q001",
        "physical/gate_skin_area_q999": "scientific/gate_skin_area_q999",
        "physical/gate_muscle_activation_spd": "scientific/gate_muscle_activation_spd",
    }
    for frame_key, case_key in gate_map.items():
        require_equal(frame_metrics[frame_key], case[case_key], f"{label} best gate")


def scan_histories(  # noqa: PLR0915
    cfg: Config,
    cases: list[dict[str, Any]],
    histories: dict[str, TemporalHistory],
    base_mesh: pv.UnstructuredGrid,
    surface: SurfaceBasis,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    tets = encoded_tetrahedra(base_mesh)
    rest_points = np.asarray(base_mesh.points, dtype=np.float64)
    rest_six = six_volume(rest_points, tets)
    if np.any(np.abs(rest_six) <= np.finfo(np.float64).eps):
        msg = "input mesh contains zero-volume tetrahedra"
        raise ValueError(msg)
    base_cells = np.asarray(base_mesh.cells).copy()
    base_celltypes = np.asarray(base_mesh.celltypes).copy()
    base_global_ids = (
        np.asarray(base_mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
        if GLOBAL_POINT_ID.vtk in base_mesh.point_data
        else np.arange(base_mesh.n_points, dtype=np.int64)
    )
    target = np.nan_to_num(
        np.asarray(base_mesh.point_data[SMILE_TARGET], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    loss_mask = np.asarray(base_mesh.point_data[SMILE_LOSS_MASK], dtype=bool)
    active = np.asarray(base_mesh.cell_data["ActivationMask"], dtype=bool)
    target_norm = float(np.linalg.norm(target[loss_mask]))
    if target_norm <= 0.0 or not np.any(active):
        msg = "input target or active muscle set is empty"
        raise ValueError(msg)
    traces: dict[str, list[dict[str, Any]]] = {}
    history_identities: dict[str, dict[str, Any]] = {}
    source_dir = cfg.input_summary.resolve().parent
    for case_index, case in enumerate(cases):
        label = str(case["candidate"])
        history = histories[label]
        logger.info("Hashing %s (%d/%d)", history.path.name, case_index + 1, len(cases))
        history_identities[label] = file_identity(history.path)
        rows: list[dict[str, Any]] = []
        prefix_valid = True
        trace = list(case["trace"])
        for step, source_row in enumerate(trace):
            frame = history.frame(step)
            displacement, activation = validate_frame_arrays(
                frame,
                base_mesh=base_mesh,
                base_cells=base_cells,
                base_celltypes=base_celltypes,
                base_global_ids=base_global_ids,
                target=target,
                loss_mask=loss_mask,
                active=active,
                step=step,
                label=label,
                trace=source_row,
            )
            residual = displacement[loss_mask] - target[loss_mask]
            error_rms = float(np.linalg.norm(residual) / math.sqrt(loss_mask.sum()))
            target_rms = target_norm / math.sqrt(loss_mask.sum())
            fidelity = error_rms / target_rms
            require_close(
                error_rms,
                float(source_row["target/error_rms"]),
                f"{label} step {step} reconstructed error RMS",
            )
            require_close(
                fidelity,
                float(source_row["target/error_rms"])
                / float(case["target/displacement_rms"]),
                f"{label} step {step} reconstructed fidelity",
            )
            require_close(
                float(source_row["loss/m2"]),
                error_rms**2 / 3.0,
                f"{label} step {step} loss/error identity",
            )
            active_values = activation[active]
            activation_rms = float(
                np.linalg.norm(active_values) / math.sqrt(active_values.size)
            )
            require_close(
                activation_rms,
                float(source_row["activation_inv/rms"]),
                f"{label} step {step} activation RMS",
            )
            quality = {
                **det_f_metrics(rest_points + displacement, tets, rest_six),
                **skin_quality_metrics(surface, displacement),
                **activation_spd_metrics(activation, active),
            }
            gates = physical_gates(quality, cfg)
            frame_valid = all(gates.values())
            prefix_valid &= frame_valid
            row = {
                "step": step,
                "fidelity/error_rms_fraction_of_target": fidelity,
                "fidelity/error_rms": error_rms,
                "fidelity/target_rms": target_rms,
                "activation_inv/rms": activation_rms,
                "activation_inv/max_abs": float(np.abs(active_values).max()),
                **quality,
                **gates,
                "physical/frame_valid": frame_valid,
                "physical/prefix_valid": prefix_valid,
            }
            rows.append(row)
            if step % 5 == 0 or step == len(trace) - 1:
                logger.info(
                    "%s physical scan %d/%d: fidelity %.6g, inv=%d, folds=%d, prefix=%s",
                    label,
                    step,
                    len(trace) - 1,
                    fidelity,
                    quality["quality/inverted_tets"],
                    quality["quality/skin_folded_triangles"],
                    prefix_valid,
                )
        traces[label] = rows
        best_step = int(case["best/step"])
        validate_source_best(case, rows[best_step], label)
        best_frame = history.frame(best_step, deep_copy=True)
        result_path = source_dir / f"{case['case']}.vtu"
        result = pv.read(result_path)
        if not isinstance(result, pv.UnstructuredGrid):
            result = result.cast_to_unstructured_grid()
        require_array_close(
            np.asarray(result.point_data["Displacement"], dtype=np.float64),
            np.asarray(best_frame.point_data["Displacement"], dtype=np.float64),
            f"case {label} result/best history displacement",
        )
        require_array_close(
            np.asarray(result.cell_data["RecoveredActivationInv"], dtype=np.float64),
            np.asarray(
                best_frame.cell_data["RecoveredActivationInv"], dtype=np.float64
            ),
            f"case {label} result/best history activation",
        )
        history_identities[label]["result"] = file_identity(result_path)
        target_path = source_dir / f"{case['case']}-target.vtu"
        target_mesh = pv.read(target_path)
        if not isinstance(target_mesh, pv.UnstructuredGrid):
            target_mesh = target_mesh.cast_to_unstructured_grid()
        require_array_close(
            np.asarray(target_mesh.point_data["TargetDisplacement"], dtype=np.float64),
            target,
            f"case {label} target artifact displacement",
        )
        require_true(
            bool(
                np.array_equal(
                    np.asarray(target_mesh.point_data["LossMask"], dtype=bool),
                    loss_mask,
                )
            ),
            f"case {label} target artifact mask",
        )
        history_identities[label]["target"] = file_identity(target_path)
    return traces, history_identities


def build_matching(
    traces: dict[str, list[dict[str, Any]]],
    labels: tuple[str, ...],
    *,
    require_physical_prefix: bool,
    max_spread: float,
) -> dict[str, Any]:
    admissible: dict[str, list[dict[str, Any]]] = {}
    failures: list[str] = []
    for label in labels:
        rows = traces[label]
        selected = (
            [row for row in rows if bool(row["physical/prefix_valid"])]
            if require_physical_prefix
            else list(rows)
        )
        if not selected:
            failures.append(f"{label}: no admissible frame")
        admissible[label] = selected
    if failures:
        return {
            "available": False,
            "require_physical_prefix": require_physical_prefix,
            "failures": failures,
            "selections": {},
            "true_matched_fidelity": False,
        }
    case_best = {
        label: min(
            admissible[label],
            key=lambda row: (
                float(row["fidelity/error_rms_fraction_of_target"]),
                int(row["step"]),
            ),
        )
        for label in labels
    }
    tau = max(
        float(row["fidelity/error_rms_fraction_of_target"])
        for row in case_best.values()
    )
    tolerance = max(1.0e-12, abs(tau) * 1.0e-10)
    selections: dict[str, dict[str, Any]] = {}
    for label in labels:
        best_step = int(case_best[label]["step"])
        eligible = [
            row
            for row in admissible[label]
            if int(row["step"]) <= best_step
            and float(row["fidelity/error_rms_fraction_of_target"]) <= tau + tolerance
        ]
        if not eligible:
            failures.append(f"{label}: cannot reach tau={tau:.17g}")
            continue
        closest = max(
            eligible,
            key=lambda row: (
                float(row["fidelity/error_rms_fraction_of_target"]),
                -int(row["step"]),
            ),
        )
        first = min(eligible, key=lambda row: int(row["step"]))
        preceding_rows = [
            row
            for row in admissible[label]
            if int(row["step"]) <= best_step and int(row["step"]) < int(first["step"])
        ]
        preceding = (
            max(preceding_rows, key=lambda row: int(row["step"]))
            if preceding_rows
            else None
        )
        first_fidelity = float(first["fidelity/error_rms_fraction_of_target"])
        preceding_fidelity = (
            float(preceding["fidelity/error_rms_fraction_of_target"])
            if preceding is not None
            else None
        )
        selections[label] = {
            "selection/tau": tau,
            "selection/case_best_step": best_step,
            "selection/case_best_fidelity": case_best[label][
                "fidelity/error_rms_fraction_of_target"
            ],
            "selection/step": int(closest["step"]),
            "selection/fidelity": closest["fidelity/error_rms_fraction_of_target"],
            "selection/first_crossing_step": int(first["step"]),
            "selection/first_crossing_fidelity": first[
                "fidelity/error_rms_fraction_of_target"
            ],
            "selection/first_crossing_preceding_step": int(preceding["step"])
            if preceding is not None
            else None,
            "selection/first_crossing_preceding_fidelity": preceding_fidelity,
            "selection/first_crossing_brackets_tau": (
                preceding_fidelity is not None
                and preceding_fidelity > tau + tolerance
                and first_fidelity <= tau + tolerance
            ),
            "selection/first_crossing_bracket_width": (
                preceding_fidelity - first_fidelity
                if preceding_fidelity is not None
                else None
            ),
            "selection/first_crossing_distance_above_tau": (
                preceding_fidelity - tau if preceding_fidelity is not None else None
            ),
            "selection/first_crossing_distance_below_tau": tau - first_fidelity,
            "selection/step_delta_from_first_crossing": int(closest["step"])
            - int(first["step"]),
        }
    if failures:
        return {
            "available": False,
            "require_physical_prefix": require_physical_prefix,
            "failures": failures,
            "tau": tau,
            "tolerance": tolerance,
            "selections": selections,
            "true_matched_fidelity": False,
        }
    selected_fidelity = np.asarray(
        [float(selections[label]["selection/fidelity"]) for label in labels]
    )
    first_crossing_fidelity = np.asarray(
        [
            float(selections[label]["selection/first_crossing_fidelity"])
            for label in labels
        ]
    )
    spread = float(np.ptp(selected_fidelity))
    spread_gate = spread <= max_spread + tolerance
    first_crossing_spread = float(np.ptp(first_crossing_fidelity))
    first_crossing_spread_gate = first_crossing_spread <= max_spread + tolerance
    return {
        "available": True,
        "require_physical_prefix": require_physical_prefix,
        "labels": list(labels),
        "tau": tau,
        "tolerance": tolerance,
        "tau_anchor_labels": [
            label
            for label in labels
            if math.isclose(
                float(case_best[label]["fidelity/error_rms_fraction_of_target"]),
                tau,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        ],
        "maximum_allowed_spread": max_spread,
        "selected_fidelity_min": float(selected_fidelity.min()),
        "selected_fidelity_max": float(selected_fidelity.max()),
        "selected_fidelity_spread": spread,
        "spread_gate_passed": spread_gate,
        "true_matched_fidelity": require_physical_prefix and spread_gate,
        "first_crossing_fidelity_min": float(first_crossing_fidelity.min()),
        "first_crossing_fidelity_max": float(first_crossing_fidelity.max()),
        "first_crossing_fidelity_spread": first_crossing_spread,
        "first_crossing_spread_gate_passed": first_crossing_spread_gate,
        "first_crossing_true_matched_fidelity": (
            require_physical_prefix and first_crossing_spread_gate
        ),
        "failures": [],
        "selections": selections,
    }


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.size != values.size or not np.any(weights > 0.0):
        return math.nan
    return float(np.sqrt(np.sum(weights * np.square(values)) / np.sum(weights)))


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    if values.size == 0 or weights.size != values.size or not np.any(weights > 0.0):
        return math.nan
    order = np.argsort(values)
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    threshold = quantile * cumulative[-1]
    index = min(
        int(np.searchsorted(cumulative, threshold, side="left")), values.size - 1
    )
    return float(ordered_values[index])


def unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def roi_metrics(
    basis: SurfaceBasis,
    displacement: np.ndarray,
    roi_name: str,
) -> dict[str, Any]:
    mask = basis.roi_masks[roi_name]
    triangles = basis.triangles
    deformed = basis.points + displacement[basis.mesh_point_ids]
    residual = displacement[basis.mesh_point_ids] - basis.target_displacement
    _, deformed_area, deformed_normals = triangle_geometry(deformed, triangles)
    normal_dot = np.clip(
        np.einsum("ij,ij->i", deformed_normals, basis.target_normals), -1.0, 1.0
    )
    normal_angle = np.arccos(normal_dot)
    log_area_error = np.log(
        np.maximum(deformed_area, np.finfo(np.float64).tiny) / basis.target_area
    )
    rest_weights = basis.rest_area[mask]
    interior = mask[basis.edge_tri_0] & mask[basis.edge_tri_1]
    n0 = deformed_normals[basis.edge_tri_0[interior]]
    n1 = deformed_normals[basis.edge_tri_1[interior]]
    target_n0 = basis.target_normals[basis.edge_tri_0[interior]]
    target_n1 = basis.target_normals[basis.edge_tri_1[interior]]
    deformed_dihedral = np.arccos(np.clip(np.einsum("ij,ij->i", n0, n1), -1.0, 1.0))
    target_dihedral = np.arccos(
        np.clip(np.einsum("ij,ij->i", target_n0, target_n1), -1.0, 1.0)
    )
    dihedral_error = deformed_dihedral - target_dihedral
    edge_weights = basis.edge_length[interior]

    roi_triangles = triangles[mask]
    triangle_residual = residual[roi_triangles].mean(axis=1)
    triangle_target = basis.target_displacement[roi_triangles].mean(axis=1)
    residual_rms = math.sqrt(
        float(np.sum(rest_weights * np.sum(np.square(triangle_residual), axis=1)))
        / float(np.sum(rest_weights))
    )
    target_rms = math.sqrt(
        float(np.sum(rest_weights * np.sum(np.square(triangle_target), axis=1)))
        / float(np.sum(rest_weights))
    )

    roi_incident = np.bincount(roi_triangles.ravel(), minlength=basis.points.shape[0])
    complete = (roi_incident == basis.incident_triangle_count) & (roi_incident > 0)
    roi_edges = unique_edges(roi_triangles)
    roi_edges = roi_edges[complete[roi_edges[:, 0]] & complete[roi_edges[:, 1]]]
    normal_residual = np.einsum("ij,ij->i", residual, basis.target_vertex_normals)
    neighbor_sum = np.zeros(basis.points.shape[0], dtype=np.float64)
    weight_sum = np.zeros(basis.points.shape[0], dtype=np.float64)
    if roi_edges.size:
        edge_length = np.linalg.norm(
            basis.points[roi_edges[:, 1]] - basis.points[roi_edges[:, 0]], axis=1
        )
        weight = 1.0 / edge_length
        np.add.at(
            neighbor_sum, roi_edges[:, 0], weight * normal_residual[roi_edges[:, 1]]
        )
        np.add.at(
            neighbor_sum, roi_edges[:, 1], weight * normal_residual[roi_edges[:, 0]]
        )
        np.add.at(weight_sum, roi_edges[:, 0], weight)
        np.add.at(weight_sum, roi_edges[:, 1], weight)
    active_vertices = complete & (weight_sum > 0.0)
    laplacian = normal_residual[active_vertices] - (
        neighbor_sum[active_vertices] / weight_sum[active_vertices]
    )
    vertex_mass = np.zeros(basis.points.shape[0], dtype=np.float64)
    for local in range(3):
        np.add.at(
            vertex_mass,
            roi_triangles[:, local],
            basis.rest_area[mask] / 3.0,
        )
    laplacian_rms = weighted_rms(laplacian, vertex_mass[active_vertices])
    prefix = f"roi/{roi_name}"
    return {
        f"{prefix}/triangles": int(mask.sum()),
        f"{prefix}/rest_area": float(basis.rest_area[mask].sum()),
        f"{prefix}/interior_edges": int(interior.sum()),
        f"{prefix}/complete_vertices": int(active_vertices.sum()),
        f"{prefix}/error_rms": residual_rms,
        f"{prefix}/target_rms": target_rms,
        f"{prefix}/error_rms_fraction_of_target": residual_rms / target_rms,
        f"{prefix}/target_relative_dihedral_rms_rad": weighted_rms(
            dihedral_error, edge_weights
        ),
        f"{prefix}/target_relative_dihedral_abs_q95_rad": weighted_quantile(
            np.abs(dihedral_error), edge_weights, 0.95
        ),
        f"{prefix}/target_relative_normal_angle_rms_rad": weighted_rms(
            normal_angle[mask], rest_weights
        ),
        f"{prefix}/target_relative_normal_angle_q95_rad": weighted_quantile(
            normal_angle[mask], rest_weights, 0.95
        ),
        f"{prefix}/log_area_error_rms": weighted_rms(
            log_area_error[mask], rest_weights
        ),
        f"{prefix}/log_area_error_abs_q95": weighted_quantile(
            np.abs(log_area_error[mask]), rest_weights, 0.95
        ),
        f"{prefix}/residual_normal_umbrella_laplacian_rms": laplacian_rms,
        f"{prefix}/residual_normal_umbrella_laplacian_over_target_rms": (
            laplacian_rms / target_rms
        ),
    }


def state_metrics(
    frame: pv.UnstructuredGrid,
    *,
    base_mesh: pv.UnstructuredGrid,
    surface: SurfaceBasis,
    legacy_edges: np.ndarray,
    loss_mask: np.ndarray,
) -> dict[str, Any]:
    displacement = np.asarray(frame.point_data["Displacement"], dtype=np.float64)
    target = np.asarray(frame.point_data["TargetDisplacement"], dtype=np.float64)
    legacy = bumpiness_metrics(
        mask=loss_mask,
        edges=legacy_edges,
        displacement=displacement,
        target=target,
    )
    for key, value in legacy.items():
        if not math.isfinite(float(value)):
            msg = f"legacy bumpiness metric {key} is non-finite"
            raise ValueError(msg)
    return {
        **legacy,
        **roi_metrics(surface, displacement, "eligible"),
        **roi_metrics(surface, displacement, "contraction"),
        **roi_metrics(surface, displacement, "expansion"),
        "mesh/n_points": base_mesh.n_points,
        "mesh/n_tets": base_mesh.n_cells,
    }


def collect_selected_metrics(
    matching: dict[str, Any],
    *,
    cases: list[dict[str, Any]],
    histories: dict[str, TemporalHistory],
    frame_traces: dict[str, list[dict[str, Any]]],
    base_mesh: pv.UnstructuredGrid,
    surface: SurfaceBasis,
    include_first_crossing: bool = True,
) -> list[dict[str, Any]]:
    loss_mask = np.asarray(base_mesh.point_data[SMILE_LOSS_MASK], dtype=bool)
    legacy_edges = surface_edges_for_mask(base_mesh, loss_mask)
    by_label = {str(case["candidate"]): case for case in cases}
    rows: list[dict[str, Any]] = []
    for label in EXPECTED_LABELS:
        selection = matching["selections"].get(label)
        if selection is None:
            source_eligible = bool(
                by_label[label].get("scientific/eligible_for_pareto", False)
            )
            rows.append(
                {
                    "candidate": label,
                    "selection/available": False,
                    "source/final_scientific_eligible": source_eligible,
                    "source/final_scientific_ineligible_reasons": list(
                        by_label[label].get("scientific/ineligible_reasons", [])
                    ),
                    "analysis/eligible_for_matched_checkpoint_pareto": False,
                    "analysis/matched_checkpoint_ineligible_reasons": [
                        "no matched selection"
                    ],
                    "promotion/eligible_for_direct_stage_b_long": False,
                    "promotion/direct_stage_b_long_ineligible_reasons": [
                        "no matched selection"
                    ],
                }
            )
            continue
        selected_step = int(selection["selection/step"])
        first_step = int(selection["selection/first_crossing_step"])
        selected_frame = histories[label].frame(selected_step, deep_copy=True)
        selected_metrics = state_metrics(
            selected_frame,
            base_mesh=base_mesh,
            surface=surface,
            legacy_edges=legacy_edges,
            loss_mask=loss_mask,
        )
        frame = frame_traces[label][selected_step]
        first_crossing_fields: dict[str, Any] = {}
        if include_first_crossing:
            first_frame = (
                selected_frame
                if first_step == selected_step
                else histories[label].frame(first_step, deep_copy=True)
            )
            first_metrics = state_metrics(
                first_frame,
                base_mesh=base_mesh,
                surface=surface,
                legacy_edges=legacy_edges,
                loss_mask=loss_mask,
            )
            first_frame_trace = frame_traces[label][first_step]
            first_crossing_fields = {
                **{
                    f"first_crossing/{key}": value
                    for key, value in first_metrics.items()
                },
                "first_crossing/activation_inv/rms": first_frame_trace[
                    "activation_inv/rms"
                ],
                "first_crossing/physical/frame_valid": first_frame_trace[
                    "physical/frame_valid"
                ],
                "first_crossing/physical/prefix_valid": first_frame_trace[
                    "physical/prefix_valid"
                ],
            }
        source_eligible = bool(
            by_label[label].get("scientific/eligible_for_pareto", False)
        )
        source_ineligible_reasons = list(
            by_label[label].get("scientific/ineligible_reasons", [])
        )
        matched_reasons: list[str] = []
        if label == NO_SKIN_LABEL:
            matched_reasons.append("no-skin is a diagnostic control")
        if not bool(frame["physical/frame_valid"]):
            matched_reasons.append("selected frame fails a physical gate")
        if not bool(frame["physical/prefix_valid"]):
            matched_reasons.append(
                "trajectory is not physically valid through selection"
            )
        if not bool(matching.get("true_matched_fidelity", False)):
            matched_reasons.append("true matched-fidelity gate did not pass")
        promotion_reasons = list(matched_reasons)
        if not source_eligible:
            promotion_reasons.append(
                "source final state is not scientifically eligible"
            )
        row = {
            "candidate": label,
            "candidate/young_min_scale": by_label[label].get(
                "candidate/young_min_scale"
            ),
            "candidate/prestrain_gain": by_label[label].get("candidate/prestrain_gain"),
            "selection/available": True,
            **selection,
            **frame,
            **selected_metrics,
            **first_crossing_fields,
            "source/best_step": by_label[label]["best/step"],
            "source/best_fidelity": by_label[label][
                "best/error_rms_fraction_of_target"
            ],
            "source/final_scientific_eligible": source_eligible,
            "source/final_scientific_ineligible_reasons": source_ineligible_reasons,
            "analysis/eligible_for_matched_checkpoint_pareto": not matched_reasons,
            "analysis/matched_checkpoint_ineligible_reasons": matched_reasons,
            "promotion/eligible_for_direct_stage_b_long": not promotion_reasons,
            "promotion/direct_stage_b_long_ineligible_reasons": promotion_reasons,
        }
        rows.append(row)
    return rows


def dominates(
    a: dict[str, Any],
    b: dict[str, Any],
    cfg: Config,
    objectives: tuple[str, str],
) -> bool:
    no_worse = True
    strictly_better = False
    for key in objectives:
        av = float(a[key])
        bv = float(b[key])
        tolerance = cfg.pareto_atol + cfg.pareto_rtol * abs(bv)
        no_worse &= av <= bv + tolerance
        strictly_better |= av < bv - tolerance
    return no_worse and strictly_better


def pareto_analysis(rows: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    objectives = (
        "roi/contraction/target_relative_dihedral_rms_rad",
        "activation_inv/rms",
    )
    eligible = [
        row
        for row in rows
        if row["candidate"] in MATERIAL_LABELS
        and bool(row.get("analysis/eligible_for_matched_checkpoint_pareto", False))
    ]
    front = [
        row
        for row in eligible
        if not any(
            dominates(other, row, cfg, objectives)
            for other in eligible
            if other is not row
        )
    ]
    return {
        "objectives": [
            "minimize roi/contraction/target_relative_dihedral_rms_rad",
            "minimize activation_inv/rms",
        ],
        "rtol": cfg.pareto_rtol,
        "atol": cfg.pareto_atol,
        "eligible_candidates": [row["candidate"] for row in eligible],
        "front": [row["candidate"] for row in front],
        "empty_reason": None
        if eligible
        else "no material candidate passed selected-frame physical-prefix and matched-fidelity gates",
    }


def first_crossing_pareto_analysis(
    rows: list[dict[str, Any]], matching: dict[str, Any], cfg: Config
) -> dict[str, Any]:
    objectives = (
        "first_crossing/roi/contraction/target_relative_dihedral_rms_rad",
        "first_crossing/activation_inv/rms",
    )
    compared = [
        row
        for row in rows
        if row["candidate"] in MATERIAL_LABELS
        and bool(row.get("selection/available", False))
        and all(row.get(key) is not None for key in objectives)
    ]
    front = [
        row
        for row in compared
        if not any(
            dominates(other, row, cfg, objectives)
            for other in compared
            if other is not row
        )
    ]
    scientific = bool(matching.get("first_crossing_true_matched_fidelity", False))
    diagnostic_reason = None
    if not scientific:
        diagnostic_reason = (
            "first-crossing selections do not have a continuous physical prefix"
            if not bool(matching.get("require_physical_prefix", False))
            else "first-crossing fidelity spread does not pass the independent matched-fidelity gate"
        )
    return {
        "objectives": [f"minimize {key}" for key in objectives],
        "rtol": cfg.pareto_rtol,
        "atol": cfg.pareto_atol,
        "compared_candidates": [row["candidate"] for row in compared],
        "front": [row["candidate"] for row in front],
        "scientific_true_matched_fidelity": scientific,
        "interpretation": "scientific" if scientific else "diagnostic_only",
        "diagnostic_reason": diagnostic_reason,
    }


def effect_sizes(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    lookup = {str(row["candidate"]): row for row in rows}

    def difference(a: str, b: str) -> float | None:
        av = lookup[a].get(key)
        bv = lookup[b].get(key)
        return None if av is None or bv is None else float(av) - float(bv)

    e_effect = {
        p: difference(f"e025-{p}", f"e100-{p}") for p in ("p000", "p050", "p100")
    }
    prestrain = {
        e: {p: difference(f"{e}-{p}", f"{e}-p000") for p in ("p050", "p100")}
        for e in ("e100", "e025")
    }
    interactions = {
        p: None
        if e_effect[p] is None or e_effect["p000"] is None
        else e_effect[p] - e_effect["p000"]
        for p in ("p050", "p100")
    }
    return {
        "metric": key,
        "young_softening_effect_e025_minus_e100": e_effect,
        "prestrain_effect_relative_to_p000": prestrain,
        "interaction_difference_in_differences": interactions,
    }


def numeric_effect_leaves(value: Any, *, prefix: str = "") -> dict[str, float | None]:
    if isinstance(value, dict):
        leaves: dict[str, float | None] = {}
        for key, item in value.items():
            if key == "metric":
                continue
            path = f"{prefix}/{key}" if prefix else str(key)
            leaves.update(numeric_effect_leaves(item, prefix=path))
        return leaves
    if value is None:
        return {prefix: None}
    if isinstance(value, int | float) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def effect_sign_stability(
    reference: dict[str, dict[str, Any]],
    sensitivity: dict[str, dict[str, Any]],
    *,
    scientific: bool,
    zero_tolerance: float,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for concept, reference_effect in reference.items():
        reference_leaves = numeric_effect_leaves(reference_effect)
        sensitivity_leaves = numeric_effect_leaves(sensitivity[concept])
        require_equal(
            set(sensitivity_leaves),
            set(reference_leaves),
            f"effect sensitivity keys for {concept}",
        )
        for path, reference_value in reference_leaves.items():
            sensitivity_value = sensitivity_leaves[path]

            def sign(value: float | None) -> int | None:
                if value is None:
                    return None
                if abs(value) <= zero_tolerance:
                    return 0
                return 1 if value > 0.0 else -1

            reference_sign = sign(reference_value)
            sensitivity_sign = sign(sensitivity_value)
            comparisons[f"{concept}/{path}"] = {
                "reference": reference_value,
                "sensitivity": sensitivity_value,
                "reference_sign": reference_sign,
                "sensitivity_sign": sensitivity_sign,
                "sign_stable": (
                    None
                    if reference_sign is None or sensitivity_sign is None
                    else reference_sign == sensitivity_sign
                ),
            }
    comparable = [
        bool(row["sign_stable"])
        for row in comparisons.values()
        if row["sign_stable"] is not None
    ]
    return {
        "scientific": scientific,
        "interpretation": "scientific" if scientific else "diagnostic_only",
        "zero_tolerance": zero_tolerance,
        "n_comparable": len(comparable),
        "n_stable": sum(comparable),
        "all_stable": all(comparable) if comparable else None,
        "comparisons": comparisons,
    }


def front_stability(
    reference: dict[str, Any], sensitivity: dict[str, Any], *, scientific: bool
) -> dict[str, Any]:
    reference_front = set(map(str, reference["front"]))
    sensitivity_front = set(map(str, sensitivity["front"]))
    union = reference_front | sensitivity_front
    return {
        "scientific": scientific,
        "interpretation": "scientific" if scientific else "diagnostic_only",
        "reference_front": sorted(reference_front),
        "sensitivity_front": sorted(sensitivity_front),
        "identical": reference_front == sensitivity_front,
        "added": sorted(sensitivity_front - reference_front),
        "removed": sorted(reference_front - sensitivity_front),
        "jaccard": len(reference_front & sensitivity_front) / len(union)
        if union
        else None,
    }


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = [
        {
            key: value
            for key, value in row.items()
            if isinstance(value, bool | int | float | str) or value is None
        }
        for row in rows
    ]
    keys = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flattened)


def format_value(value: Any, spec: str = ".5g") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def write_table(
    path: Path,
    rows: list[dict[str, Any]],
    matched_checkpoint_pareto: dict[str, Any],
) -> None:
    front = set(matched_checkpoint_pareto["front"])
    lines = [
        "| candidate | source final eligible | prefix valid | selected step | fidelity | physical | contraction dihedral | expansion error/target | activation RMS | inv tets | folds | eig min | matched-checkpoint eligible | direct Stage B | Pareto |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        label = str(row["candidate"])
        lines.append(
            "| {label} | {source} | {prefix} | {step} | {fidelity} | {physical} | "
            "{dihedral} | {expansion} | {activation} | {inverted} | {folds} | "
            "{eigenvalue} | {eligible} | {direct} | {pareto} |".format(
                label=label,
                source=format_value(row.get("source/final_scientific_eligible")),
                prefix=format_value(row.get("physical/prefix_valid")),
                step=format_value(row.get("selection/step")),
                fidelity=format_value(row.get("selection/fidelity")),
                physical=format_value(row.get("physical/frame_valid")),
                dihedral=format_value(
                    row.get("roi/contraction/target_relative_dihedral_rms_rad")
                ),
                expansion=format_value(
                    row.get("roi/expansion/error_rms_fraction_of_target")
                ),
                activation=format_value(row.get("activation_inv/rms")),
                inverted=format_value(row.get("quality/inverted_tets")),
                folds=format_value(row.get("quality/skin_folded_triangles")),
                eigenvalue=format_value(
                    row.get("quality/muscle_activation_min_eigenvalue")
                ),
                eligible=format_value(
                    row.get("analysis/eligible_for_matched_checkpoint_pareto")
                ),
                direct=format_value(
                    row.get("promotion/eligible_for_direct_stage_b_long")
                ),
                pareto="yes" if label in front else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_trajectories(
    path: Path,
    traces: dict[str, list[dict[str, Any]]],
    matching: dict[str, Any],
    cfg: Config,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(EXPECTED_LABELS)))
    selection = matching.get("selections", {})
    for color, label in zip(colors, EXPECTED_LABELS, strict=True):
        rows = traces[label]
        steps = np.asarray([row["step"] for row in rows])
        fidelity = np.asarray(
            [row["fidelity/error_rms_fraction_of_target"] for row in rows]
        )
        axes[0, 0].plot(steps, fidelity, color=color, label=label)
        selected = selection.get(label)
        if selected is not None:
            axes[0, 0].scatter(
                [selected["selection/step"]],
                [selected["selection/fidelity"]],
                color=[color],
                marker="o",
                s=35,
            )
            axes[0, 0].scatter(
                [selected["selection/first_crossing_step"]],
                [selected["selection/first_crossing_fidelity"]],
                color=[color],
                marker="x",
                s=35,
            )
        det_q = np.asarray([row["quality/detF_q001"] for row in rows])
        axes[0, 1].plot(steps, det_q, color=color, label=label)
        inverted = np.asarray([row["quality/inverted_tets"] > 0 for row in rows])
        axes[0, 1].scatter(
            steps[inverted], det_q[inverted], color=[color], marker="x", s=18
        )
        skin_margin = np.minimum(
            np.asarray([row["quality/skin_area_ratio_q001"] for row in rows])
            / cfg.min_skin_area_ratio_q001,
            cfg.max_skin_area_ratio_q999
            / np.asarray([row["quality/skin_area_ratio_q999"] for row in rows]),
        )
        axes[1, 0].plot(steps, skin_margin, color=color, label=label)
        folded = np.asarray([row["quality/skin_folded_triangles"] > 0 for row in rows])
        axes[1, 0].scatter(
            steps[folded], skin_margin[folded], color=[color], marker="x", s=18
        )
        eigenvalue = np.asarray(
            [row["quality/muscle_activation_min_eigenvalue"] for row in rows]
        )
        axes[1, 1].plot(steps, eigenvalue, color=color, label=label)
        non_spd = np.asarray(
            [row["quality/muscle_activation_non_spd_tets"] > 0 for row in rows]
        )
        axes[1, 1].scatter(
            steps[non_spd], eigenvalue[non_spd], color=[color], marker="x", s=18
        )
    if matching.get("tau") is not None:
        axes[0, 0].axhline(float(matching["tau"]), color="black", linestyle=":")
    axes[0, 0].set_title("common-mesh target fidelity")
    axes[0, 0].set_ylabel("error RMS / target RMS")
    axes[0, 1].axhline(cfg.min_det_f_q001, color="black", linestyle=":")
    axes[0, 1].set_title("tet quality (x = at least one inversion)")
    axes[0, 1].set_ylabel("detF q0.001")
    axes[1, 0].axhline(1.0, color="black", linestyle=":")
    axes[1, 0].set_title("skin area margin (x = at least one fold)")
    axes[1, 0].set_ylabel("minimum normalized area margin")
    axes[1, 1].axhline(
        cfg.min_muscle_activation_eigenvalue, color="black", linestyle=":"
    )
    axes[1, 1].set_yscale("symlog", linthresh=1.0e-6)
    axes[1, 1].set_title("muscle I + ActivationInv (x = non-SPD)")
    axes[1, 1].set_ylabel("minimum eigenvalue")
    for axis in axes.ravel():
        axis.set_xlabel("inverse evaluation step")
        axis.grid(alpha=0.3)
    axes[0, 0].legend(fontsize="small", ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def grid_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    lookup = {str(row["candidate"]): row for row in rows}
    return np.asarray(
        [
            [lookup[f"e100-{p}"].get(key, math.nan) for p in ("p000", "p050", "p100")],
            [lookup[f"e025-{p}"].get(key, math.nan) for p in ("p000", "p050", "p100")],
        ],
        dtype=np.float64,
    )


def heatmap(axis: Any, values: np.ndarray, title: str) -> None:
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(3), labels=("p000", "p050", "p100"))
    axis.set_yticks(range(2), labels=("e100", "e025"))
    axis.set_title(title)
    for row in range(2):
        for column in range(3):
            value = values[row, column]
            axis.text(
                column,
                row,
                "-" if not math.isfinite(value) else f"{value:.3g}",
                ha="center",
                va="center",
                color="white" if math.isfinite(value) else "black",
                fontsize="small",
            )
    plt.colorbar(image, ax=axis, shrink=0.82)


def plot_matched_screen(
    path: Path,
    rows: list[dict[str, Any]],
    matching: dict[str, Any],
    matched_checkpoint_pareto: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    available = [row for row in rows if bool(row.get("selection/available", False))]
    labels = [str(row["candidate"]) for row in available]
    x = np.arange(len(available))
    axes[0, 0].scatter(
        x, [float(row["selection/fidelity"]) for row in available], marker="o"
    )
    axes[0, 0].scatter(
        x,
        [float(row["selection/first_crossing_fidelity"]) for row in available],
        marker="x",
    )
    if matching.get("tau") is not None:
        axes[0, 0].axhline(float(matching["tau"]), color="black", linestyle=":")
    axes[0, 0].set_xticks(x, labels=labels, rotation=30, ha="right")
    axes[0, 0].set_ylabel("error RMS / target RMS")
    axes[0, 0].set_title(
        "matched fidelity: "
        + ("PASS" if matching.get("true_matched_fidelity") else "DIAGNOSTIC ONLY")
    )
    heatmap(
        axes[0, 1],
        grid_values(rows, "roi/contraction/target_relative_dihedral_rms_rad"),
        "contraction target-relative dihedral RMS",
    )
    heatmap(
        axes[1, 0],
        grid_values(rows, "roi/expansion/error_rms_fraction_of_target"),
        "expansion ROI error / target",
    )
    front = set(matched_checkpoint_pareto["front"])
    for row in available:
        label = str(row["candidate"])
        roughness = row.get("roi/contraction/target_relative_dihedral_rms_rad")
        activation = row.get("activation_inv/rms")
        if roughness is None or activation is None:
            continue
        eligible = bool(
            row.get("analysis/eligible_for_matched_checkpoint_pareto", False)
        )
        axes[1, 1].scatter(
            [activation],
            [roughness],
            marker="o" if eligible else "x",
            s=70 if label in front else 45,
            label=label,
        )
        axes[1, 1].annotate(
            label, (activation, roughness), xytext=(4, 4), textcoords="offset points"
        )
    axes[1, 1].set_xlabel("activation_inv RMS")
    axes[1, 1].set_ylabel("contraction dihedral RMS [rad]")
    axes[1, 1].set_title("material matched-checkpoint Pareto (x = ineligible)")
    for axis in axes.ravel():
        axis.grid(alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validate_config(cfg: Config) -> None:
    expected = {
        "max_fidelity_spread": 0.001,
        "min_det_f_q001": 0.20,
        "min_skin_area_ratio_q001": 0.10,
        "max_skin_area_ratio_q999": 10.0,
        "min_muscle_activation_eigenvalue": 1.0e-6,
        "pareto_rtol": 1.0e-6,
        "pareto_atol": 1.0e-12,
    }
    for key, value in expected.items():
        require_equal(getattr(cfg, key), value, f"fixed analyzer config {key}")
    outputs = (
        cfg.output_json,
        cfg.output_csv,
        cfg.output_table,
        cfg.output_trajectory_plot,
        cfg.output_screen_plot,
    )
    if len({path.resolve() for path in outputs}) != len(outputs):
        msg = "analyzer output paths must be distinct"
        raise ValueError(msg)
    if str(mpl.get_backend()).lower() != "agg":
        msg = f"analyzer requires Agg backend, got {mpl.get_backend()}"
        raise RuntimeError(msg)


def main(cfg: Config) -> None:  # noqa: C901, PLR0915
    validate_config(cfg)
    for path in (
        cfg.output_json,
        cfg.output_csv,
        cfg.output_table,
        cfg.output_trajectory_plot,
        cfg.output_screen_plot,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    candidates = load_candidate_artifacts(cfg)
    source, cases, histories = load_source_summary(cfg)
    base_mesh = pv.read(cfg.input_mesh)
    if not isinstance(base_mesh, pv.UnstructuredGrid):
        base_mesh = base_mesh.cast_to_unstructured_grid()
    surface = build_surface_basis(base_mesh, candidates.skins[BASELINE_LABEL])
    frame_traces, history_identities = scan_histories(
        cfg, cases, histories, base_mesh, surface
    )

    primary = build_matching(
        frame_traces,
        EXPECTED_LABELS,
        require_physical_prefix=True,
        max_spread=cfg.max_fidelity_spread,
    )
    diagnostic_fallback = None
    matching = primary
    if not primary["available"]:
        diagnostic_fallback = build_matching(
            frame_traces,
            EXPECTED_LABELS,
            require_physical_prefix=False,
            max_spread=cfg.max_fidelity_spread,
        )
        matching = diagnostic_fallback
        matching["true_matched_fidelity"] = False
        matching["diagnostic_reason"] = (
            "at least one case has no physically valid prefix; selections ignore "
            "physical validity and are not scientific matched states"
        )
    material_only = build_matching(
        frame_traces,
        MATERIAL_LABELS,
        require_physical_prefix=True,
        max_spread=cfg.max_fidelity_spread,
    )
    rows = collect_selected_metrics(
        matching,
        cases=cases,
        histories=histories,
        frame_traces=frame_traces,
        base_mesh=base_mesh,
        surface=surface,
    )
    matched_checkpoint_pareto = pareto_analysis(rows, cfg)
    first_crossing_pareto = first_crossing_pareto_analysis(rows, matching, cfg)
    matched_front = set(matched_checkpoint_pareto["front"])
    for row in rows:
        label = str(row["candidate"])
        promotion_reasons = list(
            row.get("promotion/direct_stage_b_long_ineligible_reasons", [])
        )
        if label in MATERIAL_LABELS and label not in matched_front:
            promotion_reasons.append(
                "candidate is not on the matched-checkpoint Pareto front"
            )
        row["promotion/direct_stage_b_long_ineligible_reasons"] = promotion_reasons
        row["promotion/eligible_for_direct_stage_b_long"] = (
            label in matched_front
            and bool(row.get("source/final_scientific_eligible", False))
            and not promotion_reasons
        )
    direct_stage_b_long_candidates = [
        str(row["candidate"])
        for row in rows
        if row["candidate"] in MATERIAL_LABELS
        and bool(row.get("promotion/eligible_for_direct_stage_b_long", False))
    ]
    effect_metric_keys = {
        "contraction_dihedral": ("roi/contraction/target_relative_dihedral_rms_rad"),
        "contraction_laplacian": (
            "roi/contraction/residual_normal_umbrella_laplacian_rms"
        ),
        "expansion_fidelity": "roi/expansion/error_rms_fraction_of_target",
        "activation_rms": "activation_inv/rms",
    }
    effects = {
        concept: effect_sizes(rows, key) for concept, key in effect_metric_keys.items()
    }
    first_crossing_effects = {
        concept: effect_sizes(rows, f"first_crossing/{key}")
        for concept, key in effect_metric_keys.items()
    }
    first_crossing_comparison_scientific = bool(
        matching.get("true_matched_fidelity", False)
        and matching.get("first_crossing_true_matched_fidelity", False)
    )
    first_crossing_sensitivity = {
        "fidelity_spread": matching.get("first_crossing_fidelity_spread"),
        "maximum_allowed_fidelity_spread": cfg.max_fidelity_spread,
        "fidelity_spread_gate_passed": matching.get(
            "first_crossing_spread_gate_passed", False
        ),
        "scientific_true_matched_fidelity": matching.get(
            "first_crossing_true_matched_fidelity", False
        ),
        "effect_interpretation": (
            "scientific_descriptive"
            if matching.get("first_crossing_true_matched_fidelity", False)
            else "diagnostic_only"
        ),
        "per_case_tau_brackets": {
            label: {
                key: value
                for key, value in selection.items()
                if key.startswith("selection/first_crossing_preceding_")
                or key
                in {
                    "selection/first_crossing_step",
                    "selection/first_crossing_fidelity",
                    "selection/first_crossing_brackets_tau",
                    "selection/first_crossing_bracket_width",
                    "selection/first_crossing_distance_above_tau",
                    "selection/first_crossing_distance_below_tau",
                }
            }
            for label, selection in matching.get("selections", {}).items()
        },
        "matched_checkpoint_pareto": first_crossing_pareto,
        "effect_sizes": first_crossing_effects,
        "front_stability_vs_closest_under": front_stability(
            matched_checkpoint_pareto,
            first_crossing_pareto,
            scientific=first_crossing_comparison_scientific,
        ),
        "effect_sign_stability_vs_closest_under": effect_sign_stability(
            effects,
            first_crossing_effects,
            scientific=first_crossing_comparison_scientific,
            zero_tolerance=cfg.pareto_atol,
        ),
    }

    material_only_rows = collect_selected_metrics(
        material_only,
        cases=cases,
        histories=histories,
        frame_traces=frame_traces,
        base_mesh=base_mesh,
        surface=surface,
        include_first_crossing=False,
    )
    material_only_pareto = pareto_analysis(material_only_rows, cfg)
    material_only_effects = {
        concept: effect_sizes(material_only_rows, key)
        for concept, key in effect_metric_keys.items()
    }
    material_only_comparison_scientific = bool(
        matching.get("true_matched_fidelity", False)
        and material_only.get("true_matched_fidelity", False)
    )
    material_only_sensitivity = {
        "matching": material_only,
        "tau_shift_vs_seven_case": (
            float(material_only["tau"]) - float(matching["tau"])
            if material_only.get("tau") is not None and matching.get("tau") is not None
            else None
        ),
        "selection_step_shift_vs_seven_case": {
            label: (
                int(material_only["selections"][label]["selection/step"])
                - int(matching["selections"][label]["selection/step"])
            )
            if label in material_only.get("selections", {})
            and label in matching.get("selections", {})
            else None
            for label in MATERIAL_LABELS
        },
        "effect_interpretation": (
            "scientific_descriptive"
            if material_only.get("true_matched_fidelity", False)
            else "diagnostic_only"
        ),
        "cases": [
            {
                key: value
                for key, value in row.items()
                if not key.startswith("promotion/")
            }
            for row in material_only_rows
            if row["candidate"] in MATERIAL_LABELS
        ],
        "matched_checkpoint_pareto": material_only_pareto,
        "effect_sizes": material_only_effects,
        "front_stability_vs_seven_case": front_stability(
            matched_checkpoint_pareto,
            material_only_pareto,
            scientific=material_only_comparison_scientific,
        ),
        "effect_sign_stability_vs_seven_case": effect_sign_stability(
            effects,
            material_only_effects,
            scientific=material_only_comparison_scientific,
            zero_tolerance=cfg.pareto_atol,
        ),
    }
    hard_failures: list[str] = []
    if not primary["available"]:
        hard_failures.extend(primary["failures"])
    elif not bool(primary["spread_gate_passed"]):
        hard_failures.append(
            "physical matched-fidelity spread "
            f"{primary['selected_fidelity_spread']:.6g} exceeds "
            f"{cfg.max_fidelity_spread:.6g}"
        )
    interpretation = (
        "fixed-budget-discrete-trajectory-true-matched-screen"
        if bool(primary.get("true_matched_fidelity", False))
        else "fixed-budget-discrete-trajectory-screen-unmatched"
    )
    payload = finite_or_none(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "human-face-smile-material-matched-fidelity-bumpiness-analysis",
            "complete": not hard_failures,
            "interpretation_scope": interpretation,
            "hard_failures": hard_failures,
            "source": {
                "canonical_source_policy": (
                    "live HEAD aggregate and its sibling artifacts; stale per-case "
                    "Cherries snapshots are not consulted"
                ),
                "summary": str(cfg.input_summary),
                "summary_identity": file_identity(cfg.input_summary),
                "candidate_manifest": str(cfg.input_candidates),
                "candidate_manifest_identity": file_identity(cfg.input_candidates),
                "input_mesh": str(cfg.input_mesh),
                "input_mesh_identity": file_identity(cfg.input_mesh),
                "source_complete": source["complete"],
                "source_convergence_failures": source["convergence_failures"],
                "source_pareto_candidates": source["pareto_candidates"],
                "history_and_result_identities": history_identities,
                "candidate_skin_identities": candidates.identities,
            },
            "validated_protocol": {
                "stage": "screen",
                "cases": list(EXPECTED_LABELS),
                "evaluations_per_case": EXPECTED_TRACE_STEPS,
                "total_temporal_frames": EXPECTED_TRACE_STEPS * len(EXPECTED_LABELS),
                "fresh_zero_activation": True,
                "activation_mode": "per-muscle-tet-6dof-unconstrained",
                "fixed_learning_rate": EXPECTED_LR,
                "common_mesh_correspondence": "GlobalPointId with fixed rest topology",
                "fidelity_definition": "||Displacement-TargetDisplacement||_F / ||TargetDisplacement||_F on SmileLossMask",
                "physical_prefix_definition": "all physical gates pass at every frame from step 0 through selected step",
            },
            "physical_gates": {
                "detF_no_inversions": True,
                "detF_min_positive": True,
                "detF_q001_min": cfg.min_det_f_q001,
                "skin_no_folds": True,
                "skin_area_ratio_q001_min": cfg.min_skin_area_ratio_q001,
                "skin_area_ratio_q999_max": cfg.max_skin_area_ratio_q999,
                "muscle_I_plus_ActivationInv_SPD": True,
                "muscle_min_eigenvalue": cfg.min_muscle_activation_eigenvalue,
            },
            "roi": {
                "canonical_surface": BASELINE_LABEL,
                "no_skin_surface_is_diagnostic_proxy": True,
                "eligible_triangles": int(surface.roi_masks["eligible"].sum()),
                "contraction_triangles": int(surface.roi_masks["contraction"].sum()),
                "expansion_triangles": int(surface.roi_masks["expansion"].sum()),
            },
            "matching": {
                "primary": primary,
                "diagnostic_fallback": diagnostic_fallback,
                "selection_rule": "closest-from-below before earliest admissible case best",
                "sensitivity_rule": "first crossing before earliest admissible case best",
            },
            "frame_traces": frame_traces,
            "cases": rows,
            "matched_checkpoint_pareto": matched_checkpoint_pareto,
            "promotion": {
                "direct_stage_b_long_candidates": direct_stage_b_long_candidates,
                "rule": (
                    "membership in matched_checkpoint_pareto.front AND source "
                    "final scientific eligibility"
                ),
            },
            "factor_effect_sizes": effects,
            "factor_effect_interpretation": (
                "scientific_descriptive"
                if matching.get("true_matched_fidelity", False)
                else "diagnostic_only"
            ),
            "first_crossing_sensitivity": first_crossing_sensitivity,
            "material_only_sensitivity": material_only_sensitivity,
            "limitations": [
                "single deterministic target with no replicate; effect sizes are descriptive, not inferential",
                "legacy umbrella metrics are exact only for this fixed topology and are not mesh-independent curvature",
                "no-skin surface quality is measured on the e100-p000 canonical surface as a diagnostic proxy",
                "discrete temporal states are never interpolated into a claimed physical solution",
            ],
        }
    )
    cfg.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(cfg.output_csv, rows)
    write_table(cfg.output_table, rows, matched_checkpoint_pareto)
    plot_trajectories(cfg.output_trajectory_plot, frame_traces, matching, cfg)
    plot_matched_screen(
        cfg.output_screen_plot, rows, matching, matched_checkpoint_pareto
    )
    for index, row in enumerate(rows):
        cherries.set_step(index)
        logged = {
            f"{row['candidate']}/selection_step": row.get("selection/step"),
            f"{row['candidate']}/selection_fidelity": row.get("selection/fidelity"),
            f"{row['candidate']}/physical_valid": row.get("physical/frame_valid"),
            f"{row['candidate']}/contraction_dihedral_rms": row.get(
                "roi/contraction/target_relative_dihedral_rms_rad"
            ),
        }
        cherries.log_metrics(
            {key: value for key, value in logged.items() if value is not None}
        )
    for path in (
        cfg.output_json,
        cfg.output_csv,
        cfg.output_table,
        cfg.output_trajectory_plot,
        cfg.output_screen_plot,
    ):
        logger.info("Wrote %s", path)
    if hard_failures:
        msg = (
            "material matched-fidelity analysis failed scientific gates: "
            + " | ".join(hard_failures)
        )
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(main)
