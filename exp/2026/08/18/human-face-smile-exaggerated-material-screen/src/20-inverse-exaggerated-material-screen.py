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
from _reference import PREPARED_MESH, enable_reference_modules

from liblaf import cherries

mpl.use("Agg", force=True)
enable_reference_modules()

import _human_face_runtime as reference_runtime  # noqa: E402
from _human_face_case import solve_case  # noqa: E402
from _human_face_config import (  # noqa: E402
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
    SETUP_SKIN_ESTIMATED_PRESTRAIN,
    SETUP_SKIN_NO_PRESTRAIN,
    SKIN_THICKNESS,
    InverseCase,
    configure_runtime,
)
from _human_face_forward import set_volume_material  # noqa: E402
from _human_face_runtime import CasePaths  # noqa: E402
from _material_heuristics import (  # noqa: E402
    file_sha256,
    skin_material_content_hash,
    skin_solver_content_hash,
    skin_topology_content_hash,
)

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1
EXPECTED_CANDIDATES = {
    "e100-p200": (1.0, 2.0),
    "e005-p000": (0.05, 0.0),
    "e005-p200": (0.05, 2.0),
}
EXPECTED_LABELS = tuple(EXPECTED_CANDIDATES)
EXPECTED_PROTOCOL = {
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
    "time_budget_hours": 6.0,
    "reserve_minutes": 5.0,
    "step_time_budget_s": 180.0,
    "require_convergence": False,
    "require_solver_success": True,
    "max_solver_failure_fraction": 0.0,
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_candidates: Path = cherries.input("10-exaggerated-materials-manifest.json")
    output_summary: Path = cherries.output(
        "20-exaggerated-material-screen-summary.json", mkdir=True
    )
    output_table: Path = cherries.output(
        "20-exaggerated-material-screen-table.md", mkdir=True
    )
    live_plot_dir: Path = Path("figs/live-exaggerated-material-screen")

    stage: str = "screen"
    candidate_set: str = ",".join(EXPECTED_LABELS)
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
    time_budget_hours: float = 6.0
    reserve_minutes: float = 5.0
    step_time_budget_s: float = 180.0
    require_convergence: bool = False
    require_solver_success: bool = True
    max_solver_failure_fraction: float = 0.0


def reject_json_constant(value: str) -> None:
    msg = f"manifest contains non-standard JSON constant {value!r}"
    raise ValueError(msg)


def require_finite_json(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{path} contains non-finite number {value}"
        raise ValueError(msg)
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, path=f"{path}[{index}]")


def verify_file_identity(path: Path, identity: Any, context: str) -> dict[str, Any]:
    if not isinstance(identity, dict):
        msg = f"{context} identity must be an object"
        raise TypeError(msg)
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


def load_manifest(cfg: Config) -> dict[str, Any]:
    manifest = json.loads(
        cfg.input_candidates.read_text(encoding="utf-8"),
        parse_constant=reject_json_constant,
    )
    if not isinstance(manifest, dict):
        msg = "candidate manifest must contain a JSON object"
        raise TypeError(msg)
    require_finite_json(manifest)
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        msg = f"unexpected candidate manifest schema: {manifest.get('schema_version')}"
        raise ValueError(msg)
    if manifest.get("complete") is not True:
        msg = "candidate manifest is incomplete"
        raise ValueError(msg)
    if manifest.get("design") != "exaggerated-heterogeneous-mechanism-screen":
        msg = f"unexpected experiment design: {manifest.get('design')!r}"
        raise ValueError(msg)
    verify_file_identity(
        cfg.input_mesh,
        manifest["input_mesh_identity"],
        "prepared input mesh",
    )
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(EXPECTED_LABELS):
        msg = "manifest must contain exactly the three exaggerated candidates"
        raise ValueError(msg)
    labels = tuple(str(row["label"]) for row in candidates)
    if labels != EXPECTED_LABELS:
        msg = f"candidate order changed: expected {EXPECTED_LABELS}, got {labels}"
        raise ValueError(msg)
    for row in candidates:
        label = str(row["label"])
        actual = (float(row["young_min_scale"]), float(row["prestrain_gain"]))
        if actual != EXPECTED_CANDIDATES[label]:
            msg = f"candidate {label} parameters changed: {actual}"
            raise ValueError(msg)
        if row.get("validation/ok") is not True or row.get("validation/errors") != []:
            msg = f"candidate {label} failed preparation validation"
            raise ValueError(msg)
    return manifest


def select_candidates(manifest: dict[str, Any], value: str) -> list[dict[str, Any]]:
    labels = tuple(item.strip() for item in value.split(",") if item.strip())
    if not labels or len(labels) != len(set(labels)):
        msg = f"candidate_set must be non-empty and unique, got {labels}"
        raise ValueError(msg)
    by_label = {str(row["label"]): dict(row) for row in manifest["candidates"]}
    unknown = sorted(set(labels) - set(by_label))
    if unknown:
        msg = f"unknown exaggerated candidates: {unknown}"
        raise ValueError(msg)
    return [by_label[label] for label in labels]


def verified_skin(
    cfg: Config, candidate: dict[str, Any]
) -> tuple[Path, pv.PolyData, dict[str, Any]]:
    relative = Path(str(candidate["skin/path"]))
    path = (cfg.input_candidates.parent / relative).resolve()
    root = cfg.input_candidates.parent.resolve()
    if not path.is_relative_to(root):
        msg = f"candidate skin escapes the manifest data directory: {path}"
        raise ValueError(msg)
    identity = verify_file_identity(
        path, candidate["skin/file_identity"], f"{candidate['label']} skin"
    )
    skin = pv.read(path)
    if not isinstance(skin, pv.PolyData):
        msg = f"candidate skin read back as {type(skin).__name__}"
        raise TypeError(msg)
    actual_hashes = {
        "topology_sha256": skin_topology_content_hash(skin),
        "material_sha256": skin_material_content_hash(skin),
        "solver_sha256": skin_solver_content_hash(skin),
    }
    for name, digest in actual_hashes.items():
        expected = str(candidate[f"content/{name}"])
        if digest != expected:
            msg = f"{candidate['label']} live {name} mismatch"
            raise ValueError(msg)
    cherries.log_input(path)
    provenance = {
        "provenance/skin_size_bytes": identity["size_bytes"],
        "provenance/skin_file_sha256": identity["sha256"],
        **{f"provenance/skin_{key}": value for key, value in actual_hashes.items()},
    }
    return path, skin, provenance


def build_candidate_forward(
    mesh: pv.UnstructuredGrid,
    _case: InverseCase,
    *,
    area_ratio_floor: float,
    skin_path: Path,
    skin: pv.PolyData,
    candidate: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[Any, pv.PolyData, dict[str, Any]]:
    del area_ratio_floor
    from liblaf.apple.common import GLOBAL_POINT_ID
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import Koiter, StableNeoHookean, StableNeoHookeanActive

    candidate_skin = skin.copy(deep=True)
    global_ids = np.asarray(
        candidate_skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
    )
    if (
        global_ids.size != candidate_skin.n_points
        or np.unique(global_ids).size != candidate_skin.n_points
        or global_ids.min() < 0
        or global_ids.max() >= mesh.n_points
    ):
        msg = f"{skin_path} has invalid GlobalPointId values"
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
    metrics = {
        **candidate,
        "material/candidate": str(candidate["label"]),
        "material/skin_path": str(skin_path),
        "skin/enabled": True,
        "skin/prestrain_enabled": float(candidate["prestrain_gain"]) > 0.0,
        "skin/young_spatially_varying": float(candidate["young_min_scale"]) < 1.0,
        **provenance,
    }
    return forward, candidate_skin, metrics


def normalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(summary)
    if normalized.get("initial_displacement/enabled") is not False:
        msg = "exaggerated screen unexpectedly reused an initial displacement"
        raise ValueError(msg)
    for key in ("initial_displacement/rms", "initial_displacement/max"):
        value = normalized.get(key)
        if value is not None and math.isfinite(float(value)):
            msg = f"{key} must be absent when initial displacement is disabled"
            raise ValueError(msg)
        normalized[key] = None
    return normalized


def deformation_warnings(
    result: pv.UnstructuredGrid, skin: pv.PolyData
) -> dict[str, Any]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    encoded = np.asarray(result.cells, dtype=np.int64).reshape(-1, 5)
    if not np.all(encoded[:, 0] == 4):
        msg = "deformation diagnostic expects tetrahedra"
        raise ValueError(msg)
    tets = encoded[:, 1:]
    rest = np.asarray(result.points, dtype=np.float64)
    displacement = np.asarray(result.point_data["Displacement"], dtype=np.float64)
    deformed = rest + displacement

    def six_volume(points: np.ndarray) -> np.ndarray:
        return np.einsum(
            "ij,ij->i",
            points[tets[:, 1]] - points[tets[:, 0]],
            np.cross(
                points[tets[:, 2]] - points[tets[:, 0]],
                points[tets[:, 3]] - points[tets[:, 0]],
            ),
        )

    det_f = six_volume(deformed) / six_volume(rest)
    faces = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    skin_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    result_ids = np.asarray(result.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    order = np.argsort(result_ids)
    positions = np.searchsorted(result_ids[order], skin_ids)
    if not np.array_equal(result_ids[order[positions]], skin_ids):
        msg = "skin IDs do not map to the result mesh"
        raise ValueError(msg)
    skin_displacement = displacement[order[positions]]
    skin_rest = np.asarray(skin.points, dtype=np.float64)
    skin_deformed = skin_rest + skin_displacement

    def area_vectors(points: np.ndarray) -> np.ndarray:
        return np.cross(
            points[faces[:, 1]] - points[faces[:, 0]],
            points[faces[:, 2]] - points[faces[:, 0]],
        )

    rest_area_vector = area_vectors(skin_rest)
    deformed_area_vector = area_vectors(skin_deformed)
    rest_norm = np.linalg.norm(rest_area_vector, axis=1)
    deformed_norm = np.linalg.norm(deformed_area_vector, axis=1)
    area_ratio = deformed_norm / rest_norm
    signed_normal_ratio = np.einsum(
        "ij,ij->i", deformed_area_vector, rest_area_vector
    ) / np.square(rest_norm)
    if not (
        np.isfinite(det_f).all()
        and np.isfinite(area_ratio).all()
        and np.isfinite(signed_normal_ratio).all()
    ):
        msg = "deformation diagnostic contains non-finite values"
        raise ValueError(msg)
    return {
        "warning/inverted_tets": int(np.sum(det_f <= 0.0)),
        "warning/inverted_tet_fraction": float(np.mean(det_f <= 0.0)),
        "warning/detF_min": float(det_f.min()),
        "warning/detF_q001": float(np.quantile(det_f, 0.001)),
        "warning/skin_folded_triangles": int(np.sum(signed_normal_ratio <= 0.0)),
        "warning/skin_folded_triangle_fraction": float(
            np.mean(signed_normal_ratio <= 0.0)
        ),
        "warning/skin_area_ratio_q001": float(np.quantile(area_ratio, 0.001)),
        "warning/skin_area_ratio_q999": float(np.quantile(area_ratio, 0.999)),
    }


def validate_case(  # noqa: C901, PLR0912, PLR0915
    summary: dict[str, Any], paths: CasePaths, skin: pv.PolyData, cfg: Config
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    expected_evaluations = cfg.inverse_max_steps + 1
    if float(summary.get("activation_inv/initial_rms", math.nan)) != 0.0:
        errors.append("initial activation RMS is not zero")
    if float(summary.get("activation_inv/initial_max_abs", math.nan)) != 0.0:
        errors.append("initial activation max is not zero")
    if summary.get("activation/mode") != "per-muscle-tet-6dof":
        errors.append("activation mode is not per-muscle-tet-6dof")
    if int(summary.get("n_activation_parameter_dofs", -1)) != 6 * int(
        summary.get("n_active_tets", 0)
    ):
        errors.append("activation DoF count is not six per active muscle tet")
    if summary.get("baseline/completed") is not True:
        errors.append("fixed-budget trajectory did not complete")
    if int(summary.get("inverse/evaluations", -1)) != expected_evaluations:
        errors.append("inverse evaluation count differs from the fixed budget")
    if int(summary.get("baseline/evaluations", -1)) != expected_evaluations:
        errors.append("baseline evaluation count differs from the fixed budget")
    if int(summary.get("baseline/evaluations_expected", -1)) != expected_evaluations:
        errors.append(
            "expected baseline evaluation count differs from the fixed budget"
        )
    if int(summary.get("baseline/mandatory_optimizer_steps", -1)) != (
        cfg.inverse_max_steps
    ):
        errors.append("mandatory optimizer-step budget changed")
    if float(summary.get("baseline/fixed_lr", math.nan)) != cfg.inverse_lr:
        errors.append("fixed-budget learning rate changed")
    if int(summary.get("baseline/lr_deviation_count", -1)) != 0:
        errors.append("fixed-budget trajectory changed learning rate")
    errors.extend(
        f"{key} differs from the fixed-budget frame count"
        for key in ("history/frames", "history_frames")
        if int(summary.get(key, -1)) != expected_evaluations
    )

    trace = list(summary.get("trace", []))
    if [int(row["step"]) for row in trace] != list(range(expected_evaluations)):
        errors.append("trace is not the complete contiguous fixed-budget trajectory")
    finite_keys = (
        "loss/total",
        "target/error_rms",
        "grad/norm",
        "forward/relative_grad_norm",
        "adjoint/relative_residual",
    )
    for row in trace:
        step = int(row["step"])
        if not bool(row.get("forward/success", False)):
            errors.append(f"step {step} forward solve failed")
        if not bool(row.get("adjoint/success", False)):
            errors.append(f"step {step} adjoint solve failed")
        errors.extend(
            f"step {step} has non-finite {key}"
            for key in finite_keys
            if not math.isfinite(float(row.get(key, math.nan)))
        )
    if trace:
        if float(trace[0]["activation_inv/rms"]) != 0.0:
            errors.append("step-0 activation RMS is not zero")
        if float(trace[0]["activation_inv/max_abs"]) != 0.0:
            errors.append("step-0 activation max is not zero")
        best_step = int(summary.get("best/step", -1))
        best_rows = [row for row in trace if int(row["step"]) == best_step]
        if len(best_rows) != 1:
            errors.append(f"best step {best_step} does not identify one trace row")
        elif not bool(best_rows[0].get("best/accepted", False)):
            errors.append("best inverse state was not accepted")
    if not bool(summary.get("last/forward/success", False)):
        errors.append("last forward solve failed")
    if not bool(summary.get("last/adjoint/success", False)):
        errors.append("last adjoint solve failed")

    for name, path in {
        "target": paths.target,
        "result": paths.result,
        "summary": paths.summary,
        "history": paths.history,
        "trace": paths.trace,
    }.items():
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {name} artifact: {path}")
    if paths.history.is_file():
        from vtkmodules.vtkCommonExecutionModel import (
            vtkStreamingDemandDrivenPipeline as StreamingPipeline,
        )

        reader = pv.get_reader(paths.history)
        vtk_reader = reader.reader
        vtk_reader.UpdateInformation()
        information = vtk_reader.GetOutputInformation(0)
        key = StreamingPipeline.TIME_STEPS()
        times = np.asarray(
            [information.Get(key, index) for index in range(information.Length(key))],
            dtype=np.float64,
        )
        expected_times = np.arange(expected_evaluations, dtype=np.float64)
        if expected_evaluations == 1 and times.size == 0:
            single = pv.read(paths.history)
            stored_step = int(
                np.asarray(single.field_data["inverse_step"]).reshape(-1)[0]
            )
            if stored_step != 0:
                errors.append("single-frame VTKHDF does not store inverse step 0")
        elif not np.array_equal(times, expected_times):
            errors.append("VTKHDF TIME_STEPS are not the complete fixed-budget frames")
    diagnostic: dict[str, Any] = {}
    if paths.result.is_file():
        result = pv.read(paths.result)
        if not isinstance(result, pv.UnstructuredGrid):
            errors.append(f"result read back as {type(result).__name__}")
        else:
            for name in ("Displacement", "TargetDisplacement"):
                values = np.asarray(result.point_data[name], dtype=np.float64)
                if not np.isfinite(values).all():
                    errors.append(f"result {name} contains non-finite values")
            activation = np.asarray(
                result.cell_data["RecoveredActivationInv"], dtype=np.float64
            )
            if not np.isfinite(activation).all():
                errors.append("result activation contains non-finite values")
            if not errors:
                diagnostic = deformation_warnings(result, skin)
                if int(diagnostic["warning/inverted_tets"]) > 0:
                    warnings.append(
                        f"{diagnostic['warning/inverted_tets']} inverted tets; visual warning only"
                    )
                if int(diagnostic["warning/skin_folded_triangles"]) > 0:
                    warnings.append(
                        f"{diagnostic['warning/skin_folded_triangles']} folded skin triangles; visual warning only"
                    )
    return sorted(set(errors)), warnings, diagnostic


def validate_config(cfg: Config) -> None:
    if cfg.initial_activation_mesh is not None or cfg.use_initial_displacement:
        msg = "each exaggerated candidate must start from fresh zero state"
        raise ValueError(msg)
    if str(mpl.get_backend()).lower() != "agg":
        msg = f"non-interactive Agg backend required, got {mpl.get_backend()}"
        raise RuntimeError(msg)
    protocol = {key: getattr(cfg, key) for key in EXPECTED_PROTOCOL}
    if protocol != EXPECTED_PROTOCOL:
        msg = f"fixed inverse protocol changed: expected {EXPECTED_PROTOCOL}, got {protocol}"
        raise ValueError(msg)
    labels = tuple(
        item.strip() for item in cfg.candidate_set.split(",") if item.strip()
    )
    if cfg.stage == "screen":
        if (
            labels != EXPECTED_LABELS
            or cfg.inverse_max_steps != 40
            or cfg.mandatory_baseline_steps != 40
        ):
            msg = "formal screen requires all three candidates and exactly 40 steps"
            raise ValueError(msg)
    elif cfg.stage == "smoke":
        if (
            len(labels) != 1
            or cfg.inverse_max_steps != 0
            or cfg.mandatory_baseline_steps != 0
        ):
            msg = "smoke requires one candidate and zero optimizer steps"
            raise ValueError(msg)
    else:
        msg = f"stage must be screen or smoke, got {cfg.stage!r}"
        raise ValueError(msg)
    if cfg.output_summary.resolve() == cfg.output_table.resolve():
        msg = "summary and table output paths must differ"
        raise ValueError(msg)
    stage_paths = (cfg.output_summary, cfg.output_table, cfg.live_plot_dir)
    unsafe = [str(path) for path in stage_paths if cfg.stage not in path.name]
    if unsafe:
        msg = f"stage {cfg.stage!r} must appear in every output path: {unsafe}"
        raise ValueError(msg)


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| candidate | status | evals | best step | error/target | disp Lap RMS | residual Lap RMS | inv tets | folds | warnings |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"| {row['candidate']} | {row.get('status')} | - | - | - | - | - | - | - | `{row.get('error', '')}` |"
            )
            continue
        lines.append(
            "| {candidate} | ok | {inverse/evaluations} | {best/step} | "
            "{best/error_rms_fraction_of_target:.6g} | "
            "{bumpiness/displacement_laplacian_rms:.6g} | "
            "{bumpiness/residual_laplacian_rms:.6g} | "
            "{warning/inverted_tets} | {warning/skin_folded_triangles} | {warning_text} |".format(
                **row,
                warning_text="; ".join(row.get("validation/warnings", [])) or "-",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: Config) -> None:  # noqa: PLR0915
    validate_config(cfg)
    manifest = load_manifest(cfg)
    selected = select_candidates(manifest, cfg.candidate_set)
    verified = {
        str(candidate["label"]): verified_skin(cfg, candidate) for candidate in selected
    }
    configure_runtime()
    base_mesh = pv.read(cfg.input_mesh)
    if not isinstance(base_mesh, pv.UnstructuredGrid):
        base_mesh = base_mesh.cast_to_unstructured_grid()

    original_builder: Callable[..., Any] = reference_runtime.build_forward
    rows: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    for candidate in selected:
        label = str(candidate["label"])
        skin_path, skin, provenance = verified[label]
        setup = (
            SETUP_SKIN_ESTIMATED_PRESTRAIN
            if float(candidate["prestrain_gain"]) > 0.0
            else SETUP_SKIN_NO_PRESTRAIN
        )
        case = InverseCase(
            target="smile",
            lr=cfg.inverse_lr,
            setup=setup,
            label=f"exaggerated-{label}-{cfg.stage}",
        )
        paths = CasePaths.from_case(cfg.output_summary.parent, case)
        builder_calls = 0

        def independent_builder(
            mesh: pv.UnstructuredGrid,
            inverse_case: InverseCase,
            *,
            area_ratio_floor: float,
            _candidate: dict[str, Any] = candidate,
            _skin_path: Path = skin_path,
            _skin: pv.PolyData = skin,
            _provenance: dict[str, Any] = provenance,
            _label: str = label,
        ) -> tuple[Any, pv.PolyData, dict[str, Any]]:
            nonlocal builder_calls
            builder_calls += 1
            if builder_calls != 1:
                msg = f"{_label} requested more than one forward builder"
                raise RuntimeError(msg)
            return build_candidate_forward(
                mesh,
                inverse_case,
                area_ratio_floor=area_ratio_floor,
                skin_path=_skin_path,
                skin=_skin,
                candidate=_candidate,
                provenance=_provenance,
            )

        reference_runtime.build_forward = independent_builder
        try:
            summary = normalize_summary(
                solve_case(case, base_mesh.copy(deep=True), cfg)
            )
            errors, warnings, diagnostics = validate_case(summary, paths, skin, cfg)
            if builder_calls != 1:
                errors.append(f"forward builder was called {builder_calls} times")
            row = {
                **summary,
                **diagnostics,
                "candidate": label,
                "candidate/young_min_scale": float(candidate["young_min_scale"]),
                "candidate/prestrain_gain": float(candidate["prestrain_gain"]),
                "candidate/skin_path": str(skin_path),
                "stage": cfg.stage,
                "builder/fresh_independent": builder_calls == 1,
                "builder/calls": builder_calls,
                "comparison/numerically_eligible_pending_visual_review": not errors,
                "comparison/eligibility_policy": (
                    "solver, finite trajectory, complete fixed budget, and artifact readback; "
                    "inversion/fold counts are warnings and final acceptability requires "
                    "standard-view visual review"
                ),
                "validation/errors": errors,
                "validation/warnings": warnings,
                "artifact/summary_path": str(paths.summary),
                **provenance,
                "status": "ok" if not errors else "invalid",
            }
            paths.summary.write_text(
                json.dumps(row, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            rows.append(row)
            if errors:
                hard_failures.append(f"{label}: " + "; ".join(errors))
        except Exception as error:
            logger.exception("exaggerated candidate %s failed", label)
            failed = {
                "candidate": label,
                "stage": cfg.stage,
                "status": "failed",
                "comparison/numerically_eligible_pending_visual_review": False,
                "error": f"{type(error).__name__}: {error}",
                "artifact/summary_path": str(paths.summary),
            }
            paths.summary.parent.mkdir(parents=True, exist_ok=True)
            paths.summary.write_text(
                json.dumps(failed, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            rows.append(failed)
            hard_failures.append(f"{label}: {type(error).__name__}: {error}")
        finally:
            reference_runtime.build_forward = original_builder

    aggregate = {
        "schema_version": 1,
        "complete": not hard_failures and len(rows) == len(selected),
        "design": "exaggerated-heterogeneous-mechanism-screen",
        "stage": cfg.stage,
        "candidate_set": cfg.candidate_set,
        "input_mesh": str(cfg.input_mesh),
        "input_candidates": str(cfg.input_candidates),
        "fresh_zero_activation": True,
        "activation_mode": "per-muscle-tet-6dof-unconstrained",
        "activation_shared": False,
        "activation_transferred_between_candidates": False,
        "forward_builder_shared_between_candidates": False,
        "inverse_lr": cfg.inverse_lr,
        "inverse_max_steps": cfg.inverse_max_steps,
        "plot_backend": str(mpl.get_backend()),
        "acceptance_policy": {
            "hard": [
                "complete fixed-budget trajectory",
                "every forward and adjoint solve succeeds",
                "finite metrics and readable artifacts",
            ],
            "visual_warnings_only": [
                "inverted tetrahedra",
                "folded skin triangles",
                "detF and area-ratio tails",
            ],
        },
        "hard_failures": hard_failures,
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
        msg = "exaggerated material screen failed: " + " | ".join(hard_failures)
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(run)
