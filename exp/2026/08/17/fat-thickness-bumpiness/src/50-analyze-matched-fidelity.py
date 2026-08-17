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
from _common import resolve_recorded_path, toy
from _surface_metrics import (
    ResampledSurface,
    common_xz_bounds,
    finite_scalar_metrics,
    gaussian_smooth,
    rms,
    surface_metrics,
    top_surface_ids,
    vector_rms,
)
from scipy.spatial import Delaunay, cKDTree
from vtkmodules.vtkCommonExecutionModel import (
    vtkStreamingDemandDrivenPipeline as StreamingPipeline,
)

mpl.use("Agg")
import matplotlib.pyplot as plt

from liblaf import cherries

logger = logging.getLogger(__name__)

EXPECTED_LABELS = ("thin", "current", "thick")
MIN_ROBUST_MASK_POINTS = 25
CORE_ROUGHNESS_METRICS = (
    "grid/displacement_y_highpass_rms",
    "grid/displacement_y_highpass_over_rms",
    "grid/displacement_y_laplacian_rms",
    "grid/displacement_y_laplacian_over_rms",
    "grid/interior_displacement_y_highpass_rms",
    "grid/interior_displacement_y_highpass_over_rms",
    "grid/muscle_footprint_displacement_y_highpass_rms",
    "grid/muscle_footprint_displacement_y_highpass_over_rms",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_manifest: Path = cherries.input("40-inverse-manifest.json")
    output_csv: Path = cherries.output("50-matched-fidelity.csv", mkdir=True)
    output_json: Path = cherries.output("50-matched-fidelity.json", mkdir=True)
    output_metric_plot: Path = cherries.output("50-matched-fidelity.png", mkdir=True)
    output_field_plot: Path = cherries.output(
        "50-matched-fidelity-fields.png", mkdir=True
    )

    grid_size: int = 129
    high_frequency_cutoff_cycles: float = 8.0
    laplacian_smoothing_length: float = 0.04
    max_common_fidelity_spread: float = 0.001


def finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def require_scalar(mapping: dict[str, Any], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if not finite(value):
        msg = f"{context} has no finite {key!r}: {value!r}"
        raise ValueError(msg)
    return float(value)


def case_label(case: dict[str, Any]) -> str:
    label = case.get("label")
    if not isinstance(label, str) or not label:
        msg = f"inverse case has no non-empty label: {label!r}"
        raise ValueError(msg)
    return label


def validate_config(cfg: Config) -> None:
    if cfg.grid_size < 5:
        msg = f"grid_size must be at least 5, got {cfg.grid_size}"
        raise ValueError(msg)
    if cfg.high_frequency_cutoff_cycles <= 0.0 or not math.isfinite(
        cfg.high_frequency_cutoff_cycles
    ):
        msg = (
            "high_frequency_cutoff_cycles must be finite and positive, got "
            f"{cfg.high_frequency_cutoff_cycles}"
        )
        raise ValueError(msg)
    if cfg.laplacian_smoothing_length <= 0.0 or not math.isfinite(
        cfg.laplacian_smoothing_length
    ):
        msg = (
            "laplacian_smoothing_length must be finite and positive, got "
            f"{cfg.laplacian_smoothing_length}"
        )
        raise ValueError(msg)
    if cfg.max_common_fidelity_spread < 0.0 or not math.isfinite(
        cfg.max_common_fidelity_spread
    ):
        msg = (
            "max_common_fidelity_spread must be finite and non-negative, got "
            f"{cfg.max_common_fidelity_spread}"
        )
        raise ValueError(msg)


def ordered_trace(case: dict[str, Any]) -> list[dict[str, Any]]:
    label = case_label(case)
    raw_trace = case.get("trace")
    if (
        not isinstance(raw_trace, list)
        or not raw_trace
        or not all(isinstance(row, dict) for row in raw_trace)
    ):
        msg = f"case {label!r} has no non-empty inverse trace"
        raise ValueError(msg)
    trace = list(raw_trace)
    steps: list[int] = []
    for index, row in enumerate(trace):
        step_value = require_scalar(row, "step", context=f"case {label} trace[{index}]")
        step = int(step_value)
        if step_value != step:
            msg = f"case {label!r} has non-integral trace step {step_value!r}"
            raise ValueError(msg)
        steps.append(step)
        for key in ("target/error_rms", "loss/total"):
            require_scalar(row, key, context=f"case {label} step {step}")
        for key in ("loss/residual_laplacian", "loss/activation_smooth"):
            value = require_scalar(row, key, context=f"case {label} step {step}")
            if value != 0.0:
                msg = (
                    f"case {label!r} has auxiliary loss {key}={value} "
                    f"at trace step {step}"
                )
                raise ValueError(msg)
        if row.get("forward/success") is not True:
            msg = f"case {label!r} forward failed at trace step {step}"
            raise ValueError(msg)
        if row.get("adjoint/success") is not True:
            msg = f"case {label!r} adjoint failed at trace step {step}"
            raise ValueError(msg)
    expected_steps = list(range(len(trace)))
    if steps != expected_steps:
        msg = (
            f"case {label!r} trace steps are not contiguous 0..{len(trace) - 1}: "
            f"{steps[:5]} ... {steps[-5:]}"
        )
        raise ValueError(msg)
    return trace


def validate_case_initialization(case: dict[str, Any]) -> None:
    label = case_label(case)
    initial = tuple(
        require_scalar(
            case,
            f"inverse/initial_activation_inv/{component}",
            context=f"case {label}",
        )
        for component in ("x", "y", "z", "xy", "yz", "xz")
    )
    if initial != (0.0,) * 6:
        msg = f"case {label!r} does not use fresh zero initialization: {initial}"
        raise ValueError(msg)


def validate_case_setup(case: dict[str, Any]) -> None:
    label = case_label(case)
    if case.get("status") != "ok":
        msg = f"case {label!r} status is {case.get('status')!r}, expected 'ok'"
        raise ValueError(msg)
    if case.get("validation/errors") != []:
        msg = f"case {label!r} has validation errors: {case.get('validation/errors')!r}"
        raise ValueError(msg)
    if case.get("forward/success") is not True:
        msg = f"case {label!r} does not have successful forward solves"
        raise ValueError(msg)
    if case.get("adjoint/success") is not True:
        msg = f"case {label!r} does not have successful adjoint solves"
        raise ValueError(msg)
    if case.get("activation/mode") != "per-tet" or bool(
        case.get("activation/shared", True)
    ):
        msg = f"case {label!r} is not independent per-tet activation"
        raise ValueError(msg)
    if bool(case.get("skin/energy_enabled", True)):
        msg = f"case {label!r} is not a no-skin inverse setup"
        raise ValueError(msg)
    if case.get("loss_variant") != "l2":
        msg = (
            f"case {label!r} loss variant is {case.get('loss_variant')!r}, expected l2"
        )
        raise ValueError(msg)
    for key in (
        "loss/residual_laplacian_enabled",
        "loss/activation_smooth_enabled",
    ):
        if case.get(key) is not False:
            msg = f"case {label!r} auxiliary loss flag {key} is not false"
            raise ValueError(msg)
    validate_case_initialization(case)


def validate_case_trace(case: dict[str, Any]) -> list[dict[str, Any]]:
    label = case_label(case)
    trace = ordered_trace(case)
    frames = case.get("history/frames")
    evaluations = case.get("inverse/evaluations")
    if frames != len(trace) or evaluations != len(trace):
        msg = (
            f"case {label!r} has trace/history mismatch: trace={len(trace)}, "
            f"frames={frames!r}, evaluations={evaluations!r}"
        )
        raise ValueError(msg)
    target_rms = require_scalar(
        case, "target/displacement_rms", context=f"case {label}"
    )
    if target_rms <= 0.0:
        msg = f"case {label!r} target RMS must be positive, got {target_rms}"
        raise ValueError(msg)
    return trace


def validate_case_best(case: dict[str, Any], trace: list[dict[str, Any]]) -> None:
    label = case_label(case)
    target_rms = require_scalar(
        case, "target/displacement_rms", context=f"case {label}"
    )
    best_step_value = require_scalar(case, "best/step", context=f"case {label}")
    best_step = int(best_step_value)
    if best_step_value != best_step or not 0 <= best_step < len(trace):
        msg = f"case {label!r} has invalid best step {best_step_value!r}"
        raise ValueError(msg)
    best_fraction = require_scalar(
        case,
        "best/error_rms_fraction_of_target",
        context=f"case {label}",
    )
    trace_best_fraction = (
        require_scalar(
            trace[best_step],
            "target/error_rms",
            context=f"case {label} best trace step",
        )
        / target_rms
    )
    if not math.isclose(
        best_fraction, trace_best_fraction, rel_tol=1.0e-10, abs_tol=1.0e-12
    ):
        msg = (
            f"case {label!r} best error fraction {best_fraction} disagrees with "
            f"trace step {best_step}: {trace_best_fraction}"
        )
        raise ValueError(msg)


def validate_case(case: dict[str, Any], *, manifest_path: Path) -> None:
    label = case_label(case)
    validate_case_setup(case)
    trace = validate_case_trace(case)
    validate_case_best(case, trace)
    history_path = resolve_recorded_path(
        manifest_path, str(case.get("history/path", ""))
    )
    if not history_path.is_file():
        msg = f"case {label!r} history does not exist: {history_path}"
        raise FileNotFoundError(msg)


def validate_incomplete_manifest(
    manifest: dict[str, Any], cases: list[dict[str, Any]]
) -> None:
    if manifest.get("complete") is True:
        return
    invalid_stops = [
        f"{case_label(case)}={case.get('inverse/stop_reason')!r}"
        for case in cases
        if not bool(case.get("inverse/converged", False))
        and case.get("inverse/stop_reason") != "step_limit"
    ]
    if invalid_stops:
        msg = (
            "incomplete source inverse is only accepted for step-limit cases; "
            "invalid stops: " + ", ".join(invalid_stops)
        )
        raise ValueError(msg)
    if all(bool(case.get("inverse/converged", False)) for case in cases):
        msg = "source inverse is incomplete although every case is converged"
        raise ValueError(msg)


def validate_manifest(
    manifest: dict[str, Any], *, manifest_path: Path
) -> list[dict[str, Any]]:
    if manifest.get("setup") != "no-skin-l2-per-tet-6dof":
        msg = (
            f"source setup is {manifest.get('setup')!r}, expected "
            "'no-skin-l2-per-tet-6dof'"
        )
        raise ValueError(msg)
    if manifest.get("hard_failures") != []:
        msg = f"source inverse has hard failures: {manifest.get('hard_failures')!r}"
        raise ValueError(msg)
    if manifest.get("fresh_optimizer_per_case") is not True:
        msg = "source inverse did not use a fresh optimizer per case"
        raise ValueError(msg)
    initial = manifest.get("initial_activation_inv")
    if initial != [0.0] * 6:
        msg = f"source inverse initial activation is not six zeros: {initial!r}"
        raise ValueError(msg)
    raw_cases = manifest.get("cases")
    if (
        not isinstance(raw_cases, list)
        or not raw_cases
        or not all(isinstance(case, dict) for case in raw_cases)
    ):
        msg = "source inverse manifest has no non-empty case list"
        raise ValueError(msg)
    cases = list(raw_cases)
    labels = tuple(case_label(case) for case in cases)
    if labels != EXPECTED_LABELS:
        msg = f"source labels are {labels}, expected {EXPECTED_LABELS}"
        raise ValueError(msg)
    for case in cases:
        validate_case(case, manifest_path=manifest_path)
    validate_incomplete_manifest(manifest, cases)
    return cases


def trace_fraction(case: dict[str, Any], row: dict[str, Any]) -> float:
    target_rms = require_scalar(
        case, "target/displacement_rms", context=f"case {case_label(case)}"
    )
    error_rms = require_scalar(
        row,
        "target/error_rms",
        context=f"case {case_label(case)} trace",
    )
    return error_rms / target_rms


def field_scalar(mesh: pv.DataSet, name: str) -> float:
    if name not in mesh.field_data:
        msg = f"history frame has no field_data[{name!r}]"
        raise KeyError(msg)
    values = np.asarray(mesh.field_data[name]).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        msg = f"history field {name!r} is not one finite scalar: {values!r}"
        raise ValueError(msg)
    return float(values[0])


def temporal_values(reader: Any) -> np.ndarray:
    reader.UpdateInformation()
    information = reader.GetOutputInformation(0)
    key = StreamingPipeline.TIME_STEPS()
    if not information.Has(key):
        msg = "VTKHDF reader exposes no TIME_STEPS"
        raise ValueError(msg)
    values = np.asarray(
        [information.Get(key, index) for index in range(information.Length(key))],
        dtype=np.float64,
    )
    if values.size == 0 or not np.isfinite(values).all():
        msg = f"VTKHDF reader exposes invalid time values: {values!r}"
        raise ValueError(msg)
    return values


def trace_row(case: dict[str, Any], step: int) -> dict[str, Any]:
    trace = ordered_trace(case)
    if not 0 <= step < len(trace):
        msg = f"case {case_label(case)!r} has no trace step {step}"
        raise IndexError(msg)
    row = trace[step]
    if int(float(row["step"])) != step:
        msg = f"case {case_label(case)!r} trace index {step} has wrong step"
        raise ValueError(msg)
    return row


@dataclass
class TemporalHistory:
    case: dict[str, Any]
    path: Path
    pyvista_reader: Any
    times: np.ndarray

    @classmethod
    def open(cls, case: dict[str, Any], *, manifest_path: Path) -> TemporalHistory:
        label = case_label(case)
        path = resolve_recorded_path(manifest_path, str(case["history/path"]))
        pyvista_reader = pv.get_reader(path)
        times = temporal_values(pyvista_reader.reader)
        trace = ordered_trace(case)
        expected = np.arange(len(trace), dtype=np.float64)
        if times.shape != expected.shape or not np.allclose(
            times, expected, rtol=0.0, atol=1.0e-12
        ):
            msg = (
                f"case {label!r} history times do not match trace steps: "
                f"shape={times.shape}, expected={expected.shape}"
            )
            raise ValueError(msg)
        return cls(
            case=case,
            path=path,
            pyvista_reader=pyvista_reader,
            times=times,
        )

    def frame(self, step: int, *, deep_copy: bool) -> pv.UnstructuredGrid:
        label = case_label(self.case)
        row = trace_row(self.case, step)
        if not 0 <= step < self.times.size:
            msg = f"case {label!r} history has no step {step}"
            raise IndexError(msg)
        vtk_reader = self.pyvista_reader.reader
        vtk_reader.UpdateTimeStep(float(self.times[step]))
        output = pv.wrap(vtk_reader.GetOutputDataObject(0))
        if deep_copy:
            output = output.copy(deep=True)
        if not isinstance(output, pv.UnstructuredGrid):
            output = output.cast_to_unstructured_grid()

        actual_step = field_scalar(output, "inverse_step")
        actual_error = field_scalar(output, "inverse_error_rms")
        actual_loss = field_scalar(output, "inverse_loss")
        if actual_step != step:
            msg = f"history {self.path} returned step {actual_step}, requested {step}"
            raise ValueError(msg)
        for name, actual, expected in (
            ("inverse_error_rms", actual_error, float(row["target/error_rms"])),
            ("inverse_loss", actual_loss, float(row["loss/total"])),
        ):
            if not math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=1.0e-12):
                msg = (
                    f"history {self.path} step {step} field {name}={actual} "
                    f"disagrees with trace value {expected}"
                )
                raise ValueError(msg)
        return output


@dataclass(frozen=True)
class CommonGridPlan:
    n_points: int
    n_cells: int
    top_ids: np.ndarray
    reference_top_xz: np.ndarray
    x: np.ndarray
    z: np.ndarray
    valid: np.ndarray
    valid_flat_ids: np.ndarray
    vertices: np.ndarray
    weights: np.ndarray
    missing_flat_ids: np.ndarray
    nearest_top_ids: np.ndarray

    @classmethod
    def build(
        cls,
        mesh: pv.UnstructuredGrid,
        *,
        bounds: tuple[float, float, float, float],
        grid_size: int,
    ) -> CommonGridPlan:
        ids = top_surface_ids(mesh)
        points_xz = np.asarray(mesh.points, dtype=np.float64)[ids][:, (0, 2)]
        if np.unique(points_xz, axis=0).shape[0] != points_xz.shape[0]:
            msg = "top surface is not a single-valued graph over x-z"
            raise ValueError(msg)
        xmin, xmax, zmin, zmax = bounds
        x_axis = np.linspace(xmin, xmax, grid_size)
        z_axis = np.linspace(zmin, zmax, grid_size)
        x, z = np.meshgrid(x_axis, z_axis, indexing="ij")
        query = np.column_stack((x.ravel(), z.ravel()))

        triangulation = Delaunay(points_xz)
        simplex = triangulation.find_simplex(query)
        valid_flat = simplex >= 0
        valid_flat_ids = np.flatnonzero(valid_flat)
        missing_flat_ids = np.flatnonzero(~valid_flat)
        valid = valid_flat.reshape(grid_size, grid_size)
        boundary = np.zeros_like(valid)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        missing = ~valid
        n_missing = int(missing.sum())
        max_boundary_missing = max(4, math.ceil(1.0e-3 * valid.size))
        if np.any(missing & ~boundary) or n_missing > max_boundary_missing:
            msg = (
                f"common-grid interpolation has {n_missing} missing points, "
                f"including {int((missing & ~boundary).sum())} interior points; "
                f"allowed at most {max_boundary_missing} boundary points"
            )
            raise ValueError(msg)

        valid_simplex = simplex[valid_flat]
        transforms = triangulation.transform[valid_simplex]
        delta = query[valid_flat] - transforms[:, 2, :]
        leading_weights = np.einsum("nij,nj->ni", transforms[:, :2, :], delta)
        weights = np.column_stack((leading_weights, 1.0 - leading_weights.sum(axis=1)))
        vertices = triangulation.simplices[valid_simplex]
        nearest_top_ids = np.asarray(
            cKDTree(points_xz).query(query[~valid_flat])[1], dtype=np.int64
        )
        return cls(
            n_points=mesh.n_points,
            n_cells=mesh.n_cells,
            top_ids=ids,
            reference_top_xz=points_xz,
            x=x,
            z=z,
            valid=valid,
            valid_flat_ids=valid_flat_ids,
            vertices=vertices,
            weights=weights,
            missing_flat_ids=missing_flat_ids,
            nearest_top_ids=nearest_top_ids,
        )

    def validate_mesh(self, mesh: pv.UnstructuredGrid) -> None:
        if mesh.n_points != self.n_points or mesh.n_cells != self.n_cells:
            msg = (
                "temporal frame topology changed: "
                f"points={mesh.n_points}/{self.n_points}, "
                f"cells={mesh.n_cells}/{self.n_cells}"
            )
            raise ValueError(msg)
        points_xz = np.asarray(mesh.points, dtype=np.float64)[self.top_ids][:, (0, 2)]
        if not np.allclose(points_xz, self.reference_top_xz, rtol=0.0, atol=1.0e-12):
            msg = "temporal frame changed the rest top-surface x-z coordinates"
            raise ValueError(msg)

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[0] != self.n_points:
            msg = (
                f"point values have leading size {array.shape[0]}, expected "
                f"{self.n_points}"
            )
            raise ValueError(msg)
        top_values = array[self.top_ids]
        flat_shape = (self.valid.size, *array.shape[1:])
        result = np.empty(flat_shape, dtype=np.float64)
        result[self.valid_flat_ids] = np.einsum(
            "ij,ij...->i...", self.weights, top_values[self.vertices]
        )
        if self.missing_flat_ids.size:
            result[self.missing_flat_ids] = top_values[self.nearest_top_ids]
        if not np.isfinite(result).all():
            msg = "common-grid interpolation produced non-finite values"
            raise ValueError(msg)
        return result.reshape(*self.x.shape, *array.shape[1:])

    def sample(self, mesh: pv.UnstructuredGrid) -> ResampledSurface:
        self.validate_mesh(mesh)
        displacement = np.asarray(mesh.point_data.get("Displacement"), dtype=np.float64)
        target = np.asarray(mesh.point_data.get("TargetDisplacement"), dtype=np.float64)
        expected_shape = (mesh.n_points, 3)
        if displacement.shape != expected_shape or target.shape != expected_shape:
            msg = (
                "history frame displacement/target arrays have shapes "
                f"{displacement.shape}/{target.shape}, expected {expected_shape}"
            )
            raise ValueError(msg)
        return ResampledSurface(
            x=self.x,
            z=self.z,
            rest_y=self.interpolate(np.asarray(mesh.points)[:, 1]),
            displacement=self.interpolate(displacement),
            target=self.interpolate(target),
            valid=self.valid,
        )


def common_grid_error_trace(
    case: dict[str, Any],
    history: TemporalHistory,
    plan: CommonGridPlan,
    *,
    initial_mesh: pv.UnstructuredGrid,
) -> list[dict[str, float | int]]:
    label = case_label(case)
    plan.validate_mesh(initial_mesh)
    target = np.asarray(
        initial_mesh.point_data.get("TargetDisplacement"), dtype=np.float64
    )
    if target.shape != (initial_mesh.n_points, 3):
        msg = f"case {label!r} has invalid TargetDisplacement shape {target.shape}"
        raise ValueError(msg)
    reference_top_target = target[plan.top_ids].copy()
    target_grid = plan.interpolate(target)
    target_rms = vector_rms(target_grid[plan.valid])
    if target_rms <= 0.0 or not math.isfinite(target_rms):
        msg = f"case {label!r} common-grid target RMS is invalid: {target_rms}"
        raise ValueError(msg)

    final_step = len(ordered_trace(case)) - 1
    result: list[dict[str, float | int]] = []
    for step in range(final_step + 1):
        mesh = initial_mesh if step == 0 else history.frame(step, deep_copy=False)
        plan.validate_mesh(mesh)
        frame_target = np.asarray(
            mesh.point_data.get("TargetDisplacement"), dtype=np.float64
        )
        displacement = np.asarray(mesh.point_data.get("Displacement"), dtype=np.float64)
        if frame_target.shape != target.shape or displacement.shape != target.shape:
            msg = f"case {label!r} step {step} has invalid displacement arrays"
            raise ValueError(msg)
        if not np.allclose(
            frame_target[plan.top_ids],
            reference_top_target,
            rtol=0.0,
            atol=1.0e-12,
        ):
            msg = f"case {label!r} target changed at temporal step {step}"
            raise ValueError(msg)
        residual_grid = plan.interpolate(displacement - frame_target)
        error_rms = vector_rms(residual_grid[plan.valid])
        trace = trace_row(case, step)
        result.append(
            {
                "step": step,
                "common_grid/error_rms": error_rms,
                "common_grid/target_rms": target_rms,
                "common_grid/error_rms_fraction_of_target": error_rms / target_rms,
                "native/error_rms_fraction_of_target": trace_fraction(case, trace),
            }
        )
        if step % 25 == 0 or step == final_step:
            logger.info(
                "%s common-grid scan step %d/%d: error/target %.6g",
                label,
                step,
                final_step,
                result[-1]["common_grid/error_rms_fraction_of_target"],
            )
    return result


def select_steps(
    case: dict[str, Any],
    common_trace: list[dict[str, float | int]],
    *,
    tau_common: float,
    tolerance: float,
) -> dict[str, float | int]:
    label = case_label(case)
    common_best = min(
        common_trace,
        key=lambda row: (
            float(row["common_grid/error_rms_fraction_of_target"]),
            int(row["step"]),
        ),
    )
    common_best_step = int(common_best["step"])
    eligible = [
        row
        for row in common_trace
        if int(row["step"]) <= common_best_step
        if float(row["common_grid/error_rms_fraction_of_target"])
        <= tau_common + tolerance
    ]
    if not eligible:
        msg = f"case {label!r} cannot reach common-grid tau={tau_common}"
        raise ValueError(msg)
    selected = max(
        eligible,
        key=lambda row: (
            float(row["common_grid/error_rms_fraction_of_target"]),
            -int(row["step"]),
        ),
    )
    first = min(eligible, key=lambda row: int(row["step"]))
    selected_step = int(selected["step"])
    first_step = int(first["step"])
    return {
        "selection/tau_common": tau_common,
        "selection/common_grid_best_step": common_best_step,
        "selection/step": selected_step,
        "selection/common_grid_error_rms_fraction_of_target": float(
            selected["common_grid/error_rms_fraction_of_target"]
        ),
        "selection/native_error_rms_fraction_of_target": float(
            selected["native/error_rms_fraction_of_target"]
        ),
        "selection/first_crossing_step": first_step,
        "selection/first_crossing_common_grid_error_rms_fraction_of_target": float(
            first["common_grid/error_rms_fraction_of_target"]
        ),
        "selection/first_crossing_native_error_rms_fraction_of_target": float(
            first["native/error_rms_fraction_of_target"]
        ),
        "selection/step_delta_from_first_crossing": selected_step - first_step,
    }


def load_selected_meshes(
    cases: list[dict[str, Any]],
    histories: dict[str, TemporalHistory],
    selections: dict[str, dict[str, float | int]],
) -> tuple[list[pv.UnstructuredGrid], list[pv.UnstructuredGrid]]:
    selected_meshes: list[pv.UnstructuredGrid] = []
    first_meshes: list[pv.UnstructuredGrid] = []
    for case in cases:
        label = case_label(case)
        cache: dict[int, pv.UnstructuredGrid] = {}
        for selection_key, target in (
            ("selection/step", selected_meshes),
            ("selection/first_crossing_step", first_meshes),
        ):
            step = int(selections[label][selection_key])
            if step not in cache:
                cache[step] = histories[label].frame(step, deep_copy=True)
            target.append(cache[step])
    return selected_meshes, first_meshes


@dataclass(frozen=True)
class CommonMatching:
    native_best_fractions: list[float]
    bounds: tuple[float, float, float, float]
    plans: dict[str, CommonGridPlan]
    common_traces: dict[str, list[dict[str, float | int]]]
    common_best: dict[str, dict[str, float | int]]
    tau_common: float
    tolerance: float
    tau_anchor_labels: list[str]
    tau_anchor_all_converged: bool
    tau_anchor_any_converged: bool
    selections: dict[str, dict[str, float | int]]
    selected_meshes: list[pv.UnstructuredGrid]
    first_meshes: list[pv.UnstructuredGrid]
    selected_samples: list[ResampledSurface]
    first_samples: list[ResampledSurface]


def prepare_common_matching(cases: list[dict[str, Any]], cfg: Config) -> CommonMatching:
    native_best_fractions = [
        require_scalar(
            case,
            "best/error_rms_fraction_of_target",
            context=f"case {case_label(case)}",
        )
        for case in cases
    ]
    histories = {
        case_label(case): TemporalHistory.open(case, manifest_path=cfg.input_manifest)
        for case in cases
    }
    initial_meshes = [
        histories[case_label(case)].frame(0, deep_copy=True) for case in cases
    ]
    bounds = common_xz_bounds(initial_meshes)
    plans = {
        case_label(case): CommonGridPlan.build(
            mesh, bounds=bounds, grid_size=cfg.grid_size
        )
        for case, mesh in zip(cases, initial_meshes, strict=True)
    }
    common_traces = {
        case_label(case): common_grid_error_trace(
            case,
            histories[case_label(case)],
            plans[case_label(case)],
            initial_mesh=mesh,
        )
        for case, mesh in zip(cases, initial_meshes, strict=True)
    }
    common_best = {
        label: min(
            trace,
            key=lambda row: (
                float(row["common_grid/error_rms_fraction_of_target"]),
                int(row["step"]),
            ),
        )
        for label, trace in common_traces.items()
    }
    tau_common = max(
        float(row["common_grid/error_rms_fraction_of_target"])
        for row in common_best.values()
    )
    tolerance = max(1.0e-12, abs(tau_common) * 1.0e-10)
    tau_anchor_labels = [
        label
        for label, row in common_best.items()
        if math.isclose(
            float(row["common_grid/error_rms_fraction_of_target"]),
            tau_common,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    ]
    convergence = {
        case_label(case): bool(case.get("inverse/converged", False)) for case in cases
    }
    selections = {
        case_label(case): select_steps(
            case,
            common_traces[case_label(case)],
            tau_common=tau_common,
            tolerance=tolerance,
        )
        for case in cases
    }
    selected_meshes, first_meshes = load_selected_meshes(cases, histories, selections)
    selected_samples = [
        plans[case_label(case)].sample(mesh)
        for case, mesh in zip(cases, selected_meshes, strict=True)
    ]
    first_samples = [
        plans[case_label(case)].sample(mesh)
        for case, mesh in zip(cases, first_meshes, strict=True)
    ]
    return CommonMatching(
        native_best_fractions=native_best_fractions,
        bounds=bounds,
        plans=plans,
        common_traces=common_traces,
        common_best=common_best,
        tau_common=tau_common,
        tolerance=tolerance,
        tau_anchor_labels=tau_anchor_labels,
        tau_anchor_all_converged=all(convergence[label] for label in tau_anchor_labels),
        tau_anchor_any_converged=any(convergence[label] for label in tau_anchor_labels),
        selections=selections,
        selected_meshes=selected_meshes,
        first_meshes=first_meshes,
        selected_samples=selected_samples,
        first_samples=first_samples,
    )


def prefixed_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def require_robust_mask(mask: np.ndarray, *, name: str) -> int:
    count = int(mask.sum())
    if count < MIN_ROBUST_MASK_POINTS:
        msg = (
            f"{name} has only {count} valid grid points; expected at least "
            f"{MIN_ROBUST_MASK_POINTS}"
        )
        raise ValueError(msg)
    return count


def masked_highpass_metrics(
    sample: ResampledSurface,
    *,
    smoothing_length: float,
    muscle_bounds: tuple[float, float, float, float, float, float],
) -> dict[str, float | int]:
    dx = float(sample.x[1, 0] - sample.x[0, 0])
    dz = float(sample.z[0, 1] - sample.z[0, 0])
    displacement_y = sample.displacement[..., 1]
    displacement_y_smooth = gaussian_smooth(
        displacement_y,
        dx=dx,
        dz=dz,
        smoothing_length=smoothing_length,
    )
    highpass = displacement_y - displacement_y_smooth

    interior_margin = 3.0 * smoothing_length
    interior = (
        sample.valid
        & (sample.x >= float(sample.x.min()) + interior_margin)
        & (sample.x <= float(sample.x.max()) - interior_margin)
        & (sample.z >= float(sample.z.min()) + interior_margin)
        & (sample.z <= float(sample.z.max()) - interior_margin)
    )
    muscle_footprint = (
        sample.valid
        & (sample.x >= muscle_bounds[0])
        & (sample.x <= muscle_bounds[1])
        & (sample.z >= muscle_bounds[4])
        & (sample.z <= muscle_bounds[5])
    )
    result: dict[str, float | int] = {
        "grid/interior_margin": interior_margin,
        "grid/interior_n": require_robust_mask(interior, name="interior mask"),
        "grid/muscle_footprint_n": require_robust_mask(
            muscle_footprint, name="muscle footprint mask"
        ),
    }
    for name, mask in (
        ("interior", interior),
        ("muscle_footprint", muscle_footprint),
    ):
        displacement_rms = rms(displacement_y[mask])
        highpass_rms = rms(highpass[mask])
        if displacement_rms <= 0.0 or not math.isfinite(displacement_rms):
            msg = f"{name} vertical displacement RMS is invalid: {displacement_rms}"
            raise ValueError(msg)
        result[f"grid/{name}_displacement_y_rms"] = displacement_rms
        result[f"grid/{name}_displacement_y_highpass_rms"] = highpass_rms
        result[f"grid/{name}_displacement_y_highpass_over_rms"] = (
            highpass_rms / displacement_rms
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def effect_value(comparison: float, baseline: float) -> dict[str, float]:
    relative = safe_ratio(comparison, baseline) - 1.0
    return {
        "baseline": baseline,
        "comparison": comparison,
        "absolute_change": comparison - baseline,
        "relative_change": relative,
        "percent_change": 100.0 * relative,
    }


def effect_sizes(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    lookup = {str(row["label"]): row for row in rows}
    result: dict[str, Any] = {}
    for comparison, baseline in (
        ("current", "thin"),
        ("thick", "thin"),
        ("thick", "current"),
    ):
        pair = f"{comparison}_vs_{baseline}"
        result[pair] = {}
        for metric in CORE_ROUGHNESS_METRICS:
            key = f"{prefix}{metric}"
            result[pair][metric] = effect_value(
                float(lookup[comparison][key]), float(lookup[baseline][key])
            )
    return result


def metric_sign(value: float, *, tolerance: float = 1.0e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def path_sensitivity(
    rows: list[dict[str, Any]],
    *,
    selected_effects: dict[str, Any],
    first_effects: dict[str, Any],
) -> dict[str, Any]:
    per_case: dict[str, Any] = {}
    maximum_step_delta = 0
    maximum_relative_metric_change = 0.0
    for row in rows:
        label = str(row["label"])
        step_delta = int(row["selection/step_delta_from_first_crossing"])
        maximum_step_delta = max(maximum_step_delta, abs(step_delta))
        metric_changes: dict[str, Any] = {}
        for metric in CORE_ROUGHNESS_METRICS:
            selected = float(row[metric])
            first = float(row[f"first_crossing/{metric}"])
            change = effect_value(selected, first)
            metric_changes[metric] = change
            if math.isfinite(change["relative_change"]):
                maximum_relative_metric_change = max(
                    maximum_relative_metric_change,
                    abs(change["relative_change"]),
                )
        per_case[label] = {
            "step_delta": step_delta,
            "metrics_selected_vs_first_crossing": metric_changes,
        }

    effect_sign_consistency: dict[str, Any] = {}
    all_effect_signs_consistent = True
    for pair, selected_pair in selected_effects.items():
        effect_sign_consistency[pair] = {}
        for metric, selected_effect in selected_pair.items():
            first_effect = first_effects[pair][metric]
            selected_sign = metric_sign(float(selected_effect["absolute_change"]))
            first_sign = metric_sign(float(first_effect["absolute_change"]))
            consistent = selected_sign == first_sign
            effect_sign_consistency[pair][metric] = {
                "closest_from_below_sign": selected_sign,
                "first_crossing_sign": first_sign,
                "consistent": consistent,
            }
            all_effect_signs_consistent &= consistent
    return {
        "any_step_difference": maximum_step_delta > 0,
        "maximum_step_delta": maximum_step_delta,
        "maximum_absolute_relative_metric_change": maximum_relative_metric_change,
        "all_effect_signs_consistent": all_effect_signs_consistent,
        "effect_sign_consistency": effect_sign_consistency,
        "cases": per_case,
    }


def plot_metric_comparison(
    path: Path, rows: list[dict[str, Any]], *, tau_common: float
) -> None:
    thickness = np.asarray(
        [float(row["fat_thickness/min"]) for row in rows], dtype=np.float64
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)

    axis = axes[0, 0]
    axis.axhline(tau_common, color="black", linestyle=":", label="common-grid tau")
    axis.plot(
        thickness,
        [
            float(row["selection/common_grid_error_rms_fraction_of_target"])
            for row in rows
        ],
        marker="o",
        label="common-grid closest-from-below",
    )
    axis.plot(
        thickness,
        [
            float(
                row["selection/first_crossing_common_grid_error_rms_fraction_of_target"]
            )
            for row in rows
        ],
        marker="x",
        linestyle="--",
        label="common-grid first-crossing",
    )
    axis.plot(
        thickness,
        [float(row["selection/native_error_rms_fraction_of_target"]) for row in rows],
        marker="s",
        linestyle=":",
        label="native diagnostic",
    )
    axis.set_title("matched target fidelity")
    axis.set_ylabel("target error / target RMS")
    axis.legend(fontsize="small")

    panels = (
        (
            axes[0, 1],
            "grid/displacement_y_highpass_rms",
            "vertical high-pass RMS",
        ),
        (
            axes[1, 0],
            "grid/displacement_y_highpass_over_rms",
            "normalized vertical high-pass",
        ),
        (
            axes[1, 1],
            "grid/displacement_y_laplacian_over_rms",
            "normalized smoothed Laplacian",
        ),
    )
    for axis, key, title in panels:
        axis.plot(
            thickness,
            [float(row[key]) for row in rows],
            marker="o",
            label="closest-from-below",
        )
        axis.plot(
            thickness,
            [float(row[f"first_crossing/{key}"]) for row in rows],
            marker="x",
            linestyle="--",
            label="first-crossing",
        )
        axis.set_title(title)
        axis.set_ylabel(key.removeprefix("grid/"))
        axis.legend(fontsize="small")
    for axis in axes.ravel():
        axis.set_xlabel("minimum fat thickness")
        axis.grid(alpha=0.3)
        for x, row in zip(thickness, rows, strict=True):
            axis.annotate(
                str(row["label"]),
                (x, axis.lines[-1].get_ydata()[list(thickness).index(x)]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize="small",
            )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def symmetric_limit(fields: list[np.ndarray]) -> float:
    finite_values = [field[np.isfinite(field)] for field in fields]
    limit = max(
        float(np.max(np.abs(values))) for values in finite_values if values.size
    )
    return limit if limit > 0.0 else 1.0


def masked_field(sample: ResampledSurface, field: np.ndarray) -> np.ndarray:
    return np.where(sample.valid, field, np.nan)


def highpass_field(sample: ResampledSurface, *, smoothing_length: float) -> np.ndarray:
    dx = float(sample.x[1, 0] - sample.x[0, 0])
    dz = float(sample.z[0, 1] - sample.z[0, 0])
    smooth = gaussian_smooth(
        sample.displacement[..., 1],
        dx=dx,
        dz=dz,
        smoothing_length=smoothing_length,
    )
    return masked_field(sample, sample.displacement[..., 1] - smooth)


def plot_fields(
    path: Path,
    rows: list[dict[str, Any]],
    selected_samples: list[ResampledSurface],
    first_samples: list[ResampledSurface],
    *,
    smoothing_length: float,
) -> None:
    displacement_fields = [
        masked_field(sample, sample.displacement[..., 1]) for sample in selected_samples
    ]
    selected_highpass = [
        highpass_field(sample, smoothing_length=smoothing_length)
        for sample in selected_samples
    ]
    first_highpass = [
        highpass_field(sample, smoothing_length=smoothing_length)
        for sample in first_samples
    ]
    residual_fields = [
        masked_field(sample, sample.residual[..., 1]) for sample in selected_samples
    ]
    displacement_limit = symmetric_limit(displacement_fields)
    highpass_limit = symmetric_limit([*selected_highpass, *first_highpass])
    residual_limit = symmetric_limit(residual_fields)

    fig, axes = plt.subplots(
        4,
        len(rows),
        figsize=(4.2 * len(rows), 13.2),
        squeeze=False,
        constrained_layout=True,
    )
    images: list[Any] = [None] * 4
    for column, (row, sample) in enumerate(zip(rows, selected_samples, strict=True)):
        extent = (
            float(sample.z.min()),
            float(sample.z.max()),
            float(sample.x.min()),
            float(sample.x.max()),
        )
        fields = (
            (displacement_fields[column], displacement_limit),
            (selected_highpass[column], highpass_limit),
            (first_highpass[column], highpass_limit),
            (residual_fields[column], residual_limit),
        )
        for row_index, (field, limit) in enumerate(fields):
            images[row_index] = axes[row_index, column].imshow(
                field,
                origin="lower",
                extent=extent,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                aspect="equal",
            )
            axes[row_index, column].set_ylabel("x")
        axes[0, column].set_title(
            f"{row['label']}\nclosest step={int(row['selection/step'])}, "
            f"first={int(row['selection/first_crossing_step'])}"
        )
        axes[-1, column].set_xlabel("z")
    colorbar_labels = (
        "closest vertical displacement",
        f"closest vertical high-pass (length={smoothing_length:g})",
        f"first-crossing vertical high-pass (length={smoothing_length:g})",
        "closest vertical residual",
    )
    for row_index, (image, label) in enumerate(
        zip(images, colorbar_labels, strict=True)
    ):
        fig.colorbar(
            image,
            ax=axes[row_index, :].tolist(),
            label=label,
            shrink=0.85,
        )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(cfg: Config) -> None:
    validate_config(cfg)
    for path in (
        cfg.output_csv,
        cfg.output_json,
        cfg.output_metric_plot,
        cfg.output_field_plot,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(cfg.input_manifest.read_text(encoding="utf-8"))
    cases = validate_manifest(manifest, manifest_path=cfg.input_manifest)
    matching = prepare_common_matching(cases, cfg)

    rows: list[dict[str, Any]] = []
    for index, (
        case,
        selected_mesh,
        first_mesh,
        selected_sample,
        first_sample,
    ) in enumerate(
        zip(
            cases,
            matching.selected_meshes,
            matching.first_meshes,
            matching.selected_samples,
            matching.first_samples,
            strict=True,
        )
    ):
        label = case_label(case)
        selected_muscle_bounds = toy.mesh_muscle_bounds(selected_mesh)
        first_muscle_bounds = toy.mesh_muscle_bounds(first_mesh)
        selected_metrics = {
            **surface_metrics(
                selected_sample,
                high_frequency_cutoff_cycles=cfg.high_frequency_cutoff_cycles,
                laplacian_smoothing_length=cfg.laplacian_smoothing_length,
                muscle_bounds=selected_muscle_bounds,
            ),
            **masked_highpass_metrics(
                selected_sample,
                smoothing_length=cfg.laplacian_smoothing_length,
                muscle_bounds=selected_muscle_bounds,
            ),
        }
        first_metrics = {
            **surface_metrics(
                first_sample,
                high_frequency_cutoff_cycles=cfg.high_frequency_cutoff_cycles,
                laplacian_smoothing_length=cfg.laplacian_smoothing_length,
                muscle_bounds=first_muscle_bounds,
            ),
            **masked_highpass_metrics(
                first_sample,
                smoothing_length=cfg.laplacian_smoothing_length,
                muscle_bounds=first_muscle_bounds,
            ),
        }
        selected_step = int(matching.selections[label]["selection/step"])
        first_step = int(matching.selections[label]["selection/first_crossing_step"])
        selected_trace = trace_row(case, selected_step)
        first_trace = trace_row(case, first_step)
        for name, actual, expected in (
            (
                "closest-from-below",
                float(selected_metrics["grid/error_rms_fraction_of_target"]),
                float(
                    matching.selections[label][
                        "selection/common_grid_error_rms_fraction_of_target"
                    ]
                ),
            ),
            (
                "first-crossing",
                float(first_metrics["grid/error_rms_fraction_of_target"]),
                float(
                    matching.selections[label][
                        "selection/first_crossing_common_grid_error_rms_fraction_of_target"
                    ]
                ),
            ),
        ):
            if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
                msg = (
                    f"case {label!r} {name} full metrics common-grid fidelity "
                    f"{actual} disagrees with scan {expected}"
                )
                raise ValueError(msg)
        common_best_row = matching.common_best[label]
        row: dict[str, Any] = {
            "label": label,
            "fat_thickness/min": case.get("fat_thickness/min"),
            "fat_thickness/center": case.get("fat_thickness/center"),
            "inverse/converged": case.get("inverse/converged"),
            "inverse/stop_reason": case.get("inverse/stop_reason"),
            "best/error_rms_fraction_of_target": case.get(
                "best/error_rms_fraction_of_target"
            ),
            "common_grid_scan/best_step": common_best_row["step"],
            "common_grid_scan/best_error_rms_fraction_of_target": common_best_row[
                "common_grid/error_rms_fraction_of_target"
            ],
            "common_grid_scan/native_error_rms_fraction_at_common_best": (
                common_best_row["native/error_rms_fraction_of_target"]
            ),
            **matching.selections[label],
            "selection/activation_inv_rms": selected_trace["activation_inv/rms"],
            "selection/activation_inv_max_abs": selected_trace[
                "activation_inv/max_abs"
            ],
            "selection/first_crossing_activation_inv_rms": first_trace[
                "activation_inv/rms"
            ],
            "selection/first_crossing_activation_inv_max_abs": first_trace[
                "activation_inv/max_abs"
            ],
            **selected_metrics,
            **prefixed_metrics("first_crossing/", first_metrics),
        }
        rows.append(row)
        cherries.set_step(index)
        cherries.log_metrics(
            {
                f"{label}/selection_step": selected_step,
                f"{label}/first_crossing_step": first_step,
                f"{label}/native_error_fraction": row[
                    "selection/native_error_rms_fraction_of_target"
                ],
                f"{label}/common_grid_error_fraction": row[
                    "selection/common_grid_error_rms_fraction_of_target"
                ],
                **{
                    f"{label}/{key.removeprefix('grid/')}": value
                    for key, value in finite_scalar_metrics(selected_metrics).items()
                },
            }
        )
        logger.info(
            "%s matched at step %d (native %.6g, common-grid %.6g); "
            "first crossing step %d",
            label,
            selected_step,
            row["selection/native_error_rms_fraction_of_target"],
            row["selection/common_grid_error_rms_fraction_of_target"],
            first_step,
        )

    selected_effects = effect_sizes(rows, prefix="")
    first_effects = effect_sizes(rows, prefix="first_crossing/")
    sensitivity = path_sensitivity(
        rows,
        selected_effects=selected_effects,
        first_effects=first_effects,
    )
    common_fidelities = np.asarray(
        [
            float(row["selection/common_grid_error_rms_fraction_of_target"])
            for row in rows
        ],
        dtype=np.float64,
    )
    common_fidelity_spread = float(np.ptp(common_fidelities))
    common_fidelity_gate_passed = (
        common_fidelity_spread <= cfg.max_common_fidelity_spread + matching.tolerance
    )
    write_csv(cfg.output_csv, rows)
    payload = {
        "schema_version": 1,
        "kind": "fat-thickness-matched-fidelity-analysis",
        "interpretation_scope": (
            "fixed-budget-matched-trajectory-not-converged-upper-bound"
        ),
        "source_manifest": str(cfg.input_manifest),
        "source_complete": manifest.get("complete"),
        "source_convergence_failures": manifest.get("convergence_failures", []),
        "validated_setup": {
            "fresh_optimizer_per_case": True,
            "initial_activation_inv": [0.0] * 6,
            "auxiliary_losses_enabled": False,
            "skin_energy_enabled": False,
            "activation_mode": "per-tet-6dof",
        },
        "selection": {
            "primary_rule": "common-grid-closest-from-below-before-common-best",
            "sensitivity_rule": "common-grid-first-crossing-before-common-best",
            "tau_definition": (
                "maximum across cases of the minimum common-grid target-error "
                "fraction observed over each complete temporal trace"
            ),
            "tau_common": matching.tau_common,
            "tolerance": matching.tolerance,
            "tau_anchor_labels": matching.tau_anchor_labels,
            "tau_anchor_all_converged": matching.tau_anchor_all_converged,
            "tau_anchor_any_converged": matching.tau_anchor_any_converged,
            "native_best_error_rms_fractions_of_target_diagnostic": {
                case_label(case): fraction
                for case, fraction in zip(
                    cases, matching.native_best_fractions, strict=True
                )
            },
            "common_grid_best": dict(matching.common_best),
            "common_grid_trace_frame_counts": {
                label: len(trace) for label, trace in matching.common_traces.items()
            },
        },
        "grid_size": cfg.grid_size,
        "common_xz_bounds": list(matching.bounds),
        "high_frequency_cutoff_cycles_per_unit": (cfg.high_frequency_cutoff_cycles),
        "laplacian_smoothing_length": cfg.laplacian_smoothing_length,
        "robust_masks": {
            "minimum_valid_points": MIN_ROBUST_MASK_POINTS,
            "interior": {
                "definition": (
                    "common-grid valid points at least three smoothing lengths "
                    "inside every x-z domain boundary"
                ),
                "margin": 3.0 * cfg.laplacian_smoothing_length,
            },
            "muscle_footprint": {
                "definition": (
                    "common-grid valid points inside each mesh MuscleBounds x-z "
                    "footprint"
                )
            },
        },
        "common_grid_fidelity": {
            "minimum": float(common_fidelities.min()),
            "maximum": float(common_fidelities.max()),
            "spread": common_fidelity_spread,
            "maximum_allowed_spread": cfg.max_common_fidelity_spread,
            "gate_passed": common_fidelity_gate_passed,
        },
        "common_grid_error_traces": matching.common_traces,
        "cases": rows,
        "effect_sizes": {
            "closest_from_below": selected_effects,
            "first_crossing": first_effects,
        },
        "path_sensitivity": sensitivity,
    }
    cfg.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_metric_comparison(cfg.output_metric_plot, rows, tau_common=matching.tau_common)
    plot_fields(
        cfg.output_field_plot,
        rows,
        matching.selected_samples,
        matching.first_samples,
        smoothing_length=cfg.laplacian_smoothing_length,
    )
    for path in (
        cfg.output_csv,
        cfg.output_json,
        cfg.output_metric_plot,
        cfg.output_field_plot,
    ):
        logger.info("Wrote %s", path)
    if not common_fidelity_gate_passed:
        msg = (
            "selected common-grid target-error spread "
            f"{common_fidelity_spread:.6g} exceeds configured maximum "
            f"{cfg.max_common_fidelity_spread:.6g}; selected steps are "
            + ", ".join(f"{row['label']}={int(row['selection/step'])}" for row in rows)
        )
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(main)
