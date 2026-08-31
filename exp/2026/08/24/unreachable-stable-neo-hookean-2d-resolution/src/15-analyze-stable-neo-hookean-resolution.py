# Copyright (c) 2026 liblaf
from __future__ import annotations

# ruff: noqa: EM101, EM102, PLR0915, TRY003
import csv
import importlib.util
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import scipy.ndimage as ndi
import scipy.sparse.linalg as spla

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    input_summary: Path = Path("data/10-stable-neo-hookean-resolution/summary.json")
    output_dir: Path = cherries.output("15-resolution-analysis", mkdir=True)
    common_points: int = 1921
    tangent_relative_tolerance: float = 1.0e-10
    displacement_resolution: float = 1.0e-4
    activation_scale: float = 1.0


@dataclass(frozen=True)
class Case:
    nx: int
    ny: int
    variant: str
    case_dir: Path
    payload: dict[str, Any]

    @property
    def name(self) -> str:
        return f"{self.nx}x{self.ny}-{self.variant}"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_trace(case: Case) -> list[dict[str, float]]:
    with (case.case_dir / "trace.csv").open(newline="", encoding="utf-8") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if not rows or [int(row["step"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"non-consecutive inverse trace for {case.name}")
    return rows


def verified_pre_inversion_rows(
    case: Case, trace: list[dict[str, float]]
) -> list[dict[str, float]]:
    first_invalid = case.payload["first_invalid_step"]
    rows = [
        row
        for row in trace
        if (first_invalid is None or row["step"] < first_invalid)
        and row["verified_admissible"] == 1.0
    ]
    if not rows:
        raise ValueError(f"no verified pre-inversion evaluations for {case.name}")
    return rows


def matched_data_loss_rows(
    cases: list[Case], traces: dict[str, list[dict[str, float]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in sorted({case.variant for case in cases}):
        family = sorted(
            (case for case in cases if case.variant == variant),
            key=lambda item: item.nx,
        )
        verified = {
            case.name: verified_pre_inversion_rows(case, traces[case.name])
            for case in family
        }
        common_best = max(
            min(row["objective_data"] for row in verified[case.name]) for case in family
        )
        common_initial = min(
            max(row["objective_data"] for row in verified[case.name]) for case in family
        )
        if common_best > common_initial:
            raise ValueError(f"no common verified loss interval for {variant}")
        for progress in (0.25, 0.5, 0.75, 1.0):
            target = common_initial - progress * (common_initial - common_best)
            for case in family:
                row = min(
                    verified[case.name],
                    key=lambda item: abs(item["objective_data"] - target),
                )
                output.append(
                    {
                        "variant": variant,
                        "resolution": f"{case.nx}x{case.ny}",
                        "nx": case.nx,
                        "ny": case.ny,
                        "common_progress_fraction": progress,
                        "target_data_loss": target,
                        "selected_step": int(row["step"]),
                        "actual_data_loss": row["objective_data"],
                        "relative_loss_mismatch": abs(row["objective_data"] - target)
                        / max(target, np.finfo(float).tiny),
                        "top_highpass_rms_width_0p02": row["top_highpass_rms"],
                        "top_error_rms": row["top_error_rms"],
                        "activation_l2_rms": row["activation_l2_rms"],
                        "activation_neighbor_jump_rms": row[
                            "activation_neighbor_jump_rms"
                        ],
                        "min_det_f": row["min_det_f"],
                        "min_det_g": row["min_det_g"],
                        "min_det_ainv": row["min_det_ainv"],
                        "equilibrium_residual_rms": row["equilibrium_residual_rms"],
                    }
                )
    return output


def evolution_diagnostic_rows(
    cases: list[Case], traces: dict[str, list[dict[str, float]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in cases:
        trace = traces[case.name]
        prefix = case.payload["best_admissible_prefix"]
        global_best = case.payload["best"]
        first_invalid_step = case.payload["first_invalid_step"]
        invalid = (
            trace[int(first_invalid_step)] if first_invalid_step is not None else None
        )
        output.append(
            {
                "case": case.name,
                "variant": case.variant,
                "nx": case.nx,
                "ny": case.ny,
                "evaluations": len(trace),
                "verified_comparison_step": case.payload["best_admissible_prefix_step"],
                "first_orientation_invalid_step": first_invalid_step,
                "non_equilibrated_evaluations": sum(
                    row["numerically_equilibrated"] == 0.0 for row in trace
                ),
                "comparison_data_loss": prefix["objective_data"],
                "comparison_highpass_rms": prefix["top_highpass_rms"],
                "comparison_min_det_f": prefix["min_det_f"],
                "comparison_min_det_g": prefix["min_det_g"],
                "comparison_min_det_ainv": prefix["min_det_ainv"],
                "first_invalid_data_loss": (
                    invalid["objective_data"] if invalid is not None else None
                ),
                "first_invalid_highpass_rms": (
                    invalid["top_highpass_rms"] if invalid is not None else None
                ),
                "global_best_data_loss": global_best["objective_data"],
                "global_best_highpass_rms": global_best["top_highpass_rms"],
                "global_best_is_orientation_preserving": case.payload[
                    "global_best_is_orientation_preserving"
                ],
                "global_best_loss_reduction_from_comparison": (
                    prefix["objective_data"] - global_best["objective_data"]
                ),
            }
        )
    return output


def load_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "stable_neo_hookean_resolution_runner", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import numerical runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_cases(summary_path: Path, summary: dict[str, Any]) -> list[Case]:
    cases: list[Case] = []
    root = summary_path.parent
    for payload in summary["cases"]:
        nx, ny = (int(value) for value in payload["resolution"])
        variant = str(payload["variant"])
        case_dir = root / f"{nx}x{ny}-{variant}"
        required = [
            case_dir / payload["paths"]["profile"],
            case_dir / payload["paths"]["spectrum"],
            case_dir / payload["paths"]["best_admissible_prefix_profile"],
            case_dir / payload["paths"]["best_admissible_prefix_spectrum"],
            case_dir / "best-state.npz",
            case_dir / payload["paths"]["best_admissible_prefix_state"],
        ]
        if any(not path.is_file() for path in required):
            missing = [str(path) for path in required if not path.is_file()]
            raise FileNotFoundError(
                f"case {nx}x{ny}-{variant} is incomplete: {missing}"
            )
        cases.append(Case(nx, ny, variant, case_dir, payload))
    return sorted(cases, key=lambda item: (item.variant, item.nx, item.ny))


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def interpolate_profile(
    case: Case, x_common: np.ndarray, profile_key: str
) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(
        case.case_dir / case.payload["paths"][profile_key], delimiter=",", names=True
    )
    x = np.asarray(table["x"], dtype=np.float64)
    ux = np.asarray(table["ux"], dtype=np.float64)
    uy = np.asarray(table["uy"], dtype=np.float64)
    if x.ndim != 1 or x.size < 3 or np.any(np.diff(x) <= 0.0):
        raise ValueError(f"invalid top profile for {case.name}")
    if x_common[0] < x[0] - 1.0e-12 or x_common[-1] > x[-1] + 1.0e-12:
        raise ValueError(f"common grid escapes native profile for {case.name}")
    return np.interp(x_common, x, ux), np.interp(x_common, x, uy)


def fixed_scale_metrics(
    x: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    *,
    target_y: float,
    filter_width: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    dx = float(x[1] - x[0])
    smooth = ndi.gaussian_filter1d(uy, sigma=filter_width / dx, mode="nearest")
    highpass = uy - smooth
    slope = np.gradient(uy, dx)
    curvature = np.gradient(slope, dx)
    centered = uy - np.mean(uy)
    frequency = np.fft.rfftfreq(x.size, d=dx)
    power = np.square(np.abs(np.fft.rfft(centered))) / x.size**2
    total_power = float(np.sum(power[1:]))

    def band(lo: float, hi: float) -> float:
        mask = (frequency >= lo) & (frequency < hi)
        return float(np.sum(power[mask]))

    vector_error = np.sqrt(np.square(ux) + np.square(uy - target_y))
    metrics = {
        "target_vector_rms": rms(vector_error),
        "target_vertical_rms": rms(uy - target_y),
        "top_uy_mean": float(np.mean(uy)),
        "top_uy_range": float(np.ptp(uy)),
        "highpass_rms_width_0p02": rms(highpass),
        "slope_rms": rms(slope),
        "curvature_rms": rms(curvature),
        "psd_total_non_dc": total_power,
        "psd_band_1_4": band(1.0, 4.0),
        "psd_band_4_12": band(4.0, 12.0),
        "psd_band_12_24": band(12.0, 24.0),
        "psd_fraction_12_24": band(12.0, 24.0) / max(total_power, np.finfo(float).tiny),
    }
    return metrics, highpass, frequency, power


def add_common_grid_matched_metrics(
    rows: list[dict[str, Any]], cases: list[Case], x_common: np.ndarray, target_y: float
) -> None:
    by_key = {(case.nx, case.ny, case.variant): case for case in cases}
    for row in rows:
        case = by_key[(int(row["nx"]), int(row["ny"]), str(row["variant"]))]
        step = int(row["selected_step"])
        grid = pv.read(case.case_dir / "frames" / f"step-{step:04d}.vtu")
        points = np.asarray(grid.points)
        displacement = np.asarray(grid.point_data["Displacement"])
        mask = (
            np.isclose(points[:, 1], 0.1) & (points[:, 0] > 0.0) & (points[:, 0] < 1.0)
        )
        order = np.argsort(points[mask, 0])
        x = points[mask, 0][order]
        ux = displacement[mask, 0][order]
        uy = displacement[mask, 1][order]
        ux_common = np.interp(x_common, x, ux)
        uy_common = np.interp(x_common, x, uy)
        metrics, *_ = fixed_scale_metrics(
            x_common,
            ux_common,
            uy_common,
            target_y=target_y,
            filter_width=0.02,
        )
        row["common_grid_target_vector_rms"] = metrics["target_vector_rms"]
        row["common_grid_highpass_rms_width_0p02"] = metrics["highpass_rms_width_0p02"]
        row["common_grid_psd_band_12_24"] = metrics["psd_band_12_24"]


def profile_resolution_comparisons(
    cases: list[Case],
    profiles: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    state: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in sorted({case.variant for case in cases}):
        family = sorted(
            (case for case in cases if case.variant == variant), key=lambda c: c.nx
        )
        for coarse, fine in itertools.pairwise(family):
            coarse_uy = profiles[(state, coarse.name)][1]
            fine_uy = profiles[(state, fine.name)][1]
            centered_coarse = coarse_uy - np.mean(coarse_uy)
            centered_fine = fine_uy - np.mean(fine_uy)
            denominator = float(
                np.linalg.norm(centered_coarse) * np.linalg.norm(centered_fine)
            )
            correlation = (
                float(np.dot(centered_coarse, centered_fine) / denominator)
                if denominator > np.finfo(float).tiny
                else 1.0
            )
            difference = fine_uy - coarse_uy
            rows.append(
                {
                    "state": state,
                    "variant": variant,
                    "coarse": coarse.name,
                    "fine": fine.name,
                    "profile_l2_difference": rms(difference),
                    "profile_difference_fraction_of_fine": rms(difference)
                    / max(rms(fine_uy), np.finfo(float).tiny),
                    "profile_correlation": correlation,
                }
            )
    return rows


def control_subgrid_rows(
    cases: list[Case], runner: Any, poisson: float, state: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    free = sorted(
        (case for case in cases if case.variant == "free"), key=lambda item: item.nx
    )
    for coarse, fine in itertools.pairwise(free):
        fine_mesh = runner.build_mesh(fine.nx, fine.ny, poisson)
        state_path = (
            fine.case_dir / "best-state.npz"
            if state == "global_best"
            else fine.case_dir / fine.payload["paths"]["best_admissible_prefix_state"]
        )
        fine_controls = np.asarray(np.load(state_path)["controls"])
        fine_element = fine_controls.reshape((-1, 3))
        if fine_element.shape[0] != fine_mesh.muscle_elements.size:
            raise ValueError(f"free controls are not per triangle in {fine.name}")
        parent = runner.build_control_map(fine_mesh, "tied", coarse.nx, coarse.ny)
        restricted = np.zeros((parent.n_groups, 3), dtype=np.float64)
        counts = np.zeros(parent.n_groups, dtype=np.float64)
        np.add.at(restricted, parent.element_group, fine_element)
        np.add.at(counts, parent.element_group, 1.0)
        restricted /= counts[:, None]
        prolonged = restricted[parent.element_group]
        residual = fine_element - prolonged
        rows.append(
            {
                "state": state,
                "coarse_partition": coarse.name,
                "fine_solution": fine.name,
                "fine_activation_rms": rms(fine_element),
                "subgrid_activation_rms": rms(residual),
                "subgrid_activation_fraction": rms(residual)
                / max(rms(fine_element), np.finfo(float).tiny),
            }
        )
    return rows


def selected_activation_eigen_rows(
    cases: list[Case], runner: Any, poisson: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        mesh = runner.build_mesh(case.nx, case.ny, poisson)
        cmap = runner.build_control_map(mesh, case.variant, 50, 5)
        state = np.load(
            case.case_dir / case.payload["paths"]["best_admissible_prefix_state"]
        )
        controls = np.asarray(state["controls"], dtype=np.float64)
        activation = controls.reshape((-1, 3))[cmap.element_group]
        active_inv = np.zeros((activation.shape[0], 2, 2), dtype=np.float64)
        active_inv[:, 0, 0] = 1.0 + activation[:, 0]
        active_inv[:, 1, 1] = 1.0 + activation[:, 1]
        active_inv[:, 0, 1] = activation[:, 2]
        active_inv[:, 1, 0] = activation[:, 2]
        eigenvalues = np.linalg.eigvalsh(active_inv)
        rows.append(
            {
                "case": case.name,
                "minimum_eigenvalue_ainv": float(np.min(eigenvalues)),
                "maximum_eigenvalue_ainv": float(np.max(eigenvalues)),
                "nonpositive_eigenvalues": int(np.count_nonzero(eigenvalues <= 0.0)),
                "positive_definite_selected_state": bool(np.all(eigenvalues > 0.0)),
            }
        )
    return rows


def tangent_certificate(
    runner: Any,
    nx: int,
    ny: int,
    *,
    poisson: float,
    target_y: float,
    relative_tolerance: float,
    displacement_resolution: float,
    activation_scale: float,
) -> tuple[dict[str, Any], np.ndarray]:
    mesh = runner.build_mesh(nx, ny, poisson)
    cmap = runner.build_control_map(mesh, "free", nx, ny)
    controls = np.zeros(cmap.n_controls, dtype=np.float64)
    forward = runner.solve_forward(
        mesh, controls, cmap, np.zeros(mesh.n_free), 1.0e-11, 20
    )
    _, _, hessian, mixed, *_ = runner.constitutive(
        mesh,
        forward.u_free,
        controls,
        cmap,
        need_hessian=True,
        need_mixed=True,
    )
    factor = spla.splu(hessian)
    state_sensitivity = -factor.solve(mixed.toarray())
    top_rows: list[int] = []
    for node in mesh.top_nodes:
        top_rows.extend((mesh.free_lookup[2 * node], mesh.free_lookup[2 * node + 1]))
    response = state_sensitivity[np.asarray(top_rows, dtype=np.int64)]
    singular = np.linalg.svd(response, compute_uv=False)
    tolerance = relative_tolerance * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    physical_rank = int(
        np.count_nonzero(singular * activation_scale > displacement_resolution)
    )
    target = np.tile(np.array([0.0, target_y]), mesh.top_nodes.size)
    projection = np.linalg.lstsq(response, target, rcond=relative_tolerance)[0]
    residual = response @ projection - target
    certificate = {
        "resolution": [nx, ny],
        "n_controls": int(response.shape[1]),
        "n_nonzero_observed_components": int(response.shape[0]),
        "dimension_guaranteed_nullity": int(
            max(response.shape[1] - response.shape[0], 0)
        ),
        "relative_rank_tolerance": relative_tolerance,
        "numerical_rank": rank,
        "numerical_nullity": int(response.shape[1] - rank),
        "physical_displacement_resolution": displacement_resolution,
        "physical_activation_scale": activation_scale,
        "physically_resolved_rank": physical_rank,
        "largest_singular_value": float(singular[0]),
        "smallest_singular_value": float(singular[-1]),
        "projection_residual_rms": rms(residual),
        "projection_residual_fraction_of_target": rms(residual)
        / max(rms(target), np.finfo(float).tiny),
        "scope": "local tangent at zero activation; not a global nonlinear reachability proof",
    }
    return certificate, singular


def implicit_gradient_check(
    runner: Any, poisson: float, target_y: float
) -> dict[str, Any]:
    mesh = runner.build_mesh(50, 5, poisson)
    cmap = runner.build_control_map(mesh, "free", 50, 5)

    class LocalConfig:
        forward_tolerance = 1.0e-11
        forward_max_iterations = 40
        regularization_weight = 0.0

    local = LocalConfig()
    local.target_y = target_y
    rng = np.random.default_rng(20260824)
    controls = 0.01 * rng.standard_normal(cmap.n_controls)
    direction = rng.standard_normal(cmap.n_controls)
    direction /= np.linalg.norm(direction)
    forward, loss, regularizer, gradient = runner.inverse_evaluation(
        mesh, controls, cmap, np.zeros(mesh.n_free), local
    )
    eps = 2.0e-5
    losses: list[float] = []
    for sign in (-1.0, 1.0):
        perturbed = controls + sign * eps * direction
        state = runner.solve_forward(
            mesh,
            perturbed,
            cmap,
            forward.u_free,
            local.forward_tolerance,
            local.forward_max_iterations,
        )
        value, _ = runner.target_loss_and_gradient(mesh, state.u_free, target_y)
        losses.append(value)
    finite_difference = (losses[1] - losses[0]) / (2.0 * eps)
    analytic = float(np.dot(gradient, direction))
    relative = abs(analytic - finite_difference) / max(abs(finite_difference), 1.0e-14)
    return {
        "resolution": [50, 5],
        "epsilon": eps,
        "base_loss": loss + regularizer,
        "analytic_directional_derivative": analytic,
        "finite_difference_directional_derivative": finite_difference,
        "relative_error": relative,
        "passed": bool(relative < 5.0e-4),
    }


def main(cfg: Config) -> None:  # noqa: C901
    summary_path = cfg.input_summary.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("design")
        != "exact-plane-strain-stable-neo-hookean-active-resolution-study"
    ):
        raise ValueError("input is not the Stable Neo-Hookean resolution study")
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output {cfg.output_dir}")
    if cfg.common_points < 257 or cfg.common_points % 2 == 0:
        raise ValueError("common_points must be an odd integer at least 257")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(summary_path, summary)
    traces = {case.name: load_trace(case) for case in cases}
    matched_rows = matched_data_loss_rows(cases, traces)
    evolution_rows = evolution_diagnostic_rows(cases, traces)
    write_rows(cfg.output_dir / "evolution-diagnostics.csv", evolution_rows)
    target_y = float(summary["geometry"]["target"][1])
    poisson_values = {
        float(material["poisson_ratio"]) for material in summary["materials"].values()
    }
    if len(poisson_values) != 1:
        raise ValueError("analysis expects one common Poisson ratio")
    poisson = poisson_values.pop()
    native_min = max(
        float(
            np.genfromtxt(case.case_dir / "top-profile.csv", delimiter=",", names=True)[
                "x"
            ][0]
        )
        for case in cases
    )
    native_max = min(
        float(
            np.genfromtxt(case.case_dir / "top-profile.csv", delimiter=",", names=True)[
                "x"
            ][-1]
        )
        for case in cases
    )
    x_common = np.linspace(native_min, native_max, cfg.common_points)
    add_common_grid_matched_metrics(matched_rows, cases, x_common, target_y)
    write_rows(cfg.output_dir / "matched-data-loss-evolution.csv", matched_rows)

    state_profiles = {
        "global_best": "profile",
        "admissible_prefix": "best_admissible_prefix_profile",
    }
    profiles: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    profile_columns: dict[str, np.ndarray] = {"x": x_common}
    spectrum_columns: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    common_frequency: np.ndarray | None = None
    for state, profile_key in state_profiles.items():
        for case in cases:
            ux, uy = interpolate_profile(case, x_common, profile_key)
            profiles[(state, case.name)] = (ux, uy)
            metrics, highpass, frequency, power = fixed_scale_metrics(
                x_common,
                ux,
                uy,
                target_y=target_y,
                filter_width=0.02,
            )
            metric_rows.append(
                {
                    "state": state,
                    "case": case.name,
                    "nx": case.nx,
                    "ny": case.ny,
                    "variant": case.variant,
                    **metrics,
                }
            )
            profile_columns[f"{state}_{case.name}_ux"] = ux
            profile_columns[f"{state}_{case.name}_uy"] = uy
            profile_columns[f"{state}_{case.name}_highpass"] = highpass
            if common_frequency is None:
                common_frequency = frequency
                spectrum_columns["cycles_per_unit_length"] = frequency
            elif not np.array_equal(frequency, common_frequency):
                raise AssertionError("common-grid spectra do not share frequencies")
            spectrum_columns[f"{state}_{case.name}_power"] = np.maximum(
                power, np.finfo(np.float64).tiny
            )

    with (cfg.output_dir / "common-top-profiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(profile_columns)
        writer.writerows(zip(*profile_columns.values(), strict=True))
    with (cfg.output_dir / "common-top-spectra.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(spectrum_columns)
        writer.writerows(zip(*spectrum_columns.values(), strict=True))
    write_rows(cfg.output_dir / "fixed-scale-metrics.csv", metric_rows)

    runner_path = (
        summary_path.parents[2] / "src" / "10-run-stable-neo-hookean-resolution.py"
    )
    runner = load_runner(runner_path)
    activation_eigen_rows = selected_activation_eigen_rows(cases, runner, poisson)
    write_rows(
        cfg.output_dir / "selected-activation-eigenvalues.csv",
        activation_eigen_rows,
    )
    comparison_rows = [
        row
        for state in state_profiles
        for row in profile_resolution_comparisons(cases, profiles, state)
    ]
    write_rows(cfg.output_dir / "profile-resolution-comparisons.csv", comparison_rows)
    subgrid_rows = [
        row
        for state in state_profiles
        for row in control_subgrid_rows(cases, runner, poisson, state)
    ]
    write_rows(cfg.output_dir / "activation-subgrid-content.csv", subgrid_rows)

    tangent: list[dict[str, Any]] = []
    singular_rows: list[dict[str, Any]] = []
    resolutions = sorted({(case.nx, case.ny) for case in cases})
    for nx, ny in resolutions:
        certificate, singular = tangent_certificate(
            runner,
            nx,
            ny,
            poisson=poisson,
            target_y=target_y,
            relative_tolerance=cfg.tangent_relative_tolerance,
            displacement_resolution=cfg.displacement_resolution,
            activation_scale=cfg.activation_scale,
        )
        tangent.append(certificate)
        singular_rows.extend(
            {
                "resolution": f"{nx}x{ny}",
                "index": index,
                "singular_value": float(value),
            }
            for index, value in enumerate(singular)
        )
    write_rows(cfg.output_dir / "tangent-singular-values.csv", singular_rows)
    gradient_check = implicit_gradient_check(runner, poisson, target_y)

    metrics_by_name = {(row["state"], row["case"]): row for row in metric_rows}
    finest_pair_gates: dict[str, Any] = {}
    for state in state_profiles:
        finest_pair_gates[state] = {}
        for variant in sorted({case.variant for case in cases}):
            family = sorted(
                (case for case in cases if case.variant == variant),
                key=lambda c: c.nx,
            )
            if len(family) < 2:
                continue
            coarse, fine = family[-2:]
            m0 = metrics_by_name[(state, coarse.name)]
            m1 = metrics_by_name[(state, fine.name)]
            comparison = next(
                row
                for row in comparison_rows
                if row["state"] == state
                and row["coarse"] == coarse.name
                and row["fine"] == fine.name
            )
            relative_highpass = abs(
                m1["highpass_rms_width_0p02"] - m0["highpass_rms_width_0p02"]
            ) / max(m1["highpass_rms_width_0p02"], np.finfo(float).tiny)
            relative_fit = abs(m1["target_vector_rms"] - m0["target_vector_rms"]) / max(
                m1["target_vector_rms"], np.finfo(float).tiny
            )
            relative_psd = abs(m1["psd_band_12_24"] - m0["psd_band_12_24"]) / max(
                m1["psd_band_12_24"], np.finfo(float).tiny
            )
            finest_pair_gates[state][variant] = {
                "coarse": coarse.name,
                "fine": fine.name,
                "relative_highpass_change": relative_highpass,
                "relative_target_fit_change": relative_fit,
                "relative_fixed_band_psd_change": relative_psd,
                "profile_correlation": comparison["profile_correlation"],
                "passes_5_percent_and_0p995_gate": bool(
                    relative_highpass <= 0.05
                    and relative_fit <= 0.05
                    and relative_psd <= 0.05
                    and comparison["profile_correlation"] >= 0.995
                ),
            }

    analysis = {
        "schema_version": 1,
        "design": summary["design"],
        "input_summary": str(summary_path),
        "common_profile_grid": {
            "x_min": float(x_common[0]),
            "x_max": float(x_common[-1]),
            "points": int(x_common.size),
            "spacing": float(x_common[1] - x_common[0]),
            "reason": "intersection of all native free-top observation grids",
        },
        "metrics": metric_rows,
        "profile_resolution_comparisons": comparison_rows,
        "activation_subgrid_content": subgrid_rows,
        "selected_activation_eigenvalues": activation_eigen_rows,
        "matched_data_loss_evolution": matched_rows,
        "evolution_diagnostics": evolution_rows,
        "initial_tangent_certificates": tangent,
        "implicit_gradient_finite_difference": gradient_check,
        "finest_pair_resolution_gates": finest_pair_gates,
        "claims": {
            "rank_scope": "Tangent ranks are local at zero activation, not nonlinear reachability certificates.",
            "resolution_scope": "Free per-triangle refinement changes the control space; tied and regularized families isolate this confound.",
            "roughness_scope": "All cross-mesh roughness metrics use a common physical grid, filter width, and Fourier bands.",
            "matched_loss_scope": "Trace comparisons select the nearest verified pre-inversion evaluation at four common data-loss levels; relative mismatch is reported explicitly.",
            "activation_admissibility_scope": "Every selected comparison state was checked post hoc to have positive-definite symmetric Ainv on every muscle triangle.",
        },
        "complete": bool(
            gradient_check["passed"]
            and all(
                row["positive_definite_selected_state"] for row in activation_eigen_rows
            )
        ),
    }
    write_json(cfg.output_dir / "analysis.json", analysis)


if __name__ == "__main__":
    cherries.main(main)
