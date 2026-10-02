# ruff: noqa: EM101, EM102, TRY003
"""Render measured activation-constraint outputs without interpreting them.

The matrix runner can still be writing while this script runs.  Only a case
with its atomic-looking ``summary.json`` and ``final.npz`` is an endpoint;
partial ``trace.csv`` files contribute only to convergence plots when their
case has completed.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
from experiment_profile import ProfileCometNoCommit
from matplotlib.lines import Line2D
from scipy import ndimage

from liblaf import cherries

LOG = logging.getLogger(__name__)
METHODS = ("Raw6", "G6", "G", "G-M", "G-S", "G-MS", "F", "F-M", "F-S", "F-MS", "Shared")
MAP_METHODS = ("Raw6", "G-MS", "F", "F-MS")
COLORS = {
    "Raw6": "#8c564b",
    "G6": "#9467bd",
    "G": "#1f77b4",
    "G-M": "#4c9ed9",
    "G-S": "#6baed6",
    "G-MS": "#08519c",
    "F": "#2ca25f",
    "F-M": "#74c476",
    "F-S": "#41ab5d",
    "F-MS": "#006d2c",
    "Shared": "#7f7f7f",
}


class Config(cherries.BaseConfig):
    """Analysis inputs and output locations relative to this experiment group."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    matrix_dir: Path = cherries.input("20-matrix")
    followup_dir: Path = cherries.input("30-followups")
    # This directory is created by the held-out runner after the initial matrix.
    smoothing_holdout_dir: Path = Path("data/32-smoothing-holdout")
    frequency_dir: Path = cherries.input("10-frequency")
    refinement_dir: Path = Path("data/40-refinement")
    polish_dir: Path = Path("data/35-polish")
    output_dir: Path = cherries.output("50-analysis", mkdir=True)
    highpass_length: float = 0.06


@dataclass(frozen=True)
class Endpoint:
    """One completed matrix case and its persisted result files."""

    target: str
    method: str
    group: str
    source: str
    summary: dict[str, Any]
    directory: Path
    fixture_path: Path

    @property
    def final(self) -> dict[str, Any]:
        return self.summary["final"]

    @property
    def converged(self) -> bool:
        return self.summary.get("status") == "projected_stationary"


def write_json(path: Path, value: Any) -> None:
    """Write a stable JSON artifact from only JSON-native values."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save each shareable plot as PNG and PDF."""
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_nested_endpoints(root: Path, source: str) -> list[Endpoint]:
    """Load completed group/target/method outputs with a fixture per group."""
    endpoints: list[Endpoint] = []
    for path in sorted(root.glob("*/*/*/summary.json")):
        try:
            summary = json.loads(path.read_text())
        except json.JSONDecodeError:
            LOG.warning("Skipping incomplete JSON: %s", path)
            continue
        directory = path.parent
        fixture_path = directory.parents[1] / "fixture.npz"
        if (
            "final" not in summary
            or not (directory / "final.npz").is_file()
            or not fixture_path.is_file()
        ):
            LOG.warning("Skipping non-endpoint %s: %s", source, directory)
            continue
        endpoints.append(
            Endpoint(
                str(summary.get("target", directory.parent.name)),
                str(summary.get("case", {}).get("name", directory.name)).split("__")[0],
                directory.parents[1].name,
                source,
                summary,
                directory,
                fixture_path,
            )
        )
    return endpoints


def load_endpoints(
    matrix_dir: Path, followup_dir: Path, smoothing_holdout_dir: Path
) -> list[Endpoint]:
    """Load only completed case endpoints, never a live trace as a final case."""
    endpoints: list[Endpoint] = []
    for path in sorted(matrix_dir.glob("*/*/summary.json")):
        try:
            summary = json.loads(path.read_text())
        except json.JSONDecodeError:
            LOG.warning("Skipping incomplete JSON: %s", path)
            continue
        directory = path.parent
        if "final" not in summary or not (directory / "final.npz").is_file():
            LOG.warning("Skipping non-endpoint case: %s", directory)
            continue
        target = str(summary.get("target", directory.parent.name))
        method = str(summary.get("case", {}).get("name", directory.name)).split("__")[0]
        endpoints.append(
            Endpoint(
                target,
                method,
                "initial-matrix",
                "initial-matrix",
                summary,
                directory,
                matrix_dir / "fixture.npz",
            )
        )
    endpoints.extend(load_nested_endpoints(followup_dir, "30-followups"))
    endpoints.extend(
        load_nested_endpoints(smoothing_holdout_dir, "32-smoothing-holdout")
    )
    return endpoints


def weighted_rms(
    values: np.ndarray, volumes: np.ndarray, components: float = 1.0
) -> float:
    """Compute a volume-weighted RMS with an explicit component normalization."""
    return float(
        np.sqrt(
            np.sum(volumes * np.sum(values * values, axis=tuple(range(1, values.ndim))))
            / (np.sum(volumes) * components)
        )
    )


def highpass_top(values: np.ndarray, size: int, length: float) -> np.ndarray:
    """Apply the runner's spatial high-pass filter to a structured top field."""
    field = grid(values, size)
    smooth = ndimage.gaussian_filter(
        field, sigma=(length * (size - 1), length * (size - 1)), mode="reflect"
    )
    return (field - smooth).reshape(-1)


def weighted_surface_rms(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the surface RMS using the fixture's lumped area weights."""
    return float(np.sqrt(np.average(values * values, weights=weights)))


def symmetric_log_if_spd(matrices: np.ndarray) -> np.ndarray | None:
    """Return symmetric matrix logs only when every stored inverse is SPD."""
    if not np.allclose(matrices, np.swapaxes(matrices, 1, 2), atol=1e-12, rtol=1e-10):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    if np.any(eigenvalues <= 0.0):
        return None
    return (eigenvectors * np.log(eigenvalues)[:, None, :]) @ np.swapaxes(
        eigenvectors, 1, 2
    )


def fiber_basis_metrics(H: np.ndarray, volumes: np.ndarray) -> dict[str, float | None]:
    """Measure log-strain components outside the known x-fiber basis tensor."""
    basis = np.diag((1.0, -0.5, -0.5))
    coefficient = np.einsum("nij,ij->n", H, basis) / 1.5
    remainder = H - coefficient[:, None, None] * basis
    denominator = float(np.sum(volumes * np.sum(H * H, axis=(1, 2))))
    if denominator <= 0.0:
        return {
            "off_fiber_log_energy_fraction": None,
            "negative_fiber_log_coefficient_volume_fraction": None,
        }
    return {
        "off_fiber_log_energy_fraction": float(
            np.sum(volumes * np.sum(remainder * remainder, axis=(1, 2))) / denominator
        ),
        "negative_fiber_log_coefficient_volume_fraction": float(
            np.average(coefficient < 0.0, weights=volumes)
        ),
    }


def target_residual_metrics(
    endpoint: Endpoint, length: float
) -> dict[str, float | str]:
    """Measure residuals against this endpoint's actual fixture target."""
    with (
        np.load(endpoint.fixture_path) as fixture,
        np.load(endpoint.directory / "final.npz") as final,
    ):
        top = fixture["top"]
        size = round(math.sqrt(len(top)))
        target = fixture[endpoint.target][top, 1]
        clean = fixture["clean"][top, 1]
        u = final["u"][top, 1]
        D = float(fixture["D"])
        weights = fixture["weights"]
        mode = endpoint.summary["case"]["mode"]
        volumes, truth_a, truth_ainv = (
            fixture["volumes"],
            fixture["truth_a"],
            fixture["truth_Ainv"],
        )
        log_ainv = symmetric_log_if_spd(final["Ainv"])
        log_truth = symmetric_log_if_spd(truth_ainv)
        if mode == "Raw6":
            activation_error = weighted_rms(final["Ainv"] - truth_ainv, volumes, 1.5)
            error_kind = "Ainv offset Frobenius RMS / sqrt(1.5)"
        elif mode in {"F", "Shared"}:
            q = final["q"].reshape(-1)
            if q.size == 1:
                q = np.full_like(truth_a, q.item())
            activation_error = weighted_rms((q - truth_a)[:, None], volumes)
            error_kind = "scalar log contraction RMS"
        else:
            if log_truth is None:
                raise ValueError("fixture truth_Ainv must be SPD")
            activation_error = weighted_rms(final["H"] - log_truth, volumes, 1.5)
            error_kind = "symmetric log-strain RMS / sqrt(1.5)"
        residual_hp = (
            weighted_surface_rms(highpass_top(u - target, size, length), weights) / D
        )
        target_excess_hp = (
            weighted_surface_rms(highpass_top(target - clean, size, length), weights)
            / D
        )
        if endpoint.group == "initial-matrix" and endpoint.target == "clean":
            reported = endpoint.final["excess_hp_0.06_over_D"]
            if not np.isclose(residual_hp, reported, rtol=0.0, atol=1e-10):
                raise ValueError(
                    f"clean target residual HP disagrees with persisted metric: "
                    f"{residual_hp:.16g} != {reported:.16g}"
                )
        log_metrics: dict[str, float | None] = {
            "activation_log_truth_error": None,
            "off_fiber_log_energy_fraction": None,
            "negative_fiber_log_coefficient_volume_fraction": None,
        }
        if log_ainv is not None and log_truth is not None:
            log_metrics["activation_log_truth_error"] = weighted_rms(
                log_ainv - log_truth, volumes, 1.5
            )
            log_metrics.update(fiber_basis_metrics(log_ainv, volumes))
        return {
            "residual_hp_0.06_over_D": residual_hp,
            "target_excess_hp_0.06_over_D": target_excess_hp,
            "activation_truth_error": activation_error,
            "activation_truth_error_kind": error_kind,
            **log_metrics,
        }


def endpoint_rows(endpoints: list[Endpoint], length: float) -> list[dict[str, Any]]:
    """Flatten selected endpoint metrics into portable table rows."""
    keys = (
        "fit_rms_over_D",
        "clean_rms_over_D",
        "excess_hp_0.06_over_D",
        "raw_hp_0.06_over_D",
        "projected_kkt",
        "low_frequency_amplitude",
        "low_frequency_amplitude_demeaned",
        "low_frequency_projection_residual",
        "low_frequency_projection_residual_demeaned",
        "detF_min",
        "detG_min",
        "activation_tensor_jump_rms",
        "activation_offset_jump_rms",
        "scalar_neighbor_jump_rms",
        "scalar_rms",
        "activation_log_rms",
        "activation_offset_rms",
    )
    rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        final = endpoint.final
        row = {
            "group": endpoint.group,
            "source": endpoint.source,
            "target": endpoint.target,
            "method": endpoint.method,
            "status": endpoint.summary.get("status"),
            "steps": endpoint.summary.get("steps"),
            "forward_calls": endpoint.summary.get("forward_calls"),
            "wall_s": endpoint.summary.get("wall_s"),
            "seed": endpoint.summary.get("seed"),
        }
        row.update({key: final.get(key) for key in keys})
        row.update(target_residual_metrics(endpoint, length))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write an empty-but-valid CSV when no endpoint has completed yet."""
    fields = sorted({key for row in rows for key in row}) or [
        "target",
        "method",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_three_target_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Pivot the three target endpoint metrics into one reviewable table."""
    metrics = (
        "fit_rms_over_D",
        "excess_hp_0.06_over_D",
        "clean_rms_over_D",
        "projected_kkt",
        "detF_min",
    )
    by_case = {
        (row["method"], row["target"]): row
        for row in rows
        if row["group"] == "initial-matrix"
    }
    targets = ("clean", "noisy", "mismatch")
    table: list[dict[str, Any]] = []
    for method in METHODS:
        table_row: dict[str, Any] = {"method": method}
        for target in targets:
            endpoint = by_case.get((method, target))
            table_row[f"{target}_status"] = (
                endpoint.get("status") if endpoint else "not_completed"
            )
            for metric in metrics:
                table_row[f"{target}_{metric}"] = (
                    endpoint.get(metric) if endpoint else None
                )
        table.append(table_row)
    write_csv(path, table)


def write_frequency_table(frequency_dir: Path, output: Path) -> int:
    """Preserve the measured forward-frequency summary beside inverse results."""
    path = frequency_dir / "summary.json"
    if not path.is_file():
        return 0
    value = json.loads(path.read_text())
    rows = [
        {
            "case": case.get("case"),
            "wave_number": case.get("wave_number"),
            "modulation_rms": case.get("modulation_rms"),
            "top_hp_0.06": case.get("top", {}).get("highpass_rms/ell_0.06"),
            "interface_hp_0.06": case.get("interface", {}).get("highpass_rms/ell_0.06"),
            "branch_over_signal": case.get("branch_over_signal"),
            "min_det_f": case.get("determinants", {}).get("min_det_f"),
        }
        for case in value.get("cases", [])
    ]
    write_csv(output / "forward-frequency-metrics.csv", rows)
    return len(rows)


def plot_fit_bump(
    endpoints: list[Endpoint],
    output: Path,
    metric: str,
    filename: str,
    ylabel: str,
) -> None:
    """Plot fit against one explicitly named normal high-pass metric."""
    targets = ("clean", "noisy", "mismatch")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=False, sharey=False)
    for axis, target in zip(axes, targets, strict=True):
        rows = [
            endpoint
            for endpoint in endpoints
            if endpoint.target == target and endpoint.group == "initial-matrix"
        ]
        for endpoint in rows:
            final = endpoint.final
            x = final.get("fit_rms_over_D")
            y = (
                endpoint.final.get(metric)
                if metric in endpoint.final
                else target_residual_metrics(endpoint, 0.06).get(metric)
            )
            if x is None or y is None:
                continue
            marker = "o" if endpoint.converged else "^"
            axis.scatter(
                x, y, color=COLORS.get(endpoint.method, "black"), marker=marker, s=48
            )
        axis.set_title(f"{target}: {len(rows)}/11 completed")
        axis.set_xlabel("fit RMS / D")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    fig.subplots_adjust(bottom=0.27)
    fig.legend(
        handles=method_legend_handles(),
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.08),
        fontsize=8,
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "circle: projected-stationary; triangle: other completed status",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, output / filename)


def method_legend_handles() -> list[Line2D]:
    """Return compact method labels so crowded points need no text annotations."""
    return [
        Line2D(
            [],
            [],
            color=COLORS.get(method, "black"),
            marker="o",
            linestyle="",
            label=method,
        )
        for method in METHODS
    ]


def plot_fiber_basis_metrics(rows: list[dict[str, Any]], output: Path) -> None:
    """Show recorded x-fiber-basis diagnostics for initial clean and noisy rows."""
    metrics = (
        (
            "off_fiber_log_energy_fraction",
            "off-fiber log energy fraction",
        ),
        (
            "negative_fiber_log_coefficient_volume_fraction",
            "negative x-fiber log coefficient volume fraction",
        ),
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7.2), sharex=True, constrained_layout=True
    )
    positions = np.arange(len(METHODS))
    width = 0.36
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        for offset, target, color in (
            (-width / 2, "clean", "#102a43"),
            (width / 2, "noisy", "#087e8b"),
        ):
            values: list[float] = []
            for method in METHODS:
                matches = [
                    row
                    for row in rows
                    if row["group"] == "initial-matrix"
                    and row["target"] == target
                    and row["method"] == method
                ]
                value = matches[0].get(metric) if matches else None
                values.append(math.nan if value is None else float(value))
            axis.bar(positions + offset, values, width=width, label=target, color=color)
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    axes[-1].set_xticks(positions, METHODS, rotation=35, ha="right")
    fig.suptitle("Initial matrix: x-fiber log-basis diagnostics")
    save_figure(fig, output / "fiber-basis-log-diagnostics")


def grid(values: np.ndarray, size: int) -> np.ndarray:
    """Reshape the structured top-surface ordering verified by the fixture."""
    if values.shape[0] != size * size:
        raise ValueError(f"expected {size * size} top values, got {values.shape[0]}")
    return values.reshape(size, size)


def completed_map_cases(endpoints: list[Endpoint], target: str) -> dict[str, Endpoint]:
    return {
        endpoint.method: endpoint
        for endpoint in endpoints
        if endpoint.target == target and endpoint.group == "initial-matrix"
    }


def panel_axes(count: int) -> tuple[plt.Figure, np.ndarray]:
    columns = 3
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 3.5 * rows), squeeze=False
    )
    return fig, axes.ravel()


def plot_target_maps(
    fixture: np.lib.npyio.NpzFile,
    endpoints: list[Endpoint],
    output: Path,
    length: float,
    target_name: str,
) -> None:
    """Render each initial-matrix target and its residual high-pass maps."""
    top = fixture["top"]
    points = fixture["points"]
    size = round(math.sqrt(len(top)))
    clean = fixture["clean"][top, 1]
    target = fixture[target_name][top, 1]
    cases = completed_map_cases(endpoints, target_name)
    fields: dict[str, np.ndarray | None] = {
        "Clean": clean,
        f"{target_name} target": target,
    }
    for name in MAP_METHODS:
        endpoint = cases.get(name)
        fields[name] = (
            np.load(endpoint.directory / "final.npz")["u"][top, 1] if endpoint else None
        )
    available = [value for value in fields.values() if value is not None]
    normal_limit = (
        max(float(np.max(np.abs(value))) for value in available) if available else 1.0
    )
    highpass = {
        name: None
        if name in {"Clean", f"{target_name} target"} or value is None
        else highpass_top(value - target, size, length)
        for name, value in fields.items()
    }
    hp_available = [value for value in highpass.values() if value is not None]
    hp_limit = (
        max(float(np.max(np.abs(value))) for value in hp_available)
        if hp_available
        else 1.0
    )
    extent = (
        float(points[top, 0].min()),
        float(points[top, 0].max()),
        float(points[top, 2].min()),
        float(points[top, 2].max()),
    )
    for title, data, limit, filename, label in (
        (
            "Top normal displacement",
            fields,
            normal_limit,
            f"{target_name}-normal-maps",
            "normal displacement",
        ),
        (
            f"Target-residual normal high-pass, {length:g} L",
            highpass,
            hp_limit,
            f"{target_name}-residual-highpass-maps",
            "target-residual normal high-pass",
        ),
    ):
        fig, axes = panel_axes(len(data))
        image = None
        for axis, (name, value) in zip(axes, data.items(), strict=True):
            axis.set_title(name)
            axis.set_xlabel("x / L")
            axis.set_ylabel("z / L")
            if value is None:
                axis.text(
                    0.5,
                    0.5,
                    "not completed",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                continue
            image = axis.imshow(
                grid(value, size),
                origin="lower",
                extent=extent,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                aspect="equal",
            )
        for axis in axes[len(data) :]:
            axis.set_visible(False)
        if image is not None:
            fig.colorbar(image, ax=axes[: len(data)], shrink=0.85, label=label)
        fig.suptitle(title)
        save_figure(fig, output / filename)


def plot_target_highpass_references(
    fixture: np.lib.npyio.NpzFile, output: Path, length: float
) -> None:
    """Show each fixture target's own high-pass content relative to clean."""
    top, points = fixture["top"], fixture["points"]
    size = round(math.sqrt(len(top)))
    clean = fixture["clean"][top, 1]
    fields = {
        name: highpass_top(fixture[name][top, 1] - clean, size, length)
        for name in ("clean", "noisy", "mismatch")
    }
    limit = max(float(np.max(np.abs(value))) for value in fields.values()) or 1.0
    extent = (
        float(points[top, 0].min()),
        float(points[top, 0].max()),
        float(points[top, 2].min()),
        float(points[top, 2].max()),
    )
    fig, axes = panel_axes(len(fields))
    for axis, (name, value) in zip(axes, fields.items(), strict=True):
        image = axis.imshow(
            grid(value, size),
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            aspect="equal",
        )
        axis.set(title=f"{name} target minus clean", xlabel="x / L", ylabel="z / L")
    fig.colorbar(
        image,
        ax=axes[: len(fields)],
        shrink=0.85,
        label="target excess normal high-pass",
    )
    fig.suptitle(f"Fixture target high-pass references, {length:g} L")
    save_figure(fig, output / "target-excess-highpass-references")


def plot_noisy_excess_height_fields(
    fixture: np.lib.npyio.NpzFile, endpoints: list[Endpoint], output: Path
) -> None:
    """Render top-surface normal errors as matched 3-D height fields, not geometry."""
    top, points = fixture["top"], fixture["points"]
    size = round(math.sqrt(len(top)))
    clean = fixture["clean"][top, 1]
    cases = completed_map_cases(endpoints, "noisy")
    methods = ("Raw6", "F-S", "F-MS")
    if any(method not in cases for method in methods):
        LOG.info(
            "Skipping noisy excess height fields until all requested methods finish"
        )
        return
    fields = {"Noisy 2% target": fixture["noisy"][top, 1] - clean}
    fields.update(
        {
            method: np.load(cases[method].directory / "final.npz")["u"][top, 1] - clean
            for method in methods
        }
    )
    normalized = {name: values / float(fixture["D"]) for name, values in fields.items()}
    limit = max(float(np.max(np.abs(values))) for values in normalized.values()) or 1.0
    x = grid(points[top, 0], size)
    z = grid(points[top, 2], size)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"projection": "3d"})
    colormap = plt.get_cmap("coolwarm")
    normalize = mpl.colors.Normalize(vmin=-limit, vmax=limit)
    for axis, (name, values) in zip(axes.ravel(), normalized.items(), strict=True):
        height = grid(values, size)
        axis.plot_surface(
            x,
            z,
            height,
            facecolors=colormap(normalize(height)),
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=True,
            shade=False,
        )
        axis.set(
            title=name,
            xlabel="x / L",
            ylabel="z / L",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            zlim=(-limit, limit),
        )
        axis.view_init(elev=30, azim=-60)
        axis.set_box_aspect((1.0, 1.0, 0.35))
    figure.colorbar(
        mpl.cm.ScalarMappable(norm=normalize, cmap=colormap),
        ax=axes.ravel().tolist(),
        shrink=0.72,
        pad=0.08,
        label="excess normal displacement / D",
    )
    figure.suptitle(
        "Noisy target and recovered top errors: "
        "height-error maps, not deformed geometry\n"
        "shared vertical and color scale: excess normal displacement / D"
    )
    save_figure(figure, output / "noisy-excess-height-fields")


def layer_average(
    centers: np.ndarray,
    volumes: np.ndarray,
    values: np.ndarray,
    nx: int,
) -> np.ndarray:
    """Volume-average per-tet diagnostics into the physical x-z cell grid."""
    ix = np.clip((centers[:, 0] * nx).astype(int), 0, nx - 1)
    iz = np.clip((centers[:, 2] * nx).astype(int), 0, nx - 1)
    numerator = np.zeros((nx, nx))
    denominator = np.zeros((nx, nx))
    np.add.at(numerator, (iz, ix), volumes * values)
    np.add.at(denominator, (iz, ix), volumes)
    if np.any(denominator == 0):
        raise ValueError("the active mesh did not cover a structured x-z cell")
    return numerator / denominator


def activation_quantity(endpoint: Endpoint, count: int) -> tuple[np.ndarray, str]:
    """Return a model-appropriate activation diagnostic, never mislabeled scalar RMS."""
    arrays = np.load(endpoint.directory / "final.npz")
    q, ainv, H = arrays["q"], arrays["Ainv"], arrays["H"]
    mode = endpoint.summary["case"]["mode"]
    if mode == "F":
        return q[:, 0], "scalar log contraction a"
    if mode == "Shared":
        return np.full(count, q.item()), "shared scalar log contraction a"
    if mode == "Raw6":
        return np.linalg.norm(
            ainv - np.eye(3), axis=(1, 2)
        ), "activation offset Frobenius norm"
    return np.sqrt(np.sum(H * H, axis=(1, 2)) / 1.5), "symmetric log-strain norm"


def plot_activation_maps(
    fixture: np.lib.npyio.NpzFile, endpoints: list[Endpoint], output: Path
) -> None:
    """Render layer-averaged muscle controls for completed noisy matrix rows."""
    cases = completed_map_cases(endpoints, "noisy")
    top = fixture["top"]
    nx = round(math.sqrt(len(top))) - 1
    centers = fixture["points"][fixture["tets"][fixture["active_ids"]]].mean(axis=1)
    volumes = fixture["volumes"]
    maps: dict[str, tuple[np.ndarray, str] | None] = {}
    for name in MAP_METHODS:
        endpoint = cases.get(name)
        if endpoint is None:
            maps[name] = None
            continue
        values, label = activation_quantity(endpoint, len(volumes))
        maps[name] = (layer_average(centers, volumes, values, nx), label)
    available = [item[0] for item in maps.values() if item is not None]
    limit = max(float(np.max(value)) for value in available) if available else 1.0
    fig, axes = panel_axes(len(maps))
    image = None
    for axis, (name, item) in zip(axes[: len(maps)], maps.items(), strict=True):
        if item is None:
            title = name
        else:
            compact_label = {
                "activation offset Frobenius norm": "offset Frobenius norm",
                "symmetric log-strain norm": "log-strain norm",
                "scalar log contraction a": "scalar contraction a",
                "shared scalar log contraction a": "shared scalar contraction a",
            }[item[1]]
            title = f"{name}\n{compact_label}"
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("x / L")
        axis.set_ylabel("z / L")
        if item is None:
            axis.text(
                0.5,
                0.5,
                "not completed",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            continue
        image = axis.imshow(
            item[0],
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="viridis",
            vmin=0.0,
            vmax=limit,
            aspect="equal",
        )
    for axis in axes[len(maps) :]:
        axis.set_visible(False)
    fig.subplots_adjust(top=0.80, right=0.84, hspace=0.55)
    if image is not None:
        fig.colorbar(
            image,
            ax=axes[: len(maps)],
            shrink=0.85,
            label="model-specific activation diagnostic",
        )
    fig.suptitle("Noisy target: volume-weighted muscle layer averages", y=0.97)
    save_figure(fig, output / "noisy-muscle-activation-maps")


def plot_profiles(
    fixture: np.lib.npyio.NpzFile, endpoints: list[Endpoint], output: Path
) -> None:
    """Plot the measured z=0.5 top-normal line profile for available noisy cases."""
    top, points = fixture["top"], fixture["points"]
    size = round(math.sqrt(len(top)))
    xz = points[top][:, (0, 2)]
    z_index = np.argmin(np.abs(xz.reshape(size, size, 2)[:, 0, 1] - 0.5))
    x = xz.reshape(size, size, 2)[z_index, :, 0]
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.plot(
        x,
        grid(fixture["clean"][top, 1], size)[z_index],
        color="black",
        linewidth=2.2,
        label="clean",
    )
    axis.plot(
        x,
        grid(fixture["noisy"][top, 1], size)[z_index],
        color="#d62728",
        linewidth=1.5,
        label="noisy target",
    )
    for name, endpoint in completed_map_cases(endpoints, "noisy").items():
        if name not in MAP_METHODS:
            continue
        u = np.load(endpoint.directory / "final.npz")["u"]
        axis.plot(x, grid(u[top, 1], size)[z_index], color=COLORS[name], label=name)
    axis.set_xlabel("x / L at z ≈ 0.5 L")
    axis.set_ylabel("top normal displacement")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    save_figure(fig, output / "noisy-z-half-line-profiles")


def neighbor_jump(row: dict[str, Any]) -> float | None:
    """Choose the measured neighbor diagnostic appropriate to its control space."""
    for key in (
        "scalar_neighbor_jump_rms",
        "activation_tensor_jump_rms",
        "activation_offset_jump_rms",
    ):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def plot_jump_bump(rows: list[dict[str, Any]], output: Path) -> None:
    """Plot each target's measured activation neighbor jump against surface bumping."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, target in zip(axes, ("clean", "noisy", "mismatch"), strict=True):
        for row in (
            item
            for item in rows
            if item["target"] == target and item["group"] == "initial-matrix"
        ):
            x, y = neighbor_jump(row), row.get("residual_hp_0.06_over_D")
            if x is None or y is None:
                continue
            axis.scatter(
                x,
                y,
                color=COLORS.get(row["method"], "black"),
                marker="o" if row["status"] == "projected_stationary" else "^",
                s=48,
            )
        axis.set_title(target)
        axis.set_xlabel("activation neighbor jump RMS")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("target-residual normal HP RMS at 0.06 L / D")
    fig.subplots_adjust(bottom=0.22)
    fig.legend(
        handles=method_legend_handles(),
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=8,
        frameon=False,
    )
    save_figure(fig, output / "activation-jump-versus-bump")


def load_trace(endpoint: Endpoint) -> list[dict[str, str]]:
    with (endpoint.directory / "trace.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_convergence(endpoints: list[Endpoint], output: Path) -> None:
    """Plot recorded fit and projected KKT histories for completed cases only."""
    for target in ("clean", "noisy", "mismatch"):
        selected = [
            endpoint
            for endpoint in endpoints
            if endpoint.target == target and endpoint.group == "initial-matrix"
        ]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for endpoint in selected:
            trace = load_trace(endpoint)
            if not trace:
                continue
            step = [int(row["step"]) for row in trace]
            fit = [max(float(row["fit_rms_over_D"]), 1e-16) for row in trace]
            kkt = [max(float(row["projected_kkt"]), 1e-16) for row in trace]
            style = "-" if endpoint.converged else "--"
            axes[0].semilogy(
                step,
                fit,
                style,
                color=COLORS.get(endpoint.method, "black"),
                label=endpoint.method,
            )
            axes[1].semilogy(
                step,
                kkt,
                style,
                color=COLORS.get(endpoint.method, "black"),
                label=endpoint.method,
            )
        axes[0].set(
            title=f"{target}: fit convergence",
            xlabel="outer step",
            ylabel="fit RMS / D",
        )
        axes[1].set(
            title=f"{target}: projected KKT", xlabel="outer step", ylabel="reported KKT"
        )
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
        save_figure(fig, output / f"{target}-convergence")


def write_followup_table(rows: list[dict[str, Any]], output: Path) -> int:
    """Write completed sensitivity endpoints with their fixture-specific metrics."""
    followups = [row for row in rows if row["group"] != "initial-matrix"]
    write_csv(output / "followup-endpoint-metrics.csv", followups)
    return len(followups)


def heldout_seed(row: dict[str, Any]) -> str | None:
    """Identify the named three-seed noise holdout groups and their persisted seed."""
    if not re.fullmatch(r"noise005-seed\d+", str(row["group"])):
        return None
    seed = row.get("seed")
    return str(seed) if seed is not None else row["group"].removeprefix("noise005-seed")


def write_sensitivity_table(rows: list[dict[str, Any]], output: Path) -> int:
    """Preserve the fiber, cap, and gamma sensitivity endpoints as one long table."""
    sensitivity = [
        row
        for row in rows
        if str(row["group"]).startswith(("fiber-", "cap", "gamma", "transverse-"))
    ]
    write_csv(output / "followup-sensitivity-metrics.csv", sensitivity)
    return len(sensitivity)


def write_initialization_comparison(rows: list[dict[str, Any]], output: Path) -> int:
    """Preserve matched-seed changes caused by the alternate initial activation."""
    baseline_group = "noise005-seed20260917"
    alternate_group = "noise005-init008"
    baseline = {
        row["method"]: row
        for row in rows
        if row["group"] == baseline_group and row["target"] == "noisy"
    }
    alternate = {
        row["method"]: row
        for row in rows
        if row["group"] == alternate_group and row["target"] == "noisy"
    }
    metrics = ("clean_rms_over_D", "excess_hp_0.06_over_D", "fit_rms_over_D")
    comparison: list[dict[str, Any]] = []
    for method in sorted(baseline.keys() & alternate.keys()):
        base, changed = baseline[method], alternate[method]
        row: dict[str, Any] = {
            "method": method,
            "baseline_group": baseline_group,
            "alternate_group": alternate_group,
            "baseline_status": base["status"],
            "alternate_status": changed["status"],
        }
        for metric in metrics:
            before, after = float(base[metric]), float(changed[metric])
            row[f"baseline_{metric}"] = before
            row[f"alternate_{metric}"] = after
            row[f"delta_{metric}"] = after - before
            row[f"relative_delta_{metric}"] = (after - before) / before
        comparison.append(row)
    write_csv(output / "followup-initialization-comparison.csv", comparison)
    return len(comparison)


def refinement_rows(refinement_dir: Path) -> list[dict[str, Any]]:
    """Flatten coarse and fine measurements, keeping their D scales separate."""
    path = refinement_dir / "summary.json"
    if not path.is_file():
        return []
    summary = json.loads(path.read_text())
    coarse = {item["case"]["name"]: item for item in summary["coarse_inverse"]}
    fine = {item["case"]: item for item in summary["fine_replay"]}
    transfer = summary["transfer"]
    rows: list[dict[str, Any]] = []
    for method in ("Raw6", "F-MS"):
        if method not in coarse or method not in fine:
            continue
        coarse_case, fine_case = coarse[method], fine[method]
        rows.append(
            {
                "method": method,
                "coarse_D": coarse_case["D"],
                "coarse_optimizer_status": coarse_case["status"],
                "coarse_steps": coarse_case["steps"],
                "coarse_target_fit_rms_over_D": coarse_case["final"]["fit_rms_over_D"],
                "coarse_excess_hp_0.06_over_D": coarse_case["final"][
                    "excess_hp_0.06_over_D"
                ],
                "fine_D": summary["fine_D"],
                "fine_replay_clean_error_rms_over_D": fine_case[
                    "surface_error_rms_over_D"
                ],
                "fine_replay_excess_hp_0.06_over_D": fine_case[
                    "surface_error_y_hp_0.06_rms_over_D"
                ],
                "fine_continuation_status": fine_case["continuation_forward"]["result"],
                "fine_reset_status": fine_case["reset_forward"]["result"],
                "control_transfer_kind": transfer["kind"],
                "whole_cell_containment_fraction": transfer[
                    "whole_cell_containment_fraction"
                ],
                "fine_active_tets": transfer["num_fine_active_tets"],
                "parent_coarse_active_tets": transfer[
                    "num_parent_coarse_active_tets_used"
                ],
            }
        )
    return rows


def plot_refinement_comparison(rows: list[dict[str, Any]], output: Path) -> None:
    """Plot coarse and fine metrics separately, with transfer evidence."""
    if not rows:
        return
    metrics = (
        ("coarse_target_fit_rms_over_D", "coarse target-fit RMS / coarse D"),
        ("fine_replay_clean_error_rms_over_D", "fine replay clean error RMS / fine D"),
        ("fine_replay_excess_hp_0.06_over_D", "fine replay normal HP.06 / fine D"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    methods = [row["method"] for row in rows]
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        values = [float(row[metric]) for row in rows]
        bars = axis.bar(methods, values, color=[COLORS[method] for method in methods])
        axis.bar_label(bars, fmt="%.4g", padding=3, fontsize=8)
        axis.set_ylim(top=max(values) * 1.18)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    transfer = rows[0]
    coarse_status = "; ".join(
        f"{row['method']} {row['coarse_optimizer_status']}" for row in rows
    )
    fine_status = "; ".join(
        f"{row['method']} continuation {row['fine_continuation_status']}"
        for row in rows
    )
    figure.suptitle(
        "Independent refined replay: coarse and fine normalizations kept separate"
    )
    figure.subplots_adjust(bottom=0.31, wspace=0.45)
    figure.text(
        0.5,
        0.06,
        "control transfer: "
        f"{transfer['control_transfer_kind']}; "
        f"{transfer['fine_active_tets']:,} fine active tets in "
        f"{transfer['parent_coarse_active_tets']:,} parent coarse tets; "
        f"containment {transfer['whole_cell_containment_fraction']:.0%}\n"
        f"coarse optimizer: {coarse_status}; fine replay: {fine_status}",
        ha="center",
        fontsize=8,
    )
    save_figure(figure, output / "refinement-comparison")


def polish_rows(polish_dir: Path) -> list[dict[str, Any]]:
    """Flatten polish receipts without interpreting gradient-audit outcomes."""
    rows: list[dict[str, Any]] = []
    for path in sorted(polish_dir.glob("*/*/summary.json")):
        try:
            summary = json.loads(path.read_text())
        except json.JSONDecodeError:
            LOG.warning("Skipping incomplete polish JSON: %s", path)
            continue
        final = summary.get("final")
        if not isinstance(final, dict):
            LOG.warning("Skipping polish receipt without final metrics: %s", path)
            continue
        audit = summary.get("directional_gradient_audit", {})
        errors = audit.get("finite_difference_relative_errors", [])
        case = summary.get("case", {})
        rows.append(
            {
                "source": summary.get("source"),
                "group": path.parent.parent.name,
                "method": str(case.get("name", path.parent.name)).split("__")[0],
                "target": summary.get("target"),
                "original_status": summary.get("original_status"),
                "status": summary.get("status"),
                "original_kkt": summary.get("original_kkt"),
                "projected_kkt": summary.get("projected_kkt"),
                "steps": summary.get("steps"),
                "endpoint_change_rms_over_D": summary.get("endpoint_change_rms_over_D"),
                "fit_rms_over_D": final.get("fit_rms_over_D"),
                "clean_rms_over_D": final.get("clean_rms_over_D"),
                "excess_hp_0.06_over_D": final.get("excess_hp_0.06_over_D"),
                "low_frequency_amplitude_demeaned": final.get(
                    "low_frequency_amplitude_demeaned"
                ),
                "strict_start_minus_persisted_objective": summary.get(
                    "strict_start_minus_persisted_objective"
                ),
                "fd_available": audit.get("available"),
                "fd_kind": audit.get("direction_kind"),
                "fd_mask_fraction": audit.get("direction_mask_fraction"),
                "fd_relative_error_0": errors[0] if len(errors) > 0 else None,
                "fd_relative_error_1": errors[1] if len(errors) > 1 else None,
                "fd_scale_disagreement": summary.get(
                    "finite_difference_scale_disagreement"
                ),
                "branch_reset_difference_over_D": summary.get(
                    "branch_reset_difference_over_D"
                ),
            }
        )
    return rows


def plot_heldout_comparison(rows: list[dict[str, Any]], output: Path) -> int:
    """Compare available held-out seed outcomes with method colors and seed pairing."""
    methods = ("Raw6", "G-MS", "F-S", "F-MS")
    selected = [
        row
        for row in rows
        if row["target"] == "noisy"
        and row["method"] in methods
        and heldout_seed(row) is not None
    ]
    write_csv(output / "smoothing-heldout-comparison.csv", selected)
    if not selected:
        return 0
    seeds = sorted({heldout_seed(row) for row in selected if heldout_seed(row)})
    markers = ("o", "s", "^")
    metrics = (
        ("clean_rms_over_D", "clean error RMS / D"),
        ("excess_hp_0.06_over_D", "normal HP of u minus clean / D"),
        ("fit_rms_over_D", "noisy-target fit RMS / D"),
    )
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(14, 4.4), constrained_layout=True
    )
    positions = {method: number for number, method in enumerate(methods)}
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        for seed_index, seed in enumerate(seeds):
            by_method = {
                row["method"]: row for row in selected if heldout_seed(row) == seed
            }
            line_x = [positions[method] for method in methods if method in by_method]
            line_y = [
                float(by_method[method][metric])
                for method in methods
                if method in by_method
            ]
            axis.plot(line_x, line_y, color="#9fb3c8", linewidth=1.0, zorder=1)
            for method in methods:
                row = by_method.get(method)
                if row is None:
                    continue
                axis.scatter(
                    positions[method],
                    float(row[metric]),
                    color=COLORS[method],
                    marker=markers[seed_index % len(markers)],
                    s=60,
                    zorder=2,
                )
        axis.set_xticks(range(len(methods)), methods)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    method_handles = [
        Line2D([], [], color=COLORS[method], marker="o", linestyle="", label=method)
        for method in methods
    ]
    seed_handles = [
        Line2D(
            [],
            [],
            color="#526d82",
            marker=markers[index % len(markers)],
            linestyle="",
            label=f"seed {seed}",
        )
        for index, seed in enumerate(seeds)
    ]
    fig.legend(
        handles=[*method_handles, *seed_handles],
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Held-out noise seeds: paired method comparison")
    save_figure(fig, output / "smoothing-heldout-comparison")
    return len(selected)


def main(cfg: Config) -> None:
    """Generate analysis artifacts from completed endpoints available at invocation."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    endpoints = load_endpoints(
        cfg.matrix_dir, cfg.followup_dir, cfg.smoothing_holdout_dir
    )
    rows = endpoint_rows(endpoints, cfg.highpass_length)
    write_csv(cfg.output_dir / "endpoint-metrics.csv", rows)
    write_three_target_table(cfg.output_dir / "three-target-metrics.csv", rows)
    frequency_cases = write_frequency_table(cfg.frequency_dir, cfg.output_dir)
    fixture_path = cfg.matrix_dir / "fixture.npz"
    if fixture_path.is_file():
        with np.load(fixture_path) as fixture:
            for target in ("clean", "noisy", "mismatch"):
                plot_target_maps(
                    fixture, endpoints, cfg.output_dir, cfg.highpass_length, target
                )
            plot_target_highpass_references(
                fixture, cfg.output_dir, cfg.highpass_length
            )
            plot_noisy_excess_height_fields(fixture, endpoints, cfg.output_dir)
            plot_activation_maps(fixture, endpoints, cfg.output_dir)
            plot_profiles(fixture, endpoints, cfg.output_dir)
    plot_fit_bump(
        endpoints,
        cfg.output_dir,
        "excess_hp_0.06_over_D",
        "fit-versus-excess-highpass",
        "normal HP RMS of u minus clean at 0.06 L / D",
    )
    plot_fit_bump(
        endpoints,
        cfg.output_dir,
        "residual_hp_0.06_over_D",
        "fit-versus-target-residual-highpass",
        "normal HP RMS of u minus target at 0.06 L / D",
    )
    plot_fiber_basis_metrics(rows, cfg.output_dir)
    plot_jump_bump(rows, cfg.output_dir)
    plot_convergence(endpoints, cfg.output_dir)
    followup_count = write_followup_table(rows, cfg.output_dir)
    sensitivity_count = write_sensitivity_table(rows, cfg.output_dir)
    initialization_count = write_initialization_comparison(rows, cfg.output_dir)
    heldout_count = plot_heldout_comparison(rows, cfg.output_dir)
    refinement = refinement_rows(cfg.refinement_dir)
    write_csv(cfg.output_dir / "refinement-comparison.csv", refinement)
    plot_refinement_comparison(refinement, cfg.output_dir)
    polish = polish_rows(cfg.polish_dir)
    write_csv(cfg.output_dir / "polish-comparison.csv", polish)
    summary = {
        "completed_endpoints": len(endpoints),
        "completed_by_target": {
            target: sum(endpoint.target == target for endpoint in endpoints)
            for target in ("clean", "noisy", "mismatch")
        },
        "methods_expected": list(METHODS),
        "highpass_length": cfg.highpass_length,
        "completed_followup_endpoints": followup_count,
        "completed_sensitivity_endpoints": sensitivity_count,
        "matched_initialization_comparisons": initialization_count,
        "refinement_comparison_rows": len(refinement),
        "polish_comparison_rows": len(polish),
        "completed_heldout_comparison_endpoints": heldout_count,
        "forward_frequency_cases": frequency_cases,
        "endpoint_policy": (
            "Only summary.json plus final.npz cases are treated as endpoints."
        ),
    }
    write_json(cfg.output_dir / "analysis-summary.json", summary)
    cherries.log_metrics(
        {
            "analysis/completed_endpoints": len(endpoints),
            "analysis/forward_frequency_cases": frequency_cases,
            "analysis/followup_endpoints": followup_count,
        }
    )
    LOG.info(
        "Wrote analysis for %d completed endpoints to %s",
        len(endpoints),
        cfg.output_dir,
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
