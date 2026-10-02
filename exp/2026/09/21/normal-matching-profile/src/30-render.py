"""Render matched, target-normal continuation profiles from saved histories only."""

# ruff: noqa: C901, EM101, EM102, TRY003, TRY004

from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
from experiment import Profile
from matplotlib.lines import Line2D

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

GROUP = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
VARIANTS = ("l2", "gradient-005", "gradient-025", "normal-005", "normal-025")
MODES = ("unconstrained", "contraction_only", "learned_direction", "x_contraction")
HEIGHTS = (0.05, 0.20)
COLORS = {
    "l2": "#4d4a46",
    "gradient-005": "#1769aa",
    "gradient-025": "#1769aa",
    "normal-005": "#c05621",
    "normal-025": "#c05621",
}
LABELS = {
    "l2": "Continued L2",
    "gradient-005": "Gradient β=0.05",
    "gradient-025": "Gradient β=0.25",
    "normal-005": "Normal β=0.05",
    "normal-025": "Normal β=0.25",
}
MODE_LABELS = {
    "unconstrained": "Unrestricted symmetric",
    "contraction_only": "PSD contraction-only",
    "learned_direction": "Learned-axis rank-one",
    "x_contraction": "Fixed-x contraction",
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = GROUP / "data/10-comparison"
    checks_dir: Path = GROUP / "data/20-verification"
    output_dir: Path = GROUP / "data/30-figures"


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    height: float
    shared_step: int
    selected_gradient: str | None
    selected_normal: str | None
    eligibility: dict[str, bool]
    unstable_equilibrium: bool
    backtracking_warning: bool


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _trace(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) < {"step", "objective", "projected_gradient_inf"}:
        raise ValueError(f"incomplete trace: {path}")
    result = {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
    }
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError(f"nonfinite trace: {path}")
    return result


def _history_at(path: Path, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        required = {"points", "triangles", "u", "steps", "top_all"}
        if required - set(saved.files):
            raise ValueError(
                f"history lacks {sorted(required - set(saved.files))}: {path}"
            )
        indexes = np.flatnonzero(np.asarray(saved["steps"]) == step)
        if len(indexes) != 1:
            raise ValueError(f"history has no unique saved common step {step}: {path}")
        points = np.asarray(saved["points"], dtype=float)
        triangles = np.asarray(saved["triangles"], dtype=np.int64)
        top = np.asarray(saved["top_all"], dtype=np.int64)
        u = np.asarray(saved["u"], dtype=float)[indexes[0]]
    if points.ndim != 2 or points.shape[1] != 2 or u.shape != points.shape:
        raise ValueError(f"invalid profile geometry: {path}")
    return points, triangles, u, top


def _load_cases(selection: dict[str, Any]) -> tuple[Case, ...]:
    cells = selection.get("cells")
    if not isinstance(cells, list) or len(cells) != len(MODES) * len(HEIGHTS):
        raise ValueError("selection must contain all eight comparison cells")
    result: list[Case] = []
    for raw in cells:
        if not isinstance(raw, dict):
            raise ValueError("selection cell must be an object")
        required = {"name", "mode", "height", "shared_step", "variants", "selected"}
        if required - set(raw):
            raise ValueError(f"selection cell lacks {sorted(required - set(raw))}")
        variants, selected = raw["variants"], raw["selected"]
        if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
            raise ValueError(
                f"selection variants must be exactly {VARIANTS}: {raw['name']}"
            )
        if not isinstance(selected, dict) or set(selected) != {"gradient", "normal"}:
            raise ValueError(f"selection selected contract invalid: {raw['name']}")
        eligible = {}
        for variant, metric in variants.items():
            if not isinstance(metric, dict) or "eligible" not in metric:
                raise ValueError(
                    f"selection eligibility missing: {raw['name']}/{variant}"
                )
            eligible[variant] = bool(metric["eligible"])
        gradient, normal = selected["gradient"], selected["normal"]
        if gradient is not None and (
            gradient not in VARIANTS or not eligible[gradient]
        ):
            raise ValueError(f"invalid selected gradient for {raw['name']}")
        if normal is not None and (normal not in VARIANTS or not eligible[normal]):
            raise ValueError(f"invalid selected normal for {raw['name']}")
        relevant = ("l2", normal)
        unstable = any(
            name is not None
            and float(
                variants[name]
                .get("physical_hessian", {})
                .get("physical_hessian_smallest_algebraic_eigenvalue", np.inf)
            )
            < 0
            for name in relevant
        )
        result.append(
            Case(
                name=str(raw["name"]),
                mode=str(raw["mode"]),
                height=float(raw["height"]),
                shared_step=int(raw["shared_step"]),
                selected_gradient=gradient,
                selected_normal=normal,
                eligibility=eligible,
                unstable_equilibrium=unstable,
                backtracking_warning=bool(
                    raw["qualifying_candidate_backtracking_warning"]
                ),
            )
        )
    if {(case.mode, case.height) for case in result} != {
        (mode, height) for mode in MODES for height in HEIGHTS
    }:
        raise ValueError("selection cells do not cover expected modes and heights")
    return tuple(result)


def _target(points: np.ndarray, top: np.ndarray, height: float) -> np.ndarray:
    reference = points[top]
    x = reference[:, 0]
    return reference + np.column_stack((np.zeros_like(x), 4 * height * x * (1 - x)))


def _style(axis: plt.Axes) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.15, 0.34)
    axis.grid(alpha=0.2)
    axis.set_xticks((0, 0.5, 1))
    axis.set_yticks((-0.1, 0, 0.1, 0.2, 0.3))
    axis.set_xlabel("deformed x")
    axis.set_ylabel("y")


def _variant_style(variant: str) -> str:
    """Use color for loss family and line style for beta throughout the figures."""
    return {
        "l2": "--",
        "gradient-005": "-",
        "gradient-025": ":",
        "normal-005": "-",
        "normal-025": ":",
    }[variant]


def _case_paths(source: Path, case: Case, variant: str) -> Path:
    path = source / case.name / variant
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _cell_warning(case: Case) -> str:
    messages = []
    if case.unstable_equilibrium:
        messages.append("UNSTABLE EQUILIBRIUM\nnegative physical Hessian")
    if case.backtracking_warning:
        messages.append("top-edge backtracking warning")
    return "\n".join(messages)


def _profile_overview(source: Path, cases: tuple[Case, ...], output: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(13.5, 11), sharex=True, sharey=True)
    handles = []
    for row, mode in enumerate(MODES):
        for column, height in enumerate(HEIGHTS):
            axis = axes[row, column]
            case = next(c for c in cases if c.mode == mode and c.height == height)
            points, _, start_u, top = _history_at(
                source / case.name / "l2" / "history.npz", 0
            )
            target = _target(points, top, height)
            handles.extend(
                axis.plot(target[:, 0], target[:, 1], "k--", lw=1.8, label="Target top")
            )
            handles.extend(
                axis.plot(
                    points[top, 0],
                    points[top, 1] + start_u[top, 1],
                    color="#888888",
                    lw=1.3,
                    label="Shared start",
                )
            )
            for variant in ("l2", case.selected_gradient, case.selected_normal):
                if variant is None:
                    continue
                _, _, u, _ = _history_at(
                    _case_paths(source, case, variant) / "history.npz", case.shared_step
                )
                label = LABELS[variant]
                if variant != "l2" and not case.eligibility[variant]:
                    raise AssertionError("selected candidates must be eligible")
                handles.extend(
                    axis.plot(
                        points[top, 0] + u[top, 0],
                        points[top, 1] + u[top, 1],
                        color=COLORS[variant],
                        ls=_variant_style(variant),
                        lw=1.6,
                        label=label,
                    )
                )
            rejected = [v for v in VARIANTS if v != "l2" and not case.eligibility[v]]
            selection_text = (
                f"shared step {case.shared_step}\n"
                f"gradient: {LABELS[case.selected_gradient] if case.selected_gradient else 'none eligible'}\n"
                f"normal: {LABELS[case.selected_normal] if case.selected_normal else 'none eligible'}"
            )
            if rejected:
                selection_text += "\nrejected: " + ", ".join(rejected)
            if warning := _cell_warning(case):
                selection_text += "\n" + warning
            axis.text(
                0.02,
                0.03,
                selection_text,
                transform=axis.transAxes,
                fontsize=7.4,
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
            axis.set_title(
                f"{MODE_LABELS[mode]} | target height {height:.2f}", fontsize=9
            )
            _style(axis)
    unique = {handle.get_label(): handle for handle in handles}
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncols=4,
        frameon=False,
    )
    figure.suptitle(
        "Matched continuation profiles at each verification-selected shared step",
        y=0.995,
        fontsize=14,
    )
    figure.text(
        0.5,
        0.002,
        "Equal physical axes in every panel. A missing selected shape means no candidate met the predeclared gate; it is not substituted.",
        ha="center",
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.06, right=0.985, top=0.89, bottom=0.07, wspace=0.18, hspace=0.42
    )
    figure.savefig(output / "shared-step-profile-comparison.png", dpi=200)
    plt.close(figure)


def _candidate_profiles(source: Path, cases: tuple[Case, ...], output: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(13.5, 11), sharex=True, sharey=True)
    for row, mode in enumerate(MODES):
        for column, height in enumerate(HEIGHTS):
            axis = axes[row, column]
            case = next(c for c in cases if c.mode == mode and c.height == height)
            points, _, _, top = _history_at(
                _case_paths(source, case, "l2") / "history.npz", case.shared_step
            )
            target = _target(points, top, height)
            axis.plot(target[:, 0], target[:, 1], "k--", lw=1.5, label="Target top")
            _, _, start_u, _ = _history_at(
                _case_paths(source, case, "l2") / "history.npz", 0
            )
            axis.plot(
                points[top, 0] + start_u[top, 0],
                points[top, 1] + start_u[top, 1],
                color="#888888",
                lw=1.1,
                label="Shared start",
            )
            for variant in VARIANTS:
                _, _, u, _ = _history_at(
                    _case_paths(source, case, variant) / "history.npz", case.shared_step
                )
                eligible = case.eligibility[variant]
                axis.plot(
                    points[top, 0] + u[top, 0],
                    points[top, 1] + u[top, 1],
                    color=COLORS[variant],
                    ls=_variant_style(variant),
                    lw=1.4,
                    alpha=1 if eligible or variant == "l2" else 0.45,
                    label=f"{LABELS[variant]}{' (rejected)' if variant != 'l2' and not eligible else ''}",
                )
            axis.set_title(
                f"{MODE_LABELS[mode]} | height {height:.2f} | common step {case.shared_step}",
                fontsize=8.6,
            )
            if warning := _cell_warning(case):
                axis.text(
                    0.98,
                    0.97,
                    warning,
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    color="#9c2c2c",
                    weight="bold",
                    bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#9c2c2c"},
                )
            _style(axis)
    figure.suptitle(
        "All tested continuation candidates at the common valid step", fontsize=14
    )
    figure.legend(
        handles=[
            Line2D([], [], color="black", ls="--", label="Target top"),
            Line2D([], [], color="#888888", label="Shared start"),
            Line2D([], [], color=COLORS["l2"], ls="--", label=LABELS["l2"]),
            Line2D(
                [], [], color=COLORS["gradient-005"], ls="-", label="Gradient β=0.05"
            ),
            Line2D(
                [], [], color=COLORS["gradient-025"], ls=":", label="Gradient β=0.25"
            ),
            Line2D([], [], color=COLORS["normal-005"], ls="-", label="Normal β=0.05"),
            Line2D([], [], color=COLORS["normal-025"], ls=":", label="Normal β=0.25"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncols=4,
        frameon=False,
        fontsize=8,
    )
    figure.text(
        0.5,
        0.002,
        "Blue: gradient loss; orange: normal loss; solid: β=0.05; dotted: β=0.25. Faded profiles were rejected by the verification gate and remain shown for the tradeoff.",
        ha="center",
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.06, right=0.985, top=0.90, bottom=0.07, wspace=0.18, hspace=0.42
    )
    figure.savefig(output / "all-candidate-shared-step-profiles.png", dpi=200)
    plt.close(figure)


def _wireframes(source: Path, cases: tuple[Case, ...], output: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(13.5, 11), sharex=True, sharey=True)
    for row, mode in enumerate(MODES):
        for column, height in enumerate(HEIGHTS):
            axis = axes[row, column]
            case = next(c for c in cases if c.mode == mode and c.height == height)
            points, triangles, l2_u, top = _history_at(
                _case_paths(source, case, "l2") / "history.npz", case.shared_step
            )
            target = _target(points, top, height)
            axis.plot(
                target[:, 0], target[:, 1], "k--", lw=1.6, label="Target top only"
            )
            for variant, u, color in (("l2", l2_u, COLORS["l2"]),):
                deformed = points + u
                axis.triplot(
                    deformed[:, 0],
                    deformed[:, 1],
                    triangles,
                    color=color,
                    lw=0.28,
                    alpha=0.85,
                    label=LABELS[variant],
                )
            if case.selected_normal is None:
                axis.text(
                    0.5,
                    0.12,
                    "No eligible normal candidate",
                    transform=axis.transAxes,
                    ha="center",
                    color="#9c2c2c",
                    fontsize=8,
                )
            else:
                _, _, normal_u, _ = _history_at(
                    _case_paths(source, case, case.selected_normal) / "history.npz",
                    case.shared_step,
                )
                deformed = points + normal_u
                axis.triplot(
                    deformed[:, 0],
                    deformed[:, 1],
                    triangles,
                    color=COLORS[case.selected_normal],
                    lw=0.28,
                    alpha=0.75,
                    label=LABELS[case.selected_normal],
                )
            axis.set_title(f"{MODE_LABELS[mode]} | height {height:.2f}", fontsize=8.6)
            if warning := _cell_warning(case):
                axis.text(
                    0.98,
                    0.97,
                    warning,
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    color="#9c2c2c",
                    weight="bold",
                    bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#9c2c2c"},
                )
            _style(axis)
    figure.suptitle(
        "Internal wireframes at common step: target boundary, L2, and selected normal match",
        fontsize=13,
    )
    figure.legend(
        handles=[
            Line2D([], [], color="black", ls="--", label="Target top only"),
            Line2D([], [], color=COLORS["l2"], label="Continued L2 mesh"),
            Line2D([], [], color=COLORS["normal-005"], label="Selected normal mesh"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncols=3,
        frameon=False,
        fontsize=8,
    )
    figure.text(
        0.5,
        0.002,
        "The target is a boundary profile only; no target interior displacement field is fabricated. Equal physical axes in every panel.",
        ha="center",
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.06, right=0.985, top=0.90, bottom=0.07, wspace=0.18, hspace=0.42
    )
    figure.savefig(output / "internal-wireframes.png", dpi=200)
    plt.close(figure)


def _curves(source: Path, cases: tuple[Case, ...], output: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(13.5, 11), sharex=False)
    for row, mode in enumerate(MODES):
        for column, height in enumerate(HEIGHTS):
            case = next(c for c in cases if c.mode == mode and c.height == height)
            objective_axis = axes[row, column]
            gradient_axis = objective_axis.twinx()
            for variant in VARIANTS:
                trace = _trace(_case_paths(source, case, variant) / "trace.csv")
                eligible = case.eligibility[variant]
                alpha = 1.0 if eligible or variant == "l2" else 0.42
                objective_axis.plot(
                    trace["step"],
                    trace["objective"] / trace["objective"][0],
                    color=COLORS[variant],
                    ls=_variant_style(variant),
                    lw=1.35,
                    alpha=alpha,
                    label=LABELS[variant],
                )
                gradient_axis.plot(
                    trace["step"],
                    trace["projected_gradient_inf"]
                    / trace["projected_gradient_inf"][0],
                    color=COLORS[variant],
                    ls=(0, (1, 1)),
                    lw=0.95,
                    alpha=alpha,
                )
            objective_axis.axvline(case.shared_step, color="#777777", lw=0.8, ls=":")
            gradient_axis.axvline(case.shared_step, color="#777777", lw=0.8, ls=":")
            objective_axis.set_title(
                f"{MODE_LABELS[mode]} | height {height:.2f} | common step {case.shared_step}",
                fontsize=8.5,
            )
            if warning := _cell_warning(case):
                objective_axis.text(
                    0.98,
                    0.96,
                    warning,
                    transform=objective_axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.8,
                    color="#9c2c2c",
                    weight="bold",
                    bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#9c2c2c"},
                )
            objective_axis.set_ylabel("Own objective / J(0)")
            gradient_axis.set_ylabel("Projected gradient inf / initial")
            gradient_axis.set_yscale("log")
            objective_axis.set_xlabel("Continuation update")
            objective_axis.grid(alpha=0.22)
    figure.suptitle(
        "Five candidate histories; dotted vertical line marks the shared comparison step",
        fontsize=14,
    )
    figure.legend(
        handles=[
            Line2D([], [], color=COLORS["l2"], ls="--", label="Continued L2 objective"),
            Line2D(
                [],
                [],
                color=COLORS["gradient-005"],
                ls="-",
                label="Gradient β=0.05 objective",
            ),
            Line2D(
                [],
                [],
                color=COLORS["gradient-025"],
                ls=":",
                label="Gradient β=0.25 objective",
            ),
            Line2D(
                [],
                [],
                color=COLORS["normal-005"],
                ls="-",
                label="Normal β=0.05 objective",
            ),
            Line2D(
                [],
                [],
                color=COLORS["normal-025"],
                ls=":",
                label="Normal β=0.25 objective",
            ),
            Line2D(
                [],
                [],
                color="#444444",
                ls=(0, (1, 1)),
                label="Fine-dot: projected gradient (right axis)",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncols=3,
        frameon=False,
        fontsize=7.5,
    )
    figure.text(
        0.5,
        0.002,
        "Blue: gradient; orange: normal; solid β=0.05; dotted β=0.25. Fine-dot lines use the right, log-scaled projected-gradient axis. Faded traces are rejected.\nEach objective includes its own beta-weighted term and is normalized by its own J(0), so heights are not a cross-variant fit ranking. Red labels mark verified negative physical-Hessian equilibria.",
        ha="center",
        fontsize=7.7,
    )
    figure.subplots_adjust(
        left=0.07, right=0.94, top=0.90, bottom=0.08, wspace=0.34, hspace=0.48
    )
    figure.savefig(output / "objective-and-projected-gradient-histories.png", dpi=200)
    plt.close(figure)


def _status_table(
    source: Path, cases: tuple[Case, ...], output: Path
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        for variant in VARIANTS:
            summary = _json(_case_paths(source, case, variant) / "summary.json")
            rows.append(
                {
                    "cell": case.name,
                    "variant": variant,
                    "eligible": case.eligibility[variant],
                    "shared_step": case.shared_step,
                    "accepted_updates": summary["accepted_iterations"],
                    "failure": summary["failure"],
                    "optimizer_message": summary["optimizer_message"],
                    "endpoint_normal_angle_rms_deg": summary["final"][
                        "normal_angle_rms_deg"
                    ],
                    "endpoint_fit_rms": summary["final"]["fit_rms"],
                    "endpoint_inverted_cells": summary["final"]["inverted_cells"],
                }
            )
    (output / "selection-and-endpoints.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    return rows


def main(cfg: Config) -> None:
    LOG.info("Rendering saved profile histories; no inverse solves are run.")
    source = cherries.input(cfg.comparison_dir.resolve())
    checks = cherries.input(cfg.checks_dir.resolve())
    output = cherries.output(cfg.output_dir.resolve(), mkdir=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selection_path = checks / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = _json(selection_path)
    cases = _load_cases(selection)
    _profile_overview(source, cases, output)
    _candidate_profiles(source, cases, output)
    _wireframes(source, cases, output)
    _curves(source, cases, output)
    endpoints = _status_table(source, cases, output)
    shutil.copy2(Path(__file__), output / "source-30-render.py")
    receipt = {
        "purpose": "Read-only visual comparison of target-normal and gradient continuation histories.",
        "comparison_dir": str(source),
        "selection": str(selection_path),
        "selection_contract": "all candidate variants retained; null selection means no eligible candidate and no substitution",
        "equal_profile_axes": {"x": [-0.03, 1.03], "y": [-0.15, 0.34]},
        "target": "Top-boundary target only; internal target deformation is intentionally not rendered.",
        "endpoint_rows": len(endpoints),
        "python": sys.version,
    }
    (output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for path in output.iterdir():
        cherries.log_output(path)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
