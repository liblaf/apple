"""Render the activation-direction smoothness study from saved arrays only."""

# ruff: noqa: C901, EM101, EM102, RUF001, TRY003

from __future__ import annotations

import csv
import html
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parents[1]
MODES = (
    "unconstrained",
    "contraction_only",
    "learned_direction",
    "x_contraction",
)
HEIGHTS = (0.05, 0.20)
LABELS = {
    "unconstrained": "Unconstrained symmetric B",
    "contraction_only": "Contraction only",
    "learned_direction": "Learned direction",
    "x_contraction": "Fixed x contraction",
}
ACTIVATION_LABELS = {
    "unconstrained": "Unconstrained\nsymmetric B",
    "contraction_only": "Contraction only\nfree directions",
    "learned_direction": "Contraction only\nlearned axis",
    "x_contraction": "Fixed x\ncontraction",
}
COLORS = {
    "unconstrained": "#5b3c88",
    "contraction_only": "#267f75",
    "learned_direction": "#d17a22",
    "x_contraction": "#3b6fa1",
}
PASSIVE_COLOR = "#e8dcc8"
MUSCLE_COLOR = "#d98f82"
EDGE_COLOR = "#594f48"
POSITIVE_COLOR = "#c44335"
NEGATIVE_COLOR = "#3269a8"


class ProfileFigures(profiles.Profile):
    """Normal evidence profile without automatic Git mutations."""

    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    """Input selections and output directory, relative to experiment data/."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_dirs: str = "tune-w0,tune-w001,tune-w01,tune-w1"
    output: Path = Path("70-readable-glyphs")
    selected_weight: float = 1.0
    tuning_dirs: str = "tune-w0,tune-w001,tune-w01,tune-w1"


@dataclass(frozen=True)
class Case:
    source: Path
    summary: dict[str, Any]
    trace: dict[str, np.ndarray]
    history: dict[str, np.ndarray]
    checkpoint: dict[str, np.ndarray]

    @property
    def name(self) -> str:
        return str(self.summary["name"])

    @property
    def mode(self) -> str:
        return str(self.summary["mode"])

    @property
    def height(self) -> float:
        return float(self.summary["height"])

    @property
    def weight(self) -> float:
        return float(self.summary["smooth_weight"])

    @property
    def failure(self) -> Any:
        return self.summary.get("failure")


def _data_dir(value: str) -> Path:
    path = Path(value.strip())
    if not value.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"expected a nonempty data-relative directory: {value!r}")
    return GROUP / "data" / path


def _split_dirs(value: str) -> list[Path]:
    return [_data_dir(part) for part in value.split(",") if part.strip()]


def _numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty trace: {path}")
    required = {
        "step",
        "normalized_loss",
        "objective_normalized",
        "roughness",
        "tensor_neighbor_rms",
        "min_J",
    }
    if required - rows[0].keys():
        raise ValueError(
            f"trace lacks fields {sorted(required - rows[0].keys())}: {path}"
        )
    result = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }
    if not all(np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError(f"trace contains a non-finite numeric value: {path}")
    steps = result["step"]
    if np.any(np.diff(steps) <= 0) or not np.all(steps == np.round(steps)):
        raise ValueError(f"trace steps must be strictly increasing integers: {path}")
    np.testing.assert_allclose(
        result["tensor_neighbor_rms"],
        np.sqrt(np.maximum(result["roughness"], 0.0)),
        rtol=2e-6,
        atol=1e-12,
        err_msg=f"roughness RMS mismatch in {path}",
    )
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]) for name in saved.files}


def _load_cases(directories: list[Path]) -> list[Case]:
    cases: list[Case] = []
    names: set[str] = set()
    for source in directories:
        summary_path = source / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        cherries.log_input(summary_path)
        summaries = json.loads(summary_path.read_text())
        if not isinstance(summaries, list):
            raise TypeError(f"summary must be a list: {summary_path}")
        for summary in summaries:
            required = {
                "name",
                "mode",
                "height",
                "smooth_weight",
                "accepted_iterations",
                "failure",
                "initial",
                "final",
            }
            if required - summary.keys():
                raise ValueError(
                    f"summary row lacks {sorted(required - summary.keys())}: {summary_path}"
                )
            name = str(summary["name"])
            if name in names:
                raise ValueError(
                    f"duplicate case name across input directories: {name}"
                )
            names.add(name)
            folder = source / name
            trace_path = folder / "trace.csv"
            if not trace_path.is_file():
                raise FileNotFoundError(trace_path)
            trace = _numeric_csv(trace_path)
            history = _load_npz(folder / "history.npz")
            checkpoint = _load_npz(folder / "checkpoint.npz")
            cherries.log_input(trace_path)
            cherries.log_input(folder / "history.npz")
            cherries.log_input(folder / "checkpoint.npz")
            cases.append(Case(source, summary, trace, history, checkpoint))
    return cases


def _select(cases: list[Case], weight: float) -> list[Case]:
    chosen: list[Case] = []
    for height in HEIGHTS:
        for mode in MODES:
            for candidate_weight in (0.0, weight):
                matches = [
                    case
                    for case in cases
                    if case.mode == mode
                    and np.isclose(case.height, height, rtol=0.0, atol=1e-12)
                    and np.isclose(case.weight, candidate_weight, rtol=0.0, atol=1e-12)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "expected exactly one case for "
                        f"height={height:g}, mode={mode}, weight={candidate_weight:g}; "
                        f"found {len(matches)}"
                    )
                chosen.append(matches[0])
    return chosen


def _case(cases: list[Case], height: float, mode: str, weight: float) -> Case:
    return next(
        case
        for case in cases
        if case.mode == mode
        and np.isclose(case.height, height, rtol=0.0, atol=1e-12)
        and np.isclose(case.weight, weight, rtol=0.0, atol=1e-12)
    )


def _history_contract(case: Case) -> tuple[np.ndarray, ...]:
    required = {
        "points",
        "triangles",
        "muscle",
        "top",
        "u",
        "controls",
        "steps",
        "height",
        "mode",
        "smooth_weight",
    }
    if required - case.history.keys():
        raise ValueError(
            f"history lacks {sorted(required - case.history.keys())}: {case.name}"
        )
    h = case.history
    points = np.asarray(h["points"], dtype=np.float64)
    triangles = np.asarray(h["triangles"], dtype=np.int64)
    muscle = np.asarray(h["muscle"], dtype=bool)
    top = np.asarray(h["top"], dtype=np.int64)
    frames = np.asarray(h["u"], dtype=np.float64)
    controls = np.asarray(h["controls"], dtype=np.float64)
    steps = np.asarray(h["steps"], dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"triangles must have shape (n,3): {case.name}")
    if points.ndim != 2 or points.shape[1] != 2 or frames.shape[1:] != points.shape:
        raise ValueError(f"invalid 2D point/displacement arrays: {case.name}")
    if muscle.shape != (len(triangles),):
        raise ValueError(f"muscle mask has an incompatible shape: {case.name}")
    if (
        top.ndim != 1
        or len(top) == 0
        or np.any(top < 0)
        or np.any(top >= len(points))
        or len(np.unique(top)) != len(top)
    ):
        raise ValueError(f"top must contain unique in-range node indices: {case.name}")
    if len(frames) != len(steps) or len(frames) == 0 or np.any(np.diff(steps) <= 0):
        raise ValueError(f"invalid explicit history steps: {case.name}")
    if controls.shape[0] != len(steps):
        raise ValueError(
            f"history controls and steps have different lengths: {case.name}"
        )
    if int(steps[-1]) != int(case.trace["step"][-1]):
        raise ValueError(f"history and trace end at different steps: {case.name}")
    if not np.isclose(float(np.asarray(h["height"]).item()), case.height):
        raise ValueError(f"history and summary heights disagree: {case.name}")
    if str(np.asarray(h["mode"]).item()) != case.mode:
        raise ValueError(f"history and summary modes disagree: {case.name}")
    if not np.isclose(float(np.asarray(h["smooth_weight"]).item()), case.weight):
        raise ValueError(
            f"history and summary smoothness weights disagree: {case.name}"
        )
    return points, triangles, muscle, top, frames, steps


def _failure_text(case: Case) -> str:
    if not case.failure:
        return ""
    if isinstance(case.failure, str):
        return case.failure
    return json.dumps(case.failure, sort_keys=True)


def _final_value(case: Case, key: str) -> float:
    final = case.summary["final"]
    if not isinstance(final, dict) or key not in final:
        raise ValueError(f"final summary lacks {key!r}: {case.name}")
    return float(final[key])


def _target(
    points: np.ndarray, top: np.ndarray, height: float
) -> tuple[np.ndarray, np.ndarray]:
    top_points = points[top]
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    y0 = float(np.median(top_points[:, 1]))
    x = np.linspace(x0, x1, 401)
    unit = (x - x0) / (x1 - x0)
    return x, y0 + 4.0 * height * unit * (1.0 - unit)


def _mark_failure(axis: Axes, case: Case) -> None:
    if not case.failure:
        return
    for spine in axis.spines.values():
        spine.set_color("#a71930")
        spine.set_linewidth(2.2)
    axis.text(
        0.99,
        0.03,
        "FAILED — last recorded state",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#a71930",
        fontsize=8,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "#a71930", "alpha": 0.9},
    )


def _deformation_limits(
    cases: list[Case], height: float
) -> tuple[float, float, float, float]:
    clouds: list[np.ndarray] = []
    reference: tuple[np.ndarray, np.ndarray] | None = None
    for case in cases:
        if not np.isclose(case.height, height):
            continue
        values = _history_contract(case)
        points, _, _, top, frames, _ = values
        clouds.extend((points, points + frames[-1]))
        reference = points, top
    if reference is None:
        raise ValueError(f"no saved geometry for height={height:g}")
    target = np.column_stack((*_target(*reference, height),))
    clouds.append(target)
    all_points = np.concatenate(clouds)
    span = np.ptp(all_points, axis=0)
    pad = 0.035 * max(float(span.max()), 1e-6)
    return (
        float(all_points[:, 0].min() - pad),
        float(all_points[:, 0].max() + pad),
        float(all_points[:, 1].min() - pad),
        float(all_points[:, 1].max() + pad),
    )


def _draw_deformation(axis: Axes, case: Case, limits: tuple[float, ...]) -> None:
    values = _history_contract(case)
    points, triangles, muscle, top, frames, steps = values
    deformed = points + frames[-1]
    colors = np.where(
        muscle[:, None], np.array([[0.85, 0.50, 0.45]]), np.array([[0.91, 0.85, 0.75]])
    )
    axis.add_collection(
        PolyCollection(
            deformed[triangles],
            facecolors=colors,
            edgecolors=EDGE_COLOR,
            linewidths=0.12,
            rasterized=True,
        )
    )
    x, y = _target(points, top, case.height)
    axis.plot(x, y, "k--", lw=1.0, label="target")
    final = case.summary["final"]
    fit = float(final["fit_rms"])
    min_j = float(final["min_J"])
    axis.text(
        0.02,
        0.04,
        f"last step {int(steps[-1])} · fit RMS {fit:.3g}\nmin J {min_j:.3g}",
        transform=axis.transAxes,
        fontsize=7.5,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86},
    )
    axis.set(xlim=limits[:2], ylim=limits[2:])
    axis.set_aspect("equal", adjustable="box")
    _mark_failure(axis, case)


def _save_pair(fig: plt.Figure, output: Path, stem: str) -> list[str]:
    names = []
    for suffix in ("png", "pdf"):
        name = f"{stem}.{suffix}"
        fig.savefig(output / name, dpi=220, bbox_inches="tight")
        names.append(name)
    plt.close(fig)
    return names


def _render_deformations(
    cases: list[Case], height: float, weight: float, output: Path
) -> list[str]:
    limits = _deformation_limits(cases, height)
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(11.0, 5.8 if np.isclose(height, 0.05) else 10.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for row, mode in enumerate(MODES):
        for col, candidate_weight in enumerate((0.0, weight)):
            case = _case(cases, height, mode, candidate_weight)
            axis = axes[row, col]
            _draw_deformation(axis, case, limits)
            if row == 0:
                axis.set_title(
                    "smoothness off" if col == 0 else f"smoothness on (w={weight:g})"
                )
            if col == 0:
                if np.isclose(height, 0.05):
                    axis.set_ylabel(
                        ACTIVATION_LABELS[mode],
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=12,
                        fontsize=9,
                    )
                else:
                    axis.set_ylabel(LABELS[mode])
            if row == len(MODES) - 1:
                axis.set_xlabel("x")
    fig.suptitle(
        f"Final saved deformations · target height h={height:g}\n"
        "Common physical axes; dashed curve is the target; failure panels stop at their last record",
        fontsize=14,
    )
    return _save_pair(fig, output, f"deformation-h{round(height * 1000):03d}")


def _checkpoint_b(case: Case, ntriangles: int) -> np.ndarray:
    required = {"controls", "u", "B"}
    if required - case.checkpoint.keys():
        raise ValueError(
            f"checkpoint lacks {sorted(required - case.checkpoint.keys())}: {case.name}"
        )
    b = np.asarray(case.checkpoint["B"], dtype=np.float64)
    if b.shape != (ntriangles, 2, 2) or not np.all(np.isfinite(b)):
        raise ValueError(
            f"B must contain one finite 2x2 matrix per element: {case.name}"
        )
    return b


def _activation_scale(cases: list[Case]) -> tuple[float, float]:
    absolute: list[np.ndarray] = []
    edge_lengths: list[np.ndarray] = []
    for case in cases:
        values = _history_contract(case)
        points, triangles, muscle, _, _, _ = values
        b = _checkpoint_b(case, len(triangles))
        z = 0.5 * (b + np.swapaxes(b, 1, 2)) - np.eye(2)
        absolute.append(np.abs(np.linalg.eigvalsh(z[muscle])).ravel())
        tri = points[triangles[muscle]]
        edge_lengths.append(np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1))
    if not absolute:
        raise ValueError("no selected checkpoint contains B for activation rendering")
    maximum = float(np.max(np.concatenate(absolute)))
    typical_edge = float(np.median(np.concatenate(edge_lengths)))
    if maximum <= 0.0 or typical_edge <= 0.0:
        raise ValueError("activation/elements must have positive display scales")
    return maximum, 1.4 * typical_edge


def _deformation_gradient(reference: np.ndarray, deformed: np.ndarray) -> np.ndarray:
    """Triangle edge columns give F=Ds inv(Dm), matching the saved physics."""
    dm = np.stack(
        (reference[:, 1] - reference[:, 0], reference[:, 2] - reference[:, 0]), axis=-1
    )
    ds = np.stack(
        (deformed[:, 1] - deformed[:, 0], deformed[:, 2] - deformed[:, 0]), axis=-1
    )
    return ds @ np.linalg.inv(dm)


def _transport_axes(F: np.ndarray, axes: np.ndarray) -> np.ndarray:
    transported = F @ axes
    norm = np.linalg.norm(transported, axis=1)
    assert np.all(np.isfinite(norm))
    assert np.all(norm > 0)
    return transported / norm[:, None, :]


def _activation_geometry(case: Case) -> dict[str, np.ndarray]:
    points, triangles, muscle, _, frames, _ = _history_contract(case)
    np.testing.assert_array_equal(frames[-1], case.checkpoint["u"])
    deformed = points + frames[-1]
    F = _deformation_gradient(points[triangles], deformed[triangles])
    np.testing.assert_allclose(
        np.linalg.det(F).min(), _final_value(case, "min_J"), rtol=1e-11, atol=1e-13
    )
    B = _checkpoint_b(case, len(triangles))
    eigenvalues, reference_axes = np.linalg.eigh(
        0.5 * (B + B.swapaxes(1, 2))[muscle] - np.eye(2)
    )
    return {
        "points": deformed,
        "triangles": triangles,
        "muscle": muscle,
        "centers": deformed[triangles[muscle]].mean(axis=1),
        "reference_centers": points[triangles[muscle]].mean(axis=1),
        "eigenvalues": eigenvalues,
        "reference_axes": reference_axes,
        "directions": _transport_axes(F[muscle], reference_axes),
        "F": F[muscle],
    }


def _activation_limits(
    cases: list[Case], height: float, *, zoom: bool
) -> tuple[float, float, float, float]:
    clouds = []
    for case in cases:
        if not np.isclose(case.height, height):
            continue
        geometry = _activation_geometry(case)
        if zoom:
            reference_x = geometry["reference_centers"][:, 0]
            chosen = (reference_x >= 0.25) & (reference_x <= 0.75)
            clouds.append(
                geometry["points"][
                    geometry["triangles"][geometry["muscle"]][chosen]
                ].reshape(-1, 2)
            )
        else:
            clouds.append(geometry["points"])
    values = np.concatenate(clouds)
    padding = 0.012
    lower, upper = values.min(axis=0) - padding, values.max(axis=0) + padding
    return float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])


def _draw_activation(
    axis: Axes,
    case: Case,
    maximum: float,
    max_length: float,
    limits: tuple[float, float, float, float],
) -> None:
    geometry = _activation_geometry(case)
    points, triangles, muscle = (
        geometry[key] for key in ("points", "triangles", "muscle")
    )
    axis.add_collection(
        PolyCollection(
            points[triangles],
            facecolors=np.where(
                muscle[:, None],
                np.array([[0.97, 0.94, 0.91]]),
                np.array([[0.97, 0.97, 0.96]]),
            ),
            edgecolors=np.where(muscle, "#b4aaa2", "#dedbd7"),
            linewidths=np.where(muscle, 0.18, 0.08),
            rasterized=True,
        )
    )
    values_eig, centers, directions = (
        geometry[key] for key in ("eigenvalues", "centers", "directions")
    )
    color_scale = mpl.colors.Normalize(vmin=-maximum, vmax=maximum)
    color_map = mpl.colormaps["RdBu_r"]
    for eigenmode in (0, 1):
        strength = np.abs(values_eig[:, eigenmode]) / maximum
        half = 0.5 * max_length * directions[:, :, eigenmode]
        segments = np.stack((centers - half, centers + half), axis=1)
        visible = strength > 1e-12
        if np.any(visible):
            axis.add_collection(
                LineCollection(
                    segments[visible],
                    colors=color_map(color_scale(values_eig[visible, eigenmode])),
                    linewidths=0.45,
                    capstyle="butt",
                    rasterized=True,
                )
            )
    axis.set(xlim=limits[:2], ylim=limits[2:])
    axis.set_aspect("equal", adjustable="box")
    axis.text(
        0.02,
        0.96,
        f"step {int(case.trace['step'][-1])} · max |λ(B−I)| {np.max(np.abs(values_eig)):.3g}",
        transform=axis.transAxes,
        fontsize=7,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86},
    )
    _mark_failure(axis, case)


def _render_activations(
    cases: list[Case],
    height: float,
    weight: float,
    output: Path,
    maximum: float,
    max_length: float,
    *,
    zoom: bool = False,
) -> list[str]:
    limits = _activation_limits(cases, height, zoom=zoom)
    aspect = (limits[3] - limits[2]) / (limits[1] - limits[0])
    figure_height = max(4.5, 4 * 4.4 * aspect + 2.0)
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(12, figure_height),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for row, mode in enumerate(MODES):
        for col, candidate_weight in enumerate((0.0, weight)):
            case = _case(cases, height, mode, candidate_weight)
            axis = axes[row, col]
            _draw_activation(axis, case, maximum, max_length, limits)
            if row == 0:
                axis.set_title(
                    "smoothness off" if col == 0 else f"smoothness on (w={weight:g})"
                )
            if col == 0:
                axis.set_ylabel(
                    ACTIVATION_LABELS[mode],
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                    fontsize=9,
                )
            if row == len(MODES) - 1:
                axis.set_xlabel("deformed x")
    fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=-maximum, vmax=maximum), cmap="RdBu_r"
        ),
        ax=axes,
        orientation="horizontal",
        shrink=0.55,
        aspect=45,
        pad=0.06,
        label="Signed activation λ(B−I) · shared scale across both targets",
    )
    view = "central muscle close-up" if zoom else "whole shape"
    fig.suptitle(
        f"Activation on deformed shape · h={height:g} · {view}\n"
        "Equal-length thin axes follow F n / ||F n||; color shows signed activation strength",
        fontsize=13,
    )
    stem = f"activation{'-zoom' if zoom else ''}-h{round(height * 1000):03d}"
    return _save_pair(fig, output, stem)


def _glyph_audit(cases: list[Case], output: Path) -> dict[str, Any]:
    angle = 0.4
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    expected_F = rotation @ np.array([[1.0, 0.35], [0.0, 1.0]])
    reference = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    translation = np.array([0.3, -0.2])
    deformed = reference @ expected_F.T + translation
    recovered_F = _deformation_gradient(reference, deformed)
    np.testing.assert_allclose(recovered_F[0], expected_F, rtol=0, atol=1e-14)
    axes = np.eye(2)[None]
    expected_directions = expected_F / np.linalg.norm(expected_F, axis=0)[None, :]
    np.testing.assert_allclose(
        _transport_axes(recovered_F, axes)[0], expected_directions, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        deformed.mean(axis=1)[0],
        reference.mean(axis=1)[0] @ expected_F.T + translation,
        atol=1e-14,
    )
    records = []
    for case in cases:
        geometry = _activation_geometry(case)
        displacement = geometry["centers"] - geometry["reference_centers"]
        record = {
            "case": case.name,
            "muscle_triangles": len(geometry["centers"]),
            "maximum_center_displacement": float(
                np.linalg.norm(displacement, axis=1).max()
            ),
            "direction_unit_norm_max_error": float(
                np.abs(np.linalg.norm(geometry["directions"], axis=1) - 1).max()
            ),
            "minimum_muscle_J": float(np.linalg.det(geometry["F"]).min()),
        }
        assert record["maximum_center_displacement"] > 0
        assert record["direction_unit_norm_max_error"] < 1e-12
        np.savez_compressed(output / f"{case.name}-glyphs.npz", **geometry)
        records.append(record)
    result = {
        "rotation_shear_translation_gate": "passed",
        "geometry": "saved deformed triangle centers",
        "direction": "normalize(F @ reference_eigenvector)",
        "strength": "unchanged signed eigenvalue of B-I",
        "cases": records,
    }
    (output / "glyph-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _render_histories(cases: list[Case], weight: float, output: Path) -> list[str]:
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 7.5), sharex="col", layout="constrained"
    )
    for col, height in enumerate(HEIGHTS):
        for mode in MODES:
            for candidate_weight, style in ((0.0, "-"), (weight, "--")):
                case = _case(cases, height, mode, candidate_weight)
                step = case.trace["step"]
                for row, key in enumerate(("normalized_loss", "tensor_neighbor_rms")):
                    axis = axes[row, col]
                    axis.plot(
                        step,
                        case.trace[key],
                        color=COLORS[mode],
                        ls=style,
                        lw=1.55,
                        label=f"{LABELS[mode]} · {'off' if candidate_weight == 0 else 'on'}",
                    )
                    marker = "X" if case.failure else "o"
                    axis.scatter(
                        step[-1],
                        case.trace[key][-1],
                        color=COLORS[mode],
                        marker=marker,
                        s=34,
                        zorder=4,
                    )
                    if case.failure:
                        axis.annotate(
                            "failed",
                            (step[-1], case.trace[key][-1]),
                            xytext=(3, 3),
                            textcoords="offset points",
                            fontsize=6.5,
                            color="#a71930",
                        )
        axes[0, col].set_title(f"Target height h={height:g}")
        axes[1, col].set_xlabel("accepted iteration (recorded states only)")
        for row in range(2):
            axes[row, col].grid(alpha=0.2)
    axes[0, 0].set_ylabel("pure fit: mean top error² / h²")
    axes[1, 0].set_ylabel("tensor-neighbor RMS = √roughness")
    handles = [
        Line2D([0], [0], color=COLORS[mode], lw=2, label=LABELS[mode]) for mode in MODES
    ]
    handles.extend(
        (
            Line2D([0], [0], color="0.25", ls="-", label="smoothness off"),
            Line2D(
                [0], [0], color="0.25", ls="--", label=f"smoothness on (w={weight:g})"
            ),
            Line2D([0], [0], color="0.25", marker="X", lw=0, label="failed endpoint"),
        )
    )
    fig.legend(
        handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8.5
    )
    fig.suptitle("Optimization histories stop at the final recorded state", fontsize=14)
    return _save_pair(fig, output, "loss-and-roughness")


def _stability_checks(path: Path) -> dict[str, float]:
    cherries.log_input(path)
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        raise TypeError(f"verification checks must contain a case list: {path}")
    result: dict[str, float] = {}
    for record in document["cases"]:
        name = str(record["case"])
        if name in result:
            raise ValueError(f"duplicate verification check: {name}")
        result[name] = float(record["smallest_algebraic_hessian_eigenvalue"])
    return result


def _render_tuning(
    cases: list[Case], output: Path, stability: dict[str, float]
) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    weight_markers = {0.0: "o", 0.01: "s", 0.1: "D", 1.0: "*"}
    any_point = False
    for axis, height in zip(axes, HEIGHTS, strict=True):
        for mode in MODES:
            group = sorted(
                [
                    case
                    for case in cases
                    if case.mode == mode and np.isclose(case.height, height)
                ],
                key=lambda case: case.weight,
            )
            points: list[tuple[float, float, Case]] = []
            for case in group:
                try:
                    fit = _final_value(case, "normalized_loss")
                    rms = np.sqrt(max(_final_value(case, "roughness"), 0.0))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fit) and np.isfinite(rms):
                    points.append((fit, rms, case))
            if not points:
                continue
            any_point = True
            x = np.asarray([point[1] for point in points])
            y = np.asarray([point[0] for point in points])
            axis.plot(x, y, color=COLORS[mode], lw=1.0, alpha=0.75)
            for fit, rms, case in points:
                xx, yy = rms, fit
                assert xx == np.sqrt(_final_value(case, "roughness"))
                assert yy == _final_value(case, "normalized_loss")
                unstable = stability[case.name] < 0.0
                if unstable:
                    axis.scatter(
                        xx,
                        yy,
                        facecolors="none",
                        edgecolors="#a71930",
                        marker="^",
                        linewidths=1.5,
                        s=62,
                        zorder=4,
                    )
                else:
                    axis.scatter(
                        xx,
                        yy,
                        color="#a71930" if case.failure else COLORS[mode],
                        marker="X" if case.failure else weight_markers[case.weight],
                        s=42 if case.failure else 45,
                        zorder=4,
                    )
                label = f"w={case.weight:g}"
                if unstable:
                    label += " (unstable)"
                if unstable or case.failure:
                    axis.annotate(
                        label,
                        (xx, yy),
                        xytext=(-105, 12) if unstable else (5, 5),
                        textcoords="offset points",
                        fontsize=7,
                    )
        axis.set(
            title=f"Target height h={height:g}",
            xscale="log",
            xlabel="final tensor-neighbor RMS (log scale)",
            ylabel="final pure normalized fit loss",
        )
        axis.grid(alpha=0.2)
    if not any_point:
        plt.close(fig)
        return []
    handles = [
        Line2D([0], [0], color=COLORS[mode], marker="o", label=LABELS[mode])
        for mode in MODES
    ]
    handles.extend(
        (
            Line2D(
                [0],
                [0],
                color="#a71930",
                marker="^",
                markerfacecolor="none",
                lw=0,
                label="verified unstable equilibrium",
            ),
            Line2D(
                [0],
                [0],
                color="#a71930",
                marker="X",
                lw=0,
                label="forward failed",
            ),
        )
    )
    handles.extend(
        Line2D([0], [0], color="0.35", marker=marker, lw=0, label=f"weight {weight:g}")
        for weight, marker in weight_markers.items()
    )
    fig.legend(
        handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=8
    )
    fig.suptitle(
        "Smoothness weight sweep · pure fitting loss versus activation variation",
        fontsize=14,
    )
    return _save_pair(fig, output, "tuning-tradeoff")


def _write_gallery(
    output: Path,
    figures: list[tuple[str, list[str], str]],
    cases: list[Case],
    weight: float,
) -> None:
    sections = []
    for title, files, caption in figures:
        png = next(name for name in files if name.endswith(".png"))
        pdf = next((name for name in files if name.endswith(".pdf")), None)
        pdf_link = f' · <a href="{html.escape(pdf)}">PDF</a>' if pdf else ""
        sections.append(
            f"<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}{pdf_link}</p>"
            f'<a href="{html.escape(png)}"><img src="{html.escape(png)}" alt="{html.escape(title)}"></a></section>'
        )
    failures = [case for case in cases if case.failure]
    failure_items = (
        "".join(
            f"<li><code>{html.escape(case.name)}</code>: {html.escape(_failure_text(case))}</li>"
            for case in failures
        )
        or "<li>None among selected cases.</li>"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Activation direction smoothness figures</title><style>
body{{font:16px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:2rem;color:#24211f;background:#faf9f6}}
h1,h2{{line-height:1.2}} section{{margin:2.5rem 0;padding-top:1rem;border-top:1px solid #d8d0c8}}
img{{display:block;width:100%;height:auto;background:white;border:1px solid #d8d0c8}} code{{background:#eee9e3;padding:.1rem .25rem}}
.note{{background:#f0ece6;padding:1rem 1.2rem;border-left:4px solid #7d6d60}} a{{color:#315f88}}
</style></head><body><h1>Activation direction × smoothness study</h1>
<p class="note">Selected smoothness weight: {weight:g}. Curves contain recorded states only. Failed runs are explicitly marked and are never extended. These finite optimization histories do not establish convergence.</p>
{"".join(sections)}<section><h2>Selected-run failures</h2><ul>{failure_items}</ul></section>
</body></html>
"""
    (output / "index.html").write_text(document)


def main(cfg: Config) -> None:
    if cfg.selected_weight <= 0.0:
        raise ValueError("selected_weight must be positive so off and on are distinct")
    input_dirs = _split_dirs(cfg.input_dirs)
    if not input_dirs:
        raise ValueError("input_dirs must select at least one directory")
    all_cases = _load_cases(input_dirs)
    selected = _select(all_cases, cfg.selected_weight)
    for case in selected:
        _history_contract(case)

    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)

    maximum, max_length = _activation_scale(selected)
    _glyph_audit(selected, output)
    figures: list[tuple[str, list[str], str]] = []
    for height in HEIGHTS:
        suffix = f"h={height:g}"
        figures.append(
            (
                f"Deformation comparison · {suffix}",
                _render_deformations(selected, height, cfg.selected_weight, output),
                "Equal physical axes across all eight panels.",
            )
        )
        figures.append(
            (
                f"Activation comparison · {suffix}",
                _render_activations(
                    selected, height, cfg.selected_weight, output, maximum, max_length
                ),
                "Both activation modes are placed at deformed triangle centers, with axes transported as F n / ||F n||. Thin segments have equal length for readable orientation; their color encodes the signed reference eigenvalue on one shared scale. These are transported material axes, not spatial stress eigenvectors.",
            )
        )
        figures.append(
            (
                f"Activation close-up · {suffix}",
                _render_activations(
                    selected,
                    height,
                    cfg.selected_weight,
                    output,
                    maximum,
                    max_length,
                    zoom=True,
                ),
                "Same transported activation glyphs on a common deformed-space crop of the central muscle patch; equal physical axes across all eight panels.",
            )
        )
    figures.append(
        (
            "Pure fit and roughness histories",
            _render_histories(selected, cfg.selected_weight, output),
            "Solid lines are smoothness off; dashed lines use the selected weight.",
        )
    )

    tuning_dirs = _split_dirs(cfg.tuning_dirs) if cfg.tuning_dirs.strip() else []
    stability: dict[str, float] = {}
    if tuning_dirs:
        tuning_cases = _load_cases(tuning_dirs)
        checks_path = GROUP / "data/21-regularized-checks/checks.json"
        if not checks_path.is_file():
            raise FileNotFoundError(checks_path)
        stability = _stability_checks(checks_path)
        stability.update(
            _stability_checks(GROUP / "data/20-baseline-checks/checks.json")
        )
        tuning_files = _render_tuning(tuning_cases, output, stability)
        if tuning_files:
            figures.append(
                (
                    "Tuning tradeoff",
                    tuning_files,
                    "Marker shapes indicate smoothness weight; the hollow triangle is an equilibrium with a verifier-confirmed negative smallest Hessian eigenvalue, distinct from a failed forward solve.",
                )
            )

    _write_gallery(output, figures, selected, cfg.selected_weight)
    manifest = {
        "input_dirs": [str(path) for path in input_dirs],
        "tuning_dirs": [str(path) for path in tuning_dirs],
        "selected_weight": cfg.selected_weight,
        "modes": list(MODES),
        "heights": list(HEIGHTS),
        "activation_definition": "both eigenpairs of sym(B)-I at deformed muscle triangle centers; directions normalize(F @ n)",
        "activation_signs": {
            "positive": "positive eigenmode of B-I",
            "negative": "negative eigenmode of B-I; depending on the model this permits extension or indicates an invalid activation matrix",
        },
        "activation_interpretation": "signed strengths describe saved B-I; material axes transported with F, not spatial stress eigenvectors; repeated eigenvalues have nonunique axes",
        "activation_geometry": "whole deformed shape plus common central-muscle close-up",
        "activation_glyph_style": {
            "full_length": max_length,
            "length_scale": "1.4 times median reference muscle edge; constant across strengths and cases",
            "linewidth_points": 0.45,
            "color": "RdBu_r; signed reference eigenvalue; shared linear symmetric range",
            "color_range": [-maximum, maximum],
            "hidden_modes": "absolute eigenvalue / global maximum <= 1e-12",
        },
        "activation_limits": {
            str(height): {
                "whole": _activation_limits(selected, height, zoom=False),
                "zoom": _activation_limits(selected, height, zoom=True),
            }
            for height in HEIGHTS
        },
        "global_max_abs_activation_eigenvalue": maximum,
        "failure_policy": "traces stop at the last recorded state; failed endpoints and panels are marked; no curve extension",
        "convergence_claim": "none; figures report finite saved histories and endpoints",
        "verified_unstable_tuning_endpoints": {
            name: eigenvalue
            for name, eigenvalue in stability.items()
            if eigenvalue < 0.0
        },
        "selected_cases": [
            {
                "name": case.name,
                "mode": case.mode,
                "height": case.height,
                "smooth_weight": case.weight,
                "accepted_iterations": case.summary["accepted_iterations"],
                "failure": case.failure,
                "last_trace_step": int(case.trace["step"][-1]),
            }
            for case in selected
        ],
        "figures": [{"title": title, "files": files} for title, files, _ in figures],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    cherries.log_metrics(
        {
            "figures": len(figures),
            "selected_cases": len(selected),
            "failed_cases": sum(bool(case.failure) for case in selected),
        }
    )
    LOG.info("Wrote %d figure groups and gallery to %s", len(figures), output)


if __name__ == "__main__":
    cherries.main(main, profile=None if os.environ.get("DEBUG") else ProfileFigures)
