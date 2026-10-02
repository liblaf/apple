"""Render exact accepted states from the fiber-contraction comparison.

This script never re-solves, interpolates, or smooths inverse states.  It is
deliberately separate from ``10-run-inverse.py`` so presentation assets remain
traceable to the saved ``history.npz`` inputs.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
from liblaf.cherries import core, plugins, profiles
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection

from liblaf import cherries

LOG = logging.getLogger(__name__)

MODES = ("x_contraction", "unconstrained")
MODE_LABELS = {
    "x_contraction": "fiber x-contraction",
    "unconstrained": "free symmetric B",
}
MATERIAL_COLORS = np.array(["#d9b382", "#bb5c50"])


class ProfileRecord(profiles.Profile):
    """Record a normal Comet run without creating a Git commit."""

    def init(self):
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_dir: Path = Path("10-comparison-final")
    output_dir: Path = cherries.output("20-figures", mkdir=True)
    fps: int = 20
    dpi: int = 180


@dataclass(frozen=True)
class Case:
    name: str
    points: np.ndarray
    triangles: np.ndarray
    muscle: np.ndarray
    top: np.ndarray
    u: np.ndarray
    controls: np.ndarray
    height: float
    mode: str
    trace: tuple[dict[str, str], ...]
    summary: dict

    @property
    def final_points(self) -> np.ndarray:
        return self.points + self.u[-1]


def case_name(height: float, mode: str) -> str:
    return f"h{round(height * 1000):03d}-{mode}"


def load_case(input_dir: Path, height: float, mode: str) -> Case:
    folder = input_dir / case_name(height, mode)
    archive = np.load(folder / "history.npz", allow_pickle=False)
    stored_mode = str(archive["mode"].item())
    assert stored_mode == mode, (folder, stored_mode, mode)
    stored_height = float(archive["height"].item())
    assert np.isclose(stored_height, height), (folder, stored_height, height)
    with (folder / "trace.csv").open(newline="") as handle:
        trace = tuple(csv.DictReader(handle))
    assert len(trace) == len(archive["u"]), folder
    return Case(
        name=folder.name,
        points=np.asarray(archive["points"]),
        triangles=np.asarray(archive["triangles"]),
        muscle=np.asarray(archive["muscle"], dtype=bool),
        top=np.asarray(archive["top"], dtype=np.int64),
        u=np.asarray(archive["u"]),
        controls=np.asarray(archive["controls"]),
        height=stored_height,
        mode=mode,
        trace=trace,
        summary=json.loads((folder / "summary.json").read_text()),
    )


def target(points: np.ndarray, height: float) -> np.ndarray:
    x = points[:, 0]
    return height * 4.0 * x * (1.0 - x)


def add_material_mesh(ax: Axes, case: Case, step: int) -> None:
    vertices = (case.points + case.u[step])[case.triangles]
    collection = PolyCollection(
        vertices,
        facecolors=MATERIAL_COLORS[case.muscle.astype(int)],
        edgecolors="#4a3934",
        linewidths=0.12,
    )
    ax.add_collection(collection)


def set_shape_axes(ax: Axes, cases: tuple[Case, Case]) -> None:
    # A video must keep one common view of every accepted state.  The target
    # is an absolute top coordinate: y=0.1+u_y, not merely its displacement.
    all_points = np.concatenate(
        [(case.points[None] + case.u).reshape(-1, 2) for case in cases]
    )
    xmin = min(0.0, float(all_points[:, 0].min()))
    xmax = max(1.0, float(all_points[:, 0].max()))
    ymax = max(float(all_points[:, 1].max()), *(0.1 + case.height for case in cases))
    ymin = min(0.0, float(all_points[:, 1].min()))
    pad_x = max(0.025, 0.04 * (xmax - xmin))
    pad_y = max(0.012, 0.08 * (ymax - ymin))
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (deformed coordinate)")
    ax.set_ylabel("absolute y")


def render_shape(case_a: Case, case_b: Case, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharex=True, sharey=True)
    reference_x = np.linspace(0.0, 1.0, 401)
    for axis, case in zip(axes, (case_a, case_b), strict=True):
        add_material_mesh(axis, case, -1)
        axis.plot(
            reference_x,
            0.1 + target(np.c_[reference_x, reference_x], case.height),
            "k--",
            lw=1.2,
            label="target top",
        )
        axis.set_title(MODE_LABELS[case.mode])
        axis.legend(loc="upper right", frameon=False)
    for axis in axes:
        set_shape_axes(axis, (case_a, case_b))
    fig.text(
        0.5,
        0.01,
        "tan: passive material; red: muscle; deformation shown at absolute scale",
        ha="center",
    )
    fig.suptitle(f"Parabolic top-displacement target, h={case_a.height:g}")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def render_profiles(case_a: Case, case_b: Case, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharex=True, sharey=True)
    x = case_a.points[case_a.top, 0]
    target_y = target(case_a.points[case_a.top], case_a.height)
    for axis, case in zip(axes, (case_a, case_b), strict=True):
        top_u = case.u[-1, case.top]
        axis.plot(x, target_y, "k--", lw=1.5, label="target $u_y$")
        axis.plot(x, top_u[:, 1], color="#a43e36", lw=1.8, label="final $u_y$")
        axis.plot(x, top_u[:, 0], color="#366a94", lw=1.2, label="final $u_x$")
        axis.axhline(0.0, color="0.7", lw=0.7)
        axis.set_title(MODE_LABELS[case.mode])
        axis.set_xlabel("reference x")
        axis.set_ylabel("top displacement")
        axis.legend(frameon=False)
    fig.suptitle(f"Free-top displacement profiles, h={case_a.height:g}")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def fit_series(case: Case) -> np.ndarray:
    return np.array([float(row["fit_rms"]) for row in case.trace])


def render_fit(cases: tuple[Case, ...], output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=False)
    for axis, height in zip(axes, (0.05, 0.20), strict=True):
        for mode, color in zip(MODES, ("#a43e36", "#366a94"), strict=True):
            case = next(
                item for item in cases if item.height == height and item.mode == mode
            )
            values = fit_series(case)
            axis.plot(
                np.arange(len(values)),
                values,
                color=color,
                marker="o",
                ms=2.5,
                lw=1.4,
                label=MODE_LABELS[mode],
            )
        axis.set_title(f"h={height:g}")
        axis.set_xlabel("accepted inverse-state index")
        axis.set_ylabel("top fit RMS")
        axis.legend(frameon=False)
    fig.suptitle("Exact accepted-state fit history")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def control_magnitude(case: Case) -> np.ndarray:
    if case.mode == "x_contraction":
        return case.controls[-1]
    values = case.controls[-1].reshape(-1, 3)
    return np.sqrt(values[:, 0] ** 2 + values[:, 1] ** 2 + 2.0 * values[:, 2] ** 2)


def final_b_metrics(case: Case) -> dict[str, float | int]:
    """Describe the saved final active-distortion inverse B on muscle cells."""
    if case.mode == "x_contraction":
        controls = case.controls[-1]
        matrices = np.zeros((len(controls), 2, 2))
        matrices[:, 0, 0] = 1.0 + controls
        matrices[:, 1, 1] = 1.0
    else:
        controls = case.controls[-1].reshape(-1, 3)
        matrices = np.zeros((len(controls), 2, 2))
        matrices[:, 0, 0] = 1.0 + controls[:, 0]
        matrices[:, 1, 1] = 1.0 + controls[:, 1]
        matrices[:, 0, 1] = controls[:, 2]
        matrices[:, 1, 0] = controls[:, 2]
    determinant = np.linalg.det(matrices)
    eigenvalue = np.linalg.eigvalsh(matrices)
    singular_value = np.linalg.svd(matrices, compute_uv=False)
    return {
        "muscle_triangles": len(matrices),
        "min_det_B": float(determinant.min()),
        "det_B_nonpositive_fraction": float(np.mean(determinant <= 0.0)),
        "min_eigenvalue_symmetric_B": float(eigenvalue[:, 0].min()),
        "min_singular_B": float(singular_value.min()),
        "max_singular_B": float(singular_value.max()),
    }


def render_controls(case_a: Case, case_b: Case, output: Path, dpi: int) -> None:
    values = (control_magnitude(case_a), control_magnitude(case_b))
    vmax = max(float(value.max()) for value in values)
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 3.4), sharex=True, sharey=True, layout="constrained"
    )
    for axis, case, value in zip(axes, (case_a, case_b), values, strict=True):
        field = np.zeros(len(case.triangles))
        field[case.muscle] = value
        polygons = PolyCollection(
            case.points[case.triangles],
            array=field,
            cmap="magma",
            clim=(0.0, vmax),
            edgecolors="none",
        )
        axis.add_collection(polygons)
        axis.set_title(MODE_LABELS[case.mode])
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 0.1)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("reference x")
    axes[0].set_ylabel("reference y")
    colorbar = fig.colorbar(
        polygons, ax=axes, orientation="horizontal", shrink=0.72, pad=0.18
    )
    colorbar.set_label("$||B-I||_F$ on muscle triangles")
    fig.suptitle(f"Final per-muscle-triangle control magnitude, h={case_a.height:g}")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def render_video(case_a: Case, case_b: Case, output: Path, fps: int, dpi: int) -> None:
    """Render matching saved state indices, holding a completed run at its end."""
    total = max(len(case_a.u), len(case_b.u))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharex=True, sharey=True)
    reference_x = np.linspace(0.0, 1.0, 401)

    def draw(index: int) -> None:
        for axis, case in zip(axes, (case_a, case_b), strict=True):
            axis.clear()
            saved_index = min(index, len(case.u) - 1)
            add_material_mesh(axis, case, saved_index)
            axis.plot(
                reference_x,
                0.1 + target(np.c_[reference_x, reference_x], case.height),
                "k--",
                lw=1.1,
            )
            set_shape_axes(axis, (case_a, case_b))
            status = "final held" if saved_index != index else "accepted state"
            axis.set_title(f"{MODE_LABELS[case.mode]}: {status} {saved_index}")
        fig.suptitle(
            f"h={case_a.height:g}; shared requested accepted-state index {index}"
        )

    writer = animation.FFMpegWriter(
        fps=fps, metadata={"title": f"h={case_a.height:g} comparison"}
    )
    with writer.saving(fig, str(output), dpi=dpi):
        for index in range(total):
            draw(index)
            writer.grab_frame()
    plt.close(fig)


def main(cfg: Config) -> None:
    input_dir = cherries.input(cfg.input_dir)
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    source_copy = output_dir / "source"
    source_copy.mkdir()
    shutil.copy2(Path(__file__), source_copy / Path(__file__).name)

    cases = tuple(
        load_case(input_dir, height, mode) for height in (0.05, 0.20) for mode in MODES
    )
    figures = []
    for height in (0.05, 0.20):
        pair = tuple(item for item in cases if item.height == height)
        assert len(pair) == 2
        suffix = f"h{round(height * 1000):03d}"
        for renderer, stem in (
            (render_shape, "deformation"),
            (render_profiles, "top-profile"),
            (render_controls, "controls"),
        ):
            path = output_dir / f"{stem}-{suffix}.png"
            renderer(*pair, path, cfg.dpi)
            figures.append(path.name)
        video = output_dir / f"evolution-{suffix}.mp4"
        render_video(*pair, video, cfg.fps, cfg.dpi)
        figures.append(video.name)
        cherries.log_metrics(
            {
                f"{suffix}/x_contraction_final_fit_rms": fit_series(pair[0])[-1],
                f"{suffix}/unconstrained_final_fit_rms": fit_series(pair[1])[-1],
            }
        )
    fit_path = output_dir / "fit-history.png"
    render_fit(cases, fit_path, cfg.dpi)
    figures.append(fit_path.name)
    manifest = {
        "input_dir": str(input_dir),
        "inputs": {
            case.name: {"accepted_states": len(case.u), "summary": case.summary}
            for case in cases
        },
        "rendering": "All panels use exact saved accepted states. Videos use the same requested state index; a shorter history is held at its final state.",
        "control_interpretation": {
            "x_contraction": "B=diag(1+a,1), a>=0; this is fiber-x contraction only.",
            "unconstrained": "Free symmetric B is an algebraic comparator. It is not constrained to be positive definite or orientation preserving, so det(B), its sign, and its spectrum must be read before assigning an active-strain interpretation.",
            "caution": "A positive det(F) acceptance rule prevents saved geometric inversions only. It is neither a finite-energy barrier nor proof of physical validity.",
        },
        "figures": figures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metrics = {case.name: final_b_metrics(case) for case in cases}
    (output_dir / "render-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    LOG.info("Wrote %d assets to %s", len(figures), output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileRecord)
