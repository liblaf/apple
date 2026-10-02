# ruff: noqa: EM101, EM102, SLF001, TRY003
"""Redraw the two three-rate report plots with readable diagnostic scales."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pydantic_settings as ps
from experiment_profile import ProfileCometNoCommit

mpl.use("Agg")
from matplotlib import pyplot as plt

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
RATE58_SOURCE = GROUP / "src/58-compare-rate-03.py"
RATE58_OUTPUT = GROUP / "data/58-rate03-comparison"
OUTPUT = GROUP / "data/59-rate03-report-figures"
TRACE_PATHS = {
    "original-on": GROUP / "data/25-learned-axis-smooth/trace.csv",
    "quarter-on": GROUP / "data/49-axis-on-lr-quarter-64/trace.csv",
    "rate03-on": GROUP / "data/56-axis-on-lr03-256/trace.csv",
}


class Config(cherries.BaseConfig):
    """Fresh destination for the report-only redraw."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = OUTPUT


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _digest(resolved),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _load_rate58() -> Any:
    specification = importlib.util.spec_from_file_location("rate58", RATE58_SOURCE)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load frozen plotting helpers: {RATE58_SOURCE}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _plot_trajectories(
    rate58: Any,
    traces: dict[str, list[dict[str, Any]]],
    output: Path,
) -> Path:
    figure, axes = plt.subplots(4, 2, figsize=(12, 17), constrained_layout=True)
    panels = (
        (axes[0, 0], "fit_rms_mm", "Fit RMS (mm)", "Target fit", None),
        (axes[0, 1], "motion_rms_mm", "Motion RMS (mm)", "Expression motion", None),
        (
            axes[1, 0],
            "inverted_all_cells",
            "Inverted tetrahedra (symmetric log; linear ±1)",
            "Mechanical inversions",
            1.0,
        ),
        (axes[1, 1], "detF_min", "Minimum det(F)", "Minimum volume ratio", None),
        (
            axes[2, 0],
            rate58.PRIMARY_RESIDUAL,
            "Union residual HP RMS (mm)",
            "Primary surface score",
            None,
        ),
        (
            axes[2, 1],
            "smoothness_C",
            "S(C) (symmetric log; linear ±1)",
            "Activation-axis spatial variation",
            1.0,
        ),
        (
            axes[3, 0],
            "geometry_update_face_vector_rms_mm",
            "Face geometry update RMS (mm)",
            "Physical geometry step",
            None,
        ),
        (
            axes[3, 1],
            "z_update_frobenius_rms",
            "Activation Z update RMS (symmetric log; linear ±0.01)",
            "Activation step",
            0.01,
        ),
    )
    for axis, key, ylabel, title, linthresh in panels:
        for run in rate58.RUNS:
            rate58._plot_curve(axis, traces[run], "step", key, run)
        if linthresh is not None:
            axis.set_yscale("symlog", linthresh=linthresh)
        axis.set(xlabel="Adam updates", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle(
        "Smoothed learned-axis arms · color, dash, and marker encode learning rate",
        fontsize=13,
    )
    path = output / "optimizer-trajectories.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _plot_attained_states(
    rate58: Any,
    traces: dict[str, list[dict[str, Any]]],
    output: Path,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for run in rate58.RUNS:
        rate58._plot_curve(
            axes[0, 0],
            traces[run],
            "motion_rms_mm",
            rate58.PRIMARY_RESIDUAL,
            run,
        )
        rate58._plot_curve(
            axes[0, 1],
            traces[run],
            "fit_rms_mm",
            rate58.PRIMARY_RESIDUAL,
            run,
        )
        rate58._plot_curve(
            axes[1, 0], traces[run], "motion_rms_mm", "inverted_all_cells", run
        )
        rate58._plot_curve(
            axes[1, 1], traces[run], "fit_rms_mm", "inverted_all_cells", run
        )
    axes[0, 0].set(
        xlabel="Motion RMS (mm)",
        ylabel="Union residual HP RMS (mm)",
        title="Surface score versus global motion",
    )
    axes[0, 1].set(
        xlabel="Fit RMS (mm)",
        ylabel="Union residual HP RMS (mm)",
        title="Surface score versus global fit",
    )
    axes[1, 0].set(
        xlabel="Motion RMS (mm)",
        ylabel="Inverted tetrahedra (symmetric log; linear ±1)",
        title="Inversions versus global motion",
    )
    axes[1, 1].set(
        xlabel="Fit RMS (mm)",
        ylabel="Inverted tetrahedra (symmetric log; linear ±1)",
        title="Inversions versus global fit",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    for axis in axes[1]:
        axis.set_yscale("symlog", linthresh=1.0)
    figure.suptitle(
        "Smoothed learned-axis arms · comparisons at global fit and motion",
        fontsize=13,
    )
    path = output / "global-fit-motion-comparison.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _main(cfg: Config) -> dict[str, Any]:
    output = cfg.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary58_path = RATE58_OUTPUT / "summary.json"
    required = [RATE58_SOURCE, summary58_path, *TRACE_PATHS.values()]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    summary58 = json.loads(summary58_path.read_text())
    if summary58["status"] != "completed_saved_state_three_rate_comparison":
        raise ValueError("frozen three-rate comparison did not complete")
    if _digest(RATE58_SOURCE) != str(
        summary58["source"]["live_at_generation"]["sha256"]
    ):
        raise ValueError("executed three-rate plotting source changed")
    rate58 = _load_rate58()
    if summary58["styles"] != rate58.STYLES:
        raise ValueError("executed three-rate plot styles changed")
    for path in TRACE_PATHS.values():
        receipt = summary58["inputs"].get(str(path.resolve()))
        if receipt is None or _digest(path) != str(receipt["sha256"]):
            raise ValueError(f"trace differs from executed comparison receipt: {path}")
    traces = {run: rate58._read_trace(path) for run, path in TRACE_PATHS.items()}
    plots = [
        _plot_trajectories(rate58, traces, output),
        _plot_attained_states(rate58, traces, output),
    ]
    snapshot = output / "sources/59-render-rate-03-report-plots.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), snapshot)
    snapshot.chmod(0o444)
    if _digest(snapshot) != _digest(Path(__file__)):
        raise ValueError("saved report renderer differs from live source")
    summary = {
        "status": "completed_rate03_report_plot_redraw",
        "scope": (
            "Pure Matplotlib redraw from the exact traces and styles receipted by "
            "the completed data/58 comparison; no analysis, fitting, interpolation, "
            "geometry change, or raster recoloring"
        ),
        "scale_changes": {
            "inverted_all_cells": {"scale": "symlog", "linthresh": 1.0},
            "smoothness_C": {"scale": "symlog", "linthresh": 1.0},
            "z_update_frobenius_rms": {"scale": "symlog", "linthresh": 0.01},
        },
        "inputs": {str(path.resolve()): _record(path) for path in required},
        "outputs": {str(path.relative_to(output)): _record(path) for path in plots},
        "source": {
            "snapshot": _record(snapshot),
            "live_at_generation": _record(Path(__file__)),
        },
    }
    _write_json(output / "summary.json", summary)
    return summary


def main(cfg: Config) -> None:
    summary = _main(cfg)
    cherries.log_metric("render/figures", len(summary["outputs"]))
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(
        main,
        profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit,
    )
