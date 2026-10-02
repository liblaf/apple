"""Plot solver traces for the fixed-direction Smile forward setup comparison."""

# ruff: noqa: C901, EM102, PLR0912, PLR0915, RUF007, TRY003, TRY004

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402

FORCE_GATE_N = 1.0e-4
REQUIRED_COLUMNS = {
    "iteration",
    "phase",
    "elapsed_seconds",
    "energy_j",
    "force_n",
    "inverted_cells",
    "cell_count",
    "inverted_percent",
    "minimum_detf",
}
BRANCHES = (
    (
        "inverse_setup",
        "Inverse setup: no collision, no skin",
        "#1f77b4",
        GROUP / "data/10-inverse-setup-newton",
    ),
    (
        "new_setup",
        "New setup: collision, skin, skin pre-strain",
        "#d62728",
        GROUP / "data/10-new-setup-newton-checkpointed",
    ),
)


class Config(cherries.BaseConfig):
    """Cherries-managed destination for plots derived from numerical traces."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    inverse_setup: Path = GROUP / "data/10-inverse-setup-newton"
    new_setup: Path = GROUP / "data/10-new-setup-newton-checkpointed"
    output: Path = cherries.output("20-comparison", mkdir=True)


@dataclass(frozen=True)
class Branch:
    key: str
    label: str
    color: str
    trace_path: Path
    summary_path: Path
    summary: dict[str, Any]
    iteration: np.ndarray
    elapsed_seconds: np.ndarray
    energy_j: np.ndarray
    force_n: np.ndarray
    inverted_percent: np.ndarray
    cell_count: int
    energy_label: str
    force_gate_n: float
    raw_samples: int
    final_duplicate_excluded: bool

    @property
    def energy_relative_to_initial_j(self) -> np.ndarray:
        return self.energy_j - self.energy_j[0]


def _float(row: dict[str, str], column: str, path: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"{path}: invalid {column!r} value {row.get(column)!r}"
        ) from error
    if not np.isfinite(value):
        raise ValueError(f"{path}: {column!r} must be finite, got {value!r}")
    return value


def _integer(row: dict[str, str], column: str, path: Path) -> int:
    value = _float(row, column, path)
    if not value.is_integer():
        raise ValueError(f"{path}: {column!r} must be integral, got {value!r}")
    return int(value)


def _force_gate(summary: dict[str, Any]) -> float:
    for container in (summary, summary.get("config", {}), summary.get("final", {})):
        if not isinstance(container, dict):
            continue
        for key in (
            "force_gate_N",
            "force_gate_n",
            "force_tolerance_n",
            "force_threshold_n",
        ):
            if key in container:
                value = float(container[key])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"summary {key!r} must be positive and finite")
                return value
    return FORCE_GATE_N


def _energy_label(summary: dict[str, Any]) -> str:
    for container in (
        summary,
        summary.get("material_spec", {}),
        summary.get("config", {}),
    ):
        if not isinstance(container, dict):
            continue
        for key in ("energy_zero", "energy_reference", "energy_definition"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "reported solver potential-energy reference"


def load_branch(*, key: str, label: str, color: str, directory: Path) -> Branch:
    trace_path = directory / "trace.csv"
    summary_path = directory / "summary.json"
    if not trace_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"{key} is incomplete: expected {trace_path} and {summary_path}"
        )
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path}: expected a JSON object")
    status = summary.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"{summary_path}: missing explicit branch status")
    with trace_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{trace_path}: missing CSV header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{trace_path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{trace_path}: no accepted solver samples")

    iterations: list[int] = []
    elapsed: list[float] = []
    energy: list[float] = []
    force: list[float] = []
    inversion: list[float] = []
    cell_counts: set[int] = set()
    for row in rows:
        phase = row["phase"]
        if phase not in {"initial", "pncg", "newton", "final"}:
            raise ValueError(f"{trace_path}: unexpected phase {phase!r}")
        iteration = _integer(row, "iteration", trace_path)
        seconds = _float(row, "elapsed_seconds", trace_path)
        energy_j = _float(row, "energy_j", trace_path)
        force_n = _float(row, "force_n", trace_path)
        inverted = _integer(row, "inverted_cells", trace_path)
        cells = _integer(row, "cell_count", trace_path)
        percent = _float(row, "inverted_percent", trace_path)
        if iteration < 0 or seconds < 0.0 or force_n <= 0.0:
            raise ValueError(
                f"{trace_path}: iteration/time must be nonnegative and force positive"
            )
        if cells <= 0 or not 0 <= inverted <= cells or not 0.0 <= percent <= 100.0:
            raise ValueError(f"{trace_path}: invalid inversion count or percentage")
        expected_percent = 100.0 * inverted / cells
        if not np.isclose(percent, expected_percent, rtol=0.0, atol=1.0e-9):
            raise ValueError(
                f"{trace_path}: inverted_percent {percent} disagrees with "
                f"{inverted}/{cells} ({expected_percent})"
            )
        iterations.append(iteration)
        elapsed.append(seconds)
        energy.append(energy_j)
        force.append(force_n)
        inversion.append(percent)
        cell_counts.add(cells)
    if len(cell_counts) != 1:
        raise ValueError(f"{trace_path}: cell_count changes within a branch")
    if any(
        right <= left
        for left, right in zip(iterations[:-1], iterations[1:], strict=True)
    ):
        raise ValueError(f"{trace_path}: iteration must strictly increase")
    if any(right < left for left, right in zip(elapsed[:-1], elapsed[1:], strict=True)):
        raise ValueError(f"{trace_path}: elapsed_seconds must not decrease")
    final_duplicate_excluded = False
    phases = [row["phase"] for row in rows]
    if phases[-1] == "final":
        if len(rows) < 2 or phases[-2] not in {"initial", "pncg", "newton"}:
            raise ValueError(f"{trace_path}: final sample lacks a prior solver sample")
        same_metrics = all(
            np.isclose(values[-1], values[-2], rtol=0.0, atol=1.0e-12)
            for values in (energy, force, inversion)
        ) and cell_counts == {_integer(rows[-2], "cell_count", trace_path)}
        if not same_metrics:
            raise ValueError(
                f"{trace_path}: final sample differs from its preceding solver sample"
            )
        iterations = iterations[:-1]
        elapsed = elapsed[:-1]
        energy = energy[:-1]
        force = force[:-1]
        inversion = inversion[:-1]
        final_duplicate_excluded = True
    return Branch(
        key=key,
        label=f"{label} ({status.replace('_', ' ')})",
        color=color,
        trace_path=trace_path,
        summary_path=summary_path,
        summary=summary,
        iteration=np.asarray(iterations),
        elapsed_seconds=np.asarray(elapsed),
        energy_j=np.asarray(energy),
        force_n=np.asarray(force),
        inverted_percent=np.asarray(inversion),
        cell_count=cell_counts.pop(),
        energy_label=_energy_label(summary),
        force_gate_n=_force_gate(summary),
        raw_samples=len(rows),
        final_duplicate_excluded=final_duplicate_excluded,
    )


def _caption(branches: tuple[Branch, Branch]) -> str:
    cell_count = branches[0].cell_count
    if any(branch.cell_count != cell_count for branch in branches):
        return (
            "Energy includes each setup's own reference offset. Force = free residual."
        )
    return (
        "Energy includes each setup's own reference offset. "
        f"Inversion: det(F) ≤ 0 / {cell_count:,} tets. Force = free residual."
    )


def plot_metric(
    branches: tuple[Branch, Branch],
    *,
    x_name: str,
    x_label: str,
    metric: str,
    y_label: str,
    title: str,
    output: Path,
    log_y: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.3), layout="constrained")
    for branch in branches:
        x = getattr(branch, x_name)
        y = getattr(branch, metric)
        axis.plot(
            x,
            y,
            "o-",
            color=branch.color,
            linewidth=1.5,
            markersize=3.5,
            label=branch.label,
        )
    if log_y:
        axis.set_yscale("log")
        for branch in branches:
            axis.axhline(
                branch.force_gate_n, color=branch.color, linestyle=":", linewidth=1.0
            )
    axis.set(title=title, xlabel=x_label, ylabel=y_label)
    axis.grid(alpha=0.25, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=8)
    for suffix in (".png", ".svg"):
        figure.savefig(
            output.with_suffix(suffix), dpi=220 if suffix == ".png" else None
        )
    plt.close(figure)


def plot_combined(
    branches: tuple[Branch, Branch],
    *,
    x_name: str,
    x_label: str,
    output: Path,
    relative_energy: bool = False,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 6))
    figure.subplots_adjust(top=0.86, bottom=0.19, wspace=0.28)
    energy_metric = "energy_relative_to_initial_j" if relative_energy else "energy_j"
    energy_ylabel = (
        "E - E(initial) (J)" if relative_energy else "Total potential energy (J)"
    )
    energy_title = (
        "Energy relative to initial state"
        if relative_energy
        else "Total potential energy"
    )
    specs = (
        (energy_metric, energy_ylabel, energy_title),
        ("force_n", "Free force residual (N)", "Free force residual"),
        ("inverted_percent", "Inverted cells (%)", "Inverted cells"),
    )
    for axis, (metric, y_label, title) in zip(axes, specs, strict=True):
        for branch in branches:
            axis.plot(
                getattr(branch, x_name),
                getattr(branch, metric),
                "o-",
                color=branch.color,
                linewidth=1.5,
                markersize=3.2,
                label=branch.label,
            )
        if metric == "force_n":
            axis.set_yscale("log")
            for branch in branches:
                axis.axhline(
                    branch.force_gate_n,
                    color=branch.color,
                    linestyle=":",
                    linewidth=1.0,
                )
        axis.set(title=title, xlabel=x_label, ylabel=y_label)
        axis.grid(alpha=0.25, which="both")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Stage 3 Smile safeguarded-Newton forward comparison", y=0.98, fontsize=14
    )
    figure.text(
        0.5,
        0.02,
        _caption(branches),
        ha="center",
        va="bottom",
        fontsize=7.5,
        wrap=True,
    )
    for suffix in (".png", ".svg", ".pdf"):
        figure.savefig(
            output.with_suffix(suffix),
            dpi=220 if suffix == ".png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def main(cfg: Config) -> None:
    configurations = (
        (*BRANCHES[0][:3], cfg.inverse_setup),
        (*BRANCHES[1][:3], cfg.new_setup),
    )
    branches = tuple(
        load_branch(key=key, label=label, color=color, directory=directory)
        for key, label, color, directory in configurations
    )
    assert len(branches) == 2
    output = cfg.output
    output.mkdir(parents=True, exist_ok=True)
    plot_combined(
        branches,
        x_name="iteration",
        x_label="Accepted Newton iteration",
        output=output / "forward-comparison-by-iteration",
    )
    plot_combined(
        branches,
        x_name="iteration",
        x_label="Accepted Newton iteration",
        output=output / "forward-comparison-relative-energy-by-iteration",
        relative_energy=True,
    )
    plot_combined(
        branches,
        x_name="elapsed_seconds",
        x_label="Elapsed solver time (s)",
        output=output / "forward-comparison-by-time",
    )
    for metric, ylabel, title, name, log_y in (
        (
            "energy_j",
            "Total potential energy (J)",
            "Energy by accepted iteration",
            "energy-by-iteration",
            False,
        ),
        (
            "force_n",
            "Free force residual (N)",
            "Force residual by accepted iteration",
            "force-by-iteration",
            True,
        ),
        (
            "inverted_percent",
            "Inverted cells (%)",
            "Inverted cells by accepted iteration",
            "inversion-by-iteration",
            False,
        ),
    ):
        plot_metric(
            branches,
            x_name="iteration",
            x_label="Accepted Newton iteration",
            metric=metric,
            y_label=ylabel,
            title=title,
            output=output / name,
            log_y=log_y,
        )
    metadata = {
        "schema": "smile-forward-setup-comparison-plot-v1",
        "force_gate_default_n": FORCE_GATE_N,
        "caption": _caption(branches),
        "branches": {
            branch.key: {
                "label": branch.label,
                "trace": str(branch.trace_path.resolve()),
                "summary": str(branch.summary_path.resolve()),
                "samples": int(branch.iteration.size),
                "raw_trace_samples": branch.raw_samples,
                "final_duplicate_excluded_from_plots": branch.final_duplicate_excluded,
                "cell_count": branch.cell_count,
                "energy_reference": branch.energy_label,
                "force_gate_n": branch.force_gate_n,
            }
            for branch in branches
        },
    }
    (output / "plot-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cherries.log_metrics(
        {f"{branch.key}/samples": int(branch.iteration.size) for branch in branches}
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
