"""Plot saved h=.20 loss traces without importing or running physics.

The plot is a read-only rendering of the exact CSV histories used by the
focused h=.20 materials report.  It intentionally excludes the ``zero_u``
branch and every other geometry, target, material, loss, or mesh variant.
"""

from __future__ import annotations

# ruff: noqa: C901, EM101, EM102, FBT003, ICN001, RUF059, TRY003
import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import pydantic_settings as ps

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from liblaf import cherries

logger = logging.getLogger(__name__)

CASES = (
    ("direct", "h020-direct", "canonical", "-"),
    ("shared", "h020-shared", "canonical", "--"),
    ("shared-release", "h020-shared-release", "NONSTATIONARY/EXPLORATORY", ":"),
)
COLORS = {"direct": "#111111", "shared": "#555555", "shared-release": "#888888"}


class Config(cherries.BaseConfig):
    """Only saved traces and receipts are inputs; no physics module is imported."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    direct_trace: Path = cherries.input("20-canonical-h020/h020-direct/trace.csv")
    direct_summary: Path = cherries.input("20-canonical-h020/h020-direct/summary.json")
    shared_trace: Path = cherries.input("20-canonical-h020/h020-shared/trace.csv")
    shared_summary: Path = cherries.input("20-canonical-h020/h020-shared/summary.json")
    shared_release_trace: Path = cherries.input(
        "40-exploratory-release-h020/h020-shared-release/trace.csv"
    )
    shared_release_summary: Path = cherries.input(
        "40-exploratory-release-h020/h020-shared-release/summary.json"
    )
    output_dir: Path = cherries.output("62-focused-h020-loss-curves", mkdir=True)


@dataclass(frozen=True)
class Trace:
    key: str
    case_name: str
    classification: str
    line_style: str
    trace_path: Path
    summary_path: Path
    steps: tuple[int, ...]
    objective: tuple[float, ...]
    final_objective: float
    nonstationary: bool
    source_digest: dict[str, Any]


def digest(path: Path) -> dict[str, Any]:
    """Return a content identity suitable for the rendering receipt."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def require_number(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_trace(
    key: str,
    case_name: str,
    classification: str,
    line_style: str,
    trace_path: Path,
    summary_path: Path,
) -> Trace:
    """Read a saved trace and prove its rows/endpoints match its receipt."""
    summary = json_object(summary_path)
    case = summary.get("case")
    inverse = summary.get("inverse")
    final = summary.get("metrics", {}).get("final")
    if not isinstance(case, dict) or case.get("name") != case_name:
        raise ValueError(f"wrong case receipt: {summary_path}")
    if not isinstance(inverse, dict) or not isinstance(inverse.get("evaluations"), int):
        raise TypeError(f"missing saved-state count: {summary_path}")
    if not isinstance(final, dict):
        raise TypeError(f"missing final metrics: {summary_path}")
    convergence = inverse.get("convergence")
    tail = inverse.get("tail")
    if not isinstance(convergence, dict) or not isinstance(tail, dict):
        raise TypeError(f"missing stationarity receipt: {summary_path}")
    if convergence.get("practical_stationarity_gate") or tail.get(
        "inverse_converged_1pct_tail_gate"
    ):
        raise ValueError(
            f"this focused report requires a nonstationary path: {summary_path}"
        )

    steps: list[int] = []
    objective: list[float] = []
    with trace_path.open(newline="", encoding="utf-8") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            step = int(row["step"])
            if step != index:
                raise ValueError(
                    f"saved steps must be consecutive from zero: {trace_path}"
                )
            value = require_number(row["objective"], f"objective at row {index}")
            if value <= 0:
                raise ValueError(
                    f"objective must be positive for log plotting: {trace_path}"
                )
            steps.append(step)
            objective.append(value)
    if len(steps) != inverse["evaluations"]:
        raise ValueError(f"row count does not match receipt: {trace_path}")
    if steps[-1] != final.get("step"):
        raise ValueError(f"last saved step does not match receipt: {trace_path}")
    expected_final = require_number(final.get("objective"), "summary final objective")
    if not math.isclose(objective[-1], expected_final, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"final objective does not match receipt: {trace_path}")
    return Trace(
        key=key,
        case_name=case_name,
        classification=classification,
        line_style=line_style,
        trace_path=trace_path,
        summary_path=summary_path,
        steps=tuple(steps),
        objective=tuple(objective),
        final_objective=expected_final,
        nonstationary=True,
        source_digest={"trace": digest(trace_path), "summary": digest(summary_path)},
    )


def plot(traces: tuple[Trace, ...], output: Path) -> None:
    """Create a two-panel monochrome scientific plot from saved objective rows."""
    direct, shared, release = traces
    if release.objective[0] != shared.objective[-1]:
        raise ValueError("release must begin from the recorded shared endpoint")
    release_x = [shared.steps[-1] + step for step in release.steps]

    plt.rcParams.update(
        {
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.linewidth": 0.8,
            "font.size": 10,
            "grid.color": "#d0d0d0",
            "grid.linewidth": 0.6,
            "text.color": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )
    figure, (local, continuation) = plt.subplots(1, 2, figsize=(12.5, 5.3), dpi=180)
    for trace in traces:
        local.plot(
            trace.steps,
            trace.objective,
            color=COLORS[trace.key],
            linestyle=trace.line_style,
            linewidth=1.6,
            label={
                "direct": "direct independent",
                "shared": "shared (3 DoF)",
                "shared-release": "shared → independent (exploratory)",
            }[trace.key],
        )
    local.set(
        title="Local saved-step histories",
        xlabel="saved optimization step",
        ylabel="L2 objective (log scale)",
        yscale="log",
    )
    local.grid(True, which="both")
    local.legend(frameon=False, loc="upper right")

    continuation.plot(
        direct.steps,
        direct.objective,
        color=COLORS[direct.key],
        linestyle=direct.line_style,
        linewidth=1.4,
        alpha=0.8,
        label="direct independent reference",
    )
    continuation.plot(
        shared.steps,
        shared.objective,
        color=COLORS[shared.key],
        linestyle=shared.line_style,
        linewidth=1.6,
        label="shared stage",
    )
    continuation.plot(
        release_x,
        release.objective,
        color=COLORS[release.key],
        linestyle=release.line_style,
        linewidth=1.8,
        label="released stage (exploratory)",
    )
    boundary = shared.steps[-1]
    continuation.axvline(boundary, color="#333333", linewidth=0.9)
    continuation.annotate(
        "release boundary",
        xy=(boundary, shared.objective[-1]),
        xytext=(8, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    continuation.set(
        title="Cumulative shared → release path",
        xlabel="cumulative saved step",
        ylabel="L2 objective (log scale)",
        yscale="log",
    )
    continuation.grid(True, which="both")
    continuation.legend(frameon=False, loc="upper right")

    figure.suptitle("2-D pork h=.20 band muscle: saved L2 objective histories", y=0.985)
    figure.text(
        0.5,
        0.012,
        "All displayed paths are nonstationary saved endpoints; the release path is exploratory. "
        "Plotting only: no forward or inverse physics was run.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.06, 0.98, 0.94))
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def receipt(traces: tuple[Trace, ...], output: Path) -> dict[str, Any]:
    """Build an auditable receipt for the read-only visualization."""
    direct, shared, release = traces
    return {
        "status": "ok",
        "purpose": "visualization only; no forward or inverse physics imports or solves",
        "scope": {
            "height": 0.2,
            "geometry": "long L=1 band muscle",
            "poisson": 0.49,
            "included_cases": [trace.case_name for trace in traces],
            "excluded": ["h020-shared-release_zero_u", "all mismatched experiments"],
        },
        "nonstationary": {trace.case_name: trace.nonstationary for trace in traces},
        "exploratory": [release.case_name],
        "release_handoff": {
            "shared_final_objective": shared.final_objective,
            "release_initial_objective": release.objective[0],
            "exact_objective_match": release.objective[0] == shared.final_objective,
            "release_boundary_cumulative_step": shared.steps[-1],
        },
        "series": [
            {
                "case": trace.case_name,
                "classification": trace.classification,
                "line_style": trace.line_style,
                "rows": len(trace.steps),
                "first_step": trace.steps[0],
                "last_step": trace.steps[-1],
                "final_objective": trace.final_objective,
                "source": trace.source_digest,
            }
            for trace in traces
        ],
        "render_contract": {
            "panels": ["local saved-step comparison", "cumulative shared-to-release"],
            "y_axis": "log L2 objective",
            "style": "monochrome black/dark-gray solid-dashed-dotted lines",
            "metric_colormap": False,
            "physics_run": False,
        },
        "output": digest(output),
        "renderer_source": digest(Path(__file__)),
    }


def main(cfg: Config) -> None:
    """Validate receipts, plot the existing CSV rows, and write the receipt."""
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(f"loss-curve output must be empty: {cfg.output_dir}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        (cfg.direct_trace, cfg.direct_summary),
        (cfg.shared_trace, cfg.shared_summary),
        (cfg.shared_release_trace, cfg.shared_release_summary),
    )
    traces = tuple(
        load_trace(*case, trace_path, summary_path)
        for case, (trace_path, summary_path) in zip(CASES, paths, strict=True)
    )
    output = cfg.output_dir / "loss-curves.png"
    plot(traces, output)
    receipt_path = cfg.output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt(traces, output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cherries.log_metrics(
        {f"{trace.key}/saved_rows": len(trace.steps) for trace in traces}
    )
    logger.info("Wrote %s and %s", output, receipt_path)


if __name__ == "__main__":
    cherries.main(main)
