# Copyright (c) 2026 liblaf
"""Analyze audited Raw6 Adam continuations without claiming convergence."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pydantic_settings as ps
from experiment import Profile

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

BRANCHES = (
    "smooth-off-l2",
    "smooth-off-normal",
    "smooth-on-l2",
    "smooth-on-normal",
)
LABELS = {
    "smooth-off-l2": "L2 | smooth off",
    "smooth-off-normal": "L2 + normal | smooth off",
    "smooth-on-l2": "L2 | smooth on",
    "smooth-on-normal": "L2 + normal | smooth on",
}
COLORS = {
    "smooth-off-l2": "#4d4a46",
    "smooth-off-normal": "#c05621",
    "smooth-on-l2": "#1769aa",
    "smooth-on-normal": "#9c2c2c",
}
PLOT_METRICS = (
    ("Normal angle RMS (deg)", "normal_angle_rms_deg", False),
    ("Position RMS (mm)", "fit_rms_mm", False),
    (
        "Target-relative 5 mm high-pass residual (mm)",
        "primary_union_normal_residual_highpass_5mm_rms_mm",
        False,
    ),
    ("Activation variation R", "activation_smoothness", False),
    ("Own objective / initial", "objective", True),
    ("Physical gradient RMS / initial", "physical_gradient_rms", True),
)
# These are measurements on the common fit, rather than differently weighted
# objectives. They can therefore support paired comparisons between L2, normal,
# and smoothing variants, including the common unweighted activation-variation R.
COMPARABLE_METRICS = (
    "fit_rms_mm",
    "normal_angle_rms_deg",
    "surface_gradient_rms",
    "primary_union_normal_residual_highpass_5mm_rms_mm",
    "motion_rms_mm",
    "activation_smoothness",
    "detF_min",
    "inverted_all_cells",
)
PERCENT_COMPARABLE_METRICS = tuple(
    key for key in COMPARABLE_METRICS if key not in {"detF_min", "inverted_all_cells"}
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = Path("40-continuation")
    verification: Path = Path("45-verification/checks.json")
    output: Path = Path("46-analysis")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), path
    return value


def _trace(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    assert rows, path
    assert [int(row["step"]) for row in rows] == list(range(len(rows))), path
    return rows


def _saved_steps(folder: Path) -> set[int]:
    result = {
        int(match.group(1))
        for path in folder.glob("step-*.npz")
        if (match := re.fullmatch(r"step-(\d+)\.npz", path.name))
    }
    assert result, folder
    return result


def _first_inversion(rows: list[dict[str, float]]) -> int | None:
    return next(
        (int(row["step"]) for row in rows if row["inverted_all_cells"] > 0), None
    )


def _objective_change(
    rows: list[dict[str, float]], width: int
) -> dict[str, float | int]:
    end = rows[-1]
    start = rows[max(0, len(rows) - 1 - width)]
    return {
        "from_step": int(start["step"]),
        "to_step": int(end["step"]),
        "updates": int(end["step"] - start["step"]),
        "objective_absolute_change": end["objective"] - start["objective"],
        "objective_percent_change": 100
        * (end["objective"] - start["objective"])
        / start["objective"],
    }


def _metrics(row: dict[str, float]) -> dict[str, float]:
    return {key: row[key] for key in COMPARABLE_METRICS}


def _paired(
    reference: dict[str, float], changed: dict[str, float]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in COMPARABLE_METRICS:
        delta = changed[key] - reference[key]
        result[key] = {"absolute_change": delta}
        if key in PERCENT_COMPARABLE_METRICS and reference[key] != 0:
            result[key]["percent_change"] = 100 * delta / reference[key]
    return result


def _curve_plot(
    traces: dict[str, list[dict[str, float]]],
    output: Path,
    latest_common: int,
    latest_shared_valid: int | None,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (title, key, normalized) in zip(axes.flat, PLOT_METRICS, strict=True):
        for branch, rows in traces.items():
            steps = [row["step"] for row in rows]
            baseline = rows[0][key]
            values = (
                [row[key] / baseline for row in rows]
                if normalized
                else [row[key] for row in rows]
            )
            axis.plot(
                steps,
                values,
                color=COLORS[branch],
                linestyle="--" if branch.startswith("smooth-off") else "-",
                label=LABELS[branch],
            )
        axis.axvline(
            100, color="black", linestyle=(0, (1, 2)), linewidth=1.2, alpha=0.8
        )
        axis.set(title=title, xlabel="Adam update")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    validity = "none" if latest_shared_valid is None else str(latest_shared_valid)
    endpoint_states = {
        (int(rows[-1]["step"]), int(rows[-1]["inverted_all_cells"]))
        for rows in traces.values()
    }
    if len(endpoint_states) == 1:
        endpoint, inversions = endpoint_states.pop()
        endpoint_label = (
            f"All branches: {endpoint} updates; {inversions} inversions at endpoint"
        )
    else:
        endpoint_label = "Endpoint states differ; see analysis.json"
    fig.suptitle(
        "Audited Adam continuation histories (black dotted: update 100 boundary)\n"
        "Fixed budget; convergence not certified. "
        f"{endpoint_label}. Latest common step {latest_common}; "
        f"latest shared inversion-free trace step {validity}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output / "continuation-histories.png", dpi=200)
    plt.close(fig)


def main(cfg: Config) -> None:
    source = cherries.input(cfg.comparison_dir)
    gate = _json(cherries.input(cfg.verification))
    assert gate["passed"] is True
    assert isinstance(gate["all_completed"], bool)
    assert set(gate["branches"]) == set(BRANCHES)

    traces = {name: _trace(source / name / "trace.csv") for name in BRANCHES}
    summaries = {name: _json(source / name / "summary.json") for name in BRANCHES}
    for name, rows in traces.items():
        summary = summaries[name]
        assert int(summary["last_step"]) == int(rows[-1]["step"]), name
        assert int(gate["branches"][name]["last_step"]) == int(rows[-1]["step"]), name

    endpoints = {name: int(rows[-1]["step"]) for name, rows in traces.items()}
    latest_common = min(endpoints.values())
    # Check every evaluated row rather than assuming that inversion persists after
    # its first occurrence: the traces need not be monotone in validity.
    shared_inversion_free = [
        step
        for step in range(latest_common + 1)
        if all(rows[step]["inverted_all_cells"] == 0 for rows in traces.values())
    ]
    latest_shared_valid = max(shared_inversion_free, default=None)
    common_saved = set.intersection(*(_saved_steps(source / name) for name in BRANCHES))
    shared_valid_saved = [
        step
        for step in common_saved
        if step <= latest_common
        and all(rows[step]["inverted_all_cells"] == 0 for rows in traces.values())
    ]
    latest_shared_valid_saved = max(shared_valid_saved, default=None)

    report: dict[str, Any] = {
        "gate": {
            "passed": True,
            "all_completed": gate["all_completed"],
            "path": str(cfg.verification),
        },
        "continuation_boundary_step": 100,
        "latest_common_evaluated_step": latest_common,
        "latest_shared_inversion_free_trace_step": latest_shared_valid,
        "latest_shared_inversion_free_saved_step": latest_shared_valid_saved,
        "branches": {},
        "paired_effects": {},
        "interpretation": "A completed fixed update budget is not a convergence certificate.",
    }
    for name, rows in traces.items():
        at_100 = rows[100]
        endpoint = rows[-1]
        neutral_gradient = rows[0]["physical_gradient_rms"]
        report["branches"][name] = {
            "status": summaries[name]["status"],
            "failure": summaries[name]["failure"],
            "endpoint_step": int(endpoint["step"]),
            "endpoint": _metrics(endpoint),
            "step_100": _metrics(at_100),
            "latest_common_evaluated": _metrics(rows[latest_common]),
            "first_inversion_step": _first_inversion(rows),
            "endpoint_inversion_free": endpoint["inverted_all_cells"] == 0,
            "last10_objective_change": _objective_change(rows, 10),
            "last25_objective_change": _objective_change(rows, 25),
            "physical_gradient_ratios": {
                "endpoint_vs_neutral": endpoint["physical_gradient_rms"]
                / neutral_gradient,
                "endpoint_vs_step_100": endpoint["physical_gradient_rms"]
                / at_100["physical_gradient_rms"],
                "latest_common_vs_neutral": rows[latest_common]["physical_gradient_rms"]
                / neutral_gradient,
                "latest_common_vs_step_100": rows[latest_common][
                    "physical_gradient_rms"
                ]
                / at_100["physical_gradient_rms"],
            },
        }
        if latest_shared_valid is not None:
            report["branches"][name]["latest_shared_inversion_free"] = _metrics(
                rows[latest_shared_valid]
            )
        if latest_shared_valid_saved is not None:
            report["branches"][name]["latest_shared_inversion_free_saved"] = _metrics(
                rows[latest_shared_valid_saved]
            )

    comparable_steps: dict[str, int] = {
        "step_100": 100,
        "latest_common_evaluated": latest_common,
    }
    if latest_shared_valid is not None:
        comparable_steps["latest_shared_inversion_free_trace"] = latest_shared_valid
    if latest_shared_valid_saved is not None:
        comparable_steps["latest_shared_inversion_free_saved"] = (
            latest_shared_valid_saved
        )
    pairs = {
        "normal_at_smooth_off": ("smooth-off-l2", "smooth-off-normal"),
        "normal_at_smooth_on": ("smooth-on-l2", "smooth-on-normal"),
        "smoothness_at_l2": ("smooth-off-l2", "smooth-on-l2"),
        "smoothness_at_normal": ("smooth-off-normal", "smooth-on-normal"),
    }
    for label, (reference, changed) in pairs.items():
        report["paired_effects"][label] = {
            "reference": reference,
            "changed": changed,
            "metrics": "shared fit measurements only; excludes differently weighted objectives and includes common unweighted activation variation R",
            "at_step": {
                name: _paired(traces[reference][step], traces[changed][step])
                for name, step in comparable_steps.items()
            },
        }

    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    _curve_plot(traces, output, latest_common, latest_shared_valid)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
