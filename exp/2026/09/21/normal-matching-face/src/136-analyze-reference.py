"""Analyze the fixed 2 mm / 5 degree reference-loss face fits."""

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

METRICS = (
    "fit_rms_mm",
    "normal_angle_rms_deg",
    "surface_gradient_rms",
    "primary_union_normal_residual_highpass_5mm_rms_mm",
    "activation_smoothness",
    "motion_rms_mm",
    "detF_min",
    "inverted_all_cells",
)
PLOT = (
    ("Position RMS (mm)", "fit_rms_mm", False),
    ("Normal angle RMS (deg)", "normal_angle_rms_deg", False),
    ("Surface-gradient RMS", "surface_gradient_rms", False),
    (
        "5 mm high-pass residual (mm)",
        "primary_union_normal_residual_highpass_5mm_rms_mm",
        False,
    ),
    ("Activation variation R", "activation_smoothness", False),
    ("Physical gradient RMS / initial", "physical_gradient_rms", True),
    ("Own objective / own neutral", "objective", True),
    ("Minimum det(F)", "detF_min", False),
    ("Inverted cells", "inverted_all_cells", False),
)
OLD_SPECS = {
    "smooth-off-beta0": ("40-continuation", "smooth-off-l2", "L2 only | smooth off"),
    "smooth-off-beta05": (
        "40-continuation",
        "smooth-off-normal",
        "beta .05 | smooth off",
    ),
    "smooth-off-beta25": (
        "60-strong-normal",
        "smooth-off-normal",
        "beta .25 | smooth off",
    ),
    "smooth-off-beta1": ("90-beta1", "smooth-off-normal", "beta 1 | smooth off"),
    "smooth-on-beta0": ("40-continuation", "smooth-on-l2", "L2 only | smooth on"),
    "smooth-on-beta05": ("40-continuation", "smooth-on-normal", "beta .05 | smooth on"),
    "smooth-on-beta25": (
        "60-strong-normal",
        "smooth-on-normal",
        "beta .25 | smooth on",
    ),
    "smooth-on-beta1": ("90-beta1", "smooth-on-normal", "beta 1 | smooth on"),
}
COLORS = {
    "beta0": "#4d4a46",
    "beta05": "#c05621",
    "beta25": "#6a3d9a",
    "beta1": "#008b8b",
    "reference": "#d62728",
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    reference_dir: Path = Path("132-reference-continuation")
    reference_off_branch: str = "smooth-off-normal"
    reference_on_branch: str = "smooth-on-normal"
    beta1_dir: Path = Path("90-beta1")
    strong_dir: Path = Path("60-strong-normal")
    baseline_dir: Path = Path("40-continuation")
    loss_config: Path = Path("110-shape-loss-config/loss-config.json")
    verification: Path = Path("135-verification/checks.json")
    output: Path = Path("136-analysis")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def trace(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    assert rows
    assert [int(row["step"]) for row in rows] == list(range(len(rows)))
    return rows


def saved(path: Path) -> set[int]:
    values = {
        int(match.group(1))
        for file in path.glob("step-*.npz")
        if (match := re.fullmatch(r"step-(\d+)\.npz", file.name))
    }
    assert values
    return values


def first_inversion(rows: list[dict[str, float]]) -> int | None:
    return next(
        (int(row["step"]) for row in rows if row["inverted_all_cells"] > 0), None
    )


def tail_change(rows: list[dict[str, float]], count: int) -> dict[str, float | int]:
    before, after = rows[max(0, len(rows) - 1 - count)], rows[-1]
    return {
        "from_step": int(before["step"]),
        "to_step": int(after["step"]),
        "objective_percent_change": 100
        * (after["objective"] - before["objective"])
        / before["objective"],
    }


def family(name: str) -> str:
    if name.startswith("reference"):
        return "reference"
    if "beta25" in name:
        return "beta25"
    if "beta05" in name:
        return "beta05"
    if "beta1" in name:
        return "beta1"
    return "beta0"


def main(cfg: Config) -> None:
    audit = read_json(cherries.input(cfg.verification))
    assert audit["passed"] is True
    assert audit["all_completed"] is True
    loss_config = read_json(cherries.input(cfg.loss_config))
    assert loss_config["l_ref_mm"] == 13.236093032531715
    assert loss_config["normal_weight"] == 1.0

    roots = {
        "40-continuation": cherries.input(cfg.baseline_dir),
        "60-strong-normal": cherries.input(cfg.strong_dir),
        "90-beta1": cherries.input(cfg.beta1_dir),
        "132-reference-continuation": cherries.input(cfg.reference_dir),
    }
    specs = {
        **OLD_SPECS,
        "reference-off": (
            "132-reference-continuation",
            cfg.reference_off_branch,
            "2 mm / 5 deg | smooth off",
        ),
        "reference-on": (
            "132-reference-continuation",
            cfg.reference_on_branch,
            "2 mm / 5 deg | smooth on",
        ),
    }
    rows = {
        name: trace(roots[root] / branch / "trace.csv")
        for name, (root, branch, _) in specs.items()
    }
    summaries = {
        name: read_json(roots[root] / branch / "summary.json")
        for name, (root, branch, _) in specs.items()
    }
    for name, values in rows.items():
        assert int(summaries[name]["last_step"]) == int(values[-1]["step"])
        assert summaries[name]["status"] == "completed_budget_not_convergence_certified"
    for name, (_, branch, _) in specs.items():
        if name.startswith("reference"):
            assert int(audit["branches"][branch]["last_step"]) == int(
                rows[name][-1]["step"]
            )
            assert "position_contribution" in rows[name][0]

    common = min(int(values[-1]["step"]) for values in rows.values())
    common_saved = set.intersection(
        *(saved(roots[root] / branch) for root, branch, _ in specs.values())
    )
    inversion_free = [
        step
        for step in range(common + 1)
        if all(values[step]["inverted_all_cells"] == 0 for values in rows.values())
    ]
    saved_inversion_free = [
        step
        for step in common_saved
        if step <= common
        and all(values[step]["inverted_all_cells"] == 0 for values in rows.values())
    ]

    report: dict[str, Any] = {
        "gate": {"path": str(cfg.verification), "passed": True, "all_completed": True},
        "loss": {
            "label": "2 mm / 5 deg fixed-reference data loss",
            "l_ref_mm": loss_config["l_ref_mm"],
            "normal_weight": loss_config["normal_weight"],
            "formula": "position_component_mse_mm2 / l_ref_mm**2 + normal_chord_squared + alpha / l_ref_mm**2 * R",
        },
        "provenance": {
            "reference_off": "132-reference-continuation smooth-off: fresh neutral start through update 102 from 130-reference-fit, then resumed from saved q/m/v/counter after an unexpected process stop; replay uses finite tolerances and is not a bitwise-continuity claim",
            "reference_on": "132-reference-continuation smooth-on: fresh neutral Raw6 run through the full 200-update budget",
            "reference_interruption": "130-reference-fit smooth-off stopped unexpectedly after accepted update 102 with no solver failure or graceful-shutdown record; preserve it as incomplete, not a completed fit",
            "beta25": "60-strong-normal: two fresh neutral Raw6 runs, continuous 200-update Adam trajectories",
            "beta1": "90-beta1: two fresh neutral Raw6 runs, continuous 200-update Adam trajectories",
            "beta0_beta05": "40-continuation: paused after update 100 and resumed with saved Adam state; do not treat as uninterrupted fresh trajectories",
            "source_paths": {
                "baseline": str(cfg.baseline_dir),
                "strong": str(cfg.strong_dir),
                "beta1": str(cfg.beta1_dir),
                "reference": str(cfg.reference_dir),
            },
        },
        "latest_common_trace_step": common,
        "latest_common_saved_step": max(
            step for step in common_saved if step <= common
        ),
        "latest_common_inversion_free_trace_step": max(inversion_free, default=None),
        "latest_common_inversion_free_saved_step": max(
            saved_inversion_free, default=None
        ),
        "variants": {},
        "note": "Objective magnitudes and normalized objective curves use branch-specific scalings and are only within-branch convergence diagnostics; unnormalized shape and physical metrics are comparable.",
    }
    for name, values in rows.items():
        endpoint = values[-1]
        at100 = values[100] if len(values) > 100 else None
        details: dict[str, Any] = {
            "label": specs[name][2],
            "endpoint_step": int(endpoint["step"]),
            "endpoint": {key: endpoint[key] for key in METRICS},
            "step_100": None if at100 is None else {key: at100[key] for key in METRICS},
            "latest_common": {key: values[common][key] for key in METRICS},
            "first_inversion_step": first_inversion(values),
            "last10_objective_change": tail_change(values, 10),
            "last25_objective_change": tail_change(values, 25),
            "physical_gradient_ratio_vs_neutral": endpoint["physical_gradient_rms"]
            / values[0]["physical_gradient_rms"],
            "physical_gradient_ratio_vs_100": None
            if at100 is None
            else endpoint["physical_gradient_rms"] / at100["physical_gradient_rms"],
        }
        if name.startswith("reference"):
            details["data_contributions"] = {
                "neutral": {
                    key: values[0][key]
                    for key in (
                        "position_contribution",
                        "normal_contribution",
                        "regularizer_contribution",
                    )
                },
                "endpoint": {
                    key: endpoint[key]
                    for key in (
                        "position_contribution",
                        "normal_contribution",
                        "regularizer_contribution",
                    )
                },
            }
        report["variants"][name] = details

    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    figure, axes = plt.subplots(3, 3, figsize=(16, 13))
    for panel, (axis, (title, key, normalized)) in enumerate(
        zip(axes.flat, PLOT, strict=True)
    ):
        for name, values in rows.items():
            y = (
                [row[key] / values[0][key] for row in values]
                if normalized
                else [row[key] for row in values]
            )
            axis.plot(
                [row["step"] for row in values],
                y,
                color=COLORS[family(name)],
                linestyle="--" if "off" in name else "-",
                label=specs[name][2] if panel == 0 else "_nolegend_",
            )
        axis.set(title=title, xlabel="Adam update")
        if key == "detF_min":
            axis.axhline(0, color="black", linewidth=1, alpha=0.7)
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(
        "Fixed 2 mm / 5 deg reference loss versus neutral-normalized beta controls. Objective curves are within-branch diagnostics.",
        fontsize=11,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=5,
        fontsize=7,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(out / "reference-loss-histories.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
