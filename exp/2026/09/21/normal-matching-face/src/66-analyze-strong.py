# Copyright (c) 2026 liblaf
"""Compare frozen neutral-start normal weights without comparing objectives."""

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
SPECS = {
    "smooth-off-beta0": ("40-continuation", "smooth-off-l2", "beta 0 | smooth off"),
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
    "smooth-on-beta0": ("40-continuation", "smooth-on-l2", "beta 0 | smooth on"),
    "smooth-on-beta05": ("40-continuation", "smooth-on-normal", "beta .05 | smooth on"),
    "smooth-on-beta25": (
        "60-strong-normal",
        "smooth-on-normal",
        "beta .25 | smooth on",
    ),
}
COLORS = {"beta0": "#4d4a46", "beta05": "#c05621", "beta25": "#6a3d9a"}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    strong_dir: Path = Path("60-strong-normal")
    baseline_dir: Path = Path("40-continuation")
    verification: Path = Path("65-verification/checks.json")
    output: Path = Path("66-analysis")


def load(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        r = [{k: float(v) for k, v in x.items()} for x in csv.DictReader(f)]
    assert r
    assert [int(x["step"]) for x in r] == list(range(len(r)))
    return r


def saved(p: Path) -> set[int]:
    x = {
        int(m.group(1))
        for q in p.glob("step-*.npz")
        if (m := re.fullmatch(r"step-(\d+)\.npz", q.name))
    }
    assert x
    return x


def first(r: list[dict[str, float]]) -> int | None:
    return next((int(x["step"]) for x in r if x["inverted_all_cells"] > 0), None)


def tail(r: list[dict[str, float]], n: int) -> dict[str, float | int]:
    a, b = r[max(0, len(r) - 1 - n)], r[-1]
    return {
        "from_step": int(a["step"]),
        "to_step": int(b["step"]),
        "objective_percent_change": 100
        * (b["objective"] - a["objective"])
        / a["objective"],
    }


def main(c: Config) -> None:
    gate = json.loads(cherries.input(c.verification).read_text())
    assert gate["passed"] is True
    roots = {
        "40-continuation": cherries.input(c.baseline_dir),
        "60-strong-normal": cherries.input(c.strong_dir),
    }
    traces = {n: load(roots[d] / b / "trace.csv") for n, (d, b, _l) in SPECS.items()}
    summaries = {
        n: json.loads((roots[d] / b / "summary.json").read_text())
        for n, (d, b, _l) in SPECS.items()
    }
    for n, r in traces.items():
        assert int(summaries[n]["last_step"]) == int(r[-1]["step"])
        if n.endswith("beta25"):
            assert int(gate["branches"][SPECS[n][1]]["last_step"]) == int(r[-1]["step"])
    common = min(int(r[-1]["step"]) for r in traces.values())
    valid = [
        s
        for s in range(common + 1)
        if all(r[s]["inverted_all_cells"] == 0 for r in traces.values())
    ]
    common_saved = set.intersection(
        *(saved(roots[d] / b) for d, b, _ in SPECS.values())
    )
    valid_saved = [
        s
        for s in common_saved
        if s <= common and all(r[s]["inverted_all_cells"] == 0 for r in traces.values())
    ]
    valid_step = max(valid, default=None)
    report: dict[str, Any] = {
        "gate": {
            "passed": True,
            "all_completed": gate.get("all_completed"),
            "path": str(c.verification),
        },
        "provenance": {
            "baseline": "40-continuation: beta 0 L2 and beta .05 normal branches paused at update 100 then continued with saved Adam state",
            "strong": "60-strong-normal: fresh neutral starts at beta .25",
            "source_paths": {
                "baseline": str(c.baseline_dir),
                "strong": str(c.strong_dir),
            },
        },
        "latest_common_trace_step": common,
        "latest_common_saved_step": max(s for s in common_saved if s <= common),
        "latest_common_inversion_free_trace_step": valid_step,
        "latest_common_inversion_free_saved_step": max(valid_saved, default=None),
        "variants": {},
        "common_metrics": {
            "latest_common_trace": {
                k: {n: r[common][k] for n, r in traces.items()} for k in METRICS
            },
            "latest_common_inversion_free_trace": None
            if valid_step is None
            else {k: {n: r[valid_step][k] for n, r in traces.items()} for k in METRICS},
        },
        "note": "Objectives are reported only as within-branch tail changes and are not compared across normal weights.",
    }
    for n, r in traces.items():
        e = r[-1]
        at100 = r[100] if len(r) > 100 else None
        report["variants"][n] = {
            "label": SPECS[n][2],
            "status": summaries[n]["status"],
            "failure": summaries[n]["failure"],
            "endpoint_step": int(e["step"]),
            "endpoint": {k: e[k] for k in METRICS},
            "step_100": None if at100 is None else {k: at100[k] for k in METRICS},
            "latest_common": {k: r[common][k] for k in METRICS},
            "first_inversion_step": first(r),
            "last10_objective_change": tail(r, 10),
            "last25_objective_change": tail(r, 25),
            "physical_gradient_ratio_vs_neutral": e["physical_gradient_rms"]
            / r[0]["physical_gradient_rms"],
            "physical_gradient_ratio_vs_100": None
            if at100 is None
            else e["physical_gradient_rms"] / at100["physical_gradient_rms"],
        }
    out = cherries.output(c.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    for a, (title, key, normalized) in zip(ax.flat, PLOT, strict=True):
        for n, r in traces.items():
            beta = (
                "beta25" if "beta25" in n else ("beta05" if "beta05" in n else "beta0")
            )
            y = [x[key] / r[0][key] for x in r] if normalized else [x[key] for x in r]
            a.plot(
                [x["step"] for x in r],
                y,
                color=COLORS[beta],
                linestyle="--" if "smooth-off" in n else "-",
                label=SPECS[n][2],
            )
        a.set(title=title, xlabel="Adam update")
        if key == "detF_min":
            a.axhline(0, color="black", linewidth=1, alpha=0.7)
        a.grid(alpha=0.25)
        a.legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Fresh-neutral beta .25 versus beta 0/.05 continuation; fixed budget is not convergence. "
        "Own-objective curves are within-branch diagnostics, not cross-beta rankings.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "strong-normal-histories.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
