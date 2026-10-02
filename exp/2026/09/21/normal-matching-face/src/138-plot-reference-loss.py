"""Plot the dimensionless reference-loss components for the completed face fit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pydantic_settings as ps
from experiment import Profile

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

BRANCHES = ("smooth-off-normal", "smooth-on-normal")
COMPONENTS = (
    ("Total objective", "objective"),
    ("Position contribution", "position_contribution"),
    ("Normal contribution", "normal_contribution"),
    ("Regularizer contribution", "regularizer_contribution"),
)
STYLE = {
    "smooth-off-normal": {"color": "#0072b2", "linestyle": "--", "label": "smooth off"},
    "smooth-on-normal": {"color": "#d55e00", "linestyle": "-", "label": "smooth on"},
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = Path("132-reference-continuation")
    verification: Path = Path("135-verification/checks.json")
    output: Path = Path("138-loss-components")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), path
    return value


def load(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = [
            {name: float(value) for name, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    assert rows
    assert [int(row["step"]) for row in rows] == list(range(len(rows)))
    return rows


def check_parts(rows: list[dict[str, float]]) -> float:
    errors = [
        abs(
            row["objective"]
            - row["position_contribution"]
            - row["normal_contribution"]
            - row["regularizer_contribution"]
        )
        for row in rows
    ]
    maximum = max(errors)
    assert maximum <= 2e-15, maximum
    return maximum


def plot(
    traces: dict[str, list[dict[str, float]]], out: Path, *, start: int, title: str
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    for axis, (label, key) in zip(axes.flat, COMPONENTS, strict=True):
        for branch, rows in traces.items():
            shown = rows[start:]
            axis.plot(
                [row["step"] for row in shown],
                [row[key] for row in shown],
                linewidth=2,
                **STYLE[branch],
            )
        axis.set(title=label, xlabel="Adam update", ylabel="dimensionless loss")
        axis.grid(alpha=0.25)
        if key == "regularizer_contribution":
            axis.text(
                0.98,
                0.08,
                "smooth-off regularizer\ncontribution is identically zero",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
            )
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle(title, fontsize=13)
    figure.savefig(
        out / f"loss-components-{'full' if start == 0 else 'last50'}.png", dpi=200
    )
    figure.savefig(out / f"loss-components-{'full' if start == 0 else 'last50'}.pdf")
    plt.close(figure)


def main(cfg: Config) -> None:
    source = cherries.input(cfg.comparison_dir)
    checks_path = cherries.input(cfg.verification)
    checks = read(checks_path)
    assert checks["passed"] is True
    assert checks["all_completed"] is True
    assert checks["source_protocol_record"] == {
        "path": str((source / "protocol.json").resolve()),
        "sha256": digest(source / "protocol.json"),
    }
    protocol = read(source / "protocol.json")
    assert int(protocol["config"]["steps"]) == 200
    traces = {branch: load(source / branch / "trace.csv") for branch in BRANCHES}
    max_errors = {branch: check_parts(rows) for branch, rows in traces.items()}
    for branch, rows in traces.items():
        assert int(rows[-1]["step"]) == 200
        assert checks["branches"][branch]["last_step"] == 200
        if branch == "smooth-off-normal":
            assert all(row["regularizer_contribution"] == 0.0 for row in rows)

    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "source_protocol_record": checks["source_protocol_record"],
        "verification": {
            "path": str(checks_path.resolve()),
            "sha256": digest(checks_path),
        },
        "branches": {
            branch: {
                "last_step": int(rows[-1]["step"]),
                "max_component_sum_absolute_error": max_errors[branch],
                "last50_objective_percent_change": 100
                * (rows[-1]["objective"] - rows[150]["objective"])
                / rows[150]["objective"],
            }
            for branch, rows in traces.items()
        },
        "notes": [
            "All plotted terms are dimensionless terms of the selected fixed-reference objective.",
            "The smooth-off regularizer contribution is exactly zero and is retained at zero on linear axes.",
            "The smooth-off branch resumed after accepted step 102; the smooth-on branch is a fresh neutral fit.",
        ],
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    plot(traces, out, start=0, title="Reference-loss components: updates 0-200")
    plot(traces, out, start=150, title="Reference-loss components: last 50 updates")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
