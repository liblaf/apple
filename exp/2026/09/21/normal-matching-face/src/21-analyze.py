"""Read-only convergence and paired-effect analysis for the four face runs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pydantic_settings as ps
from experiment import Profile

from liblaf import cherries

VARIANTS = (
    "smooth-off-l2",
    "smooth-off-normal",
    "smooth-on-l2",
    "smooth-on-normal",
)
KEYS = (
    "objective",
    "position_loss_component_mm2",
    "fit_rms_mm",
    "normal_angle_rms_deg",
    "normal_loss",
    "primary_union_normal_residual_highpass_5mm_rms_mm",
    "activation_smoothness",
    "motion_rms_mm",
    "physical_gradient_ratio",
    "detF_min",
    "inverted_all_cells",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = Path("10-comparison")
    output: Path = Path("21-analysis")
    gate: Path = Path("20-verification/checks.json")


def _read_trace(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    assert rows, path
    assert [int(row["step"]) for row in rows] == list(range(len(rows))), path
    return rows


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), path
    return value


def _row(row: dict[str, float]) -> dict[str, float]:
    return {key: row[key] for key in KEYS}


def _first_inversion(rows: list[dict[str, float]]) -> int | None:
    for row in rows:
        if row["inverted_all_cells"] > 0:
            return int(row["step"])
    return None


def _saved_steps(path: Path) -> set[int]:
    return {
        int(match.group(1))
        for candidate in path.glob("step-*.npz")
        if (match := re.fullmatch(r"step-(\d+)\.npz", candidate.name))
    }


def _last10(rows: list[dict[str, float]]) -> dict[str, float]:
    assert len(rows) > 10
    before, after = rows[-11], rows[-1]
    changes: dict[str, float] = {}
    for key in (
        "objective",
        "position_loss_component_mm2",
        "normal_loss",
        "normal_angle_rms_deg",
    ):
        changes[f"{key}_absolute_change"] = after[key] - before[key]
        changes[f"{key}_relative_change"] = (after[key] - before[key]) / before[key]
    changes["physical_gradient_ratio_final"] = after["physical_gradient_ratio"]
    return changes


def _effect(reference: dict[str, float], changed: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in KEYS:
        if key in {"objective", "detF_min", "inverted_all_cells"}:
            continue
        values[f"{key}_absolute_change"] = changed[key] - reference[key]
        values[f"{key}_percent_change"] = (
            100 * (changed[key] - reference[key]) / reference[key]
        )
    return values


def main(cfg: Config) -> None:
    branches = VARIANTS
    source = cherries.input(cfg.comparison_dir)
    gate = _json(cherries.input(cfg.gate))
    assert gate["passed"] is True
    traces = {name: _read_trace(source / name / "trace.csv") for name in branches}
    summaries = {name: _json(source / name / "summary.json") for name in branches}
    for name, rows in traces.items():
        summary = summaries[name]
        assert summary["status"] == "completed_budget_not_convergence_certified", name
        assert summary["failure"] is None, name
        assert int(summary["last_step"]) == int(rows[-1]["step"]), name
        assert int(rows[-1]["step"]) == 100, name
        assert summary["last_metrics"]["step"] == int(rows[-1]["step"]), name

    first_inversion = {name: _first_inversion(rows) for name, rows in traces.items()}
    for name, step in first_inversion.items():
        if step is not None:
            assert all(row["inverted_all_cells"] > 0 for row in traces[name][step:]), (
                name
            )
    valid_last = {
        name: (len(traces[name]) - 1 if step is None else step - 1)
        for name, step in first_inversion.items()
    }
    shared_valid_step = min(valid_last.values())
    common_saved_steps = set.intersection(
        *[_saved_steps(source / name) for name in branches]
    )
    shared_valid_saved_geometry_step = max(
        step for step in common_saved_steps if step <= shared_valid_step
    )
    report: dict[str, Any] = {
        "branches": list(VARIANTS),
        "shared_valid_trace_step": shared_valid_step,
        "shared_valid_saved_geometry_step": shared_valid_saved_geometry_step,
        "variants": {
            name: {
                "status": summaries[name]["status"],
                "endpoint_step": int(rows[-1]["step"]),
                "first_inversion_step": first_inversion[name],
                "last_inversion_free_step": valid_last[name],
                "endpoint": _row(rows[-1]),
                "shared_valid": _row(rows[shared_valid_step]),
                "shared_valid_saved_geometry": _row(
                    rows[shared_valid_saved_geometry_step]
                ),
                "last10": _last10(rows),
            }
            for name, rows in traces.items()
        },
        "paired_effects": {},
    }
    for smooth in ("off", "on"):
        l2, normal = f"smooth-{smooth}-l2", f"smooth-{smooth}-normal"
        if l2 in traces and normal in traces:
            report["paired_effects"][f"normal_at_smooth_{smooth}"] = {
                "endpoint": _effect(traces[l2][-1], traces[normal][-1]),
                "shared_valid": _effect(
                    traces[l2][shared_valid_step], traces[normal][shared_valid_step]
                ),
                "shared_valid_saved_geometry": _effect(
                    traces[l2][shared_valid_saved_geometry_step],
                    traces[normal][shared_valid_saved_geometry_step],
                ),
            }
    for kind in ("l2", "normal"):
        off, on = f"smooth-off-{kind}", f"smooth-on-{kind}"
        if off in traces and on in traces:
            report["paired_effects"][f"smoothness_at_{kind}"] = {
                "endpoint": _effect(traces[off][-1], traces[on][-1]),
                "shared_valid": _effect(
                    traces[off][shared_valid_step], traces[on][shared_valid_step]
                ),
                "shared_valid_saved_geometry": _effect(
                    traces[off][shared_valid_saved_geometry_step],
                    traces[on][shared_valid_saved_geometry_step],
                ),
            }
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
