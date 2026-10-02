"""Select the predeclared smoothness coefficient from completed pilot endpoints."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pydantic_settings as ps

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    pilot: Path = cherries.input("12-smooth-pilot")
    output: Path = cherries.output("15-selection/selection.json", mkdir=True)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _endpoint(folder: Path, steps: int) -> dict:
    summary = _read_json(folder / "summary.json")
    assert summary["status"] == "completed_budget_not_convergence_certified"
    assert int(summary["last_step"]) == steps
    with (folder / "trace.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == steps + 1
    assert int(rows[-1]["step"]) == steps
    metrics = summary["last_metrics"]
    assert all(float(rows[-1][key]) == float(metrics[key]) for key in metrics)
    return metrics


def main(cfg: Config) -> None:
    pilot = cfg.pilot
    protocol = _read_json(pilot / "protocol.json")
    coefficients = _read_json(pilot / "coefficients.json")
    assert protocol["config"]["phase"] == "smooth-pilot"
    steps = int(protocol["config"]["steps"])
    assert steps == int(protocol["config"]["pilot_steps"]) == 20
    assert coefficients["off"] == 0.0

    off = _endpoint(pilot / "off", steps)
    off_roughness = float(off["activation_smoothness"])
    assert off_roughness > 0.0
    off_data = float(off["data_objective"])
    assert off_data > 0.0

    candidates = []
    for name, coefficient in sorted(coefficients.items(), key=lambda item: item[1]):
        metrics = _endpoint(pilot / name, steps)
        roughness = float(metrics["activation_smoothness"])
        reduction = 1.0 - roughness / off_roughness
        row = {
            "name": name,
            "coefficient": float(coefficient),
            "fit_rms_mm": float(metrics["fit_rms_mm"]),
            "surface_gradient_rms": float(metrics["surface_gradient_rms"]),
            "motion_rms_mm": float(metrics["motion_rms_mm"]),
            "activation_smoothness": roughness,
            "roughness_reduction_fraction": reduction,
            "roughness_reduction_percent": 100.0 * reduction,
            "data_objective": float(metrics["data_objective"]),
            "data_cost_percent": 100.0
            * (float(metrics["data_objective"]) - off_data)
            / off_data,
            "min_detF": float(metrics["detF_min"]),
            "inverted_all_cells": int(metrics["inverted_all_cells"]),
            "inverted_active_cells": int(metrics["inverted_active_cells"]),
        }
        row["qualifies"] = (
            row["coefficient"] > 0.0
            and row["roughness_reduction_fraction"] >= 0.75
            and row["inverted_all_cells"] == 0
        )
        candidates.append(row)

    selected = next((row for row in candidates if row["qualifies"]), None)
    result = {
        "status": "selected" if selected is not None else "no_qualified_coefficient",
        "rule": (
            "Smallest tested positive coefficient with at least 75% activation "
            "smoothness reduction against off at update 20 and zero inverted tetrahedra."
        ),
        "pilot_steps": steps,
        "off": {
            "activation_smoothness": off_roughness,
            "data_objective": off_data,
            "fit_rms_mm": float(off["fit_rms_mm"]),
            "surface_gradient_rms": float(off["surface_gradient_rms"]),
            "motion_rms_mm": float(off["motion_rms_mm"]),
            "min_detF": float(off["detF_min"]),
            "inverted_all_cells": int(off["inverted_all_cells"]),
        },
        "candidates": candidates,
        "selected_coefficient": None if selected is None else selected["coefficient"],
        "selected_name": None if selected is None else selected["name"],
        "inputs": {
            "pilot_protocol_sha256": _digest(pilot / "protocol.json"),
            "pilot_coefficients_sha256": _digest(pilot / "coefficients.json"),
            "selector_source_sha256": _digest(Path(__file__)),
        },
    }
    cfg.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    cherries.log_metrics(
        {
            "selection/qualified_candidates": sum(
                row["qualifies"] for row in candidates
            ),
            "selection/selected_coefficient": (
                -1.0 if selected is None else selected["coefficient"]
            ),
        }
    )
    cherries.log_others(
        {
            "selection/status": result["status"],
            "selection/selected_name": result["selected_name"],
        }
    )
    assert selected is not None, result


if __name__ == "__main__":
    cherries.main(main)
