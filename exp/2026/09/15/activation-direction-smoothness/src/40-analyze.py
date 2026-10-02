"""Select a common effective smoothness weight and export measured comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
LABELS = {
    "unconstrained": "Free activation",
    "contraction_only": "Contraction-only, free directions",
    "learned_direction": "Contraction-only, learned direction",
    "x_contraction": "Contraction-only, fixed x-direction",
}


class ProfileAnalysis(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_dirs: str = "tune-w0,tune-w001,tune-w01,tune-w1"
    output: Path = Path("40-analysis")
    verification_dirs: str = "20-baseline-checks,21-regularized-checks"
    target_roughness_reduction: float = 0.75
    common_step: int = 250


def json_write(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(cfg: Config):  # noqa: PLR0915
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    checked = {}
    for directory in cfg.verification_dirs.split(","):
        evidence = json.loads((GROUP / "data" / directory / "checks.json").read_text())
        checked.update({c["case"]: c for c in evidence["cases"]})
    all_cases = []
    for name in cfg.input_dirs.split(","):
        directory = GROUP / "data" / name
        cases = json.loads((directory / "summary.json").read_text())
        assert len(cases) == 8, (directory, len(cases))
        for case in cases:
            folder = directory / case["name"]
            with (folder / "trace.csv").open() as f:
                trace = list(csv.DictReader(f))
            assert len(trace) == case["accepted_iterations"] + 1
            assert case["accepted_iterations"] >= cfg.common_step, case["name"]
            common = {
                key: float(value) for key, value in trace[cfg.common_step].items()
            }
            assert common["step"] == cfg.common_step
            case["common"] = common
            case["folder"] = str(folder)
            all_cases.append(case)
    keys = [(h, m) for h in (0.05, 0.2) for m in LABELS]
    lookup = {(c["height"], c["mode"], c["smooth_weight"]): c for c in all_cases}
    assert len(lookup) == len(all_cases)
    weights = sorted({c["smooth_weight"] for c in all_cases})
    assert weights[0] == 0
    comparisons = []
    candidates = []
    for weight in weights:
        records = []
        for height, mode in keys:
            case = lookup[(height, mode, weight)]
            base = lookup[(height, mode, 0)]
            last, first = case["final"], base["final"]
            row = {
                "height": height,
                "mode": mode,
                "weight": weight,
                "last_solved_step": case["accepted_iterations"],
                "failure": case["failure"] is not None,
                "minimum_hessian_eigenvalue": checked[case["name"]][
                    "smallest_algebraic_hessian_eigenvalue"
                ],
                "locally_stable_endpoint": checked[case["name"]][
                    "smallest_algebraic_hessian_eigenvalue"
                ]
                > 0,
                "data_L2_over_h2": last["normalized_loss"],
                "fit_RMS": last["fit_rms"],
                "R": last["roughness"],
                "R_reduction_pct": 100 * (1 - last["roughness"] / first["roughness"]),
                "fit_L2_change_pct": 100
                * (last["normalized_loss"] / first["normalized_loss"] - 1),
                "relative_R": last["relative_roughness"],
                "relative_R_reduction_pct": 100
                * (1 - last["relative_roughness"] / first["relative_roughness"]),
                "tensor_rms": last["tensor_magnitude_rms"],
                "neighbor_jump_rms": last["tensor_neighbor_rms"],
                "max_neighbor_jump": last["max_neighbor_jump"],
                "min_J": last["min_J"],
                "min_singular_B": last["min_singular_B"],
                "max_singular_B": last["max_singular_B"],
                "active_extension_fraction": last["active_extension_fraction"],
                "nonpositive_det_B_fraction": last["nonpositive_det_B_fraction"],
                "gradient_mapping_inf": last["projected_gradient_inf"],
                "last50_data_L2_improvement_pct": 100
                * (
                    float(
                        trace_val(
                            case, case["accepted_iterations"] - 50, "normalized_loss"
                        )
                    )
                    - last["normalized_loss"]
                )
                / float(
                    trace_val(case, case["accepted_iterations"] - 50, "normalized_loss")
                ),
                "common_step": cfg.common_step,
                "common_data_L2_over_h2": case["common"]["normalized_loss"],
                "common_R": case["common"]["roughness"],
                "common_R_reduction_pct": 100
                * (1 - case["common"]["roughness"] / base["common"]["roughness"]),
                "common_fit_L2_change_pct": 100
                * (
                    case["common"]["normalized_loss"]
                    / base["common"]["normalized_loss"]
                    - 1
                ),
            }
            checkpoint = np.load(Path(case["folder"]) / "checkpoint.npz")
            B = checkpoint["B"][np.load(Path(case["folder"]) / "history.npz")["muscle"]]
            strengths = np.linalg.eigvalsh(B) - 1
            row["inactive_fraction"] = float(
                np.mean(np.linalg.norm(B - np.eye(2), axis=(1, 2)) < 1e-8)
            )
            row["secondary_contraction_fraction"] = float(
                np.mean(strengths[:, 0] > 1e-6)
            )
            if mode == "learned_direction":
                q = checkpoint["controls"].reshape(-1, 2)
                angle = np.abs((q[:, 1] + np.pi / 2) % np.pi - np.pi / 2) * 180 / np.pi
                active = q[:, 0] > 1e-6
                row["learned_axis_mean_abs_angle_deg"] = float(np.mean(angle[active]))
                row["learned_axis_p95_abs_angle_deg"] = float(
                    np.quantile(angle[active], 0.95)
                )
                row["learned_strength_max"] = float(q[:, 0].max())
            else:
                row["learned_axis_mean_abs_angle_deg"] = None
                row["learned_axis_p95_abs_angle_deg"] = None
                row["learned_strength_max"] = None
            records.append(row)
        comparisons.extend(records)
        if weight > 0:
            candidates.append(
                {
                    "weight": weight,
                    "minimum_endpoint_R_reduction_pct": min(
                        r["R_reduction_pct"] for r in records
                    ),
                    "minimum_common_R_reduction_pct": min(
                        r["common_R_reduction_pct"] for r in records
                    ),
                    "maximum_endpoint_fit_L2_increase_pct": max(
                        r["fit_L2_change_pct"] for r in records
                    ),
                    "maximum_common_fit_L2_increase_pct": max(
                        r["common_fit_L2_change_pct"] for r in records
                    ),
                    "all_finished": all(not r["failure"] for r in records),
                    "all_locally_stable_endpoints": all(
                        r["locally_stable_endpoint"] for r in records
                    ),
                    "eligible": all(
                        not r["failure"]
                        and r["locally_stable_endpoint"]
                        and min(r["R_reduction_pct"], r["common_R_reduction_pct"])
                        >= 100 * cfg.target_roughness_reduction
                        for r in records
                    ),
                }
            )
    eligible = [r for r in candidates if r["eligible"]]
    selected = eligible[0]["weight"] if eligible else None
    selection = {
        "criterion": "Smallest shared alpha with at least 75% R reduction in all 4 models at both targets, at last-valid endpoints and shared update 250, all regularized runs finish 1,200 with positive endpoint Hessian.",
        "target_roughness_reduction": cfg.target_roughness_reduction,
        "common_step": cfg.common_step,
        "selected_weight": selected,
        "candidates": candidates,
        "endpoint_warning": "Unregularized free h=.20 stopped early; endpoint ratios are not equal-budget. Shared-step ratios are used as an additional selection gate.",
    }
    json_write(out / "selection.json", selection)
    json_write(out / "comparisons.json", comparisons)
    with (out / "comparisons.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparisons[0])
        writer.writeheader()
        writer.writerows(comparisons)
    lines = [
        "# Measured smoothness selection",
        "",
        json.dumps(selection, indent=2),
        "",
        "| h | Model | weight | L2/h² | RMS | R reduction | Shared-step R reduction | min J | Last update |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparisons:
        if r["weight"] in {0, selected}:
            lines.append(  # noqa: PERF401
                f"| {r['height']:.2f} | {LABELS[r['mode']]} | {r['weight']:g} | {r['data_L2_over_h2']:.6f} | {r['fit_RMS']:.6f} | {r['R_reduction_pct']:.1f}% | {r['common_R_reduction_pct']:.1f}% | {r['min_J']:.4g} | {r['last_solved_step']} |"
            )
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(selection, indent=2))


def trace_val(case: dict[str, Any], step: int, key: str):
    with (Path(case["folder"]) / "trace.csv").open() as f:
        return list(csv.DictReader(f))[step][key]


if __name__ == "__main__":
    cherries.main(main, profile=ProfileAnalysis)
