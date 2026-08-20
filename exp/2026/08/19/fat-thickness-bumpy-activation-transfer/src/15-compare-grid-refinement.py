from __future__ import annotations

# ruff: noqa: EM101, EM102, FBT003, TRY003
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pydantic_settings as ps

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
EXPECTED_GRIDS = (48, 64)
EXPECTED_WAVE_NUMBER = 4
CASE_COLORS = {
    "thin": "#56B4E9",
    "medium": "#F0E442",
    "thick": "#D55E00",
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    production_summary: Path = cherries.input(
        "10-bumpy-activation-transfer-summary.json"
    )
    refinement_summary: Path = cherries.input("../tmp/refinement-nx64/10-summary.json")
    output_json: Path = cherries.output(
        "15-grid-refinement-comparison.json", mkdir=True
    )
    output_png: Path = cherries.output("15-grid-refinement-sensitivity.png", mkdir=True)


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "ok" or summary.get("complete") is not True:
        raise ValueError(f"summary is not complete and successful: {path}")
    return summary


def cases_by_label(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = summary.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("summary must contain a nonempty cases list")
    result = {str(case["label"]): case for case in cases}
    if len(result) != len(cases):
        raise ValueError("case labels must be unique")
    return result


def relative_change(old: float, new: float) -> float:
    if old == 0.0:
        raise ZeroDivisionError("cannot compute relative change from zero")
    return new / old - 1.0


def attenuation(thin: float, thick: float) -> float:
    if thin <= 0.0 or thick < 0.0:
        raise ValueError("attenuation inputs must be nonnegative with thin > 0")
    return 1.0 - thick / thin


def gate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    cases = cases_by_label(summary)
    stages = [stage for case in cases.values() for stage in case["stages"]]
    solver = summary["solver"]
    successful = [
        stage.get("forward/success") is True
        and stage.get("forward/result") == "primary_success"
        for stage in stages
    ]
    observed = {
        "stages_total": len(stages),
        "stages_primary_success": sum(successful),
        "det_f_min_across_stages": min(float(stage["detF/min"]) for stage in stages),
        "det_f_q001_min_across_stages": min(
            float(stage["detF/q001"]) for stage in stages
        ),
        "inverted_tets_max_per_stage": max(
            int(stage["detF/inverted"]) for stage in stages
        ),
        "below_minimum_det_f_max_per_stage": max(
            int(stage["detF/below_0p2"]) for stage in stages
        ),
        "branch_difference_over_signal_max_across_cases": max(
            float(case["branch/difference_over_induced_signal"])
            for case in cases.values()
        ),
        "branch_transmission_relative_difference_max_across_cases": max(
            float(case["branch/transmission_relative_difference"])
            for case in cases.values()
        ),
    }
    thresholds = {
        "minimum_det_f": float(solver["minimum_det_f"]),
        "minimum_det_f_q001": float(solver["minimum_det_f_q001"]),
        "branch_difference_over_signal_max": float(
            solver["branch_difference_over_signal_max"]
        ),
        "require_all_stages_primary_success": True,
        "require_zero_inverted_tets": True,
        "require_zero_tets_below_minimum_det_f": True,
    }
    all_gates_passed = (
        all(successful)
        and observed["det_f_min_across_stages"] >= thresholds["minimum_det_f"]
        and observed["det_f_q001_min_across_stages"] >= thresholds["minimum_det_f_q001"]
        and observed["inverted_tets_max_per_stage"] == 0
        and observed["below_minimum_det_f_max_per_stage"] == 0
        and observed["branch_difference_over_signal_max_across_cases"]
        <= thresholds["branch_difference_over_signal_max"]
    )
    return {
        "thresholds": thresholds,
        "observed": observed,
        "all_gates_passed": all_gates_passed,
    }


def validate_pair(
    summaries: dict[int, dict[str, Any]],
) -> tuple[str, ...]:
    if tuple(sorted(summaries)) != EXPECTED_GRIDS:
        raise ValueError(
            f"expected grids {EXPECTED_GRIDS}, got {tuple(sorted(summaries))}"
        )
    designs = {str(summary["design"]) for summary in summaries.values()}
    if len(designs) != 1:
        raise ValueError(f"experiment design changed across grids: {designs}")
    wave_numbers = {
        int(summary["paired_design"]["wave_number"]) for summary in summaries.values()
    }
    if wave_numbers != {EXPECTED_WAVE_NUMBER}:
        raise ValueError(
            f"expected wave number {EXPECTED_WAVE_NUMBER}, got {wave_numbers}"
        )
    labels_by_grid = {
        grid: tuple(cases_by_label(summary)) for grid, summary in summaries.items()
    }
    if len(set(labels_by_grid.values())) != 1:
        raise ValueError(f"case labels changed across grids: {labels_by_grid}")
    labels = next(iter(labels_by_grid.values()))
    for label in labels:
        thicknesses = {
            float(cases_by_label(summary)[label]["top_fat_thickness"])
            for summary in summaries.values()
        }
        if len(thicknesses) != 1:
            raise ValueError(f"{label} thickness changed across grids: {thicknesses}")
    return labels


def make_case_comparison(
    label: str,
    summaries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    by_grid = {
        grid: cases_by_label(summary)[label] for grid, summary in summaries.items()
    }
    coarse = by_grid[48]
    fine = by_grid[64]
    amplitude_48 = float(coarse["surface/top_modal_amplitude_abs"])
    amplitude_64 = float(fine["surface/top_modal_amplitude_abs"])
    transmission_48 = float(coarse["transfer/modal_transmission"])
    transmission_64 = float(fine["transfer/modal_transmission"])
    return {
        "label": label,
        "top_fat_thickness": float(coarse["top_fat_thickness"]),
        "nx48": {
            "surface_top_modal_amplitude_abs": amplitude_48,
            "source_interface_modal_amplitude_abs": float(
                coarse["source/interface_modal_amplitude_abs"]
            ),
            "interface_normalized_transmission": transmission_48,
        },
        "nx64": {
            "surface_top_modal_amplitude_abs": amplitude_64,
            "source_interface_modal_amplitude_abs": float(
                fine["source/interface_modal_amplitude_abs"]
            ),
            "interface_normalized_transmission": transmission_64,
        },
        "relative_change_nx64_vs_nx48": {
            "surface_top_modal_amplitude_abs": relative_change(
                amplitude_48, amplitude_64
            ),
            "source_interface_modal_amplitude_abs": relative_change(
                float(coarse["source/interface_modal_amplitude_abs"]),
                float(fine["source/interface_modal_amplitude_abs"]),
            ),
            "interface_normalized_transmission": relative_change(
                transmission_48, transmission_64
            ),
        },
    }


def thick_vs_thin_attenuation(
    grid: int,
    summaries: dict[int, dict[str, Any]],
) -> dict[str, float]:
    cases = cases_by_label(summaries[grid])
    thin = cases["thin"]
    thick = cases["thick"]
    return {
        "surface_top_modal_amplitude_abs": attenuation(
            float(thin["surface/top_modal_amplitude_abs"]),
            float(thick["surface/top_modal_amplitude_abs"]),
        ),
        "interface_normalized_transmission": attenuation(
            float(thin["transfer/modal_transmission"]),
            float(thick["transfer/modal_transmission"]),
        ),
    }


def dark_style() -> None:
    plt.style.use("dark_background")
    mpl.rcParams.update(
        {
            "axes.facecolor": "#111827",
            "figure.facecolor": "#0B1020",
            "savefig.facecolor": "#0B1020",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#F8FAFC",
            "xtick.color": "#E2E8F0",
            "ytick.color": "#E2E8F0",
            "grid.color": "#64748B",
            "grid.alpha": 0.28,
            "font.size": 10,
        }
    )


def plot_sensitivity(
    cases: list[dict[str, Any]],
    attenuations: dict[str, dict[str, float]],
    path: Path,
) -> None:
    dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.79, wspace=0.20)
    grids = [48, 64]
    for case in cases:
        label = str(case["label"])
        color = CASE_COLORS.get(label, "#CC79A7")
        amplitudes = [
            float(case[f"nx{grid}"]["surface_top_modal_amplitude_abs"]) * 1.0e4
            for grid in grids
        ]
        transmissions = [
            float(case[f"nx{grid}"]["interface_normalized_transmission"])
            for grid in grids
        ]
        axes[0].plot(
            grids,
            amplitudes,
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.2,
            label=f"{label} (h={case['top_fat_thickness']:.2f})",
        )
        axes[1].plot(
            grids,
            transmissions,
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.2,
        )
        for axis, values in zip(axes, (amplitudes, transmissions), strict=True):
            axis.annotate(
                f"{values[-1] / values[0] - 1.0:+.1%}",
                (64, values[-1]),
                xytext=(-4, 9),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=9,
                weight="bold",
            )

    axes[0].set_title("Surface modal amplitude")
    axes[0].set_ylabel(r"Amplitude ($\times 10^{-4}$, native length units)")
    axes[1].set_title("Interface-normalized transmission")
    axes[1].set_ylabel("Surface / muscle-interface modal amplitude")
    for axis in axes:
        axis.set_xlabel("Horizontal grid cells per side (nx = nz)")
        axis.set_xticks(grids)
        axis.grid(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")

    attenuation_48 = attenuations["nx48"]
    attenuation_64 = attenuations["nx64"]
    fig.suptitle(
        "Grid-refinement sensitivity of bumpy-activation transfer (k=4)",
        y=0.96,
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        0.035,
        (
            "Thick-vs-thin attenuation: surface "
            f"{attenuation_48['surface_top_modal_amplitude_abs']:.1%} → "
            f"{attenuation_64['surface_top_modal_amplitude_abs']:.1%}; "
            "transmission "
            f"{attenuation_48['interface_normalized_transmission']:.1%} → "
            f"{attenuation_64['interface_normalized_transmission']:.1%}.\n"
            "Monotone trend robust; quantitative grid convergence not established."
        ),
        ha="center",
        va="bottom",
        color="#FBBF24",
        fontsize=9.5,
        weight="bold",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main(cfg: Config) -> None:
    if cfg.output_json.exists() or cfg.output_png.exists():
        raise FileExistsError("refusing to overwrite grid-refinement outputs")

    production = load_summary(cfg.production_summary)
    refinement = load_summary(cfg.refinement_summary)
    summaries = {
        int(production["mesh"]["nx"]): production,
        int(refinement["mesh"]["nx"]): refinement,
    }
    labels = validate_pair(summaries)
    cases = [make_case_comparison(label, summaries) for label in labels]
    attenuations = {
        f"nx{grid}": thick_vs_thin_attenuation(grid, summaries)
        for grid in EXPECTED_GRIDS
    }
    attenuation_change_pp = {
        metric: 100.0 * (attenuations["nx64"][metric] - attenuations["nx48"][metric])
        for metric in attenuations["nx48"]
    }
    gates = {f"nx{grid}": gate_summary(summary) for grid, summary in summaries.items()}
    monotone_by_grid = {
        f"nx{grid}": {
            metric: all(
                values[index] > values[index + 1] for index in range(len(values) - 1)
            )
            for metric, values in {
                "surface_top_modal_amplitude_abs": [
                    float(
                        cases_by_label(summaries[grid])[label][
                            "surface/top_modal_amplitude_abs"
                        ]
                    )
                    for label in labels
                ],
                "interface_normalized_transmission": [
                    float(
                        cases_by_label(summaries[grid])[label][
                            "transfer/modal_transmission"
                        ]
                    )
                    for label in labels
                ],
            }.items()
        }
        for grid in EXPECTED_GRIDS
    }
    qualitative_robust = all(
        passed
        for grid_result in monotone_by_grid.values()
        for passed in grid_result.values()
    )
    if not qualitative_robust:
        raise RuntimeError(
            f"monotone attenuation did not survive refinement: {monotone_by_grid}"
        )
    if not all(result["all_gates_passed"] for result in gates.values()):
        raise RuntimeError(f"a solver gate failed: {gates}")

    maximum_amplitude_change = max(
        abs(
            float(
                case["relative_change_nx64_vs_nx48"]["surface_top_modal_amplitude_abs"]
            )
        )
        for case in cases
    )
    maximum_transmission_change = max(
        abs(
            float(
                case["relative_change_nx64_vs_nx48"][
                    "interface_normalized_transmission"
                ]
            )
        )
        for case in cases
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "status": "ok",
        "question": (
            "Does the monotone attenuation of bumpy muscle activation by thicker "
            "fat persist when the horizontal grid is refined from 48 to 64 cells?"
        ),
        "input_summaries": {
            "nx48": str(cfg.production_summary.resolve()),
            "nx64": str(cfg.refinement_summary.resolve()),
        },
        "held_fixed": {
            "wave_number": EXPECTED_WAVE_NUMBER,
            "vertical_spacing": float(production["mesh"]["vertical_spacing"]),
            "top_fat_thicknesses": list(
                production["paired_design"]["top_fat_thicknesses"]
            ),
            "experiment_design": production["design"],
        },
        "cases": cases,
        "thick_vs_thin_attenuation": {
            **attenuations,
            "nx64_minus_nx48_percentage_points": attenuation_change_pp,
        },
        "solver_gates": gates,
        "monotone_attenuation_by_grid": monotone_by_grid,
        "maximum_absolute_relative_change_nx64_vs_nx48": {
            "surface_top_modal_amplitude_abs": maximum_amplitude_change,
            "interface_normalized_transmission": maximum_transmission_change,
        },
        "quantitatively_grid_converged": False,
        "qualitative_monotone_attenuation_robust": qualitative_robust,
        "interpretation": (
            "The attenuation ordering is unchanged and stronger at nx=64, but one "
            "refinement pair cannot establish asymptotic convergence; the raw modal "
            "amplitude and thick-case normalized transmission remain resolution-sensitive."
        ),
        "plot": str(cfg.output_png.resolve()),
    }

    cfg.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_sensitivity(cases, attenuations, cfg.output_png)

    for step, case in enumerate(cases):
        cherries.set_step(step)
        cherries.log_metrics(
            {
                f"{case['label']}/surface_amplitude_relative_change": case[
                    "relative_change_nx64_vs_nx48"
                ]["surface_top_modal_amplitude_abs"],
                f"{case['label']}/transmission_relative_change": case[
                    "relative_change_nx64_vs_nx48"
                ]["interface_normalized_transmission"],
            }
        )
    cherries.set_step(len(cases))
    cherries.log_metrics(
        {
            "refinement/max_surface_amplitude_relative_change": maximum_amplitude_change,
            "refinement/max_transmission_relative_change": maximum_transmission_change,
            "refinement/thick_vs_thin_transmission_attenuation_nx48": attenuations[
                "nx48"
            ]["interface_normalized_transmission"],
            "refinement/thick_vs_thin_transmission_attenuation_nx64": attenuations[
                "nx64"
            ]["interface_normalized_transmission"],
        }
    )
    logger.info("Wrote grid-refinement comparison to %s", cfg.output_json)
    logger.info("Wrote grid-refinement sensitivity plot to %s", cfg.output_png)


if __name__ == "__main__":
    cherries.main(main)
