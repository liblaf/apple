"""Render the completed coupled-seed continuation benchmark without fitting claims."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    benchmark_dir: Path = GROUP / "data/coupled-continuation-benchmark-002"
    output_dir: Path = GROUP / "data/coupled-continuation-visuals-002"


def number(value: Any) -> float | None:
    """Return a finite scalar, preserving unknown receipt fields as missing."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def detail_seconds(details: dict[str, Any]) -> float | None:
    predictor = details.get("predictor", {})
    predictor_seconds = number(predictor.get("seconds"))
    corrector_seconds = number(details.get("internal_corrector_wall_seconds"))
    if predictor_seconds is None:
        return None
    return predictor_seconds + (corrector_seconds or 0.0)


def contact_gap_um(details: dict[str, Any]) -> float | None:
    endpoint = details.get("geometry", {}).get("endpoint_contact", {})
    gap = number(endpoint.get("minimum_active_distance_m"))
    return None if gap is None else gap * 1e6


def completed_attempts(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    receipt = summary["seed_continuation"]
    attempts = receipt["attempts"]
    assert isinstance(attempts, list)
    assert attempts
    assert receipt["success"]
    assert receipt["progress"] == 1.0
    enriched = []
    known_timing = True
    elapsed = 0.0
    for row in attempts:
        assert isinstance(row, dict)
        assert "admitted" in row
        details = row.get("details", {})
        assert isinstance(details, dict)
        predictor_seconds = number(details.get("predictor", {}).get("seconds"))
        corrector_seconds = number(details.get("internal_corrector_wall_seconds"))
        seconds = detail_seconds(details)
        known_timing &= seconds is not None
        if predictor_seconds is not None:
            elapsed += predictor_seconds
        predictor_elapsed = elapsed
        if corrector_seconds is not None:
            elapsed += corrector_seconds
        enriched.append(
            {
                "attempt": int(row["attempt"]),
                "admitted": bool(row["admitted"]),
                "progress": float(row["progress_proposed"]),
                "elapsed_seconds": elapsed,
                "predictor_elapsed_seconds": predictor_elapsed,
                "seed_gap_um": contact_gap_um(details),
                "corrected_gap_um": contact_gap_um(
                    {
                        "geometry": {
                            "endpoint_contact": details.get(
                                "internal_corrector", {}
                            ).get("contact", {})
                        }
                    }
                ),
                "predictor_seconds": predictor_seconds,
                "internal_corrector_seconds": corrector_seconds,
                "internal_corrector_steps": number(
                    details.get("internal_corrector", {}).get("steps")
                ),
                "internal_corrector_force": number(
                    details.get("internal_corrector", {}).get("grad_norm")
                ),
                "internal_corrector_threshold": number(
                    details.get("internal_corrector", {}).get("force_threshold")
                ),
            }
        )
    return enriched, known_timing


def render(  # noqa: PLR0915
    summary: dict[str, Any], output: Path
) -> tuple[Path, dict[str, Any]]:
    rows, known_timing = completed_attempts(summary)
    accepted = [row for row in rows if row["admitted"]]
    assert accepted
    target_angle = float(summary["angle_deg"])
    final = summary["corrector"]
    assert final["success"]
    assert final["terminal_gates"]["no_intersections"]
    assert final["terminal_gates"]["minimum_active_gap_at_least_10nm"]

    x_all = (
        np.asarray([row["elapsed_seconds"] for row in rows]) / 60
        if known_timing
        else np.asarray([row["attempt"] for row in rows])
    )
    x_accepted = (
        np.asarray([row["elapsed_seconds"] for row in accepted]) / 60
        if known_timing
        else np.asarray([row["attempt"] for row in accepted])
    )
    x_label = (
        "Cumulative recorded solver time (min)"
        if known_timing
        else "Attempt index (receipt timing incomplete)"
    )
    angles = np.asarray([target_angle * row["progress"] for row in accepted])
    seed_x = (
        np.asarray([row["predictor_elapsed_seconds"] for row in accepted]) / 60
        if known_timing
        else x_accepted
    )
    seed_gaps = np.asarray([row["seed_gap_um"] for row in accepted], dtype=float)
    corrected_gaps = np.asarray(
        [row["corrected_gap_um"] for row in accepted], dtype=float
    )
    internal_steps = np.asarray(
        [row["internal_corrector_steps"] for row in accepted], dtype=float
    )
    internal_force = np.asarray(
        [row["internal_corrector_force"] for row in accepted], dtype=float
    )
    internal_threshold = np.asarray(
        [row["internal_corrector_threshold"] for row in accepted], dtype=float
    )
    predictor_seconds = sum(row["predictor_seconds"] or 0.0 for row in rows)
    internal_seconds = sum(row["internal_corrector_seconds"] or 0.0 for row in rows)
    final_seconds = float(summary["corrector_seconds"])
    seed_total_seconds = float(summary["seed_continuation"]["seconds"])
    final_x = (
        (seed_total_seconds + final_seconds) / 60
        if known_timing
        else x_accepted[-1] + 1
    )
    final_gap_um = number(final["contact"].get("minimum_active_distance_m"))
    assert final_gap_um is not None
    final_gap_um *= 1e6

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    figure.suptitle(
        "Coupled continuation to a 1° jaw + 12.33 Pa active-stress proposal",
        fontsize=14,
        fontweight="bold",
    )

    axis = axes[0, 0]
    rejected = [index for index, row in enumerate(rows) if not row["admitted"]]
    axis.plot(x_accepted, angles, "o-", color="#087d81", label="admitted seed")
    if rejected:
        axis.scatter(
            x_all[rejected], np.full(len(rejected), np.nan), marker="x", color="#bc622a"
        )
    axis.axhline(target_angle, color="#53565a", linestyle="--", label="requested 1°")
    axis.set(
        xlabel=x_label,
        ylabel="Jaw opening (degrees)",
        title="Internal continuation progress",
    )
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    finite_seed_gaps = np.isfinite(seed_gaps)
    finite_corrected_gaps = np.isfinite(corrected_gaps)
    if finite_seed_gaps.any():
        axis.plot(
            seed_x[finite_seed_gaps],
            seed_gaps[finite_seed_gaps],
            "o--",
            color="#7da5ac",
            label="admitted predictor seed",
        )
    if finite_corrected_gaps.any():
        axis.plot(
            x_accepted[finite_corrected_gaps],
            corrected_gaps[finite_corrected_gaps],
            "o-",
            color="#087d81",
            label="internal strict corrector",
        )
    if finite_seed_gaps.any() or finite_corrected_gaps.any():
        axis.scatter(
            [final_x],
            [final_gap_um],
            marker="s",
            s=42,
            color="#bc622a",
            label="final strict corrector",
        )
        axis.axhline(0.01, color="#53565a", linestyle="--", label="10 nm gate")
        axis.set_yscale("log")
        axis.legend(fontsize=8)
    else:
        axis.text(
            0.5,
            0.5,
            "No endpoint active-gap receipt",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set(
        xlabel=x_label,
        ylabel="Minimum active gap (µm)",
        title="Admitted full-contact endpoint",
    )

    axis = axes[1, 0]
    finite_steps = np.isfinite(internal_steps)
    if finite_steps.any():
        axis.plot(
            x_accepted[finite_steps],
            internal_steps[finite_steps],
            "o-",
            color="#087d81",
            label="internal strict PNCG",
        )
    axis.scatter(
        [final_x],
        [final["steps"]],
        marker="s",
        s=42,
        color="#bc622a",
        label="final strict PNCG",
    )
    axis.set(
        xlabel=x_label,
        ylabel="Strict corrector steps",
        title="Equilibrium work by continuation state",
    )
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    force_mask = np.isfinite(internal_force) & np.isfinite(internal_threshold)
    if force_mask.any():
        axis.plot(
            x_accepted[force_mask],
            internal_force[force_mask] / internal_threshold[force_mask],
            "o-",
            color="#087d81",
            label="internal strict corrector",
        )
    axis.scatter(
        [final_x],
        [final["grad_norm"] / final["force_threshold"]],
        marker="s",
        s=42,
        color="#bc622a",
        label="final strict corrector",
    )
    axis.axhline(
        1.0,
        color="#53565a",
        linestyle="--",
        label="strict threshold",
    )
    axis.set(
        xlabel=x_label,
        ylabel="Exact force / strict threshold",
        title="Strict corrector force ratio",
    )
    axis.legend(fontsize=7, ncol=2)

    filename = output / "coupled-continuation-benchmark.png"
    figure.savefig(filename, dpi=100)
    plt.close(figure)
    timings = {
        "predictor_seconds": predictor_seconds,
        "internal_corrector_seconds": internal_seconds,
        "final_corrector_seconds": final_seconds,
    }
    return filename, {
        "known_attempt_timing": known_timing,
        "timings": timings,
        "attempt_count": len(rows),
        "accepted_substeps": len(accepted),
    }


def main(cfg: Config) -> None:
    summary_path = cfg.benchmark_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["schema"] == "coupled-continuation-full-face-benchmark-v1"
    assert summary["success"]
    assert not summary["running"]
    assert summary["phase"] == "complete"
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    figure, metrics = render(summary, cfg.output_dir)
    receipt = {
        "schema": "coupled-continuation-render-v1",
        "success": True,
        "scope": "Visualization of the completed full-contact seed continuation and its strict final corrector; it is not an outer inverse-fit iteration or expression-fit convergence claim.",
        "benchmark_summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256(summary_path),
        },
        "figure": {"path": str(figure.resolve()), "sha256": sha256(figure)},
        "requested_angle_deg": summary["angle_deg"],
        "requested_active_stress_pa": summary["active_stress_mpa"] * 1e6,
        "final_contact": summary["corrector"]["contact"],
        "final_strict_corrector": summary["corrector"],
        **metrics,
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_metrics(
        {"accepted_substeps": metrics["accepted_substeps"], **metrics["timings"]}
    )
    cherries.log_output(cfg.output_dir)
    LOG.info("Rendered coupled continuation benchmark to %s", figure)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
