# Copyright (c) 2026 liblaf
"""Render an audited L2-position, normal, and activation-smoothness endpoint.

This is an additive adapter over the refinement-aware saved-state renderer.  It
retains that renderer's recursive checkpoint verification and full cranium,
mandible, and eye rendering, while substituting objective-transition curves.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
REVIEW_SOURCE = GROUP / "src/31-review-refined-pilot.py"
spec = importlib.util.spec_from_file_location("refined_review", REVIEW_SOURCE)
review = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(review)


class Config(review.Config):
    """Use the verified renderer with a regularized-objective run directory."""

    run_dir: Path = GROUP / "data/inverse-mouthopen-regularized"
    output_dir: Path = GROUP / "data/review-mouthopen-regularized"


def _terms(row: dict[str, Any]) -> dict[str, float] | None:
    terms = row.get("objective_components")
    if terms is None:
        return None
    required = {
        "position",
        "normal",
        "smooth",
        "normal_contribution",
        "smooth_contribution",
        "total",
    }
    assert set(terms) >= required
    values = {key: float(terms[key]) for key in required}
    assert all(np.isfinite(value) and value >= 0 for value in values.values())
    np.testing.assert_allclose(
        values["total"],
        values["position"]
        + values["normal_contribution"]
        + values["smooth_contribution"],
        rtol=1e-12,
        atol=1e-14,
    )
    return values


def curves(output: Path, rows: list[dict[str, Any]], error_mm: np.ndarray) -> str:
    """Plot common position loss and new components without joining objectives."""
    steps = np.asarray([row["optimizer_steps"]["q"] for row in rows])
    position = np.asarray(
        [
            _terms(row)["position"] if _terms(row) is not None else float(row["loss"])
            for row in rows
        ]
    )
    regularized = np.asarray([_terms(row) is not None for row in rows])
    assert regularized.any(), "regularized run must persist objective_components"
    first = int(np.flatnonzero(regularized)[0])
    assert all(regularized[first:])
    terms = [_terms(row) for row in rows[first:]]
    assert all(term is not None for term in terms)
    normal = np.asarray([term["normal"] for term in terms if term is not None])
    smooth = np.asarray([term["smooth"] for term in terms if term is not None])
    total = np.asarray([term["total"] for term in terms if term is not None])

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].plot(steps, position, marker="o", color="#c75f42")
    axes[0].axvline(steps[first], color="#6a4c93", ls="--", lw=1)
    axes[0].annotate(
        "objective transition",
        (steps[first], position[first]),
        xytext=(6, 12),
        textcoords="offset points",
        color="#6a4c93",
        fontsize=8,
    )
    axes[0].set(
        xlabel="Global q optimizer step",
        ylabel="Common L2 position loss",
        title="Position-only term",
    )
    # Total objective exists only after the transition, so it starts a distinct series.
    axes[1].plot(steps[first:], total, marker="o", color="#087d81", label="new total")
    axes[1].plot(
        steps[first:], normal, marker="o", color="#33658a", label="normal term"
    )
    axes[1].plot(
        steps[first:], smooth, marker="o", color="#6a4c93", label="smoothness term"
    )
    axes[1].set(
        xlabel="Global q optimizer step",
        ylabel="Dimensionless value",
        title="Regularized components",
    )
    axes[1].legend()
    axes[2].plot(np.sort(error_mm), np.linspace(0, 1, len(error_mm)), color="#6a4c93")
    axes[2].set(
        xlabel="Target error (mm)",
        ylabel="Cumulative skin fraction",
        title="Saved fit error",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    name = "position-normal-smoothness-curves.png"
    figure.savefig(output / name, dpi=180)
    plt.close(figure)
    return name


def main(cfg: Config) -> None:
    protocol = json.loads((cfg.run_dir / "protocol.json").read_text())
    objective = protocol["objective"]
    assert (
        objective["kind"]
        == "l2_position_plus_oriented_normal_plus_activation_smoothness"
    )
    assert set(objective["weights"]) == {"normal", "smooth"}
    assert all(float(value) >= 0 for value in objective["weights"].values())
    review.curves = curves
    review.main(cfg)
    receipt_path = cfg.output_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["objective_transition"] = {
        "kind": objective["kind"],
        "weights": objective["weights"],
        "curve_policy": "Position loss is comparable across lineage. The new total, normal, and smoothness series begin at the regularized transition and are never connected to predecessor losses.",
    }
    review.write_json(receipt_path, receipt)


if __name__ == "__main__":
    cherries.main(main, profile=review.ProfileJoint)
