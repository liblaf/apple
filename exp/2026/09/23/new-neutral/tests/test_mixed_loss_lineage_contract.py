# ruff: noqa: PT009
"""CPU-only contract tests for the shared coupled-review lineage helper."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

GROUP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GROUP / "src"))

from coupled_review_curves import (  # noqa: E402
    _event_markers,
    break_at_zero_update,
    combine_lineage_rows,
    objective_component_series,
    plot_objective_components,
)


class _Axis:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def axvline(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("axvline", args, kwargs))

    def scatter(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("scatter", args, kwargs))

    def annotate(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("annotate", args, kwargs))


class MixedLossLineageContract(unittest.TestCase):
    def test_both_review_scripts_use_the_shared_helper(self) -> None:
        for name in (
            "147-review-mouthopen-coupled.py",
            "181-review-expression-coupled.py",
        ):
            source = (GROUP / "src" / name).read_text()
            self.assertIn("plot_objective_components", source)

    def test_legacy_position_only_rows_allow_absent_components(self) -> None:
        series = objective_component_series(
            [
                {"iteration": 0, "loss": 2.5},
                {"iteration": 1, "loss": 1.75},
            ]
        )
        np.testing.assert_allclose(series["normalized_position_l2"], [2.5, 1.75])
        self.assertTrue(series["legacy_position_only"].all())
        self.assertTrue(np.isnan(series["weighted_normal"]).all())
        self.assertTrue(np.isnan(series["weighted_smoothness"]).all())
        self.assertTrue(np.isnan(series["raw_activation_roughness"]).all())

    def test_plot_executes_on_a_synthetic_lineage(self) -> None:
        row = {
            "iteration": 0,
            "loss": 1.0,
            "loss_components": {
                "position_loss": 0.4,
                "normal_contribution": 0.5,
                "regularizer_contribution": 0.1,
                "activation_smoothness": 3.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            name, coverage = plot_objective_components(Path(directory), [row], [])
            self.assertEqual(name, "objective-components-curves.png")
            self.assertTrue((Path(directory) / name).is_file())
        self.assertEqual(coverage["rows_with_normal_component"], 1)

    def test_same_iteration_refinement_is_retained(self) -> None:
        parent = [{"iteration": 5, "loss": 1.0}]
        child = [{"iteration": 0, "loss": 0.8, "equilibrium_refinement": {}}]
        combined = combine_lineage_rows(child, parent)
        self.assertEqual([row["iteration"] for row in combined], [5, 5])
        self.assertIn("equilibrium_refinement", combined[-1])

    def test_recovery_zero_update_is_retained_and_breaks_lines(self) -> None:
        parent = [{"iteration": 32, "loss": 1.0}]
        child = [{"iteration": 0, "loss": 0.9, "recovery_zero_update": {}}]
        combined = combine_lineage_rows(child, parent)
        self.assertEqual([row["iteration"] for row in combined], [32, 32])
        displayed = break_at_zero_update(
            combined, np.asarray([row["loss"] for row in combined])
        )
        self.assertTrue(np.isfinite(displayed[0]))
        self.assertTrue(np.isnan(displayed[1]))

    def test_same_iteration_objective_change_is_retained_and_marked(self) -> None:
        parent = [{"iteration": 5, "loss": 1.0}]
        child = [
            {
                "iteration": 0,
                "loss": 0.8,
                "objective_change": {"reason": "regularizers enabled"},
            }
        ]
        combined = combine_lineage_rows(child, parent)
        self.assertEqual([row["iteration"] for row in combined], [5, 5])

        axes = [_Axis() for _ in range(5)]
        _event_markers(axes, objective_component_series(combined), combined)
        expected = (
            "axvline",
            (5,),
            {"color": "#bc6c25", "linestyle": "--", "linewidth": 1, "alpha": 0.8},
        )
        self.assertTrue(all(axis.calls[0] == expected for axis in axes))
        self.assertTrue(
            any(
                call[0] == "scatter" and call[2]["marker"] == "X"
                for call in axes[0].calls
            )
        )


if __name__ == "__main__":
    unittest.main()
