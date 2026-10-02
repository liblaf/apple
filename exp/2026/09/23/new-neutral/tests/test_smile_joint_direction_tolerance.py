"""Dual polishing may stop at L-BFGS-B tolerance; final KKT gates remain required."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).parents[1] / "src/smile_joint_direction.py"


def load_module():
    spec = importlib.util.spec_from_file_location("smile_joint_direction", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_near_duplicate_active_rows_certify_at_shared_tolerance():
    module = load_module()
    matrix = np.array([[1.0], [1.0 + 1e-14]])
    increment, certificate = module.solve_bounded_dual(
        matrix,
        np.array([1.0, 1.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([-np.inf]),
        np.array([np.inf]),
    )
    assert increment[0] >= 1.0
    assert certificate["certified"] is True
    assert certificate["dual_projected_gradient_tolerance"] == 1e-11
    assert certificate["polish_termination"] == "projected_gradient_tolerance"
    assert certificate["polish_termination_norm"] <= 1e-11


def test_final_kkt_assertions_remain_in_the_solver():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "solve_bounded_dual"
    )
    assertions = [
        ast.unparse(item.test)
        for item in ast.walk(function)
        if isinstance(item, ast.Assert)
    ]
    assert "slack.min() >= -1e-10" in assertions
    assert "normalized_slack.min() >= -1e-08" in assertions
    assert "max(abs(normalized_complementarity)) <= 1e-08" in assertions
    assert "abs(gap / rhs_scale ** 2) <= 1e-08" in assertions
    assert "box_sign_violation <= 1e-10" in assertions
    assert "max(abs(box_stationarity)) <= 1e-10" in assertions
