"""CPU checks for the MouthOpen jaw-direction feasibility projection."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

GROUP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GROUP / "src"))

from mouthopen_pose_projection import (  # noqa: E402
    PoseProjectionError,
    active_retained_noninverted_cells,
    determinant_directional_derivative,
    determinant_ratio,
    project_pose_descent_direction,
    project_pose_direction,
)


def projection_cases() -> dict:
    requested = np.array([-2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]])
    lower = np.array([0.0, 0.5])
    projected, receipt = project_pose_direction(requested, matrix, lower)
    np.testing.assert_allclose(projected, [0.0, 1.0, 0, 0, 0, 0], atol=1e-10)
    assert receipt["projection_applied"]
    assert receipt["constraints"]["maximum_violation"] <= 1e-10
    assert len(receipt["constraints"]["multipliers"]) == 2

    feasible = np.array([2.0, 1.0, 0, 0, 0, 0])
    unchanged, unchanged_receipt = project_pose_direction(feasible, matrix, lower)
    np.testing.assert_array_equal(unchanged, feasible)
    assert not unchanged_receipt["projection_applied"]

    try:
        project_pose_direction(
            np.zeros(6), np.zeros((1, 6)), np.ones(1), max_iterations=20
        )
    except PoseProjectionError:
        infeasible = True
    else:
        infeasible = False
    assert infeasible
    return {
        "projected": projected.tolist(),
        "unneeded_projection_exact": True,
        "infeasible_qp_fails": True,
    }


def witness_cases() -> dict:
    gradient = np.array([1.0, 0, 0, 0, 0, 0])
    matrix = gradient[None, :]
    lower = np.array([0.0])
    requested = -gradient

    def project(
        *,
        strain: float = -1.0,
        target: float = -2.0,
        bounds: np.ndarray = lower,
        diagnostics: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        return project_pose_descent_direction(
            requested,
            matrix,
            bounds,
            pose_gradient=gradient,
            strain_directional=strain,
            descent_target=target,
            feasible_witness=True,
            witness_receipt=diagnostics,
        )

    direction, receipt = project()
    np.testing.assert_allclose(direction, np.zeros(6), atol=1e-10)
    assert receipt["descent"]["joint_directional_upper_bound"] == -0.5
    assert receipt["feasible_witness"]["original_lp"]["status"] == 2
    assert receipt["feasible_witness"]["kkt_stationarity_residual"] <= 1e-7
    _, feasible = project(target=-0.5)
    assert feasible["feasible_witness"]["branch"] == (
        "original_target_feasible_start_from_lp_witness"
    )
    assert feasible["descent"]["joint_directional_upper_bound"] == -0.5
    for kwargs in ({"strain": 0.0}, {"bounds": np.array([0.1])}):
        try:
            project(**kwargs)
        except PoseProjectionError:
            pass
        else:
            message = "An unavailable descent witness must fail"
            raise AssertionError(message)
    for mocked in (
        patch(
            "mouthopen_descent_witness.linprog",
            return_value=SimpleNamespace(
                status=4,
                message="numerical failure",
                x=None,
            ),
        ),
        patch(
            "mouthopen_descent_witness.minimize",
            return_value=SimpleNamespace(
                status=8,
                success=False,
                message="unresolved",
                nit=1,
                x=np.zeros(6),
            ),
        ),
        # Feasible but nonoptimal result must fail its KKT check.
        patch(
            "mouthopen_descent_witness.minimize",
            return_value=SimpleNamespace(
                status=0,
                success=True,
                message="false success",
                nit=1,
                x=np.array([0.25, 0, 0, 0, 0, 0]),
            ),
        ),
    ):
        diagnostics = {}
        with mocked:
            try:
                project(diagnostics=diagnostics)
            except PoseProjectionError:
                assert "failure" in diagnostics
            else:
                message = "Unresolved LP/QP must fail visibly"
                raise AssertionError(message)
    # The previous policy does not opt into target changes.
    try:
        project_pose_descent_direction(
            requested,
            matrix,
            lower,
            pose_gradient=gradient,
            strain_directional=-1.0,
            descent_target=-2.0,
        )
    except PoseProjectionError:
        pass
    else:
        message = "Default policy must retain original target"
        raise AssertionError(message)
    return {
        "infeasible_target_uses_certified_witness": True,
        "feasible_target_preserved": True,
        "unresolved_and_uncertified_results_fail": True,
        "default_policy_unchanged": True,
    }


def attainable_witness_cases() -> dict:
    gradient = np.array([1.0, 0, 0, 0, 0, 0])
    matrix = gradient[None, :]
    lower = np.array([0.1])

    def project(
        *,
        strain: float = -1.0,
        target: float = -2.0,
        constraints: np.ndarray = matrix,
        bounds: np.ndarray = lower,
        diagnostics: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        return project_pose_descent_direction(
            -gradient,
            constraints,
            bounds,
            pose_gradient=gradient,
            strain_directional=strain,
            descent_target=target,
            feasible_witness=True,
            witness_policy="attainable_optimum",
            witness_receipt=diagnostics,
        )

    direction, receipt = project()
    np.testing.assert_allclose(direction, 0.1 * gradient, atol=1e-10)
    witness = receipt["feasible_witness"]
    assert witness["chosen_target"] == -0.45
    assert witness["geometry_descent_lp"]["best_joint_slope"] == -0.9
    assert witness["geometry_descent_lp"]["duality_gap"] == 0
    assert witness["geometry_descent_lp"]["dual_stationarity_residual"] == 0
    _, feasible = project(target=-0.5)
    assert feasible["feasible_witness"]["chosen_target"] == -0.5
    assert "geometry_descent_lp" not in feasible["feasible_witness"]
    for kwargs in (
        {"strain": 0.0},
        {"constraints": np.vstack((matrix, -matrix)), "bounds": np.ones(2)},
    ):
        diagnostics = {}
        try:
            project(**kwargs, diagnostics=diagnostics)
        except PoseProjectionError:
            assert "failure" in diagnostics
        else:
            message = "Infeasible geometry or non-descent must stop"
            raise AssertionError(message)
    for optimum in (
        SimpleNamespace(status=4, message="numerical failure", x=None),
        SimpleNamespace(status=3, message="unbounded conflicts with first LP", x=None),
        SimpleNamespace(
            status=0,
            message="invalid dual",
            x=0.1 * gradient,
            ineqlin=SimpleNamespace(marginals=np.zeros(1)),
        ),
    ):
        diagnostics = {}
        with patch(
            "mouthopen_descent_witness.linprog",
            side_effect=[
                SimpleNamespace(status=2, message="original target infeasible", x=None),
                optimum,
            ],
        ):
            try:
                project(diagnostics=diagnostics)
            except PoseProjectionError:
                assert "failure" in diagnostics
            else:
                message = "Uncertified geometry optimum must stop"
                raise AssertionError(message)
    return {
        "nonzero_witness_certified": True,
        "original_feasible_target_unchanged": True,
        "infeasible_nondescent_and_bad_certificates_fail": True,
    }


def determinant_case() -> dict:
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tets = np.array([[0, 1, 2, 3]])
    displacement = np.zeros((4, 3))
    det_f = determinant_ratio(reference, tets, displacement)
    np.testing.assert_allclose(det_f, [1.0])
    displacement[3, 2] = -0.98
    active = active_retained_noninverted_cells(
        reference, tets, displacement, np.array([0]), det_f_threshold=0.05
    )
    np.testing.assert_array_equal(active.original_tetrahedron_ids, [0])
    np.testing.assert_allclose(active.det_f, [0.02])
    return {
        "det_f": active.det_f.tolist(),
        "active_original_ids": active.original_tetrahedron_ids.tolist(),
    }


def analytic_determinant_cases() -> dict:
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tets = np.array([[0, 1, 2, 3]])
    generator = np.random.default_rng(20260930)
    for height in (1.0, 1e-7, 0.0, -0.1):
        displacement = np.zeros_like(reference)
        displacement[3, 2] = height - 1
        direction = generator.normal(size=reference.shape)
        actual = determinant_directional_derivative(
            reference, tets, displacement, direction
        )
        h = 1e-5
        central = (
            determinant_ratio(reference, tets, displacement + h * direction)
            - determinant_ratio(reference, tets, displacement - h * direction)
        ) / (2 * h)
        np.testing.assert_allclose(actual, central, rtol=1e-7, atol=1e-8)
        # With a fixed face, det(F) is exactly linear in the free apex motion,
        # even when the old cell is flat or already inverted.
        apex_direction = np.zeros_like(reference)
        apex_direction[3] = (0.2, -0.3, 0.4)
        np.testing.assert_allclose(
            determinant_directional_derivative(
                reference, tets, displacement, apex_direction
            ),
            [0.4],
            rtol=0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            determinant_ratio(reference, tets, displacement + apex_direction),
            determinant_ratio(reference, tets, displacement) + 0.4,
            rtol=0,
            atol=1e-15,
        )
        np.testing.assert_array_equal(
            determinant_directional_derivative(
                reference, tets, displacement, np.ones_like(reference)
            ),
            [0.0],
        )
    return {
        "singular_and_inverted_cells": True,
        "fixed_face_exact_linearity": True,
        "rigid_translation_zero": True,
    }


def main() -> None:
    print(
        {
            "projection": projection_cases(),
            "witness": witness_cases(),
            "attainable_witness": attainable_witness_cases(),
            "determinant": determinant_case(),
            "analytic_determinant": analytic_determinant_cases(),
        }
    )


if __name__ == "__main__":
    main()
