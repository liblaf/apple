"""Focused CPU behavior checks for actual affine-constrained pose increments."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from mouthopen_affine_direction import project_affine_increment, save_affine_cache
from mouthopen_pose_projection import PoseProjectionError


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        axis = np.array([1.0, 0, 0, 0, 0, 0])
        ordinary = save_affine_cache(
            root / "ordinary",
            original_ids=np.array([10]),
            old_j=np.array([1.0]),
            q_delta_j=np.array([0.0]),
            pose_jacobian=axis[None, :],
            residual_delta_j=np.array([0.0]),
            pose_gradient=-axis,
            requested_dp=0.2 * axis,
            strain_slope=-0.1,
            original_target=-0.03,
        )
        quarter, qreceipt = project_affine_increment(ordinary, 0.25, root / "quarter")
        half, hreceipt = project_affine_increment(ordinary, 0.5, root / "half")
        np.testing.assert_allclose(quarter, 0.05 * axis, atol=1e-12)
        np.testing.assert_allclose(half, 2 * quarter, atol=1e-12)
        assert (
            qreceipt["actual_joint_increment_slope"]
            == 0.5 * hreceipt["actual_joint_increment_slope"]
        )
        assert qreceipt["original_increment_target"] == -0.0075
        assert not ordinary.old_j.flags.writeable
        original = np.array([0.01, 1.0, -0.2])
        residual = np.array([-0.11, 0.0, 0.0])
        cache = save_affine_cache(
            root / "repair",
            original_ids=np.array([20, 30, 40]),
            old_j=original,
            q_delta_j=np.zeros(3),
            pose_jacobian=np.vstack((axis, -2 * axis, np.zeros(6))),
            residual_delta_j=residual,
            pose_gradient=-axis,
            requested_dp=axis,
            strain_slope=0.0,
            original_target=-0.1,
        )
        original[0] = 99  # The cache owns immutable copies.
        residual[0] = 99
        actual, receipt = project_affine_increment(cache, 1.0, root / "closure")
        np.testing.assert_allclose(actual, 0.4999995 * axis, atol=1e-10)
        assert receipt["initial_active_ids"] == [20]
        assert receipt["constraint_passes"][0]["violating_original_ids"] == [30]
        assert receipt["constraint_passes"][1]["active_original_ids"] == [20, 30]
        assert receipt["existing_nonpositive_cells"] == 1
        assert receipt["all_positive_cells_checked"]
        smaller, _ = project_affine_increment(cache, 0.01, root / "separate-residual")
        np.testing.assert_allclose(smaller, 0.100001 * axis, atol=1e-10)
        with np.load(root / "separate-residual/predictions.npz") as saved:
            np.testing.assert_allclose(
                saved["predicted_affine_det_f"] - saved["predicted_seed_det_f"],
                cache.residual_delta_j,
                atol=1e-14,
            )
            assert saved["predicted_seed_det_f"][0] > 0.11
        impossible = save_affine_cache(
            root / "impossible",
            original_ids=np.array([50]),
            old_j=np.array([0.01]),
            q_delta_j=np.zeros(1),
            pose_jacobian=np.zeros((1, 6)),
            residual_delta_j=np.array([-0.02]),
            pose_gradient=-axis,
            requested_dp=axis,
            strain_slope=0.0,
            original_target=-0.1,
        )
        try:
            project_affine_increment(impossible, 0.5, root / "failure")
        except PoseProjectionError:
            failed = json.loads((root / "failure/summary.json").read_text())
            assert failed["status"] == "projection_failed"
            assert (
                failed["constraint_passes"][0]["witness"]["geometry_descent_lp"][
                    "status"
                ]
                == 2
            )
            assert (root / "failure/constraints-00.npz").is_file()
            assert (root / "failure/input-receipt.json").is_file()
        else:
            message = "Infeasible affine geometry must fail visibly"
            raise AssertionError(message)
        try:
            project_affine_increment(
                cache, 1.0, root / "pass-budget", max_constraint_passes=1
            )
        except RuntimeError:
            failed = json.loads((root / "pass-budget/summary.json").read_text())
            assert failed["status"] == "projection_failed"
            assert failed["constraint_passes"][0]["violating_original_ids"] == [30]
        else:
            message = "Unclosed full-cell constraint set must fail"
            raise AssertionError(message)
    print(
        "CPU affine-direction checks passed: alpha scaling, separate residual, immutable cache, full-positive constraint addition, visible failures"
    )


if __name__ == "__main__":
    main()
