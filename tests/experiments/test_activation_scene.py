"""Scientific identities for the active-strain comparison glyphs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOURCE = (
    Path(__file__).resolve().parents[2] / "exp/2026/09/21/stress-activation-loss/src"
)
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from activation_scene import MAX_LINE_LENGTH_M, principal_glyphs  # noqa: E402


def test_principal_mode_transports_with_physical_deformation() -> None:
    b = np.diag((1.2, 1.0, 1.0))[None]
    # The reference x axis maps to the spatial y axis, with a nonunit stretch.
    f = np.array([[[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    center = np.array([[0.1, 0.2, 0.3]])
    result = principal_glyphs(b, f, center)
    expected_percent = 100 * (1 - 1 / np.sqrt(1.44))
    np.testing.assert_allclose(result.eigenvalues_z, [0.44], atol=1e-14)
    np.testing.assert_allclose(result.signed_display_percent, [expected_percent])
    np.testing.assert_allclose(
        np.abs(result.spatial_axes), [[0.0, 1.0, 0.0]], atol=1e-14
    )
    np.testing.assert_allclose(result.endpoints.mean(axis=1), center)
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(result.endpoints, axis=1)[:, 0], axis=1),
        [MAX_LINE_LENGTH_M * expected_percent / 100],
    )
    assert bool(result.eligible[0])


def test_negative_b_eigenvalue_is_audited_as_squared_action() -> None:
    positive = principal_glyphs(
        np.diag((1.2, 1.0, 1.0))[None], np.eye(3)[None], np.zeros((1, 3))
    )
    negative = principal_glyphs(
        np.diag((-1.2, 1.0, 1.0))[None], np.eye(3)[None], np.zeros((1, 3))
    )
    np.testing.assert_allclose(
        positive.signed_display_percent, negative.signed_display_percent
    )
    assert negative.receipt["negative_b_eigenvalue_count"] == 1
    assert negative.receipt["b_has_negative_eigenvalue_tet_count"] == 1


def test_near_repeated_axis_is_hidden_and_all_negative_z_remains_signed() -> None:
    b = np.stack((np.diag((1.2, 1.2, 1.0)), np.diag((0.8, 0.9, 0.7))))
    result = principal_glyphs(b, np.tile(np.eye(3), (2, 1, 1)), np.zeros((2, 3)))
    assert not bool(result.eligible[0])
    assert result.lengths_m[0] == 0
    assert bool(result.eligible[1])
    np.testing.assert_allclose(result.eigenvalues_z[1], 0.9**2 - 1)
    assert result.signed_display_percent[1] < 0
    assert result.receipt["near_repeated_principal_count"] == 1
