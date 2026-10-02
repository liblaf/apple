"""Geometry for the verified calibrated-normal slide."""

from __future__ import annotations

import hashlib
from typing import Any

import calibrated_study as ns
import numpy as np

LABELS = (
    "Free activation",
    "Contraction only\nfree directions",
    "Contraction only\nlearned direction",
    "Contraction only\nfixed x-direction",
)


def geometry(mode: str, variant: str, metric: dict, mesh: Any) -> dict:
    source = ns.GROUP / "data/10-comparison" / mode / variant / "history.npz"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metric["history_sha256"]
    with np.load(source, allow_pickle=False) as h:
        i = np.flatnonzero(h["steps"] == metric["step"])
        assert len(i) == 1
        p = h["points"]
        tri = h["triangles"]
        muscle = h["muscle"]
        deformed = p + h["u"][i[0]]
        B = ns.gs.study.matrices(mesh, h["controls"][i[0]], mode)
        top = h["top_all"]
    F = np.einsum("eia,eib->eab", deformed[tri], mesh.grad)
    np.testing.assert_allclose(np.linalg.det(F).min(), metric["min_J"], rtol=1e-10)
    values, directions = np.linalg.eigh(B[muscle] - np.eye(2))
    direction = F[muscle] @ directions
    direction /= np.linalg.norm(direction, axis=1)[:, None, :]
    centers = deformed[tri[muscle]].mean(axis=1)
    # Fixed glyph length, matching the original orientation convention F n / |F n|.
    delta = 0.00845 * direction.swapaxes(1, 2)
    segments = np.stack(
        (centers[:, None, :] - delta, centers[:, None, :] + delta), axis=2
    )
    return {
        "points": deformed,
        "tri": tri,
        "muscle": muscle,
        "top": top,
        "eigenvalues": values.ravel(),
        "segments": segments.reshape(-1, 2, 2),
    }


def eigenvalue(metric: dict) -> float:
    v = metric["physical_hessian"]["smallest_algebraic_eigenvalue"]
    assert v is not None, "Hessian audit must finish before labeling figure."
    return float(v)
