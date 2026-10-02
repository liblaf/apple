"""Principal active-strain glyph geometry shared by comparison panels.

This is a display projection of the saved full ``B = I + S`` field.  The
forward solve uses every component of ``S``; one line cannot describe its
secondary modes.  In particular, a negative eigenvalue of unrestricted ``B``
changes the interpretation of its eigenvector, while ``B B.T`` remains a
squared stretch.  The signed scalar below is a display scale, not a physical
percentage of shortening.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PALETTE = ("#053061", "#4393c3", "#b5b6b4", "#d6604d", "#67001f")
COLOR_LIMITS_PERCENT = (-100.0, 100.0)
MAX_LINE_LENGTH_M = 0.0045
ZERO_TOL = 1e-8
GAP_TOL = 1e-6


@dataclass(frozen=True)
class PrincipalGlyphs:
    """One centered, unoriented line per active tetrahedron."""

    endpoints: np.ndarray  # (N, 2, 3), in deformed world coordinates
    centers: np.ndarray  # (N, 3)
    reference_axes: np.ndarray  # (N, 3)
    spatial_axes: np.ndarray  # (N, 3)
    eigenvalues_z: np.ndarray  # (N,)
    signed_display_percent: np.ndarray  # (N,)
    lengths_m: np.ndarray  # (N,)
    eligible: np.ndarray  # (N,), unique and nonneutral modes
    principal_unique: np.ndarray  # (N,)
    receipt: dict


def signed_display_percent(z: np.ndarray) -> np.ndarray:
    """Apply the old figure's common, bounded color and length transform."""
    z = np.asarray(z, dtype=np.float64)
    return 100.0 * np.sign(z) * (1.0 - 1.0 / np.sqrt(1.0 + np.abs(z)))


def signed_colormap() -> LinearSegmentedColormap:
    """Return the diverging color map for fixed limits of -100 to +100."""
    return LinearSegmentedColormap.from_list("signed_activation", PALETTE)


def contraction_colormap() -> LinearSegmentedColormap:
    """Return the contraction half of the original diverging color map."""
    return LinearSegmentedColormap.from_list("contraction_activation", PALETTE[2:])


def principal_glyphs(
    b: np.ndarray,
    f: np.ndarray,
    centers: np.ndarray,
    *,
    max_length_m: float = MAX_LINE_LENGTH_M,
) -> PrincipalGlyphs:
    """Project ``Z = B B.T - I`` onto deformed, principal-mode lines.

    ``f`` is the physical deformation gradient from rest to the saved shape.
    Its sole role here is to transport each reference eigenaxis.  The returned
    ``eligible`` mask excludes a nearly repeated largest eigenvalue or a
    neutral mode; visibility against the muscle surface is a separate mask.
    """
    b = np.asarray(b, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    count = len(b)
    assert b.shape == f.shape == (count, 3, 3)
    assert centers.shape == (count, 3)
    assert count > 0
    assert max_length_m > 0
    assert np.isfinite(b).all()
    assert np.isfinite(f).all()
    assert np.isfinite(centers).all()
    assert np.allclose(b, b.swapaxes(1, 2), rtol=0.0, atol=1e-10)

    z = b @ b.swapaxes(1, 2) - np.eye(3)
    values, axes = np.linalg.eigh(z)
    eigenvalue = values[:, -1]
    reference_axis = axes[:, :, -1]
    scale = np.maximum(1.0, np.max(np.abs(values), axis=1))
    gap = (values[:, -1] - values[:, -2]) / scale
    unique = gap > GAP_TOL
    eligible = unique & (np.abs(eigenvalue) > ZERO_TOL)

    residual = np.einsum("nij,nj->ni", z, reference_axis) - (
        eigenvalue[:, None] * reference_axis
    )
    normalized_residual = np.linalg.norm(residual, axis=1) / scale
    assert float(np.max(normalized_residual)) < 5e-13
    transported = np.einsum("nij,nj->ni", f, reference_axis)
    transported_norm = np.linalg.norm(transported, axis=1)
    assert np.isfinite(transported_norm).all()
    assert np.all(transported_norm > 0)
    spatial_axis = transported / transported_norm[:, None]

    signed = signed_display_percent(eigenvalue)
    length = max_length_m * np.abs(signed) / 100.0
    length[~eligible] = 0.0
    half = 0.5 * length[:, None] * spatial_axis
    endpoints = np.stack((centers - half, centers + half), axis=1)

    b_eigenvalues = np.linalg.eigvalsh(b)
    receipt = {
        "count": count,
        "principal_definition": "largest algebraic eigenpair of Z = B B.T - I",
        "direction": "normalize(F n) on the saved deformed geometry",
        "signed_display": "100 sign(z) (1 - 1/sqrt(1 + abs(z)))",
        "color_limits_percent": list(COLOR_LIMITS_PERCENT),
        "max_length_m": max_length_m,
        "zero_tolerance": ZERO_TOL,
        "principal_gap_tolerance": GAP_TOL,
        "gap_scale": "max(1, max(abs(all Z eigenvalues)))",
        "normalized_eigenpair_residual_max": float(np.max(normalized_residual)),
        "near_repeated_principal_count": int(np.count_nonzero(~unique)),
        "neutral_principal_count": int(
            np.count_nonzero(np.abs(eigenvalue) <= ZERO_TOL)
        ),
        "eligible_line_count": int(np.count_nonzero(eligible)),
        "negative_principal_z_count": int(np.count_nonzero(eigenvalue < -ZERO_TOL)),
        "negative_b_eigenvalue_count": int(np.count_nonzero(b_eigenvalues < -ZERO_TOL)),
        "b_has_negative_eigenvalue_tet_count": int(
            np.count_nonzero(np.min(b_eigenvalues, axis=1) < -ZERO_TOL)
        ),
        "principal_z_range": [float(eigenvalue.min()), float(eigenvalue.max())],
        "signed_display_range_percent": [float(signed.min()), float(signed.max())],
        "b_eigenvalue_range": [float(b_eigenvalues.min()), float(b_eigenvalues.max())],
        "display_limitation": "Principal mode only; negative B eigenvalues cannot be read as shortening from Z.",
    }
    return PrincipalGlyphs(
        endpoints=endpoints,
        centers=centers,
        reference_axes=reference_axis,
        spatial_axes=spatial_axis,
        eigenvalues_z=eigenvalue,
        signed_display_percent=signed,
        lengths_m=length,
        eligible=eligible,
        principal_unique=unique,
        receipt=receipt,
    )
