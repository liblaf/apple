"""Project local signed-volume constraints while keeping prescribed nodes fixed."""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger(__name__)


def determinants(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    edges = points[tets[:, 1:]] - points[tets[:, :1]]
    return np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))


def untangle_tetrahedra(
    reference: np.ndarray,
    candidate: np.ndarray,
    tets: np.ndarray,
    fixed_ids: np.ndarray,
    *,
    minimum_ratio: float = 0.05,
    max_sweeps: int = 200,
    step_cap_m: float = 2e-5,
) -> tuple[np.ndarray, dict]:
    """Untangle all nonfixed nodes; contact clearance must be rechecked afterward.

    Each update projects one signed tetrahedron volume toward a strictly
    positive fraction of its original volume. Sweeps include any neighbors
    newly affected by a projection. This is geometric preparation, not a FEM
    equilibrium solve, and it makes no contact-feasibility claim.
    """
    assert 0 < minimum_ratio < 1
    assert step_cap_m > 0
    x = np.asarray(candidate, dtype=np.float64).copy()
    rest = determinants(reference, tets)
    assert np.all(rest > 0)
    fixed = np.zeros(len(reference), dtype=bool)
    fixed[fixed_ids] = True
    assert np.array_equal(x[fixed], reference[fixed])
    target = minimum_ratio * rest
    trace = []
    for sweep in range(max_sweeps + 1):
        det = determinants(x, tets)
        ratio = det / rest
        bad = np.flatnonzero(ratio < minimum_ratio * (1 - 1e-8))
        row = {
            "sweep": sweep,
            "minimum_ratio": float(ratio.min()),
            "inverted": int(np.count_nonzero(ratio <= 0)),
            "below_target": len(bad),
        }
        trace.append(row)
        if sweep % 10 == 0 or not len(bad):
            LOG.info("Volume repair: %s", row)
        if not len(bad) or sweep == max_sweeps:
            break
        # Resolve the worst cells first. Recompute a cell after prior updates.
        for cid in bad[np.argsort(ratio[bad])]:
            ids = tets[cid]
            e = x[ids[1:]] - x[ids[0]]
            value = np.dot(e[0], np.cross(e[1], e[2]))
            if value >= target[cid]:
                continue
            grad = np.empty((4, 3), dtype=np.float64)
            grad[1] = np.cross(e[1], e[2])
            grad[2] = np.cross(e[2], e[0])
            grad[3] = np.cross(e[0], e[1])
            grad[0] = -grad[1:].sum(axis=0)
            grad[fixed[ids]] = 0
            denominator = float(np.sum(grad * grad))
            assert denominator > 0, (int(cid), ids.tolist())
            correction = (target[cid] - value) / denominator * grad
            largest = float(np.linalg.norm(correction, axis=1).max())
            if largest > step_cap_m:
                correction *= step_cap_m / largest
            x[ids] += correction
    assert np.array_equal(x[fixed], reference[fixed])
    receipt = {
        "schema": "reference-volume-projection-v1",
        "success": trace[-1]["below_target"] == 0,
        "minimum_requested_ratio": minimum_ratio,
        "final": trace[-1],
        "trace": trace,
        "maximum_correction_m": float(np.linalg.norm(x - candidate, axis=1).max()),
        "fixed_nodes_unchanged": True,
        "contact_clearance_checked": False,
    }
    return x, receipt
