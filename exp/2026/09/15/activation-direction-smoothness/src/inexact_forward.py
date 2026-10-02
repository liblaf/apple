"""Return the last accepted finite Newton iterate on recoverable forward failure.

The energy, derivatives, damping sequence, and positive-J Armijo rule are those
of physics2d.solve. Only its recoverable failure exits return a marked state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import study

ph = study.ph


@dataclass
class Outcome:
    state: Any
    converged: bool
    termination: str


def solve(
    mesh: Any,
    B: np.ndarray,
    seed: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 250,
) -> Outcome:
    u = seed.copy()
    for iteration in range(max_iterations + 1):
        energy, residual, H, J = ph.assemble(mesh, u, B)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(residual))
        assert np.all(np.isfinite(H.data))
        assert J.min() > 0
        state = ph.State(u, energy, residual, H, J, iteration)
        if np.linalg.norm(residual, np.inf) <= tolerance:
            return Outcome(state, converged=True, termination="converged")
        if iteration == max_iterations:
            return Outcome(state, converged=False, termination="iteration_limit")
        scale = max(np.max(np.abs(H.diagonal())), 1e-12)
        direction = None
        for damping in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0):
            d = spla.spsolve(
                H + damping * scale * sp.eye(mesh.nfree, format="csc"), -residual
            )
            if np.all(np.isfinite(d)) and residual @ d < 0:
                direction = d
                break
        if direction is None:
            return Outcome(state, converged=False, termination="no_descent_direction")
        slope = residual @ direction
        for backtrack in range(40):
            alpha = 0.5**backtrack
            trial = u + alpha * direction
            trial_energy, _, _, trial_J = ph.assemble(mesh, trial, B, hessian=False)
            rounding = 2e-15 * max(abs(energy), 1e-8)
            if (
                trial_J.min() > 1e-8
                and trial_energy <= energy + 1e-4 * alpha * slope + rounding
            ):
                u = trial
                break
        else:
            return Outcome(state, converged=False, termination="line_search_failed")
    raise AssertionError
