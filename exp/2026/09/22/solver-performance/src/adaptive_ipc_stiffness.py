# ruff: noqa: EM101, PT018, TRY003
"""Opt-in adaptive IPC barrier stiffness for one forward solve.

The controller deliberately owns only a forward solve transaction.  It uses
IPCTK's update rule verbatim, but makes the threshold an explicit experiment
parameter and invalidates every cached derivative representation when kappa
changes.
"""

from __future__ import annotations

import math
from typing import Any

import ipctk
import numpy as np


class AdaptiveIPCStiffness:
    """Monitor or adapt one collision barrier after accepted solver updates."""

    def __init__(
        self,
        collision: Any,
        *,
        initial_stiffness: float,
        enabled: bool = True,
        epsilon_scale: float = 1e-6,
        max_stiffness: float | None = None,
    ) -> None:
        if not math.isfinite(epsilon_scale) or epsilon_scale <= 0:
            raise ValueError("epsilon_scale must be finite and positive")
        if not math.isfinite(initial_stiffness) or initial_stiffness <= 0:
            raise ValueError("initial_stiffness must be finite and positive")
        if max_stiffness is not None and (
            not math.isfinite(max_stiffness) or max_stiffness <= 0
        ):
            raise ValueError("max_stiffness must be finite and positive")
        self.collision = collision
        self._initial_stiffness = initial_stiffness
        self.current_stiffness = initial_stiffness
        self.enabled = enabled
        self.epsilon_scale = epsilon_scale
        self._requested_max_stiffness = max_stiffness
        self.initial_stiffness: float | None = None
        self.max_stiffness: float | None = None
        self.bbox_diagonal: float | None = None
        self.previous_min_distance_squared: float | None = None
        self.initialized = False
        self.observations: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def _vertices(self, state: Any) -> np.ndarray:
        vertices = self.collision.vertices + state.u[self.collision.indices]
        return np.asfortranarray(vertices.numpy(force=True))

    def _minimum_distance_squared(self, state: Any) -> float | None:
        if state.collision is None:
            state.collision = self.collision.state_at(state.u)
        if len(state.collision.collisions) == 0:
            return None
        value = float(
            state.collision.collisions.compute_minimum_distance(
                self.collision.collision_mesh, self._vertices(state)
            )
        )
        if not math.isfinite(value) or value < 0:
            raise ValueError("IPCTK returned an invalid squared minimum distance")
        return value

    def initialize(self, _problem: Any, state: Any) -> dict[str, Any]:
        """Capture immutable geometry scale and the initial squared gap once."""
        if self.initialized:
            raise RuntimeError("adaptive IPC stiffness controller initialized twice")
        self.initialized = True
        reference = np.asarray(self.collision.vertices.numpy(force=True))
        extent = reference.max(axis=0) - reference.min(axis=0)
        self.bbox_diagonal = float(np.linalg.norm(extent))
        if not math.isfinite(self.bbox_diagonal) or self.bbox_diagonal <= 0:
            raise ValueError(
                "collision reference bounding-box diagonal must be positive"
            )
        self.initial_stiffness = self._initial_stiffness
        self.max_stiffness = (
            100 * self.initial_stiffness
            if self._requested_max_stiffness is None
            else self._requested_max_stiffness
        )
        if self.max_stiffness < self.initial_stiffness:
            raise ValueError("max_stiffness must not be below initial stiffness")
        self.previous_min_distance_squared = self._minimum_distance_squared(state)
        row = {
            "phase": "initial",
            "step": 0,
            "minimum_distance_squared": self.previous_min_distance_squared,
            "minimum_distance_m": (
                math.sqrt(self.previous_min_distance_squared)
                if self.previous_min_distance_squared is not None
                else None
            ),
            "stiffness": self.initial_stiffness,
            "contact_active": self.previous_min_distance_squared is not None,
        }
        self.observations.append(row)
        return row

    @staticmethod
    def _invalidate(problem: Any, state: Any, hessian: Any | None) -> None:
        invalidate = getattr(problem, "invalidate", None)
        if callable(invalidate):
            invalidate()
        if state.collision is not None:
            state.collision.hess = None
        if hessian is not None:
            hessian.invalidate()

    def after_update(
        self,
        problem: Any,
        state: Any,
        *,
        phase: str,
        step: int,
        hessian: Any | None = None,
    ) -> dict[str, Any]:
        """Observe a newly accepted state and update kappa between iterations.

        A Newton line search calls this only after accepting a displacement, so
        each line search always sees one fixed physical objective.
        """
        if not self.initialized:
            raise RuntimeError("adaptive IPC stiffness controller was not initialized")
        assert self.max_stiffness is not None and self.bbox_diagonal is not None
        previous = self.previous_min_distance_squared
        current = self._minimum_distance_squared(state)
        stiffness_before = self.current_stiffness
        if current is None:
            self.previous_min_distance_squared = None
            row = {
                "phase": phase,
                "step": step,
                "previous_minimum_distance_squared": previous,
                "minimum_distance_squared": None,
                "minimum_distance_m": None,
                "stiffness_before": stiffness_before,
                "stiffness_after": stiffness_before,
                "proposed_stiffness": None,
                "stiffness_changed": False,
                "contact_active": False,
            }
            self.observations.append(row)
            return row
        if previous is None:
            self.previous_min_distance_squared = current
            row = {
                "phase": phase,
                "step": step,
                "previous_minimum_distance_squared": None,
                "minimum_distance_squared": current,
                "minimum_distance_m": math.sqrt(current),
                "stiffness_before": stiffness_before,
                "stiffness_after": stiffness_before,
                "proposed_stiffness": None,
                "stiffness_changed": False,
                "contact_active": True,
                "baseline_established": True,
            }
            self.observations.append(row)
            return row
        proposed = ipctk.update_barrier_stiffness(
            previous,
            current,
            self.max_stiffness,
            stiffness_before,
            self.bbox_diagonal,
            self.epsilon_scale,
            self.collision.dmin,
        )
        stiffness_after = stiffness_before
        changed = self.enabled and proposed > stiffness_before
        if changed:
            stiffness_after = float(proposed)
            potential = self.collision.potential
            self.collision.potential = ipctk.BarrierPotential(
                type(potential.barrier)(),
                potential.dhat,
                stiffness_after,
                self.collision.use_physical_barrier,
            )
            self.current_stiffness = stiffness_after
            self._invalidate(problem, state, hessian)
        self.previous_min_distance_squared = current
        row = {
            "phase": phase,
            "step": step,
            "previous_minimum_distance_squared": previous,
            "minimum_distance_squared": current,
            "minimum_distance_m": math.sqrt(current),
            "stiffness_before": stiffness_before,
            "stiffness_after": stiffness_after,
            "proposed_stiffness": float(proposed),
            "stiffness_changed": changed,
            "contact_active": True,
        }
        self.observations.append(row)
        if changed:
            self.events.append(row)
        return row

    def receipt(self) -> dict[str, Any]:
        """Return JSON-safe provenance for this forward-only controller."""
        return {
            "enabled": self.enabled,
            "epsilon_scale": self.epsilon_scale,
            "bbox_diagonal_m": self.bbox_diagonal,
            "initial_stiffness": self.initial_stiffness,
            "max_stiffness": self.max_stiffness,
            "final_stiffness": self.current_stiffness,
            "observations": self.observations,
            "events": self.events,
        }
