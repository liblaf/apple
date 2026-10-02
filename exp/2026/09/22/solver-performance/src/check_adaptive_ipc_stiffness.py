# ruff: noqa: E402, I001
"""CPU contracts for IPCTK adaptive barrier stiffness integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import ipctk
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import adaptive_ipc_stiffness
from adaptive_ipc_stiffness import AdaptiveIPCStiffness


class _Potential:
    def __init__(self, stiffness: float = 0.01) -> None:
        self.stiffness = stiffness
        self.barrier = object()
        self.dhat = 1e-4


class _NormalCollisions:
    def __init__(self, distance_squared: float, count: int = 1) -> None:
        self.distance_squared = distance_squared
        self.count = count

    def __len__(self) -> int:
        return self.count

    def compute_minimum_distance(self, _mesh: object, _vertices: object) -> float:
        return self.distance_squared


class _Collision:
    def __init__(self, distance_squared: float) -> None:
        self.vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
        )
        self.indices = torch.tensor([0, 1])
        self.collision_mesh = object()
        self.potential = _Potential()
        self.use_physical_barrier = True
        self.dmin = 0.0
        self.normal = _NormalCollisions(distance_squared)

    def state_at(self, _u: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(collisions=self.normal, hess=object())


class _Problem:
    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate(self) -> None:
        self.invalidations += 1


class _Hessian:
    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate(self) -> None:
        self.invalidations += 1


def _state(collision: _Collision) -> SimpleNamespace:
    u = torch.zeros((2, 3), dtype=torch.float64)
    return SimpleNamespace(u=u, collision=collision.state_at(u))


def check_pyipctk_trigger_uses_squared_distances_and_invalidates() -> None:
    # epsilon=(1e-6)*bbox=1e-6; both values below epsilon^2 and shrinking.
    collision = _Collision(4e-14)
    state = _state(collision)
    problem, hessian = _Problem(), _Hessian()
    controller = AdaptiveIPCStiffness(
        collision, initial_stiffness=0.01, epsilon_scale=1e-6
    )
    initial = controller.initialize(problem, state)
    assert initial["minimum_distance_squared"] == 4e-14
    assert controller.bbox_diagonal == 1.0
    collision.normal.distance_squared = 1e-14
    original = adaptive_ipc_stiffness.ipctk.BarrierPotential
    adaptive_ipc_stiffness.ipctk.BarrierPotential = (
        lambda _barrier, _dhat, stiffness, _physical: _Potential(stiffness)
    )
    try:
        row = controller.after_update(
            problem, state, phase="pncg", step=1, hessian=hessian
        )
    finally:
        adaptive_ipc_stiffness.ipctk.BarrierPotential = original
    assert row["stiffness_changed"]
    assert row["previous_minimum_distance_squared"] == 4e-14
    assert row["minimum_distance_squared"] == 1e-14
    assert row["stiffness_before"] == 0.01
    assert row["stiffness_after"] == 0.02
    assert problem.invalidations == hessian.invalidations == 1
    assert state.collision.hess is None
    receipt = controller.receipt()
    assert len(receipt["observations"]) == 2
    assert receipt["events"] == [row]


def check_monitor_records_identical_gap_without_objective_mutation() -> None:
    collision = _Collision(4e-14)
    state = _state(collision)
    controller = AdaptiveIPCStiffness(
        collision, initial_stiffness=0.01, enabled=False, epsilon_scale=1e-6
    )
    controller.initialize(_Problem(), state)
    collision.normal.distance_squared = 1e-14
    row = controller.after_update(_Problem(), state, phase="newton", step=1)
    assert not row["stiffness_changed"]
    assert row["proposed_stiffness"] == 0.02
    assert row["stiffness_after"] == 0.01
    assert collision.potential.stiffness == 0.01
    assert controller.receipt()["events"] == []


def check_empty_contact_is_inactive_then_reappearance_establishes_baseline() -> None:
    collision = _Collision(4e-14)
    collision.normal.count = 0
    state = _state(collision)
    problem, hessian = _Problem(), _Hessian()
    controller = AdaptiveIPCStiffness(
        collision, initial_stiffness=0.01, epsilon_scale=1e-6
    )
    initial = controller.initialize(problem, state)
    assert not initial["contact_active"]
    assert initial["minimum_distance_squared"] is None
    inactive = controller.after_update(problem, state, phase="pncg", step=1)
    assert not inactive["contact_active"]
    assert inactive["minimum_distance_m"] is None

    collision.normal.count = 1
    collision.normal.distance_squared = 4e-14
    reappeared = controller.after_update(problem, state, phase="pncg", step=2)
    assert reappeared["baseline_established"]
    assert not reappeared["stiffness_changed"]
    collision.normal.distance_squared = 1e-14
    original = adaptive_ipc_stiffness.ipctk.BarrierPotential
    adaptive_ipc_stiffness.ipctk.BarrierPotential = (
        lambda _barrier, _dhat, stiffness, _physical: _Potential(stiffness)
    )
    try:
        changed = controller.after_update(
            problem, state, phase="pncg", step=3, hessian=hessian
        )
    finally:
        adaptive_ipc_stiffness.ipctk.BarrierPotential = original
    assert changed["stiffness_changed"]
    assert problem.invalidations == hessian.invalidations == 1


def check_real_pyipctk_replacement_doubles_contact_mechanics() -> None:
    """A real replacement retains the physical barrier and doubles all terms."""
    vertices = np.asfortranarray([[0.0, 0.0, 0.0], [2e-7, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mesh = ipctk.CollisionMesh(vertices)
    collisions = ipctk.NormalCollisions()
    collisions.vv_collisions = [ipctk.VertexVertexNormalCollision(0, 1)]

    collision = SimpleNamespace(
        collision_mesh=mesh,
        indices=torch.arange(len(vertices)),
        potential=ipctk.BarrierPotential(
            dhat=1e-4, stiffness=0.01, use_physical_barrier=True
        ),
        dmin=0.0,
        use_physical_barrier=True,
        vertices=torch.from_numpy(vertices.copy()),
    )
    collision.state_at = lambda _u: SimpleNamespace(collisions=collisions, hess=None)
    state = SimpleNamespace(
        u=torch.zeros_like(collision.vertices),
        collision=collision.state_at(torch.empty(0)),
    )
    controller = AdaptiveIPCStiffness(
        collision, initial_stiffness=0.01, epsilon_scale=1e-6
    )
    problem, hessian = _Problem(), _Hessian()
    controller.initialize(problem, state)
    # Move to a smaller active gap, then measure both potentials at this same X.
    state.u[1, 0] = -1e-7
    positions = np.asfortranarray((collision.vertices + state.u).numpy(force=True))
    old = collision.potential
    old_energy = old(collisions, mesh, positions)
    old_gradient = old.gradient(collisions, mesh, positions)
    old_hessian = old.hessian(collisions, mesh, positions)
    event = controller.after_update(
        problem, state, phase="pncg", step=1, hessian=hessian
    )
    assert event["stiffness_changed"]
    new = collision.potential
    new_energy = new(collisions, mesh, positions)
    new_gradient = new.gradient(collisions, mesh, positions)
    new_hessian = new.hessian(collisions, mesh, positions)
    np.testing.assert_allclose(new_energy, 2 * old_energy, rtol=1e-12, atol=0)
    np.testing.assert_allclose(new_gradient, 2 * old_gradient, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        new_hessian.toarray(), 2 * old_hessian.toarray(), rtol=1e-12, atol=0
    )
    assert type(new.barrier) is type(old.barrier)
    assert new.dhat == old.dhat == 1e-4
    physical = ipctk.BarrierPotential(
        type(old.barrier)(), old.dhat, 0.02, use_physical_barrier=True
    )
    unphysical = ipctk.BarrierPotential(
        type(old.barrier)(), old.dhat, 0.02, use_physical_barrier=False
    )
    np.testing.assert_allclose(new_energy, physical(collisions, mesh, positions))
    assert not np.isclose(new_energy, unphysical(collisions, mesh, positions))


if __name__ == "__main__":
    check_pyipctk_trigger_uses_squared_distances_and_invalidates()
    check_monitor_records_identical_gap_without_objective_mutation()
    check_empty_contact_is_inactive_then_reappearance_establishes_baseline()
    check_real_pyipctk_replacement_doubles_contact_mechanics()
    print("adaptive IPC stiffness CPU checks passed")
