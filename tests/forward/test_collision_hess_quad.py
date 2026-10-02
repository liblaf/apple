from types import SimpleNamespace

import ipctk
import numpy as np
import scipy.sparse
import torch

from liblaf.apple.collision import Collision


class _Stencil:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dof_flags: list[tuple[bool, bool]] = []

    def dof(
        self, values: np.ndarray, edges: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        del edges, faces
        self.dof_flags.append((values.flags.c_contiguous, values.flags.f_contiguous))
        return values.flatten()


class _Potential:
    dhat = 1.0

    def __init__(self, terms: dict[str, float]) -> None:
        self.terms = terms
        self.calls: list[str] = []

    def gauss_newton_hessian_quadratic_form(
        self, collision: _Stencil, positions: np.ndarray, p: np.ndarray
    ) -> float:
        del positions, p
        self.calls.append(collision.name)
        return self.terms[collision.name]


def _collision(potential: _Potential) -> Collision:
    mesh = SimpleNamespace(
        edges=np.empty((0, 2), dtype=np.int32),
        faces=np.empty((0, 3), dtype=np.int32),
        rest_positions=np.zeros((2, 3), dtype=np.float64),
    )
    return Collision(
        collision_mesh=mesh,
        indices=torch.tensor([0, 1]),
        potential=potential,
    )


def _state(*names: str) -> SimpleNamespace:
    return SimpleNamespace(collisions=[_Stencil(name) for name in names])


def _real_contact_fixture() -> tuple[
    Collision, SimpleNamespace, torch.Tensor, torch.Tensor
]:
    vertices = np.asfortranarray(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 3.0, 0.0], [0.5, 3.0, 0.0]]
    )
    mesh = ipctk.CollisionMesh(vertices)
    positive = ipctk.VertexVertexNormalCollision(0, 1)
    negative = ipctk.VertexVertexNormalCollision(2, 3)
    negative.weight = 10.0
    collisions = ipctk.NormalCollisions()
    collisions.vv_collisions = [positive, negative]
    potential = ipctk.BarrierPotential(1.0, 1.0)
    contact = Collision(
        collision_mesh=mesh,
        indices=torch.arange(len(vertices)),
        potential=potential,
        vertices=torch.from_numpy(vertices.copy()),
    )
    direction = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    return (
        contact,
        SimpleNamespace(collisions=collisions),
        torch.zeros_like(direction),
        direction,
    )


def test_hess_quad_clamps_each_contact_contribution() -> None:
    potential = _Potential({"negative": -5.0, "positive": 2.0})
    collision = _collision(potential)
    u = torch.zeros((2, 3), dtype=torch.float64)
    p = torch.ones_like(u)

    assert collision.raw_hess_quad_terms(_state("negative", "positive"), u, p) == (
        -5.0,
        2.0,
    )
    torch.testing.assert_close(
        collision.hess_quad(_state("negative", "positive"), u, p),
        torch.tensor(2.0),
    )
    assert potential.calls == ["negative", "positive", "negative", "positive"]


def test_raw_hess_quad_terms_converts_torch_arrays_to_fortran_order() -> None:
    potential = _Potential({"contact": 1.0})
    collision = _collision(potential)
    state = _state("contact")
    u = torch.zeros((2, 3), dtype=torch.float64)

    collision.raw_hess_quad_terms(state, u, torch.ones_like(u))

    assert state.collisions[0].dof_flags == [(False, True), (False, True)]


def test_hess_quad_clamps_real_ipctk_contacts_individually() -> None:
    contact, state, u, direction = _real_contact_fixture()
    vertices = contact.vertices.numpy(force=True)
    direction_np = direction.numpy(force=True)
    native_terms = tuple(
        contact.potential.gauss_newton_hessian_quadratic_form(
            collision=collision,
            positions=collision.dof(
                vertices, contact.collision_mesh.edges, contact.collision_mesh.faces
            ),
            p=collision.dof(
                direction_np, contact.collision_mesh.edges, contact.collision_mesh.faces
            ),
        )
        for collision in state.collisions
    )
    native_aggregate = contact.potential.gauss_newton_hessian_quadratic_form(
        collisions=state.collisions,
        mesh=contact.collision_mesh,
        vertices=vertices,
        p=direction_np.flatten(),
    )

    raw_terms = contact.raw_hess_quad_terms(state, u, direction)

    torch.testing.assert_close(torch.tensor(raw_terms), torch.tensor(native_terms))
    assert native_terms[0] > 0.0 > native_terms[1]
    assert sum(raw_terms) == native_aggregate < 0.0
    torch.testing.assert_close(
        contact.hess_quad(state, u, direction), torch.tensor(native_terms[0])
    )
    assert max(native_aggregate, 0.0) == 0.0


def test_real_contact_energy_gradient_and_exact_hvp_stay_native() -> None:
    contact, state, u, direction = _real_contact_fixture()
    vertices = contact.vertices.numpy(force=True)
    native_energy = contact.potential(
        state.collisions, contact.collision_mesh, vertices
    )
    native_gradient = contact.potential.gradient(
        state.collisions, contact.collision_mesh, vertices
    )
    state.hess = contact.potential.hessian(
        state.collisions, contact.collision_mesh, vertices
    )
    output = torch.zeros_like(direction)

    torch.testing.assert_close(contact.fun(state, u), torch.tensor(native_energy))
    contact.grad(state, u, output)
    torch.testing.assert_close(
        output, torch.as_tensor(native_gradient).reshape(contact.vertices.shape)
    )
    output.zero_()
    contact.hess_prod(state, u, direction, output)
    torch.testing.assert_close(
        output,
        torch.as_tensor(state.hess @ direction.numpy(force=True).flatten()).reshape(
            contact.vertices.shape
        ),
    )


def test_hess_quad_empty_contacts_is_zero() -> None:
    potential = _Potential({})
    collision = _collision(potential)
    u = torch.zeros((2, 3), dtype=torch.float64)

    assert collision.raw_hess_quad_terms(_state(), u, u) == ()
    torch.testing.assert_close(collision.hess_quad(_state(), u, u), torch.tensor(0.0))
    assert potential.calls == []


def test_hess_quad_preserves_nan_for_the_solver_to_reject() -> None:
    potential = _Potential({"invalid": float("nan")})
    collision = _collision(potential)
    u = torch.zeros((2, 3), dtype=torch.float64)

    assert torch.isnan(collision.hess_quad(_state("invalid"), u, u))


def test_hess_prod_keeps_the_signed_exact_hessian() -> None:
    collision = _collision(_Potential({}))
    state = SimpleNamespace(
        hess=scipy.sparse.csc_matrix(np.diag([2.0, -3.0, 4.0, 5.0, 6.0, 7.0]))
    )
    u = torch.zeros((2, 3), dtype=torch.float64)
    p = torch.ones_like(u)
    output = torch.zeros_like(u)

    collision.hess_prod(state, u, p, output)

    torch.testing.assert_close(
        output,
        torch.tensor([[2.0, -3.0, 4.0], [5.0, 6.0, 7.0]], dtype=torch.float64),
    )
