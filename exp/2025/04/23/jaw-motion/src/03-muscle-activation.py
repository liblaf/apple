from collections.abc import Sequence
from pathlib import Path
from typing import override

import attrs
import einops
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float, PyTree

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, melon

DIRECTION: Float[np.ndarray, " 3"] = np.asarray([1.0, 0.0, 0.0])
PIVOT: Float[np.ndarray, " 3"] = np.asarray([0.0, 29.141, 0.8457])


class Config(cherries.BaseConfig):
    input: Path = Path("data/input.vtu")
    output: Path = Path("data/")


def compute_d_H3_d(F: jax.Array, d: jax.Array):
    # d: 9x1, H3 9x9
    d = d.T.flatten()
    d0, d1, d2, d3, d4, d5, d6, d7, d8 = (
        d[0],
        d[1],
        d[2],
        d[3],
        d[4],
        d[5],
        d[6],
        d[7],
        d[8],
    )
    F00, F01, F02, F10, F11, F12, F20, F21, F22 = (
        F[0, 0],
        F[0, 1],
        F[0, 2],
        F[1, 0],
        F[1, 1],
        F[1, 2],
        F[2, 0],
        F[2, 1],
        F[2, 2],
    )
    return 2 * (
        F00 * d4 * d8
        - F00 * d5 * d7
        - F01 * d1 * d8
        + F01 * d2 * d7
        + F02 * d1 * d5
        - F02 * d2 * d4
        - F10 * d3 * d8
        + F10 * d5 * d6
        + F11 * d0 * d8
        - F11 * d2 * d6
        - F12 * d0 * d5
        + F12 * d2 * d3
        + F20 * d3 * d7
        - F20 * d4 * d6
        - F21 * d0 * d7
        + F21 * d1 * d6
        + F22 * d0 * d4
        - F22 * d1 * d3
    )


@apple.register_dataclass()
@attrs.frozen(kw_only=True)
class CorotatedActivationElement(apple.material.tetra.MaterialTetraElement):
    arap: apple.material.tetra.AsRigidAsPossibleElement = attrs.field(
        init=False, factory=apple.material.tetra.AsRigidAsPossibleElement
    )

    @property
    @override
    def required_params(self) -> Sequence[str]:
        return ["lambda", "mu", "activation"]

    @override
    def strain_energy_density(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        lmbda: Float[jax.Array, ""] = q["lambda"]
        mu: Float[jax.Array, ""] = q["mu"]
        A: Float[jax.Array, "3 3"] = q["activation"]
        R: Float[jax.Array, "3 3"]
        F @= jnp.linalg.pinv(A)
        R, _S = apple.math.polar_rv(F)
        J: Float[jax.Array, ""] = jnp.linalg.det(F)
        return mu * apple.math.norm_sqr(F - R) + 0.5 * lmbda * (J - 1) ** 2

    @override
    def first_piola_kirchhoff_stress(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> jax.Array:
        lmbda: Float[jax.Array, ""] = q["lambda"]
        mu: Float[jax.Array, ""] = q["mu"]
        A: Float[jax.Array, "3 3"] = q["activation"]
        R: Float[jax.Array, "3 3"]
        F @= jnp.linalg.pinv(A)
        R, _S = apple.math.polar_rv(F)
        arap_PK1: Float[jax.Array, "3 3"] = 2.0 * mu * (F - R)
        J: Float[jax.Array, ""] = jnp.linalg.det(F)
        dJdF: Float[jax.Array, "3 3"] = apple.math.det_jac(F)
        return arap_PK1 + lmbda * (J - 1) * dJdF

    @override
    def strain_energy_density_hess_diag(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "4 3"]:
        lmbda: Float[jax.Array, ""] = q["lambda"]
        mu: Float[jax.Array, ""] = q["mu"]
        dh_dX: Float[jax.Array, "4 3"] = aux["dh_dX"]
        A: Float[jax.Array, "3 3"] = q["activation"]
        F @= jnp.linalg.pinv(A)
        U: Float[jax.Array, "3 3"]
        S_diag: Float[jax.Array, " 3"]
        VH: Float[jax.Array, "3 3"]
        U, S_diag, VH = apple.math.svd_rv(F)
        s0: Float[jax.Array, ""] = S_diag[0]
        s1: Float[jax.Array, ""] = S_diag[1]
        s2: Float[jax.Array, ""] = S_diag[2]
        lambda0: Float[jax.Array, ""] = 2.0 / (s1 + s2)
        lambda0 = jnp.where(s1 + s2 < 2.0, 1.0, lambda0)
        lambda1: Float[jax.Array, ""] = 2.0 / (s0 + s2)
        lambda1 = jnp.where(s0 + s2 < 2.0, 1.0, lambda1)
        lambda2: Float[jax.Array, ""] = 2.0 / (s0 + s1)
        lambda2 = jnp.where(s0 + s1 < 2.0, 1.0, lambda2)
        U0: Float[jax.Array, " 3"] = U[:, 0]
        U1: Float[jax.Array, " 3"] = U[:, 1]
        U2: Float[jax.Array, " 3"] = U[:, 2]
        V0: Float[jax.Array, " 3"] = VH[0, :]
        V1: Float[jax.Array, " 3"] = VH[1, :]
        V2: Float[jax.Array, " 3"] = VH[2, :]
        Q0: Float[jax.Array, "3 3"] = jnp.outer(V1, U2) - jnp.outer(V2, U1)
        Q1: Float[jax.Array, "3 3"] = jnp.outer(V2, U0) - jnp.outer(V0, U2)
        Q2: Float[jax.Array, "3 3"] = jnp.outer(V1, U0) - jnp.outer(V0, U1)
        W0: Float[jax.Array, "4 3"] = (
            apple.elem.tetra.deformation_gradient_vjp(dh_dX, Q0) ** 2
        )
        W1: Float[jax.Array, "4 3"] = (
            apple.elem.tetra.deformation_gradient_vjp(dh_dX, Q1) ** 2
        )
        W2: Float[jax.Array, "4 3"] = (
            apple.elem.tetra.deformation_gradient_vjp(dh_dX, Q2) ** 2
        )
        diag_h4: Float[jax.Array, "4 3"] = lambda0 * W0 + lambda1 * W1 + lambda2 * W2
        X: Float[jax.Array, "4 3"] = apple.elem.tetra.deformation_gradient_gram(dh_dX)
        diag_ARAP: Float[jax.Array, "4 3"] = mu * (2.0 * X - diag_h4)
        g3: Float[jax.Array, "3 3"] = apple.math.det_jac(F)
        dFdx_T_g3: Float[jax.Array, "4 3"] = apple.elem.tetra.deformation_gradient_vjp(
            dh_dX, g3
        )
        diag_J: Float[jax.Array, "4 3"] = lmbda * dFdx_T_g3**2
        return diag_ARAP + diag_J

    @override
    def strain_energy_density_hess_quad(
        self,
        F: Float[jax.Array, "3 3"],
        p: Float[jax.Array, "4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, ""]:
        lmbda: Float[jax.Array, ""] = q["lambda"]
        mu: Float[jax.Array, ""] = q["mu"]
        dh_dX: Float[jax.Array, "4 3"] = aux["dh_dX"]
        A: Float[jax.Array, "3 3"] = q["activation"]
        F @= jnp.linalg.pinv(A)
        dFdx_p: Float[jax.Array, "3 3"] = apple.elem.tetra.deformation_gradient_jvp(
            dh_dX, p
        )
        ret0: Float[jax.Array, ""] = apple.math.norm_sqr(dFdx_p)
        U: Float[jax.Array, "3 3"]
        S_diag: Float[jax.Array, " 3"]
        VH: Float[jax.Array, "3 3"]
        U, S_diag, VH = apple.math.svd_rv(F)
        s0: Float[jax.Array, ""] = S_diag[0]
        s1: Float[jax.Array, ""] = S_diag[1]
        s2: Float[jax.Array, ""] = S_diag[2]
        lambda0: Float[jax.Array, ""] = 2.0 / (s1 + s2)
        lambda0 = jnp.where(s1 + s2 < 2.0, 1.0, lambda0)
        lambda1: Float[jax.Array, ""] = 2.0 / (s0 + s2)
        lambda1 = jnp.where(s0 + s2 < 2.0, 1.0, lambda1)
        lambda2: Float[jax.Array, ""] = 2.0 / (s0 + s1)
        lambda2 = jnp.where(s0 + s1 < 2.0, 1.0, lambda2)
        V: Float[jax.Array, "3 3"] = VH.T
        U0: Float[jax.Array, " 3"] = U[:, 0]
        U1: Float[jax.Array, " 3"] = U[:, 1]
        U2: Float[jax.Array, " 3"] = U[:, 2]
        V0: Float[jax.Array, " 3"] = V[:, 0]
        V1: Float[jax.Array, " 3"] = V[:, 1]
        V2: Float[jax.Array, " 3"] = V[:, 2]
        Q0: Float[jax.Array, "3 3"] = jnp.outer(V1, U2) - jnp.outer(V2, U1)
        Q1: Float[jax.Array, "3 3"] = jnp.outer(V2, U0) - jnp.outer(V0, U2)
        Q2: Float[jax.Array, "3 3"] = jnp.outer(V1, U0) - jnp.outer(V0, U1)
        ret1: Float[jax.Array, ""] = (
            lambda0 * jnp.vdot(Q0, dFdx_p) ** 2
            + lambda1 * jnp.vdot(Q1, dFdx_p) ** 2
            + lambda2 * jnp.vdot(Q2, dFdx_p) ** 2
        )
        ret_ARAP: Float[jax.Array, ""] = mu * (2.0 * ret0 - ret1)
        g3: Float[jax.Array, "3 3"] = apple.math.det_jac(F)
        ret_2: Float[jax.Array, ""] = lmbda * jnp.vdot(g3, dFdx_p) ** 2
        J: Float[jax.Array, ""] = jnp.linalg.det(F)
        ret_3: Float[jax.Array, ""] = lmbda * (J - 1.0) * compute_d_H3_d(F, dFdx_p)
        ret_3 = jnp.where(ret_3 < 0, 0.0, ret_3)
        return ret_ARAP + ret_2 + ret_3


@apple.register_dataclass()
@attrs.frozen(kw_only=True)
class CorotatedActivation(apple.material.tetra.MaterialTetra):
    elem: apple.material.tetra.MaterialTetraElement = attrs.field(
        factory=CorotatedActivationElement, metadata={"static": True}
    )


def load_problem(cfg: Config) -> apple.PhysicsProblem:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    dirichlet_mask: Bool[jax.Array, " N"] = einops.repeat(
        jnp.asarray(mesh.point_data["is-skull"]), "V -> (V 3)"
    )
    dirichlet_values: Float[jax.Array, " N"] = jnp.zeros((mesh.n_points * 3,))
    face = apple.obj.tetra.ObjectTetra(
        name="face",
        params={
            "lambda": jnp.full((mesh.n_cells,), 532879.8185941039),
            "mu": jnp.full((mesh.n_cells,), 34013.60544217687),
        },
        material=CorotatedActivation(),
        mesh=mesh,
    )
    problem = apple.PhysicsProblem(
        objects=[face], dirichlet_mask=dirichlet_mask, dirichlet_values=dirichlet_values
    )
    return problem


def update_activation(
    face: apple.obj.ObjectTetra, activation_scalar: float = 1.0
) -> PyTree:
    muscle_mask: Bool[np.ndarray, " C"] = face.mesh.cell_data["muscle-fraction"] > 1e-3
    activation: Float[jax.Array, "C 3 3"] = jnp.zeros((face.mesh.n_cells, 3, 3))
    activation = activation.at[~muscle_mask].set(jnp.identity(3))
    activation_matrix: Float[jax.Array, "3 3"] = jnp.diagflat(
        jnp.asarray(
            [
                1.0 / jnp.sqrt(activation_scalar),
                activation_scalar,
                1.0 / jnp.sqrt(activation_scalar),
            ]
        )
    )
    activation = activation.at[muscle_mask].set(activation_matrix)
    return {face.name: {"activation": activation}}


def forward(
    problem: apple.PhysicsProblem,
    face: apple.obj.ObjectTetra,
    q: PyTree | None = None,
    u0: Float[jax.Array, " DoF"] | None = None,
) -> tuple[apple.MinimizeResult, pv.UnstructuredGrid]:
    problem.prepare()
    result: apple.MinimizeResult = problem.solve(
        q=q, u0=u0, algo=apple.MinimizePNCG(eps=1e-10, iter_max=150)
    )
    solution: Float[jax.Array, " DoF"] = result["x"]
    solution: Float[jax.Array, " F"] = problem.fill(solution)
    solution: Float[jax.Array, " N"] = face.select_dof(solution)
    solution: Float[jax.Array, "V 3"] = face.unravel_u(solution)
    output: pv.UnstructuredGrid = face.mesh.copy()
    output.point_data["solution"] = np.asarray(solution)
    jac: Float[jax.Array, " DoF"] = result["jac"]  # pyright: ignore[reportAssignmentType]
    jac: Float[jax.Array, " F"] = problem.fill_zeros(jac)
    jac: Float[jax.Array, " N"] = face.select_dof(jac)
    jac: Float[jax.Array, "V 3"] = face.unravel_u(jac)
    output.point_data["jac"] = np.asarray(jac)
    output.warp_by_vector("solution", inplace=True)
    return result, output


def main(cfg: Config) -> None:
    problem: apple.PhysicsProblem = load_problem(cfg)
    face: apple.obj.ObjectTetra = problem.objects[0]  # pyright: ignore[reportAssignmentType]
    writer_face = melon.SeriesWriter(cfg.output / "face.vtp.series")
    writer_mandible = melon.SeriesWriter(cfg.output / "mandible.vtp.series")
    writer_skull = melon.SeriesWriter(cfg.output / "skull.vtp.series")
    writer_solution = melon.SeriesWriter(cfg.output / "solution.vtu.series")
    u0: Float[jax.Array, " DoF"] | None = None
    for activation in np.linspace(1.0, 0.1, 10):
        ic(activation)
        q: PyTree = update_activation(face, activation)
        result: apple.MinimizeResult
        output: pv.UnstructuredGrid
        result, output = forward(problem, face, q, u0=u0)
        u0 = result["x"]
        writer_solution.append(output)
        surface: pv.PolyData = output.extract_surface()  # pyright: ignore[reportAssignmentType]
        writer_face.append(
            melon.triangle.extract_points(surface, surface.point_data["is-face"])
        )
        writer_mandible.append(
            melon.triangle.extract_points(surface, surface.point_data["is-mandible"])
        )
        writer_skull.append(
            melon.triangle.extract_points(surface, surface.point_data["is-skull"])
        )


if __name__ == "__main__":
    cherries.run(main)
