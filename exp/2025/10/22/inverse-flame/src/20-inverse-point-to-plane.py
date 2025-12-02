import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Array, Bool, Float, Integer
from liblaf.peach import tree
from liblaf.peach.linalg import JaxCG, JaxCompositeSolver, LinearSystem
from liblaf.peach.optim import PNCG, Objective, ScipyOptimizer

from liblaf import cherries, grapes, melon
from liblaf.apple import sim
from liblaf.apple.warp import sim as sim_wp
from liblaf.apple.warp import utils as wpu
from liblaf.apple.warp.typing import vec6

logger: logging.Logger = logging.getLogger(__name__)
tree.register_fieldz(melon.NearestPointOnSurfaceResult)


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")
    target: Path = cherries.input("10-target.vtu")

    output: Path = cherries.output("20-inverse-point-to-plane.vtu")


@tree.define
class Params:
    activation: Float[Array, "a 6"]


class Aux(TypedDict):
    point_to_plane: Float[Array, ""]
    point_to_point: Float[Array, ""]
    sparse: Float[Array, ""]


@tree.define
class LossFn:
    face_point_idx: Integer[Array, " p"] = tree.array(default=None)
    nearest_point_on_surface: Float[Array, "p 3"] = tree.array(default=None)
    nearest_point_missing: Bool[Array, " p"] = tree.array(default=None)
    nearest_point_normal: Float[Array, "p 3"] = tree.array(default=None)
    rest_pos: Float[Array, "p 3"] = tree.array(default=None)
    target_disp: Float[Array, "p 3"] = tree.array(default=None)
    target_normal: Float[Array, "p 3"] = tree.array(default=None)

    muscle_id_to_idx: dict[int, Integer[Array, " m"]] = tree.field(default=None)
    muscle_id_to_volume: dict[int, Float[Array, " m"]] = tree.field(default=None)

    weight_point_to_point: float = 1.0
    weight_point_to_plane: float = 1.0
    weight_sparse: float = 1e-3

    @property
    def n_points(self) -> int:
        return self.target_disp.shape[0]

    def loss(
        self, u: Float[Array, "p 3"], act: Float[Array, "c 6"]
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        point_to_point: Float[Array, ""] = self.point_to_point(u)
        point_to_plane: Float[Array, ""] = self.point_to_plane(u)
        sparse: Float[Array, ""] = self.sparse(act)
        loss: Float[Array, ""] = point_to_point + point_to_plane + sparse
        return loss, {
            "total": loss,
            "point_to_point": point_to_point,
            "point_to_plane": point_to_plane,
            "sparse": sparse,
        }

    @eqx.filter_jit
    def loss_and_grad(
        self, u: Float[Array, "p 3"], act: Float[Array, "c 6"]
    ) -> tuple[
        tuple[Float[Array, ""], Aux], tuple[Float[Array, "p 3"], Float[Array, "c 6"]]
    ]:
        loss: Float[Array, ""]
        (loss, aux), (dLdu, dLdq) = jax.value_and_grad(
            self.loss, argnums=(0, 1), has_aux=True
        )(u, act)
        return (loss, aux), (dLdu, dLdq)

    def point_to_point(self, u: Float[Array, "p 3"]) -> Float[Array, ""]:
        if self.weight_point_to_point == 0.0:
            return jnp.zeros(())
        u = u[self.face_point_idx]
        residual: Float[Array, "p 3"] = u - self.target_disp
        loss: Float[Array, ""] = 0.5 * jnp.sum(jnp.square(residual)) / self.n_points
        return self.weight_point_to_point * loss

    def point_to_plane(self, u: Float[Array, "p 3"]) -> Float[Array, ""]:
        if self.weight_point_to_plane == 0.0:
            return jnp.zeros(())
        u = u[self.face_point_idx]
        # residual: Float[Array, "p 3"] = (
        #     self.rest_pos + u
        # ) - self.nearest_point_on_surface
        # normal: Float[Array, "p 3"] = residual / (
        #     jnp.linalg.norm(residual, axis=-1, keepdims=True) + 1e-3
        # )
        residual: Float[Array, "p 3"] = u - self.target_disp
        normal: Float[Array, "p 3"] = self.target_normal
        residual: Float[Array, " p"] = jnp.vecdot(residual, normal)
        # residual = jnp.where(self.nearest_point_missing, 0.0, residual)
        loss: Float[Array, ""] = 0.5 * jnp.sum(jnp.square(residual)) / self.n_points
        return self.weight_point_to_plane * loss

    def sparse(self, act: Float[Array, "c 6"]) -> Float[Array, ""]:
        if self.weight_sparse == 0.0:
            return jnp.zeros(())
        loss: Float[Array, ""] = jnp.zeros(())
        total_volume: Float[Array, ""] = jnp.zeros(())
        for i, indices in self.muscle_id_to_idx.items():
            act_i: Float[Array, " m 6"] = act[indices]
            vol_i: Float[Array, " m"] = self.muscle_id_to_volume[i]
            mag: Float[Array, " m"] = jnp.sum(jnp.square(act_i), axis=-1)
            loss += jnp.vdot(vol_i, mag)
            total_volume += jnp.sum(vol_i)
        loss /= total_volume
        return self.weight_sparse * loss


@tree.define
class Inverse:
    face_point_idx: Integer[Array, " f"]
    loss_fn: LossFn
    model: sim.Model
    muscle_cell_idx: Integer[Array, " a"]
    n_cells: int
    rest_pos: Float[Array, "p 3"]
    face: pv.PolyData
    nearest: melon.NearestPointOnSurfacePrepared = tree.field(default=None)
    optimizer: PNCG = tree.field(factory=lambda: PNCG(rtol=1e-6, max_steps=5000))
    step: int = 0
    u: Float[Array, "p 3"] = tree.field(default=None)

    @property
    def energy(self) -> sim_wp.Phace:
        return self.model.model_warp.energies["elastic"]  # pyright: ignore[reportReturnType]

    def value_and_grad(self, params: Params) -> tuple[Float[Array, ""], Params]:
        act: Float[Array, "c 6"]
        act_vjp: Callable[[Float[Array, "c 6"]], Params]
        act, act_vjp = jax.vjp(self.make_activation, params)
        self.set_activation(act)
        u: Float[Array, "p 3"] = self.forward()
        self.update(u)
        (loss, aux), (dLdu, dLdq) = self.loss_and_grad(u, act)
        cherries.log_metrics(aux, step=self.step)
        self.step += 1
        p: Float[Array, "p 3"] = self.adjoint(u, dLdu)
        outputs: dict[str, dict[str, Array]] = self.model.mixed_derivative_prod(u, p)
        act_grad: Float[Array, "c 6"] = dLdq + outputs[self.energy.id]["activation"]
        grad: Params = act_vjp(act_grad)
        return loss, grad

    def make_activation(self, params: Params) -> Float[Array, "c 6"]:
        activation: Float[Array, "c 6"] = jnp.zeros((self.n_cells, 6))
        activation = activation.at[self.muscle_cell_idx].set(params.activation)
        return activation

    def set_activation(self, activation: Float[Array, "c 6"]) -> None:
        wp.copy(self.energy.params.activation, wpu.to_warp(activation, dtype=vec6))

    def forward(self) -> Float[Array, "p 3"]:
        objective = Objective(
            fun=self.model.fun,
            grad=self.model.jac,
            hess_diag=self.model.hess_diag,
            hess_prod=self.model.hess_prod,
            hess_quad=self.model.hess_quad,
            value_and_grad=self.model.fun_and_jac,
            grad_and_hess_diag=self.model.jac_and_hess_diag,
        )
        u_free: Float[Array, " free"] = jnp.zeros((self.model.n_free,))
        solution: PNCG.Solution = self.optimizer.minimize(
            objective=objective, params=u_free
        )
        u_free = solution.params
        logger.info(
            "Forward time: %g sec, steps: %d, success: %s",
            solution.stats.time,
            solution.stats.n_steps,
            solution.success,
        )
        # assert solution.success
        u_full: Float[Array, "p 3"] = self.model.to_full(u_free)
        return u_full

    def adjoint(
        self, u: Float[Array, "p 3"], dLdu: Float[Array, "p 3"]
    ) -> Float[Array, "p 3"]:
        u_free: Float[Array, " free"] = self.model.dirichlet.get_free(u)
        preconditioner: Float[Array, " free"] = jnp.reciprocal(
            self.model.dirichlet.get_free(self.model.hess_diag(u))
        )
        solver = JaxCompositeSolver(
            solvers=[
                JaxCG(max_steps=self.model.n_free // 10, rtol=1e-6),
                # JaxGMRES(max_steps=self.model.n_free // 10, rtol=1e-6),
            ]
        )

        # @jax.custom_jvp
        def matvec(p_free: Float[Array, " free"]) -> Float[Array, " free"]:
            return self.model.hess_prod(u_free, p_free)

        # @matvec.defjvp
        # def matvec_jvp(
        #     primals: tuple[Float[Array, " free"]],
        #     tangents: tuple[Float[Array, " free"]],
        # ) -> tuple[Float[Array, " free"], Float[Array, " free"]]:
        #     p_free: Float[Array, " free"]
        #     (p_free,) = primals
        #     tangent: Float[Array, " free"]
        #     (tangent,) = tangents
        #     res: Float[Array, " free"] = matvec(p_free)
        #     res = jax.lax.stop_gradient(res)
        #     return res, jax.lax.stop_gradient(matvec(tangent))

        # def matvec_fwd(
        #     p_free: Float[Array, " free"], y: Float[Array, " free"]
        # ) -> tuple[Float[Array, " free"], None]:
        #     return matvec(p_free), None

        # def matvec_bwd(res: None, g: Float[Array, " free"]) -> Float[Array, " free"]:
        #     return self.model.hess_quad(u_free, g)

        # matvec.defvjp(matvec_fwd, matvec_bwd)

        system = LinearSystem(
            matvec,
            b=-self.model.dirichlet.get_free(dLdu),
            preconditioner=lambda p_free: preconditioner * p_free,
        )
        solution: JaxCompositeSolver.Solution = solver.solve(
            system, jnp.zeros((self.model.n_free,))
        )
        logger.info(
            "Adjoint time: %g sec, success: %s", solution.stats.time, solution.success
        )
        # assert solution.success
        return self.model.to_full(solution.params, zero=True)

    def loss_and_grad(
        self, u: Float[Array, "p 3"], act: Float[Array, "c 6"]
    ) -> tuple[
        tuple[Float[Array, ""], Aux], tuple[Float[Array, "p 3"], Float[Array, "c 6"]]
    ]:
        return self.loss_fn.loss_and_grad(u, act)

    def set_target(self, target: pv.UnstructuredGrid, data_name: str) -> None:
        self.step = 0

        target.point_data["PointId"] = np.arange(target.n_points)
        target.point_data["Normals"] = np.zeros((target.n_points, 3))
        target_surface: pv.PolyData = target.extract_surface()  # pyright: ignore[reportAssignmentType]
        target_surface.warp_by_vector(data_name, inplace=True)
        target.point_data["Normals"][target_surface.point_data["PointId"]] = (
            target_surface.point_normals
        )
        self.loss_fn.target_disp = jnp.asarray(target.point_data[data_name])[
            self.face_point_idx
        ]
        self.loss_fn.target_normal = jnp.asarray(target.point_data["Normals"])[
            self.face_point_idx
        ]

        target_face: pv.PolyData = target.extract_surface()  # pyright: ignore[reportAssignmentType]
        target_face = target_face.extract_points(
            target_face.point_data["IsFace"]
        ).extract_surface()  # pyright: ignore[reportAssignmentType]
        target_face.warp_by_vector(data_name, inplace=True)
        melon.save(cherries.temp("20-inverse-target.vtp"), target_face)
        self.nearest = melon.NearestPointOnSurface(
            distance_threshold=1.0, normal_threshold=None
        ).prepare(target_face)

    def update(self, u: Float[Array, "p 3"]) -> None:
        self.u = u
        # self.face.point_data["Displacement"] = np.asarray(
        #     u[self.face.point_data["PointId"]]
        # )
        # deformed: pv.PolyData = self.face.warp_by_vector("Displacement", inplace=False)  # pyright: ignore[reportAssignmentType]
        nearest_result = self.nearest.query(self.rest_pos + u[self.face_point_idx])
        # face_mask = jnp.asarray(self.face.point_data["IsFace"])
        self.loss_fn.nearest_point_on_surface = jnp.asarray(nearest_result.nearest)
        self.loss_fn.nearest_point_missing = jnp.asarray(nearest_result.missing)
        self.loss_fn.nearest_point_normal = jnp.asarray(
            self.nearest.source_pv.face_normals[nearest_result.triangle_id]
        )


def prepare(mesh: pv.UnstructuredGrid) -> Inverse:
    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    energy: sim_wp.Phace = sim_wp.Phace.from_pyvista(
        mesh, id="elastic", requires_grad=("activation",)
    )
    builder.add_energy(energy)
    model: sim.Model = builder.finish()

    MUSCLE_FRACTION_THRESHOLD: float = 1e-2
    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    face_point_idx: Integer[Array, " f"] = jnp.flatnonzero(mesh.point_data["IsFace"])
    muscle_fraction: Float[Array, " c"] = jnp.asarray(mesh.cell_data["MuscleFraction"])
    muscle_id: Integer[Array, " c"] = jnp.asarray(mesh.cell_data["MuscleId"])
    muscle_idx: Integer[Array, " a"] = jnp.flatnonzero(
        muscle_fraction > MUSCLE_FRACTION_THRESHOLD
    )
    volume: Float[Array, " c"] = jnp.asarray(mesh.cell_data["Volume"])
    muscle_id_to_idx: dict[int, Integer[Array, " m"]] = {}
    muscle_id_to_volume: dict[int, Float[Array, " m"]] = {}
    for i in grapes.track(range(jnp.max(muscle_id) + 1), description="Muscle Indices"):
        mask: Bool[Array, " c"] = (muscle_fraction > MUSCLE_FRACTION_THRESHOLD) & (
            muscle_id == i
        )
        indices: Integer[Array, " m"] = jnp.flatnonzero(mask)
        volume_i: Float[Array, " m"] = volume[indices] * muscle_fraction[indices]
        muscle_id_to_idx[i] = indices
        muscle_id_to_volume[i] = volume_i

    mesh.point_data["PointId"] = np.arange(mesh.n_points)
    surface = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    face: pv.PolyData = surface.extract_points(
        surface.point_data["IsFace"]
    ).extract_surface()  # pyright: ignore[reportAssignmentType]
    inverse = Inverse(
        face_point_idx=face_point_idx,
        model=model,
        muscle_cell_idx=muscle_idx,
        n_cells=mesh.n_cells,
        rest_pos=jnp.asarray(mesh.points[face_point_idx]),
        face=face,
        loss_fn=LossFn(
            face_point_idx=face_point_idx,
            rest_pos=jnp.asarray(mesh.points[face_point_idx]),
            muscle_id_to_idx=muscle_id_to_idx,
            muscle_id_to_volume=muscle_id_to_volume,
        ),
    )
    return inverse


def calc_inverse(
    target: pv.UnstructuredGrid, inverse: Inverse, idx: str = "000"
) -> pv.UnstructuredGrid:
    inverse.set_target(target, data_name=f"Expression{idx}")
    params: Params = Params(activation=jnp.zeros((inverse.muscle_cell_idx.shape[0], 6)))
    optimizer = ScipyOptimizer(method="L-BFGS-B", tol=1e-7, timer=True)

    with melon.SeriesWriter(
        cherries.temp(f"20-inverse-point-to-plane-{idx}.vtu.series")
    ) as writer:

        def callback(state: ScipyOptimizer.State, _stats: ScipyOptimizer.Stats) -> None:
            activation: Float[Array, "c 6"] = inverse.make_activation(state.params)
            ic(inverse.u)
            target.point_data[f"Displacement{idx}"] = np.asarray(inverse.u)
            target.point_data[f"PointToPoint{idx}"] = np.zeros((target.n_points, 3))
            target.point_data[f"PointToPoint{idx}"][
                np.asarray(inverse.face_point_idx)
            ] = np.asarray(
                inverse.u[inverse.face_point_idx] - inverse.loss_fn.target_disp
            )
            target.point_data[f"PointToPlane{idx}"] = np.zeros((target.n_points, 3))
            target.point_data[f"PointToPlane{idx}"][
                np.asarray(inverse.face_point_idx)
            ] = np.asarray(
                inverse.rest_pos
                + inverse.u[inverse.face_point_idx]
                - inverse.loss_fn.nearest_point_on_surface
            )
            target.cell_data[f"Activation{idx}"] = np.asarray(activation)
            writer.append(target)

        optimizer.tol = 1e-6
        inverse.loss_fn.weight_point_to_point = 1.0
        inverse.loss_fn.weight_point_to_plane = 0.0
        inverse.loss_fn.weight_sparse = 1e-2
        solution: ScipyOptimizer.Solution = optimizer.minimize(
            objective=Objective(value_and_grad=inverse.value_and_grad),
            params=params,
            callback=callback,
        )
        ic(solution)
        params = solution.params
        optimizer.tol = 1e-9
        inverse.loss_fn.weight_point_to_point = 0.1
        inverse.loss_fn.weight_point_to_plane = 1.0
        inverse.loss_fn.weight_sparse = 1e-4
        solution: ScipyOptimizer.Solution = optimizer.minimize(
            objective=Objective(value_and_grad=inverse.value_and_grad),
            params=params,
            callback=callback,
        )
    ic(solution)
    params: Params = solution.params
    activation: Float[Array, "c 6"] = jnp.zeros((target.n_cells, 6))
    activation = activation.at[inverse.muscle_cell_idx].set(params.activation)
    target.point_data[f"Displacement{idx}"] = np.asarray(inverse.u)
    target.cell_data[f"Activation{idx}"] = np.asarray(activation)
    return target


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    inverse: Inverse = prepare(mesh)
    mesh = calc_inverse(target, inverse, idx="000")
    mesh = calc_inverse(target, inverse, idx="001")
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
