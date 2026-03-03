import abc
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, override

import jarp
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Bool, Float
from liblaf.peach.optim import Optimizer

from liblaf import cherries, melon
from liblaf.apple import scene
from liblaf.apple.model import Forward, Model

type EnergyMaterials = Mapping[str, Array]
type ModelMaterials = Mapping[str, EnergyMaterials]
type Scalar = Float[Array, ""]
type Vector = Float[Array, "points dim"]
type BoolNumeric = Bool[Array, ""]


class Loss(abc.ABC):
    name: ClassVar[str] = "loss"

    @abc.abstractmethod
    def fun(self, u_full: Vector, materials: ModelMaterials) -> Scalar:
        raise NotImplementedError

    @jarp.jit(inline=True)
    def grad(
        self, u_full: Vector, materials: ModelMaterials
    ) -> tuple[Vector, ModelMaterials]:
        return jax.grad(self.fun, argnums=(0, 1))(u_full, materials)


class PointToPointLoss(Loss):
    name: ClassVar[str] = "point_to_point"

    face_mask: Bool[Array, " points"]
    target: Vector

    @override
    def fun(self, u_full: Vector, materials: ModelMaterials) -> Scalar:
        return jnp.mean(jnp.sum(jnp.square(u_full - self.target), axis=-1))


class Inverse:
    forward: Forward
    losses: list[Loss]

    last_adjoint_success: BoolNumeric = jarp.array(default=False)
    last_forward_success: BoolNumeric = jarp.array(default=False)

    @property
    def model(self) -> Model:
        return self.forward.model

    def fun(self, materials: ModelMaterials) -> Scalar:
        return self.loss(self.model.u_full, materials)

    def loss(self, u_full: Vector, materials: ModelMaterials) -> Scalar:
        for loss in self.losses:
            loss_value: Scalar = loss.fun(u_full, materials)
        return loss_value

    def update(self, materials: ModelMaterials) -> None:
        self.forward.update_materials(materials)
        if not self.last_forward_success:
            self.model.u_free = jnp.zeros_like(self.model.u_free)
        solution: Optimizer.Solution = self.forward.step()
        self.last_forward_success = jnp.asarray(solution.success)


class Config(cherries.BaseConfig):
    target: Path = cherries.input("20-forward-whole-act2.vtu")


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    model: Model = scene.build_phace_v3(mesh)
    forward: Forward = Forward(model)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
