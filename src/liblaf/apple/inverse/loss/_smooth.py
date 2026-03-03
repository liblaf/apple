from collections.abc import Mapping
from typing import Self, override

import jarp
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Float, Integer

from liblaf.apple.consts import MUSCLE_FRACTION
from liblaf.apple.model import Full, ModelMaterials

from ._base import Loss

type Scalar = Float[Array, ""]


@jarp.define
class SmoothActivationLoss(Loss):
    name: str = jarp.static(default="smooth", kw_only=True)
    muscle_id_to_cell_neighbors: Mapping[int, Integer[Array, "2 N"]]

    @classmethod
    def from_pyvista(cls, mesh: pv.UnstructuredGrid) -> Self:
        unique_muscle_id: Integer[Array, " muscles"] = jnp.unique(
            mesh.cell_data["MuscleId"][mesh.cell_data[MUSCLE_FRACTION] > 1e-3]
        )
        muscle_id_to_cell_neighbors: dict[int, Integer[Array, "N 2"]] = {}
        for muscle_id in unique_muscle_id.tolist():
            if muscle_id < 0:
                continue
            muscle_id_to_cell_neighbors[muscle_id] = jnp.asarray(
                mesh.field_data[f"Muscle{muscle_id}CellNeighbors"]
            )
        return cls(muscle_id_to_cell_neighbors=muscle_id_to_cell_neighbors)

    @override
    @jarp.jit(inline=True)
    def fun(self, u_full: Full, materials: ModelMaterials) -> Scalar:
        activation: Float[Array, "cells 6"] = materials["elastic"]["activation"]
        losses: list[Scalar] = []
        normalizations: list[Scalar] = []
        for cell_neighbors in self.muscle_id_to_cell_neighbors.values():
            diff: Float[Array, "N 6"] = (
                activation[cell_neighbors[:, 0]] - activation[cell_neighbors[:, 1]]
            )
            loss: Scalar
            normalization: Scalar
            loss, normalization = jnp.average(
                jnp.sum(jnp.square(diff), axis=-1), returned=True
            )
            losses.append(loss)
            normalizations.append(normalization)
        return jnp.average(jnp.stack(losses), weights=jnp.stack(normalizations))
