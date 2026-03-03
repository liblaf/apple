import abc

import jarp
import jax

from liblaf.apple.model import Full, ModelMaterials, Scalar


@jarp.define
class Loss(abc.ABC):
    name: str = jarp.static(default="loss", kw_only=True)

    @abc.abstractmethod
    def fun(self, u_full: Full, materials: ModelMaterials) -> Scalar:
        raise NotImplementedError

    @jarp.jit(filter=True, inline=True)
    def grad(
        self, u_full: Full, materials: ModelMaterials
    ) -> tuple[Full, ModelMaterials]:
        return jax.grad(self.fun, argnums=(0, 1))(u_full, materials)

    @jarp.jit(filter=True, inline=True)
    def value_and_grad(
        self, u_full: Full, materials: ModelMaterials
    ) -> tuple[Scalar, tuple[Full, ModelMaterials]]:
        return jax.value_and_grad(self.fun, argnums=(0, 1))(u_full, materials)
