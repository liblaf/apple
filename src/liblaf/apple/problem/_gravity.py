import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True, on_setattr=attrs.setters.convert)
class Gravity(apple.AbstractPhysicsProblem):
    name: str = attrs.field(default="gravity", metadata={"static": True})
    gravity: Float[jax.Array, "3"] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray([0.0, -9.8, 0.0])
    )
    mass: Float[jax.Array, "..."] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(1e3)
    )
    n_points: int = attrs.field(
        metadata={"static": True}, on_setattr=attrs.setters.frozen
    )

    @property
    def n_dof(self) -> int:
        return self.n_points * 3

    def fun(self, u: PyTree, q: PyTree | None = None) -> Float[jax.Array, ""]:
        u: Float[jax.Array, "P 3"]
        mass: Float[jax.Array, " ..."] = self.get_param("mass", q)
        gravity: Float[jax.Array, " ..."] = self.get_param("gravity", q)
        return -jnp.sum(mass * jnp.vecdot(gravity, u))

    def unravel_u(self, u_flat: Float[jax.Array, " DoF"]) -> PyTree:
        return u_flat.reshape(self.n_points, 3)


@attrs.define(kw_only=True, on_setattr=attrs.setters.convert)
class GravityBuilder(apple.AbstractPhysicsProblemBuilder):
    name: str = attrs.field(default="gravity", metadata={"static": True})
    gravity: Float[jax.Array, "..."] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray([0.0, -9.8, 0.0])
    )
    mass: Float[jax.Array, "..."] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(1e3)
    )
    n_points: int = attrs.field(
        metadata={"static": True}, on_setattr=attrs.setters.frozen
    )

    def build(self, q: PyTree | None = None) -> Gravity:
        super().build(q)
        return Gravity(
            name=self.name,
            gravity=self.get_param("gravity", q),
            mass=self.get_param("mass", q),
            n_points=self.n_points,
            _q_unravel=self._q_unravel,
        )
