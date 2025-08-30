from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.energy import Energy


@tree.pytree
class Model:
    energies: list[Energy] = tree.field(default_factory=list)
