from collections.abc import Mapping

from jaxtyping import Array, Float

type EnergyMaterials = Mapping[str, Array]
type Free = Float[Array, " free"]
type Full = Float[Array, "points dim"]
type ModelMaterials = Mapping[str, EnergyMaterials]
type Scalar = Float[Array, ""]
