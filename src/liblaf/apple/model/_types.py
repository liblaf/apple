from collections.abc import Mapping

import attrs
from jaxtyping import Array, Float

type EnergyMaterials = Mapping[str, Array]
type Free = Float[Array, " free"]
type Full = Float[Array, "points dim"]
type ModelMaterials = Mapping[str, EnergyMaterials]
type Scalar = Float[Array, ""]


@attrs.frozen
class MaterialReference:
    energy_name: str
    material_name: str


type MaterialValues = Mapping[MaterialReference, Array]
