from collections.abc import Iterator, MutableMapping

import jarp
from jaxtyping import Array, Float

type Vector = Float[Array, "points dim"]


@jarp.define
class JaxEnergyState:
    pass


@jarp.define
class JaxModelState(MutableMapping[str, JaxEnergyState]):
    u: Vector
    data: dict[str, JaxEnergyState] = jarp.field(factory=dict)

    def __getitem__(self, key: str) -> JaxEnergyState:
        return self.data[key]

    def __setitem__(self, key: str, value: JaxEnergyState) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
