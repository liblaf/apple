from collections.abc import Iterator, Mapping
from typing import Annotated

import jarp
import warp as wp
from frozendict import frozendict

type Vector = Annotated[wp.array, " points"]


@jarp.frozen_static
class WarpEnergyState:
    pass


@jarp.frozen_static
class WarpModelState(Mapping[str, WarpEnergyState]):
    data: frozendict[str, WarpEnergyState] = jarp.field(factory=lambda: frozendict())

    def __getitem__(self, key: str) -> WarpEnergyState:
        return self.data[key]

    # def __setitem__(self, key: str, value: WarpEnergyState) -> None:
    #     self.data[key] = value

    # def __delitem__(self, key: str) -> None:
    #     del self.data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)
