import abc
from typing import TYPE_CHECKING

from liblaf.apple import struct

if TYPE_CHECKING:
    from liblaf.apple.sim.abc.obj import Object


class Operator[T: Object](struct.GraphNode):
    @abc.abstractmethod
    def update(self, result: T, /) -> T:
        raise NotImplementedError
