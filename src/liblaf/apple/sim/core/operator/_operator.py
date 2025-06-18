import abc
from typing import Protocol, type_check_only

from liblaf.apple import struct


@type_check_only
class Object(Protocol): ...


class Operator(struct.GraphNode):
    @abc.abstractmethod
    def update[T: Object](self, obj: T, /) -> T:
        raise NotImplementedError
