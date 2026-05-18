from typing import Protocol


class Energy[T](Protocol):
    def fun(self, state: T) -> float: ...
