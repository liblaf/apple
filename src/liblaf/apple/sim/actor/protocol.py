from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from .actor import Actor


@runtime_checkable
class ComponentProtocol(Protocol):
    def register[T: Actor](self, actor: T) -> tuple[Self, T]: ...
    def pre_time_step[T: Actor](self, actor: T) -> tuple[Self, T]: ...
    def pre_optim_iter[T: Actor](self, actor: T) -> tuple[Self, T]: ...
