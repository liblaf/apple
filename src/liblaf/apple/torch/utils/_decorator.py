import functools
from collections.abc import Callable
from typing import Any, Protocol, cast

import torch


class HasDevice(Protocol):
    @property
    def device(self) -> torch.device: ...


def method_with_device[F: Callable[..., Any]](func: F) -> F:
    @functools.wraps(func)
    def wrapper(self: HasDevice, *args, **kwargs) -> Any:
        with self.device:
            return func(self, *args, **kwargs)

    return cast("F", wrapper)
