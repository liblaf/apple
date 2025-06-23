from collections.abc import Callable
from typing import NoReturn

from typing_extensions import TypeIs

from liblaf.apple import utils


@utils.not_implemented
def not_implemented(*args, **kwargs) -> NoReturn:
    raise NotImplementedError


def implemented(func: Callable | None, /) -> TypeIs[Callable]:
    return callable(func) and utils.is_implemented(func)
