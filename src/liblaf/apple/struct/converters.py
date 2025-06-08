from typing import Any

from .typed import Converter


def optional[T](converter: Converter[T]) -> Converter[T]:
    def wrapper(value: Any) -> T:
        if value is None:
            return None  # pyright: ignore[reportReturnType]
        return converter(value)

    return wrapper
