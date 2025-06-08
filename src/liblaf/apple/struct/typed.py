from collections.abc import Callable
from typing import Any

type Converter[T] = Callable[[Any], T]
type Validator[T] = Callable[[T], None]
