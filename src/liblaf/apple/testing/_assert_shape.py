from collections.abc import Container, Sequence
from typing import Any


def assert_shape(obj: Any, shape: Sequence[int | Container[int]]) -> None:
    assert hasattr(obj, "shape")
    assert len(obj.shape) == len(shape)
    for i, expected in zip(obj.shape, shape, strict=True):
        if isinstance(expected, Container):
            assert i in expected
        elif expected < 0:
            continue
        else:
            assert i == expected
