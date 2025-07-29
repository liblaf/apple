import warp as wp
from _typeshed import Incomplete
from typing import Any

__all__ = ['norm_huber', 'norm_l1', 'norm_l2', 'norm_pseudo_huber', 'smooth_normalize', 'transform_compose', 'transform_decompose', 'transform_from_matrix', 'transform_to_matrix']

@wp.func
def norm_l1(v: Any): ...
@wp.func
def norm_l2(v: Any): ...
@wp.func
def norm_huber(v: Any, delta: float = 1.0): ...
@wp.func
def norm_pseudo_huber(v: Any, delta: float = 1.0): ...
@wp.func
def smooth_normalize(v: Any, delta: float = 1.0): ...

transform_from_matrix: Incomplete
transform_to_matrix: Incomplete
transform_compose: Incomplete
transform_decompose: Incomplete
