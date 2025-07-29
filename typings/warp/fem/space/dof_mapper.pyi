import warp as wp
from _typeshed import Incomplete
from enum import Enum
from typing import Any

vec6: Incomplete

class DofMapper:
    value_dtype: type
    dof_dtype: type
    DOF_SIZE: int
    @wp.func
    def dof_to_value(dof: Any): ...
    @wp.func
    def value_to_dof(val: Any): ...

class IdentityMapper(DofMapper):
    value_dtype: Incomplete
    dof_dtype: Incomplete
    DOF_SIZE: Incomplete
    def __init__(self, dtype: type) -> None: ...
    @wp.func
    def dof_to_value(dof: Any): ...
    @wp.func
    def value_to_dof(val: Any): ...

class SymmetricTensorMapper(DofMapper):
    class Mapping(Enum):
        VOIGT = 0
        DB16 = 1
    value_dtype: Incomplete
    mapping: Incomplete
    dof_dtype: Incomplete
    DOF_SIZE: Incomplete
    dof_to_value: Incomplete
    value_to_dof: Incomplete
    def __init__(self, dtype: type, mapping: Mapping = ...) -> None: ...
    @wp.func
    def dof_to_value_2d(dof: wp.vec3): ...
    @wp.func
    def value_to_dof_2d(val: wp.mat22): ...
    @wp.func
    def dof_to_value_2d_voigt(dof: wp.vec3): ...
    @wp.func
    def value_to_dof_2d_voigt(val: wp.mat22): ...
    @wp.func
    def dof_to_value_3d(dof: vec6): ...
    @wp.func
    def value_to_dof_3d(val: wp.mat33): ...
    @wp.func
    def dof_to_value_3d_voigt(dof: vec6): ...
    @wp.func
    def value_to_dof_3d_voigt(val: wp.mat33): ...

class SkewSymmetricTensorMapper(DofMapper):
    value_dtype: Incomplete
    dof_dtype: Incomplete
    DOF_SIZE: Incomplete
    dof_to_value: Incomplete
    value_to_dof: Incomplete
    def __init__(self, dtype: type) -> None: ...
    @wp.func
    def dof_to_value_2d(dof: float): ...
    @wp.func
    def value_to_dof_2d(val: wp.mat22): ...
    @wp.func
    def dof_to_value_3d(dof: wp.vec3): ...
    @wp.func
    def value_to_dof_3d(val: wp.mat33): ...
