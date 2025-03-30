from ._inverse import InversePhysicsProblem
from ._linear_operator import LinearOperator, as_linear_operator
from ._physics import AbstractObject, PhysicsProblem

__all__ = [
    "AbstractObject",
    "InversePhysicsProblem",
    "LinearOperator",
    "PhysicsProblem",
    "as_linear_operator",
]
