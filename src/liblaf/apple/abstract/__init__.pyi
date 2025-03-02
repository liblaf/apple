from ._inverse import InversePhysicsProblem
from ._linear_operator import LinearOperator, as_linear_operator
from ._physics import AbstractPhysicsProblem

__all__ = [
    "AbstractPhysicsProblem",
    "InversePhysicsProblem",
    "LinearOperator",
    "as_linear_operator",
]
