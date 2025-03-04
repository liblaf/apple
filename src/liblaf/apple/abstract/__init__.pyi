from ._inverse import InversePhysicsProblem
from ._linear_operator import LinearOperator, as_linear_operator
from ._physics import AbstractPhysicsProblem, AbstractPhysicsProblemBuilder

__all__ = [
    "AbstractPhysicsProblem",
    "AbstractPhysicsProblemBuilder",
    "InversePhysicsProblem",
    "LinearOperator",
    "as_linear_operator",
]
