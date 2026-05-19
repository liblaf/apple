import enum
from typing import Self


class AttrName(enum.StrEnum):
    vtk: str
    ACTIVATION = enum.auto()
    ACTIVATION_INV = enum.auto()
    DISPLACEMENT = enum.auto()
    E = enum.auto()
    FIXED_MASK = enum.auto()
    FIXED_VALUE = enum.auto()
    FORCE = enum.auto()
    FRACTION = enum.auto()
    GLOBAL_POINT_ID = enum.auto()
    LAMBDA = "lmbda"
    MASS_DENSITY = enum.auto()
    MU = enum.auto()
    NU = enum.auto()
    PRESTRAIN = enum.auto()

    @classmethod
    def to_vtk(cls, name: str) -> str:
        try:
            name: Self = cls(name)
        except ValueError:
            return name
        else:
            return name.vtk


ACTIVATION = AttrName.ACTIVATION
ACTIVATION_INV = AttrName.ACTIVATION_INV
DISPLACEMENT = AttrName.DISPLACEMENT
E = AttrName.E
FIXED_MASK = AttrName.FIXED_MASK
FIXED_VALUE = AttrName.FIXED_VALUE
FORCE = AttrName.FORCE
FRACTION = AttrName.FRACTION
GLOBAL_POINT_ID = AttrName.GLOBAL_POINT_ID
LAMBDA = AttrName.LAMBDA
MASS_DENSITY = AttrName.MASS_DENSITY
MU = AttrName.MU
NU = AttrName.NU
PRESTRAIN = AttrName.PRESTRAIN


def snake_to_pascal(s: str) -> str:
    return s.title().replace("_", "")


for name in AttrName:
    name.vtk = snake_to_pascal(name.value)
LAMBDA.vtk = "lambda"
MU.vtk = "mu"
NU.vtk = "nu"
