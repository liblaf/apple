import enum


class AttrName(enum.StrEnum):
    vtk: str
    ACTIVATION = enum.auto()
    DISPLACEMENT = enum.auto()
    E = enum.auto()
    FIXED_MASK = enum.auto()
    FIXED_VALUE = enum.auto()
    FORCE = enum.auto()
    FRACTION = enum.auto()
    GLOBAL_POINT_ID = enum.auto()
    LAMBDA = "lambda_"
    MASS_DENSITY = enum.auto()
    MU = enum.auto()
    NU = enum.auto()
    PRESTRAIN = enum.auto()


ACTIVATION = AttrName.ACTIVATION
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
