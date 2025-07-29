from jaxtyping import Array, Float

from liblaf.apple import struct


class State(struct.ArrayDict):
    @property
    def displacement(self) -> Float[Array, " DOF"]:
        return self["displacement"]

    @property
    def velocity(self) -> Float[Array, " DOF"]:
        return self["velocity"]

    @property
    def force_ext(self) -> Float[Array, " DOF"]:
        return self["force-ext"]

    @property
    def mass(self) -> Float[Array, " DOF"]:
        return self["mass"]

    @property
    def x_prev(self) -> Float[Array, " DOF"]:
        return self["x_prev"]
