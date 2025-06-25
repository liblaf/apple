from liblaf.apple import struct

from .integrator import TimeIntegrator


@struct.pytree
class TimeIntegratorStatic(TimeIntegrator):
    pass
