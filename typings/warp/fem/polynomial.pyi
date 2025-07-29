import numpy as np
from enum import Enum

class Polynomial(Enum):
    GAUSS_LEGENDRE = 'GL'
    LOBATTO_GAUSS_LEGENDRE = 'LGL'
    EQUISPACED_CLOSED = 'closed'
    EQUISPACED_OPEN = 'open'

def is_closed(family: Polynomial): ...
def quadrature_1d(point_count: int, family: Polynomial): ...
def lagrange_scales(coords: np.array): ...
