from typing import no_type_check

import warp as wp

from liblaf.apple.func import utils
from liblaf.apple.typed.warp import mat33, vec3


@no_type_check
@wp.func
def lambdas(sigma: vec3) -> vec3:
    lambda0 = 2.0 / (sigma[0] + sigma[1])
    lambda1 = 2.0 / (sigma[1] + sigma[2])
    lambda2 = 2.0 / (sigma[2] + sigma[0])
    lambda0 = wp.clamp(lambda0, 0.0, 1.0)
    lambda1 = wp.clamp(lambda1, 0.0, 1.0)
    lambda2 = wp.clamp(lambda2, 0.0, 1.0)
    return vec3(lambda0, lambda1, lambda2)


@no_type_check
@wp.func
def Qs(U: mat33, V: mat33):  # noqa: ANN202
    U0 = utils.col(U, 0)
    U1 = utils.col(U, 1)
    U2 = utils.col(U, 2)
    V0 = utils.col(V, 0)
    V1 = utils.col(V, 1)
    V2 = utils.col(V, 2)
    Q0 = (wp.outer(U1, V0) - wp.outer(U0, V1)) / wp.sqrt(2.0)
    Q1 = (wp.outer(U1, V2) - wp.outer(U2, V1)) / wp.sqrt(2.0)
    Q2 = (wp.outer(U0, V2) - wp.outer(U2, V0)) / wp.sqrt(2.0)
    return Q0, Q1, Q2
