from ._deformation import (
    deformation_gradient,
    deformation_gradient_jvp,
    deformation_gradient_vjp,
    gradient,
)
from ._gradients import g1, g2, g3
from ._hess_diag import h1_diag, h2_diag, h3_diag, h4_diag, h5_diag, h6_diag
from ._hess_prod import h1_prod, h2_prod, h3_prod, h4_prod, h5_prod, h6_prod
from ._hess_quad import h1_quad, h2_quad, h3_quad, h4_quad, h5_quad, h6_quad
from ._identities import I1, I2, I3
from ._misc import Qs, dRdF_vjp, get_cell_displacements, lambdas, make_activation_mat33

__all__ = [
    "I1",
    "I2",
    "I3",
    "Qs",
    "dRdF_vjp",
    "deformation_gradient",
    "deformation_gradient_jvp",
    "deformation_gradient_vjp",
    "g1",
    "g2",
    "g3",
    "get_cell_displacements",
    "gradient",
    "h1_diag",
    "h1_prod",
    "h1_quad",
    "h2_diag",
    "h2_prod",
    "h2_quad",
    "h3_diag",
    "h3_prod",
    "h3_quad",
    "h4_diag",
    "h4_prod",
    "h4_quad",
    "h5_diag",
    "h5_prod",
    "h5_quad",
    "h6_diag",
    "h6_prod",
    "h6_quad",
    "lambdas",
    "make_activation_mat33",
]
