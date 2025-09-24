from typing import override

import warp as wp

import liblaf.apple.warp.utils as wp_utils
from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.region._region import Region
from liblaf.apple.warp.sim.energy.elastic._elastic import Elastic
from liblaf.apple.warp.typing import Struct, float_

from . import func


@tree.pytree
class Arap(Elastic):
    r"""As-Rigid-As-Possible.

    $$
    \Psi = \frac{\mu}{2} \|F - R\|_F^2 = \frac{\mu}{2} (I_2 - 2 I_1 + 3)
    $$
    """

    energy_density_func: wp.Function = tree.field(default=func.energy_density)
    first_piola_kirchhoff_stress_func: wp.Function = tree.field(
        default=func.first_piola_kirchhoff_stress_tensor
    )
    energy_density_hess_diag_func: wp.Function = tree.field(
        default=func.energy_density_hess_diag
    )
    energy_density_hess_prod_func: wp.Function = tree.field(
        default=func.energy_density_hess_prod
    )
    energy_density_hess_quad_func: wp.Function = tree.field(
        default=func.energy_density_hess_quad
    )
    get_cell_params: wp.Function = tree.field(default=func.get_cell_params)

    @override
    def make_params(self, region: Region) -> Struct:
        params: Struct = func.Params()
        params.mu = wp_utils.to_warp(
            region.cell_data["mu"],
            dtype=float_,
            requires_grad="mu" in self.requires_grad,
        )
        return params
