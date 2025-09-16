from typing import override

import warp as wp

import liblaf.apple.warp.utils as wp_utils
from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.region._region import Region
from liblaf.apple.warp.sim.energy.elastic._elastic import Elastic
from liblaf.apple.warp.typing import Struct, float_, vec6

from . import func


@tree.pytree
class ArapActive(Elastic):
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
        params = func.Params()
        params.activation = wp_utils.to_warp(
            region.cell_data["activation"],
            dtype=vec6,
            requires_grad="activation" in self.requires_grad,
        )
        params.mu = wp_utils.to_warp(
            region.cell_data["mu"],
            dtype=float_,
            requires_grad="mu" in self.requires_grad,
        )
        return params
