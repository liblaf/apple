from typing import override

import warp as wp
from liblaf.peach import tree

import liblaf.apple.warp.utils as wp_utils
from liblaf.apple.jax.sim.region._region import Region
from liblaf.apple.warp.sim.energy.elastic._elastic import Elastic
from liblaf.apple.warp.typing import Struct, float_, vec6

from . import func


@tree.define
class ArapActive(Elastic):
    energy_density_func: wp.Function = func.energy_density  # pyright: ignore[reportAssignmentType]
    first_piola_kirchhoff_stress_func: wp.Function = (
        func.first_piola_kirchhoff_stress_tensor
    )  # pyright: ignore[reportAssignmentType]
    energy_density_hess_diag_func: wp.Function = func.energy_density_hess_diag  # pyright: ignore[reportAssignmentType]
    energy_density_hess_prod_func: wp.Function = func.energy_density_hess_prod  # pyright: ignore[reportAssignmentType]
    energy_density_hess_quad_func: wp.Function = func.energy_density_hess_quad  # pyright: ignore[reportAssignmentType]
    get_cell_params: wp.Function = func.get_cell_params  # pyright: ignore[reportAssignmentType]

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
