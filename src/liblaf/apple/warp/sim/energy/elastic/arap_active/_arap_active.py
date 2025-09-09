from typing import override

import warp as wp

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.region._region import Region
from liblaf.apple.warp.sim.energy.elastic._elastic import Elastic
from liblaf.apple.warp.typing import float_, vec6

from . import func


@tree.pytree
class ArapActive(Elastic):
    param_dtype: type = tree.field(default=func.Params)
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

    @override
    def make_params(self, region: Region) -> wp.array:
        params: wp.array = wp.empty((region.n_cells,), dtype=self.param_dtype)
        wp.launch(
            make_params_kernel,
            (region.n_cells,),
            inputs=[
                wp.from_jax(region.cell_data["activation"], vec6),
                wp.from_jax(region.cell_data["mu"], float_),
            ],
            outputs=[params],
        )
        return params


@wp.kernel
def make_params_kernel(
    activation: wp.array(dtype=vec6),
    mu: wp.array(dtype=float_),
    output: wp.array(dtype=func.Params),
) -> None:
    cid = wp.tid()
    output[cid].activation = activation[cid]
    output[cid].mu = mu[cid]
