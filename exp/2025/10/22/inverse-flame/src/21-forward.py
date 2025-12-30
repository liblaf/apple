import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, no_type_check, override

import jax.numpy as jnp
import pyvista as pv
import warp as wp
from jaxtyping import Array, Float
from liblaf.peach import tree
from liblaf.peach.optim import PNCG

import liblaf.apple.warp.types as wpt
from liblaf import cherries, melon
from liblaf.apple import Forward, Model, ModelBuilder
from liblaf.apple.constants import ACTIVATION, LAMBDA, MU, MUSCLE_FRACTION, POINT_ID
from liblaf.apple.jax.fem import Region
from liblaf.apple.warp import Phace, math, utils
from liblaf.apple.warp.energies.elastic.hyperelastic import func
from liblaf.apple.warp.energies.elastic.hyperelastic._arap import Arap
from liblaf.apple.warp.energies.elastic.hyperelastic._arap_muscle import ArapMuscle
from liblaf.apple.warp.energies.elastic.hyperelastic._base import Hyperelastic

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    input: Path = cherries.temp(
        "20-inverse-adam-515k.vtu.d/20-inverse-adam-515k_000945.vtu"
    )


mat33 = Any
mat43 = Any
scalar = Any


@tree.define
class PhaceFixHess(Hyperelastic):
    @override
    @wp.struct
    class Params:
        activation: wp.array(dtype=wpt.vec6)
        lambda_: wp.array(dtype=wpt.float_)
        mu: wp.array(dtype=wpt.float_)
        muscle_fraction: wp.array(dtype=wpt.float_)

    @override
    @wp.struct
    class ParamsElem:
        activation: wpt.vec6
        lambda_: wpt.float_
        mu: wpt.float_
        muscle_fraction: wpt.float_

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def get_cell_params_func(params: Params, cid: int) -> ParamsElem:
        return PhaceFixHess.ParamsElem(
            activation=params.activation[cid],
            lambda_=params.lambda_[cid],
            mu=params.mu[cid],
            muscle_fraction=params.muscle_fraction[cid],
        )

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_func(F: mat33, params: ParamsElem) -> scalar:
        _1 = F.dtype(1.0)
        _2 = F.dtype(2.0)
        J = func.I3(F)  # float
        Psi_ARAP_active = ArapMuscle.energy_density_func(
            F, PhaceFixHess._arap_active_params(params)
        )  # float
        Psi_ARAP_passive = Arap.energy_density_func(
            F, PhaceFixHess._arap_params(params)
        )  # float
        Psi_ARAP = (
            params.muscle_fraction * Psi_ARAP_active
            + (_1 - params.muscle_fraction) * Psi_ARAP_passive
        )  # float
        Psi_VP = params.lambda_ * math.square(J - _1)  # float
        Psi = _2 * Psi_ARAP + Psi_VP  # float
        return Psi

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def first_piola_kirchhoff_stress_func(
        F: mat33, params: ParamsElem, *, clamp: bool = False
    ) -> mat33:
        _1 = F.dtype(1.0)
        _2 = F.dtype(2.0)
        J = func.I3(F)  # float
        g3 = func.g3(F)  # mat33
        PK1_ARAP_active = ArapMuscle.first_piola_kirchhoff_stress_func(
            F, PhaceFixHess._arap_active_params(params), clamp=clamp
        )  # mat33
        PK1_ARAP_passive = Arap.first_piola_kirchhoff_stress_func(
            F, PhaceFixHess._arap_params(params), clamp=clamp
        )  # mat33
        PK1_ARAP = (
            params.muscle_fraction * PK1_ARAP_active
            + (_1 - params.muscle_fraction) * PK1_ARAP_passive
        )  # mat33
        PK1_VP = _2 * params.lambda_ * (J - _1) * g3  # mat33
        PK1 = _2 * PK1_ARAP + PK1_VP  # mat33
        return PK1

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_hess_diag_func(
        F: mat33, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> mat33:
        _1 = F.dtype(1.0)
        _2 = F.dtype(2.0)
        J = func.I3(F)  # float
        g3 = func.g3(F)  # mat33
        diag_arap_passive = Arap.energy_density_hess_diag_func(
            F, dhdX, PhaceFixHess._arap_params(params), clamp=clamp
        )  # mat43
        diag_arap_passive = F.dtype(0.5) * params.mu * func.h5_diag(dhdX)  # mat43
        diag_arap = diag_arap_passive  # mat43
        d2Psi_dI32 = _2 * params.lambda_  # float
        dPsi_dI3 = _2 * params.lambda_ * (J - _1)  # float
        h3_diag = func.h3_diag(dhdX, g3)  # mat43
        h6_diag = func.h6_diag(dhdX, F)  # mat43
        diag_vp = d2Psi_dI32 * h3_diag + dPsi_dI3 * h6_diag  # mat43
        diag = _2 * diag_arap + diag_vp  # mat43
        return diag

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_hess_prod_func(
        F: mat33, p: mat43, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> mat33:
        _1 = F.dtype(1.0)
        _2 = F.dtype(2.0)
        J = func.I3(F)  # float
        g3 = func.g3(F)  # mat33
        prod_arap_passive = Arap.energy_density_hess_prod_func(
            F, p, dhdX, PhaceFixHess._arap_params(params), clamp=clamp
        )  # mat43
        prod_arap_passive = F.dtype(0.5) * params.mu * func.h5_prod(p, dhdX)  # mat43
        prod_arap = prod_arap_passive  # mat43
        d2Psi_dI32 = _2 * params.lambda_  # float
        dPsi_dI3 = _2 * params.lambda_ * (J - _1)  # float
        h3_prod = func.h3_prod(p, dhdX, g3)  # mat43
        h6_prod = func.h6_prod(p, dhdX, F)  # mat43
        prod_vp = d2Psi_dI32 * h3_prod + dPsi_dI3 * h6_prod  # mat43
        prod = _2 * prod_arap + prod_vp  # mat43
        return prod

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_hess_quad_func(
        F: mat33, p: mat43, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> scalar:
        _1 = F.dtype(1.0)
        _2 = F.dtype(2.0)
        J = func.I3(F)  # float
        g3 = func.g3(F)  # mat33
        quad_arap_passive = Arap.energy_density_hess_quad_func(
            F, p, dhdX, PhaceFixHess._arap_params(params), clamp=clamp
        )  # float
        quad_arap_passive = F.dtype(0.5) * params.mu * func.h5_quad(p, dhdX)  # float
        quad_arap = quad_arap_passive  # float
        d2Psi_dI32 = _2 * params.lambda_  # float
        dPsi_dI3 = _2 * params.lambda_ * (J - _1)  # float
        h3_quad = func.h3_quad(p, dhdX, g3)  # float
        h6_quad = func.h6_quad(p, dhdX, F)  # float
        quad_vp = d2Psi_dI32 * h3_quad + dPsi_dI3 * h6_quad  # float
        quad = _2 * quad_arap + quad_vp  # float
        return quad

    @staticmethod
    @no_type_check
    @wp.func
    def _arap_active_params(params: ParamsElem) -> ArapMuscle.ParamsElem:
        return ArapMuscle.ParamsElem(activation=params.activation, mu=params.mu)

    @staticmethod
    @no_type_check
    @wp.func
    def _arap_params(params: ParamsElem) -> Arap.ParamsElem:
        return Arap.ParamsElem(mu=params.mu)

    @override
    @classmethod
    def _params_fields_from_region(cls, region: Region) -> Mapping[str, wp.array]:
        fields: dict[str, wp.array] = {}
        fields["activation"] = utils.to_warp(region.cell_data[ACTIVATION], wpt.vec6)
        fields["mu"] = utils.to_warp(region.cell_data[MU], wpt.float_)
        fields["lambda_"] = utils.to_warp(region.cell_data[LAMBDA], wpt.float_)
        fields["muscle_fraction"] = utils.to_warp(
            region.cell_data[MUSCLE_FRACTION], wpt.float_
        )
        return fields


def build_model(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    # mesh.point_data["lambda"] = 0.0
    builder.add_dirichlet(mesh)
    builder.add_energy(Phace.from_pyvista(mesh, id="elastic", clamp_lambda=False))
    model: Model = builder.finalize()
    return model


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    model: Model = build_model(mesh)
    forward = Forward(
        model=model,
        optimizer=PNCG(
            max_steps=1000,
            timer=True,
            rtol=1e-5,
            max_delta=0.15 * model.edges_length_mean,
            # line_search=PNCG.default_line_search(d_hat=1, line_search_steps=0),
        ),
    )

    def callback(state: PNCG.State, stats: PNCG.Stats) -> None:
        cherries.log_metrics(
            {
                "fun": model.fun(state.params_flat),
                "relative_decrease": stats.relative_decrease,
                "grad_norm": jnp.linalg.norm(state.grad_flat),
                "grad_max_norm": jnp.linalg.norm(state.grad_flat, ord=jnp.inf),
                "alpha": state.alpha,
                "beta": state.beta,
                "delta_x_norm": jnp.linalg.norm(
                    state.alpha * state.search_direction_flat
                ),
                "delta_x_max_norm": jnp.linalg.norm(
                    state.alpha * state.search_direction_flat, ord=jnp.inf
                ),
                # "path_efficiency": state.path_efficiency,
                # "total_path_length": state.total_path_length,
                # "net_displacement": state.net_displacement,
                # "stagnation_count": state.stagnation_count,
            },
            step=stats.n_steps,
        )

    solution: PNCG.Solution = forward.step(callback=callback)
    mesh.point_data["Displacement"] = solution.params[mesh.point_data[POINT_ID]]
    melon.save(cherries.temp("21-forward.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main)
