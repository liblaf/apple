from collections.abc import Mapping
from typing import Any, no_type_check, override

import warp as wp
from liblaf.peach import tree

import liblaf.apple.warp.types as wpt
from liblaf.apple.constants import ACTIVATION, MU, MUSCLE_FRACTION
from liblaf.apple.jax.fem import Region
from liblaf.apple.warp import utils

from ._arap import Arap
from ._arap_muscle import ArapMuscle
from ._base import Hyperelastic

mat33 = Any
mat43 = Any
scalar = Any


@tree.define
class ArapActive(Hyperelastic):
    @override
    @wp.struct
    class Params:
        activation: wp.array(dtype=wpt.vec6)
        mu: wp.array(dtype=wpt.float_)
        muscle_fraction: wp.array(dtype=wpt.float_)

    @override
    @wp.struct
    class ParamsElem:
        activation: wpt.vec6
        mu: wpt.float_
        muscle_fraction: wpt.float_

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def get_cell_params_func(params: Params, cid: int) -> ParamsElem:
        return ArapActive.ParamsElem(
            activation=params.activation[cid],
            mu=params.mu[cid],
            muscle_fraction=params.muscle_fraction[cid],
        )

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_func(F: mat33, params: ParamsElem) -> scalar:
        Psi_active = ArapMuscle.energy_density_func(
            F, ArapActive._arap_active_params(params)
        )  # float
        Psi_passive = Arap.energy_density_func(
            F, ArapActive._arap_params(params)
        )  # float
        Psi = (
            params.muscle_fraction * Psi_active
            + (F.dtype(1.0) - params.muscle_fraction) * Psi_passive
        )  # float
        return Psi

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def first_piola_kirchhoff_stress_func(
        F: mat33, params: ParamsElem, *, clamp: bool = False
    ) -> mat33:
        PK1_active = ArapMuscle.first_piola_kirchhoff_stress_func(
            F, ArapActive._arap_active_params(params), clamp=clamp
        )  # mat33
        PK1_passive = Arap.first_piola_kirchhoff_stress_func(
            F, ArapActive._arap_params(params), clamp=clamp
        )  # mat33
        PK1 = (
            params.muscle_fraction * PK1_active
            + (F.dtype(1.0) - params.muscle_fraction) * PK1_passive
        )  # mat33
        return PK1

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_hess_diag_func(
        F: mat33, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> mat33:
        diag_active = ArapMuscle.energy_density_hess_diag_func(
            F, dhdX, ArapActive._arap_active_params(params), clamp=clamp
        )  # mat43
        diag_passive = Arap.energy_density_hess_diag_func(
            F, dhdX, ArapActive._arap_params(params), clamp=clamp
        )  # mat43
        diag = (
            params.muscle_fraction * diag_active
            + (F.dtype(1.0) - params.muscle_fraction) * diag_passive
        )  # mat43
        return diag

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_hess_prod_func(
        F: mat33, p: mat43, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> mat33:
        prod_active = ArapMuscle.energy_density_hess_prod_func(
            F, p, dhdX, ArapActive._arap_active_params(params), clamp=clamp
        )  # mat43
        prod_passive = Arap.energy_density_hess_prod_func(
            F, p, dhdX, ArapActive._arap_params(params), clamp=clamp
        )  # mat43
        prod = (
            params.muscle_fraction * prod_active
            + (F.dtype(1.0) - params.muscle_fraction) * prod_passive
        )  # mat43
        return prod

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_hess_quad_func(
        F: mat33, p: mat43, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> scalar:
        quad_active = ArapMuscle.energy_density_hess_quad_func(
            F, p, dhdX, ArapActive._arap_active_params(params), clamp=clamp
        )
        quad_passive = Arap.energy_density_hess_quad_func(
            F, p, dhdX, ArapActive._arap_params(params), clamp=clamp
        )
        quad = (
            params.muscle_fraction * quad_active
            + (F.dtype(1.0) - params.muscle_fraction) * quad_passive
        )
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
        fields["muscle_fraction"] = utils.to_warp(
            region.cell_data[MUSCLE_FRACTION], wpt.float_
        )
        return fields
