from collections.abc import Mapping
from typing import Any, no_type_check, override

import warp as wp
from liblaf.peach import tree

import liblaf.apple.warp.types as wpt
from liblaf.apple.constants import ACTIVATION, LAMBDA, MU, MUSCLE_FRACTION
from liblaf.apple.jax.fem import Region
from liblaf.apple.warp import math, utils

from . import func
from ._arap import Arap
from ._arap_muscle import ArapMuscle
from ._base import Hyperelastic

mat33 = Any
mat43 = Any
scalar = Any


@tree.define
class Phace(Hyperelastic):
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
        return Phace.ParamsElem(
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
            F, Phace._arap_active_params(params)
        )  # float
        Psi_ARAP_passive = Arap.energy_density_func(
            F, Phace._arap_params(params)
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
            F, Phace._arap_active_params(params), clamp=clamp
        )  # mat33
        PK1_ARAP_passive = Arap.first_piola_kirchhoff_stress_func(
            F, Phace._arap_params(params), clamp=clamp
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
        diag_arap_active = ArapMuscle.energy_density_hess_diag_func(
            F, dhdX, Phace._arap_active_params(params), clamp=clamp
        )  # mat43
        diag_arap_passive = Arap.energy_density_hess_diag_func(
            F, dhdX, Phace._arap_params(params), clamp=clamp
        )  # mat43
        diag_arap = (
            params.muscle_fraction * diag_arap_active
            + (_1 - params.muscle_fraction) * diag_arap_passive
        )  # mat43
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
        prod_arap_active = ArapMuscle.energy_density_hess_prod_func(
            F, p, dhdX, Phace._arap_active_params(params), clamp=clamp
        )  # mat43
        prod_arap_passive = Arap.energy_density_hess_prod_func(
            F, p, dhdX, Phace._arap_params(params), clamp=clamp
        )  # mat43
        prod_arap = (
            params.muscle_fraction * prod_arap_active
            + (_1 - params.muscle_fraction) * prod_arap_passive
        )  # mat43
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
        quad_arap_active = ArapMuscle.energy_density_hess_quad_func(
            F, p, dhdX, Phace._arap_active_params(params), clamp=clamp
        )
        quad_arap_passive = Arap.energy_density_hess_quad_func(
            F, p, dhdX, Phace._arap_params(params), clamp=clamp
        )
        quad_arap = (
            params.muscle_fraction * quad_arap_active
            + (_1 - params.muscle_fraction) * quad_arap_passive
        )
        d2Psi_dI32 = _2 * params.lambda_  # float
        dPsi_dI3 = _2 * params.lambda_ * (J - _1)  # float
        h3_quad = func.h3_quad(p, dhdX, g3)  # float
        h6_quad = func.h6_quad(p, dhdX, F)  # float
        quad_vp = d2Psi_dI32 * h3_quad + dPsi_dI3 * h6_quad  # float
        quad = _2 * quad_arap + quad_vp  # float
        return quad

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_hess_block_diag_func(
        F: mat33, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> tuple[mat33, mat33, mat33, mat33]:
        _1 = F.dtype(1.0)
        _2 = F.dtype(2.0)
        J = func.I3(F)  # float
        g3 = func.g3(F)  # mat33
        bd_arap_active = ArapMuscle.energy_density_hess_block_diag_func(
            F, dhdX, Phace._arap_active_params(params), clamp=clamp
        )
        bd_arap_passive = Arap.energy_density_hess_block_diag_func(
            F, dhdX, Phace._arap_params(params), clamp=clamp
        )
        d2Psi_dI32 = _2 * params.lambda_  # float
        dPsi_dI3 = _2 * params.lambda_ * (J - _1)  # float
        h3_bd = func.h3_block_diag(dhdX, g3)
        h6_bd = func.h6_block_diag(dhdX, F)
        mf = params.muscle_fraction
        imf = _1 - mf
        return (
            mf * bd_arap_active[0]
            + imf * bd_arap_passive[0]
            + d2Psi_dI32 * h3_bd[0]
            + dPsi_dI3 * h6_bd[0],
            mf * bd_arap_active[1]
            + imf * bd_arap_passive[1]
            + d2Psi_dI32 * h3_bd[1]
            + dPsi_dI3 * h6_bd[1],
            mf * bd_arap_active[2]
            + imf * bd_arap_passive[2]
            + d2Psi_dI32 * h3_bd[2]
            + dPsi_dI3 * h6_bd[2],
            mf * bd_arap_active[3]
            + imf * bd_arap_passive[3]
            + d2Psi_dI32 * h3_bd[3]
            + dPsi_dI3 * h6_bd[3],
        )

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
