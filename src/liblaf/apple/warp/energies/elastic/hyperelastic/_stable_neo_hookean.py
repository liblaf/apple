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
class StableNeoHookean(Hyperelastic):
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
        return StableNeoHookean.ParamsElem(
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
        # _1 = F.dtype(1.0)
        # _2 = F.dtype(2.0)
        A = func.make_activation_mat33(params.activation)  # mat33
        G = F @ A
        I2 = func.I2(G)  # float
        I3 = func.I3(G)  # float
        Psi_active = (
            F.dtype(0.5) * params.mu * (I2 - F.dtype(3.0))
            - params.mu * (I3 - F.dtype(1.0))
            + F.dtype(0.5) * params.lambda_ * math.square(I3 - F.dtype(1.0))
        )
        I2 = func.I2(F)  # float
        I3 = func.I3(F)  # float
        Psi_passive = (
            F.dtype(0.5) * params.mu * (I2 - F.dtype(3.0))
            - params.mu * (I3 - F.dtype(1.0))
            + F.dtype(0.5) * params.lambda_ * math.square(I3 - F.dtype(1.0))
        )
        Psi = (
            params.muscle_fraction * Psi_active
            + (F.dtype(1.0) - params.muscle_fraction) * Psi_passive
        )
        # Psi_ARAP_active = ArapMuscle.energy_density_func(
        #     F, Phace._arap_active_params(params)
        # )  # float
        # Psi_ARAP_passive = Arap.energy_density_func(
        #     F, Phace._arap_params(params)
        # )  # float
        # Psi_ARAP = (
        #     params.muscle_fraction * Psi_ARAP_active
        #     + (_1 - params.muscle_fraction) * Psi_ARAP_passive
        # )  # float
        # Psi_VP = params.lambda_ * math.square(J - _1)  # float
        # Psi = _2 * Psi_ARAP + Psi_VP  # float
        return Psi

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def first_piola_kirchhoff_stress_func(
        F: mat33, params: ParamsElem, *, clamp: bool = False
    ) -> mat33:
        # wp.printf("StableNeoHookean.first_piola_kirchhoff_stress_func called\n")
        A = func.make_activation_mat33(params.activation)  # mat33
        G = F @ A  # mat33
        # I2 = func.I2(G)  # float
        I3 = func.I3(G)  # float
        g2 = func.g2(G)  # mat33
        g3 = func.g3(G)  # mat33
        PK1_muscle = (
            F.dtype(0.5) * params.mu * G
            - params.mu * g3
            + params.lambda_ * (I3 - F.dtype(1.0)) * g3
        ) @ wp.transpose(A)  # mat33
        # I2 = func.I2(F)  # float
        I3 = func.I3(F)  # float
        g2 = func.g2(F)  # mat33
        g3 = func.g3(F)  # mat33
        PK1_passive = (
            F.dtype(0.5) * params.mu * g2
            - params.mu * g3
            + params.lambda_ * (I3 - F.dtype(1.0)) * g3
        )  # mat33
        PK1 = (
            params.muscle_fraction * PK1_muscle
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
        A = func.make_activation_mat33(params.activation)  # mat33
        G = F @ A
        I2 = func.I2(G)  # float
        I3 = func.I3(G)  # float
        g2 = func.g2(G)  # mat33
        g3 = func.g3(G)  # mat33
        dhdX_A = dhdX @ wp.transpose(A)  # mat43
        dPsi_dI2 = F.dtype(0.5) * params.mu  # float
        dPsi_dI3 = -params.mu + params.lambda_ * (I3 - F.dtype(1.0))  # float
        dPsi_dI3_2 = params.lambda_  # float
        # h2_diag = func.h2_diag(dhdX_A, g2)  # mat33
        h3_diag = func.h3_diag(dhdX_A, g3)  # mat33
        h5_diag = func.h5_diag(dhdX_A)  # mat33
        h6_diag = func.h6_diag(dhdX_A, F)  # mat33
        diag_muscle = (
            dPsi_dI2 * h5_diag + dPsi_dI3 * h6_diag + dPsi_dI3_2 * h3_diag
        )  # mat33

        I2 = func.I2(F)  # float
        I3 = func.I3(F)  # float
        g2 = func.g2(F)  # mat33
        g3 = func.g3(F)  # mat33
        dPsi_dI2 = F.dtype(0.5) * params.mu  # float
        dPsi_dI3 = -params.mu + params.lambda_ * (I3 - F.dtype(1.0))  # float
        dPsi_dI3_2 = params.lambda_  # float
        h3_diag = func.h3_diag(dhdX, g3)  # mat33
        h5_diag = func.h5_diag(dhdX)  # mat33
        h6_diag = func.h6_diag(dhdX, F)  # mat33
        diag_passive = (
            dPsi_dI2 * h5_diag + dPsi_dI3 * h6_diag + dPsi_dI3_2 * h3_diag
        )  # mat33

        diag = (
            params.muscle_fraction * diag_muscle
            + (F.dtype(1.0) - params.muscle_fraction) * diag_passive
        )  # mat33

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
            F, p, dhdX, StableNeoHookean._arap_active_params(params), clamp=clamp
        )  # mat43
        prod_arap_passive = Arap.energy_density_hess_prod_func(
            F, p, dhdX, StableNeoHookean._arap_params(params), clamp=clamp
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
        A = func.make_activation_mat33(params.activation)  # mat33
        G = F @ A
        I2 = func.I2(G)  # float
        I3 = func.I3(G)  # float
        g2 = func.g2(G)  # mat33
        g3 = func.g3(G)  # mat33
        dhdX_A = dhdX @ wp.transpose(A)  # mat43
        dPsi_dI2 = F.dtype(0.5) * params.mu  # float
        dPsi_dI3 = -params.mu + params.lambda_ * (I3 - F.dtype(1.0))  # float
        dPsi_dI3_2 = params.lambda_  # float
        # h2_quad = func.h2_quad(dhdX_A, g2)  # mat33
        h3_quad = func.h3_quad(p, dhdX_A, g3)  # mat33
        h5_quad = func.h5_quad(p, dhdX_A)  # mat33
        h6_quad = func.h6_quad(p, dhdX_A, F)  # mat33
        quad_muscle = (
            dPsi_dI2 * h5_quad + dPsi_dI3 * h6_quad + dPsi_dI3_2 * h3_quad
        )  # mat33

        I2 = func.I2(F)  # float
        I3 = func.I3(F)  # float
        g2 = func.g2(F)  # mat33
        g3 = func.g3(F)  # mat33
        dPsi_dI2 = F.dtype(0.5) * params.mu  # float
        dPsi_dI3 = -params.mu + params.lambda_ * (I3 - F.dtype(1.0))  # float
        dPsi_dI3_2 = params.lambda_  # float
        h3_quad = func.h3_quad(p, dhdX, g3)  # mat33
        h5_quad = func.h5_quad(p, dhdX)  # mat33
        h6_quad = func.h6_quad(p, dhdX, F)  # mat33
        quad_passive = (
            dPsi_dI2 * h5_quad + dPsi_dI3 * h6_quad + dPsi_dI3_2 * h3_quad
        )  # mat33

        quad = (
            params.muscle_fraction * quad_muscle
            + (F.dtype(1.0) - params.muscle_fraction) * quad_passive
        )  # mat33

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
            F, dhdX, StableNeoHookean._arap_active_params(params), clamp=clamp
        )
        bd_arap_passive = Arap.energy_density_hess_block_diag_func(
            F, dhdX, StableNeoHookean._arap_params(params), clamp=clamp
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
