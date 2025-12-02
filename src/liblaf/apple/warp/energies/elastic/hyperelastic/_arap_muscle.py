from collections.abc import Mapping
from typing import Any, no_type_check, override

import warp as wp
from liblaf.peach import tree

import liblaf.apple.warp.types as wpt
from liblaf.apple.constants import ACTIVATION, MU
from liblaf.apple.jax.fem import Region
from liblaf.apple.warp import math, utils

from . import func
from ._arap import Arap

mat33 = Any
mat43 = Any
scalar = Any


@tree.define
class ArapMuscle(Arap):
    @override
    @wp.struct
    class Params:
        activation: wp.array(dtype=wpt.vec6)
        mu: wp.array(dtype=wpt.float_)

    @override
    @wp.struct
    class ParamsElem:
        activation: wpt.vec6
        mu: wpt.float_

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def get_cell_params(params: Params, cid: int) -> ParamsElem:
        return ArapMuscle.ParamsElem(
            activation=params.activation[cid], mu=params.mu[cid]
        )

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_func(F: mat33, params: ParamsElem) -> scalar:
        A = func.make_activation_mat33(params.activation)  # mat33
        R, _ = math.polar_rv(F)
        Psi = F.dtype(0.5) * params.mu * math.fro_norm_square(F - R @ A)
        return Psi

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def first_piola_kirchhoff_stress_func(
        F: mat33, params: ParamsElem, *, clamp: bool = True
    ) -> mat33:
        A = func.make_activation_mat33(params.activation)  # mat33
        U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
        R = U @ wp.transpose(V)  # mat33
        lambdas = func.lambdas(s, clamp=clamp)  # vec3
        Q0, Q1, Q2 = func.Qs(U, V)  # mat33, mat33, mat33
        M = F - R @ A  # mat33
        PK1 = params.mu * (M - func.dRdF_vjp(M @ A, lambdas, Q0, Q1, Q2))  # mat33
        return PK1

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_hess_diag_func(
        F: mat33, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> mat33:
        U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
        h4_diag = func.h4_diag(dhdX, U, s, V, clamp=clamp)  # mat43
        h5_diag = func.h5_diag(dhdX)  # mat43
        h_diag = -F.dtype(2.0) * h4_diag + h5_diag  # mat43
        return F.dtype(0.5) * params.mu * h_diag  # mat43

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_hess_prod_func(
        F: mat33, p: mat43, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> mat33:
        U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
        h4_prod = func.h4_prod(p, dhdX, U, s, V, clamp=clamp)  # mat43
        h5_prod = func.h5_prod(p, dhdX)  # mat43
        h_prod = -F.dtype(2.0) * h4_prod + h5_prod  # mat43
        return F.dtype(0.5) * params.mu * h_prod  # mat43

    @override
    @staticmethod
    @no_type_check
    @wp.func
    def energy_density_hess_quad_func(
        F: mat33, p: mat43, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> scalar:
        U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
        h4_quad = func.h4_quad(p, dhdX, U, s, V, clamp=clamp)  # float
        h5_quad = func.h5_quad(p, dhdX)  # float
        h_quad = -F.dtype(2.0) * h4_quad + h5_quad
        return F.dtype(0.5) * params.mu * h_quad

    @override
    @classmethod
    def _params_fields_from_region(cls, region: Region) -> Mapping[str, wp.array]:
        fields: dict[str, wp.array] = {}
        fields["activation"] = utils.to_warp(region.cell_data[ACTIVATION], wpt.vec6)
        fields["mu"] = utils.to_warp(region.cell_data[MU], wpt.float_)
        return fields
