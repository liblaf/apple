from typing import Any, no_type_check, override

import warp as wp

from liblaf.apple.constants import MU
from liblaf.apple.jax.fem import Region
from liblaf.apple.warp import math, utils
from liblaf.apple.warp import types as _t

from . import func
from ._base import Hyperelastic

mat33 = Any
mat43 = Any
scalar = Any


class ARAP(Hyperelastic):
    @override
    @wp.struct
    class Params:
        mu: wp.array(dtype=_t.float_)

    @override
    @wp.struct
    class ParamsElem:
        mu: _t.float_

    @override
    @classmethod
    @no_type_check
    def make_params(cls, region: Region) -> Params:
        params = cls.Params()
        params.mu = utils.to_warp(region.cell_data[MU], _t.float_)
        return params

    @override
    @staticmethod
    @wp.func
    @no_type_check
    def get_cell_params(params: Params, cid: int) -> ParamsElem:
        return ARAP.ParamsElem(mu=params.mu[cid])

    @override
    @staticmethod
    @wp.func
    @no_type_check
    def energy_density_func(F: mat33, params: ParamsElem) -> scalar:
        R, _ = math.polar_rv(F)
        Psi = F.dtype(0.5) * params.mu * math.fro_norm_square(F - R)
        return Psi

    @override
    @staticmethod
    @wp.func
    @no_type_check
    def first_piola_kirchhoff_stress_func(F: mat33, params: ParamsElem) -> mat33:
        R, _ = math.polar_rv(F)
        PK1 = params.mu * (F - R)
        return PK1

    @override
    @staticmethod
    @wp.func
    @no_type_check
    def energy_density_hess_diag_func(
        F: mat33, dhdX: mat43, params: ParamsElem
    ) -> mat33:
        U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
        lambdas = func.lambdas(s)  # vec3
        Q0, Q1, Q2 = func.Qs(U, V)  # mat33, mat33, mat33
        h4_diag = func.h4_diag(dhdX, lambdas, Q0, Q1, Q2)  # mat43
        h5_diag = func.h5_diag(dhdX)  # mat43
        h_diag = -F.dtype(2.0) * h4_diag + h5_diag  # mat43
        return F.dtype(0.5) * params.mu * h_diag  # mat43

    @override
    @staticmethod
    @wp.func
    @no_type_check
    def energy_density_hess_prod_func(
        F: mat33, p: mat43, dhdX: mat43, params: ParamsElem
    ) -> mat33:
        U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
        lambdas = func.lambdas(s)  # vec3
        Q0, Q1, Q2 = func.Qs(U, V)  # mat33, mat33, mat33
        h4_prod = func.h4_prod(p, dhdX, lambdas, Q0, Q1, Q2)  # mat43
        h5_prod = func.h5_prod(p, dhdX)  # mat43
        h_prod = -F.dtype(2.0) * h4_prod + h5_prod  # mat43
        return F.dtype(0.5) * params.mu * h_prod  # mat43

    @override
    @staticmethod
    @wp.func
    @no_type_check
    def energy_density_hess_quad_func(
        F: mat33, p: mat43, dhdX: mat43, params: ParamsElem
    ) -> scalar:
        U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
        lambdas = func.lambdas(s)  # vec3
        Q0, Q1, Q2 = func.Qs(U, V)  # mat33, mat33, mat33
        h4_quad = func.h4_quad(p, dhdX, lambdas, Q0, Q1, Q2)
        h5_quad = func.h5_quad(p, dhdX)
        h_quad = -F.dtype(2.0) * h4_quad + h5_quad
        return F.dtype(0.5) * params.mu * h_quad
