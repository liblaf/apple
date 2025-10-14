from typing import no_type_check

import warp as wp

import liblaf.apple.warp.sim.energy.elastic.arap.func as arap
import liblaf.apple.warp.sim.energy.elastic.arap_active.func as arap_active
import liblaf.apple.warp.sim.energy.elastic.func as _f
from liblaf.apple.warp import math
from liblaf.apple.warp.typing import float_, mat33, mat43, vec6


@wp.struct
class ParamsElem:
    activation: vec6
    active_fraction: float_
    lambda_: float_
    mu: float_


@wp.struct
class Params:
    activation: wp.array(dtype=vec6)
    active_fraction: wp.array(dtype=float_)
    lambda_: wp.array(dtype=float_)
    mu: wp.array(dtype=float_)


@wp.func
@no_type_check
def energy_density(F: mat33, params: ParamsElem):  # -> float:
    _1 = type(F[0, 0])(1.0)
    _2 = type(F[0, 0])(2.0)
    J = _f.I3(F)  # float
    Psi_ARAP_active = arap_active.energy_density(
        F, arap_active.ParamsElem(activation=params.activation, mu=params.mu)
    )  # float
    Psi_ARAP_passive = arap.energy_density(F, arap.ParamsElem(mu=params.mu))  # float
    Psi_ARAP = (
        params.active_fraction * Psi_ARAP_active
        + (_1 - params.active_fraction) * Psi_ARAP_passive
    )  # float
    Psi_VP = params.lambda_ * math.square(J - _1)  # float
    Psi = _2 * Psi_ARAP + Psi_VP  # float
    return Psi


@wp.func
@no_type_check
def first_piola_kirchhoff_stress_tensor(F: mat33, params: ParamsElem):  # -> mat33:
    _1 = type(F[0, 0])(1.0)
    _2 = type(F[0, 0])(2.0)
    J = _f.I3(F)  # float
    g3 = _f.g3(F)  # mat33
    PK1_ARAP_active = arap_active.first_piola_kirchhoff_stress_tensor(
        F, arap_active.ParamsElem(activation=params.activation, mu=params.mu)
    )  # mat33
    PK1_ARAP_passive = arap.first_piola_kirchhoff_stress_tensor(
        F, arap.ParamsElem(mu=params.mu)
    )  # mat33
    PK1_ARAP = (
        params.active_fraction * PK1_ARAP_active
        + (_1 - params.active_fraction) * PK1_ARAP_passive
    )  # mat33
    PK1_VP = _2 * params.lambda_ * (J - _1) * g3  # mat33
    PK1 = _2 * PK1_ARAP + PK1_VP  # mat33
    return PK1


@wp.func
@no_type_check
def energy_density_hess_diag(F: mat33, dhdX: mat43, params: ParamsElem):  # -> mat43:
    _1 = type(F[0, 0])(1.0)
    _2 = type(F[0, 0])(2.0)
    J = _f.I3(F)  # float
    g3 = _f.g3(F)  # mat33
    diag_arap_active = arap_active.energy_density_hess_diag(
        F, dhdX, arap_active.ParamsElem(activation=params.activation, mu=params.mu)
    )  # mat43
    diag_arap_passive = arap.energy_density_hess_diag(
        F, dhdX, arap.ParamsElem(mu=params.mu)
    )  # mat43
    diag_arap = (
        params.active_fraction * diag_arap_active
        + (_1 - params.active_fraction) * diag_arap_passive
    )  # mat43
    d2Psi_dI32 = _2 * params.lambda_  # float
    dPsi_dI3 = _2 * params.lambda_ * (J - _1)  # float
    h3_diag = _f.h3_diag(dhdX, g3)  # mat43
    h6_diag = _f.h6_diag(dhdX, F)  # mat43
    diag_vp = d2Psi_dI32 * h3_diag + dPsi_dI3 * h6_diag  # mat43
    diag = _2 * diag_arap + diag_vp  # mat43
    return diag


@wp.func
@no_type_check
def energy_density_hess_prod(
    F: mat33, p: mat43, dhdX: mat43, params: ParamsElem
):  # -> mat43:
    _1 = type(F[0, 0])(1.0)
    _2 = type(F[0, 0])(2.0)
    J = _f.I3(F)  # float
    g3 = _f.g3(F)  # mat33
    prod_arap_active = arap_active.energy_density_hess_prod(
        F, p, dhdX, arap_active.ParamsElem(activation=params.activation, mu=params.mu)
    )  # mat43
    prod_arap_passive = arap.energy_density_hess_prod(
        F, p, dhdX, arap.ParamsElem(mu=params.mu)
    )  # mat43
    prod_arap = (
        params.active_fraction * prod_arap_active
        + (_1 - params.active_fraction) * prod_arap_passive
    )  # mat43
    d2Psi_dI32 = _2 * params.lambda_  # float
    dPsi_dI3 = _2 * params.lambda_ * (J - _1)  # float
    h3_prod = _f.h3_prod(p, dhdX, g3)  # mat43
    h6_prod = _f.h6_prod(p, dhdX, F)  # mat43
    prod_vp = d2Psi_dI32 * h3_prod + dPsi_dI3 * h6_prod  # mat43
    prod = _2 * prod_arap + prod_vp  # mat43
    return prod


@wp.func
@no_type_check
def energy_density_hess_quad(
    F: mat33, p: mat43, dhdX: mat43, params: ParamsElem
):  # -> float:
    _1 = type(F[0, 0])(1.0)
    _2 = type(F[0, 0])(2.0)
    J = _f.I3(F)  # float
    g3 = _f.g3(F)  # mat33
    quad_arap_active = arap_active.energy_density_hess_quad(
        F, p, dhdX, arap_active.ParamsElem(activation=params.activation, mu=params.mu)
    )
    quad_arap_passive = arap.energy_density_hess_quad(
        F, p, dhdX, arap.ParamsElem(mu=params.mu)
    )
    quad_arap = (
        params.active_fraction * quad_arap_active
        + (_1 - params.active_fraction) * quad_arap_passive
    )
    d2Psi_dI32 = _2 * params.lambda_  # float
    dPsi_dI3 = _2 * params.lambda_ * (J - _1)  # float
    h3_quad = _f.h3_quad(p, dhdX, g3)  # float
    h6_quad = _f.h6_quad(p, dhdX, F)  # float
    quad_vp = d2Psi_dI32 * h3_quad + dPsi_dI3 * h6_quad  # float
    # if quad_vp < 0.0:
    #     wp.printf(
    #         "h3_quad = %f, h6_quad = %f, quad_vp = %f\n", h3_quad, h6_quad, quad_vp
    #     )
    quad = _2 * quad_arap + quad_vp  # float
    return quad


@wp.func
@no_type_check
def get_cell_params(params: Params, cid: int) -> ParamsElem:
    cell_params = ParamsElem(
        activation=params.activation[cid],
        active_fraction=params.active_fraction[cid],
        lambda_=params.lambda_[cid],
        mu=params.mu[cid],
    )
    return cell_params
