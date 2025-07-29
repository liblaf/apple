r"""...

$$
\begin{align*}
    \Psi_{\sqrt{\text{vf}}} & = \frac{k}{2} (\|\mathbf{t}\| - \varepsilon)^2 \\
    \frac{\partial\Psi_{\sqrt{\text{vf}}}}{\partial\mathbf{x}} & = k (\|\mathbf{t}\| - \varepsilon) \frac{\mathbf{t}}{\|\mathbf{t}\|} \frac{\partial\mathbf{t}}{\partial\mathbf{x}} \\
    \frac{\partial^2\Psi_{\sqrt{\text{vf}}}}{\partial\mathbf{x}^2} & = k \left[\left(\frac{1}{\mathbf{t}^T \mathbf{t}} - \frac{\|\mathbf{t}\| - \varepsilon}{(\mathbf{t}^T \mathbf{t})^{\frac{3}{2}}}\right) \mathbf{g} \mathbf{g}^T + \frac{\|\mathbf{t}\| + \varepsilon}{\|\mathbf{t}\|} \frac{\partial\mathbf{t}}{\partial\mathbf{x}}^T \frac{\partial\mathbf{t}}{\partial\mathbf{x}}\right] \\
    \mathbf{g} & = \frac{\partial\mathbf{t}}{\partial\mathbf{x}}^T \mathbf{t}
\end{align*}
$$

References:
    1. Theodore Kim and David Eberle. 2022. Dynamic deformables: implementation and production practicalities (now with code!). In ACM SIGGRAPH 2022 Courses (SIGGRAPH '22). Association for Computing Machinery, New York, NY, USA, Article 7, 1–259. https://doi.org/10.1145/3532720.3535628. 14.5 The Actual Vertex-Face Energy Used in Fizt. P186-P190
"""

from typing import no_type_check

import warp as wp

from liblaf.apple import func, utils


@no_type_check
@utils.jax_kernel(num_outputs=6)
def collision_detect_vert_face_kernel(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    rest_length: wp.array(dtype=wp.float32),
    max_dist: wp.array(dtype=wp.float32),
    epsilon: wp.array(dtype=wp.float32),
    # outputs
    closest: wp.array(dtype=wp.vec3),
    collide: wp.array(dtype=bool),
    distance: wp.array(dtype=wp.float32),
    face_id: wp.array(dtype=wp.int32),
    face_normal: wp.array(dtype=wp.vec3),
    uv: wp.array(dtype=wp.vec2),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_sign_normal(
        mesh_id, point, max_dist=max_dist[0], epsilon=epsilon[0]
    )
    if not query.result:
        collide[tid] = False
        return
    closest_i = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    distance_i = wp.sign(query.sign) * wp.length(closest_i - point)
    if distance_i > rest_length[0]:
        collide[tid] = False
        return
    closest[tid] = closest_i
    collide[tid] = True
    distance[tid] = distance_i
    face_id[tid] = query.face
    face_normal[tid] = wp.mesh_eval_face_normal(mesh_id, query.face)
    uv[tid] = wp.vec2(query.u, query.v)


# region Function


@no_type_check
@wp.func
def collision_energy_vert_face_fun_func(
    point: wp.vec3,
    closest: wp.vec3,
    distance: float,
    rest_length: float,
    stiffness: float,
) -> wp.float32:
    t = point - closest
    rest_length = wp.sign(distance) * rest_length
    return 0.5 * stiffness * func.square(wp.length(t) - rest_length)


@no_type_check
@utils.jax_kernel
def collision_energy_vert_face_fun_kernel(
    points: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collide: wp.array(dtype=bool),
    distance: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    energy: wp.array(dtype=float),
) -> None:
    tid = wp.tid()
    if not collide[tid]:
        energy[tid] = 0.0
        return
    energy[tid] = collision_energy_vert_face_fun_func(
        point=points[tid],
        closest=closest[tid],
        distance=distance[tid],
        rest_length=rest_length[0],
        stiffness=stiffness[0],
    )


# endregion Function


# region Jacobian


@no_type_check
@wp.func
def collision_energy_vert_face_jac_func(
    point: wp.vec3,
    closest: wp.vec3,
    distance: float,
    rest_length: float,
    stiffness: float,
) -> wp.vec3:
    t = point - closest
    rest_length = wp.sign(distance) * rest_length
    return stiffness * (wp.length(t) - rest_length) * wp.normalize(t)


@no_type_check
@utils.jax_kernel
def collision_energy_vert_face_jac_kernel(
    points: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collide: wp.array(dtype=bool),
    distance: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    jac: wp.array(dtype=wp.vec3),
) -> None:
    tid = wp.tid()
    if not collide[tid]:
        jac[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    jac[tid] = collision_energy_vert_face_jac_func(
        point=points[tid],
        closest=closest[tid],
        distance=distance[tid],
        rest_length=rest_length[0],
        stiffness=stiffness[0],
    )


# endregion Jacobian

# region Hessian Diagonal


@wp.func
def collision_energy_vert_face_hess_diag_func(
    point: wp.vec3,
    closest: wp.vec3,
    distance: float,
    rest_length: float,
    stiffness: float,
) -> wp.vec3:
    t = point - closest
    rest_length = wp.sign(distance) * rest_length
    t_norm = wp.length(t)
    tTt = wp.length_sq(t)
    a = 1.0 / tTt - (t_norm - rest_length) / wp.pow(tTt, 1.5)
    b = (t_norm - rest_length) / t_norm
    # return wp.vec3(0.0, 0.0, 0.0)
    return stiffness * (a * wp.cw_mul(t, t) + b * wp.vec3(1.0, 1.0, 1.0))


@no_type_check
@utils.jax_kernel
def collision_energy_vert_face_hess_diag_kernel(
    points: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collide: wp.array(dtype=bool),
    distance: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    hess_diag: wp.array(dtype=wp.vec3),
) -> None:
    tid = wp.tid()
    if not collide[tid]:
        hess_diag[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    hess_diag_i = collision_energy_vert_face_hess_diag_func(
        point=points[tid],
        closest=closest[tid],
        distance=distance[tid],
        rest_length=rest_length[0],
        stiffness=stiffness[0],
    )
    hess_diag_i = wp.max(hess_diag_i, wp.vec3(0.0, 0.0, 0.0))
    hess_diag[tid] = hess_diag_i


# endregion Hessian Diagonal


# region Hessian Quadratic Form


@wp.func
def collision_energy_vert_face_hess_quad_func(
    point: wp.vec3,
    p: wp.vec3,
    closest: wp.vec3,
    distance: float,
    rest_length: float,
    stiffness: float,
) -> float:
    t = point - closest
    rest_length = wp.sign(distance) * rest_length
    t_norm = wp.length(t)
    tTt = wp.length_sq(t)
    a = 1.0 / tTt - (t_norm - rest_length) / wp.pow(tTt, 1.5)
    b = (t_norm - rest_length) / t_norm
    return stiffness * (a * func.square(wp.dot(t, p)) + b * wp.dot(p, p))


@no_type_check
@utils.jax_kernel
def collision_energy_vert_face_hess_quad_kernel(
    points: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collide: wp.array(dtype=bool),
    distance: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    hess_quad: wp.array(dtype=float),
) -> None:
    tid = wp.tid()
    if not collide[tid]:
        hess_quad[tid] = 0.0
        return
    hess_quad[tid] = collision_energy_vert_face_hess_quad_func(
        point=points[tid],
        p=p[tid],
        closest=closest[tid],
        distance=distance[tid],
        rest_length=rest_length[0],
        stiffness=stiffness[0],
    )


# endregion Hessian Quadratic Form
