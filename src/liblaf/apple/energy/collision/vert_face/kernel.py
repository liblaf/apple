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

from typing import cast, no_type_check

import warp as wp

from liblaf.apple import func, utils


@no_type_check
@utils.jax_kernel(num_outputs=8)
def collision_detect_vert_face_kernel(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    rest_length: wp.array(dtype=wp.float32),
    max_dist: wp.array(dtype=wp.float32),
    epsilon: wp.array(dtype=wp.float32),
    # outputs
    closest: wp.array(dtype=wp.vec3),
    collision_to_vertex: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    face_id: wp.array(dtype=wp.int32),
    face_normal: wp.array(dtype=wp.vec3),
    uv: wp.array(dtype=wp.vec2),
    vertex_to_collision: wp.array(dtype=wp.int32),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_sign_normal(
        mesh_id, point, max_dist=max_dist[0], epsilon=epsilon[0]
    )
    if not query.result:
        return
    closest_i = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    distance_i = wp.sign(query.sign) * wp.length(closest_i - point)
    if distance_i > rest_length[0]:
        return
    collision_id = wp.atomic_add(count, 0, value=1)
    closest[collision_id] = closest_i
    collision_to_vertex[collision_id] = tid
    distance[collision_id] = distance_i
    face_id[collision_id] = query.face
    face_normal[collision_id] = wp.mesh_eval_face_normal(mesh_id, query.face)
    uv[collision_id] = wp.vec2(query.u, query.v)
    vertex_to_collision[tid] = collision_id


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
@wp.kernel
def collision_energy_vert_face_fun_kernel(
    points: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collision_to_vertex: wp.array(dtype=int),
    distance: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    energy: wp.array(dtype=float),
) -> None:
    tid = wp.tid()
    vert_id = collision_to_vertex[tid]
    energy[0] += collision_energy_vert_face_fun_func(
        point=points[vert_id],
        closest=closest[tid],
        distance=distance[tid],
        rest_length=rest_length[0],
        stiffness=stiffness[0],
    )


@no_type_check
@utils.jax_callable
def collision_energy_vert_face_fun_callable(
    points: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collision_to_vertex: wp.array(dtype=int),
    distance: wp.array(dtype=float),
    count: wp.array(dtype=int),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    energy: wp.array(dtype=float),
) -> None:
    wp.launch(
        kernel=collision_energy_vert_face_fun_kernel,
        dim=(count.numpy().item(),),
        inputs=[points, closest, collision_to_vertex, distance, rest_length, stiffness],
        outputs=[energy],
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
@wp.kernel
def collision_energy_vert_face_jac_kernel(
    points: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collision_to_vertex: wp.array(dtype=int),
    distance: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    jac: wp.array(dtype=wp.vec3),
) -> None:
    tid = wp.tid()
    vert_id = collision_to_vertex[tid]
    jac[vert_id] = collision_energy_vert_face_jac_func(
        point=points[vert_id],
        closest=closest[tid],
        distance=distance[tid],
        rest_length=rest_length[0],
        stiffness=stiffness[0],
    )


@no_type_check
@utils.jax_callable
def collision_energy_vert_face_jac_callable(
    points: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collision_to_vertex: wp.array(dtype=int),
    distance: wp.array(dtype=float),
    count: wp.array(dtype=int),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    jac: wp.array(dtype=wp.vec3),
) -> None:
    jac: wp.array = cast("wp.array", jac)
    jac.zero_()
    wp.launch(
        kernel=collision_energy_vert_face_jac_kernel,
        dim=(count.numpy().item(),),
        inputs=[points, closest, collision_to_vertex, distance, rest_length, stiffness],
        outputs=[jac],
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
    return stiffness * (a * wp.cw_mul(t, t) + b * wp.vec3(1.0, 1.0, 1.0))


@no_type_check
@wp.kernel
def collision_energy_vert_face_hess_diag_kernel(
    points: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collision_to_vertex: wp.array(dtype=int),
    distance: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    hess_diag: wp.array(dtype=wp.vec3),
) -> None:
    tid = wp.tid()
    vert_id = collision_to_vertex[tid]
    hess_diag_i = collision_energy_vert_face_hess_diag_func(
        point=points[vert_id],
        closest=closest[tid],
        distance=distance[tid],
        rest_length=rest_length[0],
        stiffness=stiffness[0],
    )
    hess_diag_i = wp.max(hess_diag_i, wp.vec3(0.0, 0.0, 0.0))
    hess_diag[vert_id] = hess_diag_i


@no_type_check
@utils.jax_callable
def collision_energy_vert_face_hess_diag_callable(
    points: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collision_to_vertex: wp.array(dtype=int),
    distance: wp.array(dtype=float),
    count: wp.array(dtype=int),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    hess_diag: wp.array(dtype=wp.vec3),
) -> None:
    hess_diag: wp.array = cast("wp.array", hess_diag)
    hess_diag.zero_()
    wp.launch(
        kernel=collision_energy_vert_face_hess_diag_kernel,
        dim=(count.numpy().item(),),
        inputs=[points, closest, collision_to_vertex, distance, rest_length, stiffness],
        outputs=[hess_diag],
    )


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
) -> wp.float32:
    t = point - closest
    rest_length = wp.sign(distance) * rest_length
    t_norm = wp.length(t)
    tTt = wp.length_sq(t)
    a = 1.0 / tTt - (t_norm - rest_length) / wp.pow(tTt, 1.5)
    b = (t_norm - rest_length) / t_norm
    return stiffness * (a * func.square(wp.dot(t, p)) + b * wp.dot(p, p))


@no_type_check
@wp.kernel
def collision_energy_vert_face_hess_quad_kernel(
    points: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collision_to_vertex: wp.array(dtype=int),
    distance: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    hess_quad: wp.array(dtype=float),
) -> None:
    tid = wp.tid()
    vert_id = collision_to_vertex[tid]
    hess_quad_i = collision_energy_vert_face_hess_quad_func(
        point=points[vert_id],
        p=p[vert_id],
        closest=closest[tid],
        distance=distance[tid],
        rest_length=rest_length[0],
        stiffness=stiffness[0],
    )
    hess_quad_i = wp.max(hess_quad_i, 0.0)
    hess_quad[0] += hess_quad_i


@no_type_check
@utils.jax_callable
def collision_energy_vert_face_hess_quad_callable(
    points: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    collision_to_vertex: wp.array(dtype=int),
    distance: wp.array(dtype=float),
    count: wp.array(dtype=int),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    # outputs
    hess_quad: wp.array(dtype=float),
) -> None:
    hess_quad: wp.array = cast("wp.array", hess_quad)
    hess_quad.zero_()
    wp.launch(
        kernel=collision_energy_vert_face_hess_quad_kernel,
        dim=(count.numpy().item(),),
        inputs=[
            points,
            p,
            closest,
            collision_to_vertex,
            distance,
            rest_length,
            stiffness,
        ],
        outputs=[hess_quad],
    )


# endregion Hessian Quadratic Form
