"""
IPC Toolkit
"""

from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import scipy.sparse
import typing
from . import filib
from . import ogc
from . import tight_inclusion

__all__: list[str] = [
    "AABB",
    "ABS",
    "AUTO",
    "AdditiveCCD",
    "Barrier",
    "BarrierPotential",
    "BroadPhase",
    "BruteForce",
    "CLAMP",
    "Candidates",
    "ClampedLogBarrier",
    "ClampedLogSqBarrier",
    "CollisionFilter",
    "CollisionMesh",
    "CollisionStencil",
    "CubicBarrier",
    "EA0_EB",
    "EA0_EB0",
    "EA0_EB1",
    "EA1_EB",
    "EA1_EB0",
    "EA1_EB1",
    "EA_EB",
    "EA_EB0",
    "EA_EB1",
    "Edge2Point2Collision",
    "EdgeEdgeCandidate",
    "EdgeEdgeDistanceType",
    "EdgeEdgeNormalCollision",
    "EdgeEdgeTangentialCollision",
    "EdgeFaceCandidate",
    "EdgeVertexCandidate",
    "EdgeVertexNormalCollision",
    "EdgeVertexTangentialCollision",
    "FaceFaceCandidate",
    "FaceVertexCandidate",
    "FaceVertexNormalCollision",
    "FaceVertexTangentialCollision",
    "FrictionPotential",
    "HashGrid",
    "HashItem",
    "Hyperplane",
    "IntervalNonlinearTrajectory",
    "LBVH",
    "LBVH_Node",
    "LoggerLevel",
    "NONE",
    "NarrowPhaseCCD",
    "NonlinearCCD",
    "NonlinearTrajectory",
    "NormalAdhesionPotential",
    "NormalCollision",
    "NormalCollisions",
    "NormalPotential",
    "NormalizedClampedLogBarrier",
    "PSDProjectionMethod",
    "P_E",
    "P_E0",
    "P_E1",
    "P_E2",
    "P_T",
    "P_T0",
    "P_T1",
    "P_T2",
    "PlaneVertexCandidate",
    "PlaneVertexNormalCollision",
    "PlaneVertexTangentialCollision",
    "Point2Point2Collision",
    "PointEdgeDistanceType",
    "PointTriangleDistanceType",
    "SmoothCollision2",
    "SmoothCollisions",
    "SmoothContactParameters",
    "SmoothPotential",
    "SpatialHash",
    "SweepAndPrune",
    "TangentialAdhesionPotential",
    "TangentialCollision",
    "TangentialCollisions",
    "TangentialPotential",
    "TightInclusionCCD",
    "TwoStageBarrier",
    "VertexVertexCandidate",
    "VertexVertexNormalCollision",
    "VertexVertexTangentialCollision",
    "anisotropic_mu_eff_f",
    "barrier",
    "barrier_first_derivative",
    "barrier_force_magnitude",
    "barrier_force_magnitude_gradient",
    "barrier_second_derivative",
    "build_edge_boxes",
    "build_face_boxes",
    "build_vertex_boxes",
    "check_initial_distance",
    "compute_collision_free_stepsize",
    "critical",
    "cross_product_matrix",
    "cross_product_matrix_jacobian",
    "debug",
    "dihedral_angle",
    "dihedral_angle_gradient",
    "edge_edge_aabb_ccd",
    "edge_edge_aabb_cd",
    "edge_edge_closest_point",
    "edge_edge_closest_point_jacobian",
    "edge_edge_cross_squarednorm",
    "edge_edge_cross_squarednorm_gradient",
    "edge_edge_cross_squarednorm_hessian",
    "edge_edge_distance",
    "edge_edge_distance_gradient",
    "edge_edge_distance_hessian",
    "edge_edge_distance_type",
    "edge_edge_mollifier",
    "edge_edge_mollifier_derivative_wrt_eps_x",
    "edge_edge_mollifier_gradient",
    "edge_edge_mollifier_gradient_derivative_wrt_eps_x",
    "edge_edge_mollifier_gradient_jacobian_wrt_x",
    "edge_edge_mollifier_gradient_wrt_x",
    "edge_edge_mollifier_hessian",
    "edge_edge_mollifier_threshold",
    "edge_edge_mollifier_threshold_gradient",
    "edge_edge_parallel_distance_type",
    "edge_edge_relative_velocity",
    "edge_edge_relative_velocity_dx_dbeta",
    "edge_edge_relative_velocity_jacobian",
    "edge_edge_tangent_basis",
    "edge_edge_tangent_basis_jacobian",
    "edge_length_gradient",
    "edge_triangle_aabb_cd",
    "edges",
    "error",
    "filib",
    "get_num_threads",
    "has_intersections",
    "inexact_point_edge_ccd_2D",
    "info",
    "initial_barrier_stiffness",
    "is_edge_intersecting_triangle",
    "is_step_collision_free",
    "line_line_distance",
    "line_line_distance_gradient",
    "line_line_distance_hessian",
    "line_line_normal",
    "line_line_normal_hessian",
    "line_line_normal_jacobian",
    "line_line_signed_distance",
    "line_line_signed_distance_gradient",
    "line_line_signed_distance_hessian",
    "line_line_unnormalized_normal",
    "line_line_unnormalized_normal_hessian",
    "line_line_unnormalized_normal_jacobian",
    "make_connected_components_filter",
    "make_sparse_filter",
    "make_static_obstacle_filter",
    "make_vertex_patches_filter",
    "max_displacement_length",
    "max_edge_length",
    "max_normal_adhesion_force_magnitude",
    "mean_displacement_length",
    "mean_edge_length",
    "median_displacement_length",
    "median_edge_length",
    "normal_adhesion_potential",
    "normal_adhesion_potential_first_derivative",
    "normal_adhesion_potential_second_derivative",
    "normalization_and_jacobian",
    "normalization_and_jacobian_and_hessian",
    "normalization_hessian",
    "normalization_jacobian",
    "off",
    "ogc",
    "point_edge_aabb_ccd",
    "point_edge_aabb_cd",
    "point_edge_closest_point",
    "point_edge_closest_point_jacobian",
    "point_edge_distance",
    "point_edge_distance_gradient",
    "point_edge_distance_hessian",
    "point_edge_distance_type",
    "point_edge_relative_velocity",
    "point_edge_relative_velocity_dx_dbeta",
    "point_edge_relative_velocity_jacobian",
    "point_edge_tangent_basis",
    "point_edge_tangent_basis_jacobian",
    "point_line_distance",
    "point_line_distance_gradient",
    "point_line_distance_hessian",
    "point_line_normal",
    "point_line_signed_distance",
    "point_line_signed_distance_gradient",
    "point_line_signed_distance_hessian",
    "point_line_unnormalized_normal",
    "point_line_unnormalized_normal_jacobian",
    "point_plane_distance",
    "point_plane_distance_gradient",
    "point_plane_distance_hessian",
    "point_plane_signed_distance",
    "point_plane_signed_distance_gradient",
    "point_plane_signed_distance_hessian",
    "point_point_distance",
    "point_point_distance_gradient",
    "point_point_distance_hessian",
    "point_point_relative_velocity",
    "point_point_relative_velocity_dx_dbeta",
    "point_point_relative_velocity_jacobian",
    "point_point_tangent_basis",
    "point_point_tangent_basis_jacobian",
    "point_static_plane_ccd",
    "point_triangle_aabb_ccd",
    "point_triangle_aabb_cd",
    "point_triangle_closest_point",
    "point_triangle_closest_point_jacobian",
    "point_triangle_distance",
    "point_triangle_distance_gradient",
    "point_triangle_distance_hessian",
    "point_triangle_distance_type",
    "point_triangle_relative_velocity",
    "point_triangle_relative_velocity_dx_dbeta",
    "point_triangle_relative_velocity_jacobian",
    "point_triangle_tangent_basis",
    "point_triangle_tangent_basis_jacobian",
    "project_to_pd",
    "project_to_psd",
    "segment_segment_intersect",
    "semi_implicit_stiffness",
    "set_logger_level",
    "set_num_threads",
    "smooth_friction_f0",
    "smooth_friction_f1",
    "smooth_friction_f1_over_x",
    "smooth_friction_f2",
    "smooth_friction_f2_x_minus_f1_over_x3",
    "smooth_mu",
    "smooth_mu_a0",
    "smooth_mu_a1",
    "smooth_mu_a1_over_x",
    "smooth_mu_a2",
    "smooth_mu_a2_x_minus_mu_a1_over_x3",
    "smooth_mu_derivative",
    "smooth_mu_f0",
    "smooth_mu_f1",
    "smooth_mu_f1_over_x",
    "smooth_mu_f2",
    "smooth_mu_f2_x_minus_mu_f1_over_x3",
    "suggest_good_voxel_size",
    "tangential_adhesion_f0",
    "tangential_adhesion_f1",
    "tangential_adhesion_f1_over_x",
    "tangential_adhesion_f2",
    "tangential_adhesion_f2_x_minus_f1_over_x3",
    "tight_inclusion",
    "trace",
    "triangle_area_gradient",
    "triangle_normal",
    "triangle_normal_hessian",
    "triangle_normal_jacobian",
    "triangle_unnormalized_normal",
    "triangle_unnormalized_normal_hessian",
    "triangle_unnormalized_normal_jacobian",
    "update_barrier_stiffness",
    "vertex_to_min_edge",
    "warn",
    "world_bbox_diagonal_length",
]

class AABB:
    @staticmethod
    def conservative_inflation(
        min: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"],
        max: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"],
        inflation_radius: typing.SupportsFloat,
    ) -> tuple[
        typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ]:
        """
        Compute a conservative inflation of the AABB.
        """
    @staticmethod
    @typing.overload
    def from_point(
        p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        inflation_radius: typing.SupportsFloat = 0,
    ) -> AABB:
        """
        Construct an AABB for a static point.

        Parameters:
            p: The point's position.
            inflation_radius: Radius of a sphere around the point which the AABB encloses.

        Returns:
            The constructed AABB.
        """
    @staticmethod
    @typing.overload
    def from_point(
        p_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        p_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        inflation_radius: typing.SupportsFloat = 0,
    ) -> AABB:
        """
        Construct an AABB for a moving point (i.e. temporal edge).

        Parameters:
            p_t0: The point's position at time t=0.
            p_t1: The point's position at time t=1.
            inflation_radius: Radius of a capsule around the temporal edge which the AABB encloses.

        Returns:
            The constructed AABB.
        """
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        min: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        max: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> None: ...
    @typing.overload
    def __init__(self, aabb1: AABB, aabb2: AABB) -> None: ...
    @typing.overload
    def __init__(self, aabb1: AABB, aabb2: AABB, aabb3: AABB) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def intersects(self, other: AABB) -> bool:
        """
        Check if another AABB intersects with this one.

        Parameters:
            other: The other AABB.

        Returns:
            If the two AABBs intersect.
        """
    @property
    def max(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Maximum corner of the AABB.
        """
    @max.setter
    def max(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]
    ) -> None: ...
    @property
    def min(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Minimum corner of the AABB.
        """
    @min.setter
    def min(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]
    ) -> None: ...
    @property
    def vertex_ids(self) -> typing.Annotated[list[int], "FixedSize(3)"]:
        """
        Vertex IDs attached to the AABB.
        """
    @vertex_ids.setter
    def vertex_ids(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> None: ...

class AdditiveCCD(NarrowPhaseCCD):
    DEFAULT_CONSERVATIVE_RESCALING: typing.ClassVar[float] = 0.9
    def __init__(
        self,
        max_iterations: typing.SupportsFloat = 10000000,
        conservative_rescaling: typing.SupportsFloat = 0.9,
    ) -> None:
        """
        Construct a new AdditiveCCD object.

        Parameters:
            conservative_rescaling: The conservative rescaling of the time of impact.
        """
    @property
    def conservative_rescaling(self) -> float:
        """
        The conservative rescaling value used to avoid taking steps exactly to impact.
        """
    @conservative_rescaling.setter
    def conservative_rescaling(self, arg0: typing.SupportsFloat) -> None: ...

class Barrier:
    def __call__(self, d: typing.SupportsFloat, dhat: typing.SupportsFloat) -> float:
        """
        Evaluate the barrier function.

        Parameters:
            d: The distance.
            dhat: Activation distance of the barrier.

        Returns:
            The value of the barrier function at d.
        """
    def __init__(self) -> None: ...
    def first_derivative(
        self, d: typing.SupportsFloat, dhat: typing.SupportsFloat
    ) -> float:
        """
        Evaluate the first derivative of the barrier function wrt d.

        Parameters:
            d: The distance.
            dhat: Activation distance of the barrier.

        Returns:
            The value of the first derivative of the barrier function at d.
        """
    def second_derivative(
        self, d: typing.SupportsFloat, dhat: typing.SupportsFloat
    ) -> float:
        """
        Evaluate the second derivative of the barrier function wrt d.

        Parameters:
            d: The distance.
            dhat: Activation distance of the barrier.

        Returns:
            The value of the second derivative of the barrier function at d.
        """
    def units(self, dhat: typing.SupportsFloat) -> float:
        """
        Get the units of the barrier function.

        Parameters:
            dhat: Activation distance of the barrier.

        Returns:
            The units of the barrier function.
        """

class BarrierPotential(NormalPotential):
    @typing.overload
    def __init__(
        self,
        dhat: typing.SupportsFloat,
        stiffness: typing.SupportsFloat,
        use_physical_barrier: bool = False,
    ) -> None:
        """
        Construct a barrier potential.

        Parameters:
            dhat: The activation distance of the barrier.
            stiffness: The stiffness of the barrier.
            use_physical_barrier: Whether to use the physical barrier.
        """
    @typing.overload
    def __init__(
        self,
        barrier: Barrier,
        dhat: typing.SupportsFloat,
        stiffness: typing.SupportsFloat,
        use_physical_barrier: bool = False,
    ) -> None:
        """
        Construct a barrier potential.

        Parameters:
            barrier: The barrier function.
            dhat: The activation distance of the barrier.
            stiffness: The stiffness of the barrier.
            use_physical_barrier: Whether to use the physical barrier.
        """
    @property
    def barrier(self) -> Barrier:
        """
        Barrier function used to compute the potential.
        """
    @barrier.setter
    def barrier(self, arg1: Barrier) -> None: ...
    @property
    def dhat(self) -> float:
        """
        Barrier activation distance.
        """
    @dhat.setter
    def dhat(self, arg1: typing.SupportsFloat) -> None: ...

class BroadPhase:
    def __init__(self) -> None: ...
    @typing.overload
    def build(
        self,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        inflation_radius: typing.SupportsFloat = 0,
    ) -> None:
        """
        Build the broad phase for static collision detection.

        Parameters:
            vertices: Vertex positions
            edges: Collision mesh edges
            faces: Collision mesh faces
            inflation_radius: Radius of inflation around all elements.
        """
    @typing.overload
    def build(
        self,
        vertices_t0: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        vertices_t1: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        inflation_radius: typing.SupportsFloat = 0,
    ) -> None:
        """
        Build the broad phase for continuous collision detection.

        Parameters:
            vertices_t0: Starting vertices of the vertices.
            vertices_t1: Ending vertices of the vertices.
            edges: Collision mesh edges
            faces: Collision mesh faces
            inflation_radius: Radius of inflation around all elements.
        """
    def clear(self) -> None:
        """
        Clear any built data.
        """
    def detect_collision_candidates(self) -> ...:
        """
        Detect all collision candidates needed for a given dimensional simulation.
        """
    def detect_edge_edge_candidates(self) -> list[...]:
        """
        Find the candidate edge-edge collisions.

        Returns:
            The candidate edge-edge collisions.
        """
    def detect_edge_face_candidates(self) -> list[...]:
        """
        Find the candidate edge-face intersections.

        Returns:
            The candidate edge-face intersections.
        """
    def detect_edge_vertex_candidates(self) -> list[...]:
        """
        Find the candidate edge-vertex collisions.

        Returns:
            The candidate edge-vertex collisions.
        """
    def detect_face_face_candidates(self) -> list[...]:
        """
        Find the candidate face-face collisions.

        Returns:
            The candidate face-face collisions.
        """
    def detect_face_vertex_candidates(self) -> list[...]:
        """
        Find the candidate face-vertex collisions.

        Returns:
            The candidate face-vertex collisions.
        """
    def detect_vertex_vertex_candidates(self) -> list[...]:
        """
        Find the candidate vertex-vertex collisions.

        Returns:
            The candidate vertex-vertex collisions.
        """
    def name(self) -> str:
        """
        Get the name of the broad phase.
        """
    @property
    def can_vertices_collide(self) -> ...:
        """
        Function for determining if two vertices can collide.
        """
    @can_vertices_collide.setter
    def can_vertices_collide(self, arg0: ...) -> None: ...

class BruteForce(BroadPhase):
    def __init__(self) -> None: ...

class Candidates:
    def __getitem__(self, arg0: typing.SupportsInt) -> ...: ...
    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    @typing.overload
    def build(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        inflation_radius: typing.SupportsFloat = 0,
        broad_phase: BroadPhase = None,
    ) -> None:
        """
        Initialize the set of discrete collision detection candidates.

        Parameters:
            mesh: The surface of the collision mesh.
            vertices: Surface vertex positions (rowwise).
            inflation_radius: Amount to inflate the bounding boxes.
            broad_phase: Broad phase to use.
        """
    @typing.overload
    def build(
        self,
        mesh: ...,
        vertices_t0: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        vertices_t1: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        inflation_radius: typing.SupportsFloat = 0,
        broad_phase: BroadPhase = None,
    ) -> None:
        """
        Initialize the set of continuous collision detection candidates.

        Note:
            Assumes the trajectory is linear.

        Parameters:
            mesh: The surface of the collision mesh.
            vertices_t0: Surface vertex starting positions (rowwise).
            vertices_t1: Surface vertex ending positions (rowwise).
            inflation_radius: Amount to inflate the bounding boxes.
            broad_phase: Broad phase to use.
        """
    def clear(self) -> None: ...
    def compute_cfl_stepsize(
        self,
        mesh: ...,
        vertices_t0: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        vertices_t1: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        dhat: typing.SupportsFloat,
        min_distance: typing.SupportsFloat = 0.0,
        broad_phase: BroadPhase = None,
        narrow_phase_ccd: NarrowPhaseCCD = ...,
    ) -> float:
        """
        Computes a CFL-inspired CCD maximum step step size.

        Parameters:
            mesh: The collision mesh.
            vertices_t0: Surface vertex starting positions (rowwise).
            vertices_t1: Surface vertex ending positions (rowwise).
            dhat: Barrier activation distance.
            min_distance: Minimum distance allowable between any two elements.
            broad_phase: Broad phase algorithm to use.
            narrow_phase_ccd: Narrow phase CCD algorithm to use.

        Returns:
            A step-size :math:`\\in [0, 1]` that is collision free.
        """
    def compute_collision_free_stepsize(
        self,
        mesh: ...,
        vertices_t0: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        vertices_t1: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        min_distance: typing.SupportsFloat = 0.0,
        narrow_phase_ccd: NarrowPhaseCCD = ...,
    ) -> float:
        """
        Computes a maximal step size that is collision free using the set of collision candidates.

        Note:
            Assumes the trajectory is linear.

        Parameters:
            mesh: The collision mesh.
            vertices_t0: Surface vertex starting positions (rowwise). Assumed to be intersection free.
            vertices_t1: Surface vertex ending positions (rowwise).
            min_distance: The minimum distance allowable between any two elements.
            narrow_phase_ccd: The narrow phase CCD algorithm to use.

        Returns:
            A step-size :math:`\\in [0, 1]` that is collision free. A value of 1.0 if a full step and 0.0 is no step.
        """
    def compute_noncandidate_conservative_stepsize(
        self,
        mesh: ...,
        displacements: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        dhat: typing.SupportsFloat,
    ) -> float:
        """
        Computes a conservative bound on the largest-feasible step size for surface primitives not in collision.

        Parameters:
            mesh: The collision mesh.
            displacements: Surface vertex displacements (rowwise).
            dhat: Barrier activation distance.

        Returns:
            A step-size :math:`\\in [0, 1]` that is collision free for non-candidate elements.
        """
    def compute_per_vertex_safe_distances(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        inflation_radius: typing.SupportsFloat,
        min_distance: typing.SupportsFloat = 0.0,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the maximum distance each vertex can move independently without colliding with any other element.

        Notes:
            - Caps the value at the inflation radius used to build the candidates.

        Parameters:
            mesh: The collision mesh.
            vertices: Collision mesh vertex positions (rowwise).
            inflation_radius: The inflation radius used to build the candidates.
            min_distance: The minimum allowable distance between any two elements.

        Returns:
            A vector of minimum distances, one for each vertex.
        """
    def edge_edge_to_edge_vertex(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        is_active: typing.Any = None,
    ) -> list[...]:
        """
        Converts edge-edge candidates to edge-vertex candidates.

        Parameters:
            mesh: The collision mesh.
            vertices: Collision mesh vertex positions (rowwise).
            is_active: A function to determine if a candidate is active.
                       If None, uses the default (always true).

        Returns:
            A list of edge-vertex candidates.
        """
    def edge_vertex_to_vertex_vertex(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        is_active: typing.Any = None,
    ) -> list[...]:
        """
        Converts edge-vertex candidates to vertex-vertex candidates.

        Parameters:
            mesh: The collision mesh.
            vertices: Collision mesh vertex positions (rowwise).
            is_active: A function to determine if a candidate is active.
                       If None, uses the default (always true).

        Returns:
            A list of vertex-vertex candidates.
        """
    def empty(self) -> bool: ...
    def face_vertex_to_edge_vertex(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        is_active: typing.Any = None,
    ) -> list[...]:
        """
        Converts face-vertex candidates to edge-vertex candidates.

        Parameters:
            mesh: The collision mesh.
            vertices: Collision mesh vertex positions (rowwise).
            is_active: A function to determine if a candidate is active.
                       If None, uses the default (always true).

        Returns:
            A list of edge-vertex candidates.
        """
    def face_vertex_to_vertex_vertex(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        is_active: typing.Any = None,
    ) -> list[...]:
        """
        Converts face-vertex candidates to vertex-vertex candidates.

        Parameters:
            mesh: The collision mesh.
            vertices: Collision mesh vertex positions (rowwise).
            is_active: A function to determine if a candidate is active.
                       If None, uses the default (always true).

        Returns:
            A list of vertex-vertex candidates.
        """
    def is_step_collision_free(
        self,
        mesh: ...,
        vertices_t0: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        vertices_t1: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        min_distance: typing.SupportsFloat = 0.0,
        narrow_phase_ccd: NarrowPhaseCCD = ...,
    ) -> bool:
        """
        Determine if the step is collision free from the set of candidates.

        Note:
            Assumes the trajectory is linear.

        Parameters:
            mesh: The collision mesh.
            vertices_t0: Surface vertex starting positions (rowwise).
            vertices_t1: Surface vertex ending positions (rowwise).
            min_distance: The minimum distance allowable between any two elements.
            narrow_phase_ccd: The narrow phase CCD algorithm to use.

        Returns:
            True if <b>any</b> collisions occur.
        """
    def save_obj(
        self,
        filename: str,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
    ) -> bool: ...
    @property
    def ee_candidates(self) -> list[...]: ...
    @ee_candidates.setter
    def ee_candidates(self, arg0: collections.abc.Sequence[...]) -> None: ...
    @property
    def ev_candidates(self) -> list[...]: ...
    @ev_candidates.setter
    def ev_candidates(self, arg0: collections.abc.Sequence[...]) -> None: ...
    @property
    def fv_candidates(self) -> list[...]: ...
    @fv_candidates.setter
    def fv_candidates(self, arg0: collections.abc.Sequence[...]) -> None: ...
    @property
    def pv_candidates(self) -> list[...]: ...
    @pv_candidates.setter
    def pv_candidates(self, arg0: collections.abc.Sequence[...]) -> None: ...
    @property
    def vv_candidates(self) -> list[...]: ...
    @vv_candidates.setter
    def vv_candidates(self, arg0: collections.abc.Sequence[...]) -> None: ...

class ClampedLogBarrier(Barrier):
    """

    Smoothly clamped log barrier functions from [Li et al. 2020].

    .. math::

        b(d) = -(d-\\hat{d})^2\\ln\\left(\\frac{d}{\\hat{d}}\\right)


    """
    def __init__(self) -> None: ...

class ClampedLogSqBarrier(Barrier):
    """

    Clamped log barrier with a quadratic log term from [Huang et al. 2024].

    .. math::

        b(d) = (d-\\hat{d})^2\\ln^2\\left(\\frac{d}{\\hat{d}}\\right)


    """
    def __init__(self) -> None: ...

class CollisionFilter:
    """

    A composable, type-erased collision filter.

    Wraps any callable ``bool(int, int)`` and supports logical composition
    via ``|`` (union), ``&`` (intersection), and ``~`` (negation) operators.
    The default-constructed filter accepts all pairs.

    Example:
        .. code-block:: python

            patches = CollisionFilter(lambda i, j: patches[i] != patches[j])
            static = make_static_obstacle_filter(n_dynamic)
            active = patches & static
            if active(i, j): ...

    """
    def __and__(self, arg0: CollisionFilter) -> CollisionFilter:
        """
        Intersection: accept only if BOTH filters pass.
        """
    def __call__(self, vi: typing.SupportsInt, vj: typing.SupportsInt) -> bool:
        """
        Test whether two vertices may collide.
        """
    def __iand__(self, arg0: CollisionFilter) -> CollisionFilter: ...
    @typing.overload
    def __init__(self) -> None:
        """
        Default filter: accept all pairs.
        """
    @typing.overload
    def __init__(self, fn: collections.abc.Callable) -> None:
        """
        Construct from a Python callable ``bool(int, int)``. Python-backed filters acquire the GIL on each call and may not be safe or performant for parallel broad-phase use.
        """
    def __invert__(self) -> CollisionFilter:
        """
        Negation: accept only if this filter rejects.
        """
    def __ior__(self, arg0: CollisionFilter) -> CollisionFilter: ...
    def __or__(self, arg0: CollisionFilter) -> CollisionFilter:
        """
        Union: accept if EITHER filter passes.
        """

class CollisionMesh:
    @staticmethod
    def build_from_full_mesh(
        full_rest_positions: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ] = ...,
    ) -> CollisionMesh:
        """
        Helper function that automatically builds include_vertex using construct_is_on_surface.

        Parameters:
            full_rest_positions: The full vertices at rest (#FV × dim).
            edges: The edge matrix of mesh (#E × 2).
            faces: The face matrix of mesh (#F × 3).

        Returns:
            Constructed CollisionMesh.
        """
    @staticmethod
    def construct_faces_to_edges(
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, n]"]:
        """
        Construct a matrix that maps from the faces' edges to rows in the edges matrix.

        Parameters:
            faces: The face matrix of mesh (#F × 3).
            edges: The edge matrix of mesh (#E × 2).

        Returns:
            Matrix that maps from the faces' edges to rows in the edges matrix.
        """
    @staticmethod
    def construct_is_on_surface(
        num_vertices: typing.SupportsInt,
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        codim_vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, 1]"
        ] = ...,
    ) -> list[bool]:
        """
        Construct a vector of bools indicating whether each vertex is on the surface.

        Parameters:
            num_vertices: The number of vertices in the mesh.
            edges: The surface edges of the mesh (#E × 2).
            codim_vertices: The indices of codimensional vertices (#CV x 1).

        Returns:
            A vector of bools indicating whether each vertex is on the surface.
        """
    @typing.overload
    def __init__(
        self,
        rest_positions: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ] = ...,
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ] = ...,
        displacement_map: scipy.sparse.csc_matrix = ...,
    ) -> None:
        """
        Construct a new Collision Mesh object directly from the collision mesh vertices.

        Parameters:
            rest_positions: The vertices of the collision mesh at rest (#V × dim).
            edges: The edges of the collision mesh (#E × 2).
            faces: The faces of the collision mesh (#F × 3).
            displacement_map: The displacement mapping from displacements on the full mesh to the collision mesh.
        """
    @typing.overload
    def __init__(
        self,
        include_vertex: collections.abc.Sequence[bool],
        orient_vertex: collections.abc.Sequence[bool],
        full_rest_positions: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ] = ...,
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ] = ...,
        displacement_map: scipy.sparse.csc_matrix = ...,
    ) -> None:
        """
        Construct a new Collision Mesh object from a full mesh vertices.

        Parameters:
            include_vertex: Vector of bools indicating whether each vertex should be included in the collision mesh.
            orient_vertex: Vector of bools indicating whether each vertex is orientable.
            full_rest_positions: The vertices of the full mesh at rest (#V × dim).
            edges: The edges of the collision mesh indexed into the full mesh vertices (#E × 2).
            faces: The faces of the collision mesh indexed into the full mesh vertices (#F × 3).
            displacement_map: The displacement mapping from displacements on the full mesh to the collision mesh.
        """
    def are_adjacencies_initialized(self) -> bool:
        """
        Determine if the adjacencies have been initialized by calling init_adjacencies().
        """
    def are_area_jacobians_initialized(self) -> bool:
        """
        Determine if the area Jacobians have been initialized by calling init_area_jacobians().
        """
    def displace_vertices(
        self,
        full_displacements: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the vertex positions from vertex displacements on the full mesh.

        Parameters:
            full_displacements: The vertex displacements on the full mesh (#FV × dim).

        Returns:
            The vertex positions of the collision mesh (#V × dim).
        """
    def edge_area(self, ei: typing.SupportsInt) -> float:
        """
        Get the barycentric area of an edge.

        Parameters:
            ei: Edge ID.

        Returns:
            Barycentric area of edge ei.
        """
    def edge_area_gradient(self, ei: typing.SupportsInt) -> scipy.sparse.csc_matrix:
        """
        Get the gradient of the barycentric area of an edge wrt the rest positions of all points.

        Parameters:
            ei: Edge ID.

        Returns:
            Gradient of the barycentric area of edge ei wrt the rest positions of all points.
        """
    def edge_areas(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the barycentric area of the edges.
        """
    def init_adjacencies(self) -> None:
        """
        Initialize vertex-vertex and edge-vertex adjacencies.
        """
    def init_area_jacobians(self) -> None:
        """
        Initialize vertex and edge areas.
        """
    def is_vertex_on_boundary(self, vi: typing.SupportsInt) -> bool:
        """
        Is a vertex on the boundary of the collision mesh?

        Parameters:
            vi: Vertex ID.

        Returns:
            True if the vertex is on the boundary of the collision mesh.
        """
    def map_displacements(
        self,
        full_displacements: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Map vertex displacements on the full mesh to vertex displacements on the collision mesh.

        Parameters:
            full_displacements: The vertex displacements on the full mesh (#FV × dim).

        Returns:
            The vertex displacements on the collision mesh (#V × dim).
        """
    @typing.overload
    def to_full_dof(
        self, x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Map a vector quantity on the collision mesh to the full mesh.

        This is useful for mapping gradients from the collision mesh to the full mesh (i.e., applies the chain-rule).

        Parameters:
            x: Vector quantity on the collision mesh with size equal to ndof().

        Returns:
            Vector quantity on the full mesh with size equal to full_ndof().
        """
    @typing.overload
    def to_full_dof(self, X: scipy.sparse.csc_matrix) -> scipy.sparse.csc_matrix:
        """
        Map a matrix quantity on the collision mesh to the full mesh.

        This is useful for mapping Hessians from the collision mesh to the full mesh (i.e., applies the chain-rule).

        Parameters:
            X: Matrix quantity on the collision mesh with size equal to ndof() × ndof().

        Returns:
            Matrix quantity on the full mesh with size equal to full_ndof() × full_ndof().
        """
    @typing.overload
    def to_full_vertex_id(self, id: typing.SupportsInt) -> int:
        """
        Map a vertex ID to the corresponding vertex ID in the full mesh.

        Parameters:
            id: Vertex ID in the collision mesh.

        Returns:
            Vertex ID in the full mesh.
        """
    @typing.overload
    def to_full_vertex_id(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"]:
        """
        Get the complete mapping of vertex IDs to their corresponding vertex IDs in the full mesh.

        Returns:
            Vector of size num_vertices() where each entry is the full vertex ID corresponding to the collision mesh vertex ID.
        """
    def vertex_area(self, vi: typing.SupportsInt) -> float:
        """
        Get the barycentric area of a vertex.

        Parameters:
            vi: Vertex ID.

        Returns:
            Barycentric area of vertex vi.
        """
    def vertex_area_gradient(self, vi: typing.SupportsInt) -> scipy.sparse.csc_matrix:
        """
        Get the gradient of the barycentric area of a vertex wrt the rest positions of all points.

        Parameters:
            vi: Vertex ID.

        Returns:
            Gradient of the barycentric area of vertex vi wrt the rest positions of all points.
        """
    def vertices(
        self,
        full_positions: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the vertex positions from the positions of the full mesh.

        Parameters:
            full_positions: The vertex positions of the full mesh (#FV × dim).

        Returns:
            The vertex positions of the collision mesh (#V × dim).
        """
    @property
    def can_collide(self) -> CollisionFilter:
        """
        A function that takes two vertex IDs and returns true if the vertices (and faces or edges containing the vertices) can collide.

        By default all primitives can collide with all other primitives.
        """
    @can_collide.setter
    def can_collide(self, arg1: typing.Any) -> None: ...
    @property
    def codim_edges(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"]:
        """
        Get the indices of codimensional edges of the collision mesh (#CE x 1).
        """
    @property
    def codim_vertices(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"]:
        """
        Get the indices of codimensional vertices of the collision mesh (#CV x 1).
        """
    @property
    def dim(self) -> int:
        """
        Get the dimension of the mesh.
        """
    @property
    def edge_vertex_adjacencies(self) -> list[list[int]]:
        """
        Get the edge-vertex adjacency matrix.
        """
    @property
    def edges(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, n]"]:
        """
        Get the edges of the collision mesh (#E × 2).
        """
    @property
    def faces(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, n]"]:
        """
        Get the faces of the collision mesh (#F × 3).
        """
    @property
    def faces_to_edges(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, n]"]:
        """
        Get the mapping from faces to edges of the collision mesh (#F × 3).
        """
    @property
    def full_ndof(self) -> int:
        """
        Get the number of degrees of freedom in the full mesh.
        """
    @property
    def full_num_vertices(self) -> int:
        """
        Get the number of vertices in the full mesh.
        """
    @property
    def ndof(self) -> int:
        """
        Get the number of degrees of freedom in the collision mesh.
        """
    @property
    def num_codim_edges(self) -> int:
        """
        Get the number of codimensional edges in the collision mesh.
        """
    @property
    def num_codim_vertices(self) -> int:
        """
        Get the number of codimensional vertices in the collision mesh.
        """
    @property
    def num_edges(self) -> int:
        """
        Get the number of edges in the collision mesh.
        """
    @property
    def num_faces(self) -> int:
        """
        Get the number of faces in the collision mesh.
        """
    @property
    def num_vertices(self) -> int:
        """
        Get the number of vertices in the collision mesh.
        """
    @property
    def planes(self) -> list[Hyperplane]:
        """
        A vector of planes in the collision mesh.

        Each plane is represented as a `Hyperplane` object (wrapping `Eigen::Hyperplane<double, 3>`).
        In Python, a `Hyperplane` can be constructed from either:

        * a normal and a point on the plane: ``Hyperplane(normal, point)``, or
        * a normal and an offset: ``Hyperplane(normal, offset)``,

        where ``normal`` and ``point`` are 3D vectors and ``offset`` is a scalar.
        """
    @planes.setter
    def planes(self, arg0: collections.abc.Sequence[Hyperplane]) -> None: ...
    @property
    def rest_positions(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Get the vertices of the collision mesh at rest (#V × dim).
        """
    @property
    def vertex_areas(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the barycentric area of the vertices.
        """
    @property
    def vertex_edge_adjacencies(self) -> list[list[int]]:
        """
        Get the vertex-edge adjacency matrix.
        """
    @property
    def vertex_vertex_adjacencies(self) -> list[list[int]]:
        """
        Get the vertex-vertex adjacency matrix.
        """

class CollisionStencil:
    @staticmethod
    def contract_distance_vector_jacobian(
        coeffs: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        dim: typing.SupportsInt,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute p^T (dt/dx) efficiently as sum(c_i * p_i) (Eqs. 13-14).

        Given p = [p_0, p_1, ..., p_n]^T where p_i are dim-dimensional,
        this computes p^T (dt/dx) = sum(c_i * p_i) which is a
        dim-dimensional vector.

        Parameters:
            coeffs: The coefficients c_i.
            p: A vector of size ndof (the direction for the quadratic form).
            dim: The spatial dimension (2 or 3).

        Returns:
            p^T (dt/dx) as a dim-dimensional vector.
        """
    @staticmethod
    def diag_distance_vector_outer(
        coeffs: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        dim: typing.SupportsInt,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute diag((dt/dx)(dt/dx)^T) efficiently (Eq. 11).

        Result is [c_0^2, c_0^2, c_0^2, c_1^2, ...] (each c_i^2 repeated
        dim times).

        Parameters:
            coeffs: The coefficients c_i (from compute_coefficients).
            dim: The spatial dimension (2 or 3).

        Returns:
            The diagonal of (dt/dx)(dt/dx)^T as a vector of size ndof.
        """
    @staticmethod
    def diag_distance_vector_t_outer(
        coeffs: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        distance_vector: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, 1]"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute diag((dt/dx * t)(dt/dx * t)^T) efficiently (Eq. 12).

        Result is element-wise square of [c_0*t^T, c_1*t^T, ..., c_n*t^T].

        Parameters:
            coeffs: The coefficients c_i.
            distance_vector: The distance vector t.

        Returns:
            The diagonal of (dt/dx*t)(dt/dx*t)^T as a vector of size ndof.
        """
    def ccd(
        self,
        vertices_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        vertices_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        min_distance: typing.SupportsFloat = 0.0,
        tmax: typing.SupportsFloat = 1.0,
        narrow_phase_ccd: NarrowPhaseCCD = ...,
    ) -> tuple[bool, float]:
        """
        Perform narrow-phase CCD on the candidate.

        Parameters:
            vertices_t0: Stencil vertices at the start of the time step.
            vertices_t1: Stencil vertices at the end of the time step.
            min_distance: Minimum separation distance between primitives.
            tmax: Maximum time (normalized) to look for collisions. Should be in [0, 1].
            narrow_phase_ccd: The narrow phase CCD algorithm to use.

        Returns:
            Tuple of:
            If the candidate had a collision over the time interval.
            Computed time of impact (normalized).
        """
    @typing.overload
    def compute_coefficients(
        self,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the distance Hessian of the stencil w.r.t. the stencil's vertex positions.

        Parameters:
            vertices: Collision mesh vertices.
            edges: Collision mesh edges.
            faces: Collision mesh faces.

        Returns:
            Distance Hessian of the stencil w.r.t. the stencil's vertex positions.
        """
    @typing.overload
    def compute_coefficients(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the distance coefficients of the stencil w.r.t. the stencil's vertex positions.

        Note:
            positions can be computed as stencil.dof(vertices, edges, faces)

        Parameters:
            positions: Stencil's vertex positions.

        Returns:
            Distance of the stencil.
        """
    @typing.overload
    def compute_distance(
        self,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
    ) -> float:
        """
        Compute the distance of the stencil.

        Parameters:
            vertices: Collision mesh vertices.
            edges: Collision mesh edges.
            faces: Collision mesh faces.

        Returns:
            Distance of the stencil.
        """
    @typing.overload
    def compute_distance(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> float:
        """
        Compute the distance of the stencil.

        Note:
            positions can be computed as stencil.dof(vertices, edges, faces)

        Parameters:
            positions: Stencil's vertex positions.

        Returns:
            Distance of the stencil.
        """
    @typing.overload
    def compute_distance_gradient(
        self,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the distance gradient of the stencil w.r.t. the stencil's vertex positions.

        Parameters:
            vertices: Collision mesh vertices.
            edges: Collision mesh edges.
            faces: Collision mesh faces.

        Returns:
            Distance gradient of the stencil w.r.t. the stencil's vertex positions.
        """
    @typing.overload
    def compute_distance_gradient(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the distance gradient of the stencil w.r.t. the stencil's vertex positions.

        Note:
            positions can be computed as stencil.dof(vertices, edges, faces)

        Parameters:
            positions: Stencil's vertex positions.

        Returns:
            Distance gradient of the stencil w.r.t. the stencil's vertex positions.
        """
    @typing.overload
    def compute_distance_hessian(
        self,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the distance Hessian of the stencil w.r.t. the stencil's vertex positions.

        Parameters:
            vertices: Collision mesh vertices.
            edges: Collision mesh edges.
            faces: Collision mesh faces.

        Returns:
            Distance Hessian of the stencil w.r.t. the stencil's vertex positions.
        """
    @typing.overload
    def compute_distance_hessian(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the distance Hessian of the stencil w.r.t. the stencil's vertex positions.

        Note:
            positions can be computed as stencil.dof(vertices, edges, faces)

        Parameters:
            positions: Stencil's vertex positions.

        Returns:
            Distance Hessian of the stencil w.r.t. the stencil's vertex positions.
        """
    @typing.overload
    def compute_distance_vector(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the distance vector of the stencil: t = sum(c_i * x_i).

        The distance vector is the vector between the closest points on the
        collision primitives. Its squared norm equals the squared distance.

        Note:
            positions can be computed as stencil.dof(vertices, edges, faces)

        Parameters:
            positions: Stencil's vertex positions.

        Returns:
            The distance vector (dim-dimensional, i.e., 2D or 3D).
        """
    @typing.overload
    def compute_distance_vector(
        self,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the distance vector of the stencil.

        Parameters:
            vertices: Collision mesh vertices.
            edges: Collision mesh edges.
            faces: Collision mesh faces.

        Returns:
            The distance vector (dim-dimensional).
        """
    def compute_distance_vector_jacobian(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the Jacobian of the distance vector w.r.t. positions.

        J = [c_0 I, c_1 I, ..., c_n I]^T where I is the dim x dim identity.

        Note:
            positions can be computed as stencil.dof(vertices, edges, faces)

        Parameters:
            positions: Stencil's vertex positions.

        Returns:
            The Jacobian dt/dx as a matrix of shape (ndof, dim).
        """
    def compute_distance_vector_with_coefficients(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> tuple[
        typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ]:
        """
        Compute the distance vector and the coefficients together.

        Note:
            positions can be computed as stencil.dof(vertices, edges, faces)

        Parameters:
            positions: Stencil's vertex positions.

        Returns:
            Tuple of:
            The distance vector (dim-dimensional).
            The computed coefficients c_i.
        """
    def dim(self, ndof: typing.SupportsInt) -> int:
        """
        Get the dimension of the collision stencil.

        Parameters:
            ndof: Number of degrees of freedom in the stencil.

        Returns:
            The dimension of the collision stencil.
        """
    def dof(
        self,
        X: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Select this stencil's DOF from the full matrix of DOF.

        T Type of the DOF

        Parameters:
            X: Full matrix of DOF (rowwise).
            edges: Collision mesh edges
            faces: Collision mesh faces

        Returns:
            This stencil's DOF.
        """
    def num_vertices(self) -> int:
        """
        Get the number of vertices in the collision stencil.
        """
    def print_ccd_query(
        self,
        vertices_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        vertices_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> None:
        """
        Print the CCD query to cout.

        Parameters:
                            vertices_t0: Stencil vertices at the start of the time step.
            vertices_t1: Stencil vertices at the end of the time step.
        """
    def vertex_ids(
        self,
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[list[int], "FixedSize(4)"]:
        """
        Get the vertex IDs of the collision stencil.

        Parameters:
            edges: Collision mesh edges
            faces: Collision mesh faces

        Returns:
            The vertex IDs of the collision stencil. Size is always 4, but elements i > num_vertices() are -1.
        """
    def vertices(
        self,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[
        list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]],
        "FixedSize(4)",
    ]:
        """
        Get the vertex attributes of the collision stencil.

        T Type of the attributes

        Parameters:
            vertices: Vertex attributes
            edges: Collision mesh edges
            faces: Collision mesh faces

        Returns:
            The vertex positions of the collision stencil. Size is always 4, but elements i > num_vertices() are NaN.
        """

class CubicBarrier(Barrier):
    """

    Cubic barrier function from [Ando 2024].

    .. math::

        b(d) = -\\frac{2}{3\\hat{d}} (d - \\hat{d})^3


    """
    def __init__(self) -> None: ...

class Edge2Point2Collision(SmoothCollision2):
    def name(self) -> str:
        """
        Get the type name of collision
        """
    def num_vertices(self) -> int:
        """
        Get the number of vertices
        """

class EdgeEdgeCandidate(CollisionStencil):
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: EdgeEdgeCandidate) -> bool: ...
    @typing.overload
    def __init__(
        self, edge0_id: typing.SupportsInt, edge1_id: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, edge_ids: tuple[typing.SupportsInt, typing.SupportsInt]
    ) -> None: ...
    def __lt__(self, other: EdgeEdgeCandidate) -> bool:
        """
        Compare EdgeEdgeCandidates for sorting.
        """
    def __ne__(self, other: EdgeEdgeCandidate) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def known_dtype(self) -> ...: ...
    @property
    def edge0_id(self) -> int:
        """
        ID of the first edge.
        """
    @edge0_id.setter
    def edge0_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def edge1_id(self) -> int:
        """
        ID of the second edge.
        """
    @edge1_id.setter
    def edge1_id(self, arg0: typing.SupportsInt) -> None: ...

class EdgeEdgeDistanceType:
    """
    Members:

      EA0_EB0 : edges are closest at vertex 0 of edge A and 0 of edge B

      EA0_EB1 : edges are closest at vertex 0 of edge A and 1 of edge B

      EA1_EB0 : edges are closest at vertex 1 of edge A and 0 of edge B

      EA1_EB1 : edges are closest at vertex 1 of edge A and 1 of edge B

      EA_EB0 : edges are closest at the interior of edge A and vertex 0 of edge B

      EA_EB1 : edges are closest at the interior of edge A and vertex 1 of edge B

      EA0_EB : edges are closest at vertex 0 of edge A and the interior of edge B

      EA1_EB : edges are closest at vertex 1 of edge A and the interior of edge B

      EA_EB : edges are closest at an interior point of edge A and B

      AUTO : automatically determine the closest point
    """

    AUTO: typing.ClassVar[
        EdgeEdgeDistanceType
    ]  # value = <EdgeEdgeDistanceType.AUTO: 9>
    EA0_EB: typing.ClassVar[
        EdgeEdgeDistanceType
    ]  # value = <EdgeEdgeDistanceType.EA0_EB: 6>
    EA0_EB0: typing.ClassVar[
        EdgeEdgeDistanceType
    ]  # value = <EdgeEdgeDistanceType.EA0_EB0: 0>
    EA0_EB1: typing.ClassVar[
        EdgeEdgeDistanceType
    ]  # value = <EdgeEdgeDistanceType.EA0_EB1: 1>
    EA1_EB: typing.ClassVar[
        EdgeEdgeDistanceType
    ]  # value = <EdgeEdgeDistanceType.EA1_EB: 7>
    EA1_EB0: typing.ClassVar[
        EdgeEdgeDistanceType
    ]  # value = <EdgeEdgeDistanceType.EA1_EB0: 2>
    EA1_EB1: typing.ClassVar[
        EdgeEdgeDistanceType
    ]  # value = <EdgeEdgeDistanceType.EA1_EB1: 3>
    EA_EB: typing.ClassVar[
        EdgeEdgeDistanceType
    ]  # value = <EdgeEdgeDistanceType.EA_EB: 8>
    EA_EB0: typing.ClassVar[
        EdgeEdgeDistanceType
    ]  # value = <EdgeEdgeDistanceType.EA_EB0: 4>
    EA_EB1: typing.ClassVar[
        EdgeEdgeDistanceType
    ]  # value = <EdgeEdgeDistanceType.EA_EB1: 5>
    __members__: typing.ClassVar[
        dict[str, EdgeEdgeDistanceType]
    ]  # value = {'EA0_EB0': <EdgeEdgeDistanceType.EA0_EB0: 0>, 'EA0_EB1': <EdgeEdgeDistanceType.EA0_EB1: 1>, 'EA1_EB0': <EdgeEdgeDistanceType.EA1_EB0: 2>, 'EA1_EB1': <EdgeEdgeDistanceType.EA1_EB1: 3>, 'EA_EB0': <EdgeEdgeDistanceType.EA_EB0: 4>, 'EA_EB1': <EdgeEdgeDistanceType.EA_EB1: 5>, 'EA0_EB': <EdgeEdgeDistanceType.EA0_EB: 6>, 'EA1_EB': <EdgeEdgeDistanceType.EA1_EB: 7>, 'EA_EB': <EdgeEdgeDistanceType.EA_EB: 8>, 'AUTO': <EdgeEdgeDistanceType.AUTO: 9>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class EdgeEdgeNormalCollision(EdgeEdgeCandidate, NormalCollision):
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: EdgeEdgeNormalCollision) -> bool: ...
    @typing.overload
    def __init__(
        self,
        edge0_id: typing.SupportsInt,
        edge1_id: typing.SupportsInt,
        eps_x: typing.SupportsFloat,
        dtype: EdgeEdgeDistanceType = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        candidate: EdgeEdgeCandidate,
        eps_x: typing.SupportsFloat,
        dtype: EdgeEdgeDistanceType = ...,
    ) -> None: ...
    def __lt__(self, other: EdgeEdgeNormalCollision) -> bool: ...
    def __ne__(self, other: EdgeEdgeNormalCollision) -> bool: ...
    @property
    def dtype(self) -> EdgeEdgeDistanceType:
        """
        Cached distance type.

        Some EE collisions are mollified EV or VV collisions.
        """
    @dtype.setter
    def dtype(self, arg0: EdgeEdgeDistanceType) -> None: ...
    @property
    def eps_x(self) -> float:
        """
        Mollifier activation threshold.
        """
    @eps_x.setter
    def eps_x(self, arg0: typing.SupportsFloat) -> None: ...

class EdgeEdgeTangentialCollision(EdgeEdgeCandidate, TangentialCollision):
    @typing.overload
    def __init__(self, collision: EdgeEdgeNormalCollision) -> None: ...
    @typing.overload
    def __init__(
        self,
        collision: EdgeEdgeNormalCollision,
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        normal_potential: ...,
    ) -> None: ...

class EdgeFaceCandidate:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: EdgeFaceCandidate) -> bool: ...
    @typing.overload
    def __init__(
        self, edge_id: typing.SupportsInt, face_id: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, edge_and_face_id: tuple[typing.SupportsInt, typing.SupportsInt]
    ) -> None: ...
    def __lt__(self, other: EdgeFaceCandidate) -> bool:
        """
        Compare EdgeFaceCandidate for sorting.
        """
    def __ne__(self, other: EdgeFaceCandidate) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def edge_id(self) -> int:
        """
        ID of the edge
        """
    @edge_id.setter
    def edge_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def face_id(self) -> int:
        """
        ID of the face
        """
    @face_id.setter
    def face_id(self, arg0: typing.SupportsInt) -> None: ...

class EdgeVertexCandidate(CollisionStencil):
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: EdgeVertexCandidate) -> bool: ...
    @typing.overload
    def __init__(
        self, edge_id: typing.SupportsInt, vertex_id: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, edge_and_vertex_id: tuple[typing.SupportsInt, typing.SupportsInt]
    ) -> None: ...
    def __lt__(self, other: EdgeVertexCandidate) -> bool:
        """
        Compare EdgeVertexCandidates for sorting.
        """
    def __ne__(self, other: EdgeVertexCandidate) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def known_dtype(self) -> ...: ...
    @property
    def edge_id(self) -> int:
        """
        ID of the edge
        """
    @edge_id.setter
    def edge_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def vertex_id(self) -> int:
        """
        ID of the vertex
        """
    @vertex_id.setter
    def vertex_id(self, arg0: typing.SupportsInt) -> None: ...

class EdgeVertexNormalCollision(EdgeVertexCandidate, NormalCollision):
    @typing.overload
    def __init__(
        self, edge_id: typing.SupportsInt, vertex_id: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(self, candidate: EdgeVertexCandidate) -> None: ...

class EdgeVertexTangentialCollision(EdgeVertexCandidate, TangentialCollision):
    @typing.overload
    def __init__(self, collision: EdgeVertexNormalCollision) -> None: ...
    @typing.overload
    def __init__(
        self,
        collision: EdgeVertexNormalCollision,
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        normal_potential: ...,
    ) -> None: ...

class FaceFaceCandidate:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: FaceFaceCandidate) -> bool: ...
    @typing.overload
    def __init__(
        self, face0_id: typing.SupportsInt, face1_id: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, face_ids: tuple[typing.SupportsInt, typing.SupportsInt]
    ) -> None: ...
    def __lt__(self, other: FaceFaceCandidate) -> bool:
        """
        Compare FaceFaceCandidate for sorting.
        """
    def __ne__(self, other: FaceFaceCandidate) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def face0_id(self) -> int:
        """
        ID of the first face.
        """
    @face0_id.setter
    def face0_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def face1_id(self) -> int:
        """
        ID of the second face.
        """
    @face1_id.setter
    def face1_id(self, arg0: typing.SupportsInt) -> None: ...

class FaceVertexCandidate(CollisionStencil):
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: FaceVertexCandidate) -> bool: ...
    @typing.overload
    def __init__(
        self, face_id: typing.SupportsInt, vertex_id: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, face_and_vertex_id: tuple[typing.SupportsInt, typing.SupportsInt]
    ) -> None: ...
    def __lt__(self, other: FaceVertexCandidate) -> bool:
        """
        Compare FaceVertexCandidate for sorting.
        """
    def __ne__(self, other: FaceVertexCandidate) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def known_dtype(self) -> ...: ...
    @property
    def face_id(self) -> int:
        """
        ID of the face
        """
    @face_id.setter
    def face_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def vertex_id(self) -> int:
        """
        ID of the vertex
        """
    @vertex_id.setter
    def vertex_id(self, arg0: typing.SupportsInt) -> None: ...

class FaceVertexNormalCollision(FaceVertexCandidate, NormalCollision):
    @typing.overload
    def __init__(
        self, face_id: typing.SupportsInt, vertex_id: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(self, candidate: FaceVertexCandidate) -> None: ...

class FaceVertexTangentialCollision(FaceVertexCandidate, TangentialCollision):
    @typing.overload
    def __init__(self, collision: FaceVertexNormalCollision) -> None: ...
    @typing.overload
    def __init__(
        self,
        collision: FaceVertexNormalCollision,
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        normal_potential: ...,
    ) -> None: ...

class FrictionPotential(TangentialPotential):
    def __init__(self, eps_v: typing.SupportsFloat) -> None:
        """
        Construct a friction potential.

        Parameters:
            eps_v: The smooth friction mollifier parameter :math:`\\\\epsilon_{v}`.
        """
    @property
    def eps_v(self) -> float:
        """
        The smooth friction mollifier parameter :math:`\\epsilon_{v}`.
        """
    @eps_v.setter
    def eps_v(self, arg1: typing.SupportsFloat) -> None: ...

class HashGrid(BroadPhase):
    def __init__(self) -> None: ...
    @property
    def cell_size(self) -> float: ...
    @property
    def domain_max(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]: ...
    @property
    def domain_min(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]: ...
    @property
    def grid_size(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[3, 1]"]: ...

class HashItem:
    def __init__(self, key: typing.SupportsInt, id: typing.SupportsInt) -> None:
        """
        Construct a hash item as a (key, value) pair.
        """
    def __lt__(self, other: HashItem) -> bool:
        """
        Compare HashItems by their keys for sorting.
        """
    @property
    def id(self) -> int:
        """
        The value of the item.
        """
    @id.setter
    def id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def key(self) -> int:
        """
        The key of the item.
        """
    @key.setter
    def key(self, arg0: typing.SupportsInt) -> None: ...

class Hyperplane:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
        arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    ) -> None: ...
    def normal(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]: ...
    def offset(self) -> float: ...
    def origin(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]: ...

class IntervalNonlinearTrajectory(NonlinearTrajectory):
    @typing.overload
    def __call__(
        self, t: typing.SupportsFloat
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the point's position at time t
        """
    @typing.overload
    def __call__(self, t: ...) -> typing.Annotated[numpy.typing.NDArray[...], "[m, 1]"]:
        """
        Compute the point's position over a time interval t
        """
    def __init__(self) -> None: ...
    def max_distance_from_linear(
        self, t0: typing.SupportsFloat, t1: typing.SupportsFloat
    ) -> float:
        """
        Compute the maximum distance from the nonlinear trajectory to a linearized trajectory

        Note:
            This uses interval arithmetic to compute the maximum distance. If you know a tighter bound on the maximum distance, it is recommended to override this function.

        Parameters:
            t0: Start time of the trajectory
            t1: End time of the trajectory
        """

class LBVH(BroadPhase):
    def __init__(self) -> None: ...
    @property
    def edge_nodes(self) -> list[LBVH_Node]: ...
    @property
    def face_nodes(self) -> list[LBVH_Node]: ...
    @property
    def vertex_nodes(self) -> list[LBVH_Node]: ...

class LBVH_Node:
    def __str__(self) -> str: ...
    def intersects(self, arg0: LBVH_Node) -> bool: ...
    @property
    def aabb_max(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"]: ...
    @property
    def aabb_min(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"]: ...
    @property
    def is_inner(self) -> bool: ...
    @property
    def is_leaf(self) -> bool: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def left(self) -> int: ...
    @property
    def primitive_id(self) -> int: ...
    @property
    def right(self) -> int: ...

class LoggerLevel:
    """
    Enumeration of log levels

    Members:

      trace : Trace level

      debug : Debug level

      info : Info level

      warn : Warning level

      error : Error level

      critical : Critical level

      off : Off level
    """

    __members__: typing.ClassVar[
        dict[str, LoggerLevel]
    ]  # value = {'trace': <LoggerLevel.trace: 0>, 'debug': <LoggerLevel.debug: 1>, 'info': <LoggerLevel.info: 2>, 'warn': <LoggerLevel.warn: 3>, 'error': <LoggerLevel.error: 4>, 'critical': <LoggerLevel.critical: 5>, 'off': <LoggerLevel.off: 6>}
    critical: typing.ClassVar[LoggerLevel]  # value = <LoggerLevel.critical: 5>
    debug: typing.ClassVar[LoggerLevel]  # value = <LoggerLevel.debug: 1>
    error: typing.ClassVar[LoggerLevel]  # value = <LoggerLevel.error: 4>
    info: typing.ClassVar[LoggerLevel]  # value = <LoggerLevel.info: 2>
    off: typing.ClassVar[LoggerLevel]  # value = <LoggerLevel.off: 6>
    trace: typing.ClassVar[LoggerLevel]  # value = <LoggerLevel.trace: 0>
    warn: typing.ClassVar[LoggerLevel]  # value = <LoggerLevel.warn: 3>
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class NarrowPhaseCCD:
    def __init__(self) -> None: ...
    def edge_edge_ccd(
        self,
        ea0_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        ea1_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        eb0_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        eb1_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        ea0_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        ea1_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        eb0_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        eb1_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        min_distance: typing.SupportsFloat = 0.0,
        tmax: typing.SupportsFloat = 1.0,
    ) -> tuple[bool, float]: ...
    def point_edge_ccd(
        self,
        p_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        e0_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        e1_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        p_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        e0_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        e1_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        min_distance: typing.SupportsFloat = 0.0,
        tmax: typing.SupportsFloat = 1.0,
    ) -> tuple[bool, float]: ...
    def point_point_ccd(
        self,
        p0_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        p1_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        p0_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        p1_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        min_distance: typing.SupportsFloat = 0.0,
        tmax: typing.SupportsFloat = 1.0,
    ) -> tuple[bool, float]: ...
    def point_triangle_ccd(
        self,
        p_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        t0_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        t1_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        t2_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        p_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        t0_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        t1_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        t2_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
        min_distance: typing.SupportsFloat = 0.0,
        tmax: typing.SupportsFloat = 1.0,
    ) -> tuple[bool, float]: ...

class NonlinearCCD:
    @staticmethod
    def conservative_piecewise_linear_ccd(
        distance: collections.abc.Callable[[typing.SupportsFloat], float],
        max_distance_from_linear: collections.abc.Callable[
            [typing.SupportsFloat, typing.SupportsFloat], float
        ],
        linear_ccd: collections.abc.Callable[
            [
                typing.SupportsFloat,
                typing.SupportsFloat,
                typing.SupportsFloat,
                bool,
                typing.SupportsFloat,
            ],
            bool,
        ],
        min_distance: typing.SupportsFloat = 0,
        tmax: typing.SupportsFloat = 1.0,
        conservative_rescaling: typing.SupportsFloat = 0.8,
    ) -> tuple[bool, float]:
        """
        Perform conservative piecewise linear CCD of a nonlinear trajectories.

        Parameters:
            distance: Return the distance for a given time in [0, 1].
            max_distance_from_linear: Return the maximum distance from the linearized trajectory for a given time interval.
            linear_ccd: Perform linear CCD on a given time interval.
            tmax: Maximum time to check for collision.
            min_distance: Minimum separation distance between the objects.
            conservative_rescaling: Conservative rescaling of the time of impact.

        Returns:
            Tuple of:

            Output time of impact.
        """
    def __init__(
        self,
        tolerance: typing.SupportsFloat = 1e-06,
        max_iterations: typing.SupportsInt = 10000000,
        conservative_rescaling: typing.SupportsFloat = 0.8,
    ) -> None: ...
    def edge_edge_ccd(
        self,
        ea0: NonlinearTrajectory,
        ea1: NonlinearTrajectory,
        eb0: NonlinearTrajectory,
        eb1: NonlinearTrajectory,
        min_distance: typing.SupportsFloat = 0,
        tmax: typing.SupportsFloat = 1.0,
    ) -> tuple[bool, float]:
        """
        Perform nonlinear CCD between two linear edges moving along nonlinear trajectories.

        Parameters:
            ea0: First edge's first endpoint's trajectory
            ea1: First edge's second endpoint's trajectory
            eb0: Second edge's first endpoint's trajectory
            eb1: Second edge's second endpoint's trajectory
            min_distance: Minimum separation distance between the two edges
            tmax: Maximum time to check for collision

        Returns:
            Tuple of:
            True if the two edges collide, false otherwise.
            Output time of impact
        """
    def point_edge_ccd(
        self,
        p: NonlinearTrajectory,
        e0: NonlinearTrajectory,
        e1: NonlinearTrajectory,
        min_distance: typing.SupportsFloat = 0,
        tmax: typing.SupportsFloat = 1.0,
    ) -> tuple[bool, float]:
        """
        Perform nonlinear CCD between a point and a linear edge moving along nonlinear trajectories.

        Parameters:
            p: Point's trajectory
            e0: Edge's first endpoint's trajectory
            e1: Edge's second endpoint's trajectory
            min_distance: Minimum separation distance between the point and the edge
            tmax: Maximum time to check for collision

        Returns:
            Tuple of:
            True if the point and edge collide, false otherwise.
            Output time of impact
        """
    def point_point_ccd(
        self,
        p0: NonlinearTrajectory,
        p1: NonlinearTrajectory,
        min_distance: typing.SupportsFloat = 0,
        tmax: typing.SupportsFloat = 1.0,
    ) -> tuple[bool, float]:
        """
        Perform nonlinear CCD between two points moving along nonlinear trajectories.

        Parameters:
            p0: First point's trajectory
            p1: Second point's trajectory
            min_distance: Minimum separation distance between the two points
            tmax: Maximum time to check for collision

        Returns:
            Tuple of:
            True if the two points collide, false otherwise.
            Output time of impact
        """
    def point_triangle_ccd(
        self,
        p: NonlinearTrajectory,
        t0: NonlinearTrajectory,
        t1: NonlinearTrajectory,
        t2: NonlinearTrajectory,
        min_distance: typing.SupportsFloat = 0,
        tmax: typing.SupportsFloat = 1.0,
    ) -> tuple[bool, float]:
        """
        Perform nonlinear CCD between a point and a linear triangle moving along nonlinear trajectories.

        Parameters:
            p: Point's trajectory
            t0: Triangle's first vertex's trajectory
            t1: Triangle's second vertex's trajectory
            t2: Triangle's third vertex's trajectory
            min_distance: Minimum separation distance between the two edges
            tmax: Maximum time to check for collision

        Returns:
            Tuple of:
            True if the point and triangle collide, false otherwise.
            Output time of impact
        """
    @property
    def conservative_rescaling(self) -> float:
        """
        Conservative rescaling of the time of impact.
        """
    @conservative_rescaling.setter
    def conservative_rescaling(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def max_iterations(self) -> int:
        """
        Maximum number of iterations.
        """
    @max_iterations.setter
    def max_iterations(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def tolerance(self) -> float:
        """
        Solver tolerance.
        """
    @tolerance.setter
    def tolerance(self, arg0: typing.SupportsFloat) -> None: ...

class NonlinearTrajectory:
    def __call__(
        self, t: typing.SupportsFloat
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the point's position at time t
        """
    def __init__(self) -> None: ...
    def max_distance_from_linear(
        self, t0: typing.SupportsFloat, t1: typing.SupportsFloat
    ) -> float:
        """
        Compute the maximum distance from the nonlinear trajectory to a linearized trajectory

        Parameters:
            t0: Start time of the trajectory
            t1: End time of the trajectory
        """

class NormalAdhesionPotential(NormalPotential):
    def __init__(
        self,
        dhat_p: typing.SupportsFloat,
        dhat_a: typing.SupportsFloat,
        Y: typing.SupportsFloat,
        eps_c: typing.SupportsFloat,
    ) -> None: ...
    @property
    def Y(self) -> float:
        """
        The Young's modulus (:math:`Y`)).
        """
    @Y.setter
    def Y(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def dhat_a(self) -> float:
        """
        The adhesion activation distance (:math:`\\hat{d}_{a}`).
        """
    @dhat_a.setter
    def dhat_a(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def dhat_p(self) -> float:
        """
        The distance of largest adhesion force (:math:`\\hat{d}_{p}`) (:math:`0 < \\hat{d}_{p} < \\hat{d}_{a}`).
        """
    @dhat_p.setter
    def dhat_p(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def eps_c(self) -> float:
        """
        The critical strain (:math:`\\varepsilon_{c}`)).
        """
    @eps_c.setter
    def eps_c(self, arg0: typing.SupportsFloat) -> None: ...

class NormalCollision(CollisionStencil):
    def is_mollified(self) -> bool:
        """
        Does the distance potentially have to be mollified?
        """
    @typing.overload
    def mollifier(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> float:
        """
        Compute the mollifier for the distance.

        Parameters:
            positions: The stencil's vertex positions.

        Returns:
            The mollifier value.
        """
    @typing.overload
    def mollifier(
        self,
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        eps_x: typing.SupportsFloat,
    ) -> float:
        """
        Compute the mollifier for the distance.

        Parameters:
            positions: The stencil's vertex positions.
            eps_x: The mollifier's threshold.

        Returns:
            The mollifier value.
        """
    @typing.overload
    def mollifier_gradient(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the gradient of the mollifier for the distance wrt the positions.

        Parameters:
            positions: The stencil's vertex positions.

        Returns:
            The mollifier gradient.
        """
    @typing.overload
    def mollifier_gradient(
        self,
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        eps_x: typing.SupportsFloat,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the gradient of the mollifier for the distance wrt the positions.

        Parameters:
            positions: The stencil's vertex positions.
            eps_x: The mollifier's threshold.

        Returns:
            The mollifier gradient.
        """
    def mollifier_gradient_jacobian_wrt_x(
        self,
        rest_positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 12]"]:
        """
        Compute the jacobian of the distance mollifier's gradient w.r.t. rest positions.

        Parameters:
            rest_positions: The stencil's rest vertex positions.
            positions: The stencil's vertex positions.

        Returns:
            The jacobian of the mollifier's gradient w.r.t. rest positions.
        """
    def mollifier_gradient_wrt_x(
        self,
        rest_positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
        """
        Compute the gradient of the mollifier for the distance w.r.t. rest positions.

        Parameters:
            rest_positions: The stencil's rest vertex positions.
            positions: The stencil's vertex positions.

        Returns:
            The mollifier gradient w.r.t. rest positions.
        """
    @typing.overload
    def mollifier_hessian(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the Hessian of the mollifier for the distance wrt the positions.

        Parameters:
            positions: The stencil's vertex positions.

        Returns:
            The mollifier Hessian.
        """
    @typing.overload
    def mollifier_hessian(
        self,
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        eps_x: typing.SupportsFloat,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the Hessian of the mollifier for the distance wrt the positions.

        Parameters:
            positions: The stencil's vertex positions.
            eps_x: The mollifier's threshold.

        Returns:
            The mollifier Hessian.
        """
    def mollifier_threshold(
        self,
        rest_positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> float:
        """
        Compute the mollifier threshold for the distance.

        Parameters:
            rest_positions: The stencil's rest vertex positions.

        Returns:
            The mollifier threshold.
        """
    @property
    def dmin(self) -> float:
        """
        The minimum separation distance.
        """
    @dmin.setter
    def dmin(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def weight(self) -> float:
        """
        The term's weight (e.g., collision area)
        """
    @weight.setter
    def weight(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def weight_gradient(self) -> scipy.sparse.csc_matrix:
        """
        The gradient of the term's weight wrt the rest positions.
        """
    @weight_gradient.setter
    def weight_gradient(self, arg1: scipy.sparse.csc_matrix) -> None: ...

class NormalCollisions:
    class CollisionSetType:
        """
        Members:

          IPC : IPC set type, which uses the original formulation described in [Li et al. 2020].

          IMPROVED_MAX_APPROX : Improved max approximation set type, which uses the improved max approximation formulation described in [Li et al. 2023].

          OGC : Offset Geometric Contact set type, which uses the formulation described in [Chen et al. 2025].
        """

        IMPROVED_MAX_APPROX: typing.ClassVar[
            NormalCollisions.CollisionSetType
        ]  # value = <CollisionSetType.IMPROVED_MAX_APPROX: 1>
        IPC: typing.ClassVar[
            NormalCollisions.CollisionSetType
        ]  # value = <CollisionSetType.IPC: 0>
        OGC: typing.ClassVar[
            NormalCollisions.CollisionSetType
        ]  # value = <CollisionSetType.OGC: 2>
        __members__: typing.ClassVar[
            dict[str, NormalCollisions.CollisionSetType]
        ]  # value = {'IPC': <CollisionSetType.IPC: 0>, 'IMPROVED_MAX_APPROX': <CollisionSetType.IMPROVED_MAX_APPROX: 1>, 'OGC': <CollisionSetType.OGC: 2>}
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        def __init__(self, value: typing.SupportsInt) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: typing.SupportsInt) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    IMPROVED_MAX_APPROX: typing.ClassVar[
        NormalCollisions.CollisionSetType
    ]  # value = <CollisionSetType.IMPROVED_MAX_APPROX: 1>
    IPC: typing.ClassVar[
        NormalCollisions.CollisionSetType
    ]  # value = <CollisionSetType.IPC: 0>
    OGC: typing.ClassVar[
        NormalCollisions.CollisionSetType
    ]  # value = <CollisionSetType.OGC: 2>
    def __getitem__(self, i: typing.SupportsInt) -> NormalCollision:
        """
        Get a reference to collision at index i.

        Parameters:
            i: The index of the collision.

        Returns:
            A reference to the collision.
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of collisions.
        """
    def __str__(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> str: ...
    @typing.overload
    def build(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        dhat: typing.SupportsFloat,
        dmin: typing.SupportsFloat = 0,
        broad_phase: BroadPhase = None,
    ) -> None:
        """
        Initialize the set of collisions used to compute the barrier potential.

        Parameters:
            mesh: The collision mesh.
            vertices: Vertices of the collision mesh.
            dhat: The activation distance of the barrier.
            dmin: Minimum distance.
            broad_phase: Broad-phase to use.
        """
    @typing.overload
    def build(
        self,
        candidates: Candidates,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        dhat: typing.SupportsFloat,
        dmin: typing.SupportsFloat = 0,
    ) -> None:
        """
        Initialize the set of collisions used to compute the barrier potential.

        Parameters:
            candidates: Distance candidates from which the collision set is built.
            mesh: The collision mesh.
            vertices: Vertices of the collision mesh.
            dhat: The activation distance of the barrier.
            dmin:  Minimum distance.
        """
    def clear(self) -> None:
        """
        Clear the collision set.
        """
    def compute_minimum_distance(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> float:
        """
        Computes the minimum distance between any non-adjacent elements.

        Parameters:
            mesh: The collision mesh.
            vertices: Vertices of the collision mesh.

        Returns:
            The minimum distance between any non-adjacent elements.
        """
    def empty(self) -> bool:
        """
        Get if the collision set are empty.
        """
    def is_edge_edge(self, i: typing.SupportsInt) -> bool:
        """
        Get if the collision at i is an edge-edge collision.

        Parameters:
            i: The index of the collision.

        Returns:
            If the collision at i is an edge-edge collision.
        """
    def is_edge_vertex(self, i: typing.SupportsInt) -> bool:
        """
        Get if the collision at i is an edge-vertex collision.

        Parameters:
            i: The index of the collision.

        Returns:
            If the collision at i is an edge-vertex collision.
        """
    def is_face_vertex(self, i: typing.SupportsInt) -> bool:
        """
        Get if the collision at i is an face-vertex collision.

        Parameters:
            i: The index of the collision.

        Returns:
            If the collision at i is an face-vertex collision.
        """
    def is_plane_vertex(self, i: typing.SupportsInt) -> bool:
        """
        Get if the collision at i is an plane-vertex collision.

        Parameters:
            i: The index of the collision.

        Returns:
            If the collision at i is an plane-vertex collision.
        """
    def is_vertex_vertex(self, i: typing.SupportsInt) -> bool:
        """
        Get if the collision at i is a vertex-vertex collision.

        Parameters:
            i: The index of the collision.

        Returns:
            If the collision at i is a vertex-vertex collision.
        """
    @property
    def collision_set_type(self) -> NormalCollisions.CollisionSetType:
        """
        The type of collision set to use.

        This can be either:
          - IPC (Implicit Potential Collisions)
          - IMPROVED_MAX_APPROX (Improved Max Approximation)
          - OGC (Offset Geometric Contact)
        """
    @collision_set_type.setter
    def collision_set_type(self, arg1: NormalCollisions.CollisionSetType) -> None: ...
    @property
    def ee_collisions(self) -> list[...]: ...
    @ee_collisions.setter
    def ee_collisions(self, arg0: collections.abc.Sequence[...]) -> None: ...
    @property
    def enable_shape_derivatives(self) -> bool:
        """
        If the NormalCollisions are using the convergent formulation.
        """
    @enable_shape_derivatives.setter
    def enable_shape_derivatives(self, arg1: bool) -> None: ...
    @property
    def ev_collisions(self) -> list[...]: ...
    @ev_collisions.setter
    def ev_collisions(self, arg0: collections.abc.Sequence[...]) -> None: ...
    @property
    def fv_collisions(self) -> list[...]: ...
    @fv_collisions.setter
    def fv_collisions(self, arg0: collections.abc.Sequence[...]) -> None: ...
    @property
    def pv_collisions(self) -> list[...]: ...
    @pv_collisions.setter
    def pv_collisions(self, arg0: collections.abc.Sequence[...]) -> None: ...
    @property
    def use_area_weighting(self) -> bool:
        """
        If the NormalCollisions should use the convergent formulation.
        """
    @use_area_weighting.setter
    def use_area_weighting(self, arg1: bool) -> None: ...
    @property
    def vv_collisions(self) -> list[...]: ...
    @vv_collisions.setter
    def vv_collisions(self, arg0: collections.abc.Sequence[...]) -> None: ...

class NormalPotential:
    @typing.overload
    def __call__(
        self,
        collisions: NormalCollisions,
        mesh: ...,
        X: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> float:
        """
        Compute the potential for a set of collisions.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            X: Degrees of freedom of the collision mesh (e.g., vertices or velocities).

        Returns:
            The potential for a set of collisions.
        """
    @typing.overload
    def __call__(
        self,
        collision: NormalCollision,
        x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> float:
        """
        Compute the potential for a single collision.

        Parameters:
            collision: The collision.
            x: The collision stencil's degrees of freedom.

        Returns:
            The potential.
        """
    def force_magnitude(
        self, distance_squared: typing.SupportsFloat, dmin: typing.SupportsFloat
    ) -> float:
        """
        Compute the force magnitude for a collision.

        Parameters:
            distance_squared: The squared distance between elements.
            dmin: The minimum distance offset to the barrier.
            barrier_stiffness: The barrier stiffness.

        Returns:
            The force magnitude.
        """
    def force_magnitude_gradient(
        self,
        distance_squared: typing.SupportsFloat,
        distance_squared_gradient: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, 1]"
        ],
        dmin: typing.SupportsFloat,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the gradient of the force magnitude for a collision.

        Parameters:
            distance_squared: The squared distance between elements.
            distance_squared_gradient: The gradient of the squared distance.
            dmin: The minimum distance offset to the barrier.

        Returns:
            The gradient of the force.
        """
    @typing.overload
    def gauss_newton_hessian_diagonal(
        self,
        collisions: NormalCollisions,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the diagonal of the cumulative Gauss-Newton Hessian of the potential.

        Uses the distance-vector formulation to efficiently compute the
        diagonal of the Gauss-Newton Hessian without forming full local
        12x12 Hessian matrices. This is useful as a Jacobi preconditioner
        for iterative solvers.

        Note:
            This is a Gauss-Newton approximation (drops derivatives of
            closest-point coefficients), not the exact Hessian.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            vertices: Vertices of the collision mesh.

        Returns:
            The diagonal of the Gauss-Newton Hessian as a vector of size vertices.size().
        """
    @typing.overload
    def gauss_newton_hessian_diagonal(
        self,
        collision: NormalCollision,
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the diagonal of the Gauss-Newton Hessian for a single collision.

        Uses the distance-vector formulation (Eqs. 10-12) to efficiently
        compute the diagonal without forming the full local Hessian.

        Note:
            This is a Gauss-Newton approximation, not the exact Hessian diagonal.

        Parameters:
            collision: The collision.
            positions: The collision stencil's positions.

        Returns:
            The diagonal of the Gauss-Newton Hessian as a vector of size ndof.
        """
    @typing.overload
    def gauss_newton_hessian_quadratic_form(
        self,
        collisions: NormalCollisions,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> float:
        """
        Compute the product p^T H p for the cumulative Gauss-Newton Hessian.

        Uses the distance-vector formulation to efficiently compute the
        quadratic form without forming full local 12x12 Hessian matrices
        nor the global sparse Hessian. This is useful for nonlinear
        conjugate gradient methods.

        Note:
            This is a Gauss-Newton approximation (drops derivatives of
            closest-point coefficients), not the exact Hessian.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            vertices: Vertices of the collision mesh.
            p: The direction vector of size vertices.size().

        Returns:
            The scalar value p^T H p (approximate).
        """
    @typing.overload
    def gauss_newton_hessian_quadratic_form(
        self,
        collision: NormalCollision,
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> float:
        """
        Compute p^T H p for a single collision using the Gauss-Newton Hessian.

        Uses the distance-vector formulation (Eqs. 10, 13-14) to
        efficiently compute the quadratic form without forming the full
        local Hessian.

        Note:
            This is a Gauss-Newton approximation, not the exact Hessian.

        Parameters:
            collision: The collision.
            positions: The collision stencil's positions.
            p: The local direction vector (size ndof).

        Returns:
            The scalar value p^T H p (approximate).
        """
    @typing.overload
    def gradient(
        self,
        collisions: NormalCollisions,
        mesh: ...,
        X: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the gradient of the potential.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            X: Degrees of freedom of the collision mesh (e.g., vertices or velocities).

        Returns:
            The gradient of the potential w.r.t. X. This will have a size of X.size.
        """
    @typing.overload
    def gradient(
        self,
        collision: NormalCollision,
        x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the gradient of the potential for a single collision.

        Parameters:
            collision: The collision.
            x: The collision stencil's degrees of freedom.

        Returns:
            The gradient of the potential.
        """
    @typing.overload
    def hessian(
        self,
        collisions: NormalCollisions,
        mesh: ...,
        X: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        project_hessian_to_psd: PSDProjectionMethod = ...,
    ) -> scipy.sparse.csc_matrix:
        """
        Compute the hessian of the potential.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            X: Degrees of freedom of the collision mesh (e.g., vertices or velocities).
            project_hessian_to_psd: Make sure the hessian is positive semi-definite.

        Returns:
            The Hessian of the potential w.r.t. X. This will have a size of X.size by X.size.
        """
    @typing.overload
    def hessian(
        self,
        collision: NormalCollision,
        x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        project_hessian_to_psd: PSDProjectionMethod = ...,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the hessian of the potential for a single collision.

        Parameters:
            collision: The collision.
            x: The collision stencil's degrees of freedom.

        Returns:
            The hessian of the potential.
        """
    @typing.overload
    def shape_derivative(
        self,
        collisions: NormalCollisions,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> scipy.sparse.csc_matrix:
        """
        Compute the shape derivative of the potential.

        std::runtime_error If the collision collisions were not built with shape derivatives enabled.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            vertices: Vertices of the collision mesh.

        Returns:
            The derivative of the force with respect to X, the rest vertices.
        """
    @typing.overload
    def shape_derivative(
        self,
        collision: NormalCollision,
        vertex_ids: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        rest_positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> list[..., ...]:
        """
        Compute the shape derivative of the potential for a single collision.

        Parameters:
            collision: The collision.
            vertex_ids: The collision stencil's vertex ids.
            rest_positions: The collision stencil's rest positions.
            positions: The collision stencil's positions.
            ,out]: out Store the triplets of the shape derivative here.
        """

class NormalizedClampedLogBarrier(ClampedLogBarrier):
    """

    Normalized barrier function from [Li et al. 2023].

    .. math::

        b(d) = -\\left(\\frac{d}{\\hat{d}}-1\\right)^2\\ln\\left(\\frac{d}{\\hat{d}}\\right)


    """
    def __init__(self) -> None: ...

class PSDProjectionMethod:
    """
    Enumeration of implemented PSD projection methods.

    Members:

      NONE : No PSD projection

      CLAMP : Clamp negative eigenvalues to zero

      ABS : Flip negative eigenvalues
    """

    ABS: typing.ClassVar[PSDProjectionMethod]  # value = <PSDProjectionMethod.ABS: 2>
    CLAMP: typing.ClassVar[
        PSDProjectionMethod
    ]  # value = <PSDProjectionMethod.CLAMP: 1>
    NONE: typing.ClassVar[PSDProjectionMethod]  # value = <PSDProjectionMethod.NONE: 0>
    __members__: typing.ClassVar[
        dict[str, PSDProjectionMethod]
    ]  # value = {'NONE': <PSDProjectionMethod.NONE: 0>, 'CLAMP': <PSDProjectionMethod.CLAMP: 1>, 'ABS': <PSDProjectionMethod.ABS: 2>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class PlaneVertexCandidate(CollisionStencil):
    @staticmethod
    def __init__(*args, **kwargs) -> None: ...
    @property
    def plane(self) -> ...:
        """
        Plane of the candidate
        """
    @plane.setter
    def plane(*args, **kwargs): ...
    @property
    def vertex_id(self) -> int:
        """
        ID of the vertex
        """
    @vertex_id.setter
    def vertex_id(self, arg0: typing.SupportsInt) -> None: ...

class PlaneVertexNormalCollision(PlaneVertexCandidate, NormalCollision):
    @staticmethod
    def __init__(*args, **kwargs) -> None: ...

class PlaneVertexTangentialCollision(PlaneVertexCandidate, TangentialCollision):
    @typing.overload
    def __init__(self, collision: PlaneVertexNormalCollision) -> None: ...
    @typing.overload
    def __init__(
        self,
        collision: PlaneVertexNormalCollision,
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        normal_potential: ...,
    ) -> None: ...

class Point2Point2Collision(SmoothCollision2):
    def name(self) -> str:
        """
        Get the type name of collision
        """
    def num_vertices(self) -> int:
        """
        Get the number of vertices
        """

class PointEdgeDistanceType:
    """
    Members:

      P_E0 : point is closest to edge vertex zero

      P_E1 : point is closest to edge vertex one

      P_E : point is closest to the interior of the edge

      AUTO : automatically determine the closest point
    """

    AUTO: typing.ClassVar[
        PointEdgeDistanceType
    ]  # value = <PointEdgeDistanceType.AUTO: 3>
    P_E: typing.ClassVar[
        PointEdgeDistanceType
    ]  # value = <PointEdgeDistanceType.P_E: 2>
    P_E0: typing.ClassVar[
        PointEdgeDistanceType
    ]  # value = <PointEdgeDistanceType.P_E0: 0>
    P_E1: typing.ClassVar[
        PointEdgeDistanceType
    ]  # value = <PointEdgeDistanceType.P_E1: 1>
    __members__: typing.ClassVar[
        dict[str, PointEdgeDistanceType]
    ]  # value = {'P_E0': <PointEdgeDistanceType.P_E0: 0>, 'P_E1': <PointEdgeDistanceType.P_E1: 1>, 'P_E': <PointEdgeDistanceType.P_E: 2>, 'AUTO': <PointEdgeDistanceType.AUTO: 3>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class PointTriangleDistanceType:
    """
    Members:

      P_T0 : point is closest to triangle vertex zero

      P_T1 : point is closest to triangle vertex one

      P_T2 : point is closest to triangle vertex two

      P_E0 : point is closest to triangle edge zero (vertex zero to one)

      P_E1 : point is closest to triangle edge one (vertex one to two)

      P_E2 : point is closest to triangle edge two (vertex two to zero)

      P_T : point is closest to the interior of the triangle

      AUTO : automatically determine the closest point
    """

    AUTO: typing.ClassVar[
        PointTriangleDistanceType
    ]  # value = <PointTriangleDistanceType.AUTO: 7>
    P_E0: typing.ClassVar[
        PointTriangleDistanceType
    ]  # value = <PointTriangleDistanceType.P_E0: 3>
    P_E1: typing.ClassVar[
        PointTriangleDistanceType
    ]  # value = <PointTriangleDistanceType.P_E1: 4>
    P_E2: typing.ClassVar[
        PointTriangleDistanceType
    ]  # value = <PointTriangleDistanceType.P_E2: 5>
    P_T: typing.ClassVar[
        PointTriangleDistanceType
    ]  # value = <PointTriangleDistanceType.P_T: 6>
    P_T0: typing.ClassVar[
        PointTriangleDistanceType
    ]  # value = <PointTriangleDistanceType.P_T0: 0>
    P_T1: typing.ClassVar[
        PointTriangleDistanceType
    ]  # value = <PointTriangleDistanceType.P_T1: 1>
    P_T2: typing.ClassVar[
        PointTriangleDistanceType
    ]  # value = <PointTriangleDistanceType.P_T2: 2>
    __members__: typing.ClassVar[
        dict[str, PointTriangleDistanceType]
    ]  # value = {'P_T0': <PointTriangleDistanceType.P_T0: 0>, 'P_T1': <PointTriangleDistanceType.P_T1: 1>, 'P_T2': <PointTriangleDistanceType.P_T2: 2>, 'P_E0': <PointTriangleDistanceType.P_E0: 3>, 'P_E1': <PointTriangleDistanceType.P_E1: 4>, 'P_E2': <PointTriangleDistanceType.P_E2: 5>, 'P_T': <PointTriangleDistanceType.P_T: 6>, 'AUTO': <PointTriangleDistanceType.AUTO: 7>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SmoothCollision2:
    def __call__(
        self,
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        params: ...,
    ) -> float:
        """
        Compute the potential.

        Parameters:
            positions: The vertex positions.
            params: The parameters.

        Returns:
            The potential (not scaled by the barrier stiffness) of this collision pair.
        """
    def __getitem__(self, i: typing.SupportsInt) -> int:
        """
        Get primitive id.

        Parameters:
            i: 0 or 1.

        Returns:
            The index of the primitive.
        """
    def n_dofs(self) -> int:
        """
        Get the degree of freedom
        """

class SmoothCollisions:
    def __getitem__(self, i: typing.SupportsInt) -> SmoothCollision2:
        """
        Get a reference to collision at index i.

        Parameters:
            i: The index of the collision.

        Returns:
            A reference to the collision.
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of collisions.
        """
    def build(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        param: ...,
        use_adaptive_dhat: bool = False,
        broad_phase: BroadPhase = None,
    ) -> None:
        """
        Initialize the set of collisions used to compute the barrier potential.

        Parameters:
            mesh: The collision mesh.
            vertices: Vertices of the collision mesh.
            param: SmoothContactParameters.
            use_adaptive_dhat: If the adaptive dhat should be used.
            broad_phase: Broad phase method.
        """
    def clear(self) -> None:
        """
        Clear the collision set.
        """
    def compute_minimum_distance(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> float:
        """
        Computes the minimum distance between any non-adjacent elements.

        Parameters:
            mesh: The collision mesh.
            vertices: Vertices of the collision mesh.

        Returns:
            The minimum distance between any non-adjacent elements.
        """
    def empty(self) -> bool:
        """
        Get if the collision set is empty.
        """
    def n_candidates(self) -> int:
        """
        Get the number of candidates.
        """
    def to_string(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        param: ...,
    ) -> str: ...

class SmoothContactParameters:
    @typing.overload
    def __init__(
        self,
        dhat: typing.SupportsFloat,
        alpha_t: typing.SupportsFloat,
        beta_t: typing.SupportsFloat,
        alpha_n: typing.SupportsFloat,
        beta_n: typing.SupportsFloat,
        r: typing.SupportsInt,
    ) -> None:
        """
        Construct parameter set for smooth contact.

        Parameters:
            dhat, alpha_t, beta_t, alpha_n, beta_n, r
        """
    @typing.overload
    def __init__(
        self,
        dhat: typing.SupportsFloat,
        alpha_t: typing.SupportsFloat,
        beta_t: typing.SupportsFloat,
        r: typing.SupportsInt,
    ) -> None:
        """
        Construct parameter set for smooth contact.

        Parameters:
            dhat, alpha_t, beta_t, r
        """
    @property
    def alpha_n(self) -> float: ...
    @property
    def alpha_t(self) -> float: ...
    @property
    def beta_n(self) -> float: ...
    @property
    def beta_t(self) -> float: ...
    @property
    def dhat(self) -> float: ...
    @property
    def r(self) -> int: ...

class SmoothPotential:
    @typing.overload
    def __call__(
        self,
        collisions: SmoothCollisions,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> float:
        """
        Compute the barrier potential for a set of collisions.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            vertices: Vertices of the collision mesh.

        Returns:
            The sum of all barrier potentials (not scaled by the barrier stiffness).
        """
    @typing.overload
    def __call__(
        self,
        collision: SmoothCollision2,
        x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> float:
        """
        Compute the potential for a single collision.

        Parameters:
            collision: The collision.
            x: The collision stencil's degrees of freedom.

        Returns:
            The potential.
        """
    def __init__(self, param: SmoothContactParameters) -> None:
        """
        Construct a smooth barrier potential.

        Parameters:
            param: A set of parameters.
        """
    @typing.overload
    def gradient(
        self,
        collisions: SmoothCollisions,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the gradient of the barrier potential.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            vertices: Vertices of the collision mesh.

        Returns:
            The gradient of all barrier potentials (not scaled by the barrier stiffness). This will have a size of |vertices|.
        """
    @typing.overload
    def gradient(
        self,
        collision: SmoothCollision2,
        x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the gradient of the potential for a single collision.

        Parameters:
            collision: The collision.
            x: The collision stencil's degrees of freedom.

        Returns:
            The gradient of the potential.
        """
    @typing.overload
    def hessian(
        self,
        collisions: SmoothCollisions,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        project_hessian_to_psd: PSDProjectionMethod = ...,
    ) -> scipy.sparse.csc_matrix:
        """
        Compute the hessian of the barrier potential.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            vertices: Vertices of the collision mesh.
            project_hessian_to_psd: Make sure the hessian is positive semi-definite.

        Returns:
            The hessian of all barrier potentials (not scaled by the barrier stiffness). This will have a size of |vertices|x|vertices|.
        """
    @typing.overload
    def hessian(
        self,
        collision: SmoothCollision2,
        x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        project_hessian_to_psd: PSDProjectionMethod = ...,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the hessian of the potential for a single collision.

        Parameters:
            collision: The collision.
            x: The collision stencil's degrees of freedom.

        Returns:
            The hessian of the potential.
        """

class SpatialHash(BroadPhase):
    def __init__(self) -> None: ...
    @typing.overload
    def build(
        self,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        inflation_radius: typing.SupportsFloat = 0,
        voxel_size: typing.SupportsFloat = -1,
    ) -> None: ...
    @typing.overload
    def build(
        self,
        vertices_t0: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        vertices_t1: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        edges: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        faces: typing.Annotated[
            numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
        ],
        inflation_radius: typing.SupportsFloat = 0,
        voxel_size: typing.SupportsFloat = -1,
    ) -> None: ...

class SweepAndPrune(BroadPhase):
    def __init__(self) -> None: ...

class TangentialAdhesionPotential(TangentialPotential):
    def __init__(self, eps_a: typing.SupportsFloat) -> None:
        """
        Construct a tangential adhesion potential.

        Parameters:
            eps_a: The tangential adhesion mollifier parameter :math:`\\epsilon_a`.
        """
    @property
    def eps_a(self) -> float:
        """
        Get the tangential adhesion mollifier parameter :math:`\\epsilon_a`.
        """
    @eps_a.setter
    def eps_a(self, arg1: typing.SupportsFloat) -> None: ...

class TangentialCollision(CollisionStencil):
    def compute_closest_point(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the barycentric coordinates of the closest point.

        Parameters:
            positions: Collision stencil's vertex positions.

        Returns:
            Barycentric coordinates of the closest point.
        """
    def compute_closest_point_jacobian(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the Jacobian of the barycentric coordinates of the closest point.

        Parameters:
            positions: Collision stencil's vertex positions.

        Returns:
            Jacobian of the barycentric coordinates of the closest point.
        """
    def compute_tangent_basis(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the tangent basis of the collision.

        Parameters:
            positions: Collision stencil's vertex positions.

        Returns:
            Tangent basis of the collision.
        """
    def compute_tangent_basis_jacobian(
        self, positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the Jacobian of the tangent basis of the collision.

        Parameters:
            positions: Collision stencil's vertex positions.

        Returns:
            Jacobian of the tangent basis of the collision.
        """
    def relative_velocity(
        self,
        velocities: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the relative velocity of the collision.

        Parameters:
            velocities: Collision stencil's vertex velocities.

        Returns:
            Relative velocity of the collision.
        """
    def relative_velocity_dx_dbeta(
        self,
        closest_point: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Construct the Jacobian of the relative velocity premultiplier wrt the closest points.

        Parameters:
            closest_point: Barycentric coordinates of the closest point.

        Returns:
            Jacobian of the relative velocity premultiplier wrt the closest points.
        """
    @typing.overload
    def relative_velocity_jacobian(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Construct the premultiplier matrix for the relative velocity.

        Note:
            Uses the cached closest point.

        Returns:
            A matrix M such that `relative_velocity = M * velocities`.
        """
    @typing.overload
    def relative_velocity_jacobian(
        self,
        closest_point: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Construct the premultiplier matrix for the relative velocity.

        Parameters:
            closest_point: Barycentric coordinates of the closest point.

        Returns:
            A matrix M such that `relative_velocity = M * velocities`.
        """
    @property
    def closest_point(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Barycentric coordinates of the closest point(s)
        """
    @closest_point.setter
    def closest_point(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]
    ) -> None: ...
    @property
    def dim(self) -> int:
        """
        Get the dimension of the collision.
        """
    @property
    def mu_aniso(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]:
        """
        Tangential anisotropy scaling in the collision's tangent basis. (1,1) = isotropic (default). Positive entries are recommended for physically meaningful anisotropic scaling. Scales tau before evaluating friction.
        """
    @mu_aniso.setter
    def mu_aniso(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]
    ) -> None: ...
    @property
    def mu_k(self) -> float:
        """
        Ratio between normal and kinetic tangential forces (e.g., friction coefficient)
        """
    @mu_k.setter
    def mu_k(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def mu_k_aniso(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]:
        """
        Kinetic friction ellipse axes (2D, one per tangent). Zero → scalar mu_k. Matchstick model (CGF 2019, DOI 10.1111/cgf.13885).
        """
    @mu_k_aniso.setter
    def mu_k_aniso(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]
    ) -> None: ...
    @property
    def mu_k_effective_lagged(self) -> float:
        """
        Lagged matchstick effective kinetic μ.
        """
    @mu_k_effective_lagged.setter
    def mu_k_effective_lagged(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def mu_s(self) -> float:
        """
        Ratio between normal and static tangential forces (e.g., friction coefficient)
        """
    @mu_s.setter
    def mu_s(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def mu_s_aniso(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]:
        """
        Static friction ellipse axes (2D, one per tangent). Zero → scalar mu_s. Matchstick model (CGF 2019, DOI 10.1111/cgf.13885).
        """
    @mu_s_aniso.setter
    def mu_s_aniso(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]
    ) -> None: ...
    @property
    def mu_s_effective_lagged(self) -> float:
        """
        Lagged matchstick effective static μ (refresh via TangentialCollisions.update_lagged_anisotropic_friction_coefficients).
        """
    @mu_s_effective_lagged.setter
    def mu_s_effective_lagged(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def ndof(self) -> int:
        """
        Get the number of degrees of freedom for the collision.
        """
    @property
    def normal_force_magnitude(self) -> float:
        """
        Normal force magnitude
        """
    @normal_force_magnitude.setter
    def normal_force_magnitude(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def tangent_basis(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Tangent basis of the collision (max size 3×2)
        """
    @tangent_basis.setter
    def tangent_basis(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]
    ) -> None: ...
    @property
    def weight(self) -> float:
        """
        Weight
        """
    @weight.setter
    def weight(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def weight_gradient(self) -> scipy.sparse.csc_matrix:
        """
        Gradient of weight with respect to all DOF
        """
    @weight_gradient.setter
    def weight_gradient(self, arg1: scipy.sparse.csc_matrix) -> None: ...

class TangentialCollisions:
    @staticmethod
    def default_blend_mu(
        mu0: typing.SupportsFloat, mu1: typing.SupportsFloat
    ) -> float: ...
    def __getitem__(self, i: typing.SupportsInt) -> TangentialCollision:
        """
        Get a reference to collision at index i.

        Parameters:
            i: The index of the collision.

        Returns:
            A reference to the collision.
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of friction collisions.
        """
    @typing.overload
    def build(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        collisions: NormalCollisions,
        normal_potential: ...,
        mu: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def build(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        collisions: NormalCollisions,
        normal_potential: ...,
        mu_s: typing.SupportsFloat,
        mu_k: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def build(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        collisions: NormalCollisions,
        normal_potential: ...,
        mu_s: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        mu_k: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> None: ...
    @typing.overload
    def build(
        self,
        mesh: ...,
        vertices: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        collisions: NormalCollisions,
        normal_potential: ...,
        mu_s: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        mu_k: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        blend_mu: collections.abc.Callable[
            [typing.SupportsFloat, typing.SupportsFloat], float
        ],
    ) -> None: ...
    def clear(self) -> None:
        """
        Clear the friction collisions.
        """
    def empty(self) -> bool:
        """
        Get if the friction collisions are empty.
        """
    def reset_lagged_anisotropic_friction_coefficients(self) -> None:
        """
        Set lagged effective μ to scalar mu_s/mu_k on each collision (done automatically after build).
        """
    def update_lagged_anisotropic_friction_coefficients(
        self,
        mesh: ...,
        rest_positions: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        lagged_displacements: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        velocities: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> None:
        """
        Refresh matchstick effective μ from lagged geometry and slip. Call when mu_s_aniso is nonzero (e.g. each Newton iteration).
        """
    @property
    def ee_collisions(self) -> list[...]: ...
    @ee_collisions.setter
    def ee_collisions(self, arg0: collections.abc.Sequence[...]) -> None: ...
    @property
    def ev_collisions(self) -> list[...]: ...
    @ev_collisions.setter
    def ev_collisions(self, arg0: collections.abc.Sequence[...]) -> None: ...
    @property
    def fv_collisions(self) -> list[...]: ...
    @fv_collisions.setter
    def fv_collisions(self, arg0: collections.abc.Sequence[...]) -> None: ...
    @property
    def vv_collisions(self) -> list[...]: ...
    @vv_collisions.setter
    def vv_collisions(self, arg0: collections.abc.Sequence[...]) -> None: ...

class TangentialPotential:
    class DiffWRT:
        """
        Members:

          REST_POSITIONS : Differentiate w.r.t. rest positions

          LAGGED_DISPLACEMENTS : Differentiate w.r.t. lagged displacements

          VELOCITIES : Differentiate w.r.t. current velocities
        """

        LAGGED_DISPLACEMENTS: typing.ClassVar[
            TangentialPotential.DiffWRT
        ]  # value = <DiffWRT.LAGGED_DISPLACEMENTS: 1>
        REST_POSITIONS: typing.ClassVar[
            TangentialPotential.DiffWRT
        ]  # value = <DiffWRT.REST_POSITIONS: 0>
        VELOCITIES: typing.ClassVar[
            TangentialPotential.DiffWRT
        ]  # value = <DiffWRT.VELOCITIES: 2>
        __members__: typing.ClassVar[
            dict[str, TangentialPotential.DiffWRT]
        ]  # value = {'REST_POSITIONS': <DiffWRT.REST_POSITIONS: 0>, 'LAGGED_DISPLACEMENTS': <DiffWRT.LAGGED_DISPLACEMENTS: 1>, 'VELOCITIES': <DiffWRT.VELOCITIES: 2>}
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        def __init__(self, value: typing.SupportsInt) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: typing.SupportsInt) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    LAGGED_DISPLACEMENTS: typing.ClassVar[
        TangentialPotential.DiffWRT
    ]  # value = <DiffWRT.LAGGED_DISPLACEMENTS: 1>
    REST_POSITIONS: typing.ClassVar[
        TangentialPotential.DiffWRT
    ]  # value = <DiffWRT.REST_POSITIONS: 0>
    VELOCITIES: typing.ClassVar[
        TangentialPotential.DiffWRT
    ]  # value = <DiffWRT.VELOCITIES: 2>
    @typing.overload
    def __call__(
        self,
        collisions: TangentialCollisions,
        mesh: ...,
        X: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> float:
        """
        Compute the potential for a set of collisions.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            X: Degrees of freedom of the collision mesh (e.g., vertices or velocities).

        Returns:
            The potential for a set of collisions.
        """
    @typing.overload
    def __call__(
        self,
        collision: TangentialCollision,
        x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> float:
        """
        Compute the potential for a single collision.

        Parameters:
            collision: The collision.
            x: The collision stencil's degrees of freedom.

        Returns:
            The potential.
        """
    @typing.overload
    def force(
        self,
        collisions: TangentialCollisions,
        mesh: ...,
        rest_positions: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        lagged_displacements: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        velocities: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        normal_potential: NormalPotential,
        dmin: typing.SupportsFloat = 0,
        no_mu: bool = False,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the friction force from the given velocities.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            rest_positions: Rest positions of the vertices (rowwise).
            lagged_displacements: Previous displacements of the vertices (rowwise).
            velocities: Current displacements of the vertices (rowwise).
            normal_potential: Normal potential (used for normal force magnitude).
            dmin: Minimum distance (used for normal force magnitude).
            no_mu: whether to not multiply by mu

        Returns:
            The friction force.
        """
    @typing.overload
    def force(
        self,
        collision: TangentialCollision,
        rest_positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        lagged_displacements: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, 1]"
        ],
        velocities: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        normal_potential: NormalPotential,
        dmin: typing.SupportsFloat = 0,
        no_mu: bool = False,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the friction force.

        Parameters:
            collision: The collision
            rest_positions: Rest positions of the vertices (rowwise).
            lagged_displacements: Previous displacements of the vertices (rowwise).
            velocities: Current displacements of the vertices (rowwise).
            normal_potential: Normal potential (used for normal force magnitude).
            dmin: Minimum distance (used for normal force magnitude).
            no_mu: Whether to not multiply by mu

        Returns:
            Friction force
        """
    @typing.overload
    def force_jacobian(
        self,
        collisions: TangentialCollisions,
        mesh: ...,
        rest_positions: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        lagged_displacements: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        velocities: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        normal_potential: NormalPotential,
        wrt: TangentialPotential.DiffWRT,
        dmin: typing.SupportsFloat = 0,
    ) -> scipy.sparse.csc_matrix:
        """
        Compute the Jacobian of the friction force wrt the velocities.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            rest_positions: Rest positions of the vertices (rowwise).
            lagged_displacements: Previous displacements of the vertices (rowwise).
            velocities: Current displacements of the vertices (rowwise).
            normal_potential: Normal potential (used for normal force magnitude).
            wrt: The variable to take the derivative with respect to.
            dmin: Minimum distance (used for normal force magnitude).

        Returns:
            The Jacobian of the friction force wrt the velocities.
        """
    @typing.overload
    def force_jacobian(
        self,
        collision: TangentialCollision,
        rest_positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        lagged_displacements: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, 1]"
        ],
        velocities: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        normal_potential: NormalPotential,
        wrt: TangentialPotential.DiffWRT,
        dmin: typing.SupportsFloat = 0,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the friction force Jacobian.

        Parameters:
            collision: The collision
            rest_positions: Rest positions of the vertices (rowwise).
            lagged_displacements: Previous displacements of the vertices (rowwise).
            velocities: Current displacements of the vertices (rowwise).
            normal_potential: Normal potential (used for normal force magnitude).
            wrt: Variable to differentiate the friction force with respect to.
            dmin: Minimum distance (used for normal force magnitude).

        Returns:
            Friction force Jacobian
        """
    @typing.overload
    def gradient(
        self,
        collisions: TangentialCollisions,
        mesh: ...,
        X: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the gradient of the potential.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            X: Degrees of freedom of the collision mesh (e.g., vertices or velocities).

        Returns:
            The gradient of the potential w.r.t. X. This will have a size of X.size.
        """
    @typing.overload
    def gradient(
        self,
        collision: TangentialCollision,
        x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the gradient of the potential for a single collision.

        Parameters:
            collision: The collision.
            x: The collision stencil's degrees of freedom.

        Returns:
            The gradient of the potential.
        """
    @typing.overload
    def hessian(
        self,
        collisions: TangentialCollisions,
        mesh: ...,
        X: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        project_hessian_to_psd: PSDProjectionMethod = ...,
    ) -> scipy.sparse.csc_matrix:
        """
        Compute the hessian of the potential.

        Parameters:
            collisions: The set of collisions.
            mesh: The collision mesh.
            X: Degrees of freedom of the collision mesh (e.g., vertices or velocities).
            project_hessian_to_psd: Make sure the hessian is positive semi-definite.

        Returns:
            The Hessian of the potential w.r.t. X. This will have a size of X.size by X.size.
        """
    @typing.overload
    def hessian(
        self,
        collision: TangentialCollision,
        x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        project_hessian_to_psd: PSDProjectionMethod = ...,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Compute the hessian of the potential for a single collision.

        Parameters:
            collision: The collision.
            x: The collision stencil's degrees of freedom.

        Returns:
            The hessian of the potential.
        """

class TightInclusionCCD(NarrowPhaseCCD):
    DEFAULT_CONSERVATIVE_RESCALING: typing.ClassVar[float] = 0.8
    DEFAULT_MAX_ITERATIONS: typing.ClassVar[int] = 10000000
    DEFAULT_TOLERANCE: typing.ClassVar[float] = 1e-06
    SMALL_TOI: typing.ClassVar[float] = 1e-06
    def __init__(
        self,
        tolerance: typing.SupportsFloat = 1e-06,
        max_iterations: typing.SupportsInt = 10000000,
        conservative_rescaling: typing.SupportsFloat = 0.8,
    ) -> None:
        """
        Construct a new AdditiveCCD object.

        Parameters:
            conservative_rescaling: The conservative rescaling of the time of impact.
        """
    @property
    def conservative_rescaling(self) -> float:
        """
        Conservative rescaling of the time of impact.
        """
    @conservative_rescaling.setter
    def conservative_rescaling(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def max_iterations(self) -> int:
        """
        Maximum number of iterations.
        """
    @max_iterations.setter
    def max_iterations(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def tolerance(self) -> float:
        """
        Solver tolerance.
        """
    @tolerance.setter
    def tolerance(self, arg0: typing.SupportsFloat) -> None: ...

class TwoStageBarrier(Barrier):
    """

            Two-stage activation barrier from [Chen et al. 2025].

            .. math::

                b(d) = \\begin{cases}
                    -\\frac{\\hat{d}^2}{4} \\left(\\ln\\left(\\frac{2d}{\\hat{d}}\\right) -
                    \\tfrac{1}{2}\\right) & d < \\frac{\\hat{d}}{2}\\\\
                    \\tfrac{1}{2} (\\hat{d} - d)^2 & d < \\hat{d}\\\\
                    0 & d \\ge \\hat{d}
                \\end{cases}


    """
    def __init__(self) -> None: ...

class VertexVertexCandidate(CollisionStencil):
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: VertexVertexCandidate) -> bool: ...
    @typing.overload
    def __init__(
        self, vertex0_id: typing.SupportsInt, vertex1_id: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, vertex_ids: tuple[typing.SupportsInt, typing.SupportsInt]
    ) -> None: ...
    def __lt__(self, other: VertexVertexCandidate) -> bool:
        """
        Compare EdgeVertexCandidates for sorting.
        """
    def __ne__(self, other: VertexVertexCandidate) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def vertex0_id(self) -> int:
        """
        ID of the first vertex
        """
    @vertex0_id.setter
    def vertex0_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def vertex1_id(self) -> int:
        """
        ID of the second vertex
        """
    @vertex1_id.setter
    def vertex1_id(self, arg0: typing.SupportsInt) -> None: ...

class VertexVertexNormalCollision(VertexVertexCandidate, NormalCollision):
    @typing.overload
    def __init__(
        self, vertex0_id: typing.SupportsInt, vertex1_id: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(self, vv_candidate: VertexVertexCandidate) -> None: ...

class VertexVertexTangentialCollision(VertexVertexCandidate, TangentialCollision):
    @typing.overload
    def __init__(self, collision: VertexVertexNormalCollision) -> None: ...
    @typing.overload
    def __init__(
        self,
        collision: VertexVertexNormalCollision,
        positions: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        normal_potential: ...,
    ) -> None: ...

def anisotropic_mu_eff_f(
    tau_dir: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    mu_s_aniso: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    mu_k_aniso: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
) -> tuple[float, float]:
    """
    Effective static and kinetic friction along a unit direction for the
    elliptical (matchstick) model: μ_eff = sqrt((μ₀ t₀)² + (μ₁ t₁)²).
    Matchstick model: Erleben et al., CGF 2019, DOI 10.1111/cgf.13885.

    Parameters:
        tau_dir: Unit 2D direction (tau / ||tau||).
        mu_s_aniso: Static friction ellipse axes (2D).
        mu_k_aniso: Kinetic friction ellipse axes (2D).

    Returns:
        (mu_s_eff, mu_k_eff) along tau_dir. If anisotropic axes are
        zero, returns (0, 0) as the direct ellipse-formula result.
        Isotropic fallback is handled by higher-level anisotropic
        friction routines.
    """

def barrier(d: typing.SupportsFloat, dhat: typing.SupportsFloat) -> float:
    """
    Function that grows to infinity as d approaches 0 from the right.

    .. math::

        b(d) = -(d-\\hat{d})^2\\ln\\left(\\frac{d}{\\hat{d}}\\right)

    Parameters:
        d: The distance.
        dhat: Activation distance of the barrier.

    Returns:
        The value of the barrier function at d.
    """

def barrier_first_derivative(
    d: typing.SupportsFloat, dhat: typing.SupportsFloat
) -> float:
    """
    Derivative of the barrier function.

    .. math::

        b'(d) = (\\hat{d}-d) \\left( 2\\ln\\left( \\frac{d}{\\hat{d}} \\right) -
        \\frac{\\hat{d}}{d} + 1\\right)

    Parameters:
        d: The distance.
        dhat: Activation distance of the barrier.

    Returns:
        The derivative of the barrier wrt d.
    """

def barrier_force_magnitude(
    distance_squared: typing.SupportsFloat,
    barrier: Barrier,
    dhat: typing.SupportsFloat,
    barrier_stiffness: typing.SupportsFloat,
    dmin: typing.SupportsFloat = 0,
) -> float:
    """
    Compute the magnitude of the force due to a barrier.

    Parameters:
        distance_squared: The squared distance between elements.
        barrier: The barrier function.
        dhat: The activation distance of the barrier.
        barrier_stiffness: The stiffness of the barrier.
        dmin: The minimum distance offset to the barrier.

    Returns:
        The magnitude of the force.
    """

def barrier_force_magnitude_gradient(
    distance_squared: typing.SupportsFloat,
    distance_squared_gradient: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, 1]"
    ],
    barrier: Barrier,
    dhat: typing.SupportsFloat,
    barrier_stiffness: typing.SupportsFloat,
    dmin: typing.SupportsFloat = 0,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the gradient of the magnitude of the force due to a barrier.

    Parameters:
        distance_squared: The squared distance between elements.
        distance_squared_gradient: The gradient of the squared distance.
        barrier: The barrier function.
        dhat: The activation distance of the barrier.
        barrier_stiffness: The stiffness of the barrier.
        dmin: The minimum distance offset to the barrier.

    Returns:
        The gradient of the force.
    """

def barrier_second_derivative(
    d: typing.SupportsFloat, dhat: typing.SupportsFloat
) -> float:
    """
    Second derivative of the barrier function.

    .. math::

        b''(d) = \\left( \\frac{\\hat{d}}{d} + 2 \\right) \\frac{\\hat{d}}{d} -
        2\\ln\\left( \\frac{d}{\\hat{d}} \\right) - 3

    Parameters:
        d: The distance.
        dhat: Activation distance of the barrier.

    Returns:
        The second derivative of the barrier wrt d.
    """

def build_edge_boxes(
    vertex_boxes: collections.abc.Sequence[AABB],
    edges: typing.Annotated[
        numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
    ],
) -> list[AABB]:
    """
    Build one AABB per edge.

    Parameters:
        vertex_boxes: Vertex AABBs.
        edges: Edges (rowwise).

    Returns:
        edge_boxes: Edge AABBs.
    """

def build_face_boxes(
    vertex_boxes: collections.abc.Sequence[AABB],
    faces: typing.Annotated[
        numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
    ],
) -> list[AABB]:
    """
    Build one AABB per face.

    Parameters:
        vertex_boxes: Vertex AABBs.
        faces: Faces (rowwise).

    Returns:
        face_boxes: Face AABBs.
    """

@typing.overload
def build_vertex_boxes(
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    inflation_radius: typing.SupportsFloat = 0,
) -> list[AABB]:
    """
    Build one AABB per vertex position (row of V).

    Parameters:
        vertices: Vertex positions (rowwise).
        inflation_radius: Radius of a sphere around the points which the AABBs enclose.

    Returns:
        Vertex AABBs.
    """

@typing.overload
def build_vertex_boxes(
    vertices_t0: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    vertices_t1: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    inflation_radius: typing.SupportsFloat = 0,
) -> list[AABB]:
    """
    Build one AABB per vertex position moving linearly from t=0 to t=1.

    Parameters:
        vertices_t0: Vertex positions at t=0 (rowwise).
        vertices_t1: Vertex positions at t=1 (rowwise).
        inflation_radius: Radius of a capsule around the temporal edges which the AABBs enclose.

    Returns:
        Vertex AABBs.
    """

def check_initial_distance(
    initial_distance: typing.SupportsFloat, min_distance: typing.SupportsFloat
) -> tuple[bool, float]: ...
def compute_collision_free_stepsize(
    mesh: CollisionMesh,
    vertices_t0: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    vertices_t1: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    min_distance: typing.SupportsFloat = 0.0,
    broad_phase: BroadPhase = None,
    narrow_phase_ccd: NarrowPhaseCCD = ...,
) -> float:
    """
    Computes a maximal step size that is collision free.

    Note:
        Assumes the trajectory is linear.
        When using SweepAndTiniestQueue broad phase, tolerance and
        max_iterations are extracted from TightInclusionCCD if provided,
        otherwise defaults are used.

    Parameters:
        mesh: The collision mesh.
        vertices_t0: Vertex vertices at start as rows of a matrix. Assumes vertices_t0 is intersection free.
        vertices_t1: Surface vertex vertices at end as rows of a matrix.
        min_distance: The minimum distance allowable between any two elements.
        broad_phase: Broad phase to use.
        narrow_phase_ccd: The narrow phase CCD algorithm to use.

    Returns:
        A step-size :math:`\\in [0, 1]` that is collision free. A value of 1.0 if a full step and 0.0 is no step.
    """

def cross_product_matrix(
    v: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 3]"]:
    """
    Cross product matrix for 3D vectors.

    Parameters
    ----------
    v: Vector to create the cross product matrix for.

    Returns
    -------
    The cross product matrix of the vector.
    """

def cross_product_matrix_jacobian() -> typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[9, 3]"
]:
    """
    Computes the Jacobian of the cross product matrix.
    Returns
    -------
    The Jacobian of the cross product matrix.
    """

def dihedral_angle(
    x0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    x1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    x2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    x3: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
    """
    Compute the bending angle between two triangles sharing an edge.
        x0---x2
         | \\ |
        x1---x3

    Parameters
    ----------
    x0 : Eigen::Vector3d
        The first vertex of the edge.
    x1 : Eigen::Vector3d
        The second vertex of the edge.
    x2 : Eigen::Vector3d
        The opposite vertex of the first triangle.
    x3 : Eigen::Vector3d
        The opposite vertex of the second triangle.

    Returns
    -------
    double
        The bending angle between the two triangles.
    """

def dihedral_angle_gradient(
    x0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    x1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    x2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    x3: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
    """
    Compute the Jacobian of the bending angle between two triangles sharing an edge.
        x0---x2
         | \\ |
        x1---x3

    Parameters
    ----------
    x0 : Eigen::Vector3d
        The first vertex of the edge.
    x1 : Eigen::Vector3d
        The second vertex of the edge.
    x2 : Eigen::Vector3d
        The opposite vertex of the first triangle.
    x3 : Eigen::Vector3d
        The opposite vertex of the second triangle.

    Returns
    -------
    Eigen::Vector<double, 12>
        The Jacobian matrix of the bending angle with respect to the input vertices.
    """

def edge_edge_aabb_ccd(
    ea0_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea0_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dist: typing.SupportsFloat,
) -> bool: ...
def edge_edge_aabb_cd(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    dist: typing.SupportsFloat,
) -> bool: ...
def edge_edge_closest_point(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]:
    """
    Compute the barycentric coordinates of the closest points between two edges.

    Parameters:
        ea0: First point of the first edge
        ea1: Second point of the first edge
        eb0: First point of the second edge
        eb1: Second point of the second edge

    Returns:
        Barycentric coordinates of the closest points
    """

def edge_edge_closest_point_jacobian(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 12]"]:
    """
    Compute the Jacobian of the closest points between two edges.

    Parameters:
        ea0: First point of the first edge
        ea1: Second point of the first edge
        eb0: First point of the second edge
        eb1: Second point of the second edge

    Returns:
        Jacobian of the closest points
    """

def edge_edge_cross_squarednorm(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
    """
    Compute the squared norm of the edge-edge cross product.

    Parameters:
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.

    Returns:
        The squared norm of the edge-edge cross product.
    """

def edge_edge_cross_squarednorm_gradient(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
    """
    Compute the gradient of the squared norm of the edge cross product.

    Parameters:
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.

    Returns:
        The gradient of the squared norm of the edge cross product wrt ea0, ea1, eb0, and eb1.
    """

def edge_edge_cross_squarednorm_hessian(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 12]"]:
    """
    Compute the hessian of the squared norm of the edge cross product.

    Parameters:
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.

    Returns:
        The hessian of the squared norm of the edge cross product wrt ea0, ea1, eb0, and eb1.
    """

def edge_edge_distance(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dtype: EdgeEdgeDistanceType = ...,
) -> float:
    """
    Compute the distance between a two lines segments in 3D.

    Note:
        The distance is actually squared distance.

    Parameters:
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.
        dtype: The point edge distance type to compute.

    Returns:
        The distance between the two edges.
    """

def edge_edge_distance_gradient(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dtype: EdgeEdgeDistanceType = ...,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
    """
    Compute the gradient of the distance between a two lines segments.

    Note:
        The distance is actually squared distance.

    Parameters:
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.
        dtype: The point edge distance type to compute.

    Returns:
        The gradient of the distance wrt ea0, ea1, eb0, and eb1.
    """

def edge_edge_distance_hessian(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dtype: EdgeEdgeDistanceType = ...,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 12]"]:
    """
    Compute the hessian of the distance between a two lines segments.

    Note:
        The distance is actually squared distance.

    Parameters:
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.
        dtype: The point edge distance type to compute.

    Returns:
        The hessian of the distance wrt ea0, ea1, eb0, and eb1.
    """

def edge_edge_distance_type(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> EdgeEdgeDistanceType:
    """
    Determine the closest pair between two edges.

    Parameters:
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.

    Returns:
        The distance type of the edge-edge pair.
    """

@typing.overload
def edge_edge_mollifier(x: typing.SupportsFloat, eps_x: typing.SupportsFloat) -> float:
    """
    Mollifier function for edge-edge distance.

    Parameters:
        x: Squared norm of the edge-edge cross product.
        eps_x: Mollifier activation threshold.

    Returns:
        The mollifier coefficient to premultiply the edge-edge distance.
    """

@typing.overload
def edge_edge_mollifier(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eps_x: typing.SupportsFloat,
) -> float:
    """
    Compute a mollifier for the edge-edge distance.

    This helps smooth the non-smoothness at close to parallel edges.

    Parameters:
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.
        eps_x: Mollifier activation threshold.

    Returns:
        The mollifier coefficient to premultiply the edge-edge distance.
    """

def edge_edge_mollifier_derivative_wrt_eps_x(
    x: typing.SupportsFloat, eps_x: typing.SupportsFloat
) -> float:
    """
    The derivative of the mollifier function for edge-edge distance wrt eps_x.

    Parameters:
        x: Squared norm of the edge-edge cross product.
        eps_x: Mollifier activation threshold.

    Returns:
        The derivative of the mollifier function for edge-edge distance wrt eps_x.
    """

@typing.overload
def edge_edge_mollifier_gradient(
    x: typing.SupportsFloat, eps_x: typing.SupportsFloat
) -> float:
    """
    The gradient of the mollifier function for edge-edge distance.

    Parameters:
        x: Squared norm of the edge-edge cross product.
        eps_x: Mollifier activation threshold.

    Returns:
        The gradient of the mollifier function for edge-edge distance wrt x.
    """

@typing.overload
def edge_edge_mollifier_gradient(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eps_x: typing.SupportsFloat,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
    """
    Compute the gradient of the mollifier for the edge-edge distance.

    Parameters:
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.
        eps_x: Mollifier activation threshold.

    Returns:
        The gradient of the mollifier.
    """

def edge_edge_mollifier_gradient_derivative_wrt_eps_x(
    x: typing.SupportsFloat, eps_x: typing.SupportsFloat
) -> float:
    """
    The derivative of the gradient of the mollifier function for edge-edge distance wrt eps_x.

    Parameters:
        x: Squared norm of the edge-edge cross product.
        eps_x: Mollifier activation threshold.

    Returns:
        The derivative of the gradient of the mollifier function for edge-edge distance wrt eps_x.
    """

def edge_edge_mollifier_gradient_jacobian_wrt_x(
    ea0_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 12]"]:
    """
    Compute the jacobian of the edge-edge distance mollifier's gradient wrt rest positions.

    Note:
        This is not the hessian of the mollifier wrt rest positions, but the jacobian wrt rest positions of the mollifier's gradient wrt positions.

    Parameters:
        ea0_rest: The rest position of the first vertex of the first edge.
        ea1_rest: The rest position of the second vertex of the first edge.
        eb0_rest: The rest position of the first vertex of the second edge.
        eb1_rest: The rest position of the second vertex of the second edge.
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.

    Returns:
        The jacobian of the mollifier's gradient wrt rest positions.
    """

def edge_edge_mollifier_gradient_wrt_x(
    ea0_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
    """
    Compute the gradient of the mollifier for the edge-edge distance wrt rest positions.

    Parameters:
        ea0_rest: The rest position of the first vertex of the first edge.
        ea1_rest: The rest position of the second vertex of the first edge.
        eb0_rest: The rest position of the first vertex of the second edge.
        eb1_rest: The rest position of the second vertex of the second edge.
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.

    Returns:
        The derivative of the mollifier wrt rest positions.
    """

@typing.overload
def edge_edge_mollifier_hessian(
    x: typing.SupportsFloat, eps_x: typing.SupportsFloat
) -> float:
    """
    The hessian of the mollifier function for edge-edge distance.

    Parameters:
        x: Squared norm of the edge-edge cross product.
        eps_x: Mollifier activation threshold.

    Returns:
        The hessian of the mollifier function for edge-edge distance wrt x.
    """

@typing.overload
def edge_edge_mollifier_hessian(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eps_x: typing.SupportsFloat,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 12]"]:
    """
    Compute the hessian of the mollifier for the edge-edge distance.

    Parameters:
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.
        eps_x: Mollifier activation threshold.

    Returns:
        The hessian of the mollifier.
    """

def edge_edge_mollifier_threshold(
    ea0_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
    """
    Compute the threshold of the mollifier edge-edge distance.

    This values is computed based on the edges at rest length.

    Parameters:
        ea0_rest: The rest position of the first vertex of the first edge.
        ea1_rest: The rest position of the second vertex of the first edge.
        eb0_rest: The rest position of the first vertex of the second edge.
        eb1_rest: The rest position of the second vertex of the second edge.

    Returns:
        Threshold for edge-edge mollification.
    """

def edge_edge_mollifier_threshold_gradient(
    ea0_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1_rest: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
    """
    Compute the gradient of the threshold of the mollifier edge-edge distance.

    This values is computed based on the edges at rest length.

    Parameters:
        ea0_rest: The rest position of the first vertex of the first edge.
        ea1_rest: The rest position of the second vertex of the first edge.
        eb0_rest: The rest position of the first vertex of the second edge.
        eb1_rest: The rest position of the second vertex of the second edge.

    Returns:
        Gradient of the threshold for edge-edge mollification.
    """

def edge_edge_parallel_distance_type(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> EdgeEdgeDistanceType:
    """
    Determine the closest pair between two parallel edges.

    Parameters:
        ea0: The first vertex of the first edge.
        ea1: The second vertex of the first edge.
        eb0: The first vertex of the second edge.
        eb1: The second vertex of the second edge.

    Returns:
        The distance type of the edge-edge pair.
    """

def edge_edge_relative_velocity(
    dea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    deb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    deb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    coords: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Compute the relative velocity of the edges.

    Parameters:
        dea0: Velocity of the first endpoint of the first edge
        dea1: Velocity of the second endpoint of the first edge
        deb0: Velocity of the first endpoint of the second edge
        deb1: Velocity of the second endpoint of the second edge
        coords: Two parametric coordinates of the closest points on the edges

    Returns:
        The relative velocity of the edges
    """

def edge_edge_relative_velocity_dx_dbeta(
    coords: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[36, 2]"]:
    """
    Compute the Jacobian of the edge-edge relative velocity matrix.

    Parameters:
        coords: Two parametric coordinates of the closest points on the edges

    Returns:
        The Jacobian of the relative velocity matrix
    """

def edge_edge_relative_velocity_jacobian(
    coords: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 12]"]:
    """
    Compute the edge-edge relative velocity matrix.

    Parameters:
        coords: Two parametric coordinates of the closest points on the edges

    Returns:
        The relative velocity matrix
    """

def edge_edge_tangent_basis(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 2]"]:
    """
    Compute a basis for the space tangent to the edge-edge pair.

    Parameters:
        ea0: First point of the first edge
        ea1: Second point of the first edge
        eb0: First point of the second edge
        eb1: Second point of the second edge

    Returns:
        A 3x2 matrix whose columns are the basis vectors.
    """

def edge_edge_tangent_basis_jacobian(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 12]"]:
    """
    Compute the Jacobian of the tangent basis for the edge-edge pair.

    Parameters:
        ea0: First point of the first edge
        ea1: Second point of the first edge
        eb0: First point of the second edge
        eb1: Second point of the second edge

    Returns:
        A 12*3x2 matrix whose columns are the basis vectors.
    """

def edge_length_gradient(
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the gradient of an edge's length.

    Parameters:
        e0: The first vertex of the edge.
        e1: The second vertex of the edge.

    Returns:
        The gradient of the edge's length wrt e0, and e1.
    """

def edge_triangle_aabb_cd(
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dist: typing.SupportsFloat,
) -> bool: ...
def edges(
    F: typing.Annotated[
        numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
    ],
) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, n]"]:
    """
    Constructs a list of unique edges represented in a given mesh F

    Parameters:
        F: #F by 3 list of mesh faces (must be triangles)

    Returns:
        #E by 2 list of edges in no particular order
    """

def get_num_threads() -> int:
    """
    get maximum number of threads to use
    """

def has_intersections(
    mesh: CollisionMesh,
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    broad_phase: BroadPhase = None,
) -> bool:
    """
    Determine if the mesh has self intersections.

    Parameters:
        mesh: The collision mesh.
        vertices: Vertices of the collision mesh.
        broad_phase: Broad phase to use.

    Returns:
        A boolean for if the mesh has intersections.
    """

def inexact_point_edge_ccd_2D(
    p_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    e0_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    e1_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    p_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    e0_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    e1_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    conservative_rescaling: typing.SupportsFloat,
) -> tuple[bool, float]:
    """
    Inexact continuous collision detection between a point and an edge in 2D.

    Parameters:
        p_t0: Initial position of the point
        e0_t0: Initial position of the first endpoint of the edge
        e1_t0: Initial position of the second endpoint of the edge
        p_t1: Final position of the point
        e0_t1: Final position of the first endpoint of the edge
        e1_t1: Final position of the second endpoint of the edge
        conservative_rescaling: Conservative rescaling of the time of impact

    Returns:
        Tuple of:
        True if a collision was detected, false otherwise.
        Output time of impact
    """

def initial_barrier_stiffness(
    bbox_diagonal: typing.SupportsFloat,
    barrier: Barrier,
    dhat: typing.SupportsFloat,
    average_mass: typing.SupportsFloat,
    grad_energy: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    grad_barrier: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    min_barrier_stiffness_scale: typing.SupportsFloat = 100000000000.0,
    dmin: typing.SupportsFloat = 0,
) -> tuple[float, float]:
    """
    Compute an inital barrier stiffness using the barrier potential gradient.

    Parameters:
        bbox_diagonal: Length of the diagonal of the bounding box of the scene.
        barrier: Barrier function.
        dhat: Activation distance of the barrier.
        average_mass: Average mass of all bodies.
        grad_energy: Gradient of the elasticity energy function.
        grad_barrier: Gradient of the barrier potential.
        min_barrier_stiffness_scale: Scale used to premultiply the minimum barrier stiffness.
        dmin: Minimum distance between elements.

    Returns:
        Tuple of:
        The initial barrier stiffness.
        Maximum stiffness of the barrier.
    """

def is_edge_intersecting_triangle(
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> bool: ...
def is_step_collision_free(
    mesh: CollisionMesh,
    vertices_t0: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    vertices_t1: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    min_distance: typing.SupportsFloat = 0.0,
    broad_phase: BroadPhase = None,
    narrow_phase_ccd: NarrowPhaseCCD = ...,
) -> bool:
    """
    Determine if the step is collision free.

    Note:
        Assumes the trajectory is linear.

    Parameters:
        mesh: The collision mesh.
        vertices_t0: Surface vertex vertices at start as rows of a matrix.
        vertices_t1: Surface vertex vertices at end as rows of a matrix.
        min_distance: The minimum distance allowable between any two elements.
        broad_phase: Broad phase to use.
        narrow_phase_ccd: The narrow phase CCD algorithm to use.

    Returns:
        True if <b>any</b> collisions occur.
    """

def line_line_distance(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
    """
    Compute the distance between a two infinite lines in 3D.

    Note:
        The distance is actually squared distance.

    Warning:
        If the lines are parallel this function returns a distance of zero.

    Parameters:
        ea0: The first vertex of the edge defining the first line.
        ea1: The second vertex of the edge defining the first line.
        eb0: The first vertex of the edge defining the second line.
        eb1: The second vertex of the edge defining the second line.

    Returns:
        The distance between the two lines.
    """

def line_line_distance_gradient(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
    """
    Compute the gradient of the distance between a two lines in 3D.

    Note:
        The distance is actually squared distance.

    Warning:
        If the lines are parallel this function returns a distance of zero.

    Parameters:
        ea0: The first vertex of the edge defining the first line.
        ea1: The second vertex of the edge defining the first line.
        eb0: The first vertex of the edge defining the second line.
        eb1: The second vertex of the edge defining the second line.

    Returns:
        The gradient of the distance wrt ea0, ea1, eb0, and eb1.
    """

def line_line_distance_hessian(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 12]"]:
    """
    Compute the hessian of the distance between a two lines in 3D.

    Note:
        The distance is actually squared distance.

    Warning:
        If the lines are parallel this function returns a distance of zero.

    Parameters:
        ea0: The first vertex of the edge defining the first line.
        ea1: The second vertex of the edge defining the first line.
        eb0: The first vertex of the edge defining the second line.
        eb1: The second vertex of the edge defining the second line.

    Returns:
        The hessian of the distance wrt ea0, ea1, eb0, and eb1.
    """

def line_line_normal(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Computes the normal vector of two lines.

    Parameters
    ----------
    ea0: The first vertex of the first line.
    ea1: The second vertex of the first line.
    eb0: The first vertex of the second line.
    eb1: The second vertex of the second line.

    Returns
    -------
    The normal vector of the two lines.
    """

def line_line_normal_hessian(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[36, 12]"]:
    """
    Computes the Hessian of the normal vector of two lines.

    Parameters
    ----------
    ea0: The first vertex of the first line.
    ea1: The second vertex of the first line.
    eb0: The first vertex of the second line.
    eb1: The second vertex of the second line.

    Returns
    -------
    The Hessian of the normal vector of the two lines.
    """

def line_line_normal_jacobian(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 12]"]:
    """
    Computes the Jacobian of the normal vector of two lines.

    Parameters
    ----------
    ea0: The first vertex of the first line.
    ea1: The second vertex of the first line.
    eb0: The first vertex of the second line.
    eb1: The second vertex of the second line.

    Returns
    -------
    The Jacobian of the normal vector of the two lines.
    """

def line_line_signed_distance(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
    """
    Compute the signed distance between two lines in 3D.

    Parameters:
        ea0, ea1: Two points on the first line.
        eb0, eb1: Two points on the second line.

    Returns:
        The signed distance along the common normal between the two lines.
    """

def line_line_signed_distance_gradient(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
    """
    Compute the gradient of the signed line-line distance (3D).

    Returns a 12-vector ordered as [d/d(ea0); d/d(ea1); d/d(eb0); d/d(eb1)].
    """

def line_line_signed_distance_hessian(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 12]"]:
    """
    Compute the Hessian of the signed line-line distance (3D).

    Returns a 12x12 Hessian matrix with variables ordered as [ea0, ea1, eb0, eb1].
    """

def line_line_unnormalized_normal(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Computes the unnormalized normal vector of two lines.

    Parameters
    ----------
    ea0: The first vertex of the first line.
    ea1: The second vertex of the first line.
    eb0: The first vertex of the second line.
    eb1: The second vertex of the second line.

    Returns
    -------
    The unnormalized normal vector of the two lines.
    """

def line_line_unnormalized_normal_hessian(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[36, 12]"]:
    """
    Computes the Hessian of the unnormalized normal vector of two lines.

    Parameters
    ----------
    ea0: The first vertex of the first line.
    ea1: The second vertex of the first line.
    eb0: The first vertex of the second line.
    eb1: The second vertex of the second line.

    Returns
    -------
    The Hessian of the unnormalized normal vector of the two lines.
    """

def line_line_unnormalized_normal_jacobian(
    ea0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    ea1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    eb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 12]"]:
    """
    Computes the Jacobian of the unnormalized normal vector of two lines.

    Parameters
    ----------
    ea0: The first vertex of the first line.
    ea1: The second vertex of the first line.
    eb0: The first vertex of the second line.
    eb1: The second vertex of the second line.

    Returns
    -------
    The Jacobian of the unnormalized normal vector of the two lines.
    """

def make_connected_components_filter(
    faces: typing.Annotated[
        numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
    ],
) -> CollisionFilter:
    """
    Create a filter that prevents self-collisions within a connected
    component of the face mesh. Two vertices in the same connected
    component are blocked; cross-component pairs are allowed.

    Parameters:
        faces: Face index matrix (#F × 3).

    Returns:
        A CollisionFilter that blocks intra-component pairs.
    """

def make_sparse_filter(
    explicit_values: collections.abc.Mapping[
        tuple[typing.SupportsInt, typing.SupportsInt], bool
    ],
    default_value: bool,
) -> CollisionFilter:
    """
    Create a filter from a sparse map of explicit vertex-pair values.

    Pairs present in ``explicit_values`` use the stored boolean; all
    other pairs fall back to ``default_value``.  Only the upper triangle
    of the pair space is used — keys must satisfy ``i < j``.

    Parameters:
        explicit_values: Dict mapping ``(i, j)`` pairs (``i < j``) to
            whether those two vertices can collide.
        default_value: Value returned for pairs not in the map.

    Returns:
        A CollisionFilter backed by the sparse map.
    """

def make_static_obstacle_filter(n_dynamic: typing.SupportsInt) -> CollisionFilter:
    """
    Create a filter that prevents static obstacles from colliding with each other.
    A vertex is considered "static" if its index is >= n_dynamic.
    Pairs where both vertices are static are rejected.

    Parameters:
        n_dynamic: Number of dynamic (simulated) vertices; static vertices occupy indices [n_dynamic, n_verts).

    Returns:
        A CollisionFilter that blocks static-static pairs.
    """

def make_vertex_patches_filter(
    patch_ids: typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, 1]"],
) -> CollisionFilter:
    """
    Create a filter that only allows collisions between vertices in different patches (e.g., different garment panels or bodies).

    Parameters:
        patch_ids: Per-vertex patch label vector (one entry per vertex).

    Returns:
        A CollisionFilter that blocks same-patch pairs.
    """

def max_displacement_length(
    displacements: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
) -> float:
    """
    Compute the maximum displacement length.
    """

def max_edge_length(
    vertices_t0: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    vertices_t1: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    edges: typing.Annotated[
        numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
    ],
) -> float:
    """
    Compute the maximum edge length of a mesh.
    """

def max_normal_adhesion_force_magnitude(
    dhat_p: typing.SupportsFloat, dhat_a: typing.SupportsFloat, a2: typing.SupportsFloat
) -> float:
    """
    The maximum normal adhesion force magnitude.

    Parameters:
        dhat_p: distance of largest adhesion force (:math:`\\hat{d}_p`) where :math:`0 < \\hat{d}_p < \\hat{d}_a`
        dhat_a: adhesion activation distance (:math:`\\hat{d}_a`)
        a2: adjustable parameter relating to the maximum derivative of a (:math:`a_2`)

    Returns:
        The maximum normal adhesion force magnitude.
    """

def mean_displacement_length(
    displacements: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
) -> tuple[float, float]:
    """
    Compute the average displacement length.
    """

def mean_edge_length(
    vertices_t0: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    vertices_t1: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    edges: typing.Annotated[
        numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
    ],
) -> tuple[float, float]:
    """
    Compute the average edge length of a mesh.
    """

def median_displacement_length(
    displacements: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
) -> float:
    """
    Compute the median displacement length.
    """

def median_edge_length(
    vertices_t0: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    vertices_t1: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    edges: typing.Annotated[
        numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
    ],
) -> float:
    """
    Compute the median edge length of a mesh.
    """

def normal_adhesion_potential(
    d: typing.SupportsFloat,
    dhat_p: typing.SupportsFloat,
    dhat_a: typing.SupportsFloat,
    a2: typing.SupportsFloat,
) -> float:
    """
    The normal adhesion potential.

    Parameters:
        d: distance
        dhat_p: distance of largest adhesion force (:math:`\\hat{d}_p`) where :math:`0 < \\hat{d}_p < \\hat{d}_a`
        dhat_a: adhesion activation distance (:math:`\\hat{d}_a`)
        a2: adjustable parameter relating to the maximum derivative of a (:math:`a_2`)

    Returns:
        The normal adhesion potential.
    """

def normal_adhesion_potential_first_derivative(
    d: typing.SupportsFloat,
    dhat_p: typing.SupportsFloat,
    dhat_a: typing.SupportsFloat,
    a2: typing.SupportsFloat,
) -> float:
    """
    The first derivative of the normal adhesion potential wrt d.

    Parameters:
        d: distance
        dhat_p: distance of largest adhesion force (:math:`\\hat{d}_p`) where :math:`0 < \\hat{d}_p < \\hat{d}_a`
        dhat_a: adhesion activation distance (:math:`\\hat{d}_a`)
        a2: adjustable parameter relating to the maximum derivative of a (:math:`a_2`)

    Returns:
        The first derivative of the normal adhesion potential wrt d.
    """

def normal_adhesion_potential_second_derivative(
    d: typing.SupportsFloat,
    dhat_p: typing.SupportsFloat,
    dhat_a: typing.SupportsFloat,
    a2: typing.SupportsFloat,
) -> float:
    """
    The second derivative of the normal adhesion potential wrt d.

    Parameters:
        d: distance
        dhat_p: distance of largest adhesion force (:math:`\\hat{d}_p`) where :math:`0 < \\hat{d}_p < \\hat{d}_a`
        dhat_a: adhesion activation distance (:math:`\\hat{d}_a`)
        a2: adjustable parameter relating to the maximum derivative of a (:math:`a_2`)

    Returns:
        The second derivative of the normal adhesion potential wrt d.
    """

def normalization_and_jacobian(
    x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> tuple[
    typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"],
]:
    """
    Computes the normalization and Jacobian of a vector.

    Parameters
    ----------
    x: The input vector.

    Returns
    -------
    A tuple containing the normalized vector and its Jacobian.
    """

def normalization_and_jacobian_and_hessian(
    x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> tuple[
    typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"],
    typing.Annotated[
        list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]],
        "FixedSize(3)",
    ],
]:
    """
    Computes the normalization, Jacobian, and Hessian of a vector.

    Parameters
    ----------
    x: The input vector.

    Returns
    -------
    A tuple containing the normalized vector, its Jacobian, and its Hessian.
    """

def normalization_hessian(
    x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[
    list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]],
    "FixedSize(3)",
]:
    """
    Computes the Hessian of the normalization operation.

    Parameters
    ----------
    x: The input vector.

    Returns
    -------
    The Hessian of the normalization operation.
    """

def normalization_jacobian(
    x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Computes the Jacobian of the normalization operation.

    Parameters
    ----------
    x: The input vector.

    Returns
    -------
    The Jacobian of the normalization operation.
    """

def point_edge_aabb_ccd(
    p_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    p_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    dist: typing.SupportsFloat,
) -> bool: ...
def point_edge_aabb_cd(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    dist: typing.SupportsFloat,
) -> bool: ...
def point_edge_closest_point(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> float:
    """
    Compute the barycentric coordinate of the closest point on the edge.

    Parameters:
        p: Point
        e0: First edge point
        e1: Second edge point

    Returns:
        barycentric coordinates of the closest point
    """

def point_edge_closest_point_jacobian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the Jacobian of the closest point on the edge.

    Parameters:
        p: Point
        e0: First edge point
        e1: Second edge point

    Returns:
        Jacobian of the closest point
    """

def point_edge_distance(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    dtype: PointEdgeDistanceType = ...,
) -> float:
    """
    Compute the distance between a point and edge in 2D or 3D.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        e0: The first vertex of the edge.
        e1: The second vertex of the edge.
        dtype: The point edge distance type to compute.

    Returns:
        The distance between the point and edge.
    """

def point_edge_distance_gradient(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    dtype: PointEdgeDistanceType = ...,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the gradient of the distance between a point and edge.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        e0: The first vertex of the edge.
        e1: The second vertex of the edge.
        dtype: The point edge distance type to compute.

    Returns:
        grad The gradient of the distance wrt p, e0, and e1.
    """

def point_edge_distance_hessian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    dtype: PointEdgeDistanceType = ...,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Compute the hessian of the distance between a point and edge.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        e0: The first vertex of the edge.
        e1: The second vertex of the edge.
        dtype: The point edge distance type to compute.

    Returns:
        hess The hessian of the distance wrt p, e0, and e1.
    """

def point_edge_distance_type(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> PointEdgeDistanceType:
    """
    Determine the closest pair between a point and edge.

    Parameters:
        p: The point.
        e0: The first vertex of the edge.
        e1: The second vertex of the edge.

    Returns:
        The distance type of the point-edge pair.
    """

def point_edge_relative_velocity(
    dp: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    de0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    de1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    alpha: typing.SupportsFloat,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the relative velocity of a point and an edge

    Parameters:
        dp: Velocity of the point
        de0: Velocity of the first endpoint of the edge
        de1: Velocity of the second endpoint of the edge
        alpha: Parametric coordinate of the closest point on the edge

    Returns:
        The relative velocity of the point and the edge
    """

def point_edge_relative_velocity_dx_dbeta(
    dim: typing.SupportsInt, alpha: typing.SupportsFloat
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the Jacobian of the relative velocity premultiplier matrix

    Parameters:
        dim: Dimension (2 or 3)
        alpha: Parametric coordinate of the closest point on the edge

    Returns:
        The Jacobian of the relative velocity premultiplier matrix
    """

def point_edge_relative_velocity_jacobian(
    dim: typing.SupportsInt, alpha: typing.SupportsFloat
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Compute the point-edge relative velocity premultiplier matrix

    Parameters:
        dim: Dimension (2 or 3)
        alpha: Parametric coordinate of the closest point on the edge

    Returns:
        The relative velocity premultiplier matrix
    """

def point_edge_tangent_basis(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Compute a basis for the space tangent to the point-edge pair.

    Parameters:
        p: Point
        e0: First edge point
        e1: Second edge point

    Returns:
        A 3x2 matrix whose columns are the basis vectors.
    """

def point_edge_tangent_basis_jacobian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Compute the Jacobian of the tangent basis for the point-edge pair.

    Parameters:
        p: Point
        e0: First edge point
        e1: Second edge point

    Returns:
        A 9*3x2 matrix whose columns are the basis vectors.
    """

def point_line_distance(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> float:
    """
    Compute the distance between a point and line in 2D or 3D.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        e0: The first vertex of the edge defining the line.
        e1: The second vertex of the edge defining the line.

    Returns:
        The distance between the point and line.
    """

def point_line_distance_gradient(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the gradient of the distance between a point and line.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        e0: The first vertex of the edge defining the line.
        e1: The second vertex of the edge defining the line.

    Returns:
        The gradient of the distance wrt p, e0, and e1.
    """

def point_line_distance_hessian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Compute the hessian of the distance between a point and line.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        e0: The first vertex of the edge defining the line.
        e1: The second vertex of the edge defining the line.

    Returns:
        The hessian of the distance wrt p, e0, and e1.
    """

def point_line_normal(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Computes the normal vector of a point-line pair.

    Parameters
    ----------
    p: The point's position.
    e0: The start position of the line.
    e1: The end position of the line.

    Returns
    -------
    The normal vector.
    """

def point_line_signed_distance(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
) -> float:
    """
    Compute the signed distance from a point to a directed line segment (2D).

    Parameters:
        p: The query point (2D).
        e0: The first endpoint of the directed edge (2D).
        e1: The second endpoint of the directed edge (2D).

    Returns:
        The signed scalar distance from `p` to the (infinite) line through `e0` and `e1`.
    """

def point_line_signed_distance_gradient(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]"]:
    """
    Compute the gradient of the signed point-to-line distance (2D).

    Returns a 6-vector ordered as [dp, de0, de1].
    """

def point_line_signed_distance_hessian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 6]"]:
    """
    Compute the Hessian of the signed point-to-line distance (2D).

    Returns a 6x6 Hessian matrix with variables ordered as [p, e0, e1].
    """

def point_line_unnormalized_normal(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Computes the unnormalized normal vector of a point-line pair.

    Parameters
    ----------
    p: The point's position.
    e0: The start position of the line.
    e1: The end position of the line.

    Returns
    -------
    The unnormalized normal vector.
    """

def point_line_unnormalized_normal_jacobian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    e1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Computes the Jacobian of the unnormalized normal vector of a point-line pair.

    Parameters
    ----------
    p: The point's position.
    e0: The start position of the line.
    e1: The end position of the line.

    Returns
    -------
    The Jacobian of the unnormalized normal vector of the point-line pair.
    """

@typing.overload
def point_plane_distance(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    origin: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    normal: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
    """
    Compute the distance between a point and a plane.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        origin: The origin of the plane.
        normal: The normal of the plane.

    Returns:
        The distance between the point and plane.
    """

@typing.overload
def point_plane_distance(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
    """
    Compute the distance between a point and a plane.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        t0: The first vertex of the triangle.
        t1: The second vertex of the triangle.
        t2: The third vertex of the triangle.

    Returns:
        The distance between the point and plane.
    """

@typing.overload
def point_plane_distance_gradient(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    origin: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    normal: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Compute the gradient of the distance between a point and a plane.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        origin: The origin of the plane.
        normal: The normal of the plane.

    Returns:
        The gradient of the distance wrt p.
    """

@typing.overload
def point_plane_distance_gradient(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
    """
    Compute the gradient of the distance between a point and a plane.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        t0: The first vertex of the triangle.
        t1: The second vertex of the triangle.
        t2: The third vertex of the triangle.

    Returns:
        The gradient of the distance wrt p, t0, t1, and t2.
    """

@typing.overload
def point_plane_distance_hessian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    origin: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    normal: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 3]"]:
    """
    Compute the hessian of the distance between a point and a plane.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        origin: The origin of the plane.
        normal: The normal of the plane.

    Returns:
        The hessian of the distance wrt p.
    """

@typing.overload
def point_plane_distance_hessian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 12]"]:
    """
    Compute the hessian of the distance between a point and a plane.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        t0: The first vertex of the triangle.
        t1: The second vertex of the triangle.
        t2: The third vertex of the triangle.

    Returns:
        The hessian of the distance wrt p, t0, t1, and t2.
    """

def point_plane_signed_distance(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
    """
    Compute the signed distance from a point to the plane of a triangle (3D).

    Parameters:
        p: The query point (3D).
        t0: First vertex of the triangle (3D).
        t1: Second vertex of the triangle (3D).
        t2: Third vertex of the triangle (3D).

    Returns:
        The signed distance from `p` to the triangle plane.
    """

def point_plane_signed_distance_gradient(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
    """
    Compute the gradient of the signed point-to-plane distance (3D).

    Returns a 12-vector ordered as [dp, dt0, dt1, dt2].
    """

def point_plane_signed_distance_hessian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 12]"]:
    """
    Compute the Hessian of the signed point-to-plane distance (3D).

    Returns a 12x12 Hessian matrix with variables ordered as [p, t0, t1, t2].
    """

def point_point_distance(
    p0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    p1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> float:
    """
    Compute the distance between two points.

    Note:
        The distance is actually squared distance.

    Parameters:
        p0: The first point.
        p1: The second point.

    Returns:
        The distance between p0 and p1.
    """

def point_point_distance_gradient(
    p0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    p1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the gradient of the distance between two points.

    Note:
        The distance is actually squared distance.

    Parameters:
        p0: The first point.
        p1: The second point.

    Returns:
        The computed gradient.
    """

def point_point_distance_hessian(
    p0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    p1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Compute the hessian of the distance between two points.

    Note:
        The distance is actually squared distance.

    Parameters:
        p0: The first point.
        p1: The second point.

    Returns:
        The computed hessian.
    """

def point_point_relative_velocity(
    dp0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    dp1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the relative velocity of two points

    Parameters:
        dp0: Velocity of the first point
        dp1: Velocity of the second point

    Returns:
        The relative velocity of the two points
    """

def point_point_relative_velocity_dx_dbeta(
    dim: typing.SupportsInt,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the Jacobian of the relative velocity premultiplier matrix

    Parameters:
        dim: Dimension (2 or 3)

    Returns:
        The Jacobian of the relative velocity premultiplier matrix
    """

def point_point_relative_velocity_jacobian(
    dim: typing.SupportsInt,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Compute the point-point relative velocity premultiplier matrix

    Parameters:
        dim: Dimension (2 or 3)

    Returns:
        The relative velocity premultiplier matrix
    """

def point_point_tangent_basis(
    p0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    p1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Compute a basis for the space tangent to the point-point pair.

    Parameters:
        p0: First point
        p1: Second point

    Returns:
        A 3x2 matrix whose columns are the basis vectors.
    """

def point_point_tangent_basis_jacobian(
    p0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    p1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Compute the Jacobian of the tangent basis for the point-point pair.

    Parameters:
        p0: First point
        p1: Second point

    Returns:
        A 6*3x2 matrix whose columns are the basis vectors.
    """

def point_static_plane_ccd(
    p_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    p_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    plane_origin: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    plane_normal: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    conservative_rescaling: typing.SupportsFloat = 0.8,
) -> tuple[bool, float]:
    """
    Compute the time of impact between a point and a static plane in 3D using continuous collision detection.

    Parameters:
        p_t0: The initial position of the point.
        p_t1: The final position of the point.
        plane_origin: The origin of the plane.
        plane_normal: The normal of the plane.
        conservative_rescaling: Conservative rescaling of the time of impact.

    Returns:
        Tuple of:
        True if a collision was detected, false otherwise.
        Output time of impact
    """

def point_triangle_aabb_ccd(
    p_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2_t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    p_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2_t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dist: typing.SupportsFloat,
) -> bool: ...
def point_triangle_aabb_cd(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dist: typing.SupportsFloat,
) -> bool: ...
def point_triangle_closest_point(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]:
    """
    Compute the barycentric coordinates of the closest point on the triangle.

    Parameters:
        p: Point
        t0: Triangle's first vertex
        t1: Triangle's second vertex
        t2: Triangle's third vertex

    Returns:
        Barycentric coordinates of the closest point
    """

def point_triangle_closest_point_jacobian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 12]"]:
    """
    Compute the Jacobian of the closest point on the triangle.

    Parameters:
        p: Point
        t0: Triangle's first vertex
        t1: Triangle's second vertex
        t2: Triangle's third vertex

    Returns:
        Jacobian of the closest point
    """

def point_triangle_distance(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dtype: PointTriangleDistanceType = ...,
) -> float:
    """
    Compute the distance between a points and a triangle.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        t0: The first vertex of the triangle.
        t1: The second vertex of the triangle.
        t2: The third vertex of the triangle.
        dtype: The point-triangle distance type to compute.

    Returns:
        The distance between the point and triangle.
    """

def point_triangle_distance_gradient(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dtype: PointTriangleDistanceType = ...,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 1]"]:
    """
    Compute the gradient of the distance between a points and a triangle.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        t0: The first vertex of the triangle.
        t1: The second vertex of the triangle.
        t2: The third vertex of the triangle.
        dtype: The point-triangle distance type to compute.

    Returns:
        The gradient of the distance wrt p, t0, t1, and t2.
    """

def point_triangle_distance_hessian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dtype: PointTriangleDistanceType = ...,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[12, 12]"]:
    """
    Compute the hessian of the distance between a points and a triangle.

    Note:
        The distance is actually squared distance.

    Parameters:
        p: The point.
        t0: The first vertex of the triangle.
        t1: The second vertex of the triangle.
        t2: The third vertex of the triangle.
        dtype: The point-triangle distance type to compute.

    Returns:
        The hessian of the distance wrt p, t0, t1, and t2.
    """

def point_triangle_distance_type(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> PointTriangleDistanceType:
    """
    Determine the closest pair between a point and triangle.

    Parameters:
        p: The point.
        t0: The first vertex of the triangle.
        t1: The second vertex of the triangle.
        t2: The third vertex of the triangle.

    Returns:
        The distance type of the point-triangle pair.
    """

def point_triangle_relative_velocity(
    dp: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dt0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dt1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    dt2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    coords: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Compute the relative velocity of the point to the triangle.

    Parameters:
        dp: Velocity of the point
        dt0: Velocity of the first vertex of the triangle
        dt1: Velocity of the second vertex of the triangle
        dt2: Velocity of the third vertex of the triangle
        coords: Barycentric coordinates of the closest point on the triangle

    Returns:
        The relative velocity of the point to the triangle
    """

def point_triangle_relative_velocity_dx_dbeta(
    coords: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[36, 2]"]:
    """
    Compute the Jacobian of the point-triangle relative velocity matrix.

    Parameters:
        coords: Barycentric coordinates of the closest point on the triangle

    Returns:
        The Jacobian of the relative velocity matrix
    """

def point_triangle_relative_velocity_jacobian(
    coords: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 12]"]:
    """
    Compute the point-triangle relative velocity matrix.

    Parameters:
        coords: Barycentric coordinates of the closest point on the triangle

    Returns:
        The relative velocity matrix
    """

def point_triangle_tangent_basis(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 2]"]:
    """
    Compute a basis for the space tangent to the point-triangle pair.

    .. math::

        \\begin{bmatrix}
        \\frac{t_1 - t_0}{\\|t_1 - t_0\\|} & \\frac{((t_1 - t_0)\\times(t_2 - t_0))
        \\times(t_1 - t_0)}{\\|((t_1 - t_0)\\times(t_2 - t_0))\\times(t_1 - t_0)\\|}
        \\end{bmatrix}

    Parameters:
        p: Point
        t0: Triangle's first vertex
        t1: Triangle's second vertex
        t2: Triangle's third vertex

    Returns:
        A 3x2 matrix whose columns are the basis vectors.
    """

def point_triangle_tangent_basis_jacobian(
    p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 12]"]:
    """
    Compute the Jacobian of the tangent basis for the point-triangle pair.

    Parameters:
        p: Point
        t0: Triangle's first vertex
        t1: Triangle's second vertex
        t2: Triangle's third vertex

    Returns:
        A 12*3x2 matrix whose columns are the basis vectors.
    """

def project_to_pd(
    A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"],
    eps: typing.SupportsFloat = 1e-08,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Matrix projection onto positive definite cone

    Parameters:
        A: Symmetric matrix to project

    Returns:
        Projected matrix
    """

def project_to_psd(
    A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"],
    method: PSDProjectionMethod = ...,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Matrix projection onto positive semi-definite cone

    Parameters:
        A: Symmetric matrix to project
        method: PSD projection method

    Returns:
        Projected matrix
    """

def segment_segment_intersect(
    A: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    B: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    C: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
    D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
) -> bool:
    """
    Given two segments in 2d test whether they intersect each other using predicates orient2d

    Parameters:
        A: 1st endpoint of segment 1
        B: 2st endpoint of segment 1
        C: 1st endpoint of segment 2
        D: 2st endpoint of segment 2

    Returns:
        true if they intersect
    """

@typing.overload
def semi_implicit_stiffness(
    stencil: ...,
    vertices: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    mass: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    local_hess: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    dmin: typing.SupportsFloat,
) -> float:
    """
    Compute the semi-implicit stiffness for a single collision.

    See [Ando 2024] for details.

    Parameters:
        stencil: Collision stencil.
        vertex_ids: Vertex indices of the collision.
        vertices: Vertex positions.
        mass: Vertex masses.
        local_hess: Local hessian of the elasticity energy function.
        dmin: Minimum distance between elements.

    Returns:
        The semi-implicit stiffness.
    """

@typing.overload
def semi_implicit_stiffness(
    mesh: ...,
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    collisions: ...,
    vertex_masses: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    hess: scipy.sparse.csc_matrix,
    dmin: typing.SupportsFloat,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the semi-implicit stiffness's for all collisions.

    See [Ando 2024] for details.

    Parameters:
        mesh: Collision mesh.
        vertices: Vertex positions.
        collisions: Normal collisions.
        vertex_masses: Lumped vertex masses.
        hess: Hessian of the elasticity energy function.
        dmin: Minimum distance between elements.

    Returns:
        The semi-implicit stiffness's.
    """

@typing.overload
def semi_implicit_stiffness(
    mesh: ...,
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    collisions: ...,
    vertex_masses: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    hess: scipy.sparse.csc_matrix,
    dmin: typing.SupportsFloat,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute the semi-implicit stiffness's for all collisions.

    See [Ando 2024] for details.

    Parameters:
        mesh: Collision mesh.
        vertices: Vertex positions.
        collisions: Collisions candidates.
        vertex_masses: Lumped vertex masses.
        hess: Hessian of the elasticity energy function.
        dmin: Minimum distance between elements.

    Returns:
        The semi-implicit stiffness's.
    """

def set_logger_level(level: LoggerLevel) -> None:
    """
    Set log level
    """

def set_num_threads(nthreads: typing.SupportsInt) -> None:
    """
    set maximum number of threads to use
    """

def smooth_friction_f0(y: typing.SupportsFloat, eps_v: typing.SupportsFloat) -> float:
    """
    Smooth friction mollifier function.

    .. math::

        f_0(y)= \\begin{cases}
        -\\frac{y^3}{3\\epsilon_v^2} + \\frac{y^2}{\\epsilon_v}
        + \\frac{\\epsilon_v}{3}, & |y| < \\epsilon_v
        \\newline
        y, & |y| \\geq \\epsilon_v
        \\end{cases}

    Parameters:
        y: The tangential relative speed.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The value of the mollifier function at y.
    """

def smooth_friction_f1(y: typing.SupportsFloat, eps_v: typing.SupportsFloat) -> float:
    """
    The first derivative of the smooth friction mollifier.

    .. math::

        f_1(y) = f_0'(y) = \\begin{cases}
        -\\frac{y^2}{\\epsilon_v^2}+\\frac{2 y}{\\epsilon_v}, & |y| < \\epsilon_v
        \\newline 1, & |y| \\geq \\epsilon_v
        \\end{cases}

    Parameters:
        y: The tangential relative speed.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The value of the derivative of the smooth friction mollifier at y.
    """

def smooth_friction_f1_over_x(
    y: typing.SupportsFloat, eps_v: typing.SupportsFloat
) -> float:
    """
    Compute the derivative of the smooth friction mollifier divided by y (:math:`\\frac{f_0'(y)}{y}`).

    .. math::

        \\frac{f_1(y)}{y} = \\begin{cases}
        -\\frac{y}{\\epsilon_v^2}+\\frac{2}{\\epsilon_v}, & |y| < \\epsilon_v
        \\newline \\frac{1}{y}, & |y| \\geq \\epsilon_v
        \\end{cases}

    Parameters:
        y: The tangential relative speed.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The value of the derivative of smooth_friction_f0 divided by y.
    """

def smooth_friction_f2(y: typing.SupportsFloat, eps_v: typing.SupportsFloat) -> float:
    """
    The second derivative of the smooth friction mollifier.

    .. math::

        f_2(y) = f_0''(y) = \\begin{cases}
        -\\frac{2 y}{\\epsilon_v^2}+\\frac{2}{\\epsilon_v}, & |y| < \\epsilon_v
        \\newline 0, & |y| \\geq \\epsilon_v
        \\end{cases}

    Parameters:
        y: The tangential relative speed.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The value of the second derivative of the smooth friction mollifier at y.
    """

def smooth_friction_f2_x_minus_f1_over_x3(
    y: typing.SupportsFloat, eps_v: typing.SupportsFloat
) -> float:
    """
    The derivative of f1 times y minus f1 all divided by y cubed.

    .. math::

        \\frac{f_1'(y) y - f_1(y)}{y^3} = \\begin{cases}
        -\\frac{1}{y \\epsilon_v^2}, & |y| < \\epsilon_v \\newline
        -\\frac{1}{y^3}, & |y| \\geq \\epsilon_v
        \\end{cases}

    Parameters:
        y: The tangential relative speed.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The derivative of f1 times y minus f1 all divided by y cubed.
    """

def smooth_mu(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_v: typing.SupportsFloat,
) -> float:
    """
    Smooth coefficient from static to kinetic friction.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static friction.
        mu_k: Coefficient of kinetic friction.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The value of the μ at y.
    """

def smooth_mu_a0(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_a: typing.SupportsFloat,
) -> float:
    """
    Compute the value of the ∫ μ(y) a₁(y) dy, where a₁ is the first derivative of the smooth tangential adhesion mollifier.

    Note:
        The `a0`/`a1` are unrelated to the `a0`/`a1` in the normal adhesion.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static adhesion.
        mu_k: Coefficient of kinetic adhesion.
        eps_a: Velocity threshold below which static adhesion force is applied.

    Returns:
        The value of the integral at y.
    """

def smooth_mu_a1(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_a: typing.SupportsFloat,
) -> float:
    """
    Compute the value of the μ(y) a₁(y), where a₁ is the first derivative of the smooth tangential adhesion mollifier.

    Note:
        The `a1` is unrelated to the `a1` in the normal adhesion.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static adhesion.
        mu_k: Coefficient of kinetic adhesion.
        eps_a: Velocity threshold below which static adhesion force is applied.

    Returns:
        The value of the product at y.
    """

def smooth_mu_a1_over_x(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_a: typing.SupportsFloat,
) -> float:
    """
    Compute the value of the μ(y) a₁(y) / y, where a₁ is the first derivative of the smooth tangential adhesion mollifier.

    Notes:
        The `x` in the function name refers to the parameter `y`.
        The `a1` is unrelated to the `a1` in the normal adhesion.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static adhesion.
        mu_k: Coefficient of kinetic adhesion.
        eps_a: Velocity threshold below which static adhesion force is applied.

    Returns:
        The value of the product at y.
    """

def smooth_mu_a2(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_a: typing.SupportsFloat,
) -> float:
    """
    Compute the value of d/dy (μ(y) a₁(y)), where a₁ is the first derivative of the smooth tangential adhesion mollifier.

    Note:
        The `a1`/`a2` are unrelated to the `a1`/`a2` in the normal adhesion.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static adhesion.
        mu_k: Coefficient of kinetic adhesion.
        eps_a: Velocity threshold below which static adhesion force is applied.

    Returns:
        The value of the derivative at y.
    """

def smooth_mu_a2_x_minus_mu_a1_over_x3(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_a: typing.SupportsFloat,
) -> float:
    """
    Compute the value of the [(d/dy μ(y) a₁(y)) ⋅ y - μ(y) a₁(y)] / y³, where a₁ and a₂ are the first and second derivatives of the smooth tangential adhesion mollifier.

    Notes:
        The `x` in the function name refers to the parameter `y`.
        The `a1`/`a2` are unrelated to the `a1`/`a2` in the normal adhesion.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static adhesion.
        mu_k: Coefficient of kinetic adhesion.
        eps_a: Velocity threshold below which static adhesion force is applied.

    Returns:
        The value of the expression at y.
    """

def smooth_mu_derivative(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_v: typing.SupportsFloat,
) -> float:
    """
    Compute the derivative of the smooth coefficient from static to kinetic friction.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static friction.
        mu_k: Coefficient of kinetic friction.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The value of the derivative at y.
    """

def smooth_mu_f0(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_v: typing.SupportsFloat,
) -> float:
    """
    Compute the value of the ∫ μ(y) f₁(y) dy, where f₁ is the first derivative of the smooth friction mollifier.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static friction.
        mu_k: Coefficient of kinetic friction.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The value of the integral at y.
    """

def smooth_mu_f1(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_v: typing.SupportsFloat,
) -> float:
    """
    Compute the value of the μ(y) f₁(y), where f₁ is the first derivative of the smooth friction mollifier.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static friction.
        mu_k: Coefficient of kinetic friction.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The value of the product at y.
    """

def smooth_mu_f1_over_x(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_v: typing.SupportsFloat,
) -> float:
    """
    Compute the value of the μ(y) f₁(y) / y, where f₁ is the first derivative of the smooth friction mollifier.

    Note:
        The `x` in the function name refers to the parameter `y`.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static friction.
        mu_k: Coefficient of kinetic friction.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The value of the product at y.
    """

def smooth_mu_f2(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_v: typing.SupportsFloat,
) -> float:
    """
    Compute the value of d/dy (μ(y) f₁(y)), where f₁ is the first derivative of the smooth friction mollifier.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static friction.
        mu_k: Coefficient of kinetic friction.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The value of the derivative at y.
    """

def smooth_mu_f2_x_minus_mu_f1_over_x3(
    y: typing.SupportsFloat,
    mu_s: typing.SupportsFloat,
    mu_k: typing.SupportsFloat,
    eps_v: typing.SupportsFloat,
) -> float:
    """
    Compute the value of the [(d/dy μ(y) f₁(y)) ⋅ y - μ(y) f₁(y)] / y³, where f₁ and f₂ are the first and second derivatives of the smooth friction mollifier.

    Note:
        The `x` in the function name refers to the parameter `y`.

    Parameters:
        y: The tangential relative speed.
        mu_s: Coefficient of static friction.
        mu_k: Coefficient of kinetic friction.
        eps_v: Velocity threshold below which static friction force is applied.

    Returns:
        The value of the expression at y.
    """

@typing.overload
def suggest_good_voxel_size(
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    edges: typing.Annotated[
        numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
    ],
    inflation_radius: typing.SupportsFloat = 0,
) -> float: ...
@typing.overload
def suggest_good_voxel_size(
    vertices_t0: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    vertices_t1: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    edges: typing.Annotated[
        numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
    ],
    inflation_radius: typing.SupportsFloat = 0,
) -> float: ...
def tangential_adhesion_f0(
    y: typing.SupportsFloat, eps_a: typing.SupportsFloat
) -> float:
    """
    The tangential adhesion mollifier function.

    Parameters:
        y: The tangential relative speed.
        eps_a: Velocity threshold below which static adhesion force is applied.

    Returns:
        The tangential adhesion mollifier function at y.
    """

def tangential_adhesion_f1(
    y: typing.SupportsFloat, eps_a: typing.SupportsFloat
) -> float:
    """
    The first derivative of the tangential adhesion mollifier function.

    Parameters:
        y: The tangential relative speed.
        eps_a: Velocity threshold below which static adhesion force is applied.

    Returns:
        The first derivative of the tangential adhesion mollifier function at y.
    """

def tangential_adhesion_f1_over_x(
    y: typing.SupportsFloat, eps_a: typing.SupportsFloat
) -> float:
    """
    The first derivative of the tangential adhesion mollifier function divided by y.

    Parameters:
        y: The tangential relative speed.
        eps_a: Velocity threshold below which static adhesion force is applied.

    Returns:
        The first derivative of the tangential adhesion mollifier function divided by y.
    """

def tangential_adhesion_f2(
    y: typing.SupportsFloat, eps_a: typing.SupportsFloat
) -> float:
    """
    The second derivative of the tangential adhesion mollifier function.

    Parameters:
        y: The tangential relative speed.
        eps_a: Velocity threshold below which static adhesion force is applied.

    Returns:
        The second derivative of the tangential adhesion mollifier function at y.
    """

def tangential_adhesion_f2_x_minus_f1_over_x3(
    y: typing.SupportsFloat, eps_a: typing.SupportsFloat
) -> float:
    """
    The second derivative of the tangential adhesion mollifier function times y minus the first derivative all divided by y cubed.

    Parameters:
        y: The tangential relative speed.
        eps_a: Velocity threshold below which static adhesion force is applied.

    Returns:
        The second derivative of the tangential adhesion mollifier function times y minus the first derivative all divided by y cubed.
    """

def triangle_area_gradient(
    t0: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    t2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]"]:
    """
    Compute the gradient of the area of a triangle.

    Parameters:
        t0: The first vertex of the triangle.
        t1: The second vertex of the triangle.
        t2: The third vertex of the triangle.

    Returns:
        The gradient of the triangle's area t0, t1, and t2.
    """

def triangle_normal(
    a: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    b: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    c: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Computes the normal vector of a triangle.

    Parameters
    ----------
    a: The first vertex of the triangle.
    b: The second vertex of the triangle.
    c: The third vertex of the triangle.

    Returns
    -------
    The normal vector of the triangle.
    """

def triangle_normal_hessian(
    a: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    b: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    c: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[27, 9]"]:
    """
    Computes the Hessian of the normal vector of a triangle.

    Parameters
    ----------
    a: The first vertex of the triangle.
    b: The second vertex of the triangle.
    c: The third vertex of the triangle.

    Returns
    -------
    The Hessian of the normal vector of the triangle.
    """

def triangle_normal_jacobian(
    a: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    b: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    c: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 9]"]:
    """
    Computes the Jacobian of the normal vector of a triangle.

    Parameters
    ----------
    a: The first vertex of the triangle.
    b: The second vertex of the triangle.
    c: The third vertex of the triangle.

    Returns
    -------
    The Jacobian of the normal vector of the triangle.
    """

def triangle_unnormalized_normal(
    a: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    b: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    c: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Computes the unnormalized normal vector of a triangle.

    Parameters
    ----------
    a: The first vertex of the triangle.
    b: The second vertex of the triangle.
    c: The third vertex of the triangle.

    Returns
    -------
    The unnormalized normal vector of the triangle.
    """

def triangle_unnormalized_normal_hessian(
    a: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    b: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    c: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[27, 9]"]:
    """
    Computes the Hessian of the unnormalized normal vector of a triangle.

    Parameters
    ----------
    a: The first vertex of the triangle.
    b: The second vertex of the triangle.
    c: The third vertex of the triangle.

    Returns
    -------
    The Hessian of the unnormalized normal vector of the triangle.
    """

def triangle_unnormalized_normal_jacobian(
    a: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    b: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
    c: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 9]"]:
    """
    Computes the Jacobian of the unnormalized normal vector of a triangle.

    Parameters
    ----------
    a: The first vertex of the triangle.
    b: The second vertex of the triangle.
    c: The third vertex of the triangle.

    Returns
    -------
    The Jacobian of the unnormalized normal vector of the triangle.
    """

def update_barrier_stiffness(
    prev_min_distance: typing.SupportsFloat,
    min_distance: typing.SupportsFloat,
    max_barrier_stiffness: typing.SupportsFloat,
    barrier_stiffness: typing.SupportsFloat,
    bbox_diagonal: typing.SupportsFloat,
    dhat_epsilon_scale: typing.SupportsFloat = 1e-09,
    dmin: typing.SupportsFloat = 0,
) -> float:
    """
    Update the barrier stiffness if the distance is decreasing and less than dhat_epsilon_scale * diag.

    Parameters:
        prev_min_distance: Previous minimum distance between elements.
        min_distance: Current minimum distance between elements.
        max_barrier_stiffness: Maximum stiffness of the barrier.
        barrier_stiffness: Current barrier stiffness.
        bbox_diagonal: Length of the diagonal of the bounding box of the scene.
        dhat_epsilon_scale: Update if distance is less than this fraction of the diagonal.
        dmin: Minimum distance between elements.

    Returns:
        The updated barrier stiffness.
    """

def vertex_to_min_edge(
    num_vertices: typing.SupportsInt,
    edges: typing.Annotated[
        numpy.typing.NDArray[numpy.int32], "[m, n]", "flags.f_contiguous"
    ],
) -> list[int]: ...
def world_bbox_diagonal_length(
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
) -> float:
    """
    Compute the diagonal length of the world bounding box.

    Parameters:
        vertices: Vertex positions

    Returns:
        The diagonal length of the world bounding box.
    """

ABS: PSDProjectionMethod  # value = <PSDProjectionMethod.ABS: 2>
AUTO: EdgeEdgeDistanceType  # value = <EdgeEdgeDistanceType.AUTO: 9>
CLAMP: PSDProjectionMethod  # value = <PSDProjectionMethod.CLAMP: 1>
EA0_EB: EdgeEdgeDistanceType  # value = <EdgeEdgeDistanceType.EA0_EB: 6>
EA0_EB0: EdgeEdgeDistanceType  # value = <EdgeEdgeDistanceType.EA0_EB0: 0>
EA0_EB1: EdgeEdgeDistanceType  # value = <EdgeEdgeDistanceType.EA0_EB1: 1>
EA1_EB: EdgeEdgeDistanceType  # value = <EdgeEdgeDistanceType.EA1_EB: 7>
EA1_EB0: EdgeEdgeDistanceType  # value = <EdgeEdgeDistanceType.EA1_EB0: 2>
EA1_EB1: EdgeEdgeDistanceType  # value = <EdgeEdgeDistanceType.EA1_EB1: 3>
EA_EB: EdgeEdgeDistanceType  # value = <EdgeEdgeDistanceType.EA_EB: 8>
EA_EB0: EdgeEdgeDistanceType  # value = <EdgeEdgeDistanceType.EA_EB0: 4>
EA_EB1: EdgeEdgeDistanceType  # value = <EdgeEdgeDistanceType.EA_EB1: 5>
NONE: PSDProjectionMethod  # value = <PSDProjectionMethod.NONE: 0>
P_E: PointEdgeDistanceType  # value = <PointEdgeDistanceType.P_E: 2>
P_E0: PointTriangleDistanceType  # value = <PointTriangleDistanceType.P_E0: 3>
P_E1: PointTriangleDistanceType  # value = <PointTriangleDistanceType.P_E1: 4>
P_E2: PointTriangleDistanceType  # value = <PointTriangleDistanceType.P_E2: 5>
P_T: PointTriangleDistanceType  # value = <PointTriangleDistanceType.P_T: 6>
P_T0: PointTriangleDistanceType  # value = <PointTriangleDistanceType.P_T0: 0>
P_T1: PointTriangleDistanceType  # value = <PointTriangleDistanceType.P_T1: 1>
P_T2: PointTriangleDistanceType  # value = <PointTriangleDistanceType.P_T2: 2>
__version__: str = "1.6.0"
critical: LoggerLevel  # value = <LoggerLevel.critical: 5>
debug: LoggerLevel  # value = <LoggerLevel.debug: 1>
error: LoggerLevel  # value = <LoggerLevel.error: 4>
info: LoggerLevel  # value = <LoggerLevel.info: 2>
off: LoggerLevel  # value = <LoggerLevel.off: 6>
trace: LoggerLevel  # value = <LoggerLevel.trace: 0>
warn: LoggerLevel  # value = <LoggerLevel.warn: 3>
