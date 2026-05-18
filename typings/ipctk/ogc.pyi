"""
Offset Geometric Contact (OGC) helpers
"""

from __future__ import annotations
import ipctk
import numpy
import numpy.typing
import typing

__all__: list[str] = [
    "TrustRegion",
    "check_edge_feasible_region",
    "check_vertex_feasible_region",
    "is_edge_edge_feasible",
    "is_edge_vertex_feasible",
    "is_face_vertex_feasible",
]

class TrustRegion:
    """

    A trust region for filtering optimization steps.
    See "Offset Geometric Contact" by Chen et al. [2025]

    """
    def __init__(self, dhat: typing.SupportsFloat) -> None:
        """
        Construct a TrustRegion object.

        Parameters:
            dhat: The offset distance for contact.
        """
    def filter_step(
        self,
        mesh: ...,
        x: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        dx: typing.Annotated[
            numpy.typing.NDArray[numpy.float64],
            "[m, n]",
            "flags.writeable",
            "flags.f_contiguous",
        ],
    ) -> None:
        """
        Filter the optimization step dx to stay within the trust region.

        Parameters:
            mesh: The collision mesh.
            x: Current vertex positions.
            dx: Proposed vertex displacements.

        Note:
            Sets should_update_trust_region to true if the trust region should be updated on the next iteration.
        """
    def planar_filter_step(
        self,
        mesh: ...,
        x: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        dx: typing.Annotated[
            numpy.typing.NDArray[numpy.float64],
            "[m, n]",
            "flags.writeable",
            "flags.f_contiguous",
        ],
    ) -> None:
        """
        Filter the optimization step dx using Planar-DAT (Divide and Truncate).

        For each collision candidate (stored in ``candidates`` by the last
        call to ``update``), computes a direction-aware division plane and
        truncates only the component of displacement toward that plane.
        This eliminates the artificial damping and deadlock of the isotropic
        ``filter_step`` while retaining the penetration-free guarantee.

        See "Divide and Truncate: A Penetration and Inversion Free Framework
        for Coupled Multi-physics Systems" [ACM SIGGRAPH 2026].

        Parameters:
            mesh: The collision mesh.
            x: Current vertex positions.
            dx: Proposed vertex displacements (modified in-place).
        """
    def update(
        self,
        mesh: ...,
        x: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        collisions: ipctk.NormalCollisions,
        min_distance: typing.SupportsFloat = 0.0,
        broad_phase: ipctk.BroadPhase = None,
    ) -> None:
        """
        Update the trust region based on the current positions.

        Parameters:
            mesh: The collision mesh.
            x: Current vertex positions.
            collisions: Collisions to be updated.
            min_distance: Minimum distance between elements.
            broad_phase: Broad phase collision detection.
        """
    def update_if_needed(
        self,
        mesh: ...,
        x: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        collisions: ipctk.NormalCollisions,
        min_distance: typing.SupportsFloat = 0.0,
        broad_phase: ipctk.BroadPhase = None,
    ) -> None:
        """
        Update the trust region if needed based on the current positions.

        Parameters:
            mesh: The collision mesh.
            x: Current vertex positions.
            collisions: Collisions to be updated.
            min_distance: Minimum distance between elements.
            broad_phase: Broad phase collision detection.
        """
    def warm_start_time_step(
        self,
        mesh: ...,
        x: typing.Annotated[
            numpy.typing.NDArray[numpy.float64],
            "[m, n]",
            "flags.writeable",
            "flags.f_contiguous",
        ],
        pred_x: typing.Annotated[
            numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
        ],
        collisions: ipctk.NormalCollisions,
        dhat: typing.SupportsFloat,
        min_distance: typing.SupportsFloat = 0.0,
        broad_phase: ipctk.BroadPhase = None,
    ) -> None:
        """
        Warm start the time step by moving towards the predicted positions.

        This also initializes the trust region.

        Parameters:
            mesh: The collision mesh.
            x: Current vertex positions. (Will be modified)
            pred_x: Predicted vertex positions.
            collisions: Collisions to be initialized.
            dhat: The offset distance for contact.
            min_distance: Minimum distance between elements.
            broad_phase: Broad phase collision detection.
        """
    @property
    def candidates(self) -> ipctk.Candidates:
        """
        Collision candidates used for Planar-DAT. Updated by ``update()``.
        """
    @candidates.setter
    def candidates(self, arg0: ipctk.Candidates) -> None: ...
    @property
    def relaxed_radius_scaling(self) -> float:
        """
        Scaling factor for the relaxed trust region radius.

        Note:
            - This should be in (0, 1).
            - This is referred to as :math:`2\\gamma_p` in the paper.
        """
    @relaxed_radius_scaling.setter
    def relaxed_radius_scaling(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def should_update_trust_region(self) -> bool:
        """
        If true, the trust region will be updated on the next call to ``update_if_needed``.
        """
    @should_update_trust_region.setter
    def should_update_trust_region(self, arg0: bool) -> None: ...
    @property
    def trust_region_centers(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Centers of the trust regions for each vertex.
        """
    @trust_region_centers.setter
    def trust_region_centers(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]
    ) -> None: ...
    @property
    def trust_region_inflation_radius(self) -> float:
        """
        Inflation radius for the trust region.

        This is computed each time step based on the predicted motion.
        """
    @trust_region_inflation_radius.setter
    def trust_region_inflation_radius(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def trust_region_radii(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Radii of the trust regions for each vertex.
        """
    @trust_region_radii.setter
    def trust_region_radii(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]
    ) -> None: ...
    @property
    def update_threshold(self) -> float:
        """
        Threshold for updating the trust region.

        If more than this fraction of vertices are restricted by the trust region, we update the trust region.

        Note
        ----
            This is referred to as \\f\\(\\gamma_e\\f\\) in the paper.
        """
    @update_threshold.setter
    def update_threshold(self, arg0: typing.SupportsFloat) -> None: ...

def check_edge_feasible_region(
    mesh: ...,
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    xi: typing.SupportsInt,
    ei: typing.SupportsInt,
) -> bool:
    """
    Check if vertex `xi` is in the feasible region of edge `ei`.

    Parameters:
        mesh: Collision mesh containing the edge adjacencies
        vertices: Matrix of current vertex positions (rowwise)
        xi: Index of the vertex to check
        ei: Index of the edge for which to check the feasible region

    Returns:
        True if the vertex is in the feasible region, false otherwise
    """

@typing.overload
def check_vertex_feasible_region(
    mesh: ...,
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    point: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
    vi: typing.SupportsInt,
) -> bool:
    """
    Check if point `x` is in the feasible region of vertex `vi`.

    Parameters:
        mesh: Collision mesh containing the vertex adjacencies
        vertices: Matrix of current vertex positions (rowwise)
        point: Position of the point to check
        vi: Index of the vertex for which to check the feasible region

    Returns:
        True if the point is in the feasible region, false otherwise
    """

@typing.overload
def check_vertex_feasible_region(
    mesh: ...,
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    xi: typing.SupportsInt,
    vi: typing.SupportsInt,
) -> bool:
    """
    Check if vertex `xi` is in the feasible region of vertex `vi`.

    Parameters:
        mesh: Collision mesh containing the vertex adjacencies
        vertices: Matrix of current vertex positions (rowwise)
        xi: Index of the vertex to check
        vi: Index of the vertex for which to check the feasible region

    Returns:
        True if the vertex is in the feasible region, false otherwise
    """

def is_edge_edge_feasible(
    mesh: ...,
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    candidate: ipctk.EdgeEdgeCandidate,
    dtype: ipctk.EdgeEdgeDistanceType = ...,
) -> bool:
    """
    Check if the edge-edge candidate is feasible.

    Parameters:
        mesh: Collision mesh containing the edge adjacencies
        vertices: Matrix of current vertex positions (rowwise)
        candidate: Edge-edge candidate to check
        dtype: Edge-edge distance type to use for the check.

    Returns:
        True if the edge-edge candidate is in the feasible region, false otherwise
    """

def is_edge_vertex_feasible(
    mesh: ...,
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    candidate: ipctk.EdgeVertexCandidate,
    dtype: ipctk.PointEdgeDistanceType = ...,
) -> bool:
    """
    Check if the edge-vertex candidate is feasible.

    Parameters:
        mesh: Collision mesh containing the edge adjacencies
        vertices: Matrix of current vertex positions (rowwise)
        candidate: Edge-vertex candidate to check
        dtype: Edge-vertex distance type to use for the check.

    Returns:
        True if the edge-vertex candidate is in the feasible region, false otherwise
    """

def is_face_vertex_feasible(
    mesh: ...,
    vertices: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"
    ],
    candidate: ipctk.FaceVertexCandidate,
    dtype: ipctk.PointTriangleDistanceType = ...,
) -> bool:
    """
    Check if the face-vertex candidate is feasible.

    Parameters:
        mesh: Collision mesh containing the face adjacencies
        vertices: Matrix of current vertex positions (rowwise)
        candidate: Face-vertex candidate to check
        dtype: Point-triangle distance type to use for the check.

    Returns:
        True if the face-vertex candidate is in the feasible region, false otherwise
    """
