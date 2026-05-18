"""
Tight Inclusion CCD method of [Wang et al. 2021].
"""

from __future__ import annotations
import numpy
import numpy.typing
import typing

__all__: list[str] = [
    "BREADTH_FIRST_SEARCH",
    "CCDRootFindingMethod",
    "DEPTH_FIRST_SEARCH",
    "compute_ccd_filters",
    "edge_edge_ccd",
    "point_triangle_ccd",
]

class CCDRootFindingMethod:
    """
    Enumeration of implemented root finding methods.

    Members:

      DEPTH_FIRST_SEARCH : Depth first search

      BREADTH_FIRST_SEARCH : Breadth first search
    """

    BREADTH_FIRST_SEARCH: typing.ClassVar[
        CCDRootFindingMethod
    ]  # value = <CCDRootFindingMethod.BREADTH_FIRST_SEARCH: 1>
    DEPTH_FIRST_SEARCH: typing.ClassVar[
        CCDRootFindingMethod
    ]  # value = <CCDRootFindingMethod.DEPTH_FIRST_SEARCH: 0>
    __members__: typing.ClassVar[
        dict[str, CCDRootFindingMethod]
    ]  # value = {'DEPTH_FIRST_SEARCH': <CCDRootFindingMethod.DEPTH_FIRST_SEARCH: 0>, 'BREADTH_FIRST_SEARCH': <CCDRootFindingMethod.BREADTH_FIRST_SEARCH: 1>}
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

def compute_ccd_filters(
    min_corner: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    max_corner: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    is_vertex_face: bool,
    using_minimum_separation: bool,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Compute the numerical error filters for the input to the CCD solver.

    Before you run the simulation, you need to conservatively estimate the
    axis-aligned bounding box in which the meshes will be located during the
    whole simulation process.

    Parameters:
        min_corner: Minimum corner of the axis-aligned bounding box of the simulation scene.
        max_corner: Maximum corner of the axis-aligned bounding box of the simulation scene.
        is_vertex_face: True if checking vertex-face collision, false if checking edge-edge collision.
        using_minimum_separation: True if using minimum separation CCD, false otherwise.

    Returns:
        The numerical error filters for the input parameters.
    """

def edge_edge_ccd(
    ea0_t0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    ea1_t0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    eb0_t0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    eb1_t0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    ea0_t1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    ea1_t1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    eb0_t1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    eb1_t1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    min_distance: typing.SupportsFloat = 0,
    tmax: typing.SupportsFloat = 1,
    tolerance: typing.SupportsFloat = 1e-06,
    max_iterations: typing.SupportsInt = 10000000,
    filter: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"] = ...,
    no_zero_toi: bool = False,
    ccd_method: CCDRootFindingMethod = ...,
) -> tuple[bool, float, float]:
    """
    Determine the earliest time of impact between two edges (optionally with a minimum separation).

    Parameters:
        ea0_t0: Starting position of the first vertex of the first edge.
        ea1_t0: Start position of the second vertex of the first edge.
        eb0_t0: Start position of the first vertex of the second edge.
        eb1_t0: Start position of the second vertex of the second edge.
        ea0_t1: End position of the first vertex of the first edge.
        ea1_t1: End position of the second vertex of the first edge.
        eb0_t1: End position of the first vertex of the second edge.
        eb1_t1: End position of the second vertex of the second edge.
        min_distance: Minimum separation distance (default: 0).
        tmax: Upper bound of the time interval [0,tmax] to be checked (0<=tmax<=1).
        tolerance: Solver tolerance (default: 1e-6).
        max_iterations: Maximum number of solver iterations (default: 1e7). If negative, solver will run until convergence.
        filter: Filters calculated using get_numerical_error (default: (-1,-1,-1)). Use (-1,-1,-1) if checking a single query.
        no_zero_toi: Refine further if a zero TOI is produced (assuming not initially in contact).
        ccd_method: Root finding method (default: BREADTH_FIRST_SEARCH).

    Returns:
        Tuple of:
            True if there is a collision, false otherwise,
            the earliest time of collision if collision happens (infinity if no collision occurs), and
            if max_iterations < 0, the solver precision otherwise the input tolerance.
    """

def point_triangle_ccd(
    v_t0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    f0_t0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    f1_t0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    f2_t0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    v_t1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    f0_t1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    f1_t1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    f2_t1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    min_distance: typing.SupportsFloat = 0,
    tmax: typing.SupportsFloat = 1,
    tolerance: typing.SupportsFloat = 1e-06,
    max_iterations: typing.SupportsInt = 10000000,
    filter: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"] = ...,
    no_zero_toi: bool = False,
    ccd_method: CCDRootFindingMethod = ...,
) -> tuple[bool, float, float]:
    """
    Determine the earliest time of impact between a point and triangle (optionally with a minimum separation).

    Parameters:
        v_t0:  Starting position of the vertex.
        f0_t0: Starting position of the first vertex of the face.
        f1_t0: Starting position of the second vertex of the face.
        f2_t0: Starting position of the third vertex of the face.
        v_t1:  Ending position of the vertex.
        f0_t1: Ending position of the first vertex of the face.
        f1_t1: Ending position of the second vertex of the face.
        f2_t1: Ending position of the third vertex of the face.
        min_distance: Minimum separation distance (default: 0).
        tmax: Upper bound of the time interval [0,tmax] to be checked (0<=tmax<=1).
        tolerance: Solver tolerance (default: 1e-6).
        max_iterations: Maximum number of solver iterations (default: 1e7). If negative, solver will run until convergence.
        filter: Filters calculated using get_numerical_error (default: (-1,-1,-1)). Use (-1,-1,-1) if checking a single query.
        no_zero_toi: Refine further if a zero TOI is produced (assuming not initially in contact).
        ccd_method: Root finding method (default: BREADTH_FIRST_SEARCH).

    Returns:
        Tuple of:
            True if there is a collision, false otherwise,
            the earliest time of collision if collision happens (infinity if no collision occurs), and
            if max_iterations < 0, the solver precision otherwise the input tolerance.
    """

BREADTH_FIRST_SEARCH: (
    CCDRootFindingMethod  # value = <CCDRootFindingMethod.BREADTH_FIRST_SEARCH: 1>
)
DEPTH_FIRST_SEARCH: (
    CCDRootFindingMethod  # value = <CCDRootFindingMethod.DEPTH_FIRST_SEARCH: 0>
)
