import warp as wp
from _typeshed import Incomplete
from enum import Enum

vec2i = wp.vec2i
vec3i = wp.vec3i
vec4i = wp.vec4i
Coords: Incomplete
OUTSIDE: Incomplete
ElementIndex = int
QuadraturePointIndex = int
NodeIndex = int
NULL_ELEMENT_INDEX: Incomplete
NULL_QP_INDEX: Incomplete
NULL_NODE_INDEX: Incomplete
DofIndex = wp.vec2i
NULL_DOF_INDEX: Incomplete

@wp.func
def get_node_index_in_element(dof_idx: DofIndex): ...
@wp.func
def get_node_coord(dof_idx: DofIndex): ...

class ElementKind(Enum):
    CELL = 0
    SIDE = 1

class NodeElementIndex:
    domain_element_index: ElementIndex
    node_index_in_element: int

class Sample:
    element_index: ElementIndex
    element_coords: Coords
    qp_index: QuadraturePointIndex
    qp_weight: float
    test_dof: DofIndex
    trial_dof: DofIndex

@wp.func
def make_free_sample(element_index: ElementIndex, element_coords: Coords): ...

class Field:
    call_operator: warp.fem.operator.Operator

class Domain:
    call_operator: warp.fem.operator.Operator
