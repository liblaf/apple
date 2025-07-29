import warp.fem.operator as operator
from .field import SpaceField as SpaceField
from _typeshed import Incomplete
from warp.fem import cache as cache
from warp.fem.domain import GeometryDomain as GeometryDomain
from warp.fem.linalg import basis_coefficient as basis_coefficient, generalized_inner as generalized_inner, generalized_outer as generalized_outer
from warp.fem.quadrature import Quadrature as Quadrature
from warp.fem.space import FunctionSpace as FunctionSpace, SpacePartition as SpacePartition, SpaceRestriction as SpaceRestriction
from warp.fem.types import DofIndex as DofIndex, NULL_NODE_INDEX as NULL_NODE_INDEX, Sample as Sample, get_node_coord as get_node_coord, get_node_index_in_element as get_node_index_in_element
from warp.fem.utils import type_zero_element as type_zero_element

class AdjointField(SpaceField):
    node_dof_count: Incomplete
    value_dof_count: Incomplete
    def __init__(self, space: FunctionSpace, space_partition: SpacePartition) -> None: ...
    @property
    def name(self) -> str: ...
    @cache.cached_arg_value
    def eval_arg_value(self, device): ...
    def fill_eval_arg(self, arg, device) -> None: ...

class TestField(AdjointField):
    space_restriction: Incomplete
    domain: Incomplete
    def __init__(self, space_restriction: SpaceRestriction, space: FunctionSpace) -> None: ...

class TrialField(AdjointField):
    domain: Incomplete
    def __init__(self, space: FunctionSpace, space_partition: SpacePartition, domain: GeometryDomain) -> None: ...
    def partition_node_count(self) -> int: ...

class LocalAdjointField(SpaceField):
    INNER_DOF: Incomplete
    OUTER_DOF: Incomplete
    INNER_GRAD_DOF: Incomplete
    OUTER_GRAD_DOF: Incomplete
    DOF_TYPE_COUNT: Incomplete
    DofOffsets: Incomplete
    class EvalArg: ...
    global_field: Incomplete
    domain: Incomplete
    node_dof_count: Incomplete
    value_dof_count: Incomplete
    at_node: Incomplete
    TAYLOR_DOF_COUNT: int
    def __init__(self, field: AdjointField) -> None: ...
    def notify_operator_usage(self, ops: set[operator.Operator]): ...
    @property
    def name(self) -> str: ...
    def eval_arg_value(self, device): ...
    def fill_eval_arg(self, arg, device) -> None: ...

class LocalTestField(LocalAdjointField):
    space_restriction: Incomplete
    def __init__(self, test_field: TestField) -> None: ...

class LocalTrialField(LocalAdjointField):
    def __init__(self, trial_field: TrialField) -> None: ...

def make_linear_dispatch_kernel(test: LocalTestField, quadrature: Quadrature, accumulate_dtype: type): ...
def make_bilinear_dispatch_kernel(test: LocalTestField, trial: LocalTrialField, quadrature: Quadrature, accumulate_dtype: type): ...
