import warp.fem.domain as _domain
import warp.fem.geometry as _geometry
import warp.fem.polynomial as _polynomial
from .basis_function_space import CollocatedFunctionSpace as CollocatedFunctionSpace, ContravariantFunctionSpace as ContravariantFunctionSpace, CovariantFunctionSpace as CovariantFunctionSpace
from .basis_space import BasisSpace as BasisSpace, PointBasisSpace as PointBasisSpace, ShapeBasisSpace as ShapeBasisSpace, make_discontinuous_basis_space as make_discontinuous_basis_space
from .dof_mapper import DofMapper as DofMapper, IdentityMapper as IdentityMapper, SkewSymmetricTensorMapper as SkewSymmetricTensorMapper, SymmetricTensorMapper as SymmetricTensorMapper
from .function_space import FunctionSpace as FunctionSpace
from .grid_2d_function_space import make_grid_2d_space_topology as make_grid_2d_space_topology
from .grid_3d_function_space import make_grid_3d_space_topology as make_grid_3d_space_topology
from .hexmesh_function_space import make_hexmesh_space_topology as make_hexmesh_space_topology
from .nanogrid_function_space import make_nanogrid_space_topology as make_nanogrid_space_topology
from .partition import SpacePartition as SpacePartition, make_space_partition as make_space_partition
from .quadmesh_function_space import make_quadmesh_space_topology as make_quadmesh_space_topology
from .restriction import SpaceRestriction as SpaceRestriction
from .shape import ElementBasis as ElementBasis, ShapeFunction as ShapeFunction, get_shape_function as get_shape_function
from .tetmesh_function_space import make_tetmesh_space_topology as make_tetmesh_space_topology
from .topology import SpaceTopology as SpaceTopology
from .trimesh_function_space import make_trimesh_space_topology as make_trimesh_space_topology
from enum import Enum as Enum

def make_space_restriction(space: FunctionSpace | None = None, space_partition: SpacePartition | None = None, domain: _domain.GeometryDomain | None = None, space_topology: SpaceTopology | None = None, device=None, temporary_store: warp.fem.cache.TemporaryStore | None = None) -> SpaceRestriction: ...
def make_polynomial_basis_space(geo: _geometry.Geometry, degree: int = 1, element_basis: ElementBasis | None = None, discontinuous: bool = False, family: _polynomial.Polynomial | None = None) -> BasisSpace: ...
def make_collocated_function_space(basis_space: BasisSpace, dtype: type = ..., dof_mapper: DofMapper | None = None) -> CollocatedFunctionSpace: ...
def make_covariant_function_space(basis_space: BasisSpace) -> CovariantFunctionSpace: ...
def make_contravariant_function_space(basis_space: BasisSpace) -> ContravariantFunctionSpace: ...
def make_polynomial_space(geo: _geometry.Geometry, dtype: type = ..., dof_mapper: DofMapper | None = None, degree: int = 1, element_basis: ElementBasis | None = None, discontinuous: bool = False, family: _polynomial.Polynomial | None = None) -> CollocatedFunctionSpace: ...
