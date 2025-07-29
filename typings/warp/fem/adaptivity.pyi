import warp as wp
from warp.fem import cache as cache
from warp.fem.domain import Cells as Cells
from warp.fem.field import GeometryField as GeometryField
from warp.fem.geometry import AdaptiveNanogrid as AdaptiveNanogrid
from warp.fem.integrate import interpolate as interpolate
from warp.fem.operator import integrand as integrand, lookup as lookup
from warp.fem.types import Domain as Domain, Field as Field, NULL_ELEMENT_INDEX as NULL_ELEMENT_INDEX, Sample as Sample

def adaptive_nanogrid_from_hierarchy(grids: list[wp.Volume], grading: str | None = None, temporary_store: cache.TemporaryStore | None = None) -> AdaptiveNanogrid: ...
def adaptive_nanogrid_from_field(coarse_grid: wp.Volume, level_count: int, refinement_field: GeometryField, samples_per_voxel: int = 64, grading: str | None = None, temporary_store: cache.TemporaryStore | None = None) -> AdaptiveNanogrid: ...
def enforce_nanogrid_grading(cell_grid: wp.Volume, cell_level: wp.array, level_count: int, grading: str | None = None, temporary_store: cache.TemporaryStore | None = None) -> tuple[wp.Volume, wp.array]: ...
