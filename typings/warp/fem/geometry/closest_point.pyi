import warp as wp
from typing import Any
from warp.fem.types import Coords as Coords

@wp.func
def project_on_seg_at_origin(q: Any, seg: Any, len_sq: float): ...
@wp.func
def project_on_tri_at_origin(q: Any, e1: Any, e2: Any): ...
@wp.func
def project_on_tet_at_origin(q: wp.vec3, e1: wp.vec3, e2: wp.vec3, e3: wp.vec3): ...
