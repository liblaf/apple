import warp as wp
from _typeshed import Incomplete
from warp.render.utils import solidify_mesh as solidify_mesh, tab10_color_map as tab10_color_map

NAN: Incomplete

@wp.kernel
def compute_contact_points(body_q: None, shape_body: None, contact_count: None, contact_shape0: None, contact_shape1: None, contact_point0: None, contact_point1: None, contact_pos0: None, contact_pos1: None): ...
def CreateSimRenderer(renderer): ...

SimRendererUsd: Incomplete
SimRendererOpenGL: Incomplete
SimRenderer = SimRendererUsd
