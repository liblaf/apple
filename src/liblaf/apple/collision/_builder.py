import attrs
import ipctk
import numpy as np
import pyvista as pv
import torch
from jaxtyping import Float, Integer

from liblaf.apple.common import GLOBAL_POINT_ID

from ._collision import Collision


@attrs.define
class CollisionBuilder:
    stiffness: float
    use_physical_barrier: bool = attrs.field(default=True)

    rest_positions: Float[np.ndarray, "V dim"] = attrs.field(
        factory=lambda: np.empty((0, 3))
    )
    faces: Integer[np.ndarray, "F 3"] = attrs.field(
        factory=lambda: np.empty((0, 3), dtype=np.int32)
    )
    indices: Integer[np.ndarray, " V"] = attrs.field(
        factory=lambda: np.empty((0,), dtype=np.int32)
    )

    @property
    def n_vertices(self) -> int:
        return self.rest_positions.shape[0]

    def add_tetmesh(self, obj: pv.UnstructuredGrid) -> None:
        surface: pv.PolyData = obj.extract_surface(algorithm=None)
        self.add_trimesh(surface)

    def add_trimesh(self, obj: pv.PolyData) -> None:
        global_point_id: Integer[np.ndarray, ""] = obj.point_data[GLOBAL_POINT_ID.vtk]
        faces: Integer[np.ndarray, "F 3"] = obj.regular_faces + self.n_vertices
        self.rest_positions = np.concat([self.rest_positions, obj.points])
        self.faces = np.concat([self.faces, faces])
        self.indices = np.concat([self.indices, global_point_id])

    def finalize(self) -> Collision:
        edges: Integer[np.ndarray, "E 2"] = ipctk.edges(self.faces)
        collision_mesh: ipctk.CollisionMesh = ipctk.CollisionMesh(
            rest_positions=self.rest_positions, edges=edges, faces=self.faces
        )
        mean_edge_length, _std = ipctk.mean_edge_length(
            collision_mesh.rest_positions,
            collision_mesh.rest_positions,
            collision_mesh.edges,
        )
        potential: ipctk.BarrierPotential = ipctk.BarrierPotential(
            dhat=0.5 * mean_edge_length,
            stiffness=self.stiffness,
            use_physical_barrier=self.use_physical_barrier,
        )
        return Collision(
            collision_mesh=collision_mesh,
            indices=torch.as_tensor(self.indices),
            potential=potential,
            use_physical_barrier=self.use_physical_barrier,
        )
