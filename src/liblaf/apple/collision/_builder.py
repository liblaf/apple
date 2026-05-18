import ipctk
import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer

from liblaf import jarp
from liblaf.apple.common import GLOBAL_POINT_ID

from ._collision import Collision


@jarp.define
class CollisionBuilder:
    stiffness: float = jarp.static()
    use_physical_barrier: bool = jarp.static(default=True)

    rest_positions: Float[np.ndarray, "V dim"] = jarp.field(default=np.empty((0, 3)))
    faces: Integer[np.ndarray, "F 3"] = jarp.field(
        default=np.empty((0, 3), dtype=np.int32)
    )
    indices: Integer[np.ndarray, " V"] = jarp.field(
        default=np.empty((0,), dtype=np.int32)
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
        collisions: ipctk.NormalCollisions = ipctk.NormalCollisions()
        if self.use_physical_barrier:
            collisions.use_area_weighting = True
            collisions.collision_set_type = (
                ipctk.NormalCollisions.CollisionSetType.IMPROVED_MAX_APPROX
            )
        return Collision(
            collision_mesh=collision_mesh,
            indices=self.indices,
            potential=potential,
            collisions=collisions,
        )
