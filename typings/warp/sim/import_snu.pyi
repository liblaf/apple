from _typeshed import Incomplete

class MuscleUnit:
    name: str
    bones: Incomplete
    points: Incomplete
    def __init__(self) -> None: ...

class Skeleton:
    def __init__(self, root_xform, skeleton_file, muscle_file, builder, filter, armature: float = 0.0) -> None: ...
    node_map: Incomplete
    xform_map: Incomplete
    mesh_map: Incomplete
    coord_start: Incomplete
    dof_start: Incomplete
    def parse_skeleton(self, filename, builder, filter, root_xform, armature) -> None: ...
    muscle_start: Incomplete
    muscles: Incomplete
    def parse_muscles(self, filename, builder) -> None: ...

def parse_snu(root_xform, skeleton_file, muscle_file, builder, filter, armature: float = 0.0): ...
