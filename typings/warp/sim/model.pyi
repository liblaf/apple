import warp as wp
from .graph_coloring import ColoringAlgorithm as ColoringAlgorithm, color_trimesh as color_trimesh, combine_independent_particle_coloring as combine_independent_particle_coloring
from .inertia import compute_box_inertia as compute_box_inertia, compute_capsule_inertia as compute_capsule_inertia, compute_cone_inertia as compute_cone_inertia, compute_cylinder_inertia as compute_cylinder_inertia, compute_mesh_inertia as compute_mesh_inertia, compute_sphere_inertia as compute_sphere_inertia, transform_inertia as transform_inertia
from _typeshed import Incomplete

Vec3 = list[float]
Vec4 = list[float]
Quat = list[float]
Mat33 = list[float]
Transform = tuple[Vec3, Quat]
PARTICLE_FLAG_ACTIVE: Incomplete
GEO_SPHERE: Incomplete
GEO_BOX: Incomplete
GEO_CAPSULE: Incomplete
GEO_CYLINDER: Incomplete
GEO_CONE: Incomplete
GEO_MESH: Incomplete
GEO_SDF: Incomplete
GEO_PLANE: Incomplete
GEO_NONE: Incomplete
JOINT_PRISMATIC: Incomplete
JOINT_REVOLUTE: Incomplete
JOINT_BALL: Incomplete
JOINT_FIXED: Incomplete
JOINT_FREE: Incomplete
JOINT_COMPOUND: Incomplete
JOINT_UNIVERSAL: Incomplete
JOINT_DISTANCE: Incomplete
JOINT_D6: Incomplete
JOINT_MODE_FORCE: Incomplete
JOINT_MODE_TARGET_POSITION: Incomplete
JOINT_MODE_TARGET_VELOCITY: Incomplete

def flag_to_int(flag): ...

class ModelShapeMaterials:
    ke: None
    kd: None
    kf: None
    ka: None
    mu: None
    restitution: None

class ModelShapeGeometry:
    type: None
    is_solid: None
    thickness: None
    source: None
    scale: None

class JointAxis:
    axis: Incomplete
    limit_lower: Incomplete
    limit_upper: Incomplete
    limit_ke: Incomplete
    limit_kd: Incomplete
    action: Incomplete
    target_ke: Incomplete
    target_kd: Incomplete
    mode: Incomplete
    def __init__(self, axis, limit_lower=..., limit_upper=..., limit_ke: float = 100.0, limit_kd: float = 10.0, action=None, target_ke: float = 0.0, target_kd: float = 0.0, mode=...) -> None: ...

class SDF:
    volume: Incomplete
    I: Incomplete
    mass: Incomplete
    com: Incomplete
    has_inertia: bool
    is_solid: bool
    def __init__(self, volume=None, I=None, mass: float = 1.0, com=None) -> None: ...
    def finalize(self, device=None): ...
    def __hash__(self): ...

class Mesh:
    vertices: Incomplete
    indices: Incomplete
    is_solid: Incomplete
    has_inertia: Incomplete
    I: Incomplete
    mass: float
    com: Incomplete
    def __init__(self, vertices: list[Vec3], indices: list[int], compute_inertia: bool = True, is_solid: bool = True) -> None: ...
    mesh: Incomplete
    def finalize(self, device=None): ...
    def __hash__(self): ...

class State:
    particle_q: wp.array | None
    particle_qd: wp.array | None
    particle_f: wp.array | None
    body_q: wp.array | None
    body_qd: wp.array | None
    body_f: wp.array | None
    joint_q: wp.array | None
    joint_qd: wp.array | None
    def __init__(self) -> None: ...
    def clear_forces(self) -> None: ...
    @property
    def requires_grad(self) -> bool: ...
    @property
    def body_count(self) -> int: ...
    @property
    def particle_count(self) -> int: ...
    @property
    def joint_coord_count(self) -> int: ...
    @property
    def joint_dof_count(self) -> int: ...

class Control:
    joint_act: wp.array | None
    tri_activations: wp.array | None
    tet_activations: wp.array | None
    muscle_activations: wp.array | None
    def __init__(self, model: Model | None = None) -> None: ...
    def clear(self) -> None: ...
    def reset(self) -> None: ...

def compute_shape_mass(type, scale, src, density, is_solid, thickness): ...

class Model:
    requires_grad: bool
    num_envs: int
    particle_q: Incomplete
    particle_qd: Incomplete
    particle_mass: Incomplete
    particle_inv_mass: Incomplete
    particle_radius: Incomplete
    particle_max_radius: float
    particle_ke: float
    particle_kd: float
    particle_kf: float
    particle_mu: float
    particle_cohesion: float
    particle_adhesion: float
    particle_grid: Incomplete
    particle_flags: Incomplete
    particle_max_velocity: float
    shape_transform: Incomplete
    shape_body: Incomplete
    shape_visible: Incomplete
    body_shapes: Incomplete
    shape_materials: Incomplete
    shape_geo: Incomplete
    shape_geo_src: Incomplete
    shape_collision_group: Incomplete
    shape_collision_group_map: Incomplete
    shape_collision_filter_pairs: Incomplete
    shape_collision_radius: Incomplete
    shape_ground_collision: Incomplete
    shape_shape_collision: Incomplete
    shape_contact_pairs: Incomplete
    shape_ground_contact_pairs: Incomplete
    spring_indices: Incomplete
    spring_rest_length: Incomplete
    spring_stiffness: Incomplete
    spring_damping: Incomplete
    spring_control: Incomplete
    spring_constraint_lambdas: Incomplete
    tri_indices: Incomplete
    tri_poses: Incomplete
    tri_activations: Incomplete
    tri_materials: Incomplete
    tri_areas: Incomplete
    edge_indices: Incomplete
    edge_rest_angle: Incomplete
    edge_rest_length: Incomplete
    edge_bending_properties: Incomplete
    edge_constraint_lambdas: Incomplete
    tet_indices: Incomplete
    tet_poses: Incomplete
    tet_activations: Incomplete
    tet_materials: Incomplete
    muscle_start: Incomplete
    muscle_params: Incomplete
    muscle_bodies: Incomplete
    muscle_points: Incomplete
    muscle_activations: Incomplete
    body_q: Incomplete
    body_qd: Incomplete
    body_com: Incomplete
    body_inertia: Incomplete
    body_inv_inertia: Incomplete
    body_mass: Incomplete
    body_inv_mass: Incomplete
    body_name: Incomplete
    joint_q: Incomplete
    joint_qd: Incomplete
    joint_act: Incomplete
    joint_type: Incomplete
    joint_parent: Incomplete
    joint_child: Incomplete
    joint_ancestor: Incomplete
    joint_X_p: Incomplete
    joint_X_c: Incomplete
    joint_axis: Incomplete
    joint_armature: Incomplete
    joint_target_ke: Incomplete
    joint_target_kd: Incomplete
    joint_axis_start: Incomplete
    joint_axis_dim: Incomplete
    joint_axis_mode: Incomplete
    joint_linear_compliance: Incomplete
    joint_angular_compliance: Incomplete
    joint_enabled: Incomplete
    joint_limit_lower: Incomplete
    joint_limit_upper: Incomplete
    joint_limit_ke: Incomplete
    joint_limit_kd: Incomplete
    joint_twist_lower: Incomplete
    joint_twist_upper: Incomplete
    joint_q_start: Incomplete
    joint_qd_start: Incomplete
    articulation_start: Incomplete
    joint_name: Incomplete
    joint_attach_ke: float
    joint_attach_kd: float
    soft_contact_radius: float
    soft_contact_margin: float
    soft_contact_ke: float
    soft_contact_kd: float
    soft_contact_kf: float
    soft_contact_mu: float
    soft_contact_restitution: float
    soft_contact_count: int
    soft_contact_particle: Incomplete
    soft_contact_shape: Incomplete
    soft_contact_body_pos: Incomplete
    soft_contact_body_vel: Incomplete
    soft_contact_normal: Incomplete
    soft_contact_tids: Incomplete
    rigid_contact_max: int
    rigid_contact_max_limited: int
    rigid_mesh_contact_max: int
    rigid_contact_margin: Incomplete
    rigid_contact_torsional_friction: Incomplete
    rigid_contact_rolling_friction: Incomplete
    rigid_contact_count: Incomplete
    rigid_contact_point0: Incomplete
    rigid_contact_point1: Incomplete
    rigid_contact_offset0: Incomplete
    rigid_contact_offset1: Incomplete
    rigid_contact_normal: Incomplete
    rigid_contact_thickness: Incomplete
    rigid_contact_shape0: Incomplete
    rigid_contact_shape1: Incomplete
    rigid_contact_tids: Incomplete
    rigid_contact_pairwise_counter: Incomplete
    rigid_contact_broad_shape0: Incomplete
    rigid_contact_broad_shape1: Incomplete
    rigid_contact_point_id: Incomplete
    rigid_contact_point_limit: Incomplete
    ground: bool
    ground_plane: Incomplete
    up_vector: Incomplete
    up_axis: int
    gravity: Incomplete
    particle_count: int
    body_count: int
    shape_count: int
    joint_count: int
    joint_axis_count: int
    tri_count: int
    tet_count: int
    edge_count: int
    spring_count: int
    muscle_count: int
    articulation_count: int
    joint_dof_count: int
    joint_coord_count: int
    particle_color_groups: Incomplete
    particle_colors: Incomplete
    device: Incomplete
    def __init__(self, device=None) -> None: ...
    def state(self, requires_grad=None) -> State: ...
    def control(self, requires_grad=None, clone_variables: bool = True) -> Control: ...
    def allocate_soft_contacts(self, count, requires_grad: bool = False) -> None: ...
    shape_contact_pair_count: Incomplete
    shape_ground_contact_pair_count: Incomplete
    def find_shape_contact_pairs(self) -> None: ...
    def count_contact_points(self): ...
    def allocate_rigid_contacts(self, target=None, count=None, limited_contact_count=None, requires_grad: bool = False) -> None: ...
    @property
    def soft_contact_max(self): ...

class ModelBuilder:
    default_particle_radius: float
    default_tri_ke: float
    default_tri_ka: float
    default_tri_kd: float
    default_tri_drag: float
    default_tri_lift: float
    default_spring_ke: float
    default_spring_kd: float
    default_edge_ke: float
    default_edge_kd: float
    default_shape_ke: float
    default_shape_kd: float
    default_shape_kf: float
    default_shape_ka: float
    default_shape_mu: float
    default_shape_restitution: float
    default_shape_density: float
    default_shape_thickness: float
    default_joint_limit_ke: float
    default_joint_limit_kd: float
    num_envs: int
    particle_q: Incomplete
    particle_qd: Incomplete
    particle_mass: Incomplete
    particle_radius: Incomplete
    particle_flags: Incomplete
    particle_max_velocity: float
    particle_color_groups: Incomplete
    shape_transform: Incomplete
    shape_body: Incomplete
    shape_visible: Incomplete
    shape_geo_type: Incomplete
    shape_geo_scale: Incomplete
    shape_geo_src: Incomplete
    shape_geo_is_solid: Incomplete
    shape_geo_thickness: Incomplete
    shape_material_ke: Incomplete
    shape_material_kd: Incomplete
    shape_material_kf: Incomplete
    shape_material_ka: Incomplete
    shape_material_mu: Incomplete
    shape_material_restitution: Incomplete
    shape_collision_group: Incomplete
    shape_collision_group_map: Incomplete
    last_collision_group: int
    shape_collision_radius: Incomplete
    shape_ground_collision: Incomplete
    shape_shape_collision: Incomplete
    shape_collision_filter_pairs: Incomplete
    geo_meshes: Incomplete
    geo_sdfs: Incomplete
    spring_indices: Incomplete
    spring_rest_length: Incomplete
    spring_stiffness: Incomplete
    spring_damping: Incomplete
    spring_control: Incomplete
    tri_indices: Incomplete
    tri_poses: Incomplete
    tri_activations: Incomplete
    tri_materials: Incomplete
    tri_areas: Incomplete
    edge_indices: Incomplete
    edge_rest_angle: Incomplete
    edge_rest_length: Incomplete
    edge_bending_properties: Incomplete
    tet_indices: Incomplete
    tet_poses: Incomplete
    tet_activations: Incomplete
    tet_materials: Incomplete
    muscle_start: Incomplete
    muscle_params: Incomplete
    muscle_activations: Incomplete
    muscle_bodies: Incomplete
    muscle_points: Incomplete
    body_mass: Incomplete
    body_inertia: Incomplete
    body_inv_mass: Incomplete
    body_inv_inertia: Incomplete
    body_com: Incomplete
    body_q: Incomplete
    body_qd: Incomplete
    body_name: Incomplete
    body_shapes: Incomplete
    joint_parent: Incomplete
    joint_parents: Incomplete
    joint_child: Incomplete
    joint_axis: Incomplete
    joint_X_p: Incomplete
    joint_X_c: Incomplete
    joint_q: Incomplete
    joint_qd: Incomplete
    joint_type: Incomplete
    joint_name: Incomplete
    joint_armature: Incomplete
    joint_target_ke: Incomplete
    joint_target_kd: Incomplete
    joint_axis_mode: Incomplete
    joint_limit_lower: Incomplete
    joint_limit_upper: Incomplete
    joint_limit_ke: Incomplete
    joint_limit_kd: Incomplete
    joint_act: Incomplete
    joint_twist_lower: Incomplete
    joint_twist_upper: Incomplete
    joint_linear_compliance: Incomplete
    joint_angular_compliance: Incomplete
    joint_enabled: Incomplete
    joint_q_start: Incomplete
    joint_qd_start: Incomplete
    joint_axis_start: Incomplete
    joint_axis_dim: Incomplete
    articulation_start: Incomplete
    joint_dof_count: int
    joint_coord_count: int
    joint_axis_total_count: int
    up_vector: Incomplete
    up_axis: Incomplete
    gravity: Incomplete
    soft_contact_max: Incomplete
    rigid_mesh_contact_max: int
    rigid_contact_margin: float
    rigid_contact_torsional_friction: float
    rigid_contact_rolling_friction: float
    num_rigid_contacts_per_env: Incomplete
    def __init__(self, up_vector=(0.0, 1.0, 0.0), gravity: float = -9.80665) -> None: ...
    @property
    def shape_count(self): ...
    @property
    def body_count(self): ...
    @property
    def joint_count(self): ...
    @property
    def joint_axis_count(self): ...
    @property
    def particle_count(self): ...
    @property
    def tri_count(self): ...
    @property
    def tet_count(self): ...
    @property
    def edge_count(self): ...
    @property
    def spring_count(self): ...
    @property
    def muscle_count(self): ...
    @property
    def articulation_count(self): ...
    def add_articulation(self) -> None: ...
    def add_builder(self, builder: ModelBuilder, xform: Transform | None = None, update_num_env_count: bool = True, separate_collision_group: bool = True): ...
    def add_body(self, origin: Transform | None = None, armature: float = 0.0, com: Vec3 | None = None, I_m: Mat33 | None = None, m: float = 0.0, name: str | None = None) -> int: ...
    def add_joint(self, joint_type: wp.constant, parent: int, child: int, linear_axes: list[JointAxis] | None = None, angular_axes: list[JointAxis] | None = None, name: str | None = None, parent_xform: wp.transform | None = None, child_xform: wp.transform | None = None, linear_compliance: float = 0.0, angular_compliance: float = 0.0, armature: float = 0.01, collision_filter_parent: bool = True, enabled: bool = True) -> int: ...
    def add_joint_revolute(self, parent: int, child: int, parent_xform: wp.transform | None = None, child_xform: wp.transform | None = None, axis: Vec3 = (1.0, 0.0, 0.0), target: float | None = None, target_ke: float = 0.0, target_kd: float = 0.0, mode: int = ..., limit_lower: float = ..., limit_upper: float = ..., limit_ke: float | None = None, limit_kd: float | None = None, linear_compliance: float = 0.0, angular_compliance: float = 0.0, armature: float = 0.01, name: str | None = None, collision_filter_parent: bool = True, enabled: bool = True) -> int: ...
    def add_joint_prismatic(self, parent: int, child: int, parent_xform: wp.transform | None = None, child_xform: wp.transform | None = None, axis: Vec3 = (1.0, 0.0, 0.0), target: float | None = None, target_ke: float = 0.0, target_kd: float = 0.0, mode: int = ..., limit_lower: float = -10000.0, limit_upper: float = 10000.0, limit_ke: float | None = None, limit_kd: float | None = None, linear_compliance: float = 0.0, angular_compliance: float = 0.0, armature: float = 0.01, name: str | None = None, collision_filter_parent: bool = True, enabled: bool = True) -> int: ...
    def add_joint_ball(self, parent: int, child: int, parent_xform: wp.transform | None = None, child_xform: wp.transform | None = None, linear_compliance: float = 0.0, angular_compliance: float = 0.0, armature: float = 0.01, name: str | None = None, collision_filter_parent: bool = True, enabled: bool = True) -> int: ...
    def add_joint_fixed(self, parent: int, child: int, parent_xform: wp.transform | None = None, child_xform: wp.transform | None = None, linear_compliance: float = 0.0, angular_compliance: float = 0.0, armature: float = 0.01, name: str | None = None, collision_filter_parent: bool = True, enabled: bool = True) -> int: ...
    def add_joint_free(self, child: int, parent_xform: wp.transform | None = None, child_xform: wp.transform | None = None, armature: float = 0.0, parent: int = -1, name: str | None = None, collision_filter_parent: bool = True, enabled: bool = True) -> int: ...
    def add_joint_distance(self, parent: int, child: int, parent_xform: wp.transform | None = None, child_xform: wp.transform | None = None, min_distance: float = -1.0, max_distance: float = 1.0, compliance: float = 0.0, collision_filter_parent: bool = True, enabled: bool = True) -> int: ...
    def add_joint_universal(self, parent: int, child: int, axis_0: JointAxis, axis_1: JointAxis, parent_xform: wp.transform | None = None, child_xform: wp.transform | None = None, linear_compliance: float = 0.0, angular_compliance: float = 0.0, armature: float = 0.01, name: str | None = None, collision_filter_parent: bool = True, enabled: bool = True) -> int: ...
    def add_joint_compound(self, parent: int, child: int, axis_0: JointAxis, axis_1: JointAxis, axis_2: JointAxis, parent_xform: wp.transform | None = None, child_xform: wp.transform | None = None, linear_compliance: float = 0.0, angular_compliance: float = 0.0, armature: float = 0.01, name: str | None = None, collision_filter_parent: bool = True, enabled: bool = True) -> int: ...
    def add_joint_d6(self, parent: int, child: int, linear_axes: list[JointAxis] | None = None, angular_axes: list[JointAxis] | None = None, name: str | None = None, parent_xform: wp.transform | None = None, child_xform: wp.transform | None = None, linear_compliance: float = 0.0, angular_compliance: float = 0.0, armature: float = 0.01, collision_filter_parent: bool = True, enabled: bool = True): ...
    def plot_articulation(self, show_body_names: bool = True, show_joint_names: bool = True, show_joint_types: bool = True, plot_shapes: bool = True, show_shape_types: bool = True, show_legend: bool = True): ...
    def collapse_fixed_joints(self, verbose=...): ...
    def add_muscle(self, bodies: list[int], positions: list[Vec3], f0: float, lm: float, lt: float, lmax: float, pen: float) -> float: ...
    def add_shape_plane(self, plane: Vec4 | tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0), pos: Vec3 | None = None, rot: Quat | None = None, width: float = 10.0, length: float = 10.0, body: int = -1, ke: float | None = None, kd: float | None = None, kf: float | None = None, ka: float | None = None, mu: float | None = None, restitution: float | None = None, thickness: float | None = None, has_ground_collision: bool = False, has_shape_collision: bool = True, is_visible: bool = True, collision_group: int = -1) -> int: ...
    def add_shape_sphere(self, body, pos: Vec3 | tuple[float, float, float] = (0.0, 0.0, 0.0), rot: Quat | tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0), radius: float = 1.0, density: float | None = None, ke: float | None = None, kd: float | None = None, kf: float | None = None, ka: float | None = None, mu: float | None = None, restitution: float | None = None, is_solid: bool = True, thickness: float | None = None, has_ground_collision: bool = True, has_shape_collision: bool = True, collision_group: int = -1, is_visible: bool = True) -> int: ...
    def add_shape_box(self, body: int, pos: Vec3 | tuple[float, float, float] = (0.0, 0.0, 0.0), rot: Quat | tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0), hx: float = 0.5, hy: float = 0.5, hz: float = 0.5, density: float | None = None, ke: float | None = None, kd: float | None = None, kf: float | None = None, ka: float | None = None, mu: float | None = None, restitution: float | None = None, is_solid: bool = True, thickness: float | None = None, has_ground_collision: bool = True, has_shape_collision: bool = True, collision_group: int = -1, is_visible: bool = True) -> int: ...
    def add_shape_capsule(self, body: int, pos: Vec3 | tuple[float, float, float] = (0.0, 0.0, 0.0), rot: Quat | tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0), radius: float = 1.0, half_height: float = 0.5, up_axis: int = 1, density: float | None = None, ke: float | None = None, kd: float | None = None, kf: float | None = None, ka: float | None = None, mu: float | None = None, restitution: float | None = None, is_solid: bool = True, thickness: float | None = None, has_ground_collision: bool = True, has_shape_collision: bool = True, collision_group: int = -1, is_visible: bool = True) -> int: ...
    def add_shape_cylinder(self, body: int, pos: Vec3 | tuple[float, float, float] = (0.0, 0.0, 0.0), rot: Quat | tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0), radius: float = 1.0, half_height: float = 0.5, up_axis: int = 1, density: float | None = None, ke: float | None = None, kd: float | None = None, kf: float | None = None, ka: float | None = None, mu: float | None = None, restitution: float | None = None, is_solid: bool = True, thickness: float | None = None, has_ground_collision: bool = True, has_shape_collision: bool = True, collision_group: int = -1, is_visible: bool = True) -> int: ...
    def add_shape_cone(self, body: int, pos: Vec3 | tuple[float, float, float] = (0.0, 0.0, 0.0), rot: Quat | tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0), radius: float = 1.0, half_height: float = 0.5, up_axis: int = 1, density: float | None = None, ke: float | None = None, kd: float | None = None, kf: float | None = None, ka: float | None = None, mu: float | None = None, restitution: float | None = None, is_solid: bool = True, thickness: float | None = None, has_ground_collision: bool = True, has_shape_collision: bool = True, collision_group: int = -1, is_visible: bool = True) -> int: ...
    def add_shape_mesh(self, body: int, pos: Vec3 | None = None, rot: Quat | None = None, mesh: Mesh | None = None, scale: Vec3 | None = None, density: float | None = None, ke: float | None = None, kd: float | None = None, kf: float | None = None, ka: float | None = None, mu: float | None = None, restitution: float | None = None, is_solid: bool = True, thickness: float | None = None, has_ground_collision: bool = True, has_shape_collision: bool = True, collision_group: int = -1, is_visible: bool = True) -> int: ...
    def add_shape_sdf(self, body: int, pos: Vec3 | tuple[float, float, float] = (0.0, 0.0, 0.0), rot: Quat | tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0), sdf: SDF | None = None, scale: Vec3 | tuple[float, float, float] = (1.0, 1.0, 1.0), density: float | None = None, ke: float | None = None, kd: float | None = None, kf: float | None = None, ka: float | None = None, mu: float | None = None, restitution: float | None = None, is_solid: bool = True, thickness: float | None = None, has_ground_collision: bool = True, has_shape_collision: bool = True, collision_group: int = -1, is_visible: bool = True) -> int: ...
    def add_particle(self, pos: Vec3, vel: Vec3, mass: float, radius: float | None = None, flags: wp.uint32 = ...) -> int: ...
    def add_spring(self, i: int, j, ke: float, kd: float, control: float): ...
    def add_triangle(self, i: int, j: int, k: int, tri_ke: float | None = None, tri_ka: float | None = None, tri_kd: float | None = None, tri_drag: float | None = None, tri_lift: float | None = None) -> float: ...
    def add_triangles(self, i: list[int], j: list[int], k: list[int], tri_ke: list[float] | None = None, tri_ka: list[float] | None = None, tri_kd: list[float] | None = None, tri_drag: list[float] | None = None, tri_lift: list[float] | None = None) -> list[float]: ...
    def add_tetrahedron(self, i: int, j: int, k: int, l: int, k_mu: float = 1000.0, k_lambda: float = 1000.0, k_damp: float = 0.0) -> float: ...
    def add_edge(self, i: int, j: int, k: int, l: int, rest: float | None = None, edge_ke: float | None = None, edge_kd: float | None = None) -> None: ...
    def add_edges(self, i, j, k, l, rest: list[float] | None = None, edge_ke: list[float] | None = None, edge_kd: list[float] | None = None) -> None: ...
    def add_cloth_grid(self, pos: Vec3, rot: Quat, vel: Vec3, dim_x: int, dim_y: int, cell_x: float, cell_y: float, mass: float, reverse_winding: bool = False, fix_left: bool = False, fix_right: bool = False, fix_top: bool = False, fix_bottom: bool = False, tri_ke: float | None = None, tri_ka: float | None = None, tri_kd: float | None = None, tri_drag: float | None = None, tri_lift: float | None = None, edge_ke: float | None = None, edge_kd: float | None = None, add_springs: bool = False, spring_ke: float | None = None, spring_kd: float | None = None, particle_radius: float | None = None) -> None: ...
    def add_cloth_mesh(self, pos: Vec3, rot: Quat, scale: float, vel: Vec3, vertices: list[Vec3], indices: list[int], density: float, edge_callback=None, face_callback=None, tri_ke: float | None = None, tri_ka: float | None = None, tri_kd: float | None = None, tri_drag: float | None = None, tri_lift: float | None = None, edge_ke: float | None = None, edge_kd: float | None = None, add_springs: bool = False, spring_ke: float | None = None, spring_kd: float | None = None, particle_radius: float | None = None) -> None: ...
    def add_particle_grid(self, pos: Vec3, rot: Quat, vel: Vec3, dim_x: int, dim_y: int, dim_z: int, cell_x: float, cell_y: float, cell_z: float, mass: float, jitter: float, radius_mean: float | None = None, radius_std: float = 0.0) -> None: ...
    def add_soft_grid(self, pos: Vec3, rot: Quat, vel: Vec3, dim_x: int, dim_y: int, dim_z: int, cell_x: float, cell_y: float, cell_z: float, density: float, k_mu: float, k_lambda: float, k_damp: float, fix_left: bool = False, fix_right: bool = False, fix_top: bool = False, fix_bottom: bool = False, tri_ke: float | None = None, tri_ka: float | None = None, tri_kd: float | None = None, tri_drag: float | None = None, tri_lift: float | None = None) -> None: ...
    def add_soft_mesh(self, pos: Vec3, rot: Quat, scale: float, vel: Vec3, vertices: list[Vec3], indices: list[int], density: float, k_mu: float, k_lambda: float, k_damp: float, tri_ke: float | None = None, tri_ka: float | None = None, tri_kd: float | None = None, tri_drag: float | None = None, tri_lift: float | None = None) -> None: ...
    def set_ground_plane(self, normal=None, offset: float = 0.0, ke: float | None = None, kd: float | None = None, kf: float | None = None, mu: float | None = None, restitution: float | None = None) -> None: ...
    def set_coloring(self, particle_color_groups) -> None: ...
    def color(self, include_bending: bool = False, balance_colors: bool = True, target_max_min_color_ratio: float = 1.1, coloring_algorithm=...) -> None: ...
    def finalize(self, device=None, requires_grad: bool = False) -> Model: ...
