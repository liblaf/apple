"""Run a 2D Stable Neo-Hookean muscle-fascia FEM simulation with self-contact.

Simulates a flat meat slab (100x11 elements) with a stiff fascia middle layer.
Uses analytical First Piola-Kirchhoff stress with NumPy vectorized computation.
Adds a lightweight boundary point-edge self-collision pass for fold/contact cases.
Real-time visualization with Pygame.

Material: Stable Neo-Hookean hyperelastic
  Psi(F) = mu/2 * (tr(F^T F) - 2) - mu * (J - 1) + lam/2 * (J - 1)^2
  P      = mu * F + (-mu + lam * (J - 1)) * cof(F)

Controls:
    Up/Down    Increase/decrease wind force
    R          Reset simulation
    SPACE      Pause/resume
    C          Toggle self-collision
    W          Toggle wireframe overlay
    ESC        Quit
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import pygame

type MeshData = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
type StateData = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]
type ContactResult = tuple[int, np.ndarray]


@dataclass(frozen=True, slots=True)
class SelfCollisionData:
    """Precomputed boundary topology for point-edge self-collision."""

    boundary_nodes: np.ndarray
    boundary_edges: np.ndarray
    candidate_edges: np.ndarray


@dataclass(frozen=True, slots=True)
class ActiveContacts:
    """Vectorized point-edge contacts active in the current configuration."""

    nodes: np.ndarray
    edge_a: np.ndarray
    edge_b: np.ndarray
    t: np.ndarray
    closest: np.ndarray
    normals: np.ndarray
    depth: np.ndarray
    node_weight: np.ndarray
    edge_a_weight: np.ndarray
    edge_b_weight: np.ndarray
    denom: np.ndarray


# ============================================================
# Simulation Parameters
# ============================================================

NX, NY = 100, 11  # element grid (cols x rows)
NVX, NVY = NX + 1, NY + 1  # node grid: 101 x 12
NV = NVX * NVY  # 1212 nodes
NF = 2 * NX * NY  # 2200 triangles

DOMAIN_W = 1.0  # domain width
DOMAIN_H = DOMAIN_W * NY / NX  # domain height = 0.11

# Material
E_MUSCLE = 1000.0  # Young's modulus (muscle)
E_FASCIA = E_MUSCLE * 300  # Young's modulus (fascia) = 100x
NU = 0.3  # Poisson's ratio
RHO = 10.0  # density (normalized)

# Time integration
DT = 5e-5  # timestep
DAMPING = 15.0  # velocity damping
SUBSTEPS = 100  # substeps per frame

# Wind
WIND_STEP = 1.0  # wind increment per key press
INITIAL_WIND = 0.0

# Self-collision
CONTACT_DISTANCE = 0.9 * DOMAIN_W / NX
CONTACT_STIFFNESS = 1e4
CONTACT_DAMPING = 0.35
CONTACT_PROJECTION_ITERS = 2
COLLISION_INTERVAL = 4
MAX_DRAWN_CONTACTS = 128
EPS = 1e-12


# ============================================================
# Display Parameters
# ============================================================

SCREEN_W, SCREEN_H = 1400, 700
SCALE = 1100.0  # pixels per simulation unit
OFFSET_X = (SCREEN_W - DOMAIN_W * SCALE) / 2
OFFSET_Y = SCREEN_H * 0.78  # y=0 line in screen space

# Colors
COLOR_BG = (12, 12, 22)
COLOR_MUSCLE = (165, 55, 55)
COLOR_FASCIA = (195, 205, 215)
COLOR_EDGE = (45, 45, 58)
COLOR_CONTACT = (255, 135, 70)
COLOR_TEXT = (220, 220, 230)
COLOR_TEXT_DIM = (130, 130, 145)
TEXT_ANTIALIAS = True


# ============================================================
# Mesh & Material Setup
# ============================================================


def build_mesh() -> MeshData:
    """Build triangular mesh from a structured quad grid.

    Each quad cell (col, row) is split into 2 triangles:
      Triangle 0: [bottom-left, bottom-right, top-right]
      Triangle 1: [top-right, top-left, bottom-left]

    The middle row (row == 5) is assigned fascia material;
    all other rows are muscle.

    Returns:
        f2v:       (NF, 3) int array, triangle-to-vertex mapping
        is_fascia: (NF,) bool array, True for fascia elements
        elem_mu:   (NF,) float array, Lame parameter mu per element
        elem_lam:  (NF,) float array, Lame parameter lambda per element
    """
    f2v = np.zeros((NF, 3), dtype=np.int32)
    is_fascia = np.zeros(NF, dtype=bool)

    for row in range(NY):
        for col in range(NX):
            # Quad corner node indices
            a = row * NVX + col  # bottom-left
            b = a + 1  # bottom-right
            c = (row + 1) * NVX + col + 1  # top-right
            d = (row + 1) * NVX + col  # top-left

            k = (row * NX + col) * 2
            f2v[k] = [a, b, c]
            f2v[k + 1] = [c, d, a]

            # Middle layer = fascia
            if row == 5:
                is_fascia[k] = True
                is_fascia[k + 1] = True

    # Compute per-element Lame parameters
    E = np.where(is_fascia, E_FASCIA, E_MUSCLE)
    elem_mu = E / (2.0 * (1.0 + NU))
    elem_lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

    return f2v, is_fascia, elem_mu, elem_lam


# ============================================================
# State Initialization
# ============================================================


def init_state(f2v: np.ndarray) -> StateData:
    """Initialize node positions and precompute reference configuration.

    Precomputes:
      - Dm_inv:   (NF, 2, 2) inverse of reference edge matrix
      - Dm_inv_T: (NF, 2, 2) transpose of Dm_inv (cached for reuse)
      - W:        (NF,)      reference triangle areas

    Returns tuple:
        (pos, rest_pos, vel, Dm_inv, Dm_inv_T, W, fixed_mask, bottom_mask)
    """
    dx = DOMAIN_W / NX
    dy = DOMAIN_H / NY

    # Build node positions on a regular grid
    cols = np.arange(NVX) * dx
    rows = np.arange(NVY) * dy
    grid_x, grid_y = np.meshgrid(cols, rows)  # (NVY, NVX) each
    pos = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # (NV, 2)

    rest_pos = pos.copy()
    vel = np.zeros_like(pos)

    # --- Precompute reference configuration for all elements ---
    ia, ib, ic = f2v[:, 0], f2v[:, 1], f2v[:, 2]
    xa, xb, xc = pos[ia], pos[ib], pos[ic]

    # Reference edge matrix: Dm = [Xa - Xc | Xb - Xc]
    Dm = np.empty((NF, 2, 2))
    Dm[:, :, 0] = xa - xc
    Dm[:, :, 1] = xb - xc

    # Batch invert 2x2 matrices
    det_Dm = Dm[:, 0, 0] * Dm[:, 1, 1] - Dm[:, 0, 1] * Dm[:, 1, 0]
    inv_det = 1.0 / det_Dm

    Dm_inv = np.empty_like(Dm)
    Dm_inv[:, 0, 0] = Dm[:, 1, 1] * inv_det
    Dm_inv[:, 0, 1] = -Dm[:, 0, 1] * inv_det
    Dm_inv[:, 1, 0] = -Dm[:, 1, 0] * inv_det
    Dm_inv[:, 1, 1] = Dm[:, 0, 0] * inv_det

    Dm_inv_T = np.ascontiguousarray(Dm_inv.transpose(0, 2, 1))
    W = 0.5 * np.abs(det_Dm)  # reference triangle areas

    # --- Boundary masks ---
    node_col = np.arange(NV) % NVX
    node_row = np.arange(NV) // NVX
    fixed_mask = (node_col == 0) | (node_col == NX)  # left & right edges
    bottom_mask = node_row == 0  # bottom row

    return pos, rest_pos, vel, Dm_inv, Dm_inv_T, W, fixed_mask, bottom_mask


# ============================================================
# Self-Collision Setup & Solvers
# ============================================================


def build_boundary_edges() -> np.ndarray:
    """Build oriented edges around the outer boundary of the slab."""
    edges = [(col, col + 1) for col in range(NX)]
    edges.extend((row * NVX + NX, (row + 1) * NVX + NX) for row in range(NY))

    top_row = NY * NVX
    edges.extend((top_row + col, top_row + col - 1) for col in range(NX, 0, -1))
    edges.extend((row * NVX, (row - 1) * NVX) for row in range(NY, 0, -1))

    return np.asarray(edges, dtype=np.int32)


def build_self_collision_data(rest_pos: np.ndarray) -> SelfCollisionData:
    """Precompute boundary nodes and candidate non-incident boundary edges."""
    boundary_edges = build_boundary_edges()
    boundary_nodes = np.unique(boundary_edges)
    edge_a = boundary_edges[:, 0]
    edge_b = boundary_edges[:, 1]
    candidate_edges = np.empty(
        (boundary_nodes.size, boundary_edges.shape[0]), dtype=bool
    )

    for row, node in enumerate(boundary_nodes):
        p = rest_pos[node]
        start = rest_pos[edge_a]
        end = rest_pos[edge_b]
        _, _, rest_distance = closest_points_on_edges(p, start, end)
        non_incident = (edge_a != node) & (edge_b != node)
        separated_at_rest = rest_distance > CONTACT_DISTANCE
        candidate_edges[row] = non_incident & separated_at_rest

    return SelfCollisionData(
        boundary_nodes=boundary_nodes,
        boundary_edges=boundary_edges,
        candidate_edges=candidate_edges,
    )


def closest_points_on_edges(
    point: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return closest edge parameters, points, and distances for one point."""
    edge = edge_end - edge_start
    length_sq = np.einsum("ij,ij->i", edge, edge)
    rel = point - edge_start
    t = np.divide(
        np.einsum("ij,ij->i", rel, edge),
        length_sq,
        out=np.zeros_like(length_sq),
        where=length_sq > EPS,
    )
    t = np.clip(t, 0.0, 1.0)
    closest = edge_start + t[:, None] * edge
    delta = point - closest
    distance = np.linalg.norm(delta, axis=1)
    return t, closest, distance


def contact_normals(
    point: np.ndarray,
    closest: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    """Build stable point-edge contact normals."""
    delta = point - closest
    normals = np.zeros_like(delta)
    safe = distance > EPS
    normals[safe] = delta[safe] / distance[safe, None]

    if np.any(~safe):
        edge = edge_end[~safe] - edge_start[~safe]
        fallback = np.column_stack([-edge[:, 1], edge[:, 0]])
        fallback_norm = np.linalg.norm(fallback, axis=1)
        fallback = np.divide(
            fallback,
            fallback_norm[:, None],
            out=np.tile(np.array([[0.0, 1.0]]), (fallback.shape[0], 1)),
            where=fallback_norm[:, None] > EPS,
        )
        normals[~safe] = fallback

    return normals


def broadphase_contact_pairs(
    pos: np.ndarray,
    collision: SelfCollisionData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Find boundary point-edge pairs with overlapping contact AABBs."""
    edge_a_all = collision.boundary_edges[:, 0]
    edge_b_all = collision.boundary_edges[:, 1]
    point_pos = pos[collision.boundary_nodes]
    edge_start = pos[edge_a_all]
    edge_end = pos[edge_b_all]

    min_corner = np.minimum(edge_start, edge_end) - CONTACT_DISTANCE
    max_corner = np.maximum(edge_start, edge_end) + CONTACT_DISTANCE
    overlap_x = (point_pos[:, 0, None] >= min_corner[None, :, 0]) & (
        point_pos[:, 0, None] <= max_corner[None, :, 0]
    )
    overlap_y = (point_pos[:, 1, None] >= min_corner[None, :, 1]) & (
        point_pos[:, 1, None] <= max_corner[None, :, 1]
    )
    broadphase = collision.candidate_edges & overlap_x & overlap_y
    node_rows, edge_cols = np.nonzero(broadphase)
    if node_rows.size == 0:
        return None

    return (
        collision.boundary_nodes[node_rows],
        edge_a_all[edge_cols],
        edge_b_all[edge_cols],
    )


def compute_self_collision_forces(
    pos: np.ndarray,
    vel: np.ndarray,
    collision: SelfCollisionData,
    fixed_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Compute soft penalty forces for boundary point-edge self-contact."""
    forces = np.zeros_like(pos)
    broadphase = broadphase_contact_pairs(pos, collision)
    if broadphase is None:
        return forces, 0

    nodes, edge_a, edge_b = broadphase
    t, closest, distance = closest_points_on_edges(pos[nodes], pos[edge_a], pos[edge_b])
    active = distance < CONTACT_DISTANCE
    if not np.any(active):
        return forces, 0

    nodes = nodes[active]
    edge_a = edge_a[active]
    edge_b = edge_b[active]
    t = t[active]
    closest = closest[active]
    distance = distance[active]
    normal = contact_normals(pos[nodes], closest, pos[edge_a], pos[edge_b], distance)

    depth = CONTACT_DISTANCE - distance
    edge_vel = (1.0 - t[:, None]) * vel[edge_a] + t[:, None] * vel[edge_b]
    rel_normal_vel = np.einsum("ij,ij->i", vel[nodes] - edge_vel, normal)
    damping = CONTACT_DAMPING * np.maximum(0.0, -rel_normal_vel)
    contact_force = (CONTACT_STIFFNESS * depth + damping)[:, None] * normal

    node_free = ~fixed_mask[nodes]
    if np.any(node_free):
        np.add.at(forces, nodes[node_free], contact_force[node_free])

    edge_a_free = ~fixed_mask[edge_a]
    if np.any(edge_a_free):
        np.add.at(
            forces,
            edge_a[edge_a_free],
            -(1.0 - t[edge_a_free])[:, None] * contact_force[edge_a_free],
        )

    edge_b_free = ~fixed_mask[edge_b]
    if np.any(edge_b_free):
        np.add.at(
            forces,
            edge_b[edge_b_free],
            -t[edge_b_free][:, None] * contact_force[edge_b_free],
        )

    return forces, int(np.count_nonzero(active))


def find_active_contacts(
    pos: np.ndarray,
    collision: SelfCollisionData,
    fixed_mask: np.ndarray,
) -> ActiveContacts | None:
    """Find movable point-edge contacts for the current positions."""
    broadphase = broadphase_contact_pairs(pos, collision)
    if broadphase is None:
        return None

    nodes, edge_a, edge_b = broadphase
    t, closest, distance = closest_points_on_edges(pos[nodes], pos[edge_a], pos[edge_b])
    active = distance < CONTACT_DISTANCE
    if not np.any(active):
        return None

    nodes = nodes[active]
    edge_a = edge_a[active]
    edge_b = edge_b[active]
    t = t[active]
    closest = closest[active]
    distance = distance[active]
    normals = contact_normals(pos[nodes], closest, pos[edge_a], pos[edge_b], distance)
    depth = CONTACT_DISTANCE - distance

    node_weight = (~fixed_mask[nodes]).astype(pos.dtype)
    edge_a_weight = (~fixed_mask[edge_a]).astype(pos.dtype)
    edge_b_weight = (~fixed_mask[edge_b]).astype(pos.dtype)
    denom = node_weight + ((1.0 - t) ** 2) * edge_a_weight + (t**2) * edge_b_weight
    movable = denom > EPS
    if not np.any(movable):
        return None

    return ActiveContacts(
        nodes=nodes[movable],
        edge_a=edge_a[movable],
        edge_b=edge_b[movable],
        t=t[movable],
        closest=closest[movable],
        normals=normals[movable],
        depth=depth[movable],
        node_weight=node_weight[movable],
        edge_a_weight=edge_a_weight[movable],
        edge_b_weight=edge_b_weight[movable],
        denom=denom[movable],
    )


def project_contact_positions(pos: np.ndarray, contacts: ActiveContacts) -> np.ndarray:
    """Apply position corrections and return the raw correction vectors."""
    correction = contacts.depth[:, None] * contacts.normals
    np.add.at(
        pos,
        contacts.nodes,
        (contacts.node_weight / contacts.denom)[:, None] * correction,
    )
    np.add.at(
        pos,
        contacts.edge_a,
        -((1.0 - contacts.t) * contacts.edge_a_weight / contacts.denom)[:, None]
        * correction,
    )
    np.add.at(
        pos,
        contacts.edge_b,
        -(contacts.t * contacts.edge_b_weight / contacts.denom)[:, None] * correction,
    )
    return correction


def project_contact_velocities(vel: np.ndarray, contacts: ActiveContacts) -> None:
    """Remove inward normal velocity from active contact pairs."""
    edge_vel = (1.0 - contacts.t[:, None]) * vel[contacts.edge_a] + contacts.t[
        :, None
    ] * vel[contacts.edge_b]
    rel_normal_vel = np.einsum(
        "ij,ij->i",
        vel[contacts.nodes] - edge_vel,
        contacts.normals,
    )
    approaching = rel_normal_vel < 0.0
    if not np.any(approaching):
        return

    velocity_correction = (
        -rel_normal_vel[approaching, None] * contacts.normals[approaching]
    )
    np.add.at(
        vel,
        contacts.nodes[approaching],
        (contacts.node_weight[approaching] / contacts.denom[approaching])[:, None]
        * velocity_correction,
    )
    np.add.at(
        vel,
        contacts.edge_a[approaching],
        -(
            (1.0 - contacts.t[approaching])
            * contacts.edge_a_weight[approaching]
            / contacts.denom[approaching]
        )[:, None]
        * velocity_correction,
    )
    np.add.at(
        vel,
        contacts.edge_b[approaching],
        -(
            contacts.t[approaching]
            * contacts.edge_b_weight[approaching]
            / contacts.denom[approaching]
        )[:, None]
        * velocity_correction,
    )


def extend_contact_points(
    contact_points: list[np.ndarray],
    contacts: ActiveContacts,
    correction: np.ndarray,
) -> None:
    """Append a bounded number of contact points for rendering."""
    remaining = MAX_DRAWN_CONTACTS - len(contact_points)
    if remaining <= 0:
        return

    contact_points.extend(contacts.closest[:remaining] + 0.5 * correction[:remaining])


def project_self_collisions(
    pos: np.ndarray,
    vel: np.ndarray,
    collision: SelfCollisionData,
    fixed_mask: np.ndarray,
) -> ContactResult:
    """Project boundary point-edge pairs back to the contact distance."""
    contact_count = 0
    contact_points: list[np.ndarray] = []

    for _ in range(CONTACT_PROJECTION_ITERS):
        contacts = find_active_contacts(pos, collision, fixed_mask)
        if contacts is None:
            continue

        correction = project_contact_positions(pos, contacts)
        project_contact_velocities(vel, contacts)
        contact_count += int(contacts.nodes.size)
        extend_contact_points(contact_points, contacts, correction)

    if not contact_points:
        return contact_count, np.empty((0, 2), dtype=pos.dtype)

    return contact_count, np.asarray(contact_points, dtype=pos.dtype)


# ============================================================
# Force Computation (Analytical Stable Neo-Hookean)
# ============================================================


def compute_elastic_forces(
    pos: np.ndarray,
    f2v: np.ndarray,
    Dm_inv: np.ndarray,
    Dm_inv_T: np.ndarray,
    W: np.ndarray,
    elem_mu: np.ndarray,
    elem_lam: np.ndarray,
) -> np.ndarray:
    """Compute internal elastic forces using Stable Neo-Hookean stress.

    For each element:
      1. F = Ds @ Dm_inv
      2. J = det(F)
      3. cof(F) = dJ/dF
      4. P = mu*F + (-mu + lam*(J - 1))*cof(F)
      5. H = -W * P @ Dm_inv^T
      6. Scatter H columns to global force vector

    The polynomial determinant term avoids the log(J) singularity from the
    original Neo-Hookean script, so this force law does not divide by or clamp J.

    Returns:
        forces: (NV, 2) array of nodal forces
    """
    # Gather deformed vertex positions
    pa = pos[f2v[:, 0]]  # (NF, 2)
    pb = pos[f2v[:, 1]]
    pc = pos[f2v[:, 2]]

    # Deformed edge matrix: Ds = [xa - xc | xb - xc]
    Ds = np.empty((NF, 2, 2))
    Ds[:, :, 0] = pa - pc
    Ds[:, :, 1] = pb - pc

    # Deformation gradient: F = Ds @ Dm_inv
    F = Ds @ Dm_inv  # (NF, 2, 2)

    # Determinant: J = det(F)
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]  # (NF,)

    # Cofactor matrix dJ/dF for 2x2 F = [[a, b], [c, d]]:
    #   cof(F) = [[d, -c], [-b, a]]
    cof_F = np.empty_like(F)
    cof_F[:, 0, 0] = F[:, 1, 1]
    cof_F[:, 0, 1] = -F[:, 1, 0]
    cof_F[:, 1, 0] = -F[:, 0, 1]
    cof_F[:, 1, 1] = F[:, 0, 0]

    # First Piola-Kirchhoff stress:
    #   P = mu*F + (-mu + lam*(J - 1))*cof(F)
    mu = elem_mu[:, None, None]  # (NF, 1, 1) for broadcasting
    lam = elem_lam[:, None, None]
    j_minus_1 = (J - 1.0)[:, None, None]
    c = -mu + lam * j_minus_1
    P = mu * F + c * cof_F

    # Nodal force matrix: H = -W * P @ Dm_inv_T
    H = -W[:, None, None] * (P @ Dm_inv_T)  # (NF, 2, 2)

    # Scatter forces to global array
    #   f_a = H[:, :, 0], f_b = H[:, :, 1], f_c = -(f_a + f_b)
    forces = np.zeros_like(pos)
    fa = H[:, :, 0]  # (NF, 2)
    fb = H[:, :, 1]
    fc = -(fa + fb)

    np.add.at(forces, f2v[:, 0], fa)
    np.add.at(forces, f2v[:, 1], fb)
    np.add.at(forces, f2v[:, 2], fc)

    return forces


# ============================================================
# Time Stepping
# ============================================================


def substep(
    pos: np.ndarray,
    vel: np.ndarray,
    rest_pos: np.ndarray,
    f2v: np.ndarray,
    Dm_inv: np.ndarray,
    Dm_inv_T: np.ndarray,
    W: np.ndarray,
    elem_mu: np.ndarray,
    elem_lam: np.ndarray,
    fixed_mask: np.ndarray,
    bottom_mask: np.ndarray,
    collision: SelfCollisionData,
    wind: float,
    mass: float,
    *,
    collision_enabled: bool,
) -> ContactResult:
    """Single substep of Symplectic Euler integration.

    1. Compute elastic forces (analytical Stable Neo-Hookean)
    2. Add optional self-collision penalty forces
    3. Add wind force on bottom nodes
    4. Update velocity: vel += dt * acc, then apply damping
    5. Enforce fixed boundary: vel = 0 at left/right edges
    6. Update position: pos += dt * vel
    7. Project self-collision and restore fixed node positions
    """
    # Internal elastic forces
    forces = compute_elastic_forces(pos, f2v, Dm_inv, Dm_inv_T, W, elem_mu, elem_lam)
    contact_count = 0
    contact_points = np.empty((0, 2), dtype=pos.dtype)

    if collision_enabled:
        collision_forces, force_contacts = compute_self_collision_forces(
            pos,
            vel,
            collision,
            fixed_mask,
        )
        forces += collision_forces
        contact_count += force_contacts

    # External: upward wind on bottom nodes
    forces[bottom_mask, 1] += wind

    # Acceleration
    acc = forces / mass

    # Symplectic Euler: update velocity first, then position
    vel += DT * acc
    vel *= np.exp(-DT * DAMPING)

    # Fixed boundary: zero velocity
    vel[fixed_mask] = 0.0

    # Update positions
    pos += DT * vel

    if collision_enabled:
        projection_contacts, contact_points = project_self_collisions(
            pos,
            vel,
            collision,
            fixed_mask,
        )
        contact_count += projection_contacts

    # Restore fixed node positions exactly
    pos[fixed_mask] = rest_pos[fixed_mask]
    vel[fixed_mask] = 0.0

    return contact_count, contact_points


# ============================================================
# Coordinate Transform
# ============================================================


def sim_to_screen(sim_pos: np.ndarray) -> np.ndarray:
    """Convert simulation coordinates to screen pixel coordinates (y-flipped)."""
    screen = np.empty_like(sim_pos)
    screen[:, 0] = sim_pos[:, 0] * SCALE + OFFSET_X
    screen[:, 1] = OFFSET_Y - sim_pos[:, 1] * SCALE
    return screen.astype(np.int32)


def handle_keydown(
    key: int,
    pos: np.ndarray,
    rest_pos: np.ndarray,
    vel: np.ndarray,
    wind: float,
    *,
    paused: bool,
    show_wireframe: bool,
    collision_enabled: bool,
) -> tuple[bool, float, bool, bool, bool]:
    """Handle one key press and return updated loop state."""
    running = True

    if key == pygame.K_ESCAPE:
        running = False
    elif key == pygame.K_UP:
        wind += WIND_STEP
    elif key == pygame.K_DOWN:
        wind = max(0.0, wind - WIND_STEP)
    elif key == pygame.K_r:
        pos[:] = rest_pos
        vel[:] = 0.0
        wind = INITIAL_WIND
    elif key == pygame.K_SPACE:
        paused = not paused
    elif key == pygame.K_c:
        collision_enabled = not collision_enabled
    elif key == pygame.K_w:
        show_wireframe = not show_wireframe

    return running, wind, paused, show_wireframe, collision_enabled


def handle_events(
    pos: np.ndarray,
    rest_pos: np.ndarray,
    vel: np.ndarray,
    wind: float,
    *,
    paused: bool,
    show_wireframe: bool,
    collision_enabled: bool,
) -> tuple[bool, float, bool, bool, bool]:
    """Handle input events and return updated loop state."""
    running = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            running, wind, paused, show_wireframe, collision_enabled = handle_keydown(
                event.key,
                pos,
                rest_pos,
                vel,
                wind,
                paused=paused,
                show_wireframe=show_wireframe,
                collision_enabled=collision_enabled,
            )

    return running, wind, paused, show_wireframe, collision_enabled


def advance_simulation(
    pos: np.ndarray,
    vel: np.ndarray,
    rest_pos: np.ndarray,
    f2v: np.ndarray,
    Dm_inv: np.ndarray,
    Dm_inv_T: np.ndarray,
    W: np.ndarray,
    elem_mu: np.ndarray,
    elem_lam: np.ndarray,
    fixed_mask: np.ndarray,
    bottom_mask: np.ndarray,
    collision: SelfCollisionData,
    wind: float,
    mass: float,
    *,
    collision_enabled: bool,
) -> tuple[float, float, np.ndarray]:
    """Run one frame worth of simulation substeps and return contact stats."""
    t0 = time.perf_counter()
    frame_contacts = 0
    contact_points = np.empty((0, 2), dtype=pos.dtype)

    for step in range(SUBSTEPS):
        substep_contacts, substep_points = substep(
            pos,
            vel,
            rest_pos,
            f2v,
            Dm_inv,
            Dm_inv_T,
            W,
            elem_mu,
            elem_lam,
            fixed_mask,
            bottom_mask,
            collision,
            wind,
            mass,
            # collision_enabled=collision_enabled and step % COLLISION_INTERVAL == 0,
            collision_enabled=collision_enabled,
        )
        frame_contacts += substep_contacts
        if substep_points.size:
            contact_points = substep_points

    return (
        (time.perf_counter() - t0) * 1000.0,
        frame_contacts / SUBSTEPS,
        contact_points,
    )


def draw_mesh(
    screen: pygame.Surface,
    pos: np.ndarray,
    f2v: np.ndarray,
    muscle_elems: np.ndarray,
    fascia_elems: np.ndarray,
    *,
    show_wireframe: bool,
) -> None:
    """Draw the current triangular mesh."""
    scr_pts = sim_to_screen(pos)
    tri_verts = scr_pts[f2v]  # (NF, 3, 2), all triangle vertices

    for idx in muscle_elems:
        pygame.draw.polygon(screen, COLOR_MUSCLE, tri_verts[idx].tolist())

    for idx in fascia_elems:
        pygame.draw.polygon(screen, COLOR_FASCIA, tri_verts[idx].tolist())

    if show_wireframe:
        for i in range(NF):
            pygame.draw.polygon(screen, COLOR_EDGE, tri_verts[i].tolist(), 1)


def draw_contacts(screen: pygame.Surface, contact_points: np.ndarray) -> None:
    """Draw recent self-collision contact points."""
    if contact_points.size == 0:
        return

    for point in sim_to_screen(contact_points):
        pygame.draw.circle(screen, COLOR_CONTACT, point.tolist(), 3)


def draw_hud(
    screen: pygame.Surface,
    font_title: pygame.font.Font,
    font_hud: pygame.font.Font,
    wind: float,
    fps: float,
    sim_ms: float,
    contacts: float,
    *,
    paused: bool,
    collision_enabled: bool,
) -> None:
    """Draw the interactive simulation HUD."""
    status = "PAUSED" if paused else "RUNNING"
    collision_status = "ON" if collision_enabled else "OFF"
    hud_lines = [
        (
            font_title,
            f"Wind: {wind:.2f}  |  FPS: {fps:.0f}  |  Sim: {sim_ms:.1f}ms  |  Contacts: {contacts}  [{status}]",
            COLOR_TEXT,
        ),
        (
            font_hud,
            f"[Up/Down] Wind   [C] Collision {collision_status}   [R] Reset   [SPACE] Pause   [W] Wireframe   [ESC] Quit",
            COLOR_TEXT_DIM,
        ),
    ]

    y_pos = 12
    for font, text, color in hud_lines:
        surf = font.render(text, TEXT_ANTIALIAS, color)
        screen.blit(surf, (15, y_pos))
        y_pos += 28


def draw_frame(
    screen: pygame.Surface,
    pos: np.ndarray,
    f2v: np.ndarray,
    muscle_elems: np.ndarray,
    fascia_elems: np.ndarray,
    contact_points: np.ndarray,
    font_title: pygame.font.Font,
    font_hud: pygame.font.Font,
    wind: float,
    fps: float,
    sim_ms: float,
    contacts: float,
    *,
    show_wireframe: bool,
    paused: bool,
    collision_enabled: bool,
) -> None:
    """Draw one complete frame."""
    screen.fill(COLOR_BG)
    draw_mesh(
        screen,
        pos,
        f2v,
        muscle_elems,
        fascia_elems,
        show_wireframe=show_wireframe,
    )
    draw_contacts(screen, contact_points)
    draw_hud(
        screen,
        font_title,
        font_hud,
        wind,
        fps,
        sim_ms,
        contacts,
        paused=paused,
        collision_enabled=collision_enabled,
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    # --- Build mesh & initialize state ---
    f2v, is_fascia, elem_mu, elem_lam = build_mesh()
    pos, rest_pos, vel, Dm_inv, Dm_inv_T, W, fixed_mask, bottom_mask = init_state(f2v)
    collision = build_self_collision_data(rest_pos)

    dx = DOMAIN_W / NX
    mass = RHO * dx * dx  # lumped mass per node
    wind = INITIAL_WIND

    # Precompute element group indices for rendering
    muscle_elems = np.where(~is_fascia)[0]
    fascia_elems = np.where(is_fascia)[0]

    # --- Pygame initialization ---
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(
        "Flesh Simulation - 2D Muscle & Fascia (Stable Neo-Hookean FEM + Self-Collision)"
    )
    clock = pygame.time.Clock()

    # Use default font (avoids SysFont registry bug on Python 3.14/Windows)
    font_hud = pygame.font.Font(None, 22)
    font_title = pygame.font.Font(None, 28)

    running = True
    paused = False
    show_wireframe = False
    collision_enabled = True
    sim_ms = 0.0
    contacts = 0
    contact_points = np.empty((0, 2), dtype=pos.dtype)

    # --- Main loop ---
    while running:
        # ---- Event Handling ----
        running, wind, paused, show_wireframe, collision_enabled = handle_events(
            pos,
            rest_pos,
            vel,
            wind,
            paused=paused,
            show_wireframe=show_wireframe,
            collision_enabled=collision_enabled,
        )

        # ---- Simulation ----
        if not paused:
            sim_ms, contacts, contact_points = advance_simulation(
                pos,
                vel,
                rest_pos,
                f2v,
                Dm_inv,
                Dm_inv_T,
                W,
                elem_mu,
                elem_lam,
                fixed_mask,
                bottom_mask,
                collision,
                wind,
                mass,
                collision_enabled=collision_enabled,
            )

        # ---- Rendering ----
        draw_frame(
            screen,
            pos,
            f2v,
            muscle_elems,
            fascia_elems,
            contact_points,
            font_title,
            font_hud,
            wind,
            clock.get_fps(),
            sim_ms,
            contacts,
            show_wireframe=show_wireframe,
            paused=paused,
            collision_enabled=collision_enabled,
        )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
