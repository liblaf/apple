"""
2D Muscle-Fascia Neohookean FEM Simulation
==========================================

Simulates a flat meat slab (100x11 elements) with a stiff fascia middle layer.
Uses analytical First Piola-Kirchhoff stress with NumPy vectorized computation.
Real-time visualization with Pygame.

Material: Neohookean hyperelastic
  Psi(F) = mu/2 * (tr(F^T F) - 2) - mu * ln(J) + lam/2 * (ln J)^2
  P      = mu * (F - F^{-T}) + lam * ln(J) * F^{-T}

Controls:
    Up/Down    Increase/decrease wind force
    R          Reset simulation
    SPACE      Pause/resume
    W          Toggle wireframe overlay
    ESC        Quit
"""

import sys
import time

import numpy as np
import pygame

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
E_FASCIA = E_MUSCLE * 200  # Young's modulus (fascia) = 100x
NU = 0.3  # Poisson's ratio
RHO = 10.0  # density (normalized)

# Time integration
DT = 5e-5  # timestep
DAMPING = 15.0  # velocity damping
SUBSTEPS = 100  # substeps per frame

# Wind
WIND_STEP = 0.1  # wind increment per key press
INITIAL_WIND = 0.0


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
COLOR_TEXT = (220, 220, 230)
COLOR_TEXT_DIM = (130, 130, 145)


# ============================================================
# Mesh & Material Setup
# ============================================================


def build_mesh():
    """
    Build triangular mesh from a structured quad grid.

    Each quad cell (col, row) is split into 2 triangles:
      Triangle 0: [bottom-left, bottom-right, top-right]
      Triangle 1: [top-right, top-left, bottom-left]

    The middle row (row == 5) is assigned fascia material;
    all other rows are muscle.

    Returns:
        f2v:       (NF, 3) int array — triangle-to-vertex mapping
        is_fascia: (NF,) bool array  — True for fascia elements
        elem_mu:   (NF,) float array — Lame parameter mu per element
        elem_lam:  (NF,) float array — Lame parameter lambda per element
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


def init_state(f2v):
    """
    Initialize node positions and precompute reference configuration.

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

    # Reference edge matrix:  Dm = [Xa - Xc | Xb - Xc]
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
# Force Computation (Analytical Neohookean)
# ============================================================


def compute_elastic_forces(pos, f2v, Dm_inv, Dm_inv_T, W, elem_mu, elem_lam):
    """
    Compute internal elastic forces using the analytical Neohookean stress.

    For each element:
      1. F = Ds @ Dm_inv              (deformation gradient)
      2. J = det(F)
      3. P = mu*(F - F^{-T}) + lam*ln(J)*F^{-T}   (1st Piola-Kirchhoff stress)
      4. H = -W * P @ Dm_inv^T        (nodal force matrix)
      5. Scatter H columns to global force vector

    All operations are vectorized over the NF elements.

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
    J_safe = np.maximum(J, 1e-8)  # clamp to avoid log(0)

    # F^{-T}  (inverse transpose, computed via cofactor / J)
    inv_J = 1.0 / J_safe
    F_inv_T = np.empty_like(F)
    F_inv_T[:, 0, 0] = F[:, 1, 1] * inv_J
    F_inv_T[:, 0, 1] = -F[:, 1, 0] * inv_J
    F_inv_T[:, 1, 0] = -F[:, 0, 1] * inv_J
    F_inv_T[:, 1, 1] = F[:, 0, 0] * inv_J

    # First Piola-Kirchhoff stress:
    #   P = mu * (F - F_inv_T) + lam * ln(J) * F_inv_T
    mu = elem_mu[:, None, None]  # (NF, 1, 1) for broadcasting
    lam = elem_lam[:, None, None]
    log_J = np.log(J_safe)[:, None, None]

    P = mu * (F - F_inv_T) + lam * log_J * F_inv_T

    # Nodal force matrix: H = -W * P @ Dm_inv_T
    H = -W[:, None, None] * (P @ Dm_inv_T)  # (NF, 2, 2)

    # Scatter forces to global array
    #   f_a = H[:, :, 0],  f_b = H[:, :, 1],  f_c = -(f_a + f_b)
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
    wind,
    mass,
):
    """
    Single substep of Symplectic Euler integration.

    1. Compute elastic forces (analytical Neohookean)
    2. Add wind force on bottom nodes
    3. Update velocity: vel += dt * acc, then apply damping
    4. Enforce fixed boundary: vel = 0 at left/right edges
    5. Update position: pos += dt * vel
    6. Restore fixed node positions
    """
    # Internal elastic forces
    forces = compute_elastic_forces(pos, f2v, Dm_inv, Dm_inv_T, W, elem_mu, elem_lam)

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

    # Restore fixed node positions exactly
    pos[fixed_mask] = rest_pos[fixed_mask]


# ============================================================
# Coordinate Transform
# ============================================================


def sim_to_screen(sim_pos):
    """Convert simulation coordinates to screen pixel coordinates (y-flipped)."""
    screen = np.empty_like(sim_pos)
    screen[:, 0] = sim_pos[:, 0] * SCALE + OFFSET_X
    screen[:, 1] = OFFSET_Y - sim_pos[:, 1] * SCALE
    return screen.astype(np.int32)


# ============================================================
# Main
# ============================================================


def main():
    # --- Build mesh & initialize state ---
    f2v, is_fascia, elem_mu, elem_lam = build_mesh()
    pos, rest_pos, vel, Dm_inv, Dm_inv_T, W, fixed_mask, bottom_mask = init_state(f2v)

    dx = DOMAIN_W / NX
    mass = RHO * dx * dx  # lumped mass per node
    wind = INITIAL_WIND

    # Precompute element group indices for rendering
    muscle_elems = np.where(~is_fascia)[0]
    fascia_elems = np.where(is_fascia)[0]

    # --- Pygame initialization ---
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Flesh Simulation — 2D Muscle & Fascia (Neohookean FEM)")
    clock = pygame.time.Clock()

    # Use default font (avoids SysFont registry bug on Python 3.14/Windows)
    font_hud = pygame.font.Font(None, 22)
    font_title = pygame.font.Font(None, 28)

    running = True
    paused = False
    show_wireframe = False
    sim_ms = 0.0

    # --- Main loop ---
    while running:
        # ---- Event Handling ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    wind += WIND_STEP
                elif event.key == pygame.K_DOWN:
                    wind = max(0.0, wind - WIND_STEP)
                elif event.key == pygame.K_r:
                    pos[:] = rest_pos
                    vel[:] = 0.0
                    wind = INITIAL_WIND
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_w:
                    show_wireframe = not show_wireframe

        # ---- Simulation ----
        if not paused:
            t0 = time.perf_counter()
            for _ in range(SUBSTEPS):
                substep(
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
                    wind,
                    mass,
                )
            sim_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Rendering ----
        screen.fill(COLOR_BG)

        # Convert positions to screen coordinates
        scr_pts = sim_to_screen(pos)
        tri_verts = scr_pts[f2v]  # (NF, 3, 2)  — all triangle vertices

        # Draw filled triangles: muscle
        for idx in muscle_elems:
            pygame.draw.polygon(screen, COLOR_MUSCLE, tri_verts[idx].tolist())

        # Draw filled triangles: fascia
        for idx in fascia_elems:
            pygame.draw.polygon(screen, COLOR_FASCIA, tri_verts[idx].tolist())

        # Optional wireframe overlay
        if show_wireframe:
            for i in range(NF):
                pygame.draw.polygon(screen, COLOR_EDGE, tri_verts[i].tolist(), 1)

        # ---- HUD ----
        fps = clock.get_fps()
        status = "PAUSED" if paused else "RUNNING"

        hud_lines = [
            (
                font_title,
                f"Wind: {wind:.2f}  |  FPS: {fps:.0f}  |  Sim: {sim_ms:.1f}ms  [{status}]",
                COLOR_TEXT,
            ),
            (
                font_hud,
                "[Up/Down] Wind   [R] Reset   [SPACE] Pause   [W] Wireframe   [ESC] Quit",
                COLOR_TEXT_DIM,
            ),
        ]

        y_pos = 12
        for fnt, text, color in hud_lines:
            surf = fnt.render(text, True, color)
            screen.blit(surf, (15, y_pos))
            y_pos += 28

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
