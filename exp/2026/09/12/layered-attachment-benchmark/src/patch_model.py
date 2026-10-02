"""Synthetic, force-controlled layered tissue patch in millimetres and newtons."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pyvista as pv
import torch
import warp as wp

from liblaf.apple import common
from liblaf.apple.forward import Forward, ModelBuilder
from liblaf.apple.warp.fem import Koiter, StableNeoHookean
from liblaf.apple.warp.potential import ExternalForce, FiberSpring

LENGTH = 20.0
WIDTH = 10.0
Z_BOUNDARIES = np.array([0.0, 1.5, 2.0, 2.5])
DEEP_E = 0.003
MOBILE_E = 0.0003
SKIN_E = 0.05
NU = 0.4
SKIN_NU = 0.46
SKIN_THICKNESS = 0.5
TOTAL_LINK_STIFFNESS = 0.3


@dataclass(frozen=True)
class Case:
    name: str
    layer_young: float = MOBILE_E
    attachment_center: float | None = None
    attachment_stiffness_scale: float = 1.0
    thickness_field: bool = False
    stiffness_scale: float = 1.0
    thickness_scale: float = 1.0
    skin_young_scale: float = 1.0


@dataclass(frozen=True)
class Load:
    name: str
    axis: int
    center_x: float
    force: float = 0.002


def configure() -> None:
    assert torch.cuda.is_available()
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")
    torch.set_num_threads(4)
    wp.init()
    logging.getLogger("liblaf.apple.forward._forward").setLevel(logging.WARNING)


def geometry(
    refinement: int,
) -> tuple[pv.UnstructuredGrid, pv.PolyData, dict[str, Any]]:
    nx, ny = 8 * refinement, 4 * refinement
    xs, ys = np.linspace(0, LENGTH, nx + 1), np.linspace(0, WIDTH, ny + 1)
    zs = np.concatenate(
        [
            np.linspace(a, b, refinement, endpoint=False)
            for a, b in pairwise(Z_BOUNDARIES)
        ]
        + [Z_BOUNDARIES[-1:]]
    )
    points = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    ids = np.arange(len(points)).reshape(len(zs), ny + 1, nx + 1)
    pattern = np.array(
        [
            [0, 1, 3, 7],
            [0, 3, 2, 7],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [0, 4, 5, 7],
            [0, 5, 1, 7],
        ]
    )
    cells = []
    for k in range(len(zs) - 1):
        for j in range(ny):
            for i in range(nx):
                corners = ids[k : k + 2, j : j + 2, i : i + 2].ravel()
                cells.extend(corners[pattern])
    cells = np.asarray(cells)
    xyz = points[cells]
    determinant = np.linalg.det(np.moveaxis(xyz[:, 1:] - xyz[:, :1], 1, 2))
    assert np.all(determinant > 0)
    mesh = pv.UnstructuredGrid(
        np.column_stack([np.full(len(cells), 4), cells]).ravel(),
        np.full(len(cells), pv.CellType.TETRA, np.uint8),
        points,
    )
    top_ids = ids[-1].ravel()
    tri = []
    top_local = np.arange(len(top_ids)).reshape(ny + 1, nx + 1)
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = top_local[j : j + 2, i : i + 2].ravel()
            tri.extend([[a, b, d], [a, d, c]])
    tri = np.asarray(tri)
    skin = pv.PolyData(
        points[top_ids], np.column_stack([np.full(len(tri), 3), tri]).ravel()
    )
    skin.point_data[common.GLOBAL_POINT_ID.vtk] = top_ids
    triangle_xyz = skin.points[tri]
    areas = (
        np.linalg.norm(
            np.cross(
                triangle_xyz[:, 1] - triangle_xyz[:, 0],
                triangle_xyz[:, 2] - triangle_xyz[:, 0],
            ),
            axis=1,
        )
        / 2
    )
    node_areas = np.zeros(len(top_ids))
    np.add.at(node_areas, tri.ravel(), np.repeat(areas / 3, 3))
    return (
        mesh,
        skin,
        {
            "ids": ids,
            "zs": zs,
            "areas": areas,
            "node_areas": node_areas,
            "tets": cells,
            "top_ids": top_ids,
            "nx": nx,
            "ny": ny,
        },
    )


def attachment_arrays(mesh: pv.UnstructuredGrid, info: dict[str, Any], center: float):
    """Opposing oblique families spanning the compliant layer; no measured law."""
    ids, zs = info["ids"], info["zs"]
    lower = int(np.flatnonzero(np.isclose(zs, 1.5))[0])
    upper = int(np.flatnonzero(np.isclose(zs, 2.0))[0])
    span = info["nx"] // 8  # fixed 2.5 mm lateral offset across refinements
    pairs, weights = [], []
    node_areas = info["node_areas"].reshape(info["ny"] + 1, info["nx"] + 1)
    for j in range(info["ny"] + 1):
        for i in range(span, info["nx"] + 1 - span):
            x = mesh.points[ids[lower, j, i], 0]
            density = np.exp(-0.5 * ((x - center) / 1.5) ** 2)
            for sign in [-1, 1]:
                pairs.append([ids[lower, j, i], ids[upper, j, i + sign * span]])
                weights.append(density * node_areas[j, i] / 2)
    pairs = np.asarray(pairs, dtype=np.int32)
    weights = np.asarray(weights)
    stiffness = TOTAL_LINK_STIFFNESS * weights / weights.sum()
    vectors = mesh.points[pairs[:, 1]] - mesh.points[pairs[:, 0]]
    return pairs, vectors, stiffness


def assemble(case: Case, load: Load, refinement: int = 1):
    mesh, skin, info = geometry(refinement)
    builder = ModelBuilder()
    builder.add_vertices(mesh)
    fixed = np.zeros((mesh.n_points, 3), bool)
    fixed[np.isclose(mesh.points[:, 2], 0)] = True
    mesh.point_data[common.FIXED_MASK.vtk] = fixed
    mesh.point_data[common.FIXED_VALUE.vtk] = np.zeros((mesh.n_points, 3))
    builder.add_fixed(mesh)
    z = mesh.points[info["tets"]].mean(axis=1)[:, 2]
    mobile = (z > 1.5) & (z < 2.0)
    young = np.where(mobile, case.layer_young, DEEP_E) * case.stiffness_scale
    mu = young / (2 * (1 + NU))
    mesh.cell_data[common.MU.vtk] = mu
    mesh.cell_data[common.LAMBDA.vtk] = young * NU / ((1 + NU) * (1 - 2 * NU)) + mu
    mesh.cell_data["LayerId"] = np.where(z < 1.5, 0, np.where(z < 2, 1, 2))
    mesh.cell_data["YoungMPa"] = young
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="volume"))
    skin_e = SKIN_E * case.stiffness_scale * case.skin_young_scale
    skin.cell_data[common.LAMBDA.vtk] = np.full(
        skin.n_cells, skin_e * SKIN_NU / (1 - SKIN_NU**2)
    )
    skin.cell_data[common.MU.vtk] = np.full(skin.n_cells, skin_e / (2 * (1 + SKIN_NU)))
    thickness = np.full(skin.n_cells, SKIN_THICKNESS)
    if case.thickness_field:
        x = skin.cell_centers().points[:, 0]
        thickness *= 1 + 0.3 * np.cos(2 * np.pi * x / LENGTH)
        thickness *= SKIN_THICKNESS / np.average(thickness, weights=info["areas"])
    thickness *= case.thickness_scale
    skin.cell_data["Thickness"] = thickness
    builder.add_potential(
        Koiter.from_pyvista(
            skin,
            name="skin",
            thickness=thickness if case.thickness_field else float(thickness[0]),
        )
    )
    pairs, rest_vectors, link_stiffness = (
        np.empty((0, 2), np.int32),
        np.empty((0, 3)),
        np.empty(0),
    )
    if case.attachment_center is not None:
        pairs, rest_vectors, link_stiffness = attachment_arrays(
            mesh, info, case.attachment_center
        )
        link_stiffness *= case.stiffness_scale * case.attachment_stiffness_scale
        builder.add_potential(
            FiberSpring.from_arrays(
                pairs,
                rest_vectors,
                link_stiffness,
                tension_only=True,
                name="attachments",
            )
        )
    weights = (
        np.exp(
            -0.5
            * (
                ((skin.points[:, 0] - load.center_x) / 2.0) ** 2
                + ((skin.points[:, 1] - WIDTH / 2) / 2.0) ** 2
            )
        )
        * info["node_areas"]
    )
    weights /= weights.sum()
    force = np.zeros((skin.n_points, 3))
    force[:, load.axis] = load.force * weights
    skin.point_data[common.FORCE.vtk] = force
    builder.add_potential(ExternalForce.from_pyvista(skin, name="load"))
    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(max_steps=5000, rtol=2e-6, atol=1e-10)
    info.update(
        fixed=fixed,
        load_weights=weights,
        force=force,
        pairs=pairs,
        rest_vectors=rest_vectors,
        link_stiffness=link_stiffness,
    )
    return forward, mesh, skin, info


def solve(case: Case, load: Load, refinement: int = 1):
    start = time.monotonic()
    forward, mesh, skin, info = assemble(case, load, refinement)
    initial_gradient = float(
        torch.linalg.vector_norm(forward.problem.grad(forward.state))
    )
    solution = forward.step()
    residual = float(torch.linalg.vector_norm(forward.problem.grad(forward.state)))
    assert solution.success, (case.name, load.name, solution.result, residual)
    assert residual <= max(2e-10, initial_gradient * 3e-6), residual
    u = forward.state.u.detach().cpu().numpy().copy()
    tets = info["tets"]
    reference = mesh.points[tets]
    deformed = (mesh.points + u)[tets]
    det_ratio = np.linalg.det(deformed[:, 1:] - deformed[:, :1]) / np.linalg.det(
        reference[:, 1:] - reference[:, :1]
    )
    assert det_ratio.min() > 0.0
    top_u = u[info["top_ids"]]
    zs = info["zs"]
    low = int(np.flatnonzero(np.isclose(zs, 1.5))[0])
    high = int(np.flatnonzero(np.isclose(zs, 2.0))[0])
    relative_u = u[info["ids"][high].ravel()] - u[info["ids"][low].ravel()]
    full_gradient = forward.model.grad(forward.state).detach().cpu().numpy()
    reactions = full_gradient[np.isclose(mesh.points[:, 2], 0)].sum(axis=0)
    applied = info["force"].sum(axis=0)
    balance = np.linalg.norm(reactions + applied) / load.force
    assert balance < 5e-5
    metrics = {
        "case": case.name,
        "load": load.name,
        "refinement": refinement,
        "points": mesh.n_points,
        "tets": mesh.n_cells,
        "links": len(info["pairs"]),
        "layer_young_mpa": case.layer_young,
        "force_n": load.force,
        "load_displacement_mm": float(
            np.dot(info["load_weights"], top_u[:, load.axis])
        ),
        "surface_rms_mm": float(
            np.sqrt(np.average(np.sum(top_u**2, axis=1), weights=info["node_areas"]))
        ),
        "relative_layer_motion_rms_mm": float(
            np.sqrt(
                np.average(np.sum(relative_u**2, axis=1), weights=info["node_areas"])
            )
        ),
        "minimum_det_f": float(det_ratio.min()),
        "residual_force_n": residual,
        "relative_force_balance_error": float(balance),
        "solver_status": str(solution.result),
        "iterations": int(solution.state.step),
        "energy_n_mm": float(forward.model.fun(forward.state)),
        "elapsed_seconds": time.monotonic() - start,
        "mean_skin_thickness_mm": float(
            np.average(skin.cell_data["Thickness"], weights=info["areas"])
        ),
    }
    mesh.point_data["DisplacementMm"] = u
    skin.point_data["DisplacementMm"] = top_u
    logging.getLogger(__name__).info(
        "%s / %s: displacement %.6g mm, residual %.3g N, steps %s",
        case.name,
        load.name,
        metrics["load_displacement_mm"],
        residual,
        metrics["iterations"],
    )
    return metrics, mesh, skin, info, forward
