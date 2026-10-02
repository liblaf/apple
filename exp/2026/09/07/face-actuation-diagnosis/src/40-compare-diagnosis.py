"""Audit the per-tetrahedron regularizer and collect immutable diagnosis endpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import activation_models as am
import numpy as np
import pyvista as pv
import torch
from face_physics import active_graph

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT.parent / "face-activation-materials" / "data" / "10-fixture"
AREF = -math.log(0.8)
SMOOTH_LENGTH = 0.005


def digest(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cells(grid: pv.UnstructuredGrid) -> np.ndarray:
    return np.asarray(grid.cells).reshape(-1, 5)[:, 1:]


def surface_weights(
    reference: pv.UnstructuredGrid, skin: pv.PolyData, top: np.ndarray
) -> np.ndarray:
    global_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=int)
    triangles = global_ids[np.asarray(skin.faces).reshape(-1, 4)[:, 1:]]
    points = np.asarray(reference.points)
    xyz = points[triangles]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    weights = np.zeros(reference.n_points)
    np.add.at(weights, triangles.ravel(), np.repeat(area / 3, 3))
    weights = weights[top]
    assert weights.sum() > 0
    return weights / weights.sum()


def smooth_energy(
    field: torch.Tensor,
    ei: torch.Tensor,
    ej: torch.Tensor,
    ew: torch.Tensor,
    volume: float,
) -> torch.Tensor:
    return (
        SMOOTH_LENGTH**2
        * (ew * (field[ei] - field[ej]).square().sum(1)).sum()
        / volume
        / AREF**2
    )


def regularizer_audit(reference: pv.UnstructuredGrid) -> dict[str, Any]:
    tet = cells(reference)
    ids = np.flatnonzero(np.asarray(reference.cell_data["ActivationMask"], dtype=bool))
    region = np.asarray(reference.cell_data["ActivationControlId"], dtype=int)[ids]
    fraction = np.asarray(reference.cell_data["MuscleFraction"], dtype=float)
    points = np.asarray(reference.points, dtype=float)
    dm = np.transpose(points[tet[:, 1:]] - points[tet[:, :1]], (0, 2, 1))
    volume_all = np.linalg.det(dm) / 6
    volumes = volume_all[ids] * fraction[ids]
    ei_np, ej_np, ew_np = active_graph(points, tet, ids, region, fraction)
    ei, ej, ew = (torch.as_tensor(item) for item in (ei_np, ej_np, ew_np))
    volume = float(volumes.sum())

    # Every graph edge stays inside one MuscleId/control region, so this field
    # is exactly constant on its connected same-muscle support.
    basis = torch.tensor([0.13, -0.21, 0.34, 0.55, -0.89, 1.44])
    region_field = torch.as_tensor(region, dtype=torch.float64)[:, None] * basis
    constant_penalty = smooth_energy(region_field, ei, ej, ew, volume)

    local = torch.arange(len(ids), dtype=torch.float64)
    q = torch.stack(
        (
            torch.sin(local * 0.013),
            torch.cos(local * 0.017),
            local.remainder(11) / 11,
            torch.sin(local * 0.019),
            torch.cos(local * 0.023),
            local.remainder(7) / 7,
        ),
        dim=1,
    ).requires_grad_()
    axial_fibers = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).expand(
        len(ids), -1
    )
    _raw, h = am.matrices(q, "Raw6", axial_fibers)
    field = h.reshape(len(ids), 9) / math.sqrt(1.5)
    generic = smooth_energy(field, ei, ej, ew, volume)
    generic.backward(retain_graph=True)
    autograd = q.grad.detach().clone()
    grad_field = torch.zeros_like(field)
    coefficient = SMOOTH_LENGTH**2 / volume / AREF**2
    difference = field[ei] - field[ej]
    grad_field.index_add_(0, ei, 2 * coefficient * ew[:, None] * difference)
    grad_field.index_add_(0, ej, -2 * coefficient * ew[:, None] * difference)
    # Transform the flat field gradient through the symmetric six-coordinate map.
    manual_q = torch.autograd.grad(
        field, q, grad_outputs=grad_field, retain_graph=False
    )[0].detach()
    off_diagonal_factor = float(
        (field.square().sum(1) / h.square().sum((1, 2))).mean().detach()
    )

    fiber = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).expand(len(ids), -1)
    scalar = torch.full((len(ids), 1), 0.37, dtype=torch.float64)
    _a, fiber_h = am.matrices(scalar, "F", fiber)
    fiber_field = fiber_h.reshape(len(ids), 9) / math.sqrt(1.5)
    scalar_magnitude = float(
        (fiber_field.square().sum(1) * torch.as_tensor(volumes)).sum() / volume
    )
    tracefree = torch.tensor(
        [0.37, -0.185, -0.185, 0.0, 0.0, 0.0], dtype=torch.float64
    ).expand(len(ids), -1)
    _raw, tensor_h = am.matrices(tracefree, "Raw6", fiber)
    tensor_field = tensor_h.reshape(len(ids), 9) / math.sqrt(1.5)
    tensor_magnitude = float(
        (tensor_field.square().sum(1) * torch.as_tensor(volumes)).sum() / volume
    )
    return {
        "graph": {
            "active_cells": len(ids),
            "edges": len(ei_np),
            "same_region_edges": bool(np.all(region[ei_np] == region[ej_np])),
        },
        "constant_per_MuscleId_tensor": {
            "smooth_penalty": float(constant_penalty),
            "max_edge_difference": float(
                (region_field[ei] - region_field[ej]).abs().max()
            ),
        },
        "generic_tensor_gradient": {
            "smooth_penalty": float(generic.detach()),
            "max_abs_error": float((autograd - manual_q).abs().max()),
            "relative_l2_error": float(
                torch.linalg.vector_norm(autograd - manual_q)
                / torch.linalg.vector_norm(autograd)
            ),
            "field_over_frobenius_squared_factor": off_diagonal_factor,
        },
        "constant_fiber_vs_tracefree_tensor": {
            "scalar_field_squared": scalar_magnitude,
            "tensor_field_squared": tensor_magnitude,
            "absolute_difference": abs(scalar_magnitude - tensor_magnitude),
        },
        "interpretation": "Tensor fields are H flattened over nine entries and divided by sqrt(1.5), so off-diagonal symmetric entries receive their Frobenius multiplicity.",
    }


def common_metrics(
    reference: pv.UnstructuredGrid,
    top: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    dm_inv: np.ndarray,
    tet: np.ndarray,
    ids: np.ndarray,
    ei: np.ndarray,
    ej: np.ndarray,
    ew: np.ndarray,
    variation_volume: float,
    record: dict[str, Any],
) -> dict[str, Any]:
    endpoint = pv.read(record["path"])
    assert isinstance(endpoint, pv.UnstructuredGrid)
    assert endpoint.n_points == reference.n_points
    assert endpoint.n_cells == reference.n_cells
    rest = np.asarray(
        endpoint.point_data.get("RestPosition", reference.points), dtype=float
    )
    state = np.asarray(endpoint.points, dtype=float)
    u = state - rest
    D = float(np.sqrt(np.sum(weights[:, None] * target[top] ** 2)))
    fit = float(np.sqrt(np.sum(weights[:, None] * (u[top] - target[top]) ** 2)) / D)
    motion = float(np.sqrt(np.sum(weights[:, None] * u[top] ** 2)))
    projection = float(np.sum(weights[:, None] * u[top] * target[top]) / D**2)
    ds = np.transpose(state[tet[:, 1:]] - state[tet[:, :1]], (0, 2, 1))
    detf = np.linalg.det(ds @ dm_inv)
    variation: float | None = None
    if "ActivationInverseMatrix" in endpoint.cell_data:
        matrices = np.asarray(
            endpoint.cell_data["ActivationInverseMatrix"], dtype=float
        ).reshape(-1, 3, 3)
        field = torch.as_tensor(
            (matrices[ids] - np.eye(3)).reshape(len(ids), 9) / math.sqrt(1.5)
        )
        variation = float(smooth_energy(field, ei, ej, ew, variation_volume))
    return {
        **record,
        "fit_rms_over_D": fit,
        "fit_rms_mm": 1000 * D * fit,
        "motion_rms_mm": 1000 * motion,
        "target_projection_amplitude": projection,
        "detF_min": float(detf.min()),
        "inverted_tets": int((detf <= 0).sum()),
        "same_muscle_activation_variation_common_current_mask": variation,
        "variation_support": "current 120020-cell mask and same-MuscleId graph",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "40-diagnosis-comparison"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    reference = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    assert isinstance(reference, pv.UnstructuredGrid)
    assert isinstance(skin, pv.PolyData)
    target = np.asarray(reference.point_data["Smile"], dtype=float)
    top = np.flatnonzero(
        np.asarray(reference.point_data["IsFace"], dtype=bool)
        & np.isfinite(target).all(1)
    )
    weights = surface_weights(reference, skin, top)
    tet = cells(reference)
    points = np.asarray(reference.points, dtype=float)
    dm = np.transpose(points[tet[:, 1:]] - points[tet[:, :1]], (0, 2, 1))
    dm_inv = np.linalg.inv(dm)
    ids = np.flatnonzero(np.asarray(reference.cell_data["ActivationMask"], dtype=bool))
    region = np.asarray(reference.cell_data["ActivationControlId"], dtype=int)[ids]
    fraction = np.asarray(reference.cell_data["MuscleFraction"], dtype=float)
    volume = float((np.linalg.det(dm) / 6)[ids].dot(fraction[ids]))
    ei_np, ej_np, ew_np = active_graph(points, tet, ids, region, fraction)
    ei, ej, ew = (torch.as_tensor(x) for x in (ei_np, ej_np, ew_np))
    audit = regularizer_audit(reference)
    write_json(output / "regularizer-audit.json", audit)
    records = [
        {
            "id": "historical-no-skin",
            "status": "historical saved endpoint; no new solve",
            "path": str(ROOT / "data/11-historical-no-skin/final.vtu"),
        }
    ]
    pending_inverse_endpoints = []
    for identity, directory in (
        ("raw6-no-skin", ROOT / "data/20-raw6-no-skin"),
        ("raw6-smooth-no-skin", ROOT / "data/21-raw6-smooth-no-skin"),
    ):
        final = directory / "final.vtu"
        summary = directory / "summary.json"
        if final.is_file() and summary.is_file():
            status = json.loads(summary.read_text()).get("status", "unknown")
            records.append(
                {
                    "id": identity,
                    "status": f"inverse immutable saved endpoint; {status}",
                    "path": str(final),
                }
            )
        else:
            pending_inverse_endpoints.append(
                {
                    "id": identity,
                    "reason": "final.vtu and summary.json are both required",
                }
            )
    for path in sorted(
        (ROOT / "data/10-manual-activation").glob("skin-*/*/c*/state.vtu")
    ):
        bits = path.relative_to(ROOT / "data/10-manual-activation").parts
        records.append(
            {
                "id": "manual-" + "-".join(bits[:-1]),
                "status": "manual strict equilibrium",
                "path": str(path),
            }
        )
    results = [
        common_metrics(
            reference, top, weights, target, dm_inv, tet, ids, ei, ej, ew, volume, item
        )
        for item in records
    ]
    with (output / "endpoint-metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_json(
        output / "receipt.json",
        {
            "inputs": {
                "fixture": digest(FIXTURE / "volume.vtu"),
                "skin": digest(FIXTURE / "skin.vtp"),
            },
            "records": [
                {
                    "id": r["id"],
                    "source": digest(Path(r["path"])),
                    "status": r["status"],
                }
                for r in records
            ],
            "pending_inverse_endpoints": pending_inverse_endpoints,
            "common_support": "finite IsFace Smile target vertices with rest-skin area weights; activation variation restricted to current mask",
        },
    )


if __name__ == "__main__":
    main()
