"""Fixed L2, target-normal and physical-volume tensor objective (CPU or GPU)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PRIOR_LENGTH_M = 0.005
PRIOR_L_REF_M = 13.236093032531715e-3
PRIOR_SMOOTH_WEIGHT = 7.2e-7


def physical_volume_graph(
    reference_points: np.ndarray,
    active_tets: np.ndarray,
    muscle_ids: np.ndarray,
    physical_volumes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face neighbors in one muscle; conductance A/d times harmonic fraction.

    Arrays are in q order. Physical volumes include the muscle material fraction;
    dividing them by geometric reference volumes recovers that fraction. This is
    the graph used by the earlier active-strain chain, without cross-muscle edges.
    """
    points = np.asarray(reference_points, dtype=np.float64)
    tets = np.asarray(active_tets, dtype=np.int64)
    regions = np.asarray(muscle_ids)
    volumes = np.asarray(physical_volumes, dtype=np.float64)
    assert tets.ndim == 2
    assert tets.shape[1] == 4
    assert len(tets) > 0
    assert regions.shape == volumes.shape == (len(tets),)
    assert np.isfinite(points).all()
    assert np.isfinite(volumes).all()
    assert np.all(volumes > 0)
    xyz = points[tets]
    dm = np.transpose(xyz[:, 1:] - xyz[:, :1], (0, 2, 1))
    geometric_volumes = np.linalg.det(dm) / 6
    assert np.all(geometric_volumes > 0)
    fraction = volumes / geometric_volumes
    assert np.all(fraction <= 1 + 1e-10)
    pattern = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    faces = np.sort(tets[:, pattern].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(tets)), 4)
    order = np.lexsort(faces.T[::-1])
    faces, owner = faces[order], owner[order]
    pair = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
    assert not np.any(np.diff(pair) == 1), "Nonmanifold active tetrahedron face"
    i, j, face = owner[pair], owner[pair + 1], faces[pair]
    same = regions[i] == regions[j]
    i, j, face = i[same], j[same], face[same]
    face_points = points[face]
    area = (
        np.linalg.norm(
            np.cross(
                face_points[:, 1] - face_points[:, 0],
                face_points[:, 2] - face_points[:, 0],
            ),
            axis=1,
        )
        / 2
    )
    centers = xyz.mean(axis=1)
    distance = np.linalg.norm(centers[i] - centers[j], axis=1)
    assert np.all(distance > 0)
    assert np.all(area > 0)
    conductance = (
        area / distance * (2 * fraction[i] * fraction[j] / (fraction[i] + fraction[j]))
    )
    return i, j, conductance


def _normals(
    points: torch.Tensor, triangles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = points[triangles]
    cross = torch.linalg.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    double_area = torch.linalg.vector_norm(cross, dim=-1)
    assert bool(torch.isfinite(double_area).all())
    assert bool((double_area > 1e-14).all()), "Collapsed surface triangle"
    return cross / double_area[:, None], double_area / 2


@dataclass(frozen=True)
class MouthOpenFitObjective:
    """All weights frozen at construction; regularizer returns its weighted term."""

    reference_skin_points: torch.Tensor
    skin_ids: torch.Tensor
    triangles: torch.Tensor
    target_displacement: torch.Tensor
    target_normals: torch.Tensor
    triangle_weights: torch.Tensor
    vertex_weights: torch.Tensor
    scale2: float
    edge_i: torch.Tensor
    edge_j: torch.Tensor
    conductance: torch.Tensor
    regularizer_factor: float
    normal_weight: float
    smooth_weight: float
    active_cell_count: int
    protocol: dict

    def l2(self, u: torch.Tensor) -> torch.Tensor:
        error = u[self.skin_ids] - self.target_displacement
        return (self.vertex_weights[:, None] * error.square()).sum() / self.scale2

    def normal(self, u: torch.Tensor) -> torch.Tensor:
        normals, _ = _normals(
            self.reference_skin_points + u[self.skin_ids], self.triangles
        )
        return (
            self.triangle_weights * (normals - self.target_normals).square().sum(-1)
        ).sum()

    def smoothness(self, q: torch.Tensor) -> torch.Tensor:
        """Dimensionless R for B=I+sym(q), Raw6 order xx,yy,zz,xy,yz,xz."""
        assert q.shape == (self.active_cell_count, 6)
        delta = q[self.edge_i] - q[self.edge_j]
        frobenius2 = delta[:, :3].square().sum(-1) + 2 * delta[:, 3:].square().sum(-1)
        return self.regularizer_factor * (self.conductance * frobenius2).sum()

    def surface(self, u: torch.Tensor) -> torch.Tensor:
        return self.l2(u) + self.normal_weight * self.normal(u)

    def regularizer(self, q: torch.Tensor) -> torch.Tensor:
        """Weighted direct control objective; include its gradient exactly once."""
        return self.smooth_weight * self.smoothness(q)

    def components(self, u: torch.Tensor, q: torch.Tensor) -> dict[str, torch.Tensor]:
        l2, normal, smooth = self.l2(u), self.normal(u), self.smoothness(q)
        normal_term, smooth_term = (
            self.normal_weight * normal,
            self.smooth_weight * smooth,
        )
        return {
            "position_loss": l2,
            "normal_loss": normal,
            "activation_smoothness": smooth,
            "normal_contribution": normal_term,
            "regularizer_contribution": smooth_term,
            "surface": l2 + normal_term,
            "loss": l2 + normal_term + smooth_term,
            "fit_rms_mm": torch.sqrt(l2 * self.scale2) * 1000,
        }


def build_mouthopen_fit_objective(
    *,
    reference_points: np.ndarray,
    skin_ids: np.ndarray,
    triangles: np.ndarray,
    target_points: np.ndarray,
    area_reference_points: np.ndarray,
    vertex_weights: np.ndarray,
    scale2: float,
    active_tets: np.ndarray,
    muscle_ids: np.ndarray,
    physical_volumes: np.ndarray,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> MouthOpenFitObjective:
    """Build the prior46 objective in the current unchanged L2 units.

    Skin triangles use local indices. Target and area-reference points are skin
    arrays; reference_points is the full physical mesh. Supply the existing L2
    vertex weights and scale2 unchanged. Active cells and all graph arrays must
    follow q order. No material parameters or optimizer state are changed.
    """
    assert math.isfinite(scale2)
    assert scale2 > 0
    assert dtype == torch.float64

    def real(array: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.asarray(array).copy(), dtype=dtype, device=device)

    def index(array: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.asarray(array).copy(), dtype=torch.int64, device=device)

    skin = np.asarray(skin_ids, dtype=np.int64)
    ref = real(np.asarray(reference_points)[skin])
    target = real(target_points)
    areas_ref = real(area_reference_points)
    faces = index(triangles)
    weights = real(vertex_weights)
    assert ref.shape == target.shape == areas_ref.shape == (len(skin), 3)
    assert weights.shape == (len(skin),)
    assert bool(torch.isfinite(ref).all())
    assert bool(torch.isfinite(target).all())
    assert bool(torch.isfinite(weights).all())
    assert bool((weights >= 0).all())
    assert math.isclose(float(weights.sum()), 1.0, rel_tol=0, abs_tol=1e-12)
    _, areas = _normals(areas_ref, faces)
    normals, _ = _normals(target, faces)
    i, j, conductance = physical_volume_graph(
        reference_points, active_tets, muscle_ids, physical_volumes
    )
    volume = float(np.sum(physical_volumes))
    factor = PRIOR_LENGTH_M**2 / volume
    conversion = 3 * PRIOR_L_REF_M**2 / scale2
    normal_weight = (0.002**2 / scale2) / (2 * (1 - math.cos(math.radians(5))))
    smooth_weight = PRIOR_SMOOTH_WEIGHT * conversion
    graph = coo_matrix(
        (np.ones(len(i)), (i, j)), shape=(len(active_tets), len(active_tets))
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    graph_hashes = {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in {
            "edge_i": i,
            "edge_j": j,
            "conductance": conductance,
            "muscle_ids": np.asarray(muscle_ids),
            "physical_volumes": np.asarray(physical_volumes, dtype=np.float64),
        }.items()
    }
    protocol = {
        "graph_array_sha256": graph_hashes,
        "graph_hash_encoding": "C-contiguous raw NumPy bytes; edge indices int64 and conductance/volumes float64",
        "graph_connected_components": int(component_count),
        "graph_singleton_cells": int(np.count_nonzero(component_sizes == 1)),
        "objective": "L2 + normal_weight * mean_target_normal_chord_squared + smooth_weight * R",
        "l2": "existing vertex-area weighted squared vector error / existing scale2; no factor 1/3",
        "scale2_m2": scale2,
        "normal": "oriented target unit face normals; fixed area-reference triangle weights normalized to sum one",
        "normal_weight": normal_weight,
        "normal_weight_rule": "2 mm vector RMS contribution equals uniform 5 degree normal contribution",
        "smooth_weight": smooth_weight,
        "smooth_weight_prior": PRIOR_SMOOTH_WEIGHT,
        "prior_l_ref_m": PRIOR_L_REF_M,
        "prior_objective_conversion": conversion,
        "prior_source": "exp/2026/09/21/stress-activation-loss/src/46-run-active-strain-chain.py",
        "smoothness": "R = length^2 / total physical active volume * sum_same_muscle_face_edges(A/d * harmonic physical fraction * Frobenius(Bi-Bj)^2)",
        "physical_fraction": "physical active volume / positive geometric reference tet volume",
        "tensor": "B=I+sym(q); Raw6 xx,yy,zz,xy,yz,xz; squared offdiagonal differences weighted twice",
        "smooth_length_m": PRIOR_LENGTH_M,
        "total_physical_active_volume_m3": volume,
        "active_cell_count": len(active_tets),
        "same_muscle_edge_count": len(i),
        "regularizer_factor": factor,
        "regularizer_method_returns": "weighted direct control term, not raw R",
        "fit_rms": "sqrt(L2 * scale2) * 1000; excludes added objective terms",
    }
    return MouthOpenFitObjective(
        ref,
        index(skin),
        faces,
        target - ref,
        normals,
        areas / areas.sum(),
        weights,
        float(scale2),
        index(i),
        index(j),
        real(conductance),
        factor,
        normal_weight,
        smooth_weight,
        len(active_tets),
        protocol,
    )
