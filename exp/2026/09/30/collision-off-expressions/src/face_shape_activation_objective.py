# Copyright (c) 2026 liblaf
# ruff: noqa: PLR0915, PT018, SIM300
"""Dimensionless skin shape and within-muscle activation losses for one face mesh.

The inverse activation coordinates are raw symmetric six-vectors in the order
``xx, yy, zz, xy, yz, xz``. They are not Mandel-scaled. This module has no
forward solver, optimizer, collision, or checkpoint side effects.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ObjectiveWeights:
    """Dimensionless coefficients, selected and frozen by a separate calibration."""

    normal: float
    smooth: float

    def __post_init__(self) -> None:
        assert math.isfinite(self.normal) and self.normal >= 0
        assert math.isfinite(self.smooth) and self.smooth >= 0


class SkinShapeLoss:
    """Area-weighted corresponding positions and oriented triangle normals.

    The skin correspondence and triangle winding are fixed across neutral,
    target, and predicted surfaces. Degenerate triangles fail visibly.
    """

    def __init__(
        self,
        constitutive_skin_points_m: np.ndarray,
        neutral_skin_points_m: np.ndarray,
        target_skin_points_m: np.ndarray,
        skin_global_ids: np.ndarray,
        skin_triangles: np.ndarray,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
        minimum_double_area_m2: float = 1e-14,
        minimum_area_ratio: float = 1e-4,
    ) -> None:
        assert math.isfinite(minimum_double_area_m2) and minimum_double_area_m2 > 0
        assert math.isfinite(minimum_area_ratio) and 0 < minimum_area_ratio < 1
        self.minimum_double_area_m2 = minimum_double_area_m2
        self.minimum_area_ratio = minimum_area_ratio
        self.input_sha256 = {
            "constitutive_skin_points_m": _array_sha256(
                np.asarray(constitutive_skin_points_m)
            ),
            "neutral_skin_points_m": _array_sha256(np.asarray(neutral_skin_points_m)),
            "target_skin_points_m": _array_sha256(np.asarray(target_skin_points_m)),
            "skin_global_ids": _array_sha256(np.asarray(skin_global_ids)),
            "skin_triangles": _array_sha256(np.asarray(skin_triangles)),
        }
        self.constitutive = torch.as_tensor(
            constitutive_skin_points_m, device=device, dtype=dtype
        )
        self.neutral = torch.as_tensor(
            neutral_skin_points_m, device=device, dtype=dtype
        )
        self.target = torch.as_tensor(target_skin_points_m, device=device, dtype=dtype)
        self.skin_ids = torch.as_tensor(
            skin_global_ids, device=device, dtype=torch.long
        )
        self.triangles = torch.as_tensor(
            skin_triangles, device=device, dtype=torch.long
        )
        assert self.constitutive.ndim == 2 and self.constitutive.shape[1] == 3
        assert self.neutral.shape == self.target.shape == self.constitutive.shape
        assert self.skin_ids.shape == (len(self.neutral),)
        assert torch.unique(self.skin_ids).numel() == self.skin_ids.numel()
        assert bool(torch.all(self.skin_ids >= 0))
        assert self.triangles.ndim == 2 and self.triangles.shape[1] == 3
        assert self.triangles.numel() > 0
        assert bool(
            torch.all((0 <= self.triangles) & (self.triangles < len(self.neutral)))
        )
        for value in (self.constitutive, self.neutral, self.target):
            assert bool(torch.isfinite(value).all())
        self.neutral_normals, neutral_double_area = self._normals(self.neutral)
        self.target_normals, target_double_area = self._normals(self.target)
        self.neutral_double_area = neutral_double_area
        assert bool(
            torch.all(target_double_area / neutral_double_area > minimum_area_ratio)
        ), "target skin triangle approaches collapse"
        self.neutral_area = neutral_double_area / 2
        self.total_neutral_area_m2 = self.neutral_area.sum()
        assert bool(self.total_neutral_area_m2 > 0)
        vertex_area = torch.zeros(len(self.neutral), dtype=dtype, device=device)
        vertex_area.scatter_add_(
            0,
            self.triangles.reshape(-1),
            (self.neutral_area[:, None] / 3).expand(-1, 3).reshape(-1),
        )
        self.vertex_weight = vertex_area / vertex_area.sum()
        self.target_displacement = self.target - self.constitutive
        self.target_motion_scale2_m2 = (
            self.vertex_weight[:, None] * (self.target - self.neutral).square()
        ).sum()
        assert bool(torch.isfinite(self.target_motion_scale2_m2))
        assert bool(self.target_motion_scale2_m2 > 0)
        negative_alignment = (self.target_normals * self.neutral_normals).sum(dim=1) < 0
        self.negative_reference_target_normal_alignment_ids = (
            torch.nonzero(negative_alignment).flatten().cpu().tolist()
        )
        self.negative_reference_target_normal_alignment_area_weights = (
            (self.neutral_area[negative_alignment] / self.total_neutral_area_m2)
            .cpu()
            .tolist()
        )
        self.minimum_target_to_neutral_area_ratio = float(
            (target_double_area / neutral_double_area).min()
        )

    def _normals(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        triangle = points[self.triangles]
        cross = torch.linalg.cross(
            triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
        )
        double_area = torch.linalg.vector_norm(cross, dim=1)
        assert bool(torch.isfinite(double_area).all())
        assert bool(torch.all(double_area > self.minimum_double_area_m2)), (
            "collapsed skin triangle"
        )
        return cross / double_area[:, None], double_area

    def components(
        self, full_displacement_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized position loss and area-mean squared normal chord."""
        assert full_displacement_m.ndim == 2 and full_displacement_m.shape[1] == 3
        assert full_displacement_m.shape[0] > int(self.skin_ids.max())
        assert full_displacement_m.dtype == self.constitutive.dtype
        assert full_displacement_m.device == self.constitutive.device
        predicted_displacement = full_displacement_m[self.skin_ids]
        position = (
            self.vertex_weight[:, None]
            * (predicted_displacement - self.target_displacement).square()
        ).sum() / self.target_motion_scale2_m2
        predicted_normals, predicted_double_area = self._normals(
            self.constitutive + predicted_displacement
        )
        assert bool(
            torch.all(
                predicted_double_area / self.neutral_double_area
                > self.minimum_area_ratio
            )
        ), "predicted skin triangle approaches collapse"
        normal = (
            self.neutral_area
            * (predicted_normals - self.target_normals).square().sum(dim=1)
        ).sum() / self.total_neutral_area_m2
        assert bool(torch.isfinite(position) and torch.isfinite(normal))
        return position, normal

    @torch.no_grad()
    def metrics(self, full_displacement_m: torch.Tensor) -> dict[str, float | int]:
        position, normal = self.components(full_displacement_m)
        predicted = self.constitutive + full_displacement_m[self.skin_ids]
        normals, double_area = self._normals(predicted)
        dot = (normals * self.target_normals).sum(dim=1).clamp(-1, 1)
        angle = torch.arccos(dot)
        angle_rms = torch.sqrt(
            (self.neutral_area * angle.square()).sum() / self.total_neutral_area_m2
        )
        return {
            "position_normalized": float(position),
            "position_rms_mm": float(
                torch.sqrt(position * self.target_motion_scale2_m2) * 1000
            ),
            "normal_chord2_area_mean": float(normal),
            "normal_chord_rms": float(torch.sqrt(normal)),
            "normal_angle_rms_deg": float(torch.rad2deg(angle_rms)),
            "minimum_predicted_to_neutral_triangle_area_ratio": float(
                (double_area / (2 * self.neutral_area)).min()
            ),
            "minimum_target_to_neutral_triangle_area_ratio": self.minimum_target_to_neutral_area_ratio,
            "negative_reference_target_normal_alignment_count": len(
                self.negative_reference_target_normal_alignment_ids
            ),
            "negative_reference_predicted_normal_alignment_count": int(
                ((normals * self.neutral_normals).sum(dim=1) < 0).sum()
            ),
        }

    def contract(self) -> dict[str, float | int | str]:
        """Return stable loss units and target-geometry diagnostics for a protocol."""
        return {
            "position": "sum(neutral vertex area weight * squared corresponding position error) / target motion scale squared",
            "normal": "sum(neutral triangle area * squared oriented unit-normal chord) / total neutral triangle area",
            "skin_vertex_count": len(self.neutral),
            "skin_triangle_count": len(self.triangles),
            "neutral_skin_area_m2": float(self.total_neutral_area_m2),
            "target_motion_scale_m": math.sqrt(float(self.target_motion_scale2_m2)),
            "minimum_double_area_m2": self.minimum_double_area_m2,
            "minimum_target_and_predicted_area_ratio": self.minimum_area_ratio,
            "negative_reference_target_normal_alignment_ids": self.negative_reference_target_normal_alignment_ids,
            "negative_reference_target_normal_alignment_area_weights": self.negative_reference_target_normal_alignment_area_weights,
            "minimum_target_to_neutral_triangle_area_ratio": self.minimum_target_to_neutral_area_ratio,
            "input_sha256": self.input_sha256,
        }


class ActivationSmoothness:
    """Geometry/volume normalized raw6 tensor variation within each muscle.

    Shared-face graph edges use repaired-reference face area divided by cell
    centroid distance, times the harmonic muscle fraction. Only retained active
    cells with the same ``MuscleId`` are joined. The prior length is explicit.
    """

    def __init__(
        self,
        points_m: np.ndarray,
        tetrahedra: np.ndarray,
        retained_active_cell_ids: np.ndarray,
        muscle_fraction: np.ndarray,
        muscle_id: np.ndarray,
        *,
        smooth_length_m: float | None = None,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        assert smooth_length_m is None or (
            math.isfinite(smooth_length_m) and smooth_length_m > 0
        )
        points = np.asarray(points_m, dtype=np.float64)
        tets = np.asarray(tetrahedra, dtype=np.int64)
        ids = np.asarray(retained_active_cell_ids, dtype=np.int64)
        fraction = np.asarray(muscle_fraction, dtype=np.float64)
        label = np.asarray(muscle_id, dtype=np.int64)
        assert points.ndim == 2 and points.shape[1] == 3
        assert tets.ndim == 2 and tets.shape[1] == 4
        assert ids.ndim == 1 and ids.size > 0
        assert fraction.shape == label.shape == (len(tets),)
        assert np.isfinite(points).all() and np.isfinite(fraction).all()
        assert np.all((0 <= tets) & (tets < len(points)))
        assert np.all((0 <= ids) & (ids < len(tets)))
        assert len(np.unique(ids)) == len(ids)
        active = tets[ids]
        active_fraction = fraction[ids]
        active_label = label[ids]
        assert np.all((active_fraction > 0) & (active_fraction <= 1))
        assert np.all(active_label >= 0), "active muscle cell lacks MuscleId"
        self.input_sha256 = {
            "points_m": _array_sha256(points),
            "tetrahedra": _array_sha256(tets),
            "retained_active_cell_ids": _array_sha256(ids),
            "active_muscle_fraction": _array_sha256(active_fraction),
            "active_muscle_id": _array_sha256(active_label),
        }
        dm = np.transpose(points[active[:, 1:]] - points[active[:, :1]], (0, 2, 1))
        volume = np.linalg.det(dm) / 6
        assert np.isfinite(volume).all() and np.all(volume > 0), (
            "invalid reference tetrahedron"
        )
        effective_volume = volume * active_fraction
        total_effective_volume = float(effective_volume.sum())
        assert math.isfinite(total_effective_volume) and total_effective_volume > 0
        face_pattern = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        faces = np.sort(active[:, face_pattern].reshape(-1, 3), axis=1)
        owner = np.repeat(np.arange(len(ids), dtype=np.int64), 4)
        order = np.lexsort(faces.T[::-1])
        faces, owner = faces[order], owner[order]
        pair = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
        assert not np.any(np.diff(pair) == 1), "nonmanifold active tetrahedral face"
        edge_i, edge_j, shared_face = owner[pair], owner[pair + 1], faces[pair]
        assert np.all(edge_i != edge_j)
        within_muscle = active_label[edge_i] == active_label[edge_j]
        cross_muscle_edge_count = int(np.count_nonzero(~within_muscle))
        edge_i, edge_j, shared_face = (
            edge_i[within_muscle],
            edge_j[within_muscle],
            shared_face[within_muscle],
        )
        assert len(edge_i) > 0, "no same-muscle retained active face adjacency"
        xyz = points[shared_face]
        face_area = (
            np.linalg.norm(
                np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
            )
            / 2
        )
        center = points[active].mean(axis=1)
        distance = np.linalg.norm(center[edge_i] - center[edge_j], axis=1)
        assert np.isfinite(face_area).all() and np.all(face_area > 0)
        assert np.isfinite(distance).all() and np.all(distance > 0)
        fi, fj = active_fraction[edge_i], active_fraction[edge_j]
        conductance = face_area / distance * (2 * fi * fj / (fi + fj))
        assert np.isfinite(conductance).all() and np.all(conductance > 0)
        graph_length_m = math.sqrt(total_effective_volume / float(conductance.sum()))
        provided_length = smooth_length_m is not None
        if smooth_length_m is None:
            smooth_length_m = graph_length_m
        self.edge_i = torch.as_tensor(edge_i.copy(), device=device, dtype=torch.long)
        self.edge_j = torch.as_tensor(edge_j.copy(), device=device, dtype=torch.long)
        self.conductance_m = torch.as_tensor(
            conductance.copy(), device=device, dtype=dtype
        )
        self.mass_weight = torch.as_tensor(
            effective_volume / total_effective_volume, device=device, dtype=dtype
        )
        self.factor_per_m = smooth_length_m**2 / total_effective_volume
        self.smooth_length_m = smooth_length_m
        self.graph_normalization_length_m = graph_length_m
        self.length_selection = (
            "explicit modeling length"
            if provided_length
            else "geometry-derived neighbor normalization"
        )
        self.total_effective_volume_m3 = total_effective_volume
        self.active_cell_count = len(ids)
        self.edge_count = len(edge_i)
        self.cross_muscle_edge_count = cross_muscle_edge_count
        self.graph_sha256 = {
            "edge_i": _array_sha256(edge_i),
            "edge_j": _array_sha256(edge_j),
            "conductance_m": _array_sha256(conductance),
        }
        self.muscle_count = len(np.unique(active_label))
        self.isolated_cell_count = int(
            np.count_nonzero(
                np.bincount(np.r_[edge_i, edge_j], minlength=len(ids)) == 0
            )
        )

    def __call__(self, raw6_activation_inv: torch.Tensor) -> torch.Tensor:
        assert raw6_activation_inv.shape == (self.active_cell_count, 6)
        assert raw6_activation_inv.device == self.conductance_m.device
        assert raw6_activation_inv.dtype == self.conductance_m.dtype
        difference = raw6_activation_inv[self.edge_i] - raw6_activation_inv[self.edge_j]
        frobenius2 = difference[:, :3].square().sum(dim=1) + 2 * difference[
            :, 3:
        ].square().sum(dim=1)
        value = self.factor_per_m * (self.conductance_m * frobenius2).sum()
        assert bool(torch.isfinite(value))
        return value

    def contract(self) -> dict[str, float | int | str]:
        return {
            "formula": "length^2 / total effective active volume * sum_same_muscle_face_edges((face area / centroid distance) * harmonic muscle fraction * squared raw6 tensor Frobenius difference)",
            "smooth_length_m": self.smooth_length_m,
            "graph_normalization_length_m": self.graph_normalization_length_m,
            "length_selection": self.length_selection,
            "effective_active_volume_m3": self.total_effective_volume_m3,
            "retained_active_cell_count": self.active_cell_count,
            "same_muscle_edge_count": self.edge_count,
            "cross_muscle_shared_faces_excluded": self.cross_muscle_edge_count,
            "muscle_count": self.muscle_count,
            "isolated_retained_active_cell_count": self.isolated_cell_count,
            "activation_order": "xx, yy, zz, xy, yz, xz; Frobenius square doubles off-diagonals",
            "input_sha256": self.input_sha256,
            "graph_sha256": self.graph_sha256,
        }


class FaceShapeActivationObjective:
    """Combine dimensionless position, normal, and smoothness terms."""

    def __init__(
        self,
        skin: SkinShapeLoss,
        smoothness: ActivationSmoothness,
        weights: ObjectiveWeights,
    ) -> None:
        self.skin = skin
        self.smoothness = smoothness
        self.weights = weights

    def components(
        self, full_displacement_m: torch.Tensor, raw6_activation_inv: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        position, normal = self.skin.components(full_displacement_m)
        smooth = self.smoothness(raw6_activation_inv)
        return {"position": position, "normal": normal, "smooth": smooth}

    def __call__(
        self, full_displacement_m: torch.Tensor, raw6_activation_inv: torch.Tensor
    ) -> torch.Tensor:
        terms = self.components(full_displacement_m, raw6_activation_inv)
        value = (
            terms["position"]
            + self.weights.normal * terms["normal"]
            + self.weights.smooth * terms["smooth"]
        )
        assert bool(torch.isfinite(value))
        return value

    @torch.no_grad()
    def metrics(
        self, full_displacement_m: torch.Tensor, raw6_activation_inv: torch.Tensor
    ) -> dict[str, float | int]:
        skin_metrics = self.skin.metrics(full_displacement_m)
        smooth = float(self.smoothness(raw6_activation_inv))
        normal_contribution = self.weights.normal * float(
            skin_metrics["normal_chord2_area_mean"]
        )
        smooth_contribution = self.weights.smooth * smooth
        return {
            **skin_metrics,
            "activation_smoothness": smooth,
            "normal_coefficient": self.weights.normal,
            "smooth_coefficient": self.weights.smooth,
            "normal_contribution": normal_contribution,
            "smooth_contribution": smooth_contribution,
            "objective": float(skin_metrics["position_normalized"])
            + normal_contribution
            + smooth_contribution,
        }


def raw6_dual_volume_norm(
    gradient: torch.Tensor, mass_weight: torch.Tensor
) -> torch.Tensor:
    """Physical dual norm of a raw6 covector under normalized effective volume."""
    assert gradient.ndim == 2 and gradient.shape[1] == 6
    assert mass_weight.shape == (len(gradient),)
    assert gradient.device == mass_weight.device and gradient.dtype == mass_weight.dtype
    assert bool(torch.isfinite(gradient).all() and torch.isfinite(mass_weight).all())
    assert bool(torch.all(mass_weight > 0))
    torch.testing.assert_close(
        mass_weight.sum(), mass_weight.new_tensor(1.0), rtol=1e-10, atol=1e-12
    )
    squared = gradient[:, :3].square().sum(dim=1) + 0.5 * gradient[:, 3:].square().sum(
        dim=1
    )
    return (squared / mass_weight).sum().sqrt()


def normal_anchor_coefficient(
    target_motion_scale2_m2: float,
    *,
    position_error_rms_m: float = 0.002,
    normal_angle_deg: float = 5.0,
) -> float:
    """Equate a declared vector-RMS error to one oriented normal angle.

    The normal term uses squared chord distance; this coefficient is a
    modeling hypothesis for calibration, not a coefficient transferred from a
    different mesh or objective normalization.
    """
    assert math.isfinite(target_motion_scale2_m2) and target_motion_scale2_m2 > 0
    assert math.isfinite(position_error_rms_m) and position_error_rms_m > 0
    assert math.isfinite(normal_angle_deg) and 0 < normal_angle_deg < 180
    chord2 = 4 * math.sin(math.radians(normal_angle_deg) / 2) ** 2
    return (position_error_rms_m**2 / target_motion_scale2_m2) / chord2


def calibrate_smooth_weight(
    position_gradient_q: torch.Tensor,
    normal_gradient_q: torch.Tensor,
    smooth_gradient_q: torch.Tensor,
    mass_weight: torch.Tensor,
    *,
    normal_coefficient: float,
    smooth_target_ratio: float,
) -> dict[str, float]:
    """Balance smoothness against position plus anchored normal at one state.

    All three inputs are derivatives of the *unweighted* dimensionless terms
    with respect to the same retained raw6 activation coordinates. The caller
    must obtain position and normal gradients through the same accepted
    collision-off equilibrium and validated unshifted adjoint. No solve or
    gradient approximation is performed here.
    """
    assert math.isfinite(normal_coefficient) and normal_coefficient > 0
    assert 0 < smooth_target_ratio < 1
    position_norm = float(raw6_dual_volume_norm(position_gradient_q, mass_weight))
    normal_norm = float(raw6_dual_volume_norm(normal_gradient_q, mass_weight))
    smooth_norm = float(raw6_dual_volume_norm(smooth_gradient_q, mass_weight))
    data_norm = float(
        raw6_dual_volume_norm(
            position_gradient_q + normal_coefficient * normal_gradient_q, mass_weight
        )
    )
    assert position_norm > 0 and normal_norm > 0 and smooth_norm > 0 and data_norm > 0
    return {
        "position_gradient_dual_norm": position_norm,
        "normal_gradient_dual_norm": normal_norm,
        "smooth_gradient_dual_norm": smooth_norm,
        "data_gradient_dual_norm": data_norm,
        "normal_coefficient": normal_coefficient,
        "weighted_normal_to_position_gradient_ratio": normal_coefficient
        * normal_norm
        / position_norm,
        "smooth_target_ratio": smooth_target_ratio,
        "smooth_coefficient": smooth_target_ratio * data_norm / smooth_norm,
    }


def normal_pose_gradient_ratio(
    position_gradient_pose_normalized: torch.Tensor,
    normal_gradient_pose_normalized: torch.Tensor,
    *,
    normal_coefficient: float,
) -> float:
    """Report anchored normal versus position in the six normalized pose DOFs."""
    assert math.isfinite(normal_coefficient) and normal_coefficient > 0
    assert (
        position_gradient_pose_normalized.shape
        == normal_gradient_pose_normalized.shape
        == (6,)
    )
    assert (
        position_gradient_pose_normalized.device
        == normal_gradient_pose_normalized.device
    )
    assert (
        position_gradient_pose_normalized.dtype == normal_gradient_pose_normalized.dtype
    )
    assert bool(torch.isfinite(position_gradient_pose_normalized).all())
    assert bool(torch.isfinite(normal_gradient_pose_normalized).all())
    position_norm = float(torch.linalg.vector_norm(position_gradient_pose_normalized))
    normal_norm = float(torch.linalg.vector_norm(normal_gradient_pose_normalized))
    assert position_norm > 0
    return normal_coefficient * normal_norm / position_norm
