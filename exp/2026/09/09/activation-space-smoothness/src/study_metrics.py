# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""CPU-only frozen diagnostics for the activation-space smoothness study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[6]
DEFAULT_FIXTURE = (
    ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
)
DEFAULT_CLOSEUPS = (
    ROOT / "exp/2026/09/08/physical-volume-closeups/data/21-diagnostics/summary.json"
)
ROUGHNESS_REGIONS = (
    "right_mouth_corner",
    "right_lateral_cheek",
    "right_lower_cheek_jaw",
)
LOW_FREQUENCY_REGIONS = (*ROUGHNESS_REGIONS, "right_nose_to_mouth")
PRIMARY_SURFACE_REGION = "union_three_roughness_rois"
HIGH_PASS_SCALE_M = 0.005


def _file_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        raise ValueError("RMS has empty support")
    result = float(np.sqrt(np.mean(np.square(values))))
    if not math.isfinite(result):
        raise FloatingPointError("non-finite RMS")
    return result


def _vector_rms(values: np.ndarray) -> float:
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"expected (n, 3) vector field, got {values.shape}")
    result = float(np.sqrt(np.mean(np.sum(np.square(values), axis=1))))
    if not math.isfinite(result):
        raise FloatingPointError("non-finite vector RMS")
    return result


def _matrix_frobenius_rms(values: np.ndarray) -> float:
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError(f"expected (n, 3, 3) matrix field, got {values.shape}")
    result = float(np.sqrt(np.mean(np.sum(np.square(values), axis=(1, 2)))))
    if not math.isfinite(result):
        raise FloatingPointError("non-finite matrix Frobenius RMS")
    return result


def _roi_mask(points: np.ndarray, box: tuple[tuple[float, float], ...]) -> np.ndarray:
    return np.all(
        [
            (points[:, axis] >= limits[0]) & (points[:, axis] <= limits[1])
            for axis, limits in enumerate(box)
        ],
        axis=0,
    )


def _active_graph(
    points: np.ndarray,
    tets: np.ndarray,
    active_ids: np.ndarray,
    region: np.ndarray,
    fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return same-muscle shared-face finite-volume conductances."""
    active = tets[active_ids]
    face_pattern = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
    )
    faces = np.sort(active[:, face_pattern].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(active_ids), dtype=np.int64), 4)
    order = np.lexsort(faces.T[::-1])
    faces, owner = faces[order], owner[order]
    pair = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
    if np.any(np.diff(pair) == 1):
        raise ValueError("active mesh has a nonmanifold shared face")
    i, j = owner[pair], owner[pair + 1]
    same = region[i] == region[j]
    i, j, face = i[same], j[same], faces[pair[same]]
    xyz = points[face]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    centers = points[active].mean(axis=1)
    distance = np.linalg.norm(centers[i] - centers[j], axis=1)
    if np.any(distance <= 0.0):
        raise ValueError("active graph has coincident tetrahedron centers")
    frac = fraction[active_ids]
    weight = area / distance * (2.0 * frac[i] * frac[j] / (frac[i] + frac[j]))
    if not len(weight) or np.any(weight <= 0.0) or not np.isfinite(weight).all():
        raise ValueError("active graph has invalid conductance weights")
    return i, j, weight


class StudyMetrics:
    """Evaluate fixed surface and common physical-activation diagnostics.

    Construction freezes all masks, the 5 mm cotangent filter, active-cell
    volume weights, and same-muscle adjacency. ``evaluate`` performs CPU array
    operations only; it does not solve physics or modify an optimizer state.
    """

    def __init__(
        self,
        fixture: Path = DEFAULT_FIXTURE,
        closeups_summary: Path = DEFAULT_CLOSEUPS,
    ) -> None:
        fixture = Path(fixture)
        if fixture.is_dir():
            fixture = fixture / "volume.vtu"
        self.fixture_path = fixture.resolve()
        self.closeups_summary_path = Path(closeups_summary).resolve()
        mesh = pv.read(self.fixture_path)
        skin_path = self.fixture_path.with_name("skin.vtp")
        skin = pv.read(skin_path)

        self.rest = np.asarray(mesh.points, dtype=np.float64)
        self.target = np.asarray(mesh.point_data["Smile"], dtype=np.float64)
        self.top = np.flatnonzero(
            np.asarray(mesh.point_data["IsFace"], dtype=bool)
            & np.isfinite(self.target).all(axis=1)
        )
        if len(self.top) != 15_302:
            raise ValueError(
                f"expected 15302 finite IsFace target points, got {len(self.top)}"
            )
        self.target_top = self.target[self.top]
        self.target_points_top = self.rest[self.top] + self.target_top
        self.target_squared_norm = float(np.sum(self.target_top**2))
        if self.target_squared_norm <= 0.0:
            raise ValueError("target direction has zero norm")

        payload = json.loads(self.closeups_summary_path.read_text())
        reported = payload["exploratory_target_space_rois"]["rois"]
        self.roi_masks: dict[str, np.ndarray] = {}
        self.roi_boxes: dict[str, tuple[tuple[float, float], ...]] = {}
        for name, values in reported.items():
            raw_box = values["box_xyz_m"]
            box = tuple(
                tuple(float(v) for v in raw_box[axis]) for axis in ("x", "y", "z")
            )
            mask = _roi_mask(self.target_points_top, box)
            if int(mask.sum()) != int(values["count"]):
                raise ValueError(f"frozen ROI {name} no longer has the recorded count")
            self.roi_boxes[name] = box
            self.roi_masks[name] = mask

        self.skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
        if np.any(self.skin_ids < 0) or np.any(self.skin_ids >= len(self.rest)):
            raise ValueError("skin GlobalPointId is outside fixture point range")
        if not np.array_equal(
            np.asarray(skin.points, dtype=np.float64), self.rest[self.skin_ids]
        ):
            raise ValueError("skin rest points do not exactly map to fixture points")
        if not np.isfinite(self.target[self.skin_ids]).all():
            raise ValueError("skin target contains non-finite values")
        faces = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
        if np.any(faces[:, 0] != 3):
            raise ValueError("skin must contain only triangles")
        self.triangles = faces[:, 1:]
        self.mass, self.normals, stiffness = self._surface_operators(
            np.asarray(skin.points, dtype=np.float64)
        )
        time = HIGH_PASS_SCALE_M**2 / 4.0
        self.lowpass_solver = spla.factorized(
            (sp.diags(self.mass) + time * stiffness).tocsc()
        )
        constant = np.ones(len(self.mass))
        if np.max(np.abs(constant - self._lowpass(constant))) > 1e-10:
            raise ValueError("5 mm low-pass operator fails constant invariance")

        target_skin_points = self.rest[self.skin_ids] + self.target[self.skin_ids]
        self.skin_roi_masks = {
            name: _roi_mask(target_skin_points, self.roi_boxes[name])
            for name in LOW_FREQUENCY_REGIONS
        }
        if any(not mask.any() for mask in self.skin_roi_masks.values()):
            raise ValueError("a frozen surface ROI has empty support")
        self.primary_surface_mask = np.logical_or.reduce(
            [self.skin_roi_masks[name] for name in ROUGHNESS_REGIONS]
        )
        if not self.primary_surface_mask.any():
            raise ValueError("primary three-ROI surface union has empty support")
        self.target_normal = np.einsum(
            "ij,ij->i", self.target[self.skin_ids], self.normals
        )
        self.target_normal_lowpass = self._lowpass(self.target_normal)
        for name, mask in self.skin_roi_masks.items():
            weight = self.mass[mask]
            projection_denominator = float(
                np.sum(weight * self.target_normal_lowpass[mask] ** 2)
            )
            target_mean = float(
                np.sum(weight * self.target_normal_lowpass[mask]) / weight.sum()
            )
            if projection_denominator <= 0.0 or abs(target_mean) <= 1e-15:
                raise ValueError(f"surface ROI {name} has no retained target direction")
        primary_weight = self.mass[self.primary_surface_mask]
        primary_target_low = self.target_normal_lowpass[self.primary_surface_mask]
        self.primary_low_frequency_projection_denominator = float(
            np.sum(primary_weight * primary_target_low**2)
        )
        if self.primary_low_frequency_projection_denominator <= 0.0:
            raise ValueError("primary surface union has no retained target direction")

        cells = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)
        if np.any(cells[:, 0] != 4):
            raise ValueError("volume fixture must contain only tetrahedra")
        tets = cells[:, 1:]
        self.active_ids = np.flatnonzero(
            np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
        )
        region = np.asarray(mesh.cell_data["ActivationControlId"], dtype=np.int64)[
            self.active_ids
        ]
        fraction = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
        volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
        self.active_volume = volume[self.active_ids] * fraction[self.active_ids]
        if (
            np.any(self.active_volume <= 0.0)
            or not np.isfinite(self.active_volume).all()
        ):
            raise ValueError("active muscle-volume weights must be finite and positive")
        self.graph_i, self.graph_j, self.graph_weight = _active_graph(
            self.rest, tets, self.active_ids, region, fraction
        )

    def _surface_operators(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
        tri = self.triangles
        p0, p1, p2 = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
        cross = np.cross(p1 - p0, p2 - p0)
        double_area = np.linalg.norm(cross, axis=1)
        if np.any(double_area <= 0.0):
            raise ValueError("skin has a degenerate rest triangle")
        mass = np.zeros(len(points), dtype=np.float64)
        np.add.at(mass, tri.ravel(), np.repeat(double_area / 6.0, 3))
        normals = np.zeros_like(points)
        for column in range(3):
            np.add.at(normals, tri[:, column], cross)
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 0.0) or np.any(mass <= 0.0):
            raise ValueError("skin has an isolated vertex or undefined normal")
        normals /= lengths[:, None]
        cot0 = np.einsum("ij,ij->i", p1 - p0, p2 - p0) / double_area
        cot1 = np.einsum("ij,ij->i", p2 - p1, p0 - p1) / double_area
        cot2 = np.einsum("ij,ij->i", p0 - p2, p1 - p2) / double_area
        i = np.concatenate((tri[:, 1], tri[:, 2], tri[:, 0]))
        j = np.concatenate((tri[:, 2], tri[:, 0], tri[:, 1]))
        weight = 0.5 * np.concatenate((cot0, cot1, cot2))
        stiffness = sp.coo_matrix(
            (
                np.concatenate((-weight, -weight, weight, weight)),
                (np.concatenate((i, j, i, j)), np.concatenate((j, i, i, j))),
            ),
            shape=(len(points), len(points)),
        ).tocsr()
        if np.max(np.abs(np.asarray(stiffness.sum(axis=1)).reshape(-1))) > 1e-10:
            raise ValueError("cotangent stiffness fails constant preservation")
        return mass, normals, stiffness

    def _scalar_rms_mm(self, scalar: np.ndarray, mask: np.ndarray) -> float:
        weight = self.mass[mask]
        return 1000.0 * math.sqrt(
            float(np.sum(weight * scalar[mask] ** 2) / weight.sum())
        )

    def _weighted_mean_mm(self, scalar: np.ndarray, mask: np.ndarray) -> float:
        weight = self.mass[mask]
        return 1000.0 * float(np.sum(weight * scalar[mask]) / weight.sum())

    def _lowpass(self, scalar: np.ndarray) -> np.ndarray:
        low = np.asarray(self.lowpass_solver(self.mass * scalar), dtype=np.float64)
        if not np.isfinite(low).all():
            raise FloatingPointError("non-finite 5 mm low-pass field")
        return low

    def evaluate_surface(self, u_skin: np.ndarray) -> dict[str, float]:
        """Return frozen surface metrics from skin-vertex displacements only."""
        u_skin = np.asarray(u_skin, dtype=np.float64)
        if u_skin.shape != (len(self.skin_ids), 3):
            raise ValueError(
                f"skin displacement has shape {u_skin.shape}; "
                f"expected {(len(self.skin_ids), 3)}"
            )
        if not np.isfinite(u_skin).all():
            raise FloatingPointError("skin displacement must be finite")

        normal_displacement = np.einsum("ij,ij->i", u_skin, self.normals)
        normal_residual = normal_displacement - self.target_normal
        low_displacement = self._lowpass(normal_displacement)
        low_residual = low_displacement - self.target_normal_lowpass
        high_displacement = normal_displacement - low_displacement
        high_residual = normal_residual - low_residual

        primary_mask = self.primary_surface_mask
        primary_weight = self.mass[primary_mask]
        primary_target_low = self.target_normal_lowpass[primary_mask]
        values = {
            "primary_union_normal_residual_highpass_5mm_rms_mm": (
                self._scalar_rms_mm(high_residual, primary_mask)
            ),
            "primary_union_normal_displacement_highpass_5mm_rms_mm": (
                self._scalar_rms_mm(high_displacement, primary_mask)
            ),
            "primary_union_low_frequency_normal_target_projection": float(
                np.sum(
                    primary_weight * low_displacement[primary_mask] * primary_target_low
                )
                / self.primary_low_frequency_projection_denominator
            ),
        }
        for name, mask in self.skin_roi_masks.items():
            weight = self.mass[mask]
            target_low = self.target_normal_lowpass[mask]
            denominator = float(np.sum(weight * target_low**2))
            target_mean = self._weighted_mean_mm(self.target_normal_lowpass, mask)
            displacement_mean = self._weighted_mean_mm(low_displacement, mask)
            prefix = f"low_frequency_{name}_normal"
            values[f"{prefix}_target_projection"] = float(
                np.sum(weight * low_displacement[mask] * target_low) / denominator
            )
            values[f"{prefix}_target_rms_mm"] = self._scalar_rms_mm(
                self.target_normal_lowpass, mask
            )
            values[f"{prefix}_displacement_rms_mm"] = self._scalar_rms_mm(
                low_displacement, mask
            )
            values[f"{prefix}_residual_rms_mm"] = self._scalar_rms_mm(
                low_residual, mask
            )
            values[f"{prefix}_target_mean_mm"] = target_mean
            values[f"{prefix}_displacement_mean_mm"] = displacement_mean
            values[f"{prefix}_mean_retention"] = displacement_mean / target_mean
        for name in ROUGHNESS_REGIONS:
            mask = self.skin_roi_masks[name]
            values[f"roughness_{name}_normal_displacement_highpass_5mm_rms_mm"] = (
                self._scalar_rms_mm(high_displacement, mask)
            )
            values[f"roughness_{name}_normal_residual_highpass_5mm_rms_mm"] = (
                self._scalar_rms_mm(high_residual, mask)
            )
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("surface metric is non-finite")
        return values

    @property
    def provenance(self) -> dict[str, Any]:
        """Return JSON-compatible receipts for every frozen measurement."""
        return {
            "scope": "CPU-only fixed diagnostics; no solve or optimizer update",
            "inputs": {
                "fixture": _file_record(self.fixture_path),
                "skin": _file_record(self.fixture_path.with_name("skin.vtp")),
                "closeups_summary": _file_record(self.closeups_summary_path),
            },
            "target": {
                "field": "Smile",
                "selector": "IsFace & finite(Smile)",
                "count": len(self.top),
                "direction_projection": "unweighted Euclidean projection of face displacement onto the frozen target vector",
            },
            "target_space_rois": {
                name: {
                    "box_xyz_m": {
                        axis: self.roi_boxes[name][index]
                        for index, axis in enumerate(("x", "y", "z"))
                    },
                    "count": int(mask.sum()),
                    "mask_sha256": _array_sha256(mask),
                }
                for name, mask in self.roi_masks.items()
            },
            "surface": {
                "global_point_ids_sha256": _array_sha256(self.skin_ids),
                "triangles_sha256": _array_sha256(self.triangles),
                "point_count": len(self.skin_ids),
                "triangle_count": len(self.triangles),
            },
            "frequency_split": {
                "scale_mm": 5.0,
                "heat_time_m2": HIGH_PASS_SCALE_M**2 / 4.0,
                "lowpass_operator": "solve((M + t*K), M*scalar) on frozen rest skin",
                "highpass_operator": "scalar minus lowpass; rest-normal scalar field",
                "selection_space": "target-deformed skin coordinates frozen at construction",
                "low_frequency_regions": {
                    name: {
                        "count": int(mask.sum()),
                        "mask_sha256": _array_sha256(mask),
                    }
                    for name, mask in self.skin_roi_masks.items()
                },
                "roughness_regions": list(ROUGHNESS_REGIONS),
                "primary_surface_region": {
                    "name": PRIMARY_SURFACE_REGION,
                    "definition": (
                        "union of the three roughness-region masks; each frozen "
                        "skin vertex counted once"
                    ),
                    "count": int(self.primary_surface_mask.sum()),
                    "mask_sha256": _array_sha256(self.primary_surface_mask),
                    "area_m2": float(self.mass[self.primary_surface_mask].sum()),
                    "low_frequency_target_projection_denominator_m4": (
                        self.primary_low_frequency_projection_denominator
                    ),
                    "primary_metric": (
                        "area-weighted RMS of 5 mm high-pass rest-normal target "
                        "residual over the union"
                    ),
                },
            },
            "activation": {
                "name": "Z",
                "common_definition": {
                    "physical_volume_baseline": "B @ B.T - I",
                    "psd_tensile": "Q_eff / mu",
                },
                "active_ids_sha256": _array_sha256(self.active_ids),
                "active_count": len(self.active_ids),
                "volume_weight": "fixture Volume * MuscleFraction",
                "same_muscle_graph": {
                    "edge_count": len(self.graph_weight),
                    "i_sha256": _array_sha256(self.graph_i),
                    "j_sha256": _array_sha256(self.graph_j),
                    "conductance_sha256": _array_sha256(self.graph_weight),
                    "conductance": "shared face area / center distance times harmonic MuscleFraction",
                },
            },
        }

    def evaluate(
        self,
        u: np.ndarray,
        z_numpy: np.ndarray,
        previous_u: np.ndarray | None = None,
        previous_z: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Return model-comparable scalar metrics for one evaluated state."""
        u = np.asarray(u, dtype=np.float64)
        z = np.asarray(z_numpy, dtype=np.float64)
        if u.shape != self.rest.shape:
            raise ValueError(f"u has shape {u.shape}; expected {self.rest.shape}")
        expected_z_shape = (len(self.active_ids), 3, 3)
        if z.shape != expected_z_shape:
            raise ValueError(f"Z has shape {z.shape}; expected {expected_z_shape}")
        if not np.isfinite(u).all() or not np.isfinite(z).all():
            raise FloatingPointError("study fields must be finite")
        if not np.allclose(z, np.swapaxes(z, 1, 2), rtol=0.0, atol=1e-10):
            raise ValueError("common physical activation Z must be symmetric")
        if previous_u is None:
            previous_u = u
        if previous_z is None:
            previous_z = z
        previous_u = np.asarray(previous_u, dtype=np.float64)
        previous_z = np.asarray(previous_z, dtype=np.float64)
        if previous_u.shape != u.shape or previous_z.shape != z.shape:
            raise ValueError("previous field shape differs from current field")
        if not np.isfinite(previous_u).all() or not np.isfinite(previous_z).all():
            raise FloatingPointError("previous study fields must be finite")

        pred = u[self.top]
        error = pred - self.target_top
        projection = float(np.sum(pred * self.target_top) / self.target_squared_norm)
        fit = 1000.0 * _vector_rms(error)
        motion = 1000.0 * _vector_rms(pred)
        values: dict[str, float] = {
            "fit_rms_mm": fit,
            "motion_rms_mm": motion,
            "fit_vector_rms_mm": fit,
            "motion_vector_rms_mm": motion,
            "target_projection": projection,
            "target_projection_amplitude": projection,
            "target_direction_residual_vector_rms_mm": 1000.0
            * _vector_rms(pred - projection * self.target_top),
            "geometry_update_vector_rms_mm": 1000.0 * _vector_rms(u - previous_u),
            "geometry_update_face_vector_rms_mm": 1000.0
            * _vector_rms(pred - previous_u[self.top]),
            "z_update_frobenius_rms": _matrix_frobenius_rms(z - previous_z),
            "z_frobenius_rms": _matrix_frobenius_rms(z),
        }

        volume_sum = float(self.active_volume.sum())
        z_squared = np.sum(z**2, axis=(1, 2))
        values["z_volume_weighted_frobenius_rms"] = math.sqrt(
            float(np.sum(self.active_volume * z_squared) / volume_sum)
        )
        trace = np.trace(z, axis1=1, axis2=2)
        deviatoric = z - trace[:, None, None] * np.eye(3) / 3.0
        values["z_volume_weighted_trace_rms"] = math.sqrt(
            float(np.sum(self.active_volume * trace**2) / volume_sum)
        )
        values["z_volume_weighted_deviatoric_frobenius_rms"] = math.sqrt(
            float(
                np.sum(self.active_volume * np.sum(deviatoric**2, axis=(1, 2)))
                / volume_sum
            )
        )
        eigenvalues = np.linalg.eigvalsh(z)
        values["z_eigen_min"] = float(eigenvalues.min())
        values["z_eigen_max"] = float(eigenvalues.max())
        graph_jump_squared = np.sum(
            (z[self.graph_i] - z[self.graph_j]) ** 2, axis=(1, 2)
        )
        weighted_jump_sum = float(np.sum(self.graph_weight * graph_jump_squared))
        values["z_same_muscle_jump_frobenius_rms"] = math.sqrt(
            weighted_jump_sum / float(self.graph_weight.sum())
        )
        values["z_same_muscle_dirichlet_density_per_m2"] = (
            weighted_jump_sum / volume_sum
        )

        squared_error = np.sum(error**2, axis=1)
        total_squared_error = float(squared_error.sum())
        if total_squared_error <= 0.0:
            raise ValueError(
                "exact zero target residual has undefined ROI objective share"
            )
        for name, mask in self.roi_masks.items():
            values[f"roi_{name}_fit_vector_rms_mm"] = 1000.0 * _vector_rms(error[mask])
            values[f"roi_{name}_objective_share"] = float(
                squared_error[mask].sum() / total_squared_error
            )

        values.update(self.evaluate_surface(u[self.skin_ids]))
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("study metric is non-finite")
        return values
