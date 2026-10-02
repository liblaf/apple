# ruff: noqa: EM101, EM102, TRY003
"""CPU-only frozen diagnostics for the physical-volume continuation."""

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
MARKED_REGIONS = (
    "right_mouth_corner",
    "right_lateral_cheek",
    "right_lower_cheek_jaw",
)
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
    value = float(np.sqrt(np.mean(np.square(values))))
    if not math.isfinite(value):
        raise FloatingPointError("non-finite RMS")
    return value


def _vector_rms(values: np.ndarray) -> float:
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"expected (n, 3) vector field, got {values.shape}")
    value = float(np.sqrt(np.mean(np.sum(np.square(values), axis=1))))
    if not math.isfinite(value):
        raise FloatingPointError("non-finite vector RMS")
    return value


def _matrix_frobenius_rms(values: np.ndarray) -> float:
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError(f"expected (n, 3, 3) matrix field, got {values.shape}")
    value = float(np.sqrt(np.mean(np.sum(np.square(values), axis=(1, 2)))))
    if not math.isfinite(value):
        raise FloatingPointError("non-finite matrix Frobenius RMS")
    return value


def _roi_mask(points: np.ndarray, box: tuple[tuple[float, float], ...]) -> np.ndarray:
    return np.all(
        [
            (points[:, axis] >= limits[0]) & (points[:, axis] <= limits[1])
            for axis, limits in enumerate(box)
        ],
        axis=0,
    )


def _packed_ainv(q: np.ndarray) -> np.ndarray:
    """Map the driver's Raw6 controls to its symmetric Ainv matrices."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError(f"expected q with shape (active_tets, 6), got {q.shape}")
    ainv = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    ainv[:, 0, 0] += q[:, 0]
    ainv[:, 1, 1] += q[:, 1]
    ainv[:, 2, 2] += q[:, 2]
    ainv[:, 0, 1] = ainv[:, 1, 0] = q[:, 3]
    ainv[:, 1, 2] = ainv[:, 2, 1] = q[:, 4]
    ainv[:, 0, 2] = ainv[:, 2, 0] = q[:, 5]
    return ainv


class ContinuationMetrics:
    """Evaluate fixed target-space and rest-surface diagnostic fields.

    Construction fixes every mask and the 5 mm cotangent high-pass operator.
    ``evaluate`` only reads arrays; it performs no forward, adjoint, or optimizer work.
    """

    def __init__(
        self,
        fixture: Path = DEFAULT_FIXTURE,
        closeups_summary: Path = DEFAULT_CLOSEUPS,
    ) -> None:
        fixture = Path(fixture)
        if fixture.is_dir():
            fixture = fixture / "volume.vtu"
        closeups_summary = Path(closeups_summary)
        self.fixture_path = fixture.resolve()
        self.closeups_summary_path = closeups_summary.resolve()
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
        faces = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
        if np.any(faces[:, 0] != 3):
            raise ValueError("skin must contain only triangles")
        self.triangles = faces[:, 1:]
        self.mass, self.normals, stiffness = self._surface_operators(
            np.asarray(skin.points, dtype=np.float64)
        )
        time = HIGH_PASS_SCALE_M**2 / 4.0
        self.highpass_solver = spla.factorized(
            (sp.diags(self.mass) + time * stiffness).tocsc()
        )
        constant = np.ones(len(self.mass))
        if (
            np.max(np.abs(constant - self.highpass_solver(self.mass * constant)))
            > 1e-10
        ):
            raise ValueError("5 mm high-pass operator fails constant invariance")
        target_skin_points = self.rest[self.skin_ids] + self.target[self.skin_ids]
        self.marked_skin_masks = {
            name: _roi_mask(target_skin_points, self.roi_boxes[name])
            for name in MARKED_REGIONS
        }
        if any(not mask.any() for mask in self.marked_skin_masks.values()):
            raise ValueError("a marked roughness region has empty skin support")

    def _surface_operators(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
        tri = self.triangles
        p0, p1, p2 = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
        cross = np.cross(p1 - p0, p2 - p0)
        double_area = np.linalg.norm(cross, axis=1)
        if np.any(double_area <= 0):
            raise ValueError("skin has a degenerate rest triangle")
        mass = np.zeros(len(points), dtype=np.float64)
        np.add.at(mass, tri.ravel(), np.repeat(double_area / 6.0, 3))
        normals = np.zeros_like(points)
        for column in range(3):
            np.add.at(normals, tri[:, column], cross)
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 0) or np.any(mass <= 0):
            raise ValueError("skin has an isolated vertex or undefined normal")
        normals /= lengths[:, None]
        cot0 = np.einsum("ij,ij->i", p1 - p0, p2 - p0) / double_area
        cot1 = np.einsum("ij,ij->i", p2 - p1, p0 - p1) / double_area
        cot2 = np.einsum("ij,ij->i", p0 - p2, p1 - p2) / double_area
        i = np.concatenate((tri[:, 1], tri[:, 2], tri[:, 0]))
        j = np.concatenate((tri[:, 2], tri[:, 0], tri[:, 1]))
        w = 0.5 * np.concatenate((cot0, cot1, cot2))
        stiffness = sp.coo_matrix(
            (
                np.concatenate((-w, -w, w, w)),
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

    def _highpass(self, scalar: np.ndarray) -> np.ndarray:
        low = self.highpass_solver(self.mass * scalar)
        high = scalar - low
        if not np.isfinite(high).all():
            raise FloatingPointError("non-finite high-pass field")
        return high

    @property
    def provenance(self) -> dict[str, Any]:
        """JSON-compatible receipts and definitions for the fixed measurements."""
        return {
            "scope": "CPU-only fixed diagnostics; no forward solve, adjoint solve, objective change, or optimizer update",
            "inputs": {
                "fixture": _file_record(self.fixture_path),
                "closeups_summary": _file_record(self.closeups_summary_path),
            },
            "target": {
                "field": "Smile",
                "selector": "IsFace & finite(Smile)",
                "count": len(self.top),
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
            "highpass": {
                "scale_mm": 5.0,
                "heat_time_m2": HIGH_PASS_SCALE_M**2 / 4.0,
                "operator": "rest-normal scalar minus solve((M + t*K), M*scalar); lumped area mass and cotangent stiffness on frozen rest skin",
                "roughness_regions": {
                    name: {
                        "selection_space": "target deformed skin coordinates frozen at construction",
                        "count": int(mask.sum()),
                        "mask_sha256": _array_sha256(mask),
                    }
                    for name, mask in self.marked_skin_masks.items()
                },
            },
            "activation": {
                "name": "Ainv",
                "parameterization": "I plus Raw6 packed symmetric q: (00,11,22,01,12,02)",
            },
        }

    def evaluate(
        self,
        u: np.ndarray,
        q: np.ndarray,
        previous_u: np.ndarray | None = None,
        previous_q: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Return scalar metrics suitable for one continuation CSV row."""
        u = np.asarray(u, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if u.shape != self.rest.shape:
            raise ValueError(f"u has shape {u.shape}; expected {self.rest.shape}")
        if q.ndim != 2 or q.shape != (len(q), 6):
            raise ValueError(f"q has shape {q.shape}; expected (active_tets, 6)")
        if not np.isfinite(u).all() or not np.isfinite(q).all():
            raise FloatingPointError("continuation fields must be finite")
        if previous_u is None:
            previous_u = u
        if previous_q is None:
            previous_q = q
        previous_u = np.asarray(previous_u, dtype=np.float64)
        previous_q = np.asarray(previous_q, dtype=np.float64)
        if previous_u.shape != u.shape or previous_q.shape != q.shape:
            raise ValueError("previous field shape differs from current field")
        error = u[self.top] - self.target_top
        values: dict[str, float] = {
            "fit_vector_rms_mm": 1000.0 * _vector_rms(error),
            "motion_vector_rms_mm": 1000.0 * _vector_rms(u[self.top]),
            "target_projection": float(
                np.sum(u[self.top] * self.target_top) / np.sum(self.target_top**2)
            ),
            "geometry_update_vector_rms_mm": 1000.0 * _vector_rms(u - previous_u),
            "geometry_update_face_vector_rms_mm": 1000.0
            * _vector_rms(u[self.top] - previous_u[self.top]),
            "q_update_rms": _rms(q - previous_q),
            "q_rms": _rms(q),
        }
        ainv = _packed_ainv(q)
        previous_ainv = _packed_ainv(previous_q)
        values["ainv_update_frobenius_rms"] = _matrix_frobenius_rms(
            ainv - previous_ainv
        )
        values["ainv_frobenius_rms"] = _matrix_frobenius_rms(ainv)
        squared = np.sum(error**2, axis=1)
        total_squared = float(squared.sum())
        for name, mask in self.roi_masks.items():
            values[f"roi_{name}_fit_vector_rms_mm"] = 1000.0 * _vector_rms(error[mask])
            values[f"roi_{name}_objective_share"] = float(
                squared[mask].sum() / total_squared
            )
        u_skin = u[self.skin_ids]
        residual_skin = u_skin - self.target[self.skin_ids]
        normal_displacement = np.einsum("ij,ij->i", u_skin, self.normals)
        normal_residual = np.einsum("ij,ij->i", residual_skin, self.normals)
        high_displacement = self._highpass(normal_displacement)
        high_residual = self._highpass(normal_residual)
        for name, mask in self.marked_skin_masks.items():
            values[f"roughness_{name}_normal_displacement_highpass_5mm_rms_mm"] = (
                self._scalar_rms_mm(high_displacement, mask)
            )
            values[f"roughness_{name}_normal_residual_highpass_5mm_rms_mm"] = (
                self._scalar_rms_mm(high_residual, mask)
            )
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("continuation metric is non-finite")
        return values
