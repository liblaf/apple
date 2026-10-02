"""CPU diagnostics for the exact Koiter membrane metric used by this experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

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
PROTECTED_REGION = "right_nose_to_mouth"


def _file_record(path: Path) -> dict[str, Any]:
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


def _box_mask(points: np.ndarray, box: tuple[tuple[float, float], ...]) -> np.ndarray:
    return np.logical_and.reduce(
        tuple(
            (points[:, axis] >= lower) & (points[:, axis] <= upper)
            for axis, (lower, upper) in enumerate(box)
        )
    )


def _metric(vertices: np.ndarray) -> np.ndarray:
    a = vertices[:, 1] - vertices[:, 0]
    b = vertices[:, 2] - vertices[:, 0]
    return np.stack(
        (
            np.stack(
                (np.einsum("ij,ij->i", a, a), np.einsum("ij,ij->i", a, b)), axis=1
            ),
            np.stack(
                (np.einsum("ij,ij->i", a, b), np.einsum("ij,ij->i", b, b)), axis=1
            ),
        ),
        axis=1,
    )


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    if value.shape[0] != weight.shape[0] or not np.any(weight > 0):
        raise ValueError("weighted statistic has invalid or empty support")
    return float(np.sum(weight * value) / np.sum(weight))


class SkinStressDiagnostics:
    """Evaluate stretches and stress signs without invoking the face solver.

    ROI support uses lumped reference area: each triangle contributes one third
    of its reference area for each vertex selected by the frozen target-space ROI.
    """

    def __init__(
        self,
        fixture: Path = DEFAULT_FIXTURE,
        closeups_summary: Path = DEFAULT_CLOSEUPS,
        *,
        young_mpa: float = 0.2,
        nu: float = 0.46,
        thickness_m: float = 0.001,
    ) -> None:
        fixture = Path(fixture)
        if fixture.is_dir():
            fixture = fixture / "volume.vtu"
        self.volume_path = fixture.resolve()
        self.skin_path = self.volume_path.with_name("skin.vtp")
        self.closeups_path = Path(closeups_summary).resolve()
        if not (young_mpa > 0 and 0 < nu < 0.5 and thickness_m > 0):
            raise ValueError(
                "skin material constants must be positive and 0 < nu < 0.5"
            )
        self.young_mpa = float(young_mpa)
        self.nu = float(nu)
        self.thickness_m = float(thickness_m)
        self.lmbda_mpa = self.young_mpa * self.nu / (1 - self.nu**2)
        self.mu_mpa = self.young_mpa / (2 * (1 + self.nu))

        volume = pv.read(self.volume_path)
        skin = pv.read(self.skin_path)
        self.rest_volume = np.asarray(volume.points, dtype=np.float64)
        self.smile = np.asarray(volume.point_data["Smile"], dtype=np.float64)
        self.point_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
        self.rest = np.asarray(skin.points, dtype=np.float64)
        if not np.array_equal(self.rest, self.rest_volume[self.point_ids]):
            raise ValueError("skin points do not exactly map to volume GlobalPointId")
        faces = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
        if np.any(faces[:, 0] != 3):
            raise ValueError("skin must contain triangles only")
        self.triangles = faces[:, 1:]
        rest_vertices = self.rest[self.triangles]
        cross = np.cross(
            rest_vertices[:, 1] - rest_vertices[:, 0],
            rest_vertices[:, 2] - rest_vertices[:, 0],
        )
        self.reference_area = np.linalg.norm(cross, axis=1) / 2
        if (
            np.any(self.reference_area <= 0)
            or not np.isfinite(self.reference_area).all()
        ):
            raise ValueError("skin contains a non-finite or degenerate rest triangle")
        self.rest_metric = _metric(rest_vertices)
        self.rest_metric_inv = np.linalg.inv(self.rest_metric)

        payload = json.loads(self.closeups_path.read_text())
        reported = payload["exploratory_target_space_rois"]["rois"]
        target_points = self.rest + self.smile[self.point_ids]
        if not np.isfinite(target_points).all():
            raise ValueError("skin target coordinates are non-finite")
        self.roi_vertex_masks: dict[str, np.ndarray] = {}
        self.roi_boxes: dict[str, tuple[tuple[float, float], ...]] = {}
        for name, record in reported.items():
            box = tuple(
                tuple(float(value) for value in record["box_xyz_m"][axis])
                for axis in ("x", "y", "z")
            )
            mask = _box_mask(target_points, box)
            if int(mask.sum()) != int(record["count"]):
                raise ValueError(f"frozen ROI {name} has changed support")
            self.roi_boxes[name] = box
            self.roi_vertex_masks[name] = mask
        marked = np.logical_or.reduce(
            tuple(self.roi_vertex_masks[name] for name in MARKED_REGIONS)
        )
        self.roi_vertex_masks["marked_union"] = marked
        self.region_area_weights = {
            "all_skin": self.reference_area.copy(),
            **{
                name: self.reference_area * mask[self.triangles].mean(axis=1)
                for name, mask in self.roi_vertex_masks.items()
            },
        }
        if any(not np.any(weight > 0) for weight in self.region_area_weights.values()):
            raise ValueError("a skin stress ROI has empty support")

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "scope": "CPU-only diagnostics over supplied equilibrated displacement; no solve, adjoint, objective, or update",
            "inputs": {
                "fixture_volume": _file_record(self.volume_path),
                "fixture_skin": _file_record(self.skin_path),
                "closeups_summary": _file_record(self.closeups_path),
            },
            "material": {
                "young_mpa": self.young_mpa,
                "nu": self.nu,
                "thickness_m": self.thickness_m,
                "plane_stress_lambda_mpa": self.lmbda_mpa,
                "mu_mpa": self.mu_mpa,
            },
            "koiter_convention": {
                "effective_inverse_metric": "S = Ainv @ inverse(G0) @ transpose(Ainv)",
                "principal_elastic_stretch_squared": "eigenvalues(S @ g)",
                "strain_eigenvalue": "m_i = stretch_i**2 - 1",
                "principal_metric_stress_indicator_mpa": "lambda*sum(m) + 2*mu*m_i; common positive thickness*fraction*reference-area/4 weight omitted, so signs are exact",
                "energy": "reference_area*thickness*fraction/4 * (0.5*lambda*trace(M)**2 + mu*trace(M**2))",
                "bending_included": False,
            },
            "roi_support": {
                "selection_space": "frozen target coordinates rest_skin + Smile[GlobalPointId]",
                "triangle_weight": "reference area times fraction of triangle vertices inside ROI",
                "regions": {
                    name: {
                        "vertex_count": int(mask.sum()),
                        "vertex_mask_sha256": _array_sha256(mask),
                        "lumped_reference_area_m2": float(
                            self.region_area_weights[name].sum()
                        ),
                    }
                    for name, mask in self.roi_vertex_masks.items()
                },
            },
            "topology": {
                "skin_points": len(self.rest),
                "triangles": len(self.triangles),
                "global_point_ids_sha256": _array_sha256(self.point_ids),
                "triangles_sha256": _array_sha256(self.triangles),
                "reference_area_sha256": _array_sha256(self.reference_area),
            },
        }

    def evaluate(
        self,
        u: np.ndarray,
        activation_inv: np.ndarray,
        *,
        label: str,
        stress_zero_tol_mpa: float = 1e-12,
    ) -> dict[str, Any]:
        """Return JSON-compatible area-weighted stress and stretch diagnostics."""
        u = np.asarray(u, dtype=np.float64)
        activation_inv = np.asarray(activation_inv, dtype=np.float64)
        if u.shape != self.rest_volume.shape or not np.isfinite(u).all():
            raise ValueError("u must be a finite displacement for every volume point")
        if activation_inv.shape != (len(self.triangles), 3):
            raise ValueError("skin activation_inv must have shape (skin triangles, 3)")
        if not np.isfinite(activation_inv).all():
            raise ValueError("skin activation_inv is non-finite")
        if not (np.isfinite(stress_zero_tol_mpa) and stress_zero_tol_mpa >= 0):
            raise ValueError("stress tolerance must be finite and nonnegative")

        current = self.rest_volume + u
        current_metric = _metric(current[self.point_ids][self.triangles])
        ainv = np.zeros((len(self.triangles), 2, 2), dtype=np.float64)
        ainv[:, 0, 0] = 1 + activation_inv[:, 0]
        ainv[:, 1, 1] = 1 + activation_inv[:, 1]
        ainv[:, 0, 1] = ainv[:, 1, 0] = activation_inv[:, 2]
        if np.any(np.linalg.det(ainv) <= 0):
            raise ValueError("skin Ainv must be positive definite")
        inverse_metric = ainv @ self.rest_metric_inv @ np.transpose(ainv, (0, 2, 1))
        elastic_squared = np.linalg.eigvals(inverse_metric @ current_metric)
        if np.max(np.abs(elastic_squared.imag)) > 1e-10:
            raise FloatingPointError("elastic metric has non-real eigenvalues")
        elastic_squared = np.sort(elastic_squared.real, axis=1)
        if np.any(elastic_squared <= 0) or not np.isfinite(elastic_squared).all():
            raise FloatingPointError("elastic metric is non-positive or non-finite")
        stretch = np.sqrt(elastic_squared)
        strain_eigenvalue = elastic_squared - 1
        stress = (
            self.lmbda_mpa * strain_eigenvalue.sum(axis=1, keepdims=True)
            + 2 * self.mu_mpa * strain_eigenvalue
        )
        energy_density_mpa = 0.5 * self.lmbda_mpa * strain_eigenvalue.sum(
            axis=1
        ) ** 2 + self.mu_mpa * np.sum(strain_eigenvalue**2, axis=1)
        fraction = np.ones(len(self.triangles), dtype=np.float64)
        energy_j = (
            self.reference_area
            * self.thickness_m
            * fraction
            / 4
            * energy_density_mpa
            * 1e6
        )
        biaxial_tension = stress[:, 0] > stress_zero_tol_mpa
        biaxial_compression = stress[:, 1] < -stress_zero_tol_mpa
        mixed = (stress[:, 0] < -stress_zero_tol_mpa) & (
            stress[:, 1] > stress_zero_tol_mpa
        )
        neutral = ~(biaxial_tension | biaxial_compression | mixed)

        region_weights = {
            **self.region_area_weights,
            "prestrain_support": self.reference_area
            * (np.linalg.norm(activation_inv, axis=1) > 0),
        }
        regions = {}
        for name, weight in region_weights.items():
            total = float(weight.sum())
            if total <= 0:
                regions[name] = {"empty": True}
                continue
            regions[name] = {
                "empty": False,
                "reference_area_support_m2": total,
                "principal_elastic_stretch_mean": [
                    _weighted_mean(stretch[:, column], weight) for column in (0, 1)
                ],
                "principal_elastic_stretch_min": [
                    float(stretch[weight > 0, column].min()) for column in (0, 1)
                ],
                "principal_elastic_stretch_max": [
                    float(stretch[weight > 0, column].max()) for column in (0, 1)
                ],
                "principal_metric_stress_indicator_mean_mpa": [
                    _weighted_mean(stress[:, column], weight) for column in (0, 1)
                ],
                "area_weighted_fraction_biaxial_tension": _weighted_mean(
                    biaxial_tension, weight
                ),
                "area_weighted_fraction_biaxial_compression": _weighted_mean(
                    biaxial_compression, weight
                ),
                "area_weighted_fraction_mixed_sign": _weighted_mean(mixed, weight),
                "area_weighted_fraction_neutral_or_unclassified": _weighted_mean(
                    neutral, weight
                ),
                "area_weighted_fraction_any_tension": _weighted_mean(
                    stress[:, 1] > stress_zero_tol_mpa, weight
                ),
                "area_weighted_fraction_any_compression": _weighted_mean(
                    stress[:, 0] < -stress_zero_tol_mpa, weight
                ),
                "koiter_energy_j": float(
                    np.sum(energy_j * weight / self.reference_area)
                ),
            }
            category_sum = sum(
                regions[name][key]
                for key in (
                    "area_weighted_fraction_biaxial_tension",
                    "area_weighted_fraction_biaxial_compression",
                    "area_weighted_fraction_mixed_sign",
                    "area_weighted_fraction_neutral_or_unclassified",
                )
            )
            if abs(category_sum - 1) > 1e-12:
                raise AssertionError(f"stress-sign categories do not partition {name}")
        return {
            "label": label,
            "stress_zero_tolerance_mpa": stress_zero_tol_mpa,
            "arrays": {
                "u_sha256": _array_sha256(u),
                "activation_inv_sha256": _array_sha256(activation_inv),
            },
            "regions": regions,
        }
