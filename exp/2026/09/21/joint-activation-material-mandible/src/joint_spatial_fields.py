"""Inactive 80-coordinate spatial shared-baseline prototype.

The basis is an immutable external asset from the accepted exact-rest force
audit.  This module validates that asset and owns only field coordinates,
reconstruction, projection, and regularizers.  No live inverse runner imports
it, and no default path silently selects a basis.

Bulk anchor coordinates use the same Frobenius-orthonormal convention as
``joint_fields`` and are normalized by each tissue's shear modulus.  The two
skin coordinates remain a uniform isotropic resultant and a global log
stiffness multiplier.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joint_fields import (
    BULK_TISSUES,
    N_SHARED_COEFFICIENTS,
    SharedFieldParameters,
    research_informed_material_config,
    symmetric_coordinates,
    symmetric_matrices,
)
from torch import nn

SPATIAL_FIELD_SCHEMA = "joint-additive-spatial-stress-fields-v1"
AUDIT_SCHEMA = "joint-neutral-spatial-baseline-audit-v2"
EXPECTED_BASIS_SHA256 = (
    "693e07f7ba4b71415ec41e5e0791edf4a6fb65d0e4452f2fbbfcd7f5c96f095d"
)
EXPECTED_INPUT_ARRAYS_SHA256 = (
    "72b69e30759b2c1922220a625aa54b763c684d9b9887233bbe4e3e358e59b5bd"
)
EXPECTED_INPUT_MANIFEST_SHA256 = (
    "822263773ab2ca18c7eaf342498a93dfd9a5547aaaf620df7fde2a20d42b5cd2"
)
EXPECTED_MESH_CELL_COUNT = 1_146_517
ANCHOR_COUNTS: dict[str, int] = {
    "fat": 4,
    "aponeurosis": 4,
    "muscle": 5,
}
ANCHOR_COORDINATE_SLICES: dict[str, slice] = {
    "fat": slice(0, 24),
    "aponeurosis": slice(24, 48),
    "muscle": slice(48, 78),
}
N_BULK_COORDINATES = 78
N_SPATIAL_SHARED_COEFFICIENTS = 80
SKIN_BASELINE_INDEX = 78
SKIN_LOG_MULTIPLIER_INDEX = 79


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def spatial_field_config() -> dict[str, Any]:
    """Return the frozen material configuration with the spatial layout added."""
    config = research_informed_material_config()
    config["schema"] = SPATIAL_FIELD_SCHEMA
    config["status"] = (
        "inactive spatial sensitivity prototype; not calibrated or activated"
    )
    config["parameterization"] = {
        "shared_parameter_attribute": "coefficients",
        "shared_coefficient_count": N_SPATIAL_SHARED_COEFFICIENTS,
        "bulk_anchor_counts": copy.deepcopy(ANCHOR_COUNTS),
        "bulk_anchor_coordinate_slices": {
            name: [value.start, value.stop]
            for name, value in ANCHOR_COORDINATE_SLICES.items()
        },
        "skin_isotropic_resultant_coordinate": SKIN_BASELINE_INDEX,
        "skin_log_stiffness_multiplier_coordinate": SKIN_LOG_MULTIPLIER_INDEX,
        "activation_coordinates_per_active_tet": 6,
        "symmetric_coordinate_order": [
            "xx",
            "yy",
            "zz",
            "sqrt2_xy",
            "sqrt2_yz",
            "sqrt2_xz",
        ],
        "coordinate_status": "fixed inactive prototype convention",
    }
    config["spatial_basis"] = {
        "audit_schema": AUDIT_SCHEMA,
        "basis_sha256": EXPECTED_BASIS_SHA256,
        "input_arrays_sha256": EXPECTED_INPUT_ARRAYS_SHA256,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
        "mesh_cell_count": EXPECTED_MESH_CELL_COUNT,
        "regularizer_reduction": (
            "unweighted mean over three individually volume-normalized tissues"
        ),
        "activation_requirement": (
            "an activating runner must freeze and record a nonzero strong spatial "
            "roughness weight"
        ),
    }
    return config


def constant20_to_spatial80(coefficients: torch.Tensor) -> torch.Tensor:
    """Embed approved constant fields exactly into the 4/4/5 anchor basis."""
    if coefficients.shape[-1] != N_SHARED_COEFFICIENTS:
        msg = f"expected final dimension 20, got {coefficients.shape}"
        raise ValueError(msg)
    result = coefficients.new_empty(*coefficients.shape[:-1], 80)
    for tissue_index, tissue in enumerate(BULK_TISSUES):
        source = coefficients[..., 6 * tissue_index : 6 * (tissue_index + 1)]
        target = ANCHOR_COORDINATE_SLICES[tissue]
        result[..., target] = (
            source.unsqueeze(-2)
            .expand(*source.shape[:-1], ANCHOR_COUNTS[tissue], 6)
            .reshape(*source.shape[:-1], -1)
        )
    result[..., SKIN_BASELINE_INDEX] = coefficients[..., 18]
    result[..., SKIN_LOG_MULTIPLIER_INDEX] = coefficients[..., 19]
    return result


class SpatialSharedFieldParameters(nn.Module):
    """Validated inactive 80-coordinate shared spatial field.

    ``basis_path`` and ``audit_summary_path`` are mandatory by design.  The
    large immutable basis buffers are nonpersistent; checkpoints must carry the
    receipt returned by :meth:`basis_receipt` and reload the exact external
    artifact.
    """

    def __init__(  # noqa: C901, PLR0912, PLR0915 - fail-fast asset validation.
        self,
        basis_path: Path,
        audit_summary_path: Path,
        mesh_cell_count: int,
        *,
        material_config: Mapping[str, Any] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        basis_path = Path(basis_path)
        audit_summary_path = Path(audit_summary_path)
        if not basis_path.is_file():
            msg = f"spatial basis does not exist: {basis_path}"
            raise FileNotFoundError(msg)
        if not audit_summary_path.is_file():
            msg = f"spatial audit metadata does not exist: {audit_summary_path}"
            raise FileNotFoundError(msg)
        if mesh_cell_count != EXPECTED_MESH_CELL_COUNT:
            msg = (
                f"spatial basis requires {EXPECTED_MESH_CELL_COUNT} mesh cells, "
                f"got {mesh_cell_count}"
            )
            raise ValueError(msg)

        canonical_material = research_informed_material_config()
        if material_config is not None and dict(material_config) != canonical_material:
            msg = "spatial prototype accepts only the frozen approved material config"
            raise ValueError(msg)
        # Reuse the approved constant class's complete material/scale validation,
        # but do not retain its independent trainable parameter.
        validated = SharedFieldParameters(canonical_material, device="cpu")
        del validated
        self.config = spatial_field_config()

        # ``nn.Parameter(torch.zeros(..., device=None))`` follows PyTorch's
        # global default device, whereas ``Tensor.to(device=None)`` is a no-op.
        # Resolve the implicit device once so parameters and immutable basis
        # buffers cannot silently land on different devices.
        target_device = (
            torch.device(torch.get_default_device())
            if device is None
            else torch.device(device)
        )

        summary = json.loads(audit_summary_path.read_text())
        if summary.get("schema") != AUDIT_SCHEMA or summary.get("success") is not True:
            msg = "spatial audit metadata is not an accepted v2 receipt"
            raise ValueError(msg)
        if summary.get("active_model_changed") is not False:
            msg = "spatial audit must describe an inactive model"
            raise ValueError(msg)
        if int(summary.get("bulk_columns", -1)) != N_BULK_COORDINATES:
            msg = "spatial audit must contain exactly 78 bulk columns"
            raise ValueError(msg)
        if (
            int(summary.get("shared_coefficients_if_activated", -1))
            != N_SPATIAL_SHARED_COEFFICIENTS
        ):
            msg = "spatial audit must declare exactly 80 shared coefficients"
            raise ValueError(msg)
        bounds = summary.get("bounds", {})
        constraints = self.config["constraints"]
        expected_lower = -(1.0 - float(constraints["baseline_epsilon"]))
        expected_upper = float(constraints["baseline_upper_mu_multiple"])
        if bounds.get("kind") != "per-anchor tensor eigenvalues" or not math.isclose(
            float(bounds.get("lower", math.nan)), expected_lower, abs_tol=1.0e-15
        ):
            msg = "spatial audit lower spectral bound does not match the field contract"
            raise ValueError(msg)
        if not math.isclose(
            float(bounds.get("upper", math.nan)), expected_upper, abs_tol=1.0e-15
        ):
            msg = "spatial audit upper spectral bound does not match the field contract"
            raise ValueError(msg)
        hashes = summary.get("hashes", {})
        actual_basis_sha256 = _sha256(basis_path)
        if actual_basis_sha256 != EXPECTED_BASIS_SHA256:
            msg = "spatial basis bytes do not match the frozen accepted artifact"
            raise ValueError(msg)
        if hashes.get("basis_arrays") != actual_basis_sha256:
            msg = "spatial audit metadata does not identify the supplied basis"
            raise ValueError(msg)
        if hashes.get("input_arrays") != EXPECTED_INPUT_ARRAYS_SHA256:
            msg = "spatial audit input-array lineage does not match the frozen fixture"
            raise ValueError(msg)
        if hashes.get("input_manifest") != EXPECTED_INPUT_MANIFEST_SHA256:
            msg = "spatial audit manifest lineage does not match the frozen fixture"
            raise ValueError(msg)

        target_dtype = torch.get_default_dtype() if dtype is None else dtype
        if not target_dtype.is_floating_point:
            msg = "spatial fields require a floating-point dtype"
            raise TypeError(msg)
        arrays = np.load(basis_path, allow_pickle=False)
        local_g: dict[str, np.ndarray] = {}
        local_m: dict[str, np.ndarray] = {}
        offset = 0
        reconstructed_g = np.zeros((78, 78))
        reconstructed_m = np.zeros((78, 78))
        support_counts: dict[str, int] = {}
        for tissue in BULK_TISSUES:
            count = ANCHOR_COUNTS[tissue]
            ids = arrays[f"{tissue}_cell_ids"]
            phi = arrays[f"{tissue}_phi"]
            g = arrays[f"{tissue}_G"]
            m = arrays[f"{tissue}_M"]
            if ids.dtype != np.int64 or ids.ndim != 1 or len(ids) == 0:
                msg = f"{tissue} cell IDs must be a nonempty int64 vector"
                raise ValueError(msg)
            if np.any(np.diff(ids) <= 0) or ids[0] < 0 or ids[-1] >= mesh_cell_count:
                msg = f"{tissue} cell IDs are not sorted unique valid mesh IDs"
                raise ValueError(msg)
            if phi.dtype != np.float64 or phi.shape != (len(ids), count):
                msg = f"{tissue} basis must be float64 with {count} anchors"
                raise ValueError(msg)
            if not np.isfinite(phi).all() or phi.min() < 0.0:
                msg = f"{tissue} basis contains invalid partition weights"
                raise ValueError(msg)
            if np.max(np.abs(phi.sum(axis=1) - 1.0)) > 5.0e-15:
                msg = f"{tissue} basis rows do not form a partition of unity"
                raise ValueError(msg)
            for label, matrix in (("G", g), ("M", m)):
                if matrix.dtype != np.float64 or matrix.shape != (count, count):
                    msg = f"{tissue} {label} has the wrong shape or dtype"
                    raise ValueError(msg)
                if not np.isfinite(matrix).all() or not np.allclose(
                    matrix, matrix.T, rtol=0.0, atol=1.0e-14
                ):
                    msg = f"{tissue} {label} is not finite symmetric"
                    raise ValueError(msg)
            if np.linalg.eigvalsh(g).min() < -1.0e-12:
                msg = f"{tissue} G is not positive semidefinite"
                raise ValueError(msg)
            if np.linalg.eigvalsh(m).min() <= 0.0:
                msg = f"{tissue} M is not positive definite"
                raise ValueError(msg)
            if np.max(np.abs(g @ np.ones(count))) > 1.0e-12:
                msg = f"{tissue} G does not annihilate a constant field"
                raise ValueError(msg)
            basis_receipt = summary.get("basis", {}).get(tissue, {})
            if int(basis_receipt.get("positive_cells", -1)) != len(ids):
                msg = f"{tissue} metadata support count does not match the basis"
                raise ValueError(msg)
            if len(basis_receipt.get("anchor_cell_ids", [])) != count:
                msg = f"{tissue} metadata anchor count does not match the basis"
                raise ValueError(msg)

            self.register_buffer(
                f"_{tissue}_cell_ids",
                torch.from_numpy(ids.copy()).to(device=target_device),
                persistent=False,
            )
            self.register_buffer(
                f"_{tissue}_phi",
                torch.from_numpy(phi.copy()).to(
                    device=target_device, dtype=target_dtype
                ),
                persistent=False,
            )
            self.register_buffer(
                f"_{tissue}_G",
                torch.from_numpy(g.copy()).to(device=target_device, dtype=target_dtype),
                persistent=False,
            )
            self.register_buffer(
                f"_{tissue}_M",
                torch.from_numpy(m.copy()).to(device=target_device, dtype=target_dtype),
                persistent=False,
            )
            local_g[tissue], local_m[tissue] = g, m
            width = 6 * count
            reconstructed_g[offset : offset + width, offset : offset + width] = (
                np.kron(g, np.eye(6)) / 3.0
            )
            reconstructed_m[offset : offset + width, offset : offset + width] = (
                np.kron(m, np.eye(6)) / 3.0
            )
            support_counts[tissue] = len(ids)
            offset += width
        if not np.allclose(arrays["G"], reconstructed_g, rtol=1.0e-13, atol=1.0e-15):
            msg = "stored global G does not equal the three-tissue mean block form"
            raise ValueError(msg)
        if not np.allclose(arrays["M"], reconstructed_m, rtol=1.0e-13, atol=1.0e-15):
            msg = "stored global M does not equal the three-tissue mean block form"
            raise ValueError(msg)
        arrays.close()

        self.mesh_cell_count = mesh_cell_count
        self._bulk_mu_mpa = tuple(
            float(self.config["materials"][name]["mu_mpa"]) for name in BULK_TISSUES
        )
        skin = self.config["materials"]["skin"]
        self._skin_resultant_scale_n_per_m = float(
            skin["reference_resultant_scale_n_per_m"]
        )
        self._skin_resultant_scale_mpa_m = float(
            skin["reference_resultant_scale_mpa_m"]
        )
        self._basis_receipt = {
            "schema": "joint-spatial-basis-binding-v1",
            "audit_schema": AUDIT_SCHEMA,
            "basis_sha256": actual_basis_sha256,
            "audit_summary_sha256": _sha256(audit_summary_path),
            "input_arrays_sha256": hashes["input_arrays"],
            "input_manifest_sha256": hashes["input_manifest"],
            "mesh_cell_count": mesh_cell_count,
            "anchor_counts": copy.deepcopy(ANCHOR_COUNTS),
            "support_cell_counts": support_counts,
            "basis_path": str(basis_path.resolve()),
            "audit_summary_path": str(audit_summary_path.resolve()),
            "persistent_basis_buffers": False,
        }
        self.coefficients = nn.Parameter(
            torch.zeros(80, device=target_device, dtype=target_dtype)
        )

    @property
    def skin_baseline_index(self) -> int:
        return SKIN_BASELINE_INDEX

    @property
    def skin_log_multiplier_index(self) -> int:
        return SKIN_LOG_MULTIPLIER_INDEX

    @property
    def skin_baseline_coordinate(self) -> torch.Tensor:
        return self.coefficients[SKIN_BASELINE_INDEX]

    @property
    def skin_log_multiplier(self) -> torch.Tensor:
        return self.coefficients[SKIN_LOG_MULTIPLIER_INDEX]

    @property
    def skin_resultant_scale_n_per_m(self) -> float:
        return self._skin_resultant_scale_n_per_m

    @property
    def skin_resultant_scale_mpa_m(self) -> float:
        return self._skin_resultant_scale_mpa_m

    @property
    def bulk_mu_mpa(self) -> tuple[float, float, float]:
        return self._bulk_mu_mpa

    @property
    def activation_reference_mpa(self) -> float:
        return float(self.config["constraints"]["activation_reference_mpa"])

    @property
    def activation_maximum_dimensionless(self) -> float:
        return float(self.config["constraints"]["activation_maximum_dimensionless"])

    @property
    def free_coordinate_ids(self) -> tuple[int, ...]:
        """Coordinates free when the neutral skin target at index 78 is fixed."""
        return (*range(SKIN_BASELINE_INDEX), SKIN_LOG_MULTIPLIER_INDEX)

    @property
    def free_coordinate_indices(self) -> torch.Tensor:
        """Device-local indices for the fixed-skin neutral contract."""
        return torch.tensor(
            self.free_coordinate_ids,
            device=self.coefficients.device,
            dtype=torch.long,
        )

    @property
    def all_coordinate_indices(self) -> torch.Tensor:
        return torch.arange(
            N_SPATIAL_SHARED_COEFFICIENTS,
            device=self.coefficients.device,
            dtype=torch.long,
        )

    def basis_receipt(self) -> dict[str, Any]:
        """Return owned immutable lineage metadata for a checkpoint."""
        return copy.deepcopy(self._basis_receipt)

    def cell_ids(self, tissue: str) -> torch.Tensor:
        self._validate_tissue(tissue)
        return getattr(self, f"_{tissue}_cell_ids")

    def basis_weights(self, tissue: str) -> torch.Tensor:
        self._validate_tissue(tissue)
        return getattr(self, f"_{tissue}_phi")

    def anchor_coordinates(self, tissue: str) -> torch.Tensor:
        self._validate_tissue(tissue)
        return self.coefficients[ANCHOR_COORDINATE_SLICES[tissue]].reshape(
            ANCHOR_COUNTS[tissue], 6
        )

    def bulk_dimensionless_support_coordinates(self, tissue: str) -> torch.Tensor:
        """Reconstruct one tissue's normalized six-coordinate support field."""
        return self.basis_weights(tissue) @ self.anchor_coordinates(tissue)

    def bulk_support_stress_mpa(self, tissue: str) -> torch.Tensor:
        """Return physical 3x3 stress on the tissue's positive-fraction cells."""
        tissue_index = BULK_TISSUES.index(tissue)
        values = self.bulk_dimensionless_support_coordinates(tissue)
        return self._bulk_mu_mpa[tissue_index] * symmetric_matrices(values)

    def bulk_stress_mpa(self, tissue: str) -> torch.Tensor:
        """Return one full-cell physical stress field, zero off its support."""
        support = self.bulk_support_stress_mpa(tissue)
        result = support.new_zeros((self.mesh_cell_count, 3, 3))
        return result.index_copy(0, self.cell_ids(tissue), support)

    def bulk_stresses_mpa(self) -> torch.Tensor:
        """Return full physical stress fields with shape ``[3,cells,3,3]``."""
        return torch.stack([self.bulk_stress_mpa(t) for t in BULK_TISSUES])

    def skin_resultant_n_per_m(self) -> torch.Tensor:
        identity = torch.eye(
            2, dtype=self.coefficients.dtype, device=self.coefficients.device
        )
        return (
            self._skin_resultant_scale_n_per_m
            * self.skin_baseline_coordinate
            * identity
        )

    def skin_resultant_mpa_m(self) -> torch.Tensor:
        identity = torch.eye(
            2, dtype=self.coefficients.dtype, device=self.coefficients.device
        )
        return (
            self._skin_resultant_scale_mpa_m * self.skin_baseline_coordinate * identity
        )

    def skin_stiffness_multiplier(self) -> torch.Tensor:
        return self.skin_log_multiplier.exp()

    @torch.no_grad()
    def set_skin_resultant_target_(self, target_n_per_m: float) -> None:
        """Set the uniform isotropic skin coordinate to an explicit fixed target."""
        if not math.isfinite(target_n_per_m):
            msg = "skin target must be finite"
            raise ValueError(msg)
        self.skin_baseline_coordinate.copy_(
            self.coefficients.new_tensor(
                target_n_per_m / self._skin_resultant_scale_n_per_m
            )
        )

    @torch.no_grad()
    def load_constant20_(self, coefficients: torch.Tensor) -> None:
        """Initialize from an exact constant-model checkpoint embedding."""
        embedded = constant20_to_spatial80(
            coefficients.to(
                device=self.coefficients.device, dtype=self.coefficients.dtype
            )
        )
        if embedded.ndim != 1:
            msg = "a checkpoint embedding must be one 20-coordinate vector"
            raise ValueError(msg)
        self.coefficients.copy_(embedded)

    @torch.no_grad()
    def project_(self) -> dict[str, float]:
        """Project all bulk anchors and both constrained skin coordinates."""
        constraints = self.config["constraints"]
        epsilon = float(constraints["baseline_epsilon"])
        upper = float(constraints["baseline_upper_mu_multiple"])
        lower = -(1.0 - epsilon)
        before = self.coefficients.detach().clone()
        for tissue in BULK_TISSUES:
            values, vectors = torch.linalg.eigh(
                symmetric_matrices(self.anchor_coordinates(tissue))
            )
            bounded = values.clamp(min=lower, max=upper)
            projected = (vectors * bounded.unsqueeze(-2)) @ vectors.transpose(-1, -2)
            self.anchor_coordinates(tissue).copy_(symmetric_coordinates(projected))
        lower_multiplier, upper_multiplier = map(
            float, constraints["skin_multiplier_bounds"]
        )
        self.skin_log_multiplier.clamp_(
            min=math.log(lower_multiplier), max=math.log(upper_multiplier)
        )
        multiplier = float(self.skin_stiffness_multiplier())
        self.skin_baseline_coordinate.clamp_(
            min=lower * multiplier, max=upper * multiplier
        )
        change = self.coefficients - before
        return {
            "projection_coordinate_rms": float(change.square().mean().sqrt()),
            "minimum_bulk_anchor_eigenvalue": min(
                float(
                    torch.linalg.eigvalsh(
                        symmetric_matrices(self.anchor_coordinates(tissue))
                    )
                    .min()
                    .detach()
                )
                for tissue in BULK_TISSUES
            ),
            "maximum_bulk_anchor_eigenvalue": max(
                float(
                    torch.linalg.eigvalsh(
                        symmetric_matrices(self.anchor_coordinates(tissue))
                    )
                    .max()
                    .detach()
                )
                for tissue in BULK_TISSUES
            ),
            "skin_stiffness_multiplier": multiplier,
        }

    def regularizers(self) -> dict[str, torch.Tensor]:
        """Return exact coarse G/M forms; roughness is never folded into priors."""
        result: dict[str, torch.Tensor] = {}
        roughness_fields, magnitude_fields = [], []
        bulk_prior = self.coefficients.new_zeros(())
        weights = self.config["prior_weights"]
        for tissue in BULK_TISSUES:
            coordinates = self.anchor_coordinates(tissue)
            g = getattr(self, f"_{tissue}_G")
            m = getattr(self, f"_{tissue}_M")
            roughness = torch.sum(coordinates * (g @ coordinates))
            magnitude = torch.sum(coordinates * (m @ coordinates))
            result[f"{tissue}_spatial_roughness"] = roughness
            result[f"{tissue}_magnitude"] = magnitude
            roughness_fields.append(roughness)
            magnitude_fields.append(magnitude)
            bulk_prior = bulk_prior + float(weights[tissue]) * magnitude
        roughness_by_tissue = torch.stack(roughness_fields)
        magnitude_by_tissue = torch.stack(magnitude_fields)
        result["bulk_spatial_roughness_by_tissue"] = roughness_by_tissue
        result["bulk_spatial_magnitude_by_tissue"] = magnitude_by_tissue
        result["bulk_spatial_roughness"] = roughness_by_tissue.mean()
        result["bulk_spatial_magnitude"] = magnitude_by_tissue.mean()
        result["bulk_magnitude_prior"] = bulk_prior
        skin_magnitude = self.skin_baseline_coordinate.square()
        skin_prior = float(weights["skin_baseline"]) * skin_magnitude
        stiffness_prior = (
            float(weights["skin_log_multiplier"]) * self.skin_log_multiplier.square()
        )
        result["skin_baseline_magnitude"] = skin_magnitude
        result["skin_baseline_prior"] = skin_prior
        result["skin_log_multiplier_prior"] = stiffness_prior
        result["prior_total"] = bulk_prior + skin_prior + stiffness_prior
        # An activating runner must add a separately frozen nonzero weight times
        # bulk_spatial_roughness.  Keeping it outside prior_total prevents a
        # missing or zero strong weight from being hidden.
        return result

    @staticmethod
    def _validate_tissue(tissue: str) -> None:
        if tissue not in BULK_TISSUES:
            msg = f"unknown bulk tissue: {tissue}"
            raise KeyError(msg)


__all__ = [
    "ANCHOR_COORDINATE_SLICES",
    "ANCHOR_COUNTS",
    "AUDIT_SCHEMA",
    "EXPECTED_BASIS_SHA256",
    "EXPECTED_MESH_CELL_COUNT",
    "N_BULK_COORDINATES",
    "N_SPATIAL_SHARED_COEFFICIENTS",
    "SKIN_BASELINE_INDEX",
    "SKIN_LOG_MULTIPLIER_INDEX",
    "SPATIAL_FIELD_SCHEMA",
    "SpatialSharedFieldParameters",
    "constant20_to_spatial80",
    "spatial_field_config",
]
