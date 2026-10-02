"""Field parameterizations for the joint facial inverse experiment.

The module owns parameter coordinates, projections, and regularizers only.  It
does not assemble FEM energies or mutate solver state.  Six-vectors use a
Frobenius-orthonormal symmetric basis throughout::

    xx, yy, zz, sqrt(2) xy, sqrt(2) yz, sqrt(2) xz.

Bulk baseline coordinates are normalized by the corresponding infinitesimal
shear modulus.  Activation coordinates are normalized by a fixed caller-supplied
reference stress.  These choices make Euclidean coordinate norms equal tensor
Frobenius norms in the declared dimensionless fields.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

SQRT2 = math.sqrt(2.0)
BULK_TISSUES = ("fat", "aponeurosis", "muscle")
N_SHARED_COEFFICIENTS = 20
DEFAULT_EIGEN_BATCH_SIZE = 4096


# This is a sensitivity configuration, not a calibrated material set.  Every
# entry records whether it is a direct measurement, inverse fit, proxy, or
# implementation assumption so downstream reports cannot silently merge them.
APPROVED_FIELD_CONFIG: dict[str, Any] = {
    "schema": "joint-additive-stress-fields-v1",
    "status": "research-informed sensitivity configuration; not calibrated",
    "bulk_order": list(BULK_TISSUES),
    "parameterization": {
        "shared_parameter_attribute": "coefficients",
        "shared_coefficient_count": 20,
        "bulk_symmetric_coordinates": {
            "fat": [0, 6],
            "aponeurosis": [6, 12],
            "muscle": [12, 18],
        },
        "skin_isotropic_resultant_coordinate": 18,
        "skin_log_stiffness_multiplier_coordinate": 19,
        "activation_coordinates_per_active_tet": 6,
        "symmetric_coordinate_order": [
            "xx",
            "yy",
            "zz",
            "sqrt2_xy",
            "sqrt2_yz",
            "sqrt2_xz",
        ],
        "coordinate_status": "fixed implementation convention",
    },
    "materials": {
        "fat": {
            "young_mpa": 0.0112,
            "poisson": 0.46,
            "mu_mpa": 0.0038356164383561643,
            "lambda_code_mpa": 0.04794520547945208,
            "baseline_stress_scale_mpa": 0.0038356164383561643,
            "young_status": (
                "research-informed seed from apparent shear-wave elasticity; "
                "not a static Stable Neo-Hookean Young modulus"
            ),
            "poisson_status": "modeling assumption",
            "source": "https://doi.org/10.5114/ada.2018.79778",
            "source_scope": "deep medial cheek fat; 89 women; 11.2 +/- 6.9 kPa",
            "baseline_stress_prior": "zero-centered modeling prior; no measurement",
        },
        "aponeurosis": {
            "young_mpa": 1.693,
            "poisson": 0.35,
            "mu_mpa": 0.6270370370370371,
            "lambda_code_mpa": 2.0901234567901232,
            "baseline_stress_scale_mpa": 0.6270370370370371,
            "young_status": (
                "distant stiffness sensitivity proxy from native cervical "
                "SMAS/platysma tensile testing; not cheek aponeurosis calibration"
            ),
            "poisson_status": "modeling assumption",
            "source": "https://doi.org/10.1093/asjof/ojaf126",
            "source_scope": "native cervical SMAS/platysma; n=7; 1.693 +/- 0.543 MPa",
            "baseline_stress_prior": (
                "zero-centered modeling prior; no measured facial-aponeurosis "
                "baseline-stress distribution was identified"
            ),
        },
        "muscle": {
            "young_mpa": 0.0120,
            "alternate_young_mpa": [0.0183],
            "poisson": 0.46,
            "mu_mpa": 0.004109589041095891,
            "lambda_code_mpa": 0.051369863013698655,
            "baseline_stress_scale_mpa": 0.004109589041095891,
            "young_status": (
                "research-informed seed from probe-dependent apparent "
                "elastography; not passive static muscle calibration"
            ),
            "poisson_status": "modeling assumption",
            "source": "https://doi.org/10.4236/jbise.2019.1211037",
            "source_scope": (
                "relaxed zygomaticus major; 15 volunteers; 12.0 +/- 4.3 or "
                "18.3 +/- 3.7 kPa depending on probe"
            ),
            "baseline_stress_prior": "zero-centered modeling prior; no measurement",
        },
        "skin": {
            "reference_map": {
                "kind": "uniform",
                "young_mpa": 0.2,
                "mapping_status": (
                    "uniform computational map centered on a derived central-cheek "
                    "zero-stress tangent; not a registered full-face map"
                ),
            },
            "poisson": 0.46,
            "thickness_m": 0.001,
            "mu_mpa": 0.06849315068493152,
            "lambda_code_mpa": 0.8561643835616444,
            "baseline_stress_scale_mpa": 0.06849315068493152,
            "reference_resultant_scale_n_per_m": 68.49315068493152,
            "reference_resultant_scale_mpa_m": 6.849315068493152e-05,
            "young_status": (
                "derived tangent scale from an incompressible Ogden-QLV inverse "
                "fit; not a directly measured transferable SNH modulus"
            ),
            "poisson_status": "modeling assumption",
            "thickness_status": "fixed modeling assumption",
            "source": "https://doi.org/10.1016/j.jmbbm.2013.03.004",
            "source_scope": (
                "in-vivo facial skin inverse fit; reported regional prestresses "
                "are model-dependent and are not a registered stress map"
            ),
            "baseline_resultant_prior_n_per_m": 0.0,
            "baseline_resultant_prior_status": (
                "zero-centered initialization; Flynn central-cheek directional "
                "89.4/71.8 N/m conversion at 1 mm is retained as optional "
                "sensitivity evidence, not as this asset's measurement"
            ),
            "continuation": {
                "source_proxy_mean_n_per_m": 80.6,
                "initial_fraction": 0.01,
                "initial_target_n_per_m": 0.806,
                "source_proxy": "Flynn central-cheek 89.4/71.8 N/m directional values",
                "status": (
                    "modeling continuation target from a model-dependent literature "
                    "proxy; increase only after valid neutral equilibrium"
                ),
            },
        },
    },
    "constraints": {
        "baseline_epsilon": 0.10,
        "baseline_upper_mu_multiple": 10.0,
        "activation_reference_mpa": 0.012328767123287673,
        "activation_reference_status": (
            "fixed modeling scale equal to three times the configured muscle mu; "
            "not a measured activation stress"
        ),
        "activation_maximum_dimensionless": 10.0,
        "activation_cap_mpa": 0.12328767123287673,
        "skin_multiplier_bounds": [1.0 / 3.0, 3.0],
        "smooth_length_m": 0.005,
        "constraint_status": "fixed computational choices; not biological intervals",
    },
    "prior_weights": {
        "fat": 1.0,
        "aponeurosis": 1.0,
        "muscle": 1.0,
        "skin_baseline": 1.0,
        "skin_log_multiplier": 1.0,
        "weight_status": "dimensionless placeholders to be frozen by the runner protocol",
    },
}


def research_informed_material_config() -> dict[str, Any]:
    """Return an owned copy of the source- and assumption-labeled configuration."""
    return copy.deepcopy(APPROVED_FIELD_CONFIG)


def lame_from_young_poisson(young_mpa: float, poisson: float) -> tuple[float, float]:
    """Return ``(mu, lambda_code)`` for Apple's no-log polynomial energy."""
    if not young_mpa > 0.0:
        msg = "Young's modulus must be positive"
        raise ValueError(msg)
    if not 0.0 < poisson < 0.5:
        msg = "Poisson's ratio must lie strictly between zero and one half"
        raise ValueError(msg)
    mu = young_mpa / (2.0 * (1.0 + poisson))
    classical_lambda = young_mpa * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, classical_lambda + mu


def symmetric_matrices(coordinates: torch.Tensor) -> torch.Tensor:
    """Unpack Frobenius-orthonormal symmetric tensor coordinates."""
    if coordinates.shape[-1] != 6:
        msg = f"expected final coordinate dimension 6, got {coordinates.shape}"
        raise ValueError(msg)
    xx, yy, zz, xy, yz, xz = coordinates.unbind(-1)
    values = torch.stack(
        (
            xx,
            xy / SQRT2,
            xz / SQRT2,
            xy / SQRT2,
            yy,
            yz / SQRT2,
            xz / SQRT2,
            yz / SQRT2,
            zz,
        ),
        dim=-1,
    )
    return values.reshape(*coordinates.shape[:-1], 3, 3)


def symmetric_coordinates(matrices: torch.Tensor) -> torch.Tensor:
    """Pack symmetric matrices in the Frobenius-orthonormal basis."""
    if matrices.shape[-2:] != (3, 3):
        msg = f"expected final matrix dimensions (3, 3), got {matrices.shape}"
        raise ValueError(msg)
    return torch.stack(
        (
            matrices[..., 0, 0],
            matrices[..., 1, 1],
            matrices[..., 2, 2],
            SQRT2 * matrices[..., 0, 1],
            SQRT2 * matrices[..., 1, 2],
            SQRT2 * matrices[..., 0, 2],
        ),
        dim=-1,
    )


def activation_stresses_mpa(
    coordinates: torch.Tensor, reference_mpa: float
) -> torch.Tensor:
    """Map dimensionless activation coordinates to physical stress matrices."""
    if not reference_mpa > 0.0:
        msg = "activation reference stress must be positive"
        raise ValueError(msg)
    return reference_mpa * symmetric_matrices(coordinates)


@torch.no_grad()
def project_activation_(
    coordinates: torch.Tensor,
    maximum_dimensionless: float,
    *,
    batch_size: int = DEFAULT_EIGEN_BATCH_SIZE,
) -> dict[str, float]:
    """Project dense activation coordinates onto ``0 <= A/Aref <= maximum I``."""
    if coordinates.shape[-1] != 6:
        msg = "activation coordinates must end in six components"
        raise ValueError(msg)
    if not maximum_dimensionless > 0.0:
        msg = "activation maximum must be positive"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "projection batch size must be positive"
        raise ValueError(msg)
    flat = coordinates.reshape(-1, 6)
    squared_change = coordinates.new_zeros(())
    negative = 0
    upper = 0
    total_eigenvalues = 3 * flat.shape[0]
    for block in flat.split(batch_size):
        before = block.clone()
        values, vectors = torch.linalg.eigh(symmetric_matrices(block))
        bounded = values.clamp(0.0, maximum_dimensionless)
        projected = (vectors * bounded.unsqueeze(-2)) @ vectors.transpose(-1, -2)
        block.copy_(symmetric_coordinates(projected))
        squared_change += (block - before).square().sum()
        negative += int((values < 0.0).sum())
        upper += int((values > maximum_dimensionless).sum())
    denominator = max(flat.numel(), 1)
    eigen_denominator = max(total_eigenvalues, 1)
    return {
        "projection_coordinate_rms": float((squared_change / denominator).sqrt()),
        "projected_negative_eigenvalue_fraction": negative / eigen_denominator,
        "projected_upper_eigenvalue_fraction": upper / eigen_denominator,
    }


@dataclass(frozen=True)
class VolumeGraph:
    """Frozen neutral-reference graph for physical activation regularization."""

    i: torch.Tensor
    j: torch.Tensor
    conductance_m: torch.Tensor
    effective_cell_volume_m3: torch.Tensor
    smooth_length_m: float

    def __post_init__(  # noqa: C901, PLR0912 - validate the graph contract eagerly.
        self,
    ) -> None:
        if self.i.dtype != torch.long or self.j.dtype != torch.long:
            msg = "graph indices must use torch.long"
            raise TypeError(msg)
        if self.i.ndim != 1 or self.i.shape != self.j.shape:
            msg = "graph endpoints must be equal-length vectors"
            raise ValueError(msg)
        if self.conductance_m.shape != self.i.shape:
            msg = "one conductance is required per graph edge"
            raise ValueError(msg)
        if self.effective_cell_volume_m3.ndim != 1:
            msg = "effective cell volumes must be a vector"
            raise ValueError(msg)
        if not self.smooth_length_m > 0.0:
            msg = "smooth length must be positive"
            raise ValueError(msg)
        if bool((self.conductance_m <= 0.0).any()):
            msg = "all graph conductances must be positive"
            raise ValueError(msg)
        if bool((self.effective_cell_volume_m3 < 0.0).any()):
            msg = "effective cell volumes cannot be negative"
            raise ValueError(msg)
        if not float(self.effective_cell_volume_m3.sum()) > 0.0:
            msg = "effective tissue volume must be positive"
            raise ValueError(msg)
        if self.i.numel():
            n_cells = self.effective_cell_volume_m3.numel()
            if int(self.i.min()) < 0 or int(self.j.min()) < 0:
                msg = "graph indices cannot be negative"
                raise ValueError(msg)
            if int(self.i.max()) >= n_cells or int(self.j.max()) >= n_cells:
                msg = "graph index exceeds the cell count"
                raise ValueError(msg)
            if bool((self.i == self.j).any()):
                msg = "self edges are not allowed"
                raise ValueError(msg)
        devices = {
            self.i.device,
            self.j.device,
            self.conductance_m.device,
            self.effective_cell_volume_m3.device,
        }
        if len(devices) != 1:
            msg = "all graph tensors must share one device"
            raise ValueError(msg)

    @property
    def tissue_volume_m3(self) -> torch.Tensor:
        return self.effective_cell_volume_m3.sum()

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> VolumeGraph:
        return VolumeGraph(
            i=self.i.to(device=device),
            j=self.j.to(device=device),
            conductance_m=self.conductance_m.to(device=device, dtype=dtype),
            effective_cell_volume_m3=self.effective_cell_volume_m3.to(
                device=device, dtype=dtype
            ),
            smooth_length_m=self.smooth_length_m,
        )


def activation_regularizers(
    coordinates: torch.Tensor, graph: VolumeGraph
) -> dict[str, torch.Tensor]:
    """Return the exact normalized graph smoothness and magnitude contracts.

    ``coordinates`` may have shape ``(cells, 6)`` or
    ``(..., cells, 6)``.  Leading fields, normally expressions, are averaged
    only in the scalar ``smoothness`` and ``magnitude`` outputs; the per-field
    values remain available for reporting.
    """
    if coordinates.shape[-1] != 6:
        msg = "activation coordinates must end in six components"
        raise ValueError(msg)
    if coordinates.shape[-2] != graph.effective_cell_volume_m3.numel():
        msg = "activation cell count does not match the graph"
        raise ValueError(msg)
    if coordinates.device != graph.i.device:
        msg = "activation coordinates and graph must share one device"
        raise ValueError(msg)
    if coordinates.dtype != graph.conductance_m.dtype:
        msg = "activation coordinates and graph weights must share a dtype"
        raise ValueError(msg)
    delta = coordinates[..., graph.i, :] - coordinates[..., graph.j, :]
    edge_energy = (graph.conductance_m * delta.square().sum(dim=-1)).sum(dim=-1)
    smoothness_by_field = (
        graph.smooth_length_m**2 / graph.tissue_volume_m3 * edge_energy
    )
    normalized_mass = graph.effective_cell_volume_m3 / graph.tissue_volume_m3
    magnitude_by_field = (normalized_mass * coordinates.square().sum(dim=-1)).sum(
        dim=-1
    )
    return {
        "smoothness": smoothness_by_field.mean(),
        "magnitude": magnitude_by_field.mean(),
        "smoothness_by_field": smoothness_by_field,
        "magnitude_by_field": magnitude_by_field,
    }


class SharedFieldParameters(nn.Module):
    """The approved 20-coefficient shared-field initialization."""

    def __init__(  # noqa: C901 - validate the frozen material configuration eagerly.
        self,
        config: Mapping[str, Any] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = (
            research_informed_material_config()
            if config is None
            else copy.deepcopy(dict(config))
        )
        if tuple(self.config["bulk_order"]) != BULK_TISSUES:
            msg = f"bulk order must be {BULK_TISSUES}"
            raise ValueError(msg)
        self.coefficients = nn.Parameter(
            torch.zeros(N_SHARED_COEFFICIENTS, device=device, dtype=dtype)
        )
        parameterization = self.config["parameterization"]
        if parameterization["shared_parameter_attribute"] != "coefficients":
            msg = "shared parameter attribute must be coefficients"
            raise ValueError(msg)
        if int(parameterization["shared_coefficient_count"]) != N_SHARED_COEFFICIENTS:
            msg = "shared coefficient count must be 20"
            raise ValueError(msg)
        self._bulk_mu_mpa = tuple(
            float(self.config["materials"][name]["mu_mpa"]) for name in BULK_TISSUES
        )
        for name, configured_mu in zip(BULK_TISSUES, self._bulk_mu_mpa, strict=True):
            material = self.config["materials"][name]
            derived_mu, derived_lambda = lame_from_young_poisson(
                float(material["young_mpa"]), float(material["poisson"])
            )
            if not math.isclose(configured_mu, derived_mu, rel_tol=1.0e-14):
                msg = f"{name} configured mu is inconsistent with E and nu"
                raise ValueError(msg)
            if not math.isclose(
                float(material["lambda_code_mpa"]), derived_lambda, rel_tol=1.0e-14
            ):
                msg = f"{name} configured lambda is inconsistent with E and nu"
                raise ValueError(msg)
        skin = self.config["materials"]["skin"]
        derived_skin_mu, derived_skin_lambda = lame_from_young_poisson(
            float(skin["reference_map"]["young_mpa"]), float(skin["poisson"])
        )
        self._skin_mu_mpa = float(skin["mu_mpa"])
        self._skin_resultant_scale_n_per_m = float(
            skin["reference_resultant_scale_n_per_m"]
        )
        self._skin_resultant_scale_mpa_m = float(
            skin["reference_resultant_scale_mpa_m"]
        )
        expected_resultant = self._skin_mu_mpa * 1.0e6 * float(skin["thickness_m"])
        if not math.isclose(self._skin_mu_mpa, derived_skin_mu, rel_tol=1.0e-14):
            msg = "skin configured mu is inconsistent with E and nu"
            raise ValueError(msg)
        if not math.isclose(
            float(skin["lambda_code_mpa"]), derived_skin_lambda, rel_tol=1.0e-14
        ):
            msg = "skin configured lambda is inconsistent with E and nu"
            raise ValueError(msg)
        if not math.isclose(
            self._skin_resultant_scale_n_per_m, expected_resultant, rel_tol=1.0e-14
        ):
            msg = "skin resultant scale is inconsistent with mu and thickness"
            raise ValueError(msg)
        if not math.isclose(
            self._skin_resultant_scale_mpa_m,
            self._skin_resultant_scale_n_per_m * 1.0e-6,
            rel_tol=1.0e-14,
        ):
            msg = "skin MPa*m and N/m resultant scales are inconsistent"
            raise ValueError(msg)

    @property
    def bulk_coordinates(self) -> torch.Tensor:
        return self.coefficients[:18].reshape(3, 6)

    @property
    def skin_baseline_coordinate(self) -> torch.Tensor:
        return self.coefficients[18]

    @property
    def skin_log_multiplier(self) -> torch.Tensor:
        return self.coefficients[19]

    @property
    def activation_reference_mpa(self) -> float:
        return float(self.config["constraints"]["activation_reference_mpa"])

    @property
    def activation_maximum_dimensionless(self) -> float:
        return float(self.config["constraints"]["activation_maximum_dimensionless"])

    @property
    def activation_cap_mpa(self) -> float:
        return float(self.config["constraints"]["activation_cap_mpa"])

    @property
    def bulk_mu_mpa(self) -> tuple[float, float, float]:
        return self._bulk_mu_mpa

    @property
    def skin_resultant_scale_n_per_m(self) -> float:
        return self._skin_resultant_scale_n_per_m

    @property
    def skin_resultant_scale_mpa_m(self) -> float:
        return self._skin_resultant_scale_mpa_m

    def bulk_stresses_mpa(self) -> torch.Tensor:
        scales = self.coefficients.new_tensor(self._bulk_mu_mpa)
        return scales[:, None, None] * symmetric_matrices(self.bulk_coordinates)

    def skin_resultant_n_per_m(self) -> torch.Tensor:
        identity = torch.eye(
            2, device=self.coefficients.device, dtype=self.coefficients.dtype
        )
        return (
            self._skin_resultant_scale_n_per_m
            * self.skin_baseline_coordinate
            * identity
        )

    def skin_resultant_mpa_m(self) -> torch.Tensor:
        identity = torch.eye(
            2, device=self.coefficients.device, dtype=self.coefficients.dtype
        )
        return (
            self._skin_resultant_scale_mpa_m * self.skin_baseline_coordinate * identity
        )

    def skin_stiffness_multiplier(self) -> torch.Tensor:
        return self.skin_log_multiplier.exp()

    @torch.no_grad()
    def project_(self) -> dict[str, float]:
        constraints = self.config["constraints"]
        epsilon = float(constraints["baseline_epsilon"])
        upper_multiple = float(constraints["baseline_upper_mu_multiple"])
        if not 0.0 < epsilon <= 1.0:
            msg = "baseline epsilon must lie in (0, 1]"
            raise ValueError(msg)
        if not upper_multiple > 0.0:
            msg = "baseline upper multiple must be positive"
            raise ValueError(msg)
        before = self.coefficients.detach().clone()
        for index, mu_mpa in enumerate(self._bulk_mu_mpa):
            matrix = mu_mpa * symmetric_matrices(self.bulk_coordinates[index])
            values, vectors = torch.linalg.eigh(matrix)
            bounded = values.clamp(
                min=-(1.0 - epsilon) * mu_mpa,
                max=upper_multiple * mu_mpa,
            )
            projected = (vectors * bounded.unsqueeze(-2)) @ vectors.transpose(-1, -2)
            self.bulk_coordinates[index].copy_(
                symmetric_coordinates(projected) / mu_mpa
            )
        lower_multiplier, upper_multiplier = map(
            float, constraints["skin_multiplier_bounds"]
        )
        if not 0.0 < lower_multiplier <= upper_multiplier:
            msg = "invalid skin multiplier bounds"
            raise ValueError(msg)
        self.skin_log_multiplier.clamp_(
            min=math.log(lower_multiplier), max=math.log(upper_multiplier)
        )
        multiplier = float(self.skin_stiffness_multiplier())
        self.skin_baseline_coordinate.clamp_(
            min=-(1.0 - epsilon) * multiplier,
            max=upper_multiple * multiplier,
        )
        change = self.coefficients - before
        return {
            "projection_coordinate_rms": float(change.square().mean().sqrt()),
            "skin_stiffness_multiplier": multiplier,
        }

    def regularizers(self) -> dict[str, torch.Tensor]:
        """Return separate block priors and honest constant-field smoothness."""
        weights = self.config["prior_weights"]
        result: dict[str, torch.Tensor] = {}
        prior_total = self.coefficients.new_zeros(())
        for index, name in enumerate(BULK_TISSUES):
            magnitude = self.bulk_coordinates[index].square().sum()
            weighted = float(weights[name]) * magnitude
            result[f"{name}_magnitude"] = magnitude
            result[f"{name}_prior"] = weighted
            prior_total = prior_total + weighted
        skin_magnitude = self.skin_baseline_coordinate.square()
        skin_prior = float(weights["skin_baseline"]) * skin_magnitude
        stiffness_prior = (
            float(weights["skin_log_multiplier"]) * self.skin_log_multiplier.square()
        )
        result["skin_baseline_magnitude"] = skin_magnitude
        result["skin_baseline_prior"] = skin_prior
        result["skin_log_multiplier_prior"] = stiffness_prior
        result["prior_total"] = prior_total + skin_prior + stiffness_prior
        # The approved first basis is spatially constant.  Returning detached
        # mathematical zeros prevents reports from claiming a regularization
        # force that this basis cannot possess.
        result["bulk_spatial_smoothness"] = self.coefficients.new_zeros(())
        result["skin_spatial_smoothness"] = self.coefficients.new_zeros(())
        return result
