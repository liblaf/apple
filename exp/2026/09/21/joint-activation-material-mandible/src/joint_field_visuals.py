"""Small coordinate-only summaries for constant and spatial baseline figures."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from joint_fields import BULK_TISSUES, symmetric_matrices


def coefficient_summary(coefficients: Any, materials: dict[str, Any]) -> dict[str, Any]:
    """Summarize anchor spectra, not inferred anatomical or volume averages."""
    layout = materials["parameterization"]
    values = torch.as_tensor(coefficients, dtype=torch.float64, device="cpu")
    assert values.shape == (layout["shared_coefficient_count"],)
    assert torch.isfinite(values).all()
    spatial = materials["schema"] == "joint-additive-spatial-stress-fields-v1"
    if not spatial:
        assert materials["schema"] == "joint-additive-stress-fields-v1"
    slices = layout[
        "bulk_anchor_coordinate_slices" if spatial else "bulk_symmetric_coordinates"
    ]
    spectra = []
    norms = []
    for name in BULK_TISSUES:
        start, stop = slices[name]
        anchors = values[start:stop].reshape(-1, 6)
        expected_count = layout["bulk_anchor_counts"][name] if spatial else 1
        assert anchors.shape[0] == expected_count
        scale = materials["materials"][name]["baseline_stress_scale_mpa"]
        principal = torch.linalg.eigvalsh(symmetric_matrices(anchors)) * scale * 1000
        spectra.append(
            torch.stack(
                (principal.min(0).values, principal.mean(0), principal.max(0).values)
            )
            .detach()
            .numpy()
        )
        norms.append(float(torch.sqrt(anchors.square().sum(1).mean())))
    skin = materials["materials"]["skin"]
    return {
        "spatial": spatial,
        "principal_anchor_min_mean_max_kpa": np.asarray(spectra).tolist(),
        "anchor_coordinate_rms_frobenius": norms,
        "skin_resultant_n_per_m": float(
            values[layout["skin_isotropic_resultant_coordinate"]]
        )
        * skin["reference_resultant_scale_n_per_m"],
        "skin_stiffness_multiplier": float(
            values[layout["skin_log_stiffness_multiplier_coordinate"]].exp()
        ),
        "scope": "Unweighted anchor means and ranges; not volume averages or measured stresses",
    }
