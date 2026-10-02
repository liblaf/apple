"""CPU validation for the staged activation parameterizations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import torch
from activation_models import (
    STRESS_REF_MPA,
    controls_from_matrix,
    initialize_from_stress_numpy,
    initialize_learned_zero_amplitude_axes,
    learned_controls_from_fixed,
    mandel_frobenius2,
    mandel_to_matrix,
    matrices,
    project_,
    project_stress_numpy,
    zero_amplitude_count,
)
from experiment import Profile

from liblaf import cherries

DTYPE = torch.float64
MODES = ("symmetric6", "psd6", "rankone_fixed", "rankone_learned")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("05-activation-validation-inverse-v2-003")
    finite_difference_step: float = 1e-6


def close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-10) -> None:
    assert torch.allclose(actual, expected, rtol=1e-10, atol=atol), (actual, expected)


def fd_error(
    mode: str, q: torch.Tensor, axes: torch.Tensor | None, step: float
) -> float:
    q = q.detach().clone().requires_grad_(requires_grad=True)
    weight = torch.tensor(
        ((0.3, -0.2, 0.1), (-0.2, 0.5, 0.4), (0.1, 0.4, -0.7)), dtype=DTYPE
    )
    loss = (matrices(q, mode, axes) * weight).sum()
    loss.backward()
    assert q.grad is not None
    analytic = q.grad.detach().clone()
    numeric = torch.empty_like(q)
    for index in range(q.numel()):
        plus, minus = q.detach().clone(), q.detach().clone()
        plus.flatten()[index] += step
        minus.flatten()[index] -= step
        numeric.flatten()[index] = (
            (matrices(plus, mode, axes) * weight).sum()
            - (matrices(minus, mode, axes) * weight).sum()
        ) / (2 * step)
    return float((analytic - numeric).abs().max())


def main(cfg: Config) -> None:  # noqa: PLR0915
    assert STRESS_REF_MPA == 0.012 / (2.0 * 1.49)
    assert cfg.finite_difference_step > 0
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)

    raw = torch.tensor(
        ((0.8, 0.2, -0.1), (0.2, -0.4, 0.3), (-0.1, 0.3, 0.1)), dtype=DTYPE
    )
    q1, _ = controls_from_matrix(raw, "symmetric6")
    close(matrices(q1, "symmetric6"), raw)
    q2 = q1.clone()
    project_(q2, "psd6")
    psd = matrices(q2, "psd6")
    assert float(torch.linalg.eigvalsh(psd).min()) >= -1e-12

    q3, axis3 = controls_from_matrix(psd, "rankone_fixed")
    assert axis3 is not None
    rank1 = matrices(q3, "rankone_fixed", axis3)
    values, vectors = torch.linalg.eigh(psd)
    expected_rank1 = values[-1] * torch.outer(vectors[:, -1], vectors[:, -1])
    close(rank1, expected_rank1)
    q4, axes4 = controls_from_matrix(rank1, "rankone_learned")
    assert axes4 is None
    close(matrices(q4, "rankone_learned"), rank1)

    q_bad = torch.tensor((-2.0, 0.3, 0.0, 0.0), dtype=DTYPE)
    project_(q_bad, "rankone_learned")
    assert q_bad[0] == 0
    close(torch.linalg.vector_norm(q_bad[1:]), torch.ones((), dtype=DTYPE))
    assert torch.linalg.eigvalsh(matrices(q_bad, "rankone_learned")).min() >= -1e-12

    # Near a largest-component tie, axis-sign canonicalization would make one
    # optimizer update discontinuous. Projection must only normalize the axis.
    q_tied = torch.tensor(
        ((0.7, 1.0, -(1.0 + 1e-13), 0.2), (0.7, 1.0 + 1e-13, -1.0, 0.2)),
        dtype=DTYPE,
    )
    expected_axes = q_tied[:, 1:] / torch.linalg.vector_norm(
        q_tied[:, 1:], dim=-1, keepdim=True
    )
    project_(q_tied, "rankone_learned")
    close(q_tied[:, 1:], expected_axes, atol=1e-14)

    fd_inputs = {
        "symmetric6": (
            torch.tensor((0.4, -0.2, 0.3, 0.1, -0.05, 0.08), dtype=DTYPE),
            None,
        ),
        "psd6": (torch.tensor((0.8, 0.7, 0.9, 0.03, -0.02, 0.01), dtype=DTYPE), None),
        "rankone_fixed": (
            torch.tensor((0.7,), dtype=DTYPE),
            torch.tensor((0.2, -0.3, 0.9), dtype=DTYPE),
        ),
        "rankone_learned": (torch.tensor((0.7, 0.2, -0.3, 0.9), dtype=DTYPE), None),
    }
    fd = {
        mode: fd_error(mode, q, axes, cfg.finite_difference_step)
        for mode, (q, axes) in fd_inputs.items()
    }
    assert max(fd.values()) < 2e-8, fd

    angle = 0.61
    rotation = torch.tensor(
        (
            (np.cos(angle), -np.sin(angle), 0),
            (np.sin(angle), np.cos(angle), 0),
            (0, 0, 1),
        ),
        dtype=DTYPE,
    )
    rotated = rotation @ rank1 @ rotation.mT
    qr, ar = controls_from_matrix(rotated, "rankone_fixed")
    assert ar is not None
    close(matrices(qr, "rankone_fixed", ar), rotated)
    close(
        matrices(q4, "rankone_learned"),
        matrices(torch.cat((q4[:1], -q4[1:])), "rankone_learned"),
    )

    zero = torch.zeros((3, 3), dtype=DTYPE)
    qzero, azero = controls_from_matrix(zero, "rankone_fixed")
    assert azero is not None
    close(azero, torch.tensor((1.0, 0.0, 0.0), dtype=DTYPE))
    assert qzero.item() == 0
    assert zero_amplitude_count(zero) == 1

    learned_zero = learned_controls_from_fixed(qzero, azero)
    zero_gradient = torch.diag(torch.tensor((1.0, -1.0, 1.0), dtype=DTYPE))
    released = initialize_learned_zero_amplitude_axes(learned_zero, zero_gradient)
    close(matrices(released, "rankone_learned"), zero)
    activated = released.clone()
    activated[0] = 0.1
    assert float((matrices(activated, "rankone_learned") * zero_gradient).sum()) < 0

    stress = np.array(((0.004, 0.001, 0), (0.001, -0.002, 0), (0, 0, 0.003)))
    stress_psd = project_stress_numpy(stress)
    assert np.linalg.eigvalsh(stress_psd).min() >= -1e-12
    q_np, _ = initialize_from_stress_numpy(stress_psd, "psd6")
    reconstructed = mandel_to_matrix(torch.from_numpy(q_np)).numpy() * STRESS_REF_MPA
    assert np.allclose(reconstructed, stress_psd, rtol=1e-12, atol=1e-12)
    close(mandel_frobenius2(q1), raw.square().sum())

    receipt = {
        "passed": True,
        "modes": MODES,
        "mandel_order": "(xx, yy, zz, sqrt2*xy, sqrt2*yz, sqrt2*xz)",
        "stress_ref_mpa": STRESS_REF_MPA,
        "transitions": {
            "symmetric_to_psd": True,
            "psd_to_rankone": True,
            "rankone_to_learned": True,
        },
        "psd_projection": True,
        "finite_difference_max_abs_error": fd,
        "rotation_covariance": True,
        "axis_sign_invariance": True,
        "axis_projection_continuity_near_component_tie": True,
        "zero_amplitude_count": zero_amplitude_count(zero),
        "zero_amplitude_descent_axis_release": True,
        "learned_axis_zero_amplitude_direction_gradient": "zero; activate along a negative tensor-gradient eigenvector without changing the initial tensor",
        "physical_gradient_note": "Mandel6 Euclidean norm equals symmetric-tensor Frobenius norm, including off-diagonal factor two.",
        "activation_source": {
            "path": str(Path(__file__).with_name("activation_models.py").resolve()),
            "sha256": hashlib.sha256(
                Path(__file__).with_name("activation_models.py").read_bytes()
            ).hexdigest(),
        },
        "validation_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_metrics(
        {
            "finite_difference/max_abs_error": max(fd.values()),
            "stress_ref_mpa": STRESS_REF_MPA,
            "zero_amplitude_count": 1,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
