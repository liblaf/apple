"""CPU checks for normalized mixed-loss algebra and autograd."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pydantic_settings as ps
import torch
from study import normalized_data

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = cherries.output("05-loss/checks.json", mkdir=True)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check() -> dict[str, object]:
    L20 = torch.tensor(8.656092875221388, dtype=torch.float64)
    Lg0 = torch.tensor(0.07413172241432929, dtype=torch.float64)
    K = L20.clone()
    checks: dict[str, object] = {"beta": {}}

    for beta in (0.25, 1.0, 4.0):
        l2 = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
        gradient = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
        mixed, components = normalized_data(
            l2,
            gradient,
            l2_0=L20,
            gradient_0=Lg0,
            scale=K,
            kind="mixed",
            beta=beta,
        )
        expected = K * (l2 / L20 + beta * gradient / Lg0) / (1 + beta)
        assert torch.allclose(mixed, expected)
        mixed.backward()
        expected_l2_gradient = torch.tensor(1 / (1 + beta), dtype=torch.float64)
        expected_gradient_gradient = K * beta / ((1 + beta) * Lg0)
        assert torch.allclose(l2.grad, expected_l2_gradient)
        assert torch.allclose(gradient.grad, expected_gradient_gradient)
        assert torch.allclose(
            components["position_data_contribution"], l2.detach() / (1 + beta)
        )
        checks["beta"][str(beta)] = {
            "objective": float(mixed.detach()),
            "l2_gradient": float(l2.grad),
            "gradient_gradient": float(gradient.grad),
            "expected_l2_gradient": float(expected_l2_gradient),
            "expected_gradient_gradient": float(expected_gradient_gradient),
        }

    l2_only, _ = normalized_data(
        torch.tensor(1.7, dtype=torch.float64),
        torch.tensor(0.03, dtype=torch.float64),
        l2_0=L20,
        gradient_0=Lg0,
        scale=K,
        kind="l2",
        beta=0.0,
    )
    gradient_only, _ = normalized_data(
        torch.tensor(1.7, dtype=torch.float64),
        torch.tensor(0.03, dtype=torch.float64),
        l2_0=L20,
        gradient_0=Lg0,
        scale=K,
        kind="gradient",
        beta=0.0,
    )
    assert torch.allclose(l2_only, torch.tensor(1.7, dtype=torch.float64))
    assert torch.allclose(gradient_only, K * 0.03 / Lg0)

    neutral, _ = normalized_data(
        L20,
        Lg0,
        l2_0=L20,
        gradient_0=Lg0,
        scale=K,
        kind="mixed",
        beta=4.0,
    )
    assert torch.allclose(neutral, K)
    checks.update(
        L20=float(L20),
        Lg0=float(Lg0),
        K=float(K),
        l2_endpoint=float(l2_only),
        gradient_endpoint=float(gradient_only),
        neutral_mixed=float(neutral),
    )
    return checks


def main(cfg: Config) -> None:
    checks = check()
    payload = {
        "passed": True,
        "checks": checks,
        "source_sha256": {
            "05-verify-loss.py": digest(Path(__file__)),
            "study.py": digest(Path(__file__).with_name("study.py")),
        },
    }
    cfg.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    cherries.log_metrics(
        {
            "loss/L20": checks["L20"],
            "loss/Lg0": checks["Lg0"],
            "loss/K": checks["K"],
        }
    )


if __name__ == "__main__":
    cherries.main(main)
