"""Full-face directional derivative checks for the new signed-stress mechanics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import torch
from activation_models import STRESS_REF_MPA
from experiment import Profile
from run_support import archive, metrics, receipt, write_json
from stress_study import FIXTURE, L_REF_MM, SMOOTH_LENGTH_M, StressStudy

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("10-validation-inverse-v2")
    activation: Path = Path("05-activation-validation-inverse-v2-003/receipt.json")


def main(cfg: Config) -> None:
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    cpu = cherries.input(cfg.activation)
    cpu_data = json.loads(cpu.read_text())
    assert cpu_data.get("passed", cpu_data.get("status") == "passed")
    study = StressStudy()
    p = study.physics
    production_tolerance = dict(p.forward_tolerance)
    study.save_geometry(out / "mesh.npz")
    with p.accuracy(0.1):
        validation_tolerance = dict(p.forward_tolerance)
        q = torch.nn.Parameter(torch.zeros((len(p.ids), 6)))
        zero = np.zeros_like(p.points)
        t = time.perf_counter()
        neutral = study.evaluate(q, "symmetric6", None, zero, 1.0, 0.0, backward=True)
        assert float(np.max(np.abs(neutral["u"]))) < 1e-9
        neutral_metrics = metrics(neutral)
        neutral_metrics["wall_seconds"] = time.perf_counter() - t
        write_json(
            out / "neutral.json",
            {
                "metrics": neutral_metrics,
                "forward": neutral["forward"],
                "adjoint": neutral["adjoint"],
            },
        )
        q0 = torch.zeros_like(q)
        q0[:, :3] = 0.04
        center_coords = p.points[p.tets[p.ids]].mean(axis=1)
        z = (center_coords - center_coords.mean(0)) / np.ptp(center_coords, axis=0)
        d1 = torch.zeros_like(q)
        d1[:, 0] = 1.0
        d2 = torch.zeros_like(q)
        d2[:, 0] = torch.as_tensor(z[:, 0])
        d2[:, 1] = -torch.as_tensor(z[:, 0])
        d2[:, 3] = 0.25
        checks = []
        for beta in (0.0, 1.0):
            center = torch.nn.Parameter(q0.clone())
            result = study.evaluate(
                center, "symmetric6", None, zero, beta, 0.001, backward=True
            )
            for name, direction in (("uniform-xx", d1), ("smooth-deviatoric", d2)):
                analytic = float((result["gradient"] * direction).sum())
                assert abs(analytic) > 1e-9
                for eps in (0.002, 0.001):
                    vals = []
                    for sign in (-1.0, 1.0):
                        trial = (q0 + sign * eps * direction).detach()
                        trial_result = study.evaluate(
                            trial,
                            "symmetric6",
                            None,
                            result["u"],
                            beta,
                            0.001,
                            backward=False,
                        )
                        vals.append(trial_result["objective"])
                    numeric = (vals[1] - vals[0]) / (2 * eps)
                    row = {
                        "normal_weight": beta,
                        "direction": name,
                        "epsilon": eps,
                        "analytic": analytic,
                        "finite_difference": numeric,
                        "relative_error": abs(numeric - analytic) / abs(analytic),
                    }
                    checks.append(row)
                    write_json(
                        out / "checks.json",
                        {"passed": False, "status": "running", "checks": checks},
                    )
                    assert row["relative_error"] < 0.02, row
    np.savez_compressed(
        out / "neutral-gradient.npz", gradient=neutral["gradient"].cpu().numpy()
    )
    assert p.forward_tolerance == production_tolerance
    records = archive(out)
    protocol = {
        "materials": p.material_spec,
        "stress_reference_MPa": STRESS_REF_MPA,
        "l_ref_mm": L_REF_MM,
        "smooth_length_m": SMOOTH_LENGTH_M,
        "forward_tolerance": p.forward_tolerance,
        "validation_forward_tolerance": validation_tolerance,
        "validation_adjoint_rtol": validation_tolerance["adjoint_rtol"],
        "fixture": {
            f: receipt(FIXTURE / f) for f in ("volume.vtu", "skin.vtp", "summary.json")
        },
        "sources": records,
        "cpu_validation": receipt(cpu),
        "geometry": receipt(out / "mesh.npz"),
    }
    write_json(out / "protocol.json", protocol)
    write_json(
        out / "checks.json",
        {
            "passed": True,
            "checks": checks,
            "neutral": neutral_metrics,
            "maximum_relative_error": max(c["relative_error"] for c in checks),
            "source_protocol": receipt(out / "protocol.json"),
        },
    )
    cherries.log_metric(
        "maximum_relative_error", max(c["relative_error"] for c in checks)
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
