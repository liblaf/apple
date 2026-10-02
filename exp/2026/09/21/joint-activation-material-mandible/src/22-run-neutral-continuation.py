"""Run the remaining neutral prestress stages, stopping at any failed gate."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    initial_checkpoint: Path
    output_dir: Path
    contact_spec: Path = GROUP / "data/contact/config.json"
    outer_method: Literal["spg", "bfgs"]
    max_accepted_steps: int = 200
    max_forward_evaluations: int = 1000
    wall_budget_seconds_per_stage: float = 14400


def validate_checkpoint(path: Path, fraction: float, contact_hash: str) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["schema"] == "joint-inverse-checkpoint-v1"
    assert state["stage"] == "neutral"
    for key in (
        "preparation_complete",
        "optimizer_converged",
        "neutral_budget_met",
        "contact_validated",
        "neutral_converged",
        "inverse_converged",
    ):
        assert state[key] is True, (path, key)
    protocol = state["protocol"]
    assert protocol["schema"] == "joint-neutral-convergence-protocol-v1"
    assert protocol["skin_prestress_fraction"] == fraction
    assert abs(protocol["skin_target_n_per_m"] - 80.6 * fraction) < 1e-12
    assert protocol["contact"]["spec_sha256"] == contact_hash
    solver = protocol["forward_solver"]
    assert solver["method"] == "newton_cg"
    assert solver["newton_linear_rtol"] == 1e-3
    assert solver["newton_max_steps"] == 12
    assert solver.get("fallback") is None
    skin = state["materials"]["materials"]["skin"]
    layout = state["materials"]["parameterization"]
    assert state["shared_coefficients"].shape == (layout["shared_coefficient_count"],)
    resultant = (
        float(
            state["shared_coefficients"][layout["skin_isotropic_resultant_coordinate"]]
        )
        * skin["reference_resultant_scale_n_per_m"]
    )
    assert abs(resultant - 80.6 * fraction) < 1e-10
    return state


def shared_basis_arguments(state: dict) -> list[str]:
    """Continue the exact admitted basis; never select a new basis by default."""
    protocol = state["protocol"]
    basis_name = protocol.get("shared_basis", "constant20")
    if basis_name == "constant20":
        assert state["materials"]["schema"] == "joint-additive-stress-fields-v1"
        return ["--shared-basis", "constant20"]
    assert basis_name == "spatial80"
    assert state["materials"]["schema"] == "joint-additive-spatial-stress-fields-v1"
    basis = protocol["spatial_basis"]
    assert sha256(Path(basis["basis_path"])) == basis["basis_sha256"]
    assert sha256(Path(basis["audit_summary_path"])) == basis["audit_summary_sha256"]
    validation = protocol["spatial_validation"]
    for name in ("cpu", "full_face_directional"):
        item = validation[name]
        assert sha256(Path(item["path"])) == item["sha256"]
        assert item["receipt"]["success"] is True
        assert item["receipt"]["basis"] == basis
    smoothness = protocol["shared_field"]["spatial_smoothness"]
    assert smoothness["weight"] == 100.0
    assert smoothness["factor"] == 0.5
    return [
        "--shared-basis",
        "spatial80",
        "--spatial-basis-path",
        basis["basis_path"],
        "--spatial-audit-summary",
        basis["audit_summary_path"],
        "--spatial-cpu-validation",
        validation["cpu"]["path"],
        "--spatial-face-gradient-validation",
        validation["full_face_directional"]["path"],
        "--spatial-smoothness-weight",
        "100",
    ]


def validate_visual_receipt(directory: Path, schema: str) -> dict:
    receipt = json.loads((directory / "summary.json").read_text())
    assert receipt["schema"] == schema
    assets = receipt["assets"] if "assets" in receipt else receipt["figures"]
    assert assets
    for asset in assets:
        filename = asset["filename"] if isinstance(asset, dict) else asset
        with (directory / filename).open("rb") as stream:
            assert stream.read(8) == b"\x89PNG\r\n\x1a\n", filename
    return receipt


def main(cfg: Config) -> None:  # noqa: PLR0915 - fixed sequence with artifact gates.
    global COMPLETED  # noqa: PLW0603
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    contact_hash = sha256(cfg.contact_spec)
    initial = torch.load(cfg.initial_checkpoint, map_location="cpu", weights_only=False)
    initial_fraction = initial["protocol"]["skin_prestress_fraction"]
    assert initial_fraction in (0.1, 0.25, 0.5, 1.0)
    validate_checkpoint(cfg.initial_checkpoint, initial_fraction, contact_hash)
    basis_arguments = shared_basis_arguments(initial)
    checkpoint = cfg.initial_checkpoint.resolve()
    commands: list[dict] = []
    stages: list[dict] = []

    def run(script: str, arguments: list[str], label: str) -> None:
        command = [sys.executable, str(GROUP / "src" / script), *arguments]
        environment = dict(os.environ)
        environment["CHERRIES_NAME"] = label
        environment["CHERRIES_TAGS"] = "joint-inverse,neutral,contact,continuation"
        record = {"command": command, "name": label, "started_unix": time.time()}
        commands.append(record)
        write_json(cfg.output_dir / "commands.json", commands)
        log = cfg.output_dir / f"command-{len(commands):02d}.log"
        with log.open("x") as stream:
            result = subprocess.run(
                command,
                cwd=GROUP,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record.update({"exit_code": result.returncode, "finished_unix": time.time()})
        write_json(cfg.output_dir / "commands.json", commands)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)

    for fraction in (0.25, 0.5, 1.0):
        if fraction <= initial_fraction:
            continue
        stage = cfg.output_dir / f"neutral-{round(100 * fraction):03d}"
        parent_hash = sha256(checkpoint)
        run(
            "21-neutral-converge.py",
            [
                "--initial-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(stage.resolve()),
                "--contact-spec",
                str(cfg.contact_spec.resolve()),
                "--skin-prestress-fraction",
                str(fraction),
                "--forward-method",
                "newton_cg",
                "--outer-method",
                cfg.outer_method,
                "--max-accepted-steps",
                str(cfg.max_accepted_steps),
                "--max-forward-evaluations",
                str(cfg.max_forward_evaluations),
                "--wall-budget-seconds",
                str(cfg.wall_budget_seconds_per_stage),
                *basis_arguments,
            ],
            f"Neutral contact convergence {100 * fraction:g} percent",
        )
        summary = json.loads((stage / "summary.json").read_text())
        assert summary["success"] is True, summary
        assert summary["preparation_complete"] is True, summary
        checkpoint = stage.resolve() / "terminal.pt"
        state = validate_checkpoint(checkpoint, fraction, contact_hash)
        assert shared_basis_arguments(state) == basis_arguments
        assert state["protocol"]["initial_checkpoint_sha256"] == parent_hash
        contact_maps = stage / "contact-visuals"
        shape_maps = stage / "shape-visuals"
        convergence_plots = stage / "convergence-visuals"
        run(
            "44-render-contact.py",
            [
                "--checkpoint",
                str(checkpoint),
                "--contact-spec",
                str(cfg.contact_spec.resolve()),
                "--output-dir",
                str(contact_maps.resolve()),
            ],
            f"Converged {100 * fraction:g} percent neutral contact maps",
        )
        run(
            "45-render-neutral-state.py",
            [
                "--run-dir",
                str(stage.resolve()),
                "--checkpoint",
                str(checkpoint),
                "--contact-visuals-dir",
                str(contact_maps.resolve()),
                "--output-dir",
                str(shape_maps.resolve()),
            ],
            f"Converged {100 * fraction:g} percent neutral shape maps",
        )
        run(
            "42-render-convergence.py",
            [
                "--run-dir",
                str(stage.resolve()),
                "--output-dir",
                str(convergence_plots.resolve()),
            ],
            f"Converged {100 * fraction:g} percent neutral trajectories",
        )
        checkpoint_hash = sha256(checkpoint)
        contact_receipt = validate_visual_receipt(
            contact_maps, "joint-contact-visual-audit-v1"
        )
        assert contact_receipt["state"]["checkpoint"]["sha256"] == checkpoint_hash
        assert contact_receipt["runtime_contact"]["contact_numerically_valid"] is True
        shape_receipt = validate_visual_receipt(
            shape_maps, "joint-contact-neutral-state-visuals-v1"
        )
        assert shape_receipt["checkpoint"]["sha256"] == checkpoint_hash
        plot_receipt = validate_visual_receipt(
            convergence_plots, "joint-preparation-plots-v1"
        )
        assert plot_receipt["checkpoint_sha256"] == checkpoint_hash
        assert plot_receipt["neutral_converged"] is True
        stages.append(
            {
                "fraction": fraction,
                "run_dir": str(stage.resolve()),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "summary_sha256": sha256(stage / "summary.json"),
            }
        )
        write_json(cfg.output_dir / "stages.json", stages)
        LOG.info("Converged and rendered neutral continuation %.0f%%", 100 * fraction)
    validate_checkpoint(checkpoint, 1.0, contact_hash)
    write_json(
        cfg.output_dir / "summary.json",
        {
            "schema": "joint-neutral-continuation-sequence-v1",
            "success": True,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "stages": stages,
            "scope": "converged neutral preparation; not final joint trends",
        },
    )
    cherries.log_output(cfg.output_dir)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
