"""Run the prepared calibration, control, joint, and visualization sequence once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch

GROUP = Path(__file__).resolve().parent.parent
SRC = GROUP / "src"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), path
    return value


def resolve(path: Path) -> Path:
    return (path if path.is_absolute() else GROUP / path).resolve()


def options(**values: object) -> list[str]:
    result: list[str] = []
    for name, value in values.items():
        if value is not None:
            result.extend((f"--{name.replace('_', '-')}", str(value)))
    return result


def python_command(script: str, **values: object) -> list[str]:
    return [
        "uv",
        "run",
        "--frozen",
        "python",
        str(SRC / script),
        *options(**values),
    ]


def solver_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": args.forward_method,
        "newton_linear_rtol": args.newton_linear_rtol,
        "newton_max_steps": args.newton_max_steps,
        "fallback": None,
    }


def validate_solver(value: dict[str, Any], expected: dict[str, Any]) -> None:
    assert value["method"] == expected["method"], (value, expected)
    assert float(value["newton_linear_rtol"]) == float(
        expected["newton_linear_rtol"]
    ), (value, expected)
    assert int(value["newton_max_steps"]) == int(expected["newton_max_steps"]), (
        value,
        expected,
    )
    assert value.get("fallback") is None, value


def validate_neutral(
    path: Path,
    expected_solver: dict[str, Any],
    contact_spec_hash: str,
    contact_validation_hash: str,
) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    assert value["schema"] == "joint-inverse-checkpoint-v1"
    assert value["stage"] == "neutral"
    for key in (
        "neutral_converged",
        "optimizer_converged",
        "neutral_budget_met",
        "inverse_converged",
        "contact_validated",
        "preparation_complete",
    ):
        assert value[key] is True, key
    protocol = value["protocol"]
    assert protocol["schema"] == "joint-neutral-convergence-protocol-v1"
    assert protocol["stage"] == "neutral"
    assert float(protocol["skin_prestress_fraction"]) == 1.0
    assert float(protocol["skin_target_n_per_m"]) == 80.6
    validate_solver(protocol["forward_solver"], expected_solver)
    contact = protocol["contact"]
    assert contact["enabled"] is True
    assert contact["required_for_final"] is True
    assert contact["spec_sha256"] == contact_spec_hash
    assert contact["validation_sha256"] == contact_validation_hash
    metrics = value["metrics"]
    assert metrics["preparation_complete"] is True
    assert metrics["contact"]["contact_numerically_valid"] is True
    convergence = metrics["convergence"]
    assert convergence["stationarity_met"] is True
    assert convergence["objective_stabilized"] is True
    coordinate = int(
        value["materials"]["parameterization"]["skin_isotropic_resultant_coordinate"]
    )
    scale = float(
        value["materials"]["materials"]["skin"]["reference_resultant_scale_n_per_m"]
    )
    resultant = float(value["shared_coefficients"][coordinate]) * scale
    assert abs(resultant - 80.6) <= 1.0e-10, resultant
    validate_shared_basis(value, expected_solver, contact_spec_hash)
    return value


def validate_shared_basis(
    checkpoint: dict[str, Any],
    expected_solver: dict[str, Any],
    contact_spec_hash: str,
) -> None:
    """Admit the exact declared basis and its recorded derivative evidence."""
    protocol = checkpoint["protocol"]
    materials = checkpoint["materials"]
    count = materials["parameterization"]["shared_coefficient_count"]
    assert checkpoint["shared_coefficients"].shape == (count,)
    if materials["schema"] == "joint-additive-stress-fields-v1":
        assert count == 20
        assert protocol.get("shared_basis", "constant20") == "constant20"
        return
    assert materials["schema"] == "joint-additive-spatial-stress-fields-v1"
    assert count == 80
    assert protocol["shared_basis"] == "spatial80"
    basis = protocol["spatial_basis"]
    assert sha256(Path(basis["basis_path"])) == basis["basis_sha256"]
    assert sha256(Path(basis["audit_summary_path"])) == basis["audit_summary_sha256"]
    assert basis["input_arrays_sha256"] == protocol["input_arrays_sha256"]
    assert basis["input_manifest_sha256"] == protocol["input_manifest_sha256"]
    receipts = {}
    for name in ("cpu", "full_face_directional"):
        evidence = protocol["spatial_validation"][name]
        path = Path(evidence["path"])
        assert sha256(path) == evidence["sha256"]
        receipt = load_json(path)
        assert receipt == evidence["receipt"]
        assert receipt["success"] is True
        assert receipt["basis"] == basis
        receipts[name] = receipt
    cpu = receipts["cpu"]
    assert cpu["schema"] == "joint-spatial-field-validation-v1"
    for source, digest in cpu["sources"].items():
        assert sha256(Path(source)) == digest, source
    face = receipts["full_face_directional"]
    assert face["schema"] == "joint-spatial-face-gradient-validation-v1"
    assert face["contact_enabled"] is True
    assert face["contact_spec_sha256"] == contact_spec_hash
    assert face["input_arrays_sha256"] == protocol["input_arrays_sha256"]
    assert face["input_manifest_sha256"] == protocol["input_manifest_sha256"]
    assert face["spatial_smoothness_weight"] == 100.0
    assert face["spatial_smoothness_factor"] == 0.5
    assert len(face["checks"]) == 16
    assert face["maximum_relative_error"] < 0.02
    assert face["maximum_mechanical_relative_error"] < 0.02
    validate_solver(face["forward_solver"], expected_solver)
    for source, digest in face["implementation_sha256"].items():
        assert sha256(Path(source)) == digest, source


def validate_readiness_receipts(
    args: argparse.Namespace,
    expected_solver: dict[str, Any],
    contact_spec_hash: str,
) -> None:
    contact = load_json(args.contact_validation)
    assert contact["schema"] == "joint-contact-validation-v1"
    assert contact["success"] is True
    assert contact["contact_spec_sha256"] == contact_spec_hash
    assert contact["contact_diagnostics"]["contact_numerically_valid"] is True
    synthetic = load_json(args.newton_synthetic_validation)
    assert synthetic["schema"] == "joint-contact-validation-v1"
    assert synthetic["success"] is True
    assert synthetic["contact_spec_sha256"] == contact_spec_hash
    assert synthetic["contact_diagnostics"]["contact_numerically_valid"] is True
    validate_solver(synthetic["forward_solver"], expected_solver)
    full_face = load_json(args.newton_face_gradient_validation)
    assert full_face["success"] is True
    assert full_face["contact_enabled"] is True
    assert full_face["contact_spec_sha256"] == contact_spec_hash
    assert len(full_face["checks"]) == 16
    assert float(full_face["maximum_relative_error"]) < 0.02
    assert full_face["last_forward"]["contact"]["contact_numerically_valid"] is True
    validate_solver(full_face["forward_solver"], expected_solver)
    assert full_face["forward_tolerances"] == {
        "rtol": 1.0e-6,
        "atol": 1.0e-12,
        "adjoint_rtol": 1.0e-7,
        "max_steps": 10000,
    }
    rigid = load_json(args.rigid_bone_ccd_validation)
    assert rigid["schema"] == "joint-rigid-bone-ccd-validation-v1"
    assert rigid["success"] is True
    assert rigid["input_arrays_sha256"] == sha256(args.prepared_dir / "inputs.npz")
    assert rigid["input_manifest_sha256"] == sha256(args.prepared_dir / "manifest.json")
    assert rigid["reference"]["numerically_admissible"] is True
    assert (
        rigid["seed_adapter_checks"]["pose_and_displacement_routes_identical"] is True
    )
    assert rigid["seed_adapter_checks"]["moving_cranium_seed_rejected"] is True
    assert rigid["seed_adapter_checks"]["receipt"]["numerically_admissible"] is True
    assert rigid["whole_pose_box_validated"] is False
    for source, digest in rigid["sources"].items():
        assert sha256(Path(source)) == digest, source


class Sequence:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.logs = directory / "logs"
        self.logs.mkdir()
        self.commands: list[dict[str, Any]] = []
        self.source_hashes = {
            str(path): sha256(path)
            for directory in (
                SRC,
                GROUP.parents[4] / "src/liblaf/apple",
                GROUP.parents[4] / "exp/2026/09/07/tensor-active-stress/src",
            )
            for path in sorted(directory.rglob("*.py"))
        }
        write_json(directory / "source-hashes.json", self.source_hashes)
        write_json(directory / "commands.json", self.commands)

    def validate_sources(self) -> None:
        for source, expected in self.source_hashes.items():
            assert sha256(Path(source)) == expected, f"Source changed: {source}"

    def run(
        self,
        label: str,
        command: list[str],
        *,
        cherries_name: str,
        cherries_tags: str,
    ) -> None:
        assert command
        assert all(isinstance(value, str) for value in command)
        self.validate_sources()
        log_path = self.logs / f"{len(self.commands) + 1:02d}-{label}.log"
        record = {
            "label": label,
            "argv": command,
            "shell_display": shlex.join(command),
            "cwd": str(GROUP),
            "cherries_name": cherries_name,
            "cherries_tags": cherries_tags,
            "log": str(log_path),
            "started_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "exit_code": None,
        }
        self.commands.append(record)
        write_json(self.directory / "commands.json", self.commands)
        environment = os.environ.copy()
        environment["CHERRIES_NAME"] = cherries_name
        environment["CHERRIES_TAGS"] = cherries_tags
        started = time.perf_counter()
        with log_path.open("w") as stream:
            stream.write(f"$ {record['shell_display']}\n")
            stream.write(f"CHERRIES_NAME={cherries_name}\n")
            stream.write(f"CHERRIES_TAGS={cherries_tags}\n\n")
            stream.flush()
            process = subprocess.Popen(
                command,
                cwd=GROUP,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            exit_code = process.wait()
        record.update(
            {
                "status": "completed" if exit_code == 0 else "failed",
                "exit_code": exit_code,
                "elapsed_seconds": time.perf_counter() - started,
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        write_json(self.directory / "commands.json", self.commands)
        if exit_code != 0:
            message = f"{label} failed with exit code {exit_code}; see {log_path}"
            raise RuntimeError(message)
        self.validate_sources()


def common_joint_args(args: argparse.Namespace, output: Path) -> list[str]:
    return python_command(
        "30-joint-pilot.py",
        output_dir=output,
        prepared_dir=args.prepared_dir,
        neutral_checkpoint=args.neutral_checkpoint,
        contact_spec=args.contact_spec,
        contact_validation=args.contact_validation,
        newton_synthetic_validation=args.newton_synthetic_validation,
        newton_face_gradient_validation=args.newton_face_gradient_validation,
        rigid_bone_ccd_validation=args.rigid_bone_ccd_validation,
        activation_neighbor_rms_budget_dimensionless=(
            args.activation_neighbor_rms_budget_dimensionless
        ),
        preparation_max_updates=args.preparation_max_updates,
        preparation_wall_budget_seconds=args.preparation_wall_budget_seconds,
        final_updates=args.final_updates,
        final_wall_budget_seconds=args.final_wall_budget_seconds,
        forward_method=args.forward_method,
        newton_linear_rtol=args.newton_linear_rtol,
        newton_max_steps=args.newton_max_steps,
    )


def validate_jaw_preflight(  # noqa: PLR0915
    path: Path,
    args: argparse.Namespace,
    neutral: dict[str, Any],
    expected_solver: dict[str, Any],
) -> dict[str, Any]:
    value = load_json(path)
    assert value["schema"] == "joint-jaw-preflight-v2"
    assert value["success"] is True
    assert value["admission_mode"] == "final_launch_ready"
    assert value["diagnostic_only"] is False
    assert value["final_launch_ready"] is True
    for key, source in (
        ("neutral_checkpoint_sha256", args.neutral_checkpoint),
        ("input_arrays_sha256", args.prepared_dir / "inputs.npz"),
        ("input_manifest_sha256", args.prepared_dir / "manifest.json"),
        ("contact_spec_sha256", args.contact_spec),
        ("contact_validation_sha256", args.contact_validation),
        ("rigid_bone_ccd_validation_sha256", args.rigid_bone_ccd_validation),
    ):
        assert value[key] == sha256(source), key
    validate_solver(value["forward_solver"], expected_solver)
    for key in ("shared_basis", "shared_field", "spatial_basis"):
        assert value[key] == neutral["protocol"][key], key
    assert value["probe_count"] == 15
    assert len(value["rows"]) == 15
    assert value["broad_axis_endpoint_count"] == 12
    assert value["geometry_reproducibility"]["budget_m"] == 1e-6
    assert value["geometry_reproducibility"]["met"] is True
    expected_tolerances = {
        "rtol": 1e-6,
        "atol": 1e-12,
        "adjoint_rtol": 1e-7,
        "max_steps": 10000,
    }
    assert value["forward_tolerances"] == expected_tolerances
    expected_roles = {
        "zero": "required_zero_geometry_reproducibility",
        **{
            f"coordinate_{coordinate}_{sign:+d}": "diagnostic_broad_axis_endpoint"
            for coordinate in range(6)
            for sign in (-1, 1)
        },
        "legacy_one_degree_center": "diagnostic_legacy_one_degree_center",
        "world_x_positive_0p01_degree": "required_nonzero_world_x_0p01_degree",
    }
    assert [row["label"] for row in value["rows"]] == list(expected_roles)
    assert value["required_labels"] == ["zero", "world_x_positive_0p01_degree"]
    for row in value["rows"]:
        assert row["role"] == expected_roles[row["label"]]
        assert row["seed_sha256"] == value["seed_sha256"]
        assert row["seed_matches_neutral"] is True
        if row["label"] not in value["required_labels"]:
            continue
        assert row["status"] == "accepted"
        assert row["numerically_admissible"] is True
        assert row["forward_attempted"] is True
        assert row["rigid_bone_ccd"]["numerically_admissible"] is True
        assert row["contact"]["contact_numerically_valid"] is True
        assert all(flag is True for flag in row["shape_checks"].values())
        assert row["geometry"]["schema"] == "joint-final-run-geometry-v1"
        assert row["geometry"]["numerical_geometry_admissible"] is True
        forward = row["forward"]
        assert forward["success"] is True
        assert math.isfinite(float(forward["grad_norm"]))
        assert forward["tolerances"] == expected_tolerances
        validate_solver(forward["forward_solver"], expected_solver)
        implementation = (
            "newton_cg"
            if forward["result"] == "initial_equilibrium"
            else "inexact_newton_cg"
        )
        assert forward["method"] == implementation
        state = row["state"]
        assert state["schema"] == "joint-jaw-preflight-state-v1"
        assert sha256(Path(state["path"])) == state["sha256"]
        assert state["displacement_shape"] == list(neutral["primal"]["neutral"].shape)
        assert state["displacement_dtype"] == "<f8"
        if row["label"] == "zero":
            metrics = row["geometry_reproducibility"]
            assert metrics == value["geometry_reproducibility"]["metrics"]
            assert metrics["reference"] == "frozen_neutral_seed_displacement"
            assert metrics["candidate"] == "resolved_zero_pose_displacement"
            assert metrics["coordinate_frame"] == "world_m"
            assert metrics["maximum_euclidean_nodal_difference_budget_m"] == 1e-6
            assert metrics["passed"] is True
            for name in (
                "maximum_euclidean_nodal_displacement_difference_m",
                "surface_node_rms_euclidean_displacement_difference_m",
                "volume_node_rms_euclidean_displacement_difference_m",
            ):
                assert 0 <= metrics[name] <= 1e-6, name
            assert metrics["volume_node_count"] == neutral["primal"]["neutral"].shape[0]
            assert 0 < metrics["surface_node_count"] <= metrics["volume_node_count"]
            assert row["pose_rad_m"] == [0.0] * 6
        else:
            assert row["normalized_pose"] == [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]
            assert math.isclose(row["pose_rad_m"][0], math.radians(0.01), rel_tol=1e-14)
            assert row["pose_rad_m"][1:] == [0.0] * 5
    for key in (
        "zero_geometry_reproducibility_met",
        "required_nonzero_world_x_0p01_degree_met",
    ):
        assert value[key] is True, key
    for key in (
        "whole_pose_box_validated",
        "anatomical_validation",
        "bone_bone_energy_added",
        "rotation_arc_checked",
    ):
        assert value[key] is False, key
    expected_sources = {
        (SRC / name).resolve()
        for name in (
            "28-run-jaw-preflight.py",
            "30-joint-pilot.py",
            "joint_final_geometry.py",
            "joint_rigid_bone_collision.py",
        )
    }
    assert {Path(source).resolve() for source in value["sources"]} == expected_sources
    for source, digest in value["sources"].items():
        assert sha256(Path(source)) == digest, source
    return value


def validate_calibration(
    path: Path,
    neutral_hash: str,
    expected_solver: dict[str, Any],
    contact_spec_hash: str,
    contact_validation_hash: str,
) -> dict[str, Any]:
    value = load_json(path)
    assert value["schema"] == "strong-smoothness-calibration-v1"
    assert value["stage"] == "calibrate"
    assert value["success"] is True
    assert value["neutral_checkpoint_sha256"] == neutral_hash
    assert value["contact_spec_sha256"] == contact_spec_hash
    assert value["contact_validation_sha256"] == contact_validation_hash
    validate_solver(value["forward_solver"], expected_solver)
    return value


def validate_control(
    run_dir: Path,
    neutral_hash: str,
    calibration_hash: str,
    expected_solver: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = load_json(run_dir / "summary.json")
    checkpoint_path = run_dir / "terminal.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert summary["schema"] == "joint-final-summary-v1"
    assert summary["stage"] == "control_converge"
    assert summary["success"] is True
    assert summary["preparation_complete"] is True
    assert summary["inverse_converged"] is True
    assert summary["terminal_checkpoint_sha256"] == sha256(checkpoint_path)
    validate_solver(summary["forward_solver"], expected_solver)
    assert checkpoint["schema"] == "joint-inverse-checkpoint-v1"
    assert checkpoint["stage"] == "control_converge"
    assert checkpoint["preparation_complete"] is True
    assert checkpoint["inverse_converged"] is True
    assert checkpoint["neutral_checkpoint_sha256"] == neutral_hash
    assert checkpoint["calibration_sha256"] == calibration_hash
    assert checkpoint["comparison_fingerprint"] == summary["comparison_fingerprint"]
    validate_solver(checkpoint["forward_solver"], expected_solver)
    lineage = checkpoint["lineage"]
    assert lineage[0]["role"] == "converged_neutral"
    assert lineage[0]["checkpoint_sha256"] == neutral_hash
    return summary, checkpoint


def validate_joint(
    run_dir: Path,
    neutral_hash: str,
    calibration_hash: str,
    control_checkpoint_hash: str,
    control_receipt_hash: str,
    expected_solver: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = load_json(run_dir / "summary.json")
    checkpoint_path = run_dir / "terminal.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert summary["schema"] == "joint-final-summary-v1"
    assert summary["stage"] == "joint_trend"
    assert summary["success"] is True
    assert int(summary["accepted_updates"]) >= 1
    assert summary["minimum_joint_updates_met"] is True
    assert summary["parent_control_checkpoint_sha256"] == control_checkpoint_hash
    assert summary["control_receipt_sha256"] == control_receipt_hash
    assert summary["terminal_checkpoint_sha256"] == sha256(checkpoint_path)
    validate_solver(summary["forward_solver"], expected_solver)
    assert checkpoint["schema"] == "joint-inverse-checkpoint-v1"
    assert checkpoint["stage"] == "joint_trend"
    assert checkpoint["neutral_checkpoint_sha256"] == neutral_hash
    assert checkpoint["calibration_sha256"] == calibration_hash
    assert checkpoint["parent_control_checkpoint_sha256"] == control_checkpoint_hash
    validate_solver(checkpoint["forward_solver"], expected_solver)
    assert any(
        item["role"] == "converged_control_parent"
        and item["checkpoint_sha256"] == control_checkpoint_hash
        for item in checkpoint["lineage"]
    )
    return summary, checkpoint


def render_stage(
    sequence: Sequence,
    args: argparse.Namespace,
    stage: Literal["control", "joint"],
    run_dir: Path,
) -> tuple[Path, Path]:
    spatial = sequence.directory / f"{stage}-spatial"
    visuals = sequence.directory / f"{stage}-visuals"
    sequence.run(
        f"{stage}-render-31",
        python_command(
            "31-render-final.py",
            run_dir=run_dir,
            prepared_dir=args.prepared_dir,
            output_dir=spatial,
        ),
        cherries_name=f"Joint sequence {stage} spatial fields {sequence.directory.name}",
        cherries_tags=f"joint-inverse,sequence,{stage},render31,visualization",
    )
    spatial_summary = load_json(spatial / "summary.json")
    assert spatial_summary["schema"] == "joint-final-visualization-summary-v1"
    assert spatial_summary["success"] is True
    assert Path(spatial_summary["run_dir"]).resolve() == run_dir.resolve()
    sequence.run(
        f"{stage}-render-46",
        python_command(
            "46-render-optimization-state.py",
            run_dir=run_dir,
            spatial_dir=spatial,
            prepared_dir=args.prepared_dir,
            contact_spec=args.contact_spec,
            output_dir=visuals,
        ),
        cherries_name=f"Joint sequence {stage} accepted states {sequence.directory.name}",
        cherries_tags=(
            f"joint-inverse,sequence,{stage},render46,contact,visualization"
        ),
    )
    visual_summary = load_json(visuals / "summary.json")
    assert visual_summary["schema"] == "joint-optimization-spatial-visuals-v1"
    assert Path(visual_summary["run"]["directory"]).resolve() == run_dir.resolve()
    assert visual_summary["run"]["stage"] == (
        "control_converge" if stage == "control" else "joint_trend"
    )
    assert visual_summary["assets"]
    return spatial, visuals


def build_review(
    sequence: Sequence,
    args: argparse.Namespace,
    *,
    control_dir: Path,
    joint_dir: Path | None,
    optimization_dirs: list[Path],
) -> None:
    command = python_command(
        "43-build-review.py",
        output_dir=args.review_site_dir,
        control_run_dir=control_dir,
        neutral_run_dirs=json.dumps([str(path) for path in args.neutral_run_dirs]),
        neutral_visual_dir=args.neutral_visual_dir,
        neutral_contact_visual_dir=args.neutral_contact_visual_dir,
        neutral_lineage_visual_dir=args.neutral_lineage_visual_dir,
        jaw_visual_dir=sequence.directory / "jaw-preflight-visuals",
        contact_validation_path=args.contact_validation,
        newton_face_validation_path=args.newton_face_gradient_validation,
        optimization_plot_dirs=json.dumps([str(path) for path in optimization_dirs]),
    )
    if args.convergence_plot_dirs:
        command.extend(
            options(
                convergence_plot_dirs=json.dumps(
                    [str(path) for path in args.convergence_plot_dirs]
                )
            )
        )
    if joint_dir is not None:
        command.extend(options(joint_run_dir=joint_dir))
    label = "joint" if joint_dir is not None else "control"
    sequence.run(
        f"review-{label}",
        command,
        cherries_name=f"Joint sequence {label} review site {sequence.directory.name}",
        cherries_tags=f"joint-inverse,sequence,{label},review-site,visualization",
    )
    status = load_json(args.review_site_dir / "status.json")
    assert status["schema"] == "joint-mobile-review-v1"
    assert status["control"]["terminal_checkpoint_sha256"] == sha256(
        control_dir / "terminal.pt"
    )
    if joint_dir is not None:
        assert status["joint"]["terminal_checkpoint_sha256"] == sha256(
            joint_dir / "terminal.pt"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one foreground prepared joint sequence; no retries or scheduler."
    )
    parser.add_argument("--neutral-checkpoint", required=True, type=Path)
    parser.add_argument("--sequence-dir", required=True, type=Path)
    parser.add_argument("--prepared-dir", type=Path, default=GROUP / "data/prepared")
    parser.add_argument(
        "--contact-spec", type=Path, default=GROUP / "data/contact/config.json"
    )
    parser.add_argument(
        "--contact-validation",
        type=Path,
        default=GROUP / "data/contact-validation/summary.json",
    )
    parser.add_argument(
        "--newton-synthetic-validation",
        type=Path,
        default=GROUP / "data/contact-validation-newton/summary.json",
    )
    parser.add_argument(
        "--newton-face-gradient-validation",
        type=Path,
        default=GROUP / "data/face-gradient-validation-contact-newton/summary.json",
    )
    parser.add_argument(
        "--rigid-bone-ccd-validation",
        type=Path,
        default=GROUP / "data/rigid-bone-ccd-validation-003/summary.json",
    )
    parser.add_argument(
        "--review-site-dir", type=Path, default=GROUP / "data/review-site"
    )
    parser.add_argument("--neutral-run-dir", action="append", type=Path, default=[])
    parser.add_argument("--neutral-visual-dir", type=Path)
    parser.add_argument("--neutral-contact-visual-dir", type=Path)
    parser.add_argument("--neutral-lineage-visual-dir", required=True, type=Path)
    parser.add_argument(
        "--convergence-plot-dir", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--forward-method", choices=("pncg", "newton_cg"), default="newton_cg"
    )
    parser.add_argument("--newton-linear-rtol", type=float, default=1.0e-3)
    parser.add_argument("--newton-max-steps", type=int, default=12)
    parser.add_argument(
        "--activation-neighbor-rms-budget-dimensionless", type=float, default=0.05
    )
    parser.add_argument("--preparation-max-updates", type=int, default=1000)
    parser.add_argument("--preparation-wall-budget-seconds", type=float, default=86400)
    parser.add_argument("--final-updates", type=int, default=100)
    parser.add_argument("--final-wall-budget-seconds", type=float, default=43200)
    args = parser.parse_args()
    for name in (
        "neutral_checkpoint",
        "sequence_dir",
        "prepared_dir",
        "contact_spec",
        "contact_validation",
        "newton_synthetic_validation",
        "newton_face_gradient_validation",
        "rigid_bone_ccd_validation",
        "review_site_dir",
        "neutral_lineage_visual_dir",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    args.neutral_run_dirs = [resolve(path) for path in args.neutral_run_dir]
    if not args.neutral_run_dirs:
        args.neutral_run_dirs = [args.neutral_checkpoint.parent]
    args.convergence_plot_dirs = [resolve(path) for path in args.convergence_plot_dir]
    args.neutral_visual_dir = (
        resolve(args.neutral_visual_dir)
        if args.neutral_visual_dir is not None
        else args.neutral_checkpoint.parent / "shape-visuals"
    )
    args.neutral_contact_visual_dir = (
        resolve(args.neutral_contact_visual_dir)
        if args.neutral_contact_visual_dir is not None
        else args.neutral_checkpoint.parent / "contact-visuals"
    )
    return args


def main() -> None:  # noqa: PLR0915 - linear fail-fast execution protocol.
    args = parse_args()
    sequence = Sequence(args.sequence_dir)
    expected_solver = solver_contract(args)
    try:
        assert args.forward_method == "newton_cg"
        assert args.newton_linear_rtol == 1.0e-3
        assert args.newton_max_steps == 12
        assert args.activation_neighbor_rms_budget_dimensionless == 0.05
        assert args.final_updates == 100
        assert args.final_wall_budget_seconds == 43200
        for path in (
            args.neutral_checkpoint,
            args.prepared_dir / "inputs.npz",
            args.prepared_dir / "manifest.json",
            args.contact_spec,
            args.contact_validation,
            args.newton_synthetic_validation,
            args.newton_face_gradient_validation,
            args.rigid_bone_ccd_validation,
        ):
            assert path.is_file(), path
        contact_spec_hash = sha256(args.contact_spec)
        assert load_json(args.contact_spec)["surface_selection"] == (
            "pure-soft-vs-complete-source-bones"
        ), (
            "The user requires complete skull colliders; partial FEM collider launches are superseded."
        )
        contact_validation_hash = sha256(args.contact_validation)
        validate_readiness_receipts(args, expected_solver, contact_spec_hash)
        neutral = validate_neutral(
            args.neutral_checkpoint,
            expected_solver,
            contact_spec_hash,
            contact_validation_hash,
        )
        neutral_hash = sha256(args.neutral_checkpoint)
        neutral_lineage = load_json(args.neutral_lineage_visual_dir / "summary.json")
        assert neutral_lineage["schema"] == "joint-neutral-lineage-visualization-v1"
        assert neutral_lineage["lineage_proven"] is True
        assert neutral_lineage["branch_tip"]["sha256"] == neutral_hash
        neutral_visuals = load_json(args.neutral_visual_dir / "summary.json")
        assert neutral_visuals["schema"] == "joint-contact-neutral-state-visuals-v1"
        assert neutral_visuals["checkpoint"]["sha256"] == neutral_hash
        neutral_contact_visuals = load_json(
            args.neutral_contact_visual_dir / "summary.json"
        )
        assert neutral_contact_visuals["schema"] == "joint-contact-visual-audit-v1"
        assert neutral_contact_visuals["state"]["checkpoint"]["sha256"] == neutral_hash
        assert (
            neutral_contact_visuals["runtime_contact"]["contact_numerically_valid"]
            is True
        )
        manifest = {
            "schema": "joint-prepared-sequence-v1",
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "neutral_checkpoint": str(args.neutral_checkpoint),
            "neutral_checkpoint_sha256": neutral_hash,
            "neutral_visual_dir": str(args.neutral_visual_dir),
            "neutral_contact_visual_dir": str(args.neutral_contact_visual_dir),
            "neutral_lineage_visual_dir": str(args.neutral_lineage_visual_dir),
            "frozen_sources_sha256": sha256(sequence.directory / "source-hashes.json"),
            "neutral_protocol": neutral["protocol"],
            "forward_solver": expected_solver,
            "activation_neighbor_rms_budget_dimensionless": (
                args.activation_neighbor_rms_budget_dimensionless
            ),
            "final_budget": {
                "accepted_updates": args.final_updates,
                "wall_seconds": args.final_wall_budget_seconds,
            },
            "source_sha256": {
                name: sha256(SRC / name)
                for name in (
                    "30-joint-pilot.py",
                    "31-render-final.py",
                    "43-build-review.py",
                    "46-render-optimization-state.py",
                    "50-run-prepared-joint-sequence.py",
                )
            },
        }
        write_json(sequence.directory / "manifest.json", manifest)

        jaw_preflight_dir = sequence.directory / "jaw-preflight"
        sequence.run(
            "jaw-preflight",
            python_command(
                "28-run-jaw-preflight.py",
                prepared_dir=args.prepared_dir,
                neutral_checkpoint=args.neutral_checkpoint,
                contact_spec=args.contact_spec,
                contact_validation=args.contact_validation,
                newton_synthetic_validation=args.newton_synthetic_validation,
                rigid_bone_ccd_validation=args.rigid_bone_ccd_validation,
                output_dir=jaw_preflight_dir,
                admission_mode="final_launch_ready",
                forward_method=args.forward_method,
                newton_linear_rtol=args.newton_linear_rtol,
                newton_max_steps=args.newton_max_steps,
            ),
            cherries_name=f"Joint sequence jaw preflight {sequence.directory.name}",
            cherries_tags="joint-inverse,sequence,jaw,preflight,contact,newton-cg",
        )
        jaw_preflight_path = jaw_preflight_dir / "summary.json"
        jaw_preflight = validate_jaw_preflight(
            jaw_preflight_path, args, neutral, expected_solver
        )
        manifest["jaw_preflight"] = {
            "path": str(jaw_preflight_path),
            "sha256": sha256(jaw_preflight_path),
            "status": jaw_preflight["status"],
        }
        write_json(sequence.directory / "manifest.json", manifest)

        jaw_visual_dir = sequence.directory / "jaw-preflight-visuals"
        sequence.run(
            "jaw-preflight-render",
            python_command(
                "49-render-jaw-domain.py",
                validation=args.rigid_bone_ccd_validation,
                jaw_preflight=jaw_preflight_path,
                prepared_dir=args.prepared_dir,
                output_dir=jaw_visual_dir,
            ),
            cherries_name=f"Joint sequence jaw preflight views {sequence.directory.name}",
            cherries_tags="joint-inverse,sequence,jaw,contact,visualization",
        )
        jaw_visuals = load_json(jaw_visual_dir / "summary.json")
        assert jaw_visuals["schema"] == "joint-jaw-domain-visuals-v1"
        assert jaw_visuals["preflight"]["sha256"] == sha256(jaw_preflight_path)
        assert jaw_visuals["preflight"]["final_launch_ready"] is True
        assert jaw_visuals["assets"]

        calibration_dir = sequence.directory / "calibration"
        command = common_joint_args(args, calibration_dir)
        command.extend(options(stage="calibrate"))
        sequence.run(
            "calibration",
            command,
            cherries_name=f"Joint sequence calibration {sequence.directory.name}",
            cherries_tags="joint-inverse,sequence,calibration,strong-smoothness,newton-cg",
        )
        calibration_path = calibration_dir / "calibration.json"
        validate_calibration(
            calibration_path,
            neutral_hash,
            expected_solver,
            contact_spec_hash,
            contact_validation_hash,
        )
        calibration_hash = sha256(calibration_path)

        control_dir = sequence.directory / "control"
        command = common_joint_args(args, control_dir)
        command.extend(options(stage="control_converge", calibration=calibration_path))
        sequence.run(
            "control-converge",
            command,
            cherries_name=f"Joint sequence converged control {sequence.directory.name}",
            cherries_tags="joint-inverse,sequence,control,convergence,newton-cg",
        )
        control_summary, _ = validate_control(
            control_dir, neutral_hash, calibration_hash, expected_solver
        )
        control_checkpoint = control_dir / "terminal.pt"
        control_receipt = control_dir / "summary.json"
        control_checkpoint_hash = sha256(control_checkpoint)
        control_receipt_hash = sha256(control_receipt)
        control_spatial, control_visuals = render_stage(
            sequence, args, "control", control_dir
        )
        optimization_dirs = [control_spatial, control_visuals]
        build_review(
            sequence,
            args,
            control_dir=control_dir,
            joint_dir=None,
            optimization_dirs=optimization_dirs,
        )

        joint_dir = sequence.directory / "joint"
        command = common_joint_args(args, joint_dir)
        command.extend(
            options(
                stage="joint_trend",
                calibration=calibration_path,
                initial_checkpoint=control_checkpoint,
                control_receipt=control_receipt,
            )
        )
        sequence.run(
            "joint-trend",
            command,
            cherries_name=f"Joint sequence final trend {sequence.directory.name}",
            cherries_tags="joint-inverse,sequence,joint,final-trend,newton-cg",
        )
        joint_summary, _ = validate_joint(
            joint_dir,
            neutral_hash,
            calibration_hash,
            control_checkpoint_hash,
            control_receipt_hash,
            expected_solver,
        )
        joint_spatial, joint_visuals = render_stage(sequence, args, "joint", joint_dir)
        optimization_dirs.extend((joint_spatial, joint_visuals))
        build_review(
            sequence,
            args,
            control_dir=control_dir,
            joint_dir=joint_dir,
            optimization_dirs=optimization_dirs,
        )

        manifest.update(
            {
                "status": "completed",
                "success": True,
                "finished_at": datetime.now(UTC).isoformat(),
                "calibration_sha256": calibration_hash,
                "control_checkpoint_sha256": control_checkpoint_hash,
                "control_receipt_sha256": control_receipt_hash,
                "joint_checkpoint_sha256": sha256(joint_dir / "terminal.pt"),
                "joint_receipt_sha256": sha256(joint_dir / "summary.json"),
                "control": control_summary,
                "joint": joint_summary,
                "review_site": str(args.review_site_dir),
                "review_url": str(args.review_site_dir),
            }
        )
        write_json(sequence.directory / "summary.json", manifest)
        write_json(sequence.directory / "manifest.json", manifest)
    except Exception as error:
        failure = {
            "schema": "joint-prepared-sequence-failure-v1",
            "success": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "commands_completed": len(sequence.commands),
            "commands": sequence.commands,
            "failed_at": datetime.now(UTC).isoformat(),
        }
        write_json(sequence.directory / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
