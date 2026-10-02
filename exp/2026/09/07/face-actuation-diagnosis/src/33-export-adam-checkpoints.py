"""Export explicit saved Adam checkpoints to verified VTU history frames."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

STEP_PATTERN = re.compile(r"step-(\d+)\.npz")


def digest(path: Path) -> dict[str, str | int]:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def receipt_map(path: Path) -> dict[int, dict[str, Any]]:
    receipts = [json.loads(line) for line in path.read_text().splitlines() if line]
    by_step = {int(receipt["step"]): receipt for receipt in receipts}
    if len(by_step) != len(receipts):
        raise ValueError("solver receipt steps are not unique")
    return by_step


def checkpoint_paths(run_dir: Path, requested: list[int] | None) -> list[Path]:
    discovered: dict[int, Path] = {}
    for path in run_dir.glob("step-*.npz"):
        match = STEP_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step in discovered:
            raise ValueError(f"duplicate checkpoint step {step}")
        discovered[step] = path
    steps = sorted(discovered) if requested is None else requested
    if len(set(steps)) != len(steps) or steps != sorted(steps):
        raise ValueError("requested steps must be unique and increasing")
    missing = [step for step in steps if step not in discovered]
    if missing:
        raise FileNotFoundError(f"missing explicit checkpoint steps: {missing}")
    if not steps:
        raise ValueError("no explicit checkpoint files selected")
    return [discovered[step] for step in steps]


def full_activation(
    q: np.ndarray, active: np.ndarray, n_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    if q.shape != (int(active.sum()), 6):
        raise ValueError(f"checkpoint q shape {q.shape} does not match active mask")
    packed = np.zeros((n_cells, 6), dtype=q.dtype)
    packed[active] = q
    matrix = np.broadcast_to(np.eye(3, dtype=q.dtype), (n_cells, 3, 3)).copy()
    matrix[active, 0, 0] += q[:, 0]
    matrix[active, 1, 1] += q[:, 1]
    matrix[active, 2, 2] += q[:, 2]
    matrix[active, 0, 1] += q[:, 3]
    matrix[active, 1, 0] += q[:, 3]
    matrix[active, 1, 2] += q[:, 4]
    matrix[active, 2, 1] += q[:, 4]
    matrix[active, 0, 2] += q[:, 5]
    matrix[active, 2, 0] += q[:, 5]
    return packed, matrix


def verify_export(
    path: Path,
    reference: pv.UnstructuredGrid,
    expected_points: np.ndarray,
    u: np.ndarray,
    packed: np.ndarray,
    matrix: np.ndarray,
    *,
    require_packed_q_array: bool = True,
) -> dict[str, Any]:
    saved = pv.read(path)
    if not isinstance(saved, pv.UnstructuredGrid):
        raise TypeError(f"exported frame is not an unstructured grid: {path}")
    if not np.array_equal(saved.cells, reference.cells) or not np.array_equal(
        saved.celltypes, reference.celltypes
    ):
        raise ValueError(f"exported topology changed: {path}")
    checks = {
        "points_max_abs_error": float(np.max(np.abs(saved.points - expected_points))),
        "rest_position_max_abs_error": float(
            np.max(np.abs(saved.point_data["RestPosition"] - reference.points))
        ),
        "displacement_max_abs_error": float(
            np.max(np.abs(saved.point_data["Displacement"] - u))
        ),
        "packed_q_field_max_abs_error": (
            float(np.max(np.abs(saved.cell_data["ActivationOffsetPacked6"] - packed)))
            if "ActivationOffsetPacked6" in saved.cell_data
            else None
        ),
        "packed_q_reconstructed_from_final_npz": True,
        "activation_matrix_max_abs_error": float(
            np.max(
                np.abs(
                    saved.cell_data["ActivationInverseMatrix"].reshape(-1, 3, 3)
                    - matrix
                )
            )
        ),
        "activation_mask_exact": bool(
            np.array_equal(
                np.asarray(saved.cell_data["ActivationMask"], dtype=bool),
                np.asarray(reference.cell_data["ActivationMask"], dtype=bool),
            )
        ),
        "topology_exact": True,
    }
    errors = [
        checks["points_max_abs_error"],
        checks["rest_position_max_abs_error"],
        checks["displacement_max_abs_error"],
        checks["activation_matrix_max_abs_error"],
    ]
    if require_packed_q_array:
        errors.append(checks["packed_q_field_max_abs_error"])
    if not checks["activation_mask_exact"] or any(error != 0.0 for error in errors):
        raise ValueError(f"exported frame failed exact round-trip checks: {checks}")
    return checks


def final_state_verification(
    run_dir: Path,
    summary_path: Path,
    receipts: dict[int, dict[str, Any]],
    reference: pv.UnstructuredGrid,
    active: np.ndarray,
) -> dict[str, Any]:
    """Prove the claimed final VTU is the receipt-valid best state in final.npz."""
    summary = json.loads(summary_path.read_text())
    convergence = summary.get("convergence")
    if not isinstance(convergence, dict) or not isinstance(
        convergence.get("best_valid_step"), int
    ):
        raise TypeError("completed summary lacks integer convergence.best_valid_step")
    final_npz, final_vtu = run_dir / "final.npz", run_dir / "final.vtu"
    if not final_npz.is_file() or not final_vtu.is_file():
        raise FileNotFoundError("completed run requires final.npz and final.vtu")
    with np.load(final_npz) as saved:
        required = {
            "q",
            "u",
            "Ainv",
            "step",
            "forward_success",
            "adjoint_success",
            "solver_valid",
        }
        if missing := required.difference(saved.files):
            raise KeyError(f"final.npz lacks {sorted(missing)}")
        step = int(saved["step"])
        if step != convergence["best_valid_step"]:
            raise ValueError("final.npz step differs from summary best_valid_step")
        if not (
            bool(saved["forward_success"])
            and bool(saved["adjoint_success"])
            and bool(saved["solver_valid"])
        ):
            raise ValueError("final.npz is not marked solver-valid")
        receipt = receipts.get(step)
        if receipt is None or not (
            receipt["forward"]["success"] and receipt["adjoint"]["success"]
        ):
            raise ValueError("final.npz best step lacks a solver-valid receipt")
        q, u, ainv = (np.asarray(saved[name]) for name in ("q", "u", "Ainv"))
    packed, matrix = full_activation(q, active, reference.n_cells)
    if not np.array_equal(ainv, matrix[active]):
        raise ValueError("final.npz q and Ainv disagree")
    if u.shape != reference.points.shape:
        raise ValueError("final.npz displacement shape differs from fixture")
    verification = verify_export(
        final_vtu,
        reference,
        np.asarray(reference.points) + u,
        u,
        packed,
        matrix,
        require_packed_q_array=False,
    )
    return {
        "status": "verified_completed_best_state",
        "summary_best_valid_step": step,
        "final_npz": digest(final_npz),
        "final_vtu": digest(final_vtu),
        "solver_receipt": receipt,
        "verification": verification,
        "packed_q_contract": "packed q is reconstructed exactly from final.npz q and the fixture ActivationMask; final VTU ActivationInverseMatrix is checked against that reconstruction",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    config_path = run_dir / "config.json"
    receipts_path = run_dir / "solver-receipts.jsonl"
    summary_path = run_dir / "summary.json"
    if not config_path.is_file() or not receipts_path.is_file():
        raise FileNotFoundError("run config and solver receipts are required")
    if not summary_path.is_file() and not args.allow_incomplete:
        raise FileNotFoundError(
            "completed-run summary is absent; use --allow-incomplete only for a pilot"
        )
    config = json.loads(config_path.read_text())
    fixture = Path(config["fixture"])
    if not fixture.is_absolute():
        fixture = run_dir / fixture
    volume_path = fixture.resolve() / "volume.vtu"
    reference = pv.read(volume_path)
    if not isinstance(reference, pv.UnstructuredGrid):
        raise TypeError("run fixture is not an unstructured grid")
    active = np.asarray(reference.cell_data["ActivationMask"], dtype=bool)
    receipts = receipt_map(receipts_path)
    paths = checkpoint_paths(run_dir, args.steps)
    completed_verification = (
        final_state_verification(run_dir, summary_path, receipts, reference, active)
        if summary_path.is_file()
        else None
    )
    frames: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []

    for checkpoint in paths:
        match = STEP_PATTERN.fullmatch(checkpoint.name)
        assert match is not None
        filename_step = int(match.group(1))
        with np.load(checkpoint) as saved:
            required = {
                "q",
                "u",
                "Ainv",
                "step",
                "forward_success",
                "adjoint_success",
                "solver_valid",
            }
            if missing := required.difference(saved.files):
                raise KeyError(f"checkpoint {checkpoint} lacks {sorted(missing)}")
            step = int(saved["step"])
            if step != filename_step:
                raise ValueError(
                    f"checkpoint filename/payload step mismatch: {checkpoint}"
                )
            if step not in receipts:
                raise KeyError(f"missing solver receipt for checkpoint step {step}")
            forward_success = bool(saved["forward_success"])
            adjoint_success = bool(saved["adjoint_success"])
            solver_valid = bool(saved["solver_valid"])
            receipt = receipts[step]
            receipt_valid = bool(
                receipt["forward"]["success"] and receipt["adjoint"]["success"]
            )
            if solver_valid != (forward_success and adjoint_success) or (
                solver_valid != receipt_valid
            ):
                raise ValueError(f"solver status mismatch at checkpoint step {step}")
            if not solver_valid:
                omissions.append(
                    {
                        "step": step,
                        "checkpoint": digest(checkpoint),
                        "reason": "solver_valid is false; frame omitted",
                        "receipt": receipt,
                    }
                )
                continue
            q = np.asarray(saved["q"])
            u = np.asarray(saved["u"])
            ainv = np.asarray(saved["Ainv"])
        if u.shape != reference.points.shape:
            raise ValueError(f"checkpoint displacement shape changed at step {step}")
        packed, matrix = full_activation(q, active, reference.n_cells)
        if not np.array_equal(ainv, matrix[active]):
            raise ValueError(f"saved q and Ainv disagree at step {step}")
        expected_points = np.asarray(reference.points) + u
        frame = reference.copy(deep=True)
        frame.points = expected_points
        frame.point_data["RestPosition"] = np.asarray(reference.points)
        frame.point_data["Displacement"] = u
        frame.cell_data["ActivationOffsetPacked6"] = packed
        frame.cell_data["ActivationInverseMatrix"] = matrix.reshape(-1, 9)
        frame.cell_data["DetAinv"] = np.linalg.det(matrix)
        frame.field_data["OptimizationStep"] = np.asarray([step], dtype=np.int64)
        frame.field_data["ForwardSuccess"] = np.asarray(
            [forward_success], dtype=np.uint8
        )
        frame.field_data["AdjointSuccess"] = np.asarray(
            [adjoint_success], dtype=np.uint8
        )
        frame.field_data["SolverValid"] = np.asarray([solver_valid], dtype=np.uint8)
        frame_path = output / f"step-{step:04d}.vtu"
        frame.save(frame_path)
        verification = verify_export(
            frame_path, reference, expected_points, u, packed, matrix
        )
        frames.append(
            {
                "step": step,
                "file": frame_path.name,
                "checkpoint": digest(checkpoint),
                "output": digest(frame_path),
                "solver_receipt": receipt,
                "verification": verification,
            }
        )

    if not frames:
        raise RuntimeError("no solver-valid checkpoint frames were available")
    series = {
        "file-series-version": "1.0",
        "files": [{"name": frame["file"], "time": frame["step"]} for frame in frames],
    }
    series_path = output / "checkpoints.vtu.series"
    write_json(series_path, series)
    run_summary = digest(summary_path) if summary_path.is_file() else None
    manifest = {
        "schema_version": 1,
        "status": "completed_run_export" if run_summary else "incomplete_pilot_export",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "direct conversion of explicit saved NPZ checkpoints; no solve, "
            "interpolation, optimization, or smoothing"
        ),
        "run": {
            "directory": str(run_dir),
            "config": digest(config_path),
            "summary": run_summary,
            "solver_receipts": digest(receipts_path),
        },
        "fixture": {
            "volume": digest(volume_path),
            "points": reference.n_points,
            "tetrahedra": reference.n_cells,
            "active_tetrahedra": int(active.sum()),
        },
        "history": {
            "frames": [frame["file"] for frame in frames],
            "steps": [frame["step"] for frame in frames],
            "series": series_path.name,
        },
        "frames": frames,
        "omissions": omissions,
        "exporter": digest(Path(__file__)),
        "final_state_verification": completed_verification,
        "final_best_vtu": (
            digest(run_dir / "final.vtu") if (run_dir / "final.vtu").is_file() else None
        ),
        "final_best_policy": "referenced when present; never recreated by this exporter",
    }
    write_json(output / "manifest.json", manifest)
    shutil.copy2(Path(__file__), output / Path(__file__).name)


if __name__ == "__main__":
    main()
