"""Validate adjoint-only GPU-contact scope against saved CPU references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    output_dir: Path = EXPERIMENT / "data/adjoint-scope-validation-001"
    cpu_baseline_dir: Path = EXPERIMENT / "data/adjoint-reference-baseline20-cpu-003"
    cpu_zero_dir: Path = EXPERIMENT / "data/adjoint-reference-zero19-cpu-003"
    gpu_baseline_dir: Path = EXPERIMENT / "data/adjoint-scope-baseline20-gpu-003"
    gpu_zero_dir: Path = EXPERIMENT / "data/adjoint-scope-zero19-gpu-003"
    old_cpu_baseline_dir: Path = EXPERIMENT / "data/adjoint-tolerance-baseline20-001"
    old_cpu_zero_dir: Path = EXPERIMENT / "data/adjoint-tolerance-zero19-001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def vector_error(
    value: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | None]:
    value = value.detach().cpu().flatten().to(torch.float64)
    reference = reference.detach().cpu().flatten().to(torch.float64)
    value_l2 = float(torch.linalg.vector_norm(value))
    reference_l2 = float(torch.linalg.vector_norm(reference))
    difference_l2 = float(torch.linalg.vector_norm(value - reference))
    relative_l2 = None if reference_l2 == 0 else difference_l2 / reference_l2
    cosine = (
        None
        if value_l2 == 0 or reference_l2 == 0
        else float(torch.dot(value, reference) / (value_l2 * reference_l2))
    )
    return {
        "value_l2": value_l2,
        "reference_l2": reference_l2,
        "difference_l2": difference_l2,
        "relative_l2": relative_l2,
        "cosine": cosine,
    }


def load_summary(directory: Path) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    assert summary_path.is_file(), summary_path
    value = json.loads(summary_path.read_text())
    assert value["success"] is True, summary_path
    assert len(value["arms"]) == 1, summary_path
    return value


def one_reference(directory: Path) -> tuple[dict[str, Any], Path]:
    paths = sorted((directory / "references").glob("*.pt"))
    assert len(paths) == 1, paths
    reference_path = paths[0]
    value = torch.load(reference_path, map_location="cpu", weights_only=False)
    assert value["rtol"] == 1e-8
    for key in (
        "data_gradient_q",
        "data_gradient_jaw",
        "total_gradient_q",
        "total_gradient_jaw",
        "next_projected_adam_q_update",
        "next_projected_adam_jaw_update",
    ):
        assert isinstance(value[key], torch.Tensor), key
    return value, reference_path


def local_path(run_dir: Path, record: dict[str, Any]) -> Path:
    """Resolve an archived relative path or a remote absolute run path."""
    raw = Path(record["path"])
    if raw.is_file():
        path = raw
    elif not raw.is_absolute():
        path = run_dir / raw
    else:
        parts = raw.parts
        assert "data" in parts, raw
        path = EXPERIMENT / Path(*parts[parts.index("data") :])
    assert path.is_file(), path
    assert sha256(path) == record["sha256"], path
    return path


def raw_vectors(run_dir: Path, row: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    record = row["raw_vectors"]
    assert isinstance(record, dict)
    path = local_path(run_dir, record)
    value = torch.load(path, map_location="cpu", weights_only=False)
    assert value["rtol"] == row["rtol_requested"]
    return value, path


def old_cpu_seconds(directory: Path) -> float:
    summary = load_summary(directory)
    rows = next(iter(summary["arms"].values()))["rows"]
    row = next(item for item in rows if item["rtol_requested"] == 1e-7)
    return float(row["owned_adjoint_seconds"])


def compare_arm(
    *,
    label: str,
    cpu_dir: Path,
    gpu_dir: Path,
    old_cpu_dir: Path,
) -> dict[str, Any]:
    load_summary(cpu_dir)
    gpu_summary = load_summary(gpu_dir)
    reference, reference_path = one_reference(cpu_dir)
    cpu_checkpoint = reference["checkpoint"]
    rows = next(iter(gpu_summary["arms"].values()))["rows"]
    assert {row["rtol_requested"] for row in rows} == {1e-8, 1e-4}
    comparison_rows = []
    for row in rows:
        assert row["success"] is True
        assert row["checkpoint"]["sha256"] == cpu_checkpoint["sha256"]
        assert row["no_primal"]["forward_count"] == 0
        assert row["gpu_contact"]["enabled"] is True
        assert row["gpu_contact"]["products"] > 0
        vector, vector_path = raw_vectors(gpu_dir, row)
        metrics = {
            key: vector_error(vector[key], reference[key])
            for key in (
                "data_gradient_q",
                "data_gradient_jaw",
                "total_gradient_q",
                "total_gradient_jaw",
                "next_projected_adam_q_update",
                "next_projected_adam_jaw_update",
            )
        }
        comparison_rows.append(
            {
                "rtol": row["rtol_requested"],
                "raw_vectors": file_record(vector_path),
                "adjoint": row["adjoint"],
                "gpu_contact": row["gpu_contact"],
                "no_primal": row["no_primal"],
                "comparisons_to_cpu_1e-8": metrics,
            }
        )
    return {
        "label": label,
        "cpu_reference": file_record(reference_path),
        "cpu_reference_checkpoint": cpu_checkpoint,
        "cpu_summary": file_record(cpu_dir / "summary.json"),
        "gpu_summary": file_record(gpu_dir / "summary.json"),
        "old_cpu_1e-7_adjoint_seconds": old_cpu_seconds(old_cpu_dir),
        "rows": comparison_rows,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    cfg.output_dir.mkdir(parents=True)
    arms = [
        compare_arm(
            label="baseline20",
            cpu_dir=cfg.cpu_baseline_dir,
            gpu_dir=cfg.gpu_baseline_dir,
            old_cpu_dir=cfg.old_cpu_baseline_dir,
        ),
        compare_arm(
            label="zero19",
            cpu_dir=cfg.cpu_zero_dir,
            gpu_dir=cfg.gpu_zero_dir,
            old_cpu_dir=cfg.old_cpu_zero_dir,
        ),
    ]
    summary = {
        "schema": "smile-adjoint-scope-cpu-gpu-validation-v1",
        "success": True,
        "scope": "fixed saved states; direct CPU 1e-8 vector comparison to GPU adjoint-only contact replay; no primal solve",
        "script": file_record(Path(__file__)),
        "arms": arms,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_output(cfg.output_dir)
    cherries.log_metrics({"adjoint_scope_validation/success": 1.0})


if __name__ == "__main__":
    cherries.main(main)
