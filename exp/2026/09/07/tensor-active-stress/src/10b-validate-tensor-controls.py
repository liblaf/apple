"""Validate chunked tensor-control projection and eigenvalue diagnostics."""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pydantic_settings as ps
import torch
from experiment_profile import ProfileCometNoCommit
from tensor_controls import (
    EIGEN_BATCH_SIZE,
    coordinates,
    eigenvalues,
    matrices,
    project,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parent.parent
REPOSITORY = GROUP.parents[4]


class Config(cherries.BaseConfig):
    """Settings for small-reference and face-sized control audits."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("10b-tensor-controls-validation", mkdir=True)
    seed: int = 20260907
    small_count: int = 4097
    face_active_cells: int = 288235
    maximum_mpa: float = 0.030201342281879196
    memory_limit_mib: float = 768.0


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
            default=lambda value: (
                value.item() if isinstance(value, torch.Tensor) else str(value)
            ),
        )
        + "\n"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def max_abs(value: torch.Tensor) -> float:
    return float(torch.max(torch.abs(value)).cpu())


def unbatched_projection(
    q: torch.Tensor, maximum: float
) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(matrices(q))
    bounded = values.clamp(0.0, maximum)
    result = coordinates((vectors * bounded[:, None, :]) @ vectors.transpose(-1, -2))
    return result, values


def small_reference_audit(
    *, device: torch.device, count: int, seed: int, maximum: float
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(seed)
    q = 0.025 * torch.randn(
        (count, 6), dtype=torch.float64, device=device, generator=generator
    )
    reconstructed = coordinates(matrices(q))
    expected_q, expected_input_eig = unbatched_projection(q, maximum)
    expected_output_eig = torch.linalg.eigvalsh(matrices(expected_q))
    actual_q = q.clone()
    stats = project(actual_q, maximum)
    actual_output_eig = eigenvalues(matrices(actual_q))
    expected_change_rms = torch.sqrt(torch.mean((expected_q - q).square()))
    expected_negative_fraction = torch.mean((expected_input_eig < 0).to(torch.float64))
    expected_upper_fraction = torch.mean(
        (expected_input_eig > maximum).to(torch.float64)
    )
    idempotent = actual_q.clone()
    idempotent_stats = project(idempotent, maximum)
    result = {
        "device": str(device),
        "count": count,
        "chunk_size": EIGEN_BATCH_SIZE,
        "chunk_count": (count + EIGEN_BATCH_SIZE - 1) // EIGEN_BATCH_SIZE,
        "coordinate_roundtrip_max_abs": max_abs(reconstructed - q),
        "projection_vs_unbatched_max_abs_MPa": max_abs(actual_q - expected_q),
        "eigenvalues_vs_unbatched_max_abs_MPa": max_abs(
            actual_output_eig - expected_output_eig
        ),
        "projection_rms_metric_abs_error_MPa": abs(
            stats["projection_rms"] - float(expected_change_rms.cpu())
        ),
        "negative_fraction_metric_abs_error": abs(
            stats["projected_negative_eigenvalue_fraction"]
            - float(expected_negative_fraction.cpu())
        ),
        "upper_fraction_metric_abs_error": abs(
            stats["projected_upper_eigenvalue_fraction"]
            - float(expected_upper_fraction.cpu())
        ),
        "output_eigenvalue_min_MPa": float(actual_output_eig.min().cpu()),
        "output_eigenvalue_max_MPa": float(actual_output_eig.max().cpu()),
        "finite": bool(
            torch.isfinite(actual_q).all() and torch.isfinite(actual_output_eig).all()
        ),
        "idempotence_max_abs_MPa": max_abs(idempotent - actual_q),
        "idempotent_projection_rms_MPa": idempotent_stats["projection_rms"],
        "projection_stats": stats,
    }
    limits = {
        "coordinate_roundtrip_max_abs": 1.0e-15,
        "projection_vs_unbatched_max_abs_MPa": 1.0e-14,
        "eigenvalues_vs_unbatched_max_abs_MPa": 1.0e-14,
        "projection_rms_metric_abs_error_MPa": 1.0e-14,
        "negative_fraction_metric_abs_error": 1.0e-15,
        "upper_fraction_metric_abs_error": 1.0e-15,
        "idempotence_max_abs_MPa": 1.0e-14,
    }
    for name, limit in limits.items():
        if not float(result[name]) < limit:
            raise AssertionError(f"{name} exceeds its comparison limit")
    if not result["finite"]:
        raise AssertionError("small projection produced a nonfinite value")
    if result["output_eigenvalue_min_MPa"] < -1.0e-14:
        raise AssertionError("small projection violated the PSD lower bound")
    if result["output_eigenvalue_max_MPa"] > maximum + 1.0e-14:
        raise AssertionError("small projection violated the upper bound")
    result["limits"] = limits
    result["status"] = "passed"
    return result


def face_sized_cuda_audit(cfg: Config) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the face-sized allocation audit")
    device = torch.device("cuda:0")
    device_index = 0
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    q = 0.025 * torch.randn(
        (cfg.face_active_cells, 6),
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    sample_ids = torch.as_tensor(
        (0, 1, 1023, 1024, 8191, 65535, cfg.face_active_cells - 1),
        dtype=torch.int64,
        device=device,
    )
    sample_before = q[sample_ids].cpu()
    input_eig = eigenvalues(matrices(q))
    expected_negative_fraction = float(
        torch.mean((input_eig < 0).to(torch.float64)).cpu()
    )
    expected_upper_fraction = float(
        torch.mean((input_eig > cfg.maximum_mpa).to(torch.float64)).cpu()
    )
    stats = project(q, cfg.maximum_mpa)
    output_eig = eigenvalues(matrices(q))
    sample_after = q[sample_ids].cpu()
    expected_sample, _ = unbatched_projection(sample_before, cfg.maximum_mpa)
    torch.cuda.synchronize(device)
    peak_allocated_mib = torch.cuda.max_memory_allocated(device_index) / 2**20
    peak_reserved_mib = torch.cuda.max_memory_reserved(device_index) / 2**20
    result = {
        "status": "passed",
        "device": str(device),
        "dtype": str(q.dtype),
        "count": len(q),
        "coordinates": q.shape[1],
        "chunk_size": EIGEN_BATCH_SIZE,
        "chunk_count": (len(q) + EIGEN_BATCH_SIZE - 1) // EIGEN_BATCH_SIZE,
        "finite_coordinates": bool(torch.isfinite(q).all().cpu()),
        "finite_eigenvalues": bool(torch.isfinite(output_eig).all().cpu()),
        "output_eigenvalue_min_MPa": float(output_eig.min().cpu()),
        "output_eigenvalue_max_MPa": float(output_eig.max().cpu()),
        "sample_projection_vs_cpu_unbatched_max_abs_MPa": max_abs(
            sample_after - expected_sample
        ),
        "negative_fraction_metric_abs_error": abs(
            stats["projected_negative_eigenvalue_fraction"] - expected_negative_fraction
        ),
        "upper_fraction_metric_abs_error": abs(
            stats["projected_upper_eigenvalue_fraction"] - expected_upper_fraction
        ),
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "declared_peak_allocated_limit_mib": cfg.memory_limit_mib,
        "projection_stats": stats,
    }
    if result["count"] != cfg.face_active_cells:
        raise AssertionError("face-sized audit did not use the declared cell count")
    if not result["finite_coordinates"] or not result["finite_eigenvalues"]:
        raise AssertionError("face-sized projection produced a nonfinite value")
    if result["output_eigenvalue_min_MPa"] < -1.0e-13:
        raise AssertionError("face-sized projection violated the PSD lower bound")
    if result["output_eigenvalue_max_MPa"] > cfg.maximum_mpa + 1.0e-13:
        raise AssertionError("face-sized projection violated the upper bound")
    if result["sample_projection_vs_cpu_unbatched_max_abs_MPa"] >= 1.0e-13:
        raise AssertionError(
            "face-sized chunked projection disagrees with CPU reference"
        )
    if result["negative_fraction_metric_abs_error"] >= 1.0e-15:
        raise AssertionError("face-sized negative projection fraction is incorrect")
    if result["upper_fraction_metric_abs_error"] >= 1.0e-15:
        raise AssertionError("face-sized upper projection fraction is incorrect")
    if peak_allocated_mib >= cfg.memory_limit_mib:
        raise AssertionError("face-sized eigensolver exceeded the allocation limit")
    return result


def snapshot_sources(out: Path) -> dict[str, object]:
    source_dir = out / "sources"
    source_dir.mkdir()
    files = (
        Path(__file__).resolve(),
        Path(__file__).with_name("tensor_controls.py").resolve(),
        Path(__file__).with_name("experiment_profile.py").resolve(),
    )
    records = []
    for source in files:
        destination = source_dir / source.name
        shutil.copy2(source, destination)
        records.append(
            {
                "path": str(source.relative_to(REPOSITORY)),
                "sha256": sha256(source),
                "snapshot": str(destination.relative_to(GROUP)),
            }
        )
    return {
        "sources": records,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "comet_auto_log_git_metadata": os.environ.get("COMET_AUTO_LOG_GIT_METADATA"),
        "comet_auto_log_git_patch": os.environ.get("COMET_AUTO_LOG_GIT_PATCH"),
        "comet_auto_log_env_details": os.environ.get("COMET_AUTO_LOG_ENV_DETAILS"),
    }


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("choose an empty output directory")
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    provenance = snapshot_sources(out)
    write_json(out / "provenance.json", provenance)
    torch.set_default_dtype(torch.float64)
    small_cpu = small_reference_audit(
        device=torch.device("cpu"),
        count=cfg.small_count,
        seed=cfg.seed,
        maximum=cfg.maximum_mpa,
    )
    small_cuda = small_reference_audit(
        device=torch.device("cuda:0"),
        count=cfg.small_count,
        seed=cfg.seed,
        maximum=cfg.maximum_mpa,
    )
    face_cuda = face_sized_cuda_audit(cfg)
    summary = {
        "status": "passed",
        "scope": "chunked tensor-control projection and eigenvalues; no face solve",
        "small_cpu": small_cpu,
        "small_cuda": small_cuda,
        "face_sized_cuda": face_cuda,
        "provenance": provenance,
    }
    write_json(out / "summary.json", summary)
    cherries.log_metrics(
        {
            "tensor_controls/small_cpu_projection_error": small_cpu[
                "projection_vs_unbatched_max_abs_MPa"
            ],
            "tensor_controls/small_cuda_projection_error": small_cuda[
                "projection_vs_unbatched_max_abs_MPa"
            ],
            "tensor_controls/face_sample_projection_error": face_cuda[
                "sample_projection_vs_cpu_unbatched_max_abs_MPa"
            ],
            "tensor_controls/face_peak_allocated_mib": face_cuda["peak_allocated_mib"],
        },
        step=0,
    )
    LOG.info("Tensor-control chunk audit passed: %s", out / "summary.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
