from __future__ import annotations

# Build an atomic, native-ParaView visualization bundle from production outputs.
# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from liblaf import cherries

logger = logging.getLogger(__name__)

EXPECTED_PARAVIEW_VERSION = "6.1.1"
GROUP_DIR = Path(__file__).resolve().parents[1]
PVBATCH = Path("/usr/bin/pvbatch")
RENDERER = Path(__file__).with_name("20-render-bumpy-transfer-paraview.py")
OUTPUT_ROOT_NAME = "21-bumpy-transfer-paraview"
MANIFEST_NAME = "21-bumpy-transfer-paraview-manifest.json"
RENDERER_RECEIPT_NAME = "21-renderer-receipt.json"
CASE_ORDER = ("thin", "medium", "thick")


class Config(cherries.BaseConfig):
    input_summary: Path = cherries.input("10-bumpy-activation-transfer-summary.json")
    # Individual files are logged below. Queuing this directory itself makes the
    # Cherries Local plugin collide with the already-populated snapshot folder.
    output_root: Path = GROUP_DIR / "data" / OUTPUT_ROOT_NAME
    output_manifest: Path = cherries.output(MANIFEST_NAME, mkdir=True)
    warp_factor: float = 40.0
    image_width: int = 1800
    image_height: int = 1350


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if read_json(temporary) != payload:
        raise RuntimeError(f"JSON readback changed for {path}")
    temporary.replace(path)


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def validate_inputs(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = read_json(summary_path)
    if summary.get("status") != "ok" or summary.get("complete") is not True:
        raise ValueError("simulation summary is incomplete")
    cases = summary.get("cases")
    if not isinstance(cases, list) or [row.get("label") for row in cases] != list(
        CASE_ORDER
    ):
        raise ValueError("unexpected simulation cases")
    source_root = summary_path.parent / "10-bumpy-activation-transfer"
    identities: dict[str, Any] = {"summary": identity(summary_path)}
    for label in CASE_ORDER:
        path = source_root / label / f"10-{label}-bumpy-activation-transfer.vtu"
        identities[label] = identity(path)
    hashes = {
        str(row["bumpy_activation_sha256"])
        for row in cases
        if "bumpy_activation_sha256" in row
    }
    active_ids = {
        str(row["active_ids_sha256"]) for row in cases if "active_ids_sha256" in row
    }
    active_centers = {
        str(row["active_centers_xz_sha256"])
        for row in cases
        if "active_centers_xz_sha256" in row
    }
    if len(hashes) != 1 or len(active_ids) != 1 or len(active_centers) != 1:
        raise ValueError("activation source is not shared across thickness cases")
    return summary, identities


def validate_renderer_receipt(
    receipt_path: Path, temporary_root: Path, cfg: Config
) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    if (
        receipt.get("status") != "ok"
        or receipt.get("complete") is not True
        or receipt.get("paraview_version") != EXPECTED_PARAVIEW_VERSION
        or receipt.get("native_paraview_rendering") is not True
    ):
        raise ValueError("ParaView renderer receipt is invalid")
    if receipt.get("resolution") != [cfg.image_width, cfg.image_height]:
        raise ValueError("renderer resolution changed")
    surface = receipt.get("surface_visualization")
    if not isinstance(surface, dict) or surface.get("warp_factor") != cfg.warp_factor:
        raise ValueError("renderer warp factor changed")
    pairs = [*receipt.get("cases", []), receipt.get("activation_source")]
    if len(pairs) != 4 or any(not isinstance(pair, dict) for pair in pairs):
        raise ValueError("renderer did not produce four standalone pairs")
    for pair in pairs:
        for kind in ("png", "pvsm"):
            record = pair[kind]
            path = temporary_root / record["relative_path"]
            actual = identity(path)
            if {
                "size_bytes": actual["size_bytes"],
                "sha256": actual["sha256"],
            } != {
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }:
                raise ValueError(f"renderer identity mismatch: {path}")
            if kind == "png" and png_size(path) != (
                cfg.image_width,
                cfg.image_height,
            ):
                raise ValueError(f"renderer PNG dimensions changed: {path}")
            if (
                kind == "pvsm"
                and "ServerManagerState"
                not in path.read_text(encoding="utf-8", errors="strict")[:2048]
            ):
                raise ValueError(f"invalid ParaView state: {path}")
    return receipt


def final_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        identity(path)
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    ]


def main(cfg: Config) -> None:
    summary_path = Path(cfg.input_summary).resolve()
    output_root = Path(cfg.output_root).resolve()
    manifest_path = Path(cfg.output_manifest).resolve()
    expected_data_dir = (GROUP_DIR / "data").resolve()
    if summary_path != (
        expected_data_dir / "10-bumpy-activation-transfer-summary.json"
    ):
        raise ValueError("production summary path cannot be overridden")
    if output_root != (expected_data_dir / OUTPUT_ROOT_NAME):
        raise ValueError("output root cannot be overridden")
    if manifest_path != (expected_data_dir / MANIFEST_NAME):
        raise ValueError("manifest path cannot be overridden")
    if output_root.exists() or manifest_path.exists():
        raise FileExistsError("refusing to replace existing ParaView assets")
    if not RENDERER.is_file() or not PVBATCH.is_file():
        raise FileNotFoundError("ParaView renderer or pvbatch is missing")
    if cfg.warp_factor <= 0.0 or min(cfg.image_width, cfg.image_height) < 800:
        raise ValueError("invalid render scale or resolution")

    summary, input_identities_before = validate_inputs(summary_path)
    renderer_identity_before = identity(RENDERER)
    pvbatch_identity_before = identity(PVBATCH)
    version = subprocess.run(
        [str(PVBATCH), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    version_text = f"{version.stdout}\n{version.stderr}".strip()
    if not version_text.endswith(EXPECTED_PARAVIEW_VERSION):
        raise RuntimeError(f"unexpected ParaView version: {version_text}")

    temporary_parent = GROUP_DIR / "tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix="21-paraview-build-", dir=temporary_parent)
    ).resolve()
    renderer_receipt_path = temporary_root / RENDERER_RECEIPT_NAME
    command = [
        str(PVBATCH),
        "--force-offscreen-rendering",
        str(RENDERER.resolve()),
        "--input-root",
        str((summary_path.parent / "10-bumpy-activation-transfer").resolve()),
        "--summary",
        str(summary_path),
        "--output-root",
        str(temporary_root / OUTPUT_ROOT_NAME),
        "--renderer-receipt",
        str(renderer_receipt_path),
        "--warp-factor",
        str(cfg.warp_factor),
        "--resolution",
        str(cfg.image_width),
        str(cfg.image_height),
    ]
    logger.info("Running native ParaView renderer: %s", command)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=GROUP_DIR,
    )
    if completed.stdout:
        logger.info("pvbatch stdout:\n%s", completed.stdout.rstrip())
    if completed.stderr:
        logger.info("pvbatch stderr:\n%s", completed.stderr.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(f"pvbatch failed with exit code {completed.returncode}")
    renderer_receipt = validate_renderer_receipt(
        renderer_receipt_path, temporary_root / OUTPUT_ROOT_NAME, cfg
    )

    input_identities_after = validate_inputs(summary_path)[1]
    if input_identities_after != input_identities_before:
        raise RuntimeError("production inputs changed during rendering")
    if identity(RENDERER) != renderer_identity_before:
        raise RuntimeError("renderer changed during rendering")
    if identity(PVBATCH) != pvbatch_identity_before:
        raise RuntimeError("pvbatch changed during rendering")

    built_root = temporary_root / OUTPUT_ROOT_NAME
    built_root.replace(output_root)
    renderer_receipt_final = output_root / RENDERER_RECEIPT_NAME
    shutil.move(str(renderer_receipt_path), renderer_receipt_final)
    shutil.rmtree(temporary_root)
    inventory = final_inventory(output_root)
    manifest = {
        "schema_version": 1,
        "design": "fat-thickness-bumpy-activation-transfer-paraview-v1",
        "complete": True,
        "status": "ok",
        "execution_profile": "debug" if os.environ.get("DEBUG") == "1" else "default",
        "execution": {
            "simulation_executed": False,
            "asset_build_only": True,
            "native_paraview_rendering_executed": True,
            "standalone_png_count": 4,
            "standalone_pvsm_count": 4,
            "combined_images": False,
        },
        "command": command,
        "input_summary": summary,
        "input_identities_before": input_identities_before,
        "input_identities_after": input_identities_after,
        "renderer": renderer_identity_before,
        "pvbatch": {**pvbatch_identity_before, "version": EXPECTED_PARAVIEW_VERSION},
        "renderer_receipt": {
            **identity(renderer_receipt_final),
            "payload": renderer_receipt,
        },
        "output_root": str(output_root),
        "output_inventory": inventory,
    }
    write_json(manifest_path, manifest)
    for path in sorted(path for path in output_root.rglob("*") if path.is_file()):
        cherries.log_output(path)
    cherries.log_metrics(
        {
            "assets/standalone_png_count": 4,
            "assets/standalone_pvsm_count": 4,
            "render/warp_factor": cfg.warp_factor,
            "render/image_width": cfg.image_width,
            "render/image_height": cfg.image_height,
        }
    )
    logger.info("Wrote native ParaView asset manifest to %s", manifest_path)


if __name__ == "__main__":
    cherries.main(main)
