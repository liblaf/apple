from __future__ import annotations

# Prepare archived inputs, delegate every 3D pixel to native ParaView 6.1.1,
# and create two separate scalar-metric charts.
# ruff: noqa: C901, EM101, EM102, FBT003, PERF401, PLR0912, RUF001, TRY003
import hashlib
import json
import logging
import math
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pyvista as pv

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
DESIGN = "cuboid-fat-thickness-meeting-assets-v2"
EXPECTED_PARAVIEW_VERSION = "6.1.1"
IMAGE_RESOLUTION = (2400, 1800)
CASE_ORDER = ("top-fat-0p04", "top-fat-0p08", "top-fat-0p12")

GROUP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = GROUP_DIR / "data"
DOCS_DIR = GROUP_DIR / "docs"
BUNDLE_ROOT = DATA_DIR / "30-cuboid-fat-thickness"
INPUT_ROOT = BUNDLE_ROOT / "inputs"
PARAVIEW_ROOT = BUNDLE_ROOT / "paraview"
METRICS_ROOT = BUNDLE_ROOT / "metrics"
CONTRACT = BUNDLE_ROOT / "30-cuboid-paraview-contract.json"
RENDERER_RECEIPT = BUNDLE_ROOT / "30-cuboid-paraview-renderer-receipt.json"
RECEIPT = DATA_DIR / "30-cuboid-assets-receipt.json"
REPORT = DOCS_DIR / "30-cuboid-assets.md"
RENDERER = Path(__file__).with_name("30-cuboid-render-paraview.py")
PVBATCH = Path("/usr/bin/pvbatch")

ARCHIVE_ROOT = Path(
    "/home/liblaf/mnt/DATA41/cherries/liblaf/apple/runs/2026/08/11/"
    "fat-layer-thickness-sandwich/30-run-large-deformation-fat-thickness/"
    "2026-08-12T141945-fat-large-deformation-thickness"
)


@dataclass(frozen=True)
class Identity:
    relative_path: str
    size_bytes: int
    sha256: str
    local_name: str


SOURCE_FILES = {
    "summary_json": Identity(
        "data/30-large-deformation-summary.json",
        30_414,
        "1d8369e4157c872059401e5523b3b139e0ef3918cc33e9dff9ef68d19ccbcc8b",
        "30-cuboid-source-summary.json",
    ),
    "summary_csv": Identity(
        "data/30-large-deformation-summary.csv",
        5_308,
        "1557c94c755e0968ad9798c8c119148f617ac8c34502e1560c9a9d7fd14e1e49",
        "30-cuboid-source-summary.csv",
    ),
    "report": Identity(
        "docs/30-large-deformation-report.md",
        3_968,
        "f9cd1fca10ef01e2797a9254862f98b1eb6d9876d86f2fbcdb7f9087fdacf07a",
        "30-cuboid-source-report.md",
    ),
    "source": Identity(
        "src/30-run-large-deformation-fat-thickness.py",
        40_962,
        "6b338818a3ff5e5be905c5f9c0e28679abf89660375e190252a4218ff724a5b0",
        "30-cuboid-source-runner.py",
    ),
    "vtu_0p04": Identity(
        "data/30-top-fat-0p04-pressure-0p6.vtu",
        2_180_253,
        "d7ad9001330c5638a91526eeee5a408c2b54b75ae92cf955eed6469e1ac62114",
        "30-cuboid-top-fat-0p04-pressure-0p60.vtu",
    ),
    "vtu_0p08": Identity(
        "data/30-top-fat-0p08-pressure-0p6.vtu",
        2_892_315,
        "d34aceecd9965854ce89fe78ce39953b8c34c860e7bf3641a2e6b8457d9b1a0a",
        "30-cuboid-top-fat-0p08-pressure-0p60.vtu",
    ),
    "vtu_0p12": Identity(
        "data/30-top-fat-0p12-pressure-0p6.vtu",
        3_339_421,
        "9355da8265f57fea964811723f2aa25702e57ee993504947027ed176faa477ec",
        "30-cuboid-top-fat-0p12-pressure-0p60.vtu",
    ),
    "grid_0p04": Identity(
        "data/30-top-fat-0p04-pressure-0p6-top-grid.npz",
        79_276,
        "b28b6a28ffb1acd483e9958bfb7ea612e33eecbd57ce76b28a953c1fa08acf57",
        "30-cuboid-top-fat-0p04-pressure-0p60-top-grid.npz",
    ),
    "grid_0p08": Identity(
        "data/30-top-fat-0p08-pressure-0p6-top-grid.npz",
        79_031,
        "68def5ba4ffb80701bd9b4d4a06c1a6de577c2f2aaf022eeb2644d52e04d83d3",
        "30-cuboid-top-fat-0p08-pressure-0p60-top-grid.npz",
    ),
    "grid_0p12": Identity(
        "data/30-top-fat-0p12-pressure-0p6-top-grid.npz",
        78_864,
        "ac89c13d55e92f0949959a5636f4100976e812b9b15b2554a08ecc3750d4d96f",
        "30-cuboid-top-fat-0p12-pressure-0p60-top-grid.npz",
    ),
}

CASE_SOURCE_KEYS = {
    "top-fat-0p04": "vtu_0p04",
    "top-fat-0p08": "vtu_0p08",
    "top-fat-0p12": "vtu_0p12",
}
EXPECTED_MESH_COUNTS = {
    "top-fat-0p04": (9_148, 42_540),
    "top-fat-0p08": (12_289, 59_907),
    "top-fat-0p12": (14_606, 73_876),
}


class Config(cherries.BaseConfig):
    input_summary: Path = cherries.input(
        ARCHIVE_ROOT / SOURCE_FILES["summary_json"].relative_path
    )
    output_receipt: Path = cherries.output(RECEIPT, mkdir=True)


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


def require_source(label: str, spec: Identity) -> dict[str, Any]:
    path = ARCHIVE_ROOT / spec.relative_path
    actual = identity(path)
    expected = {"size_bytes": spec.size_bytes, "sha256": spec.sha256}
    if {key: actual[key] for key in expected} != expected:
        raise ValueError(f"archived source identity changed for {label}: {actual}")
    return actual


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
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


def validate_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if summary.get("render_pressure") != 0.6:
        raise ValueError("archived render pressure changed")
    if summary.get("debug_local_only") is not True:
        raise ValueError("archive is no longer the documented DEBUG run")
    controlled = summary.get("controlled_config")
    expected_controlled = {
        "bottom_fat_thickness": 0.04,
        "smas_thickness": 0.02,
        "E_fat": 1.0,
        "E_smas": 100.0,
        "nu": 0.49,
        "smas_prestrain": [0.8, 1.0, 0.8, 0.0, 0.0, 0.0],
        "tetwild_lr": 0.02,
        "grid_size": 101,
        "grid_margin": 0.02,
    }
    if not isinstance(controlled, dict):
        raise TypeError("controlled_config changed")
    for key, expected in expected_controlled.items():
        if controlled.get(key) != expected:
            raise ValueError(f"controlled config changed at {key}")
    rows = summary.get("cases")
    if not isinstance(rows, list):
        raise TypeError("summary cases changed")
    selected = [row for row in rows if row.get("pressure") == 0.6]
    selected.sort(key=lambda row: float(row["top_fat_thickness"]))
    if [row["top_fat_thickness"] for row in selected] != [0.04, 0.08, 0.12]:
        raise ValueError("pressure-0.60 case set changed")
    for row in selected:
        if (
            row.get("solver/result") != "PRIMARY_SUCCESS"
            or row.get("solver/success") != 1
            or row.get("display_valid") != 1
            or row.get("finite") != 1
            or row.get("deformation/detF_inverted") != 0
            or row.get("top_normal/flipped") != 0
        ):
            raise ValueError(f"invalid archived case {row.get('case')}")
        for key, value in row.items():
            if isinstance(value, int | float) and not math.isfinite(float(value)):
                raise ValueError(f"non-finite {key} in {row.get('case')}")
    return selected


def copy_sources() -> dict[str, dict[str, Any]]:
    INPUT_ROOT.mkdir(parents=True, exist_ok=False)
    copied: dict[str, dict[str, Any]] = {}
    for label, spec in SOURCE_FILES.items():
        source = ARCHIVE_ROOT / spec.relative_path
        source_identity = require_source(label, spec)
        destination = INPUT_ROOT / spec.local_name
        shutil.copy2(source, destination)
        destination_identity = identity(destination)
        if (
            destination_identity["size_bytes"] != source_identity["size_bytes"]
            or destination_identity["sha256"] != source_identity["sha256"]
        ):
            raise ValueError(f"copied source changed for {label}")
        cherries.log_input(source)
        copied[label] = {
            "archive": source_identity,
            "local_copy": destination_identity,
            "byte_equal": True,
        }
    return copied


def prepare_contract(
    rows: list[dict[str, Any]], copied: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    u_y_parts: list[np.ndarray] = []
    for case_id, row in zip(CASE_ORDER, rows, strict=True):
        source_key = CASE_SOURCE_KEYS[case_id]
        path = Path(copied[source_key]["local_copy"]["path"])
        mesh = pv.read(path)
        if not isinstance(mesh, pv.UnstructuredGrid):
            raise TypeError(f"{case_id} is not an UnstructuredGrid")
        expected_points, expected_cells = EXPECTED_MESH_COUNTS[case_id]
        if (mesh.n_points, mesh.n_cells) != (expected_points, expected_cells):
            raise ValueError(f"mesh dimensions changed for {case_id}")
        displacement = np.asarray(mesh.point_data["Displacement"], dtype=np.float64)
        if displacement.shape != (mesh.n_points, 3) or not np.isfinite(
            displacement
        ).all():
            raise ValueError(f"invalid displacement for {case_id}")
        u_y_parts.append(displacement[:, 1])
        cases.append(
            {
                "case_id": case_id,
                "top_fat_thickness": float(row["top_fat_thickness"]),
                "pressure": float(row["pressure"]),
                "input_path": str(path.resolve()),
                "input_size_bytes": path.stat().st_size,
                "input_sha256": sha256(path),
                "n_points": mesh.n_points,
                "n_cells": mesh.n_cells,
                "metrics": {
                    "p95_p05": float(row["top_grid/u_y_p95_minus_p05"]),
                    "laplacian_rms": float(row["top_grid/laplacian_rms"]),
                    "laplacian_rms_normalized": float(
                        row["top_grid/laplacian_rms_normalized"]
                    ),
                    "max_displacement": float(row["max_displacement"]),
                    "min_det_f": float(row["deformation/detF_min"]),
                    "q001_det_f": float(row["deformation/detF_q001"]),
                },
            }
        )
    pooled_u_y = np.concatenate(u_y_parts)
    shared_range = [float(pooled_u_y.min()), float(pooled_u_y.max())]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "renderer": "native ParaView 6.1.1",
        "case_order": list(CASE_ORDER),
        "image_resolution": list(IMAGE_RESOLUTION),
        "shared_u_y_range": shared_range,
        "camera": {
            "position": [1.65, 1.13, 1.65],
            "focal_point": [0.5, 0.09, 0.5],
            "view_up": [0.0, 1.0, 0.0],
            "parallel_scale": 0.78,
            "projection": "parallel",
        },
        "warp": {"vectors": "Displacement", "scale_factor": 1.0},
        "cases": cases,
    }
    write_json(CONTRACT, contract)
    return contract, identity(CONTRACT)


def plot_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from matplotlib import pyplot as plt

    thickness = np.asarray([float(row["top_fat_thickness"]) for row in rows])
    variation = np.asarray(
        [float(row["top_grid/u_y_p95_minus_p05"]) for row in rows]
    )
    laplacian = np.asarray([float(row["top_grid/laplacian_rms"]) for row in rows])
    variation_reduction = 1.0 - variation[-1] / variation[0]
    laplacian_reduction = 1.0 - laplacian[-1] / laplacian[0]

    plt.style.use("dark_background")
    panels = (
        (
            "p95_p05",
            variation,
            "Top-surface p95-p05 vs top-fat thickness",
            "model length",
            variation_reduction,
            "#F59E75",
            "30-cuboid-top-surface-p95-p05.png",
        ),
        (
            "laplacian_rms",
            laplacian,
            "Laplacian RMS vs top-fat thickness",
            "model length⁻¹",
            laplacian_reduction,
            "#65B7A8",
            "30-cuboid-top-surface-laplacian-rms.png",
        ),
    )
    outputs: dict[str, Any] = {}
    for metric_id, values, title, ylabel, reduction, color, filename in panels:
        figure, axis = plt.subplots(
            figsize=(2400 / 200, 1500 / 200), constrained_layout=True
        )
        figure.patch.set_facecolor("#0B0D10")
        axis.set_facecolor("#0B0D10")
        axis.plot(
            thickness,
            values,
            color=color,
            linewidth=4.5,
            marker="o",
            markersize=14,
        )
        for x_value, y_value in zip(thickness, values, strict=True):
            axis.annotate(
                f"{y_value:.5g}",
                (x_value, y_value),
                xytext=(0, 16),
                textcoords="offset points",
                ha="center",
                fontsize=20,
                color="white",
            )
        axis.text(
            0.04,
            0.08,
            f"0.04 → 0.12: −{reduction:.1%}",
            transform=axis.transAxes,
            fontsize=24,
            fontweight="bold",
            color=color,
        )
        axis.set_title(
            f"{title}\ncontrolled cuboid · pressure = 0.60 model stress",
            fontsize=27,
            fontweight="bold",
            pad=22,
        )
        axis.set_xlabel("top-fat thickness [model length]", fontsize=21)
        axis.set_ylabel(ylabel, fontsize=21)
        axis.set_xticks(thickness)
        axis.tick_params(labelsize=18)
        axis.grid(True, alpha=0.18, linewidth=1.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.12, y=0.24)
        path = METRICS_ROOT / filename
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        figure.savefig(temporary, dpi=200, facecolor=figure.get_facecolor())
        plt.close(figure)
        if png_size(temporary) != (2400, 1500) or temporary.stat().st_size < 100_000:
            raise ValueError(f"metric plot PNG validation failed for {metric_id}")
        temporary.replace(path)
        outputs[metric_id] = {
            **identity(path),
            "width": 2400,
            "height": 1500,
            "reduction_fraction_0p04_to_0p12": float(reduction),
        }
    return {
        "matplotlib_version": mpl.__version__,
        "standalone_plot_count": len(outputs),
        "plots": outputs,
    }


def validate_renderer_receipt(contract: dict[str, Any]) -> dict[str, Any]:
    renderer = read_json(RENDERER_RECEIPT)
    if (
        renderer.get("complete") is not True
        or renderer.get("status") != "ok"
        or renderer.get("paraview_version") != EXPECTED_PARAVIEW_VERSION
        or renderer.get("shared_u_y_range") != contract["shared_u_y_range"]
        or renderer.get("camera") != contract["camera"]
    ):
        raise ValueError("ParaView renderer receipt changed")
    outputs = renderer.get("outputs")
    if not isinstance(outputs, list) or [row["case_id"] for row in outputs] != list(
        CASE_ORDER
    ):
        raise ValueError("ParaView output order changed")
    for row in outputs:
        png = Path(row["png"]["path"])
        pvsm = Path(row["pvsm"]["path"])
        if png_size(png) != IMAGE_RESOLUTION or png.stat().st_size < 100_000:
            raise ValueError(f"invalid screenshot {png}")
        if identity(png)["sha256"] != row["png"]["sha256"]:
            raise ValueError(f"screenshot changed for {row['case_id']}")
        if identity(pvsm)["sha256"] != row["pvsm"]["sha256"]:
            raise ValueError(f"ParaView state changed for {row['case_id']}")
    return renderer


def write_report(rows: list[dict[str, Any]], receipt: dict[str, Any]) -> None:
    plots = receipt["metric_plots"]["plots"]
    p95_reduction = plots["p95_p05"]["reduction_fraction_0p04_to_0p12"]
    lap_reduction = plots["laplacian_rms"]["reduction_fraction_0p04_to_0p12"]
    lines = [
        "# Cuboid fat-thickness meeting assets",
        "",
        "## Meeting-safe result",
        "",
        (
            "At bottom pressure `0.60` model stress, increasing the controlled "
            "cuboid's top-fat thickness from `0.04` to `0.12` reduced absolute "
            f"top-surface p95-p05 variation by `{p95_reduction:.1%}` and finite-difference "
            f"Laplacian RMS by `{lap_reduction:.1%}`."
        ),
        "",
        "This supports reduced absolute surface variation in this toy block only. "
        "It does not establish scale-invariant smoothing or an anatomical-face result.",
        "",
        "## Controlled setup",
        "",
        "- Block footprint: `1 × 1` model length.",
        "- Bottom fat: `0.04`; SMAS: `0.02`; top fat: `0.04 / 0.08 / 0.12`.",
        "- Fat: `E = 1`; SMAS: `E = 100`; all `nu = 0.49` in model units.",
        "- Fixed SMAS pre-strain: `(0.8, 1.0, 0.8, 0, 0, 0)`.",
        "- All displacement components fixed on four vertical sides; positive-y "
        "pressure applied on the free bottom-interior surface.",
        "- Each thickness was independently remeshed and solved by continuation to `0.60`.",
        "",
        "## Pressure-0.60 metrics",
        "",
        "| top fat | p95-p05 | Laplacian RMS | max displacement | min detF |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {float(row['top_fat_thickness']):.2f} | "
            f"{float(row['top_grid/u_y_p95_minus_p05']):.8f} | "
            f"{float(row['top_grid/laplacian_rms']):.6f} | "
            f"{float(row['max_displacement']):.8f} | "
            f"{float(row['deformation/detF_min']):.6f} |"
        )
    lines.extend(
        [
            "",
            "All three cases were `PRIMARY_SUCCESS`, finite, display-valid, with zero "
            "inverted tetrahedra and zero flipped top triangles.",
            "",
            "## Asset contract",
            "",
            "- The three 3D PNGs are separate native ParaView 6.1.1 renders.",
            "- Each PNG has a separate `.pvsm` state.",
            "- Camera, parallel scale, warp factor `1.0`, and vertical-displacement "
            "color range are identical across all three images.",
            "- The white outline is the undeformed rest shape.",
            "- The p95-p05 and Laplacian RMS summaries are two separate standalone "
            "Matplotlib images; there is no combined meeting chart.",
            "",
            "## Standalone meeting asset inventory",
            "",
            "- `paraview/top-fat-0p04/30-cuboid-top-fat-0p04-paraview.png` "
            "with its `.pvsm`.",
            "- `paraview/top-fat-0p08/30-cuboid-top-fat-0p08-paraview.png` "
            "with its `.pvsm`.",
            "- `paraview/top-fat-0p12/30-cuboid-top-fat-0p12-paraview.png` "
            "with its `.pvsm`.",
            "- `metrics/30-cuboid-top-surface-p95-p05.png`.",
            "- `metrics/30-cuboid-top-surface-laplacian-rms.png`.",
            "",
            "## Provenance and limitations",
            "",
            f"- Authoritative archived snapshot: `{ARCHIVE_ROOT}`.",
            "- The archived run used `DEBUG=1`, so it is local-only and has no Comet run.",
            "- This asset build reran no simulation, inverse, adjoint, or optimizer; it "
            "only copied identity-pinned results and rendered/plot them.",
            "- Maximum displacement changed only about `4.7%`, but normalized Laplacian "
            "was non-monotone (`180.99 → 138.80 → 141.92`).",
            "- The thicknesses were independently remeshed; this does not remove volumetric "
            "remeshing bias. Self-collision was disabled.",
            "- At `0.60`, integrated force differed by about `0.36%` across cases.",
            "- The `0.12` case had one positive tetrahedron below `detF = 0.5` "
            "(`min detF = 0.461`) but passed the documented display gates.",
            "",
            "Machine-readable identities and commands are in "
            "`../data/30-cuboid-assets-receipt.json`.",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main(cfg: Config) -> None:
    if os.environ.get("DEBUG") != "1":
        raise RuntimeError("set DEBUG=1; this historical asset build is local-only")
    if Path(cfg.input_summary).resolve() != (
        ARCHIVE_ROOT / SOURCE_FILES["summary_json"].relative_path
    ).resolve():
        raise ValueError("source summary cannot be overridden")
    if Path(cfg.output_receipt).resolve() != RECEIPT.resolve():
        raise ValueError("receipt output cannot be overridden")
    stale = [path for path in (BUNDLE_ROOT, RECEIPT, REPORT) if path.exists()]
    if stale:
        raise FileExistsError(f"refusing stale cuboid outputs: {stale}")
    if not RENDERER.is_file() or not PVBATCH.is_file():
        raise FileNotFoundError("ParaView renderer or pvbatch missing")

    pvbatch_identity_before = identity(PVBATCH)
    renderer_identity_before = identity(RENDERER)
    completed = subprocess.run(
        [str(PVBATCH), "--version"], check=True, capture_output=True, text=True
    )
    version_text = f"{completed.stdout}\n{completed.stderr}".strip()
    if not version_text.endswith(EXPECTED_PARAVIEW_VERSION):
        raise RuntimeError(f"unexpected ParaView version: {version_text}")

    source_pre = {
        label: require_source(label, spec) for label, spec in SOURCE_FILES.items()
    }
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=False)
    copied = copy_sources()
    local_summary = INPUT_ROOT / SOURCE_FILES["summary_json"].local_name
    summary = read_json(local_summary)
    rows = validate_summary(summary)
    PARAVIEW_ROOT.mkdir()
    METRICS_ROOT.mkdir()
    contract, contract_identity = prepare_contract(rows, copied)
    metric_plots = plot_metrics(rows)

    command = [
        str(PVBATCH),
        str(RENDERER.resolve()),
        "--contract",
        str(CONTRACT.resolve()),
        "--input-root",
        str(INPUT_ROOT.resolve()),
        "--output-root",
        str(PARAVIEW_ROOT.resolve()),
        "--renderer-receipt",
        str(RENDERER_RECEIPT.resolve()),
    ]
    logger.info("Running native ParaView cuboid renderer: %s", command)
    render = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=GROUP_DIR,
    )
    if render.stdout:
        logger.info("pvbatch stdout:\n%s", render.stdout)
    if render.stderr:
        logger.info("pvbatch stderr:\n%s", render.stderr)
    if render.returncode != 0:
        raise RuntimeError(f"pvbatch failed with exit code {render.returncode}")
    renderer_receipt = validate_renderer_receipt(contract)

    source_post = {
        label: require_source(label, spec) for label, spec in SOURCE_FILES.items()
    }
    if source_pre != source_post:
        raise RuntimeError("archived sources changed during asset build")
    if identity(PVBATCH) != pvbatch_identity_before:
        raise RuntimeError("pvbatch changed during asset build")
    if identity(RENDERER) != renderer_identity_before:
        raise RuntimeError("ParaView renderer changed during asset build")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "status": "ok",
        "execution_profile": "debug",
        "execution": {
            "asset_build_only": True,
            "simulation_executed": False,
            "inverse_executed": False,
            "adjoint_executed": False,
            "optimizer_executed": False,
            "native_paraview_rendering_executed": True,
            "matplotlib_metric_plots_executed": True,
            "matplotlib_metric_plot_count": 2,
        },
        "authoritative_snapshot": str(ARCHIVE_ROOT.resolve()),
        "source_run": {
            "timestamp": "2026-08-12T141945+08:00",
            "debug_local_only": True,
            "comet_run": None,
            "recorded_command": (
                "DEBUG=1 CHERRIES_NAME=fat-large-deformation-thickness "
                "CHERRIES_TAGS=fat,thickness,large-deformation,continuation "
                "/home/liblaf/github/liblaf/apple/.venv/bin/python3 "
                "src/30-run-large-deformation-fat-thickness.py --overwrite true"
            ),
        },
        "source_pre": source_pre,
        "source_post": source_post,
        "copied_sources": copied,
        "controlled_config": summary["controlled_config"],
        "pressure_0p60_cases": rows,
        "contract": contract_identity,
        "paraview": {
            "version": EXPECTED_PARAVIEW_VERSION,
            "pvbatch": pvbatch_identity_before,
            "renderer": renderer_identity_before,
            "command": command,
            "receipt": {**identity(RENDERER_RECEIPT), "payload": renderer_receipt},
        },
        "metric_plots": metric_plots,
        "meeting_asset_inventory": {
            "combined_images": False,
            "paraview_3d_pair_count": len(renderer_receipt["outputs"]),
            "paraview_3d_pairs": renderer_receipt["outputs"],
            "standalone_metric_plot_count": metric_plots["standalone_plot_count"],
            "standalone_metric_plots": metric_plots["plots"],
        },
        "authority": (
            "all 3D pixels and all PVSM states were generated by native ParaView 6.1.1; "
            "PyVista was used only for strict mesh/array readback and pooled scalar-range preparation"
        ),
    }
    write_json(cfg.output_receipt, receipt)
    write_report(rows, receipt)
    for path in sorted(path for path in BUNDLE_ROOT.rglob("*") if path.is_file()):
        cherries.log_output(path)
    cherries.log_output(REPORT)
    logger.info("Wrote cuboid asset receipt to %s", cfg.output_receipt)


if __name__ == "__main__":
    if os.environ.get("DEBUG") != "1":
        raise RuntimeError("set DEBUG=1 before Cherries starts")
    cherries.main(main)
