# Copyright (c) 2026 liblaf
from __future__ import annotations

# Executed with ParaView 6.1.1's pvpython, not the project interpreter.
# ruff: noqa: C901, EM101, EM102, TRY003
import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import paraview.simple as pvs

EXPECTED_PARAVIEW_VERSION = "6.1.1"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_DESIGN = "exact-plane-strain-stable-neo-hookean-active-resolution-study"
PROGRESS = 0.75
VARIANTS = ("free", "tied", "regularized")
RESOLUTIONS = ((50, 5), (100, 10), (200, 20))
COLORS = {
    "free": (0.12, 0.62, 0.88),
    "tied": (0.92, 0.47, 0.14),
    "regularized": (0.22, 0.67, 0.34),
}


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"invalid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def paraview_version() -> str:
    from paraview import servermanager

    manager = servermanager.vtkSMProxyManager
    return ".".join(
        str(value)
        for value in (
            manager.GetVersionMajor(),
            manager.GetVersionMinor(),
            manager.GetVersionPatch(),
        )
    )


def read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(fields) - set(reader.fieldnames):
            raise ValueError(f"missing CSV fields in {path}: {fields}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def finite(row: dict[str, str], field: str, path: Path) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in {path}")
    return value


def validate_inputs(
    analysis_dir: Path, numerical_dir: Path
) -> tuple[Path, list[dict[str, str]], dict[str, Path]]:
    analysis_json = analysis_dir / "analysis.json"
    analysis = read_json(analysis_json)
    if analysis.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError("analysis schema changed")
    if (
        analysis.get("design") != EXPECTED_DESIGN
        or analysis.get("complete") is not True
    ):
        raise ValueError("analysis is incomplete or has a different design")
    matched = analysis_dir / "matched-data-loss-evolution.csv"
    fields = (
        "variant",
        "nx",
        "ny",
        "common_progress_fraction",
        "target_data_loss",
        "actual_data_loss",
        "relative_loss_mismatch",
        "common_grid_highpass_rms_width_0p02",
        "activation_neighbor_jump_rms",
        "min_det_f",
        "min_det_g",
        "min_det_ainv",
        "equilibrium_residual_rms",
    )
    rows = read_csv(matched, fields)
    selected = [
        row
        for row in rows
        if math.isclose(
            finite(row, "common_progress_fraction", matched), PROGRESS, abs_tol=1.0e-12
        )
    ]
    if len(selected) != len(VARIANTS) * len(RESOLUTIONS):
        raise ValueError(f"expected nine matched-loss rows at progress {PROGRESS}")
    expected = {(variant, nx, ny) for variant in VARIANTS for nx, ny in RESOLUTIONS}
    observed = {(row["variant"], int(row["nx"]), int(row["ny"])) for row in selected}
    if observed != expected:
        raise ValueError(f"unexpected matched-loss keys: {observed}")
    for row in selected:
        for field in fields[4:]:
            finite(row, field, matched)
        if (
            min(
                finite(row, field, matched)
                for field in ("min_det_f", "min_det_g", "min_det_ainv")
            )
            <= 0.0
        ):
            raise ValueError("matched-loss row is not orientation-preserving")

    tangent_paths = {
        f"{nx}x{ny}": numerical_dir / f"tangent-{nx}x{ny}-singular-values.csv"
        for nx, ny in RESOLUTIONS
    }
    combined = analysis_dir / "tangent-singular-values.csv"
    combined_rows = read_csv(combined, ("resolution", "index", "singular_value"))
    for resolution, path in tangent_paths.items():
        source_rows = read_csv(path, ("index", "singular_value", "relative_to_largest"))
        selected_rows = [
            row for row in combined_rows if row["resolution"] == resolution
        ]
        if len(selected_rows) != len(source_rows):
            raise ValueError(f"tangent length mismatch for {resolution}")
        for source, aggregate in zip(source_rows, selected_rows, strict=True):
            if int(source["index"]) != int(aggregate["index"]):
                raise ValueError(f"tangent index mismatch for {resolution}")
            value = finite(source, "singular_value", path)
            if value <= 0.0 or not math.isclose(
                value,
                finite(aggregate, "singular_value", combined),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise ValueError(f"invalid tangent singular value for {resolution}")
    return matched, selected, tangent_paths


def write_variant_csvs(rows: list[dict[str, str]], output_dir: Path) -> dict[str, Path]:
    fields = (
        "nx",
        "common_grid_highpass_rms_width_0p02",
        "activation_neighbor_jump_rms",
    )
    paths: dict[str, Path] = {}
    for variant in VARIANTS:
        path = output_dir / f"matched-data-loss-progress-0p75-{variant}.csv"
        variant_rows = sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: int(row["nx"]),
        )
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(
                {field: row[field] for field in fields} for row in variant_rows
            )
        paths[variant] = path
    return paths


def configure_chart(
    view: Any,
    *,
    title: str,
    x_title: str,
    y_title: str,
    y_range: tuple[float, float],
    log_y: bool = False,
) -> None:
    view.ChartTitle = title
    view.ChartTitleColor = [0.08, 0.10, 0.13]
    view.BottomAxisTitle = x_title
    view.LeftAxisTitle = y_title
    for property_name in (
        "BottomAxisColor",
        "BottomAxisLabelColor",
        "BottomAxisTitleColor",
        "LeftAxisColor",
        "LeftAxisLabelColor",
        "LeftAxisTitleColor",
    ):
        setattr(view, property_name, [0.08, 0.10, 0.13])
    view.BottomAxisGridColor = [0.78, 0.80, 0.83]
    view.LeftAxisGridColor = [0.78, 0.80, 0.83]
    view.ShowLegend = 1
    view.LegendLocation = "Bottom Right"
    view.ShowBottomAxisGrid = 1
    view.ShowLeftAxisGrid = 1
    view.LeftAxisLogScale = int(log_y)
    view.LeftAxisUseCustomRange = 1
    view.LeftAxisRangeMinimum = float(y_range[0])
    view.LeftAxisRangeMaximum = float(y_range[1])
    view.BottomAxisUseCustomRange = 1
    view.BottomAxisRangeMinimum = 45.0
    view.BottomAxisRangeMaximum = 205.0


def show_series(
    reader: Any,
    view: Any,
    *,
    x: str,
    y: str,
    label: str,
    color: tuple[float, float, float],
) -> None:
    display = pvs.Show(reader, view, "XYChartRepresentation")
    display.UseIndexForXAxis = 0
    display.XArrayName = x
    display.SeriesVisibility = [y, "1"]
    display.SeriesLabel = [y, label]
    display.SeriesColor = [y, *(str(value) for value in color)]
    display.SeriesLineThickness = [y, "3"]


def save_screenshot(
    path: Path, layout: Any, requested: tuple[int, int]
) -> dict[str, Any]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    pvs.SaveScreenshot(
        str(temporary),
        layout,
        ImageResolution=list(requested),
        TransparentBackground=0,
        FontScaling="Do not scale fonts",
    )
    width, height = png_size(temporary)
    if (
        width != requested[0]
        or abs(height - requested[1]) > 16
        or temporary.stat().st_size < 20_000
    ):
        raise ValueError(f"invalid screenshot: {temporary}")
    temporary.replace(path)
    return {**identity(path), "width": width, "height": height}


def save_state(path: Path) -> dict[str, Any]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    pvs.SaveState(str(temporary))
    if "ServerManagerState" not in temporary.read_text(encoding="utf-8")[:2048]:
        raise ValueError(f"invalid ParaView state: {temporary}")
    temporary.replace(path)
    return identity(path)


def render_bumpiness(
    rows: list[dict[str, str]], source: Path, output_dir: Path
) -> dict[str, Any]:
    pvs.ResetSession()
    derived = write_variant_csvs(rows, output_dir)
    highpass = [
        finite(row, "common_grid_highpass_rms_width_0p02", source) for row in rows
    ]
    jump = [finite(row, "activation_neighbor_jump_rms", source) for row in rows]
    layout = pvs.CreateLayout(name="Matched-data-loss bumpiness by resolution")
    layout.SplitVertical(0, 0.5)
    top_location = int(layout.SMProxy.GetFirstChild(0))
    bottom_location = int(layout.SMProxy.GetSecondChild(0))
    highpass_view = pvs.CreateView("XYChartView")
    jump_view = pvs.CreateView("XYChartView")
    configure_chart(
        highpass_view,
        title="Matched-data-loss bumpiness at 75% progress",
        x_title="mesh resolution n_x (n_y=n_x/10)",
        y_title="common-grid top high-pass RMS (Gaussian width = 0.02)",
        y_range=(0.0, max(highpass) * 1.10),
    )
    configure_chart(
        jump_view,
        title="Control roughness at the same matched data loss and progress",
        x_title="mesh resolution n_x (n_y=n_x/10)",
        y_title="activation-neighbor-jump RMS",
        y_range=(0.0, max(jump) * 1.10),
    )
    if not layout.AssignView(top_location, highpass_view) or not layout.AssignView(
        bottom_location, jump_view
    ):
        raise RuntimeError("failed to assign bumpiness chart views")
    resolution = (1800, 1200)
    layout.SetSize(*resolution)
    for variant in VARIANTS:
        highpass_reader = pvs.CSVReader(
            registrationName=f"{variant} high-pass at 75 percent",
            FileName=[str(derived[variant])],
        )
        highpass_reader.UpdatePipeline()
        show_series(
            highpass_reader,
            highpass_view,
            x="nx",
            y="common_grid_highpass_rms_width_0p02",
            label=variant,
            color=COLORS[variant],
        )
        jump_reader = pvs.CSVReader(
            registrationName=f"{variant} jump at 75 percent",
            FileName=[str(derived[variant])],
        )
        jump_reader.UpdatePipeline()
        show_series(
            jump_reader,
            jump_view,
            x="nx",
            y="activation_neighbor_jump_rms",
            label=variant,
            color=COLORS[variant],
        )
    pvs.Render(highpass_view)
    pvs.Render(jump_view)
    png = save_screenshot(
        output_dir / "matched-loss-bumpiness-progress-0p75.png", layout, resolution
    )
    pvsm = save_state(output_dir / "matched-loss-bumpiness-progress-0p75.pvsm")
    return {
        "source": identity(source),
        "derived_variant_csvs": {
            variant: identity(path) for variant, path in derived.items()
        },
        "highpass_range": [0.0, max(highpass) * 1.10],
        "neighbor_jump_range": [0.0, max(jump) * 1.10],
        "png": png,
        "pvsm": pvsm,
    }


def render_tangent_spectra(
    analysis_dir: Path, tangent_paths: dict[str, Path], output_dir: Path
) -> dict[str, Any]:
    pvs.ResetSession()
    all_values: list[float] = []
    maximum_index = 0
    for path in tangent_paths.values():
        for row in read_csv(path, ("index", "singular_value", "relative_to_largest")):
            all_values.append(finite(row, "singular_value", path))
            maximum_index = max(maximum_index, int(row["index"]))
    layout = pvs.CreateLayout(name="Initial tangent singular-value spectra")
    view = pvs.CreateView("XYChartView")
    if not layout.AssignView(0, view):
        raise RuntimeError("failed to assign tangent chart view")
    resolution = (1800, 850)
    layout.SetSize(*resolution)
    configure_chart(
        view,
        title="Initial tangent singular-value spectra (unregularized free-control map)",
        x_title="singular-value index (descending)",
        y_title="singular value (log scale)",
        y_range=(min(all_values) * 0.80, max(all_values) * 1.20),
        log_y=True,
    )
    view.BottomAxisRangeMinimum = 0.0
    view.BottomAxisRangeMaximum = float(maximum_index)
    labels = {"50x5": "50x5", "100x10": "100x10", "200x20": "200x20"}
    colors = {
        "50x5": (0.12, 0.62, 0.88),
        "100x10": (0.92, 0.47, 0.14),
        "200x20": (0.22, 0.67, 0.34),
    }
    for key in ("50x5", "100x10", "200x20"):
        reader = pvs.CSVReader(
            registrationName=f"initial tangent {key}",
            FileName=[str(tangent_paths[key])],
        )
        reader.UpdatePipeline()
        show_series(
            reader,
            view,
            x="index",
            y="singular_value",
            label=labels[key],
            color=colors[key],
        )
    pvs.Render(view)
    png = save_screenshot(
        output_dir / "initial-tangent-singular-value-spectra.png", layout, resolution
    )
    pvsm = save_state(output_dir / "initial-tangent-singular-value-spectra.pvsm")
    return {
        "analysis_aggregate": identity(analysis_dir / "tangent-singular-values.csv"),
        "numerical_tangent_csvs": {
            key: identity(path) for key, path in tangent_paths.items()
        },
        "singular_value_range": [min(all_values) * 0.80, max(all_values) * 1.20],
        "maximum_index": maximum_index,
        "png": png,
        "pvsm": pvsm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--numerical-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    version = paraview_version()
    if version != EXPECTED_PARAVIEW_VERSION:
        raise RuntimeError(
            f"requires ParaView {EXPECTED_PARAVIEW_VERSION}, found {version}"
        )
    analysis_dir = args.analysis_dir.resolve()
    numerical_dir = args.numerical_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory must be empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    matched, rows, tangent_paths = validate_inputs(analysis_dir, numerical_dir)
    bumpiness = render_bumpiness(rows, matched, output_dir)
    tangent = render_tangent_spectra(analysis_dir, tangent_paths, output_dir)
    write_json(
        output_dir / "render-receipt.json",
        {
            "schema_version": EXPECTED_SCHEMA_VERSION,
            "design": EXPECTED_DESIGN,
            "complete": True,
            "status": "ok",
            "paraview_version": version,
            "native_paraview_rendering": True,
            "analysis_dir": str(analysis_dir),
            "numerical_dir": str(numerical_dir),
            "progress_fraction": PROGRESS,
            "matched_data_loss_bumpiness": bumpiness,
            "initial_tangent_singular_value_spectra": tangent,
        },
    )


if __name__ == "__main__":
    main()
