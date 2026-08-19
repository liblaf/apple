# ruff: noqa: C901, EM101, EM102, FBT003, PLR0912, PLR0915, TRY003
"""ParaView-only, meeting-authoritative rendering of the four skin cases.

Run this file with ParaView 6.1.1 ``pvbatch``.  It deliberately imports neither
PyVista nor Cherries: the saved PVSM contains native ParaView readers, cell-data
coloring, camera settings, labels, and the 2 x 4 comparison layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

# This is independent from the material producer and Cherries wrapper blockers.
# Direct pvbatch invocation therefore cannot bypass the static-render review.
PARAVIEW_RENDER_APPROVED_AFTER_STATIC_REVIEW = True
APPROVAL_BLOCKER = (
    "NO-GO: ParaView material rendering awaits static review; do not execute "
    "until this source-level blocker is explicitly changed"
)

EXPECTED_SCHEMA_VERSION = 1
EXPECTED_DESIGN = "corrected-isface-four-case-selective-e000-c020-inverse-materials"
EXPECTED_CASE_ORDER = ("H0P0", "H0P1", "H1P1", "H1P0")
EXPECTED_PARAVIEW_VERSION = "6.1.1"
EXPECTED_POINTS = 15_299
EXPECTED_TRIANGLES = 29_899
FIELDS = (
    ("SkinYoungModulusMPa", "Skin Young's modulus E (MPa)", (0.0, 0.2)),
    ("StressFreeAreaRatio", "Stress-free area ratio rho", (0.4802, 1.0)),
)
CASE_LABELS = {
    "H0P0": "H0P0  baseline\nhomogeneous E, no prestrain",
    "H0P1": "H0P1  prestrain\nhomogeneous E + c020",
    "H1P1": "H1P1  combined\nselective E=0 + c020",
    "H1P0": "H1P0  softening\nselective E=0, no prestrain",
}
IMAGE_RESOLUTION = (4_000, 2_160)
BACKGROUND = (0.97, 0.97, 0.97)
TEXT_COLOR = (0.08, 0.08, 0.08)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def _validate_paths(args: argparse.Namespace) -> None:
    if not args.manifest.is_file():
        raise FileNotFoundError(f"missing material manifest: {args.manifest}")
    outputs = (args.screenshot, args.state, args.receipt)
    stale = [
        str(path)
        for path in (*outputs, *(_temporary_path(path) for path in outputs))
        if path.exists()
    ]
    if stale:
        raise FileExistsError(
            "refusing to overwrite ParaView material assets or partial files: "
            + str(stale)
        )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)


def _validate_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(manifest_path)
    expected_header = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "design": EXPECTED_DESIGN,
        "complete": True,
    }
    changed = {
        key: (manifest.get(key), expected)
        for key, expected in expected_header.items()
        if manifest.get(key) != expected
    }
    if changed:
        raise ValueError(f"material manifest header changed: {changed}")
    order = manifest.get("case_order")
    if order != list(EXPECTED_CASE_ORDER):
        raise ValueError(f"material case order changed: {order}")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != len(EXPECTED_CASE_ORDER):
        raise ValueError("material manifest must contain exactly four cases")
    cases_by_id = {row.get("case_id"): row for row in raw_cases}
    if set(cases_by_id) != set(EXPECTED_CASE_ORDER):
        raise ValueError("material manifest case identifiers changed")
    cases: list[dict[str, Any]] = []
    for case_id in EXPECTED_CASE_ORDER:
        row = cases_by_id[case_id]
        if row.get("validation") != {"ok": True, "errors": []}:
            raise ValueError(f"{case_id} material validation is not clean")
        skin = row.get("skin")
        if not isinstance(skin, dict):
            raise TypeError(f"{case_id} skin manifest is malformed")
        path = Path(str(skin.get("path")))
        identity = skin.get("file_identity")
        if not path.is_file() or not isinstance(identity, dict):
            raise FileNotFoundError(f"{case_id} material skin is unavailable: {path}")
        actual = _file_identity(path)
        expected = {
            "size_bytes": int(identity.get("size_bytes", -1)),
            "sha256": str(identity.get("sha256")),
        }
        if actual != expected:
            raise ValueError(f"{case_id} material skin identity changed")
        if (
            skin.get("points") != EXPECTED_POINTS
            or skin.get("triangles") != EXPECTED_TRIANGLES
        ):
            raise ValueError(f"{case_id} material skin dimensions changed")
        arrays = skin.get("arrays")
        if not isinstance(arrays, dict):
            raise TypeError(f"{case_id} material array contract is malformed")
        for field, _label, limits in FIELDS:
            record = arrays.get(field)
            if not isinstance(record, dict):
                raise KeyError(f"{case_id} manifest lacks cell field {field}")
            expected_record = {
                "association": "cell",
                "shape": [EXPECTED_TRIANGLES],
                "dtype": "<f8",
                "finite": True,
            }
            record_changed = {
                key: (record.get(key), value)
                for key, value in expected_record.items()
                if record.get(key) != value
            }
            if record_changed:
                raise ValueError(
                    f"{case_id} {field} array record changed: {record_changed}"
                )
            if not (
                math.isfinite(float(record.get("min")))
                and math.isfinite(float(record.get("max")))
                and limits[0] - 1.0e-14 <= float(record["min"])
                and float(record["max"]) <= limits[1] + 1.0e-14
            ):
                raise ValueError(f"{case_id} {field} escapes reviewed display range")
        cases.append(row)
    return manifest, cases


def _paraview_version() -> str:
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


def _field_names(dataset: Any) -> set[str]:
    cell_data = dataset.GetCellData()
    return {
        str(cell_data.GetArrayName(index))
        for index in range(cell_data.GetNumberOfArrays())
    }


def _fixed_camera(bounds: tuple[float, ...]) -> dict[str, Any]:
    if len(bounds) != 6 or not all(math.isfinite(float(value)) for value in bounds):
        raise ValueError(f"invalid skin bounds: {bounds}")
    center = (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )
    extents = (
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )
    span = max(extents)
    if span <= 0.0:
        raise ValueError("skin bounds have zero extent")
    return {
        "projection": "parallel",
        "view_direction": [0.0, 0.0, 1.0],
        "focal_point": [float(value) for value in center],
        "position": [center[0], center[1], center[2] + 4.0 * span],
        "view_up": [0.0, 1.0, 0.0],
        "parallel_scale": float(0.55 * max(extents[0], extents[1])),
    }


def _configure_view(view: Any, camera: dict[str, Any]) -> None:
    view.Background = list(BACKGROUND)
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = camera["focal_point"]
    view.CameraPosition = camera["position"]
    view.CameraViewUp = camera["view_up"]
    view.CameraParallelScale = camera["parallel_scale"]


def _configure_lut(field: str, limits: tuple[float, float], display: Any) -> Any:
    from paraview.simple import GetColorTransferFunction

    lut = GetColorTransferFunction(field, display, separate=True)
    lut.RGBPoints = [
        limits[0],
        0.267004,
        0.004874,
        0.329415,
        limits[1],
        0.993248,
        0.906157,
        0.143936,
    ]
    lut.ColorSpace = "Lab"
    lut.ScalarRangeInitialized = 1.0
    return lut


def _build_layout(
    cases: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    from paraview import servermanager
    from paraview.simple import (
        ColorBy,
        CreateLayout,
        CreateRenderView,
        GetScalarBar,
        Show,
        Text,
        XMLPolyDataReader,
    )

    readers: dict[str, Any] = {}
    source_rows: list[dict[str, Any]] = []
    bounds: tuple[float, ...] | None = None
    for row in cases:
        case_id = str(row["case_id"])
        path = str(row["skin"]["path"])
        reader = XMLPolyDataReader(
            registrationName=f"{case_id} skin material", FileName=[path]
        )
        reader.CellArrayStatus = [field for field, _label, _limits in FIELDS]
        reader.UpdatePipeline()
        fetched = servermanager.Fetch(reader)
        names = _field_names(fetched)
        missing = sorted({field for field, _label, _limits in FIELDS} - names)
        if missing:
            raise KeyError(
                f"{case_id} ParaView reader lacks true cell arrays: {missing}"
            )
        if (
            fetched.GetNumberOfPoints() != EXPECTED_POINTS
            or fetched.GetNumberOfCells() != EXPECTED_TRIANGLES
        ):
            raise ValueError(f"{case_id} ParaView readback dimensions changed")
        candidate_bounds = tuple(float(value) for value in fetched.GetBounds())
        if bounds is None:
            bounds = candidate_bounds
        elif any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-14)
            for left, right in zip(bounds, candidate_bounds, strict=True)
        ):
            raise ValueError(f"{case_id} bounds differ from H0P0")
        readers[case_id] = reader
        source_rows.append(
            {
                "case_id": case_id,
                "path": path,
                "file_identity": _file_identity(Path(path)),
                "cell_arrays_verified": sorted(names),
                "points": int(fetched.GetNumberOfPoints()),
                "triangles": int(fetched.GetNumberOfCells()),
            }
        )
    if bounds is None:
        raise RuntimeError("no material cases were loaded")
    camera = _fixed_camera(bounds)

    views: dict[tuple[int, int], Any] = {}
    displays: dict[tuple[int, int], Any] = {}
    for column, case_id in enumerate(EXPECTED_CASE_ORDER):
        for row_index, (field, row_label, limits) in enumerate(FIELDS):
            view = CreateRenderView()
            _configure_view(view, camera)
            display = Show(readers[case_id], view, "GeometryRepresentation")
            display.Representation = "Surface"
            display.InterpolateScalarsBeforeMapping = 0
            display.Ambient = 0.35
            display.Diffuse = 0.65
            ColorBy(display, ("CELLS", field))
            lut = _configure_lut(field, limits, display)
            display.LookupTable = lut
            text = Text(
                registrationName=f"{case_id} {field} label",
                Text=f"{CASE_LABELS[case_id]}\n{row_label}",
            )
            text_display = Show(text, view, "TextSourceRepresentation")
            text_display.WindowLocation = "Upper Center"
            text_display.Color = list(TEXT_COLOR)
            text_display.FontSize = 18
            text_display.Bold = 1
            if column == len(EXPECTED_CASE_ORDER) - 1:
                display.SetScalarBarVisibility(view, True)
                scalar_bar = GetScalarBar(lut, view)
                scalar_bar.Title = "E" if field == "SkinYoungModulusMPa" else "rho"
                scalar_bar.ComponentTitle = (
                    "MPa" if field == "SkinYoungModulusMPa" else ""
                )
                scalar_bar.TitleColor = list(TEXT_COLOR)
                scalar_bar.LabelColor = list(TEXT_COLOR)
                scalar_bar.TitleFontSize = 16
                scalar_bar.LabelFontSize = 14
                scalar_bar.ScalarBarLength = 0.32
                scalar_bar.WindowLocation = "Any Location"
                scalar_bar.Position = [0.82, 0.08]
            views[(row_index, column)] = view
            displays[(row_index, column)] = display

    layout = CreateLayout(name="Four-case skin materials (ParaView 6.1.1)")
    layout.SplitHorizontal(0, 0.25)
    column_0 = layout.GetFirstChild(0)
    remaining_123 = layout.GetSecondChild(0)
    layout.SplitHorizontal(remaining_123, 1.0 / 3.0)
    column_1 = layout.GetFirstChild(remaining_123)
    remaining_23 = layout.GetSecondChild(remaining_123)
    layout.SplitHorizontal(remaining_23, 0.5)
    column_2 = layout.GetFirstChild(remaining_23)
    column_3 = layout.GetSecondChild(remaining_23)
    for column, column_cell in enumerate((column_0, column_1, column_2, column_3)):
        layout.SplitVertical(column_cell, 0.5)
        top_location = layout.GetFirstChild(column_cell)
        bottom_location = layout.GetSecondChild(column_cell)
        layout.AssignView(top_location, views[(0, column)])
        layout.AssignView(bottom_location, views[(1, column)])
    layout.SetSize(*IMAGE_RESOLUTION)
    return layout, camera, source_rows


def main() -> None:
    if not PARAVIEW_RENDER_APPROVED_AFTER_STATIC_REVIEW:
        raise RuntimeError(APPROVAL_BLOCKER)
    args = _parse_args()
    _validate_paths(args)
    manifest_identity_before = _file_identity(args.manifest)
    _manifest, cases = _validate_manifest(args.manifest)
    version = _paraview_version()
    if version != EXPECTED_PARAVIEW_VERSION:
        raise RuntimeError(
            f"ParaView version changed: {version} != {EXPECTED_PARAVIEW_VERSION}"
        )

    from paraview.simple import RenderAllViews, SaveScreenshot, SaveState

    layout, camera, sources = _build_layout(cases)
    RenderAllViews()
    screenshot_temporary = _temporary_path(args.screenshot)
    state_temporary = _temporary_path(args.state)
    SaveScreenshot(
        str(screenshot_temporary),
        layout,
        ImageResolution=list(IMAGE_RESOLUTION),
        TransparentBackground=0,
        FontScaling="Scale fonts proportionally",
    )
    SaveState(str(state_temporary))
    if not screenshot_temporary.is_file() or screenshot_temporary.stat().st_size == 0:
        raise RuntimeError("ParaView did not produce a non-empty screenshot")
    if not state_temporary.is_file() or state_temporary.stat().st_size == 0:
        raise RuntimeError("ParaView did not produce a non-empty state")
    screenshot_temporary.replace(args.screenshot)
    state_temporary.replace(args.state)
    manifest_identity_after = _file_identity(args.manifest)
    if manifest_identity_after != manifest_identity_before:
        raise RuntimeError("material manifest changed during ParaView rendering")
    source_recheck = []
    for row in sources:
        actual = _file_identity(Path(row["path"]))
        if actual != row["file_identity"]:
            raise RuntimeError(f"{row['case_id']} skin changed during rendering")
        source_recheck.append({**row, "unchanged": True})
    receipt = {
        "schema_version": 1,
        "design": "paraview-6.1.1-four-case-skin-material-sheet",
        "complete": True,
        "paraview_version": version,
        "manifest": {"path": str(args.manifest), **manifest_identity_after},
        "case_order": list(EXPECTED_CASE_ORDER),
        "layout": "2 rows (E, stress-free area ratio) x 4 cases",
        "color_association": "CELLS",
        "fields": [
            {"name": name, "label": label, "fixed_range": list(limits)}
            for name, label, limits in FIELDS
        ],
        "camera": camera,
        "image_resolution": list(IMAGE_RESOLUTION),
        "sources": source_recheck,
        "outputs": {
            "screenshot": {
                "path": str(args.screenshot),
                **_file_identity(args.screenshot),
            },
            "state": {"path": str(args.state), **_file_identity(args.state)},
        },
        "meeting_authority": (
            "native ParaView 6.1.1 XMLPolyDataReader; both rows use true cell-data "
            "arrays with fixed ranges and identical parallel-projection cameras"
        ),
    }
    _write_json_atomic(args.receipt, receipt)
    if _read_json(args.receipt) != receipt:
        raise RuntimeError("ParaView render receipt strict readback failed")


if __name__ == "__main__":
    main()
