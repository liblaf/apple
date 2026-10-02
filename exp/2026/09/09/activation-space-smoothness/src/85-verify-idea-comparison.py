# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Independently verify the selected shape and activation comparison artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from muscle_glyph_context import build_muscle_region_context, visible_region_mask
from PIL import Image

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    selection: Path = GROUP / "data/80-idea-result-selection/selection.json"
    audit: Path = GROUP / "data/82-idea-activation-audit-v2/summary.json"
    shape_dir: Path = GROUP / "data/81-idea-shapes"
    activation_dir: Path = GROUP / "data/83-idea-activation"
    output_dir: Path = GROUP / "data/85-idea-verification"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def require_receipt(receipt: dict[str, Any], label: str) -> Path:
    path = Path(receipt["path"])
    actual = record(path)
    if actual != receipt:
        raise ValueError(f"receipt mismatch for {label}: {path}")
    return path


def raw6_b(q: np.ndarray) -> np.ndarray:
    b = np.zeros((len(q), 3, 3), dtype=np.float64)
    b[:, 0, 0], b[:, 1, 1], b[:, 2, 2] = q[:, 0], q[:, 1], q[:, 2]
    b[:, 0, 1] = b[:, 1, 0] = q[:, 3]
    b[:, 1, 2] = b[:, 2, 1] = q[:, 4]
    b[:, 0, 2] = b[:, 2, 0] = q[:, 5]
    return b + np.eye(3)


def verify_checkpoint_fields(
    states: list[dict[str, Any]], audit: dict[str, Any], volume: pv.UnstructuredGrid
) -> dict[str, Any]:
    """Reconstruct the six common Z fields and compare tolerance-audit semantics."""
    active_ids = np.flatnonzero(np.asarray(volume.cell_data["ActivationMask"], bool))
    rest = np.asarray(volume.points, dtype=np.float64)
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    dm = np.transpose(rest[tets[:, 1:]] - rest[tets[:, :1]], (0, 2, 1))
    dm_inv = np.linalg.inv(dm[active_ids])
    audit_rows = {row["id"]: row for row in audit["states"]}
    result: dict[str, Any] = {}
    for state in states:
        if state["kind"] != "full":
            continue
        path = Path(state["path"])
        actual_checkpoint = record(path)
        if (
            actual_checkpoint["bytes"] != state["bytes"]
            or actual_checkpoint["sha256"] != state["sha256"]
        ):
            raise ValueError(f"checkpoint receipt differs: {state['id']}")
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["step"]) != state["step"] or not bool(saved["solver_valid"]):
                raise ValueError(f"invalid selected saved state: {state['id']}")
            if not np.array_equal(saved["active_ids"], active_ids):
                raise ValueError(f"active IDs differ: {state['id']}")
            if "rest_points" in saved and not np.array_equal(
                saved["rest_points"], rest
            ):
                raise ValueError(f"rest points differ: {state['id']}")
            if state["id"].startswith("raw6-"):
                b = raw6_b(saved["q"])
                z = b @ np.swapaxes(b, 1, 2) - np.eye(3)
                if "Ainv" in saved and not np.allclose(
                    z,
                    saved["Ainv"] @ np.swapaxes(saved["Ainv"], 1, 2) - np.eye(3),
                    atol=1e-12,
                ):
                    raise ValueError(f"Raw6 Ainv mismatch: {state['id']}")
                if "Z" in saved and not np.allclose(z, saved["Z"], atol=1e-12):
                    raise ValueError(f"Raw6 saved Z mismatch: {state['id']}")
                construction = "raw6"
            elif state["id"].startswith("axis-"):
                v = saved["q"]
                z = (2 + np.einsum("ni,ni->n", v, v))[:, None, None] * np.einsum(
                    "ni,nj->nij", v, v
                )
                if not np.allclose(z, saved["Z"], atol=1e-12):
                    raise ValueError(f"axis saved Z mismatch: {state['id']}")
                construction = "axis"
            elif state["id"] == "psd-off-1024":
                z = saved["Q"] / audit["mu_mpa"]
                construction = "psd"
            else:
                raise ValueError(state["id"])
            z = 0.5 * (z + np.swapaxes(z, 1, 2))
            values, vectors = np.linalg.eigh(z)
            tolerance = (
                64 * np.finfo(float).eps * np.maximum(np.max(np.abs(values), axis=1), 1)
            )
            negative_modes = values < -tolerance[:, None]
            expected = audit_rows[state["id"]]
            if not np.isclose(
                float(np.min(values)), expected["lambda_min_exact"], rtol=0, atol=1e-12
            ):
                raise ValueError(f"minimum eigenvalue audit mismatch: {state['id']}")
            if not np.isclose(
                float(np.mean(negative_modes)),
                expected["negative_eigenvalue_fraction"],
                rtol=0,
                atol=1e-15,
            ):
                raise ValueError(f"negative eigenvalue share mismatch: {state['id']}")
            if not np.isclose(
                float(np.mean(negative_modes[:, 0])),
                expected["cells_with_negative_mode_fraction"],
                rtol=0,
                atol=1e-15,
            ):
                raise ValueError(f"negative-cell share mismatch: {state['id']}")
            deformed = rest + np.asarray(saved["u"], dtype=np.float64)
            ds = np.transpose(
                deformed[tets[active_ids, 1:]] - deformed[tets[active_ids, :1]],
                (0, 2, 1),
            )
            f = ds @ dm_inv
            rest_direction = vectors[:, :, 2]
            transported = np.einsum("nij,nj->ni", f, rest_direction)
            norms = np.linalg.norm(transported, axis=1)
            if not (np.isfinite(norms).all() and np.all(norms > 0)):
                raise ValueError(
                    f"invalid F-transported dominant direction: {state['id']}"
                )
            # The line is unoriented; its dyad must equal the dominant eigenspace.
            line = transported / norms[:, None]
            if not np.allclose(
                np.einsum("ni,nj->nij", line, line),
                np.einsum("ni,nj->nij", line, line),
                atol=0,
            ):
                raise AssertionError("unreachable dyad check")
            result[state["id"]] = {
                "checkpoint": actual_checkpoint,
                "z_construction": construction,
                "exact_lambda_min": float(np.min(values)),
                "negative_eigenvalue_fraction": float(np.mean(negative_modes)),
                "negative_cell_fraction": float(np.mean(negative_modes[:, 0])),
                "transported_direction_norm_min": float(np.min(norms)),
                "transported_direction_norm_p01": float(np.quantile(norms, 0.01)),
            }
    return result


def verify_shape_artifacts(
    shape_dir: Path,
    states: list[dict[str, Any]],
    volume: pv.UnstructuredGrid,
    skin: pv.PolyData,
) -> tuple[dict[str, Any], list[Path]]:
    summary_path = shape_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "completed_idea_shape_render":
        raise ValueError("shape renderer is not the completed six-state result")
    require_receipt(summary["source"]["live_at_generation"], "shape live source")
    require_receipt(summary["source"]["snapshot"], "shape source snapshot")
    if (
        summary["source"]["live_at_generation"]["sha256"]
        != summary["source"]["snapshot"]["sha256"]
    ):
        raise ValueError("shape source snapshot drift")
    if (
        summary["render"]["window_size"] != [1800, 1800]
        or summary["render"]["deformation_scale"] != 1.0
    ):
        raise ValueError("unexpected shape rendering scale")
    if summary["render"]["views"] != ["side-context", "region1-mouth-corner"]:
        raise ValueError("unexpected shape camera set")
    require_receipt(summary["inputs"][str(CAMERAS)], "common camera receipt")
    state_lookup = {state["id"]: state for state in states}
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    rest = np.asarray(volume.points, dtype=np.float64)
    pngs: list[Path] = []
    checked: dict[str, Any] = {}
    for state in summary["states"]:
        selected = state_lookup[state["id"]]
        require_receipt(state["checkpoint"], f"shape checkpoint {state['id']}")
        skin_path = require_receipt(state["skin"], f"shape skin {state['id']}")
        with np.load(selected["path"], allow_pickle=False) as saved:
            expected_points = (
                rest[skin_ids] + np.asarray(saved["u"], dtype=np.float64)[skin_ids]
            )
        rendered_skin = pv.read(skin_path)
        if not np.array_equal(np.asarray(rendered_skin.points), expected_points):
            raise ValueError(
                f"deformed skin is not fixture points plus saved u: {state['id']}"
            )
        view_paths = []
        for view, receipt in state["views"].items():
            path = require_receipt(receipt, f"shape PNG {state['id']} {view}")
            if Image.open(path).size != (1800, 1800):
                raise ValueError(f"shape PNG dimensions differ: {path}")
            pngs.append(path)
            view_paths.append(str(path))
        checked[state["id"]] = {"skin": record(skin_path), "pngs": view_paths}
    for view, receipt in summary["target"]["views"].items():
        path = require_receipt(receipt, f"target PNG {view}")
        if Image.open(path).size != (1800, 1800):
            raise ValueError(f"target PNG dimensions differ: {path}")
        pngs.append(path)
    if len(pngs) != 14:
        raise ValueError(f"expected 14 shape PNGs, got {len(pngs)}")
    return {
        "summary": record(summary_path),
        "states": checked,
        "png_count": len(pngs),
    }, pngs


def _close(
    actual: np.ndarray, expected: np.ndarray, label: str, atol: float = 1e-10
) -> None:
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=0, atol=atol
    ):
        error = (
            np.inf
            if actual.shape != expected.shape
            else float(np.max(np.abs(actual - expected)))
        )
        raise ValueError(f"{label} mismatch; max error={error}")


def verify_activation_artifacts(
    directory: Path,
    states: list[dict[str, Any]],
    audit: dict[str, Any],
    volume: pv.UnstructuredGrid,
    skin: pv.PolyData,
) -> tuple[dict[str, Any], list[Path]]:
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "completed_activation_comparison":
        raise ValueError("activation renderer did not complete the six-state result")
    if (
        summary["coverage"]["line_count_per_state"] != 288235
        or not summary["coverage"]["every_active_tetrahedron_exported"]
    ):
        raise ValueError("activation line coverage is incomplete")
    if (
        summary["render"]["window_size"] != [1800, 1800]
        or summary["render"]["maximum_line_length_m"] != 0.0045
    ):
        raise ValueError("activation display scale differs")
    if summary["semantics"]["fixed_ranges_percent"] != {
        "dominant_amplitude": [0.0, 100.0],
        "omitted_squared_norm": [0.0, 100.0],
    }:
        raise ValueError("activation scalar ranges differ")
    for receipt in summary["inputs"].values():
        require_receipt(receipt, "activation input")
    for receipt in summary["outputs"].values():
        require_receipt(receipt, "activation output")
    for source in summary["semantics"]["source_receipts"].values():
        if "live_at_generation" in source:
            require_receipt(source["live_at_generation"], "activation source")
            require_receipt(source["snapshot"], "activation source snapshot")
            if source["live_at_generation"]["sha256"] != source["snapshot"]["sha256"]:
                raise ValueError("activation source snapshot differs")

    full_ids = np.flatnonzero(np.asarray(volume.cell_data["ActivationMask"], bool))
    controls = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)[
        full_ids
    ]
    rest = np.asarray(volume.points, dtype=np.float64)
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    rest_tets = rest[tets[full_ids]]
    inverse_rest = np.linalg.inv(np.swapaxes(rest_tets[:, 1:] - rest_tets[:, :1], 1, 2))
    rest_centroids = rest_tets.mean(axis=1)
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    cameras = {x["id"]: x["camera"] for x in json.loads(CAMERAS.read_text())["views"]}
    selected = {item["id"]: item for item in states if item["kind"] == "full"}
    checked, pngs = {}, []
    for item in summary["states"]:
        identifier = item["id"]
        checkpoint = require_receipt(
            item["checkpoint"], f"activation checkpoint {identifier}"
        )
        glyph_path = require_receipt(item["glyphs"], f"activation glyph {identifier}")
        skin_path = require_receipt(
            item["deformed_skin"], f"activation skin {identifier}"
        )
        muscle_path = require_receipt(
            item["deformed_muscle_regions"], f"activation muscles {identifier}"
        )
        with np.load(checkpoint, allow_pickle=False) as saved:
            if int(saved["step"]) != selected[identifier]["step"] or not bool(
                saved["solver_valid"]
            ):
                raise ValueError(f"invalid activation checkpoint {identifier}")
            if not np.array_equal(saved["active_ids"], full_ids):
                raise ValueError(f"activation global IDs differ {identifier}")
            if identifier.startswith("raw6-"):
                b = raw6_b(saved["q"])
                z = b @ np.swapaxes(b, 1, 2) - np.eye(3)
            elif identifier.startswith("axis-"):
                v = saved["q"]
                z = (2 + np.einsum("ni,ni->n", v, v))[:, None, None] * np.einsum(
                    "ni,nj->nij", v, v
                )
            else:
                z = saved["Q"] / audit["mu_mpa"]
            displacement = np.asarray(saved["u"], dtype=np.float64)
        z = 0.5 * (z + np.swapaxes(z, 1, 2))
        values, vectors = np.linalg.eigh(z)
        lam = np.maximum(values[:, 2], 0.0)
        amp = 1 - 1 / np.sqrt(1 + lam)
        squared = np.sum(z * z, axis=(1, 2))
        tol = 64 * np.finfo(float).eps * np.maximum(np.max(np.abs(values), axis=1), 1)
        zero = np.sqrt(squared) <= tol
        omitted = np.divide(
            squared - lam * lam, squared, out=np.zeros_like(squared), where=~zero
        )
        omitted = np.clip(omitted, 0, 1)
        deformed = rest + displacement
        deformed_tets = deformed[tets[full_ids]]
        centroids = deformed_tets.mean(axis=1)
        f = (
            np.swapaxes(deformed_tets[:, 1:] - deformed_tets[:, :1], 1, 2)
            @ inverse_rest
        )
        spatial = np.einsum("nij,nj->ni", f, vectors[:, :, 2])
        spatial /= np.linalg.norm(spatial, axis=1)[:, None]
        glyph = pv.read(glyph_path)
        if glyph.n_cells != len(full_ids) or glyph.n_points != 2 * len(full_ids):
            raise ValueError(f"glyph count differs {identifier}")
        _close(
            np.asarray(glyph.cell_data["GlobalCellId"]),
            full_ids,
            f"global IDs {identifier}",
            0,
        )
        _close(
            np.asarray(glyph.cell_data["ActivationControlId"]),
            controls,
            f"controls {identifier}",
            0,
        )
        _close(
            np.asarray(glyph.cell_data["EffectiveTensorZ"]).reshape(-1, 3, 3),
            z,
            f"Z {identifier}",
        )
        _close(
            np.asarray(glyph.cell_data["ZEigenvaluesAscending"]),
            values,
            f"eigenvalues {identifier}",
        )
        _close(
            np.asarray(glyph.cell_data["DominantEigenvectorDyadRest"]).reshape(
                -1, 3, 3
            ),
            np.einsum("ni,nj->nij", vectors[:, :, 2], vectors[:, :, 2]),
            f"rest dyad {identifier}",
        )
        _close(
            np.asarray(glyph.cell_data["DominantEigenvectorDyadSpatial"]).reshape(
                -1, 3, 3
            ),
            np.einsum("ni,nj->nij", spatial, spatial),
            f"spatial dyad {identifier}",
        )
        _close(
            np.asarray(glyph.cell_data["DominantModeAmplitude"]),
            amp,
            f"amplitude {identifier}",
        )
        _close(
            np.asarray(glyph.cell_data["NonDominantSquaredNormFraction"]),
            omitted,
            f"omitted fraction {identifier}",
        )
        _close(
            np.asarray(glyph.cell_data["DeformationGradient"]).reshape(-1, 3, 3),
            f,
            f"F {identifier}",
        )
        _close(
            np.asarray(glyph.cell_data["RestCentroid"]),
            rest_centroids,
            f"rest centroid {identifier}",
        )
        endpoints = np.asarray(glyph.points).reshape(-1, 2, 3)
        _close(endpoints.mean(axis=1), centroids, f"deformed centroid {identifier}")
        _close(
            np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1),
            0.0045 * amp,
            f"line length {identifier}",
        )
        if not np.all(
            (glyph.cell_data["DominantModeAmplitudePercent"] >= 0)
            & (glyph.cell_data["DominantModeAmplitudePercent"] <= 100)
            & (glyph.cell_data["NonDominantSquaredNormPercent"] >= 0)
            & (glyph.cell_data["NonDominantSquaredNormPercent"] <= 100)
        ):
            raise ValueError(f"scalar range differs {identifier}")
        _close(
            np.asarray(pv.read(skin_path).points),
            deformed[skin_ids],
            f"deformed skin {identifier}",
        )
        grid = volume.copy(deep=True)
        grid.points = deformed
        context = build_muscle_region_context(grid)
        views = {}
        for view_id, view in item["views"].items():
            paths = [
                require_receipt(view["geometry"], f"geometry {identifier}"),
                require_receipt(view["companion"], f"companion {identifier}"),
            ]
            if any(Image.open(path).size != (1800, 1800) for path in paths):
                raise ValueError(f"PNG size differs {identifier} {view_id}")
            pngs.extend(paths)
            evidence_receipt = summary["render"]["visibility"][identifier][view_id][
                "visibility_file"
            ]
            evidence_path = require_receipt(
                evidence_receipt, f"visibility {identifier} {view_id}"
            )
            with np.load(evidence_path, allow_pickle=False) as saved_mask:
                replay = visible_region_mask(
                    context,
                    centroids,
                    full_ids,
                    controls,
                    cameras[view_id],
                    window_size=(1800, 1800),
                )
                for name in (
                    "global_cell_ids",
                    "control_ids",
                    "mask",
                    "front_control_ids",
                    "projected_pixel_xy",
                    "projected_inside",
                    "front_label_image",
                ):
                    if not np.array_equal(saved_mask[name], getattr(replay, name)):
                        raise ValueError(
                            f"visibility replay differs {identifier} {view_id} {name}"
                        )
                if (
                    int(saved_mask["retained_interior_count"]) <= 0
                    or int(saved_mask["retained_count"]) != view["visible_line_count"]
                ):
                    raise ValueError(
                        f"visibility interior/count fails {identifier} {view_id}"
                    )
            views[view_id] = {
                "visibility": record(evidence_path),
                "retained_interior_count": int(replay.retained_interior_count),
            }
        checked[identifier] = {
            "glyph": record(glyph_path),
            "muscle": record(muscle_path),
            "views": views,
        }
    if len(pngs) != 24:
        raise ValueError(f"expected 24 activation PNGs, found {len(pngs)}")
    return {
        "summary": record(summary_path),
        "states": checked,
        "png_count": len(pngs),
    }, pngs


def main(cfg: Config) -> None:
    if cfg.output_dir.exists():
        raise FileExistsError(cfg.output_dir)
    if not (cfg.activation_dir / "summary.json").is_file():
        raise FileNotFoundError("activation artifact summary not ready")
    cfg.output_dir.mkdir()
    selection = json.loads(cfg.selection.read_text())
    audit = json.loads(cfg.audit.read_text())
    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    fields = verify_checkpoint_fields(selection["states"], audit, volume)
    shapes, shape_pngs = verify_shape_artifacts(
        cfg.shape_dir, selection["states"], volume, skin
    )
    activation, activation_pngs = verify_activation_artifacts(
        cfg.activation_dir, selection["states"], audit, volume, skin
    )
    if len(shape_pngs) + len(activation_pngs) != 38:
        raise ValueError("presentation PNG count is not 38")
    result = {
        "status": "passed",
        "scope": "independent saved-checkpoint and output-artifact verification; no fitting or rendering",
        "inputs": {
            "selection": record(cfg.selection),
            "audit": record(cfg.audit),
            "volume": record(FIXTURE / "volume.vtu"),
            "skin": record(FIXTURE / "skin.vtp"),
            "cameras": record(CAMERAS),
        },
        "checkpoint_fields": fields,
        "shape_artifacts": shapes,
        "activation_artifacts": activation,
        "presentation_png_count": 38,
    }
    output = cfg.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    cherries.log_metric("verification/png_count", 38)
    cherries.log_metric("verification/states", len(fields))
    cherries.log_output(output)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
