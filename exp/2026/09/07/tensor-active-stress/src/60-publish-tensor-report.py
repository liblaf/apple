# Copyright (c) 2026 liblaf
# ruff: noqa: PLR0915
"""Build and validate the tensor active-stress report from completed evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EARLIER = ROOT.parent / "face-actuation-diagnosis"
CASE_IDS = ("raw6", "psd", "psd-smooth", "psd-smooth-rank")
HISTORY_STEPS = (0, 16, 32, 48, 64)
RUN_DIRECTORIES = {
    "raw6": "20-raw6-reference",
    "psd": "21-psd",
    "psd-smooth": "22-psd-smooth",
    "psd-smooth-rank": "23-psd-smooth-rank",
}


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EARLIER / "src" / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PUBLISH = load("existing_diagnosis_publisher", "60-publish-diagnosis.py")
CHECK = load("existing_diagnosis_validator", "61-validate-diagnosis-artifacts.py")
PUBLISH.ROOT = ROOT
PUBLISH.APPLE_ROOT = ROOT.parents[4]
PUBLISH.LINKED_SUFFIXES |= {".log", ".txt", ".toml", ".lock"}


def require_inside_root(path: Path, label: str) -> Path:
    """Require a publication input to stay inside this experiment."""
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        message = f"{label} escapes the experiment root: {resolved}"
        raise ValueError(message) from error
    return resolved


def file_record(path: Path) -> dict[str, object]:
    """Record one immutable publication input."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": PUBLISH.file_sha256(path),
    }


def validate_inventory(directory: Path, inventory: dict, *, exact: bool) -> int:
    """Validate a relative-path output inventory against one directory."""
    assert isinstance(inventory, dict)
    assert inventory
    seen: set[str] = set()
    for raw, record in inventory.items():
        relative = Path(raw)
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        path = (directory / relative).resolve()
        path.relative_to(directory.resolve())
        assert path.is_file(), path
        assert path.stat().st_size == record["bytes"], path
        assert PUBLISH.file_sha256(path) == record["sha256"], path
        seen.add(relative.as_posix())
    if exact:
        actual = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.name != "summary.json"
        }
        assert seen == actual
    return len(seen)


def validate_render_summary(viewer: Path) -> tuple[dict, dict[str, dict], dict]:
    """Validate source 50's complete render receipt before copying its bundle."""
    summary_path = viewer / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == 1
    assert summary["status"] == "completed"
    assert summary["history_steps"] == list(HISTORY_STEPS)
    assert summary["viewer"]["fiber_data"] is None
    assert summary["viewer"]["cutaway"] is None
    case_records = summary["inputs"]["cases"]
    assert len(case_records) == len(CASE_IDS)
    receipts = {row["id"]: row for row in case_records}
    assert tuple(receipts) == CASE_IDS
    assert len(receipts) == len(CASE_IDS)
    for identity, receipt in receipts.items():
        receipt_path = viewer / f"{identity}-receipt.json"
        assert json.loads(receipt_path.read_text()) == receipt
    for name, record in (
        ("viewer.html", summary["viewer"]["html"]),
        ("viewer-manifest.json", summary["viewer"]["manifest"]),
        ("face-comparison-front.png", summary["static"]["front"]["png"]),
        ("face-comparison-front.pdf", summary["static"]["front"]["pdf"]),
        ("face-comparison-mouth.png", summary["static"]["mouth"]["png"]),
        ("face-comparison-mouth.pdf", summary["static"]["mouth"]["pdf"]),
    ):
        CHECK.validate_file_record(record, f"source50 {name}", viewer / name)
    inventory = summary["output_inventory"]
    validate_inventory(viewer, inventory, exact=True)
    return file_record(summary_path), receipts, inventory


def validate_comparison_summary(path: Path) -> dict[str, object]:
    """Require the verified source-41 comparison that the report interprets."""
    summary = json.loads(path.read_text())
    assert summary["status"] == "verified_completed_fixed_budget_comparison"
    assert tuple(row["id"] for row in summary["cases"]) == CASE_IDS
    assert len(summary["cases"]) == len(CASE_IDS)
    assert isinstance(summary["comparison_limit"], str)
    assert summary["comparison_limit"].strip()
    checks = summary["checks"]
    for name in (
        "all_states_forward_and_adjoint_valid",
        "all_arms_have_65_states_including_rest",
        "common_fixture_active_ids_zero_control_initialization_and_optimizer",
        "common_passive_materials_and_zero_skin_energy",
        "active_ids_are_exactly_positive_muscle_fraction",
        "all_tensor_endpoints_symmetric_and_in_spectral_box",
        "controls_npz_state_and_vtk_fields_are_bound",
        "surface_endpoint_hashes_match_final_meshes",
    ):
        assert checks[name] is True
    return {
        **file_record(path),
        "status": summary["status"],
        "case_ids": list(CASE_IDS),
        "comparison_limit": summary["comparison_limit"],
    }


def validate_tensor_receipts(viewer: Path, expected_receipts: dict[str, dict]) -> int:
    """Validate the tensor wrapper's explicit per-case source/history contract."""
    manifest = json.loads((viewer / "viewer-manifest.json").read_text())
    assert tuple(case["id"] for case in manifest["cases"]) == CASE_IDS
    assert manifest["fibers"] is None
    reference_paths = set()
    for case in manifest["cases"]:
        assert set(case["states"]) == {"reference", "endpoint", "target_skin"}
        assert [frame["step"] for frame in case["history"]] == list(HISTORY_STEPS)
        assert all("cutaway_file" not in frame for frame in case["history"])
        assert case["receipt"] == f"{case['id']}-receipt.json"
        receipt_path = (viewer / case["receipt"]).resolve()
        receipt_path.relative_to(viewer.resolve())
        receipt = json.loads(receipt_path.read_text())
        assert receipt == expected_receipts[case["id"]]
        assert receipt["id"] == case["id"]
        assert receipt["label"] == case["label"]
        assert receipt["deformation_scale"] == 1.0
        assert receipt["cutaway"] is None
        assert [frame["step"] for frame in receipt["history"]] == list(HISTORY_STEPS)
        run_summary = receipt["run_summary"]
        assert run_summary["status"] == "completed_fixed_budget"
        assert run_summary["primary_step"] == 64
        assert run_summary["solver_valid"] is True
        reference_path = CHECK.validate_file_record(
            receipt["reference"], f"{case['id']} reference"
        )
        endpoint_path = CHECK.validate_file_record(
            receipt["endpoint"], f"{case['id']} endpoint"
        )
        summary_path = CHECK.validate_file_record(
            run_summary["source"], f"{case['id']} summary"
        )
        reference_paths.add(reference_path.resolve())
        assert endpoint_path.name == "final.vtu"
        assert endpoint_path.parent.name == RUN_DIRECTORIES[case["id"]]
        assert summary_path == endpoint_path.parent / "summary.json"
        run_document = json.loads(Path(run_summary["source"]["path"]).read_text())
        assert run_document["status"] == "completed_fixed_budget"
        assert run_document["inverse_convergence_claimed"] is False
        assert run_document["primary_endpoint"]["step"] == 64
        assert run_document["primary_endpoint"]["solver_valid"] is True
        for frame in receipt["history"]:
            expected_stem = f"step-{frame['step']:04d}"
            vtu_path = CHECK.validate_file_record(
                frame["vtu"], f"{case['id']} {expected_stem} VTU"
            )
            npz_path = CHECK.validate_file_record(
                frame["paired_npz"], f"{case['id']} {expected_stem} paired NPZ"
            )
            assert vtu_path == endpoint_path.parent / f"{expected_stem}.vtu"
            assert npz_path == endpoint_path.parent / f"{expected_stem}.npz"
    assert len(reference_paths) == 1
    return len(manifest["cases"])


def reproduction_files(manifest: dict) -> list[Path]:
    """Use explicit scoped inputs and require every declared pattern to match."""
    paths: set[Path] = set()
    allowed = {
        ".py",
        ".pyi",
        ".typed",
        ".md",
        ".json",
        ".jsonl",
        ".csv",
        ".log",
        ".txt",
        ".css",
        ".js",
        ".toml",
        ".lock",
    }
    for pattern in manifest["reproduction_patterns"]:
        matches = [p for p in ROOT.glob(pattern) if p.is_file()]
        assert matches, pattern
        for path in matches:
            assert path.suffix in allowed, path
            path.resolve().relative_to(ROOT.resolve())
            paths.add(path)
    return sorted(paths)


def zip_paths(paths: list[Path], destination: Path) -> None:
    """Package this experiment with its own reproduction instructions."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
        archive.writestr(
            "MANIFEST.txt",
            "\n".join(path.relative_to(ROOT).as_posix() for path in paths) + "\n",
        )
        archive.writestr(
            "REPRODUCIBILITY_README.md",
            """# Tensor active-stress archive

Paths are relative to exp/2026/09/07/tensor-active-stress in the Apple repository.
MANIFEST.txt lists the included files. The figures archive provides five separate
figures in both PNG and PDF. The evidence archive provides experiment source,
protocols, configurations, traces, solver receipts, and frozen source snapshots.

Runtime versions and core Python sources are recorded under data/05-runtime/.
External surface, rendering, and publishing helpers plus supplemental .pyi and
py.typed files are recorded under data/06-helper-sources/. Their summary files
identify original paths and source hashes. The executed face source snapshots
are also stored within each of data/20-raw6-reference, data/21-psd,
data/22-psd-smooth, and data/23-psd-smooth-rank.

Rerunning a numerical experiment requires the repository checkout, the exact
large fixture/state files named in its provenance, and the recorded Python/CUDA
environment. Full VTK/NPZ arrays and optimizer checkpoints remain in the local
workspace and are not included here. The static site carries derived browser
geometry separately. See docs/60-tensor-active-stress-report.md and the verified
data/41-face-comparison/summary.json for the actual update-64 comparison and its
limits. Failed attempts are distinguished from completed runs in their records.
""",
        )


def write_records_manifest(
    path: Path,
    *,
    report: Path,
    viewer_manifest: Path,
    archives: dict,
    viewer_records: dict,
    render_summary: dict,
    comparison_summary: dict,
    comparison_staged_path: Path,
    publication_manifest: Path,
) -> None:
    """Write the self-contained publication provenance record."""
    path.write_text(
        json.dumps(
            {
                "report_markdown": report.name,
                "viewer_manifest_sha256": PUBLISH.file_sha256(viewer_manifest),
                "archives": archives,
                "viewers": viewer_records,
                "source50_render_summary": render_summary,
                "source41_comparison_summary": {
                    **comparison_summary,
                    "staged_path": comparison_staged_path.as_posix(),
                },
                "publication_inputs_sha256": PUBLISH.file_sha256(publication_manifest),
                "scope": (
                    "Three bounded PSD tensor active-stress arms and one unbounded "
                    "Raw6 historical reference at a common fixed 64-update budget, "
                    "with unchanged passive materials and small-model validation. "
                    "Full volume states and fixtures remain in the workspace."
                ),
            },
            indent=2,
        )
        + "\n"
    )


def publish(manifest_path: Path, output: Path) -> dict:
    manifest_path = require_inside_root(manifest_path, "publication manifest")
    document = json.loads(manifest_path.read_text())
    report = require_inside_root(
        manifest_path.parent / document["report"], "report Markdown"
    )
    viewer = require_inside_root(
        manifest_path.parent / document["viewer"], "source50 viewer"
    )
    comparison_path = require_inside_root(
        manifest_path.parent / document["comparison_summary_path"],
        "source41 comparison summary",
    )
    assert report.suffix.lower() == ".md"
    assert document.get("extra_viewers", []) == []
    markdown = report.read_text()
    assert "PENDING_" not in markdown
    assert "FINAL_" not in markdown
    assert not output.exists(), "Choose a new publication output directory"
    assert (viewer / "viewer.html").is_file()
    render_summary_record, expected_receipts, source50_inventory = (
        validate_render_summary(viewer)
    )
    comparison_record = validate_comparison_summary(comparison_path)
    reproduction = reproduction_files(document)
    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tensor-site-", dir=ROOT / "tmp") as temp:
        stage = Path(temp) / "site"
        shutil.copytree(viewer, stage)
        shutil.copy2(ROOT / "src/report-theme.css", stage / "report.css")
        shutil.copy2(ROOT / "src/report-theme.js", stage / "report-theme.js")
        rewritten = PUBLISH.copy_report_artifacts(markdown, report, stage)
        records = stage / "records"
        records.mkdir()
        staged_comparison = records / "source41-comparison-summary.json"
        shutil.copy2(comparison_path, staged_comparison)
        assert PUBLISH.file_sha256(staged_comparison) == comparison_record["sha256"]
        assert staged_comparison.stat().st_size == comparison_record["bytes"]
        PUBLISH.downloadable_markdown_closure(report, stage)
        figures = PUBLISH.figures(report, viewer)
        zip_paths(figures, records / "figures.zip")
        zip_paths(reproduction, records / "reproducibility.zip")
        archives = {
            name: {
                "sha256": PUBLISH.file_sha256(records / name),
                "bytes": (records / name).stat().st_size,
            }
            for name in ("figures.zip", "reproducibility.zip")
        }
        viewer_records = {
            ".": {
                "path": "viewer-manifest.json",
                "sha256": PUBLISH.file_sha256(stage / "viewer-manifest.json"),
                "bytes": (stage / "viewer-manifest.json").stat().st_size,
            }
        }
        write_records_manifest(
            records / "manifest.json",
            report=report,
            viewer_manifest=stage / "viewer-manifest.json",
            archives=archives,
            viewer_records=viewer_records,
            render_summary=render_summary_record,
            comparison_summary=comparison_record,
            comparison_staged_path=staged_comparison.relative_to(stage),
            publication_manifest=manifest_path,
        )
        body = PUBLISH.render_markdown(rewritten)
        html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tensor active stress</title><script src="report-theme.js"></script><link rel="stylesheet" href="report.css"></head>
<body><header class="site-header"><div class="site-header__inner">
	<a class="wordmark" href="index.html">Tensor active stress</a><nav>
	<a href="viewer.html">3D viewer</a>
	<a href="records/figures.zip">PNG / PDF figures</a>
<a href="records/reproducibility.zip">Scripts and evidence</a>
<a href="records/{report.name}">Markdown</a>

</nav><label class="theme-picker" for="report-theme">Theme
<select id="report-theme" disabled><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select>
</label></div></header><main class="report">{body}</main>
<footer class="site-footer">Actual saved states. Common cameras. Original geometric scale.</footer></body></html>"""
        (stage / "index.html").write_text(html)
        CHECK.validate_record_manifest(stage)
        saved_records = json.loads((records / "manifest.json").read_text())
        staged_render_summary = stage / "summary.json"
        assert staged_render_summary.stat().st_size == render_summary_record["bytes"]
        assert (
            PUBLISH.file_sha256(staged_render_summary)
            == render_summary_record["sha256"]
        )
        for record in saved_records["viewers"].values():
            path = stage / record["path"]
            assert PUBLISH.file_sha256(path) == record["sha256"]
            assert path.stat().st_size == record["bytes"]
        verification = {
            "status": "passed",
            "source50_render_summary": render_summary_record,
            "source41_comparison_summary": comparison_record,
            "viewers": {
                ".": {
                    "geometry": CHECK.validate_geometry(stage),
                    "case_receipts": validate_tensor_receipts(stage, expected_receipts),
                    "javascript_modules": CHECK.validate_modules(stage),
                    "source50_inventory_files": validate_inventory(
                        stage, source50_inventory, exact=False
                    ),
                }
            },
            "html_files": CHECK.validate_site(stage),
            "downloadable_markdown_links": CHECK.validate_staged_markdown(stage),
            "archives": {name: CHECK.validate_zip(records / name) for name in archives},
            "publication_inputs": {
                "path": str(manifest_path.resolve()),
                "sha256": PUBLISH.file_sha256(manifest_path),
            },
            "report": {"path": str(report), "sha256": PUBLISH.file_sha256(report)},
            "publisher": {
                "path": str(Path(__file__).resolve()),
                "sha256": PUBLISH.file_sha256(Path(__file__)),
            },
        }
        (stage / "validation.json").write_text(
            json.dumps(verification, indent=2) + "\n"
        )
        hashes = {
            p.relative_to(stage).as_posix(): {
                "bytes": p.stat().st_size,
                "sha256": PUBLISH.file_sha256(p),
            }
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        (stage / "file-manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/60-tensor-report-site"
    )
    args = parser.parse_args()
    print(json.dumps(publish(args.manifest.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
