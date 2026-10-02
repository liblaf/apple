# Copyright (c) 2026 liblaf
# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, PT018, TRY003, TRY004
"""Stage a validated refined optimizer-continuation report; never deploy it."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EARLIER = ROOT.parent / "face-actuation-diagnosis"


def load(name: str, filename: str) -> Any:
    """Import the established generic publishing helper."""
    spec = importlib.util.spec_from_file_location(name, EARLIER / "src" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PUBLISH = load("optimizer_publisher_helpers", "60-publish-diagnosis.py")
CHECK = load("optimizer_validator_helpers", "61-validate-diagnosis-artifacts.py")
PUBLISH.ROOT = ROOT
PUBLISH.APPLE_ROOT = ROOT.parents[4]
PUBLISH.LINKED_SUFFIXES |= {".jsonl", ".log", ".txt", ".toml", ".lock"}


def require_inside_root(path: Path, label: str) -> Path:
    """Reject a publication input outside this experiment."""
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes experiment root: {resolved}") from error
    return resolved


def record(path: Path) -> dict[str, object]:
    """Return a byte-exact file record."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": PUBLISH.file_sha256(path),
    }


def validate_record(value: object, label: str, expected: Path | None = None) -> Path:
    """Validate a file record, optionally binding it to one named path."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} needs a file record")
    path = Path(value.get("path", "")).resolve()
    require_inside_root(path, label)
    if expected is not None and path != expected.resolve():
        raise ValueError(f"{label} path differs: {path}")
    CHECK.validate_file_record(value, label, path)
    return path


def validate_inventory(directory: Path, inventory: object, *, exact: bool) -> int:
    """Validate a relative output inventory, optionally requiring closure."""
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("nonempty output inventory required")
    seen: set[str] = set()
    for name, value in inventory.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe inventory path: {name}")
        path = (directory / relative).resolve()
        path.relative_to(directory.resolve())
        CHECK.validate_file_record(
            {"path": str(path), **value}, f"inventory {name}", path
        )
        seen.add(relative.as_posix())
    if exact:
        actual = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.name != "summary.json"
        }
        if seen != actual:
            raise ValueError("render output inventory is not exact")
    return len(seen)


def pinned(
    manifest: Path, value: object, label: str, suffix: str | None = None
) -> Path:
    """Resolve and byte-check one renderer manifest input."""
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise TypeError(f"{label} needs a pinned path")
    path = require_inside_root((manifest.parent / value["path"]).resolve(), label)
    if suffix is not None and path.suffix != suffix:
        raise ValueError(f"{label} needs {suffix}: {path}")
    if "latest" in path.name.lower():
        raise ValueError(f"{label} must not use moving latest artifact")
    if PUBLISH.file_sha256(path) != value.get("sha256"):
        raise ValueError(f"{label} hash differs: {path}")
    return path


def validate_comparison(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Require completed source80 results and return endpoint proof records."""
    summary = json.loads(path.read_text())
    if summary.get("status") != "completed_postprocessing":
        raise ValueError("source80 comparison is not completed_postprocessing")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("source80 comparison lacks artifacts")
    for name, value in artifacts.items():
        validate_record(value, f"source80 artifact {name}")
    endpoints = summary.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("source80 comparison lacks endpoints")
    by_id = {row.get("id"): row for row in endpoints if isinstance(row, dict)}
    if len(by_id) != len(endpoints) or None in by_id:
        raise ValueError("source80 endpoint IDs are not unique")
    for identity, endpoint in by_id.items():
        files = endpoint.get("files")
        history = endpoint.get("final_history")
        if not isinstance(files, dict) or not isinstance(history, dict):
            raise ValueError(f"source80 endpoint lacks source hashes: {identity}")
        for name, value in files.items():
            validate_record(value, f"source80 {identity} file {name}")
        for name, value in history.items():
            validate_record(value, f"source80 {identity} final history {name}")
    return summary, by_id


def validate_renderer(viewer: Path, comparison_path: Path) -> tuple[dict, dict, int]:
    """Validate source85's complete inventory, manifest, receipts, and proof."""
    summary_path = viewer / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("schema_version") != 1 or summary.get("status") != "completed":
        raise ValueError("source85 render is not completed")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("source85 render lacks inputs")
    manifest_path = validate_record(inputs.get("manifest"), "source85 manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("source85 manifest schema differs")
    if not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise ValueError("source85 manifest lacks cases")
    pinned(
        manifest_path, manifest.get("reference", {}).get("vtu"), "reference VTU", ".vtu"
    )
    pinned(
        manifest_path, manifest.get("reference", {}).get("npz"), "reference NPZ", ".npz"
    )
    source80 = pinned(
        manifest_path,
        manifest.get("verified_comparison", {}).get("summary_json"),
        "source80 summary",
        ".json",
    )
    if source80 != comparison_path:
        raise ValueError("source85 does not bind the requested source80 summary")
    comparison, endpoints = validate_comparison(comparison_path)
    receipts = inputs.get("cases")
    if not isinstance(receipts, list) or len(receipts) != len(manifest["cases"]):
        raise ValueError("source85 receipt count differs from manifest")
    receipt_by_id = {
        item.get("id"): item for item in receipts if isinstance(item, dict)
    }
    if len(receipt_by_id) != len(manifest["cases"]) or None in receipt_by_id:
        raise ValueError("source85 receipts lack unique IDs")
    for case in manifest["cases"]:
        identity = case.get("id")
        receipt = receipt_by_id.get(identity)
        if not isinstance(identity, str) or not isinstance(receipt, dict):
            raise ValueError("source85 case/receipt identity differs")
        receipt_path = viewer / f"{identity}-receipt.json"
        if (
            not receipt_path.is_file()
            or json.loads(receipt_path.read_text()) != receipt
        ):
            raise ValueError(f"source85 receipt differs: {identity}")
        endpoint = case.get("endpoint", {})
        for key, suffix in (("npz", ".npz"), ("vtu", ".vtu")):
            path = pinned(
                manifest_path, endpoint.get(key), f"{identity} endpoint {key}", suffix
            )
            saved = receipt.get("endpoint", {}).get(key)
            validate_record(saved, f"{identity} receipt endpoint {key}", path)
        history = case.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"source85 history missing: {identity}")
        if len(receipt.get("history", [])) != len(history):
            raise ValueError(f"source85 history receipt differs: {identity}")
        for declared, saved in zip(history, receipt["history"], strict=True):
            if declared.get("step") != saved.get("step"):
                raise ValueError(f"source85 history step differs: {identity}")
            for key, suffix in (("npz", ".npz"), ("vtu", ".vtu")):
                path = pinned(
                    manifest_path,
                    declared.get(key),
                    f"{identity} history {key}",
                    suffix,
                )
                validate_record(
                    saved.get(key), f"{identity} receipt history {key}", path
                )
        reference = validate_record(
            receipt.get("reference"), f"{identity} receipt reference"
        )
        expected_reference = validate_record(
            inputs.get("reference", {}).get("vtu"), "source85 reference VTU"
        )
        if reference != expected_reference:
            raise ValueError(f"source85 receipt reference differs: {identity}")
        if case.get("kind") == "completed_endpoint":
            endpoint_id = case.get("source80_endpoint_id")
            matched = endpoints.get(endpoint_id)
            proof = receipt.get("proof", {})
            if (
                not isinstance(matched, dict)
                or proof.get("endpoint_id") != endpoint_id
                or proof.get("summary", {}).get("sha256")
                != PUBLISH.file_sha256(comparison_path)
            ):
                raise ValueError(f"source80 proof missing for {identity}")
            files = matched.get("files", {})
            for name, key in (("final.npz", "npz"), ("final.vtu", "vtu")):
                if (
                    files.get(name, {}).get("sha256")
                    != receipt["endpoint"][key]["sha256"]
                ):
                    raise ValueError(
                        f"source80 endpoint proof differs: {identity} {name}"
                    )
        elif case.get("kind") == "parent_or_intermediate":
            proof = receipt.get("proof", {})
            validate_record(proof.get("summary"), f"{identity} original summary")
            validate_record(proof.get("trace"), f"{identity} original trace")
        else:
            raise ValueError(f"unsupported source85 case kind: {case.get('kind')}")
    viewer_manifest = viewer / "viewer-manifest.json"
    validate_record(
        summary.get("viewer", {}).get("html"),
        "source85 viewer HTML",
        viewer / "viewer.html",
    )
    validate_record(
        summary.get("viewer", {}).get("manifest"),
        "source85 viewer manifest",
        viewer_manifest,
    )
    if json.loads(viewer_manifest.read_text()).get("fibers") is not None:
        raise ValueError("source85 viewer unexpectedly exports fibers")
    inventory_count = validate_inventory(
        viewer, summary.get("output_inventory"), exact=True
    )
    return record(summary_path), comparison, inventory_count


def reproduction_files(document: dict[str, Any]) -> list[Path]:
    """Collect only declared, reproducible text/source evidence."""
    patterns = document.get("reproduction_patterns")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("nonempty reproduction_patterns required")
    allowed = {
        ".py",
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
    files: set[Path] = set()
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise TypeError("reproduction pattern must be text")
        matches = [path for path in ROOT.glob(pattern) if path.is_file()]
        if not matches:
            raise ValueError(f"reproduction pattern matched nothing: {pattern}")
        for path in matches:
            require_inside_root(path, "reproduction file")
            if path.suffix not in allowed:
                raise ValueError(f"unsupported reproduction file: {path}")
            files.add(path)
    return sorted(files)


def zip_paths(paths: list[Path], destination: Path) -> None:
    """Package declared sources and a readable refined optimizer-continuation guide."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
        archive.writestr(
            "MANIFEST.txt",
            "\n".join(path.relative_to(ROOT).as_posix() for path in paths) + "\n",
        )
        archive.writestr(
            "REPRODUCIBILITY_README.md",
            """# Optimizer continuation report evidence

MANIFEST.txt lists the exact contents of this archive. The figures archive contains
separate PNG/PDF assets. The reproducibility archive contains the declared source
and text evidence for sources 90-99, including records identifying saved local states.
Large VTK/NPZ states, checkpoints, and fixtures remain local;
the staged site contains derived viewer geometry and byte-hash manifests instead.

Use the report Markdown, source80 comparison receipt, source85 renderer receipt, and
per-case receipts to identify exact pinned source files. The publisher stages evidence
only; it does not run solvers, select best states, or deploy content.
""",
        )


def write_records_manifest(
    path: Path, *, archives: dict[str, dict], inputs: dict
) -> None:
    """Write fields accepted by CHECK plus optimizer-specific provenance."""
    path.write_text(
        json.dumps({"archives": archives, "optimizer_continuation": inputs}, indent=2)
        + "\n"
    )


def publish(manifest_path: Path, output: Path) -> dict:
    """Build a new static staging directory without deploying it."""
    manifest_path = require_inside_root(manifest_path, "publication manifest")
    document = json.loads(manifest_path.read_text())
    report = require_inside_root(
        manifest_path.parent / document["report"], "report Markdown"
    )
    viewer = require_inside_root(
        manifest_path.parent / document["viewer"], "source85 viewer"
    )
    comparison = require_inside_root(
        manifest_path.parent / document["comparison_summary_path"], "source80 summary"
    )
    if (
        report.suffix != ".md"
        or not report.is_file()
        or not (viewer / "viewer.html").is_file()
    ):
        raise ValueError("report Markdown and viewer.html are required")
    markdown = report.read_text()
    if "PENDING_" in markdown or "FINAL_" in markdown:
        raise ValueError("report has unresolved placeholders")
    if output.exists():
        raise FileExistsError("choose a new empty publication output directory")
    render_summary, comparison_summary, inventory_count = validate_renderer(
        viewer, comparison
    )
    reproduction = reproduction_files(document)
    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="optimizer-site-", dir=ROOT / "tmp"
    ) as temporary:
        stage = Path(temporary) / "site"
        shutil.copytree(viewer, stage)
        shutil.copy2(ROOT / "src/report-theme.css", stage / "report.css")
        shutil.copy2(ROOT / "src/report-theme.js", stage / "report-theme.js")
        rewritten = PUBLISH.copy_report_artifacts(markdown, report, stage)
        records = stage / "records"
        records.mkdir()
        shutil.copy2(comparison, records / "source80-comparison-summary.json")
        PUBLISH.downloadable_markdown_closure(report, stage)
        figures = PUBLISH.figures(report, viewer)
        zip_paths(figures, records / "figures.zip")
        zip_paths(reproduction, records / "reproducibility.zip")
        archives = {
            name: {
                "bytes": (records / name).stat().st_size,
                "sha256": PUBLISH.file_sha256(records / name),
            }
            for name in ("figures.zip", "reproducibility.zip")
        }
        write_records_manifest(
            records / "manifest.json",
            archives=archives,
            inputs={
                "scope": "optimizer continuation sources 90-99 and saved-state provenance records",
                "report": record(report),
                "source80": record(comparison),
                "source85": render_summary,
                "publication_manifest": record(manifest_path),
            },
        )
        body = PUBLISH.render_markdown(rewritten)
        (stage / "index.html").write_text(
            f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Optimizer repeatability and continuation</title><script src="report-theme.js"></script><link rel="stylesheet" href="report.css"></head>
<body><header class="site-header"><div class="site-header__inner">
<a class="wordmark" href="index.html">Optimizer repeatability</a><nav>
<a href="viewer.html">3D viewer</a><a href="records/figures.zip">PNG / PDF figures</a>
<a href="records/reproducibility.zip">Scripts and evidence</a>
<a href="records/{report.name}">Markdown</a></nav>
<label class="theme-picker" for="report-theme">Theme
<select id="report-theme" disabled><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select>
</label></div></header><main class="report">{body}</main>
<footer class="site-footer">Actual saved states. Common cameras. Original geometric scale.</footer></body></html>"""
        )
        CHECK.validate_record_manifest(stage)
        verification = {
            "status": "passed",
            "source80": record(comparison),
            "source85": render_summary,
            "source85_inventory_files": inventory_count,
            "source80_status": comparison_summary["status"],
            "geometry": CHECK.validate_geometry(stage),
            "javascript_modules": CHECK.validate_modules(stage),
            "html_files": CHECK.validate_site(stage),
            "downloadable_markdown_links": CHECK.validate_staged_markdown(stage),
            "archives": {name: CHECK.validate_zip(records / name) for name in archives},
            "report": record(report),
            "publisher": record(Path(__file__)),
            "publication_manifest": record(manifest_path),
        }
        (stage / "validation.json").write_text(
            json.dumps(verification, indent=2) + "\n"
        )
        hashes = {
            path.relative_to(stage).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": PUBLISH.file_sha256(path),
            }
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        (stage / "file-manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.manifest.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
