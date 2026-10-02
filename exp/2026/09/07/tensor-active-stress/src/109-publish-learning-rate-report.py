# ruff: noqa: EM101, TRY003
"""Stage, validate, and label the learning-rate report; never deploy it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src/99-publish-refined-optimizer-report.py"


def load() -> Any:
    """Load the immutable publication validation framework."""
    spec = importlib.util.spec_from_file_location(
        "learning_rate_publication_framework", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FRAMEWORK = load()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def zip_paths(paths: list[Path], destination: Path) -> None:
    """Package the declared 100--109 evidence with its correct scope caption."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
        archive.writestr(
            "MANIFEST.txt",
            "\n".join(path.relative_to(ROOT).as_posix() for path in paths) + "\n",
        )
        archive.writestr(
            "REPRODUCIBILITY_README.md",
            "# Learning-rate continuation report evidence\n\nMANIFEST.txt lists the exact contents of this archive. The figures archive contains separate PNG/PDF assets. The reproducibility archive contains the declared source and text evidence for sources 100-109, including records identifying saved local states. Large VTK/NPZ states, checkpoints, and fixtures remain local; the staged site contains derived viewer geometry and byte-hash manifests instead.\n\nUse the report Markdown, source105 comparison receipt, source107 renderer receipt, and per-case receipts to identify exact pinned source files. The publisher stages evidence only; it does not run solvers, choose rates, select best states, or deploy content.\n",
        )


def write_records_manifest(path: Path, *, archives: dict, inputs: dict) -> None:
    """Preserve framework provenance while giving the staged evidence its scope."""
    scoped = {
        **inputs,
        "scope": "learning-rate continuation sources 100-109 and saved-state provenance records",
        "framework_publisher": record(SOURCE),
        "learning_rate_publisher": record(Path(__file__)),
    }
    path.write_text(
        json.dumps({"archives": archives, "optimizer_continuation": scoped}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def rewrite_identity(output: Path, manifest: Path) -> dict[str, Any]:
    """Bind the staged static output to the 100--109 evidence range."""
    index = output / "index.html"
    html = index.read_text(encoding="utf-8")
    html = html.replace(
        "Optimizer repeatability and continuation",
        "Learning-rate repeatability and continuation",
    )
    html = html.replace("Optimizer repeatability", "Learning-rate repeatability")
    index.write_text(html, encoding="utf-8")
    validation_path = output / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["learning_rate_report"] = {
        "evidence_caption": "Learning-rate experiment evidence, sources 100-109; actual saved states and manifest-pinned postprocessing only.",
        "publication_manifest": record(manifest),
        "publisher": record(Path(__file__)),
    }
    FRAMEWORK.CHECK.validate_record_manifest(output)
    validation["html_files"] = FRAMEWORK.CHECK.validate_site(output)
    validation["downloadable_markdown_links"] = (
        FRAMEWORK.CHECK.validate_staged_markdown(output)
    )
    validation["javascript_modules"] = FRAMEWORK.CHECK.validate_modules(output)
    validation_path.write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    file_manifest = {
        path.relative_to(output).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "file-manifest.json"
    }
    (output / "file-manifest.json").write_text(
        json.dumps(file_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return validation


def publish(manifest: Path, output: Path) -> dict[str, Any]:
    """Apply the established validation framework, then attach new evidence identity."""
    manifest, output = manifest.resolve(), output.resolve()
    document = json.loads(manifest.read_text(encoding="utf-8"))
    required = {"report", "viewer", "comparison_summary_path", "reproduction_patterns"}
    if set(document) - {"schema_version", *required} or not required <= set(document):
        raise ValueError("publication manifest schema differs")
    if document.get("schema_version") != 1:
        raise ValueError("publication manifest schema_version differs")
    FRAMEWORK.zip_paths = zip_paths
    FRAMEWORK.write_records_manifest = write_records_manifest
    FRAMEWORK.publish(manifest, output)
    return rewrite_identity(output, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.manifest, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
