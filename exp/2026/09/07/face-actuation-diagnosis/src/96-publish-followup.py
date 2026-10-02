"""Build and validate an independent static report for the follow-up experiments."""

# ruff: noqa: CPY001

from __future__ import annotations

import argparse
import html as html_module
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PUBLISH = load("existing_diagnosis_publisher", "60-publish-diagnosis.py")
CHECK = load("existing_diagnosis_validator", "61-validate-diagnosis-artifacts.py")


def reproduction_files(manifest: dict) -> list[Path]:
    """Use explicit scoped inputs and require every declared pattern to match."""
    paths: set[Path] = set()
    allowed = {".py", ".md", ".json", ".jsonl", ".csv", ".log", ".txt", ".css", ".js"}
    for pattern in manifest["reproduction_patterns"]:
        matches = [p for p in ROOT.glob(pattern) if p.is_file()]
        assert matches, pattern
        for path in matches:
            assert path.suffix in allowed, path
            path.resolve().relative_to(ROOT.resolve())
            paths.add(path)
    return sorted(paths)


def publish(manifest_path: Path, output: Path) -> dict:
    document = json.loads(manifest_path.read_text())
    report = (manifest_path.parent / document["report"]).resolve()
    viewer = (manifest_path.parent / document["viewer"]).resolve()
    markdown = report.read_text()
    assert "PENDING_" not in markdown
    assert "FINAL_" not in markdown
    assert not output.exists(), "Choose a new publication output directory"
    assert (viewer / "viewer.html").is_file()
    viewers = {".": viewer}
    for extra in document.get("extra_viewers", []):
        destination = Path(extra["destination"])
        assert not destination.is_absolute()
        assert ".." not in destination.parts
        assert destination.as_posix() not in viewers
        source = (manifest_path.parent / extra["source"]).resolve()
        assert (source / "viewer.html").is_file()
        viewers[destination.as_posix()] = source
    reproduction = reproduction_files(document)
    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="followup-site-", dir=ROOT / "tmp") as temp:
        stage = Path(temp) / "site"
        shutil.copytree(viewer, stage)
        for destination, source in viewers.items():
            if destination != ".":
                shutil.copytree(source, stage / destination)
        shutil.copy2(ROOT / "src/report-theme.css", stage / "report.css")
        shutil.copy2(ROOT / "src/report-theme.js", stage / "report-theme.js")
        rewritten = PUBLISH.copy_report_artifacts(markdown, report, stage)
        records = stage / "records"
        records.mkdir()
        PUBLISH.downloadable_markdown_closure(report, stage)
        figures = sorted(
            {p for source in viewers.values() for p in PUBLISH.figures(report, source)}
        )
        PUBLISH.zip_paths(figures, records / "figures.zip")
        PUBLISH.zip_paths(reproduction, records / "reproducibility.zip")
        archives = {
            name: {
                "sha256": PUBLISH.file_sha256(records / name),
                "bytes": (records / name).stat().st_size,
            }
            for name in ("figures.zip", "reproducibility.zip")
        }
        viewer_records = {
            destination: {
                "path": (Path(destination) / "viewer-manifest.json").as_posix(),
                "sha256": PUBLISH.file_sha256(
                    stage / destination / "viewer-manifest.json"
                ),
                "bytes": (stage / destination / "viewer-manifest.json").stat().st_size,
            }
            for destination in viewers
        }
        (records / "manifest.json").write_text(
            json.dumps(
                {
                    "report_markdown": report.name,
                    "viewer_manifest_sha256": PUBLISH.file_sha256(
                        stage / "viewer-manifest.json"
                    ),
                    "archives": archives,
                    "viewers": viewer_records,
                    "publication_inputs_sha256": PUBLISH.file_sha256(manifest_path),
                    "scope": "Follow-up field diffusion, skin transmission, and independent active tension; large source state arrays remain in the workspace.",
                },
                indent=2,
            )
            + "\n"
        )
        body = PUBLISH.render_markdown(rewritten)
        extra_nav = "".join(
            f'<a href="{html_module.escape(destination, quote=True)}/viewer.html">'
            f"{html_module.escape(destination.title())} comparison</a>\n"
            for destination in viewers
            if destination != "."
        )
        html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face activation follow-up</title><script src="report-theme.js"></script><link rel="stylesheet" href="report.css"></head>
<body><header class="site-header"><div class="site-header__inner">
<a class="wordmark" href="index.html">Face activation follow-up</a><nav>
<a href="viewer.html">3D viewer</a>
{extra_nav}
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
        for record in saved_records["viewers"].values():
            path = stage / record["path"]
            assert PUBLISH.file_sha256(path) == record["sha256"]
            assert path.stat().st_size == record["bytes"]
        verification = {
            "status": "passed",
            "viewers": {
                destination: {
                    "geometry": CHECK.validate_geometry(stage / destination),
                    "case_receipts": CHECK.validate_receipts(stage / destination),
                    "javascript_modules": CHECK.validate_modules(stage / destination),
                }
                for destination in viewers
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
    parser.add_argument("--output", type=Path, default=ROOT / "data/96-followup-site")
    args = parser.parse_args()
    print(json.dumps(publish(args.manifest.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
