"""Publish the final measured Markdown report as a self-contained static site.

This helper is intentionally presentation-only: it copies existing figures and records,
rewrites local report links, and validates every local HTML reference before replacing
``site/index.html``.  It does not calculate or interpret experimental results.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
REPORT = ROOT / "docs" / "20-results.md"
ANALYSIS = ROOT / "data" / "50-analysis"
FREQUENCY = ROOT / "data" / "10-frequency"
POLISH = ROOT / "data" / "35-polish"
REFINEMENT = ROOT / "data" / "40-refinement"
AUDIT_REFINEMENT = ROOT / "data" / "45-audit-refinement"
AUDIT = ROOT / "data" / "45-audit"
ASSET_EXTENSIONS = {".png", ".pdf"}
RECORD_EXTENSIONS = {".csv", ".json"}
LOCAL_REFERENCE = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")
IMAGE_TAG = re.compile(
    r'<img\b(?P<before>[^>]*?)\bsrc="(?P<src>[^"]+)"(?P<after>[^>]*)>'
)


class PublishError(Exception):
    """Raised when report inputs cannot be safely published."""

    def __init__(self, problem: str, detail: object) -> None:
        super().__init__(f"{problem}: {detail}")


def slugify(value: str, used: set[str]) -> str:
    """Create a stable, unique fragment identifier for one Markdown heading."""
    base = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "section"
    slug = base
    number = 2
    while slug in used:
        slug = f"{base}-{number}"
        number += 1
    used.add(slug)
    return slug


def render_markdown(markdown: str) -> str:
    """Render CommonMark with heading links and scrollable tables."""
    renderer = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable(
        "table"
    )
    used_ids: set[str] = set()

    def heading_open(tokens, index, _options, _env):  # noqa: ANN001
        token = tokens[index]
        inline = tokens[index + 1]
        identifier = slugify(inline.content, used_ids)
        anchor = (
            f'<a class="heading-anchor" href="#{identifier}" '
            f'aria-label="Link to {html.escape(inline.content, quote=True)}">#</a>'
        )
        return f'<{token.tag} id="{identifier}">{anchor}'

    renderer.renderer.rules["heading_open"] = heading_open
    renderer.renderer.rules["table_open"] = lambda _tokens, _index, _options, _env: (
        '<div class="table-scroll"><table>\n'
    )
    renderer.renderer.rules["table_close"] = lambda _tokens, _index, _options, _env: (
        "</table></div>\n"
    )
    return renderer.render(markdown)


def figure_sources() -> Iterable[Path]:
    """Yield the report-ready figure files, with collision-safe output names."""
    for directory in (ANALYSIS, FREQUENCY):
        if directory.is_dir():
            yield from sorted(
                path for path in directory.iterdir() if path.suffix in ASSET_EXTENSIONS
            )


def copy_assets(destination: Path) -> dict[str, str]:
    """Copy figures and return Markdown-path to site-path rewrites."""
    destination.mkdir(parents=True, exist_ok=True)
    rewrites: dict[str, str] = {}
    seen_names: set[str] = set()
    for source in figure_sources():
        if source.name in seen_names:
            problem = "asset filename collision"
            raise PublishError(problem, source.name)
        seen_names.add(source.name)
        shutil.copy2(source, destination / source.name)
        relative = source.relative_to(ROOT).as_posix()
        rewrites[f"../{relative}"] = f"assets/{source.name}"
    return rewrites


def write_figures_zip(destination: Path) -> None:
    """Bundle all reusable PNG/PDF figures under their checked flat asset names."""
    sources = list(figure_sources())
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        problem = "figure bundle filename collision"
        raise PublishError(problem, ", ".join(names))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, source.name)


def copy_records(destination: Path) -> dict[str, str]:
    """Copy report, tabular records, and JSON summaries for direct inspection."""
    destination.mkdir(parents=True, exist_ok=True)
    rewrites: dict[str, str] = {}
    report_copy = destination / REPORT.name
    shutil.copy2(REPORT, report_copy)
    rewrites["../docs/20-results.md"] = f"records/{REPORT.name}"
    for source in (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))):
        if source == REPORT:
            continue
        target = destination / source.name
        if target.exists():
            problem = "record filename collision"
            raise PublishError(problem, source.name)
        shutil.copy2(source, target)
        rewrites[f"../{source.relative_to(ROOT).as_posix()}"] = f"records/{source.name}"
    for directory in (
        ANALYSIS,
        FREQUENCY,
        ROOT / "data" / "32-smoothing-holdout",
        POLISH,
        ROOT / "data" / "37-gradient-scale",
        ROOT / "data" / "37-gradient-scale-fiber10",
        REFINEMENT,
        AUDIT_REFINEMENT,
        AUDIT,
        ROOT / "data" / "45-audit-polish",
    ):
        if not directory.is_dir():
            continue
        for source in sorted(directory.glob("*")):
            if source.suffix not in RECORD_EXTENSIONS:
                continue
            target_name = f"{directory.name}-{source.name}"
            shutil.copy2(source, destination / target_name)
            rewrites[f"../{source.relative_to(ROOT).as_posix()}"] = (
                f"records/{target_name}"
            )
    return rewrites


def rewrite_local_paths(markdown: str, rewrites: dict[str, str]) -> str:
    """Replace report-local file paths only; ordinary prose remains untouched."""
    for original, replacement in sorted(
        rewrites.items(), key=lambda item: -len(item[0])
    ):
        markdown = markdown.replace(f"]({original})", f"]({replacement})")
    return markdown


def wrap_images(html_body: str) -> str:
    """Make each report figure open its full-size asset in a new browser tab."""

    def replace(match: re.Match[str]) -> str:
        tag = match.group(0)
        source = match.group("src")
        return (
            f'<a class="figure-link" href="{source}" '
            f'target="_blank" rel="noopener">{tag}</a>'
        )

    return IMAGE_TAG.sub(replace, html_body)


def page_document(body: str) -> str:
    """Return the fixed report shell without adding scientific claims."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Activation constraints — measured report</title>
  <link rel="stylesheet" href="report.css">
</head>
<body>
  <header class="site-header">
    <div class="site-header__inner">
      <a class="wordmark" href="#top">Activation constraints</a>
      <nav aria-label="Report navigation">
        <a href="#top">Report</a>
        <a href="records/20-results.md">Source Markdown</a>
        <a href="records/figures.zip">Figure bundle</a>
        <a href="records/reproducibility.zip">Reproducibility ZIP</a>
      </nav>
    </div>
  </header>
  <main id="top" class="report">{body}</main>
  <footer class="site-footer">
    Measured experiment records and reproducibility materials are linked above.
  </footer>
</body>
</html>
"""


def local_references(html_document: str) -> Iterable[str]:
    """Yield ordinary relative href/src paths, excluding fragments and external URLs."""
    for raw in LOCAL_REFERENCE.findall(html_document):
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc or raw.startswith(("#", "data:")):
            continue
        if not parsed.path:
            continue
        yield unquote(parsed.path)


def validate_links(site_root: Path, html_document: str) -> None:
    """Fail before publishing if any generated local link does not resolve."""
    missing = sorted(
        {
            path
            for path in local_references(html_document)
            if not (site_root / path).is_file()
        }
    )
    if missing:
        problem = "unresolved local report references"
        raise PublishError(problem, ", ".join(missing))


def reproducibility_inputs() -> Iterable[Path]:
    """Select experiment sources and compact records, excluding site output."""
    yield from sorted((ROOT / "src").glob("*.py"))
    yield from sorted((ROOT / "docs").glob("*.md"))
    data = ROOT / "data"
    for extension in ("*.json", "*.csv"):
        yield from sorted(
            path for path in data.rglob(extension) if ".cherries" not in path.parts
        )
    yield from sorted(
        path
        for path in data.rglob("*.py")
        if "sources" in path.parts and ".cherries" not in path.parts
    )
    if (ROOT / "README.md").is_file():
        yield ROOT / "README.md"


def write_reproducibility_zip(destination: Path) -> None:
    """Create a compact archive without large simulation-state NPZ files."""
    paths = sorted(set(reproducibility_inputs()))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
        archive.writestr(
            "MANIFEST.txt",
            "\n".join(path.relative_to(ROOT).as_posix() for path in paths) + "\n",
        )


def publish() -> None:
    """Build in a temporary sibling then atomically replace final site content."""
    if not REPORT.is_file():
        problem = "final report is not ready"
        raise PublishError(problem, REPORT)
    if not (SITE / "report.css").is_file():
        problem = "report stylesheet is missing"
        raise PublishError(problem, SITE / "report.css")
    markdown = REPORT.read_text()
    if "<!-- FINAL_" in markdown:
        problem = "unfinished final report placeholder"
        raise PublishError(problem, "remove every <!-- FINAL_... --> marker")
    with tempfile.TemporaryDirectory(
        prefix="activation-report-", dir=ROOT
    ) as temporary:
        stage = Path(temporary) / "site"
        assets, records = stage / "assets", stage / "records"
        rewrites = copy_assets(assets)
        rewrites.update(copy_records(records))
        shutil.copy2(SITE / "report.css", stage / "report.css")
        write_reproducibility_zip(records / "reproducibility.zip")
        write_figures_zip(records / "figures.zip")
        rewrites.update(
            {
                "../site/records/figures.zip": "records/figures.zip",
                "../site/records/reproducibility.zip": "records/reproducibility.zip",
            }
        )
        rewritten_markdown = rewrite_local_paths(markdown, rewrites)
        document = page_document(wrap_images(render_markdown(rewritten_markdown)))
        validate_links(stage, document)
        (stage / "index.html").write_text(document)
        for name in ("assets", "records"):
            shutil.rmtree(SITE / name, ignore_errors=True)
            (stage / name).replace(SITE / name)
        (stage / "index.html").replace(SITE / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    publish()


if __name__ == "__main__":
    main()
