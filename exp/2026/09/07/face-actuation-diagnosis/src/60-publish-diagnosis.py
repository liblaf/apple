# ruff: noqa: EM101, S608, TRY003
"""Publish the face-actuation diagnosis report and its inspectable evidence.

The publisher only copies already-produced artifacts.  It never invokes a solver,
renderer, GPU program, or HTTP server.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
APPLE_ROOT = ROOT.parents[4]
DEFAULT_REPORT = ROOT / "docs" / "30-face-results.md"
DEFAULT_SITE = ROOT / "site"
LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^\s)]+)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\)")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^\s)]+)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\)")
LOCAL_ATTRIBUTES = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")
PUBLISHABLE_SUFFIXES = {".png", ".pdf", ".json", ".jsonl", ".csv"}
LINKED_SUFFIXES = PUBLISHABLE_SUFFIXES | {".md", ".py"}
ARCHIVE_EXCLUDED_SUFFIXES = {".npz", ".pt", ".pth", ".ckpt", ".vtu"}
MATCHED_STEP20_VIEWER = ROOT / "data" / "62-interim-adam-step20-viewer"
CONTINUATION_STEP10_VIEWER = ROOT / "data" / "67-continuation-step10-viewer"


class PublishError(RuntimeError):
    """A publication input does not meet the static-site contract."""

    def __init__(self, problem: str, detail: object = "") -> None:
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{problem}{suffix}")


def inside_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True


def published_relative(source: Path) -> Path | None:
    """Map a small approved report dependency into the staged artifact tree."""
    for base, prefix in ((ROOT, Path()), (APPLE_ROOT, Path("external"))):
        try:
            return prefix / source.resolve().relative_to(base.resolve())
        except ValueError:
            continue
    return None


def url_path(raw: str) -> Path | None:
    parsed = urlsplit(raw.strip("<>"))
    if parsed.scheme or parsed.netloc or not parsed.path or raw.startswith("#"):
        return None
    return Path(unquote(parsed.path))


def stable_slug(text: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"
    result, number = base, 2
    while result in used:
        result = f"{base}-{number}"
        number += 1
    used.add(result)
    return result


def render_markdown(markdown: str) -> str:
    renderer = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable(
        "table"
    )
    used: set[str] = set()

    def heading_open(tokens, index, _options, _env):  # noqa: ANN001
        token, inline = tokens[index], tokens[index + 1]
        identifier = stable_slug(inline.content, used)
        label = html.escape(inline.content, quote=True)
        return f'<{token.tag} id="{identifier}"><a class="heading-anchor" href="#{identifier}" aria-label="Link to {label}">#</a>'

    renderer.renderer.rules["heading_open"] = heading_open
    renderer.renderer.rules["table_open"] = lambda *_: (
        '<div class="table-scroll"><table>\n'
    )
    renderer.renderer.rules["table_close"] = lambda *_: "</table></div>\n"
    return renderer.render(markdown)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise PublishError("viewer bundle is not a directory", source)
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def copy_matched_step20_viewer(stage: Path) -> None:
    """Preserve the immutable viewer already shared for the matched step-20 check."""
    if not (MATCHED_STEP20_VIEWER / "viewer.html").is_file():
        raise PublishError(
            "matched step-20 viewer lacks viewer.html", MATCHED_STEP20_VIEWER
        )
    copy_tree(MATCHED_STEP20_VIEWER, stage / "matched-step20")


def copy_continuation_step10_viewer(stage: Path) -> None:
    """Preserve the immutable ongoing continuation step-10 viewer in the final site."""
    if not (CONTINUATION_STEP10_VIEWER / "viewer.html").is_file():
        raise PublishError(
            "continuation step-10 viewer lacks viewer.html", CONTINUATION_STEP10_VIEWER
        )
    copy_tree(CONTINUATION_STEP10_VIEWER, stage / "continuation-step10")


def copy_report_artifacts(markdown: str, report: Path, stage: Path) -> str:
    """Copy report-linked inspectable records and point Markdown at the copies."""
    copied: dict[Path, str] = {}

    def replacement(match: re.Match[str]) -> str:
        alt, raw = match.group(1), match.group(2)
        source_path = url_path(raw)
        if source_path is None:
            return match.group(0)
        source = (report.parent / source_path).resolve()
        relative = published_relative(source)
        if (
            relative is None
            or not source.is_file()
            or source.suffix.lower() not in LINKED_SUFFIXES
        ):
            return match.group(0)
        if source not in copied:
            target = stage / "artifacts" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[source] = target.relative_to(stage).as_posix()
        parsed = urlsplit(raw.strip("<>"))
        new_url = urlunsplit(("", "", copied[source], parsed.query, parsed.fragment))
        return f"[{alt}]({new_url})"

    # Preserve the leading exclamation mark when rewriting image links.
    def image_replacement(match: re.Match[str]) -> str:
        rewritten_image = replacement(match)
        if rewritten_image == match.group(0):
            return rewritten_image
        return "!" + rewritten_image

    rewritten = IMAGE.sub(image_replacement, markdown)
    return LINK.sub(replacement, rewritten)


def rewrite_download_markdown(source: Path, destination: Path, stage: Path) -> None:
    """Copy a Markdown record and recursively close its local, publishable links."""
    pending = [(source, destination)]
    done: set[Path] = set()

    def replace(match: re.Match[str], origin: Path, target: Path) -> str:
        alt, raw = match.group(1), match.group(2)
        relative = url_path(raw)
        if relative is None:
            return match.group(0)
        if relative.as_posix() in {
            "viewer.html",
            "records/figures.zip",
            "records/reproducibility.zip",
        }:
            # These are generated site destinations, including when this report
            # is reached again through a nested supporting Markdown document.
            staged = stage / relative
        else:
            linked = (origin.parent / relative).resolve()
            if (
                not linked.is_file()
                or linked.suffix.lower() not in LINKED_SUFFIXES
                or published_relative(linked) is None
            ):
                return match.group(0)
            staged = stage / "artifacts" / published_relative(linked)
            staged.parent.mkdir(parents=True, exist_ok=True)
            if not staged.exists():
                shutil.copy2(linked, staged)
            if linked.suffix.lower() == ".md" and linked not in done:
                pending.append((linked, staged))
        parsed = urlsplit(raw.strip("<>"))
        href = os.path.relpath(staged, target.parent).replace(os.sep, "/")
        url = urlunsplit(("", "", href, parsed.query, parsed.fragment))
        return f"[{alt}]({url})"

    while pending:
        origin, target = pending.pop()
        if origin in done:
            continue
        done.add(origin)
        target.parent.mkdir(parents=True, exist_ok=True)
        markdown = origin.read_text()

        def image_replace(
            match: re.Match[str], origin: Path = origin, target: Path = target
        ) -> str:
            rewritten_image = replace(match, origin, target)
            return (
                match.group(0)
                if rewritten_image == match.group(0)
                else "!" + rewritten_image
            )

        rewritten = IMAGE.sub(image_replace, markdown)
        rewritten = LINK.sub(
            lambda match, origin=origin, target=target: replace(match, origin, target),
            rewritten,
        )
        target.write_text(rewritten)


def downloadable_markdown_closure(report: Path, stage: Path) -> None:
    """Rewrite report and support Markdown into paths valid from their staged files."""
    records = stage / "records"
    rewrite_download_markdown(report, records / report.name, stage)
    for path in list((stage / "artifacts").rglob("*.md")):
        staged_relative = path.relative_to(stage / "artifacts")
        source = (
            APPLE_ROOT / staged_relative.relative_to("external")
            if staged_relative.parts[0] == "external"
            else ROOT / staged_relative
        )
        if source.is_file():
            rewrite_download_markdown(source, path, stage)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def zip_paths(paths: list[Path], destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
        archive.writestr(
            "MANIFEST.txt",
            "\n".join(path.relative_to(ROOT).as_posix() for path in paths) + "\n",
        )
        archive.writestr(
            "REPRODUCIBILITY_README.md",
            """# Archive contents and reproduction

MANIFEST.txt lists this archive's files. The figures download contains separate PNG/PDF assets. The reproduction download contains diagnosis source, experiment documentation, compact solver/configuration/provenance records, and frozen source snapshots from the numerical runs and conversion tools. Neither download contains full VTK/NPZ/optimizer state arrays or copied preview bundles. Reproduction requires the repository checkout, the exact fixture/state files named by each receipt or provenance record, and the recorded Python/CUDA environment. The static viewer carries its derived browser geometry separately.

Frozen runtime helpers under `data/runtime-sources/` are included in the reproduction download with their source hashes.
""",
        )


def reproducibility_inputs() -> list[Path]:
    """Return compact source, configuration, and provenance records.

    Preview bundles contain copied geometry and are deliberately excluded: the
    final viewer already carries its exact lazy geometry separately.
    """
    candidates: set[Path] = set()
    candidates.update(
        path for path in (ROOT / "src").rglob("*.py") if "__pycache__" not in path.parts
    )
    candidates.update((ROOT / "docs").glob("*.md"))
    candidates.update((ROOT / "docs").glob("*.json"))
    candidates.update((ROOT / "data" / "runtime-sources").rglob("*"))
    # A run's own copied sources are the reproducibility record. Discover every
    # direct pre-50 group so this diagnosis cannot silently inherit another
    # experiment's frozen-run inventory.
    for run in (ROOT / "data").iterdir():
        match = re.match(r"(\d+)-", run.name)
        if run.is_dir() and match and int(match.group(1)) < 50:
            candidates.update((run / "sources").rglob("*.py"))
            # The Adam checkpoint exporter is a frozen conversion tool stored
            # directly in its pilot record rather than under sources/.
            candidates.update(run.glob("*.py"))

    candidates.update(path for path in (ROOT / "data").glob("environment*.json"))
    # Final trend decisions and surface analyses share the later numbering with
    # preview assets. Include their compact evidence without copying geometry.
    candidates.update(
        path
        for pattern in (
            "60-final-diagnosis-render-service-receipt.json",
            "61-final-viewer-validation.json",
            "68-*.json",
            "69-completed-step60-surface/*.json",
            "70-final-comparison-figure/manifest.json",
        )
        for path in (ROOT / "data").glob(pattern)
    )
    for suffix in ("*.json", "*.jsonl"):
        for path in (ROOT / "data").rglob(suffix):
            relative = path.relative_to(ROOT / "data")
            first = relative.parts[0] if relative.parts else ""
            match = re.match(r"(\d+)-", first)
            # Pre-50 records are solver/configuration/audit provenance. Preserve
            # compact JSON and JSONL receipts, while excluding preview bundles and
            # any unexpectedly large state-like payload.
            if (
                match
                and int(match.group(1)) < 50
                and (path.suffix == ".jsonl" or path.stat().st_size <= 1_000_000)
            ):
                candidates.add(path)
    # Failure receipts under pre-50 run groups are JSON and are included above.
    # Preserve any compact terminal logs that accompany those direct run groups.
    candidates.update((ROOT / "logs").glob("*-terminal.log"))
    for path in (ROOT / "data").rglob("*.csv"):
        relative = path.relative_to(ROOT / "data")
        first = relative.parts[0] if relative.parts else ""
        preview = re.match(r"(?:[5-9]\d|[1-9]\d{2,})-", first)
        if not preview and "geometry" not in relative.parts:
            candidates.add(path)
    for name in ("README.md", "pyproject.toml", "uv.lock"):
        path = ROOT / name
        if path.is_file():
            candidates.add(path)
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix not in ARCHIVE_EXCLUDED_SUFFIXES
    )


def figures(report: Path, viewer_bundle: Path) -> list[Path]:
    found: set[Path] = set()
    for raw in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)", report.read_text()):
        relative = url_path(raw)
        if relative is None:
            continue
        source = (report.parent / relative).resolve()
        if (
            inside_root(source)
            and source.is_file()
            and source.suffix.lower() in {".png", ".pdf"}
        ):
            found.add(source)
    # Full-size static views are reusable evidence. Include all rendered case
    # PNG/PDF assets, but never browser geometry, VTUs, or history-frame images.
    for path in viewer_bundle.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in {".png", ".pdf"}
            and "history-frames" not in path.parts
        ):
            found.add(path)
    return sorted(found)


def validate_tet_asset(path: Path) -> None:
    """Reject any tetrahedron viewer input that is not four exact saved vertices."""
    payload = json.loads(path.read_text())
    states = payload.get("states")
    if not isinstance(states, list) or not states:
        raise PublishError("tetrahedron asset lacks states", path)
    for state in states:
        if not isinstance(state, dict) or len(state.get("vertices", [])) != 4:
            raise PublishError(
                "tetrahedron state must have exactly four vertices", path
            )
        if any(len(vertex) != 3 for vertex in state["vertices"]):
            raise PublishError("tetrahedron vertex is not a 3-vector", path)
        if len(state.get("triangles", [])) != 4:
            raise PublishError("tetrahedron state must have four supplied faces", path)


def tet_viewer(browser_asset: str) -> str:
    """A small exact-vertex viewer; each select change replaces the tetrahedron."""
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Material tetrahedron</title><link rel="stylesheet" href="report.css">
<body class="tet-page"><header class="site-header"><div class="site-header__inner"><a class="wordmark" href="index.html">Face activation report</a><label>state <select id="state"></select></label><span>Four saved vertices; no interpolation.</span></div></header><main><canvas id="view"></canvas></main>
<script type="module">
import * as THREE from './vendor/three.module.js'; import {{OrbitControls}} from './vendor/OrbitControls.js';
const data=await (await fetch('{browser_asset}')).json(), select=document.querySelector('#state'), canvas=document.querySelector('#view');
for(const s of data.states) select.add(new Option(`${{s.label}} (J=${{s.J.toFixed(3)}})`,s.key));
const renderer=new THREE.WebGLRenderer({{canvas,antialias:true}});renderer.setPixelRatio(devicePixelRatio);renderer.setClearColor(0xf7f5ef);const scene=new THREE.Scene();scene.add(new THREE.HemisphereLight(0xffffff,0x384957,2.2));const light=new THREE.DirectionalLight(0xffffff,1.6);light.position.set(2,3,4);scene.add(light);const camera=new THREE.PerspectiveCamera(35,1,.001,100),controls=new OrbitControls(camera,canvas);controls.enableDamping=true;
const points=data.states.flatMap(s=>s.vertices), box=new THREE.Box3();for(const p of points)box.expandByPoint(new THREE.Vector3(...p));const center=box.getCenter(new THREE.Vector3()), radius=Math.max(box.getSize(new THREE.Vector3()).length()/2,.01);camera.position.copy(center).add(new THREE.Vector3(1.5,1.2,1.8).normalize().multiplyScalar(radius*3.2));camera.near=radius/100;camera.far=radius*100;controls.target.copy(center);controls.update();let mesh;
function show(){{if(mesh){{scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose()}}const s=data.states.find(s=>s.key===select.value),g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(s.vertices.flat(),3));g.setIndex(s.triangles.flat());g.computeVertexNormals();mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({{color:0x3d8c8c,roughness:.72,side:THREE.DoubleSide}}));scene.add(mesh)}}
function resize(){{renderer.setSize(canvas.clientWidth,canvas.clientHeight,false);camera.aspect=canvas.clientWidth/canvas.clientHeight;camera.updateProjectionMatrix()}}new ResizeObserver(resize).observe(canvas);select.onchange=show;show();(function frame(){{requestAnimationFrame(frame);controls.update();renderer.render(scene,camera)}})();
</script></body></html>"""


def stylesheet() -> str:
    return """*{box-sizing:border-box}body{margin:0;background:#f7f5ef;color:#1b2827;font:16px/1.6 system-ui,sans-serif}.site-header{background:#173b3a;color:#fff}.site-header__inner{max-width:1100px;margin:auto;padding:.8rem 1rem;display:flex;gap:1rem;align-items:center;flex-wrap:wrap}.site-header a{color:#fff}.wordmark{font-weight:700;text-decoration:none}.site-header nav{display:flex;gap:.8rem;flex-wrap:wrap;font-size:.9rem}.report{max-width:960px;margin:2rem auto;padding:0 1rem 3rem}h1,h2,h3{line-height:1.2;margin-top:2rem}.heading-anchor{color:#779b98;text-decoration:none;margin-right:.4rem;font-size:.75em}a{color:#075e66}img{max-width:100%;height:auto}.figure-link{display:inline-block}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:.45rem;border:1px solid #c8d0ca;text-align:left}.site-footer{padding:1rem;text-align:center;background:#e7ebe6;color:#4a5b57;font-size:.9rem}.tet-page main{height:calc(100vh - 4.5rem)}#view{width:100%;height:100%;display:block}@media(max-width:600px){.report{margin-top:1rem}.site-header__inner{align-items:flex-start}.site-header nav{width:100%}}"""


def page_document(body: str, report_name: str, tet_link: str) -> str:
    tet = tet_link
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Face activation report</title><link rel="stylesheet" href="report.css"></head><body><header class="site-header"><div class="site-header__inner"><a class="wordmark" href="#top">Face activation report</a><nav><a href="#top">Report</a><a href="viewer.html">3D viewer</a><a href="matched-step20/viewer.html">Matched step 20</a><a href="continuation-step10/viewer.html">Continuation step 10</a>{tet}<a href="records/{report_name}">Source Markdown</a><a href="records/figures.zip">Figures ZIP</a><a href="records/reproducibility.zip">Reproduction ZIP</a></nav></div></header><main id="top" class="report">{body}</main><footer class="site-footer">Static copies of the linked records accompany this report.</footer></body></html>"""


def validate_links(stage: Path, document: str) -> None:
    missing = []
    for raw in LOCAL_ATTRIBUTES.findall(document):
        parsed = urlsplit(raw)
        if (
            parsed.scheme
            or parsed.netloc
            or raw.startswith(("#", "data:"))
            or not parsed.path
        ):
            continue
        if not (stage / unquote(parsed.path)).is_file():
            missing.append(raw)
    if missing:
        raise PublishError(
            "unresolved local references: " + ", ".join(sorted(set(missing)))
        )


def publish(viewer_bundle: Path, report: Path, site: Path) -> None:
    report, viewer_bundle = report.resolve(), viewer_bundle.resolve()
    if not report.is_file():
        raise PublishError("report is not ready", report)
    if not viewer_bundle.is_dir() or not (viewer_bundle / "viewer.html").is_file():
        raise PublishError("viewer bundle lacks viewer.html", viewer_bundle)
    markdown = report.read_text()
    if "<!-- FINAL_" in markdown:
        raise PublishError("unfinished report placeholder")
    with tempfile.TemporaryDirectory(prefix="face-diagnosis-", dir=ROOT) as temporary:
        stage = Path(temporary) / "site"
        copy_tree(viewer_bundle, stage)
        copy_matched_step20_viewer(stage)
        copy_continuation_step10_viewer(stage)
        (stage / "report.css").write_text(stylesheet())
        rewritten = copy_report_artifacts(markdown, report, stage)
        records = stage / "records"
        records.mkdir()
        downloadable_markdown_closure(report, stage)
        figure_zip = records / "figures.zip"
        repro_zip = records / "reproducibility.zip"
        zip_paths(figures(report, viewer_bundle), figure_zip)
        zip_paths(reproducibility_inputs(), repro_zip)
        (records / "manifest.json").write_text(
            json.dumps(
                {
                    "report_markdown": report.name,
                    "viewer_manifest_sha256": file_sha256(
                        stage / "viewer-manifest.json"
                    ),
                    "matched_step20_viewer_sha256": file_sha256(
                        stage / "matched-step20" / "viewer.html"
                    ),
                    "continuation_step10_viewer_sha256": file_sha256(
                        stage / "continuation-step10" / "viewer.html"
                    ),
                    "archives": {
                        "figures.zip": {"sha256": file_sha256(figure_zip)},
                        "reproducibility.zip": {"sha256": file_sha256(repro_zip)},
                    },
                    "figure_scope": "report-linked PNG/PDF plus final viewer static PNG/PDF; excludes lazy geometry, VTU, NPZ, and history-frame images",
                },
                indent=2,
            )
            + "\n"
        )
        tet_source = ROOT / "data" / "44-material-tet" / "browser.json"
        if not tet_source.is_file():
            choices = sorted((ROOT / "data" / "44-material-tet").glob("*-browser.json"))
            tet_source = choices[0] if choices else tet_source
        has_tet = tet_source.is_file()
        if has_tet:
            validate_tet_asset(tet_source)
            target = stage / "artifacts" / tet_source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tet_source, target)
            (stage / "tet-viewer.html").write_text(
                tet_viewer(target.relative_to(stage).as_posix())
            )
        tet_link = (
            '<a href="tet-viewer.html">Material tetrahedron</a>' if has_tet else ""
        )
        document = page_document(render_markdown(rewritten), report.name, tet_link)
        validate_links(stage, document)
        (stage / "index.html").write_text(document)
        site.parent.mkdir(parents=True, exist_ok=True)
        backup = site.with_name(site.name + ".previous")
        shutil.rmtree(backup, ignore_errors=True)
        if site.exists():
            site.replace(backup)
        stage.replace(site)
        shutil.rmtree(backup, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer-bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    args = parser.parse_args()
    publish(args.viewer_bundle, args.report, args.site)


if __name__ == "__main__":
    main()
