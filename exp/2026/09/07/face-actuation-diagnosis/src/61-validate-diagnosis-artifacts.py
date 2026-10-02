"""Verify closure and provenance of a rendered diagnosis viewer and optional site.

This CPU-only validator never runs a solver, renderer, browser, or server. It
checks lazy browser geometry, generated receipts, local web links and modules,
ZIP CRCs/contents, and any persisted SHA-256 provenance records it can resolve.
"""

# ruff: noqa: C901, EM101, EM102, PLR0912, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_COPY_CORRECTION = ROOT / "data" / "36-checkpoint-copy-correction.json"
CHECKPOINT_COPY_CORRECTION_BEFORE = (
    ROOT / "data" / "36-checkpoint-copy-correction-before.json"
)
SOURCE_SHA256_BEFORE = (
    "f4a1509aa8ac31149c14be01584c4256ceb65a3f9b04b658b39cc033846f0337"
)
SOURCE_SHA256_AFTER = "691872807eaee0e33ecf222de03ed4880f45344dc7bfeaa5195b17394863e5f6"
ATTR = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")
IMPORT = re.compile(r"(?:import|export)\s+(?:[^'\"]+?\s+from\s+)?[\"']([^\"']+)[\"']")
MARKDOWN = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)")


class ClosureError(RuntimeError):
    """A viewer or publication artifact has a broken local dependency."""


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_file_record(
    record: Any, context: str, expected_path: Path | None = None
) -> Path:
    """Validate one literal path/bytes/SHA-256 provenance record."""
    if not isinstance(record, dict):
        raise ClosureError(f"invalid file record: {context}")
    path = Path(record.get("path", ""))
    if expected_path is not None and path.resolve() != expected_path.resolve():
        raise ClosureError(f"unexpected file record path: {context}")
    if (
        not path.is_file()
        or path.stat().st_size != record.get("bytes")
        or sha256(path) != record.get("sha256")
    ):
        raise ClosureError(f"file record digest mismatch: {context}")
    return path


def validate_checkpoint_copy_correction(
    directory: Path, provenance: dict[str, Any]
) -> int:
    """Verify the additive deep-copy correction without erasing its old revision."""
    current_source = provenance["sources"].get("35-continue-historical-adam.py")
    correction_record = provenance.get("checkpoint_copy_correction")
    if current_source is None:
        return 0
    revision = directory / "revisions" / "01-cpu-tensor-snapshot-copy"
    if correction_record is None:
        if current_source != SOURCE_SHA256_AFTER:
            raise ClosureError(
                f"unexpected fresh continuation source hash: {directory}"
            )
        if revision.exists():
            raise ClosureError(
                f"checkpoint-copy revision lacks its marker: {directory}"
            )
        # A fresh run began with the corrected source, so it has no historical
        # migration to claim or validate.
        return 0
    if current_source != SOURCE_SHA256_AFTER:
        raise ClosureError(f"unexpected continuation source hash: {directory}")
    correction_path = validate_file_record(
        correction_record, "checkpoint-copy correction", CHECKPOINT_COPY_CORRECTION
    )
    correction = json.loads(correction_path.read_text())
    if (
        correction.get("schema_version") != 1
        or correction.get("source_sha256_before") != SOURCE_SHA256_BEFORE
        or correction.get("source_sha256_after") != SOURCE_SHA256_AFTER
        or correction.get("physics_materials_objective_tolerances_unchanged")
        is not True
        or correction.get("adam_moment_reset_performed") is not False
        or correction.get("resume_pt_preserved_byte_for_byte") is not True
        or correction.get("affected_failure_path_executed_before_fix") is not False
    ):
        raise ClosureError("checkpoint-copy correction contract differs")
    before_path = validate_file_record(
        correction.get("before_change_record"),
        "checkpoint-copy correction before record",
        CHECKPOINT_COPY_CORRECTION_BEFORE,
    )
    before = json.loads(before_path.read_text())
    if before.get("old_source_sha256") != SOURCE_SHA256_BEFORE:
        raise ClosureError("checkpoint-copy before record source hash differs")
    run_key = (
        "raw6"
        if directory.name == "38-historical-adam-raw6-continuation"
        else "raw6-s"
        if directory.name == "39-historical-adam-raw6-s-continuation"
        else None
    )
    if run_key is None:
        raise ClosureError(f"unknown corrected continuation directory: {directory}")
    old_run = before.get("runs", {}).get(run_key)
    if (
        not isinstance(old_run, dict)
        or Path(old_run.get("run_directory", "")).resolve() != directory
    ):
        raise ClosureError(f"checkpoint-copy before record run differs: {directory}")
    frozen_before = old_run.get("frozen_before_change")
    if not isinstance(frozen_before, dict) or not frozen_before:
        raise ClosureError(f"checkpoint-copy before record is incomplete: {directory}")
    for name, record in frozen_before.items():
        validate_file_record(
            record, f"checkpoint-copy retained input {directory} {name}"
        )
    old_source = frozen_before.get("35-continue-historical-adam.py")
    old_source_path = validate_file_record(
        old_source, f"checkpoint-copy archived old source {directory}"
    )
    if (
        old_source_path.parent.name != "01-cpu-tensor-snapshot-copy"
        or old_source.get("sha256") != SOURCE_SHA256_BEFORE
    ):
        raise ClosureError(
            f"checkpoint-copy initial source revision differs: {directory}"
        )
    updated = correction.get("updated_archived_sources", {}).get(run_key)
    updated_path = validate_file_record(
        updated, f"checkpoint-copy updated source {directory}"
    )
    current_path = directory / "sources" / "35-continue-historical-adam.py"
    if updated_path != current_path or sha256(current_path) != SOURCE_SHA256_AFTER:
        raise ClosureError(f"checkpoint-copy updated source differs: {directory}")
    return 3 + len(frozen_before)


def local(raw: str) -> str | None:
    parsed = urlsplit(raw.strip("<>"))
    if parsed.scheme or parsed.netloc or raw.startswith("#") or not parsed.path:
        return None
    return unquote(parsed.path)


def ensure_local(
    base: Path, raw: str, context: str, *, root: Path | None = None
) -> Path | None:
    path = local(raw)
    if path is None:
        return None
    target = (base / path).resolve()
    allowed_root = (base if root is None else root).resolve()
    try:
        target.relative_to(allowed_root)
    except ValueError as error:
        raise ClosureError(
            f"local path escapes artifact root in {context}: {raw}"
        ) from error
    if not target.is_file():
        raise ClosureError(f"missing local path in {context}: {raw}")
    return target


def validate_geometry(viewer: Path) -> dict[str, int]:
    document = json.loads((viewer / "viewer-manifest.json").read_text())
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ClosureError("viewer manifest has no cases")
    state_files: set[Path] = set()
    topology_files: set[Path] = set()
    topology_vertex_counts: dict[Path, set[int]] = {}
    state_count = 0
    for case in cases:
        if not isinstance(case, dict):
            raise ClosureError("viewer manifest contains non-object case")
        pointers = list(case.get("states", {}).values())
        pointers.extend(frame.get("file") for frame in case.get("history", []))
        pointers.extend(
            frame.get("cutaway_file")
            for frame in case.get("history", [])
            if frame.get("cutaway_file")
        )
        for pointer in pointers:
            if not isinstance(pointer, str):
                raise ClosureError(f"bad geometry pointer in {case.get('id')}")
            state_path = ensure_local(viewer, pointer, "viewer manifest")
            assert state_path is not None
            state_files.add(state_path)
    for state_path in state_files:
        state = json.loads(state_path.read_text())
        positions, normals, topology_name = (
            state.get("positions"),
            state.get("normals"),
            state.get("topology"),
        )
        if (
            not isinstance(positions, list)
            or not isinstance(normals, list)
            or len(positions) == 0
            or len(positions) % 3
            or len(normals) != len(positions)
            or not isinstance(topology_name, str)
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in [*positions, *normals]
            )
        ):
            raise ClosureError(f"invalid lazy state geometry: {state_path}")
        topology_path = ensure_local(
            viewer, f"geometry/{topology_name}", f"lazy state {state_path.name}"
        )
        assert topology_path is not None
        topology_files.add(topology_path)
        topology_vertex_counts.setdefault(topology_path, set()).add(len(positions) // 3)
        state_count += 1
    for topology_path in topology_files:
        topology = json.loads(topology_path.read_text())
        indices = topology.get("indices")
        classes = topology.get("material_classes")
        if not isinstance(indices, list) or not indices or len(indices) % 3:
            raise ClosureError(f"invalid lazy topology: {topology_path}")
        vertex_counts = topology_vertex_counts[topology_path]
        if len(vertex_counts) != 1:
            raise ClosureError(
                f"lazy topology has inconsistent state vertex counts: {topology_path}"
            )
        vertices = next(iter(vertex_counts))
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= vertices
            for index in indices
        ):
            raise ClosureError(
                f"topology index is outside lazy state vertices: {topology_path}"
            )
        if classes is not None and (
            not isinstance(classes, list) or len(classes) != len(indices) // 3
        ):
            raise ClosureError(f"material class count mismatch: {topology_path}")
    return {
        "cases": len(cases),
        "lazy_states": state_count,
        "lazy_topologies": len(topology_files),
    }


def validate_receipts(viewer: Path) -> int:
    count = 0
    for receipt_path in viewer.glob("*/receipt.json"):
        receipt = json.loads(receipt_path.read_text())
        for section in ("sources",):
            for name, record in receipt.get(section, {}).items():
                if not isinstance(record, dict):
                    raise ClosureError(f"invalid {section}.{name}: {receipt_path}")
                path = Path(record.get("path", ""))
                if not path.is_file() or sha256(path) != record.get("sha256"):
                    raise ClosureError(f"source digest mismatch: {receipt_path} {name}")
                if path.stat().st_size != record.get("bytes"):
                    raise ClosureError(
                        f"source byte count mismatch: {receipt_path} {name}"
                    )
        for name, record in receipt.get("outputs", {}).items():
            generated = receipt_path.parent / name
            if not generated.is_file() or sha256(generated) != record.get("sha256"):
                raise ClosureError(f"output digest mismatch: {generated}")
        history = receipt.get("history")
        if isinstance(history, dict):
            for record in [*history.get("frames", []), history.get("video", {})]:
                path = Path(record.get("path", ""))
                if not path.is_file() or sha256(path) != record.get("sha256"):
                    raise ClosureError(f"history digest mismatch: {receipt_path}")
        count += 1
    if not count:
        raise ClosureError("viewer has no case receipts")
    return count


def without_comments(text: str) -> str:
    """Remove JavaScript comments before extracting executable import specifiers."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.DOTALL)


def validate_run_provenance(directory: Path) -> int:
    """Match persisted input/source SHA-256 values against frozen run artifacts."""
    provenance_path, config_path = (
        directory / "provenance.json",
        directory / "config.json",
    )
    if not provenance_path.is_file() or not config_path.is_file():
        return 0
    provenance = json.loads(provenance_path.read_text())
    config = json.loads(config_path.read_text())
    checked = 0
    for name, expected in provenance.get("sources", {}).items():
        source = directory / "sources" / name
        if not source.is_file() or sha256(source) != expected:
            raise ClosureError(f"frozen source hash mismatch: {directory} {name}")
        checked += 1
    checked += validate_checkpoint_copy_correction(directory, provenance)
    frozen_inputs = provenance.get("frozen_inputs")
    source_checkpoint = provenance.get("source_checkpoint")
    if frozen_inputs is not None or source_checkpoint is not None:
        if not isinstance(frozen_inputs, dict) or not frozen_inputs:
            raise ClosureError(f"continuation lacks frozen_inputs: {directory}")
        if not isinstance(source_checkpoint, dict):
            raise ClosureError(f"continuation lacks source_checkpoint: {directory}")
        for name, record in {
            **frozen_inputs,
            "source_checkpoint": source_checkpoint,
        }.items():
            if not isinstance(record, dict):
                raise ClosureError(
                    f"invalid continuation provenance record: {directory} {name}"
                )
            path = Path(record.get("path", ""))
            if (
                not path.is_file()
                or path.stat().st_size != record.get("bytes")
                or sha256(path) != record.get("sha256")
            ):
                raise ClosureError(
                    f"continuation provenance hash mismatch: {directory} {name}"
                )
            checked += 1
        if (
            config.get("steps") != 150
            or config.get("checkpoint_interval") != 10
            or config.get("resume") is not True
            or config.get("source_checkpoint") != source_checkpoint["path"]
        ):
            raise ClosureError(f"continuation config contract differs: {directory}")
        summary_path = directory / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            if summary.get("provenance") != provenance:
                raise ClosureError(
                    f"continuation summary provenance differs: {directory}"
                )
            continuation = summary.get("continuation")
            if not isinstance(continuation, dict):
                raise ClosureError(f"continuation summary lacks metadata: {directory}")
            if (
                continuation.get("source_global_step") != 50
                or continuation.get("optimizer_moments_at_source")
                != "explicitly reset for both methods"
                or continuation.get("uninterrupted_original_trajectory_claimed")
                is not False
            ):
                raise ClosureError(f"continuation reset contract differs: {directory}")
            bootstrap = continuation.get("bootstrap_re_equilibration")
            source = continuation.get("source")
            if (
                not isinstance(source, dict)
                or source.get("checkpoint") != source_checkpoint
                or source.get("inputs") != frozen_inputs
            ):
                raise ClosureError(f"continuation source evidence differs: {directory}")
            if (
                not isinstance(bootstrap, dict)
                or bootstrap.get("source_seed_reused") is not True
                or bootstrap.get("forward", {}).get("success") is not True
                or bootstrap.get("adjoint", {}).get("success") is not True
            ):
                raise ClosureError(
                    f"continuation bootstrap record differs: {directory}"
                )
        return checked
    fixture = config.get("fixture")
    if isinstance(fixture, str):
        fixture_path = Path(fixture)
        fixture_path = (
            fixture_path if fixture_path.is_absolute() else directory / fixture_path
        )
        for name, expected in provenance.get("inputs", {}).items():
            source = fixture_path / name
            if not source.is_file() or sha256(source) != expected:
                raise ClosureError(f"fixture input hash mismatch: {directory} {name}")
            checked += 1
    return checked


def validate_inventory_provenance(inventory: Path) -> int:
    document = json.loads(inventory.read_text())
    seen: set[Path] = set()
    for case in document.get("cases", []):
        endpoint = case.get("endpoint_vtu")
        if isinstance(endpoint, str):
            path = (inventory.parent / endpoint).resolve()
            if path.parent.is_dir():
                seen.add(path.parent)
        selection = case.get("endpoint_selection")
        if isinstance(selection, dict) and isinstance(selection.get("summary"), str):
            seen.add((inventory.parent / selection["summary"]).resolve().parent)
    return sum(validate_run_provenance(path) for path in seen)


def validate_modules(viewer: Path) -> int:
    pending = [viewer / "viewer.html"]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        text = path.read_text()
        for raw in ATTR.findall(text):
            ensure_local(path.parent, raw, str(path.relative_to(viewer)), root=viewer)
        if path.suffix in {".html", ".js"}:
            for raw in IMPORT.findall(without_comments(text)):
                target = ensure_local(
                    path.parent, raw, str(path.relative_to(viewer)), root=viewer
                )
                if target is not None and target.suffix == ".js":
                    pending.append(target)
    return len(seen)


def validate_markdown(report: Path) -> int:
    count = 0
    for raw in MARKDOWN.findall(report.read_text()):
        path = local(raw)
        if path is not None:
            target = (report.parent / path).resolve()
            if not target.is_file():
                raise ClosureError(f"missing Markdown local link: {raw}")
            count += 1
    return count


def validate_staged_markdown(site: Path) -> int:
    """Check every downloadable/support Markdown link from its staged location."""
    count = 0
    paths = [*(site / "records").glob("*.md"), *(site / "artifacts").rglob("*.md")]
    for path in paths:
        for raw in MARKDOWN.findall(path.read_text()):
            local_path = local(raw)
            if local_path is None:
                continue
            target = (path.parent / local_path).resolve()
            try:
                target.relative_to(site.resolve())
            except ValueError as error:
                raise ClosureError(
                    f"staged Markdown link escapes site: {path.relative_to(site)} -> {raw}"
                ) from error
            if not target.is_file():
                raise ClosureError(
                    f"staged Markdown link is missing: {path.relative_to(site)} -> {raw}"
                )
            count += 1
    return count


def validate_record_manifest(site: Path) -> None:
    """Require recorded SHA-256 values for the two downloadable archives."""
    path = site / "records" / "manifest.json"
    if not path.is_file():
        raise ClosureError("published site lacks records/manifest.json")
    document = json.loads(path.read_text())
    for name, record in document.get("archives", {}).items():
        archive = site / "records" / name
        if not archive.is_file() or record.get("sha256") != sha256(archive):
            raise ClosureError(f"record archive hash mismatch: {name}")


def validate_site(site: Path) -> int:
    count = 0
    for path in site.rglob("*.html"):
        for raw in ATTR.findall(path.read_text()):
            ensure_local(path.parent, raw, str(path.relative_to(site)), root=site)
        count += 1
    return count


def validate_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ClosureError(f"ZIP CRC failure in {path}: {bad}")
        names = archive.namelist()
        if "MANIFEST.txt" not in names:
            raise ClosureError(f"ZIP lacks MANIFEST.txt: {path}")
        listed = set(archive.read("MANIFEST.txt").decode().splitlines())
        members = {
            name
            for name in names
            if name not in {"MANIFEST.txt", "REPRODUCIBILITY_README.md"}
        }
        if listed != members:
            raise ClosureError(f"ZIP manifest mismatch: {path}")
        if (
            path.name == "reproducibility.zip"
            and "REPRODUCIBILITY_README.md" not in names
        ):
            raise ClosureError(f"reproducibility ZIP lacks provenance readme: {path}")
        member_hash = hashlib.sha256()
        for name in sorted(members):
            member_hash.update(
                name.encode() + b"\0" + hashlib.sha256(archive.read(name)).digest()
            )
        return {"members": len(members), "content_sha256": member_hash.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    viewer = args.viewer.resolve()
    if (
        not (viewer / "viewer.html").is_file()
        or not (viewer / "viewer-manifest.json").is_file()
    ):
        raise ClosureError(f"not a rendered viewer bundle: {viewer}")
    result: dict[str, Any] = {
        "viewer": str(viewer),
        "geometry": validate_geometry(viewer),
        "receipts": validate_receipts(viewer),
        "module_files": validate_modules(viewer),
    }
    if args.report is not None and args.site is None:
        result["markdown_local_links"] = validate_markdown(args.report.resolve())
    if args.inventory is not None:
        result["provenance_hashes"] = validate_inventory_provenance(
            args.inventory.resolve()
        )
    if args.site is not None:
        site = args.site.resolve()
        result["site_html_files"] = validate_site(site)
        nested_viewers = sorted(
            path.parent for path in site.rglob("viewer-manifest.json")
        )
        if not nested_viewers:
            raise ClosureError("published site has no viewer manifests")
        result["site_viewers"] = {
            str(viewer.relative_to(site)): {
                "geometry": validate_geometry(viewer),
                "module_files": validate_modules(viewer),
            }
            for viewer in nested_viewers
        }
        result["staged_markdown_local_links"] = validate_staged_markdown(site)
        validate_record_manifest(site)
        zips = sorted((site / "records").glob("*.zip"))
        if not zips:
            raise ClosureError("published site has no evidence ZIPs")
        result["zips"] = {path.name: validate_zip(path) for path in zips}
    if args.receipt is not None:
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
