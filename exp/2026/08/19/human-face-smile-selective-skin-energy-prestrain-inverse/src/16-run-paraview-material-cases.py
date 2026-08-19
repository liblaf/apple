# ruff: noqa: C901, EM101, EM102, TRY003

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pydantic_settings as ps

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DESIGN = "paraview-6.1.1-four-case-skin-material-sheet-runner"
GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
RUNNER = Path(__file__).resolve()

# The wrapper and the pure pvbatch script have separate source-level blockers.
# Both must be reviewed and flipped before rendering can begin.
EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True
APPROVAL_BLOCKER = (
    "NO-GO: ParaView material runner awaits static review; do not execute until "
    "this source-level blocker is explicitly changed"
)

MANIFEST = GROUP_DIR / "data/10-prepared-material-cases-manifest.json"
RENDERER = GROUP_DIR / "src/15-render-material-cases-paraview.py"
PVBATCH = Path("/usr/bin/pvbatch")
OUTPUT_SCREENSHOT = GROUP_DIR / "data/15-paraview-material-cases.png"
OUTPUT_STATE = GROUP_DIR / "data/15-paraview-material-cases.pvsm"
OUTPUT_RECEIPT = GROUP_DIR / "data/15-paraview-material-cases-receipt.json"

EXPECTED_MATERIAL_SCHEMA_VERSION = 1
EXPECTED_MATERIAL_DESIGN = (
    "corrected-isface-four-case-selective-e000-c020-inverse-materials"
)
EXPECTED_CASE_ORDER = ("H0P0", "H0P1", "H1P1", "H1P0")
EXPECTED_PARAVIEW_VERSION = "6.1.1"
EXPECTED_RENDERER_NORMALIZED_SHA256 = (
    "2618c6af14d5ee5adfccfcdea789a396235f30e777dbd2c6792c9e1b2deaacbb"
)
EXPECTED_PVBATCH_SIZE_BYTES = 18_608
EXPECTED_PVBATCH_SHA256 = (
    "be482a75b1e52a8b5d9df6c5687c743cc0b5312e30916622d54652a998eb8871"
)
RENDERER_FALSE_MARKER = b"PARAVIEW_RENDER_APPROVED_AFTER_STATIC_REVIEW = False"
RENDERER_TRUE_MARKER = b"PARAVIEW_RENDER_APPROVED_AFTER_STATIC_REVIEW = True"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_manifest: Path = cherries.input(MANIFEST)
    input_renderer: Path = cherries.input(RENDERER)
    output_screenshot: Path = cherries.output(
        "15-paraview-material-cases.png", mkdir=True
    )
    output_state: Path = cherries.output("15-paraview-material-cases.pvsm", mkdir=True)
    output_receipt: Path = cherries.output(
        "15-paraview-material-cases-receipt.json", mkdir=True
    )


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


def _require_exact_path(actual: Path, expected: Path, *, name: str) -> None:
    if actual.resolve() != expected.resolve():
        raise ValueError(f"{name} must remain {expected}, got {actual}")


def _normalized_renderer_sha256(path: Path) -> str:
    source = path.read_bytes()
    false_count = source.count(RENDERER_FALSE_MARKER)
    true_count = source.count(RENDERER_TRUE_MARKER)
    if false_count + true_count != 1:
        raise ValueError(
            "renderer must contain exactly one recognized source approval assignment"
        )
    normalized = source.replace(RENDERER_TRUE_MARKER, RENDERER_FALSE_MARKER)
    return hashlib.sha256(normalized).hexdigest()


def _validate_config(cfg: Config) -> None:
    if not EXECUTION_APPROVED_AFTER_STATIC_REVIEW:
        raise RuntimeError(APPROVAL_BLOCKER)
    exact_paths = (
        (cfg.input_manifest, MANIFEST, "input_manifest"),
        (cfg.input_renderer, RENDERER, "input_renderer"),
        (cfg.output_screenshot, OUTPUT_SCREENSHOT, "output_screenshot"),
        (cfg.output_state, OUTPUT_STATE, "output_state"),
        (cfg.output_receipt, OUTPUT_RECEIPT, "output_receipt"),
    )
    for actual, expected, name in exact_paths:
        _require_exact_path(actual, expected, name=name)
    outputs = (OUTPUT_SCREENSHOT, OUTPUT_STATE, OUTPUT_RECEIPT)
    stale = [
        str(path)
        for path in (*outputs, *(_temporary_path(path) for path in outputs))
        if path.exists()
    ]
    if stale:
        raise FileExistsError(
            "refusing to overwrite ParaView outputs or partial files: " + str(stale)
        )


def _validate_renderer() -> dict[str, Any]:
    if not RENDERER.is_file():
        raise FileNotFoundError(f"missing reviewed ParaView renderer: {RENDERER}")
    normalized_sha = _normalized_renderer_sha256(RENDERER)
    if normalized_sha != EXPECTED_RENDERER_NORMALIZED_SHA256:
        raise ValueError(
            "ParaView renderer changed beyond its approval bit: "
            f"{normalized_sha} != {EXPECTED_RENDERER_NORMALIZED_SHA256}"
        )
    return {
        "path": str(RENDERER),
        **_file_identity(RENDERER),
        "normalized_static_review_sha256": normalized_sha,
    }


def _validate_pvbatch() -> dict[str, Any]:
    if not PVBATCH.is_file():
        raise FileNotFoundError(f"missing pinned pvbatch executable: {PVBATCH}")
    actual = _file_identity(PVBATCH)
    expected = {
        "size_bytes": EXPECTED_PVBATCH_SIZE_BYTES,
        "sha256": EXPECTED_PVBATCH_SHA256,
    }
    if actual != expected:
        raise ValueError(f"pvbatch executable identity changed: {actual} != {expected}")
    completed = subprocess.run(
        [str(PVBATCH), "--version"],
        cwd=GROUP_DIR,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    version_output = completed.stdout.strip()
    if version_output != f"paraview version {EXPECTED_PARAVIEW_VERSION}":
        raise RuntimeError(f"pvbatch version changed: {version_output!r}")
    return {
        "path": str(PVBATCH),
        **actual,
        "version": EXPECTED_PARAVIEW_VERSION,
        "version_output": version_output,
    }


def _validate_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(path)
    expected = {
        "schema_version": EXPECTED_MATERIAL_SCHEMA_VERSION,
        "design": EXPECTED_MATERIAL_DESIGN,
        "complete": True,
        "case_order": list(EXPECTED_CASE_ORDER),
    }
    changed = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if changed:
        raise ValueError(f"material manifest contract changed: {changed}")
    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        raise TypeError("material manifest lacks producer identity")
    producer_path = Path(str(producer.get("path")))
    producer_expected = producer.get("file_identity", producer)
    if not producer_path.is_file() or not isinstance(producer_expected, dict):
        raise FileNotFoundError("material manifest producer is unavailable")
    producer_actual = _file_identity(producer_path)
    if producer_actual != {
        "size_bytes": int(producer_expected.get("size_bytes", -1)),
        "sha256": str(producer_expected.get("sha256")),
    }:
        raise ValueError("material producer changed after preparation")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 4:
        raise ValueError("material manifest does not contain exactly four cases")
    cases_by_id = {row.get("case_id"): row for row in raw_cases}
    if set(cases_by_id) != set(EXPECTED_CASE_ORDER):
        raise ValueError("material case identifiers changed")
    identities: list[dict[str, Any]] = []
    for case_id in EXPECTED_CASE_ORDER:
        row = cases_by_id[case_id]
        if row.get("validation") != {"ok": True, "errors": []}:
            raise ValueError(f"{case_id} material validation is not clean")
        skin = row.get("skin")
        if not isinstance(skin, dict):
            raise TypeError(f"{case_id} material skin row is malformed")
        skin_path = Path(str(skin.get("path")))
        identity = skin.get("file_identity")
        if not skin_path.is_file() or not isinstance(identity, dict):
            raise FileNotFoundError(f"{case_id} material skin is unavailable")
        actual = _file_identity(skin_path)
        expected_identity = {
            "size_bytes": int(identity.get("size_bytes", -1)),
            "sha256": str(identity.get("sha256")),
        }
        if actual != expected_identity:
            raise ValueError(f"{case_id} material skin changed")
        identities.append({"case_id": case_id, "path": str(skin_path), **actual})
    return manifest, identities


def _run_renderer(cfg: Config) -> None:
    command = [
        str(PVBATCH),
        str(cfg.input_renderer),
        "--manifest",
        str(cfg.input_manifest),
        "--screenshot",
        str(cfg.output_screenshot),
        "--state",
        str(cfg.output_state),
        "--receipt",
        str(cfg.output_receipt),
    ]
    logger.info("Running native ParaView command: %s", command)
    process = subprocess.Popen(
        command,
        cwd=GROUP_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.stdout is None:
        raise RuntimeError("pvbatch stdout pipe was not created")
    for line in process.stdout:
        logger.info("pvbatch | %s", line.rstrip())
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"pvbatch material renderer exited with {return_code}")


def _validate_outputs(
    cfg: Config, *, manifest_identity: dict[str, Any]
) -> dict[str, Any]:
    for name, path in (
        ("screenshot", cfg.output_screenshot),
        ("state", cfg.output_state),
        ("receipt", cfg.output_receipt),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"ParaView output {name} is missing or empty: {path}")
    if cfg.output_screenshot.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("ParaView screenshot is not a PNG")
    state_prefix = cfg.output_state.read_bytes()[:512].lstrip()
    if b"<ParaView" not in state_prefix and b"<ServerManagerState" not in state_prefix:
        raise ValueError("ParaView state lacks an XML state root")
    receipt = _read_json(cfg.output_receipt)
    expected_header = {
        "schema_version": 1,
        "design": "paraview-6.1.1-four-case-skin-material-sheet",
        "complete": True,
        "paraview_version": EXPECTED_PARAVIEW_VERSION,
        "case_order": list(EXPECTED_CASE_ORDER),
        "color_association": "CELLS",
    }
    changed = {
        key: (receipt.get(key), value)
        for key, value in expected_header.items()
        if receipt.get(key) != value
    }
    if changed:
        raise ValueError(f"ParaView receipt contract changed: {changed}")
    if receipt.get("manifest") != {
        "path": str(cfg.input_manifest),
        **manifest_identity,
    }:
        raise ValueError("ParaView receipt manifest identity changed")
    expected_outputs = {
        "screenshot": {
            "path": str(cfg.output_screenshot),
            **_file_identity(cfg.output_screenshot),
        },
        "state": {"path": str(cfg.output_state), **_file_identity(cfg.output_state)},
    }
    if receipt.get("outputs") != expected_outputs:
        raise ValueError("ParaView receipt output identities changed")
    return receipt


def main(cfg: Config) -> None:
    _validate_config(cfg)
    renderer_before = _validate_renderer()
    pvbatch = _validate_pvbatch()
    _manifest, cases_before = _validate_manifest(cfg.input_manifest)
    manifest_identity = _file_identity(cfg.input_manifest)
    _run_renderer(cfg)
    receipt = _validate_outputs(cfg, manifest_identity=manifest_identity)
    renderer_after = _validate_renderer()
    if renderer_after != renderer_before:
        raise RuntimeError("ParaView renderer changed during execution")
    if _file_identity(cfg.input_manifest) != manifest_identity:
        raise RuntimeError("material manifest changed during ParaView execution")
    _manifest_after, cases_after = _validate_manifest(cfg.input_manifest)
    if cases_after != cases_before:
        raise RuntimeError("material case files changed during ParaView execution")
    cherries.log_metrics(
        {
            "paraview/version_major": 6,
            "paraview/version_minor": 1,
            "paraview/version_patch": 1,
            "paraview/cases": len(cases_after),
            "paraview/panels": 8,
            "paraview/screenshot_bytes": cfg.output_screenshot.stat().st_size,
            "paraview/state_bytes": cfg.output_state.stat().st_size,
        }
    )
    logger.info(
        "ParaView material sheet complete: version=%s screenshot=%s state=%s receipt=%s",
        pvbatch["version"],
        cfg.output_screenshot,
        cfg.output_state,
        receipt["design"],
    )


if __name__ == "__main__":
    cherries.main(main)
