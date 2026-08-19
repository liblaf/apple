from __future__ import annotations

# The wrapper keeps experiment provenance and intentionally reports detailed guards.
# ruff: noqa: C901, EM101, EM102, TRY003, TRY004
import hashlib
import json
import logging
import struct
import subprocess
from pathlib import Path
from typing import Any

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
EXPECTED_PARAVIEW_VERSION = "6.1.1"
PVBATCH = Path("/usr/bin/pvbatch")
GROUP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = GROUP_DIR / "data"
ANALYSIS = DATA_DIR / "30-selective-skin-prestrain-analysis.json"
ANALYZER = Path(__file__).with_name("30-analyze-selective-skin-prestrain.py")
RENDERER = Path(__file__).with_name("35-render-selective-skin-prestrain-paraview.py")
INPUT_ROOT = DATA_DIR / "30-paraview-inputs"
OUTPUT_DIR = DATA_DIR / "35-paraview-results"
OUTPUT_MANIFEST = DATA_DIR / "36-paraview-render-manifest.json"

# A later isolated approval edit must set all three hashes from completed, reviewed
# artifacts and change only this boolean. There is no CLI bypass.
PARAVIEW_EXECUTION_APPROVED_AFTER_ANALYSIS_REVIEW = True
EXPECTED_ANALYSIS_SHA256 = (
    "120b03b02cec7e30dc4ecabb8d3ac8197168a347d405bacd78dceb7f8af2d520"
)
EXPECTED_ANALYZER_SHA256 = (
    "d3225740992d57edfc852026416fe11c1bd4ab94c13c955debb4323c7c280548"
)
EXPECTED_RENDERER_SHA256 = (
    "3cd737c45f377c6cee0ebc2990a41e9112baf5e379d98f3d20e0f2fbd323e737"
)
EXPECTED_PVBATCH_SIZE_BYTES = 18_608
EXPECTED_PVBATCH_SHA256 = (
    "be482a75b1e52a8b5d9df6c5687c743cc0b5312e30916622d54652a998eb8871"
)

CASE_ORDER = ("H0P0", "H0P1", "H1P1", "H1P0")
COHORT_ORDER = ("terminal", "baseline-fidelity", "common-tau")
MODE_ORDER = ("geometry", "normal-residual")
IMAGE_SIZE = (4000, 3000)


class Config(cherries.BaseConfig):
    input_analysis: Path = cherries.input(ANALYSIS)
    output_manifest: Path = cherries.output(OUTPUT_MANIFEST, mkdir=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _require_hash(path: Path, expected: str, *, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash changed: {actual} != {expected}")


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _validate_config(cfg: Config) -> None:
    if Path(cfg.input_analysis).resolve() != ANALYSIS.resolve():
        raise ValueError("analysis input cannot be overridden")
    if Path(cfg.output_manifest).resolve() != OUTPUT_MANIFEST.resolve():
        raise ValueError("ParaView manifest output cannot be overridden")
    stale = [path for path in (OUTPUT_DIR, OUTPUT_MANIFEST) if path.exists()]
    if stale:
        raise FileExistsError(f"refusing stale ParaView outputs: {stale}")
    if not PARAVIEW_EXECUTION_APPROVED_AFTER_ANALYSIS_REVIEW:
        raise RuntimeError(
            "NO-GO: ParaView rendering awaits completed analysis review and isolated source approval"
        )
    _require_hash(ANALYSIS, EXPECTED_ANALYSIS_SHA256, label="analysis JSON")
    _require_hash(ANALYZER, EXPECTED_ANALYZER_SHA256, label="numeric analyzer")
    _require_hash(RENDERER, EXPECTED_RENDERER_SHA256, label="ParaView renderer")
    approved_marker = (
        "PARAVIEW_RENDERER_EXECUTION_APPROVED_AFTER_ANALYSIS_REVIEW = True"
    )
    if approved_marker not in RENDERER.read_text(encoding="utf-8"):
        raise RuntimeError("reviewed ParaView renderer is still source-blocked")


def _paraview_version() -> str:
    pvbatch_identity = _identity(PVBATCH)
    expected_identity = {
        "path": str(PVBATCH.resolve()),
        "size_bytes": EXPECTED_PVBATCH_SIZE_BYTES,
        "sha256": EXPECTED_PVBATCH_SHA256,
    }
    if pvbatch_identity != expected_identity:
        raise ValueError("pvbatch executable identity changed")
    completed = subprocess.run(
        [str(PVBATCH), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    combined = f"{completed.stdout}\n{completed.stderr}".strip()
    if not combined.endswith(EXPECTED_PARAVIEW_VERSION):
        raise RuntimeError(
            f"requires ParaView {EXPECTED_PARAVIEW_VERSION}; got {combined!r}"
        )
    return EXPECTED_PARAVIEW_VERSION


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"{path} is not a valid PNG header")
    return struct.unpack(">II", header[16:24])


def _validate_analysis(analysis: dict[str, Any]) -> None:
    if not bool(analysis.get("complete")):
        raise ValueError("numeric analysis is incomplete")
    if analysis.get("case_order") != list(CASE_ORDER):
        raise ValueError("numeric analysis case order changed")
    paraview = analysis.get("paraview")
    if not isinstance(paraview, dict):
        raise ValueError("numeric analysis has no ParaView contract")
    if paraview.get("cohort_order") != list(COHORT_ORDER):
        raise ValueError("numeric analysis cohort order changed")
    if paraview.get("renderer") != (
        "ParaView 6.1.1 only; Matplotlib geometry render is prohibited"
    ):
        raise ValueError("numeric analysis renderer contract changed")
    for cohort in COHORT_ORDER:
        inputs = paraview["inputs"][cohort]
        if set(inputs) != set(CASE_ORDER):
            raise ValueError(f"{cohort} ParaView input case identifiers changed")
        for case_id in CASE_ORDER:
            path = Path(str(inputs[case_id]["path"])).resolve()
            if INPUT_ROOT.resolve() not in path.parents:
                raise ValueError(f"{cohort}/{case_id} input escapes the pinned root")
            actual = _identity(path)
            expected = {
                "path": str(path),
                "size_bytes": int(inputs[case_id]["size_bytes"]),
                "sha256": str(inputs[case_id]["sha256"]),
            }
            if actual != expected:
                raise ValueError(f"{cohort}/{case_id} ParaView input changed")


def _expected_outputs() -> list[Path]:
    return [
        OUTPUT_DIR / f"35-paraview-{cohort}-{mode}.{suffix}"
        for cohort in COHORT_ORDER
        for mode in MODE_ORDER
        for suffix in ("png", "pvsm")
    ]


def _validate_outputs() -> list[dict[str, Any]]:
    expected = _expected_outputs()
    actual = sorted(OUTPUT_DIR.iterdir())
    if actual != sorted(expected):
        raise ValueError(
            f"ParaView output inventory changed: expected {expected}, got {actual}"
        )
    rows: list[dict[str, Any]] = []
    for path in expected:
        if path.suffix == ".png":
            if _png_size(path) != IMAGE_SIZE:
                raise ValueError(f"{path} does not have the locked 4000x3000 size")
            if path.stat().st_size < 100_000:
                raise ValueError(f"{path} is suspiciously small")
        else:
            head = path.read_text(encoding="utf-8", errors="strict")[:512]
            if "ParaView" not in head and "ServerManagerState" not in head:
                raise ValueError(f"{path} is not a recognizable ParaView state")
        rows.append(_identity(path))
        cherries.log_output(path)
    return rows


def main(cfg: Config) -> None:
    _validate_config(cfg)
    analysis = _read_json(cfg.input_analysis)
    _validate_analysis(analysis)
    version = _paraview_version()
    command = [
        str(PVBATCH),
        str(RENDERER.resolve()),
        "--analysis",
        str(ANALYSIS.resolve()),
        "--input-root",
        str(INPUT_ROOT.resolve()),
        "--output-dir",
        str(OUTPUT_DIR.resolve()),
    ]
    logger.info("Running pinned ParaView renderer: %s", command)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=GROUP_DIR,
    )
    if completed.stdout:
        logger.info("pvbatch stdout:\n%s", completed.stdout)
    if completed.stderr:
        logger.info("pvbatch stderr:\n%s", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"pvbatch failed with exit code {completed.returncode}")
    _require_hash(ANALYSIS, EXPECTED_ANALYSIS_SHA256, label="analysis JSON postrun")
    _require_hash(ANALYZER, EXPECTED_ANALYZER_SHA256, label="numeric analyzer postrun")
    _require_hash(RENDERER, EXPECTED_RENDERER_SHA256, label="ParaView renderer postrun")
    _validate_analysis(_read_json(ANALYSIS))
    outputs = _validate_outputs()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "design": "meeting-authoritative-paraview-selective-skin-prestrain-inverse",
        "complete": True,
        "paraview_version": version,
        "pvbatch": str(PVBATCH),
        "pvbatch_identity": _identity(PVBATCH),
        "command": command,
        "analysis": _identity(ANALYSIS),
        "analyzer": _identity(ANALYZER),
        "renderer": _identity(RENDERER),
        "case_order": list(CASE_ORDER),
        "cohort_order": list(COHORT_ORDER),
        "view_order": ["front", "30-degree", "mouth", "eye-cheek+x"],
        "modes": list(MODE_ORDER),
        "image_size": list(IMAGE_SIZE),
        "outputs": outputs,
        "authority": (
            "geometry and target-normal-residual fixed-view images and states were "
            "generated by ParaView 6.1.1; Matplotlib is used only for trajectories "
            "and Pareto plots"
        ),
    }
    cfg.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote ParaView render manifest to %s", cfg.output_manifest)


if __name__ == "__main__":
    # ParaView reporting must never stage or commit this dirty research
    # worktree.  Keep Local + Logging provenance through the explicit debug
    # profile, whose Git plugin is non-committing.
    cherries.main(main, profile="debug")
