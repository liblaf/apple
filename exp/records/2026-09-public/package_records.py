#!/usr/bin/env python3
# ruff: noqa: C901, EM102, PLR0915, TRY003
"""Build a small, public September 2026 experiment-record release.

The input is intentionally an allow-list of result tables, receipts, and
synthetic/profile plots.  It excludes runtime snapshots, source copies, mesh
and checkpoint payloads, meeting material, and anatomical renders.  The script
never edits the experiment outputs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "bundles"
MAX_TEXT_BYTES = 1_100_000
TEXT_SUFFIXES = {".csv", ".json", ".txt", ".md", ".yaml", ".yml"}
SENSITIVE = re.compile(
    r"(?:/(?:home|root)/|\\\\|\b(?:10|172|192)\.(?:\d{1,3}\.){2}\d{1,3}\b|"
    r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.(?:\d{1,3}\.)\d{1,3}\b|"
    r"\b[\w.-]+\.ts\.net\b|"
    r"(?:api[_-]?key|secret|password|token)\s*[:=])",
    re.IGNORECASE,
)

# Paths are repository-relative so the release can be rebuilt from any clone.
GROUPS = {
    "activation-space-constraints": [
        "exp/2026/09/06/activation-space-constraints/data/50-analysis/analysis-summary.json",
        "exp/2026/09/06/activation-space-constraints/data/50-analysis/endpoint-metrics.csv",
        "exp/2026/09/06/activation-space-constraints/data/50-analysis/followup-endpoint-metrics.csv",
        "exp/2026/09/06/activation-space-constraints/data/50-analysis/refinement-comparison.csv",
        "exp/2026/09/06/activation-space-constraints/data/50-analysis/smoothing-heldout-comparison.csv",
        "exp/2026/09/06/activation-space-constraints/data/50-analysis/clean-convergence.png",
        "exp/2026/09/06/activation-space-constraints/data/50-analysis/noisy-convergence.png",
        "exp/2026/09/06/activation-space-constraints/data/50-analysis/smoothing-heldout-comparison.png",
    ],
    "dominant-activation-ablation": [
        "exp/2026/09/14/dominant-activation-ablation/data/90-released-axes/summary.json",
        "exp/2026/09/14/dominant-activation-ablation/data/94-released-axes-verification/receipt.json",
        "exp/2026/09/14/dominant-activation-ablation/data/98-aligned-errors/summary.json",
    ],
    "activation-direction-smoothness": [
        "exp/2026/09/15/activation-direction-smoothness/data/30-figures/manifest.json",
        "exp/2026/09/15/activation-direction-smoothness/data/30-figures/tuning-tradeoff.png",
        "exp/2026/09/15/activation-direction-smoothness/data/30-figures/loss-and-roughness.png",
        "exp/2026/09/15/activation-direction-smoothness/data/150-reset-figures/reset-vs-no-reset-history.png",
        "exp/2026/09/15/activation-direction-smoothness/data/190-completed-height-figures/free-activation-height-comparison-16x9-preview.png",
    ],
    "gradient-only-profile": [
        "exp/2026/09/19/gradient-only-profile/data/10-gradient/summary.json",
        "exp/2026/09/19/gradient-only-profile/data/30-figures/summary.csv",
        "exp/2026/09/19/gradient-only-profile/data/30-figures/summary.json",
        "exp/2026/09/19/gradient-only-profile/data/30-figures/final-shape-metrics.png",
        "exp/2026/09/19/gradient-only-profile/data/30-figures/profiles-all-cases.png",
        "exp/2026/09/19/gradient-only-profile/data/30-figures/slope-error-profiles.png",
    ],
    "normal-matching-profile": [
        "exp/2026/09/21/normal-matching-profile/data/10-comparison/summary.json",
        "exp/2026/09/21/normal-matching-profile/data/30-figures/selection-and-endpoints.json",
        "exp/2026/09/21/normal-matching-profile/data/30-figures/summary.json",
        "exp/2026/09/21/normal-matching-profile/data/30-figures/shared-step-profile-comparison.png",
        "exp/2026/09/21/normal-matching-profile/data/30-figures/objective-and-projected-gradient-histories.png",
    ],
    "solver-performance": [
        "exp/2026/09/22/solver-performance/data/neutral-ablation-full-001/summary.json",
        "exp/2026/09/22/solver-performance/data/neutral-adaptive-ipc-final-fixed-002/summary.json",
        "exp/2026/09/22/solver-performance/data/adaptive-pncg-cold-003/summary.json",
    ],
    "new-neutral": [
        "exp/2026/09/23/new-neutral/data/forward-isfixed-001/summary.json",
        "exp/2026/09/23/new-neutral/data/forward-repaired-reference-001/summary.json",
        "exp/2026/09/23/new-neutral/data/review-isfixed-001/mouthopen-continuation/receipt.json",
        "exp/2026/09/23/new-neutral/data/pose-jump-002/summary.json",
        "exp/2026/09/23/new-neutral/data/mouthopen-joint-constraints-001/attempt-00/summary.json",
    ],
    "joint-activation-material-mandible": [
        "exp/2026/09/21/joint-activation-material-mandible/data/frozen-neutral-004/manifest.json",
        "exp/2026/09/21/joint-activation-material-mandible/data/neutral-convergence-010-contact-metric-bfgs-segment-003/summary.json",
        "exp/2026/09/21/joint-activation-material-mandible/data/contact-validation-newton/summary.json",
        "exp/2026/09/21/joint-activation-material-mandible/data/expression-fitting-008/summary.json",
        "exp/2026/09/21/joint-activation-material-mandible/data/face-gradient-validation-contact-newton/summary.json",
        "exp/2026/09/21/joint-activation-material-mandible/data/precise-hvp-expression-validation-001/summary.json",
    ],
    "mouthopen-activation": [
        "exp/2026/09/29/mouthopen-activation/data/91-smile-mouthopen-transition-003/summary.json",
        "exp/2026/09/29/mouthopen-activation/data/92-smile-mouthopen-transition-render/manifest.json",
    ],
    "collision-off-expressions": [
        "exp/2026/09/30/collision-off-expressions/data/smile-baseline-resolution-001/summary.json",
        "exp/2026/09/30/collision-off-expressions/data/mouthopen-probe-001/summary.json",
        "exp/2026/09/30/collision-off-expressions/data/feasible-descent-partial-2313.json",
    ],
    "projected-newton-neutral": [
        "exp/2026/09/30/projected-newton-neutral/data/forward-exact-control-001/summary.json",
    ],
    "stage3-mouthopen-smile-contact": [
        "exp/2026/09/30/stage3-mouthopen-smile-contact-comparison/data/22-ten-frame-collision-off/summary.json",
    ],
    "mouthopen-smile-collisions": [
        "exp/2026/09/30/mouthopen-smile-collisions/data/10-contact-checks-005/summary.json",
        "exp/2026/09/30/mouthopen-smile-collisions/data/17-contact-set-reproducibility/summary.json",
        "exp/2026/09/30/mouthopen-smile-collisions/data/20-contact-transition/summary.json",
        "exp/2026/09/30/mouthopen-smile-collisions/data/50-fixed-reference/summary.json",
        "exp/2026/09/30/mouthopen-smile-collisions/data/54-fixed-contact-audit-003/summary.json",
        "exp/2026/09/30/mouthopen-smile-collisions/data/72-linear-cap-pilot/summary.json",
    ],
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unsafe_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    if path.stat().st_size > MAX_TEXT_BYTES:
        return "text file exceeds the public-record limit"
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "text candidate is not UTF-8"
    return "sensitive text pattern" if SENSITIVE.search(text) else None


def sanitized_json(path: Path) -> object | None:
    """Keep public scalar/numeric evidence while removing operational fields."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    banned_key = re.compile(
        r"(?:path|directory|dir$|file|host|argv|command|credential|secret|"
        r"token|password|snapshot|source|log|cache)",
        re.IGNORECASE,
    )

    def clean(item: object, depth: int = 0) -> object | None:
        if depth > 12:
            return "<depth-limit>"
        if item is None or isinstance(item, bool | int | float):
            return item
        if isinstance(item, str):
            return None if SENSITIVE.search(item) else item
        if isinstance(item, list):
            kept = [clean(x, depth + 1) for x in item[:256]]
            kept = [x for x in kept if x is not None]
            if len(item) > 256:
                kept.append("<truncated-list>")
            return kept
        if isinstance(item, dict):
            return {
                key: cleaned
                for key, child in item.items()
                if isinstance(key, str)
                and not banned_key.search(key)
                and not SENSITIVE.search(key)
                and (cleaned := clean(child, depth + 1)) is not None
            }
        return None

    return clean(value)


def normalized_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Remove filesystem timestamps and owners from bundle headers."""
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def bundle_member(group: str, source_name: str) -> str:
    return f"{group}/source/{source_name}"


def coverage_inventory(
    selected: list[dict[str, object]], excluded: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Report the audited data scope, including studies outside the allow-list."""
    selected_by_group = {str(x["group"]) for x in selected}
    excluded_by_group = {str(x["group"]) for x in excluded}
    studies: list[dict[str, object]] = []
    for data_dir in sorted(ROOT.glob("exp/2026/09/*/data")):
        files = [path for path in data_dir.rglob("*") if path.is_file()]
        studies.append(
            {
                "study": str(data_dir.parent.relative_to(ROOT)),
                "data_files": len(files),
                "data_bytes": sum(path.stat().st_size for path in files),
                "allowlisted_group": data_dir.parent.name
                in selected_by_group | excluded_by_group,
            }
        )
    return studies


def verify_members(archive: Path, expected: list[dict[str, object]]) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        for item in expected:
            member = str(item["bundle_member"])
            if member not in names:
                raise RuntimeError(f"missing bundle member {member}")
            handle = tar.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read bundle member {member}")
            h = hashlib.sha256(handle.read()).hexdigest()
            if h != item["member_sha256"]:
                raise RuntimeError(f"bundle digest mismatch for {member}")


def main() -> int:
    if not (ROOT / ".git").exists():
        raise RuntimeError(f"expected repository root at {ROOT}")
    for names in GROUPS.values():
        for name in names:
            if not (ROOT / name).is_file():
                raise FileNotFoundError(f"allow-listed source is missing: {name}")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    selected: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    derived: list[dict[str, object]] = []
    expected_by_group: dict[str, list[dict[str, object]]] = {}
    for group, names in GROUPS.items():
        stage = OUT / group
        stage.mkdir()
        expected_by_group[group] = []
        for name in names:
            source = ROOT / name
            reason = unsafe_text(source)
            item = {
                "group": group,
                "path": name,
                "bytes": source.stat().st_size,
                "sha256": digest(source),
            }
            if reason:
                item["reason"] = reason
                excluded.append(item)
                cleaned = sanitized_json(source)
                if cleaned is not None:
                    derived_name = f"derived/{item['sha256'][:12]}.sanitized.json"
                    derived_path = stage / derived_name
                    derived_path.parent.mkdir(parents=True, exist_ok=True)
                    derived_record = {
                        "group": group,
                        "source_path": name,
                        "source_sha256": item["sha256"],
                        "transformation": "JSON scalar/numeric evidence retained; operational keys and sensitive strings omitted",
                        "bundle_member": f"{group}/{derived_name}",
                        "data": cleaned,
                    }
                    derived_path.write_text(json.dumps(derived_record, indent=2) + "\n")
                    if unsafe_text(derived_path):
                        raise RuntimeError(f"sanitization failed for {name}")
                    derived_record["member_sha256"] = digest(derived_path)
                    derived_record["member_bytes"] = derived_path.stat().st_size
                    derived.append(derived_record)
                    expected_by_group[group].append(derived_record)
                continue
            destination = stage / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            item["bundle_member"] = bundle_member(group, name)
            item["member_sha256"] = digest(destination)
            selected.append(item)
            expected_by_group[group].append(item)
        archive = OUT / f"{group}.tar.gz"
        # A fixed gzip timestamp makes unchanged inputs produce unchanged bundles.
        with (
            archive.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
            tarfile.open(fileobj=zipped, mode="w") as tar,
        ):
            tar.add(stage, arcname=group, filter=normalized_tarinfo)
        shutil.rmtree(stage)
        verify_members(archive, expected_by_group[group])
    for item in selected + derived:
        item["bundle"] = f"{item['group']}.tar.gz"
    (OUT / "selected-manifest.json").write_text(json.dumps(selected, indent=2) + "\n")
    (OUT / "excluded-manifest.json").write_text(json.dumps(excluded, indent=2) + "\n")
    derived_manifest = [
        {key: value for key, value in item.items() if key != "data"} for item in derived
    ]
    (OUT / "derived-sanitized-manifest.json").write_text(
        json.dumps(derived_manifest, indent=2) + "\n"
    )
    (OUT / "coverage-inventory.json").write_text(
        json.dumps(coverage_inventory(selected, excluded), indent=2) + "\n"
    )
    with (OUT / "checksums.sha256").open("w", encoding="utf-8") as f:
        for path in sorted(OUT.glob("*.tar.gz")):
            f.write(f"{digest(path)}  {path.name}\n")
    inventory = {
        "scope": "exp/2026/09/**/data",
        "initial_selection_boundary": "allow-listed result tables, receipts, and synthetic/profile plots; no runtime archives, copied code, raw meshes, checkpoints, meeting/proposal material, downloaded anatomy, or model-face renders",
        "selected_files": len(selected),
        "derived_sanitized_records": len(derived),
        "selected_source_bytes": sum(int(x["bytes"]) for x in selected),
        "excluded_candidates": len(excluded),
        "bundle_bytes": sum(p.stat().st_size for p in OUT.glob("*.tar.gz")),
        "full_text_scan": "all allow-listed text and derived JSON scanned for local paths, private and Tailnet addresses, tailnet names, and credential markers",
    }
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
