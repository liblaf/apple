from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
TOY_HELPER_DIR = REPO_ROOT / "exp/2026/06/10/unreachable-toy-skin-tetwild/src"
if str(TOY_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(TOY_HELPER_DIR))

import _toy_skin_tetwild as toy  # noqa: E402

__all__ = ["toy"]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        msg = f"label {value!r} does not contain a usable slug"
        raise ValueError(msg)
    return slug


def resolve_recorded_path(manifest_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    manifest_path = manifest_path.resolve()
    for parent in (manifest_path.parent, *manifest_path.parents):
        candidate = parent / path
        if candidate.exists():
            return candidate
    return path


def case_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        msg = "manifest must contain a non-empty cases list"
        raise ValueError(msg)
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or "label" not in case:
            msg = "every manifest case must be an object with a label"
            raise ValueError(msg)
        label = str(case["label"])
        if label in result:
            msg = f"duplicate case label {label!r}"
            raise ValueError(msg)
        result[label] = case
    return result
