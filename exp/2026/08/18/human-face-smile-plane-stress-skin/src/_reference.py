from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]

MATERIAL_REFERENCE_GROUP = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep"
)
MATERIAL_REFERENCE_SRC = MATERIAL_REFERENCE_GROUP / "src"
RUNTIME_REFERENCE_GROUP = REPO_ROOT / "exp/2026/06/17/human-face-smile-prestrain-v2"
RUNTIME_REFERENCE_SRC = RUNTIME_REFERENCE_GROUP / "src"
LEGACY_VALIDATION_GROUP = (
    REPO_ROOT / "exp/2026/08/18/human-face-smile-exaggerated-material-screen"
)
LEGACY_PREPARE = LEGACY_VALIDATION_GROUP / "src/10-prepare-exaggerated-materials.py"
LEGACY_INVERSE = (
    LEGACY_VALIDATION_GROUP / "src/20-inverse-exaggerated-material-screen.py"
)
KOITER_IMPLEMENTATION = REPO_ROOT / "src/liblaf/apple/warp/fem/_koiter.py"
VOLUME_LAME_IMPLEMENTATION = RUNTIME_REFERENCE_SRC / "_human_face_mesh.py"
VOLUME_FORWARD_IMPLEMENTATION = RUNTIME_REFERENCE_SRC / "_human_face_forward.py"

SOURCE_MANIFEST = MATERIAL_REFERENCE_GROUP / "data/10-material-candidates-manifest.json"
SOURCE_SKIN = (
    MATERIAL_REFERENCE_GROUP / "data/10-material-candidates/skin-e100-p000.vtp"
)
PREPARED_MESH = RUNTIME_REFERENCE_GROUP / "data/10-human-face-prepared.vtu"

SOURCE_MANIFEST_SIZE_BYTES = 79_539
SOURCE_MANIFEST_SHA256 = (
    "f5b4a16183171bf68336db748bbd547621def8c9c3aa536fc4ee821869be3cd9"
)
SOURCE_SKIN_SIZE_BYTES = 38_742_137
SOURCE_SKIN_SHA256 = "ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f"
SOURCE_TOPOLOGY_SHA256 = (
    "ae1261614865d2fa39e674994043e25fba85db3c5f2622fe588e676213ca5aff"
)
SOURCE_MATERIAL_SHA256 = (
    "7fdd896528f867d35b5740763030c297ff53abad33590abd610edaa8689da6b5"
)
SOURCE_SOLVER_SHA256 = (
    "b9acd84d49b0f9ae6e82c4aad773c222dd5e2c24773f7cdc9712b4d3c6f0291f"
)

LEGACY_PREPARE_SHA256 = (
    "ab48e6e5e403b62f247f8eaaa45a2e071f4307dc0886d60b7c3cfe85667567f4"
)
LEGACY_INVERSE_SHA256 = (
    "78f7297265d67c2e3e937b185f7c89d75cf95a4c36a0c2f2c0b753329c721392"
)
KOITER_IMPLEMENTATION_SHA256 = (
    "f7b7c9547c82976a130a88faf8df5172312309238c2b0cf8c8e762e1ec463e8c"
)
VOLUME_LAME_IMPLEMENTATION_SHA256 = (
    "f1e1cdc806273c4ce5a37e52e3032d357b44bfd201de3fc58c35d793d11454bc"
)
VOLUME_FORWARD_IMPLEMENTATION_SHA256 = (
    "2d0ff39b13555300c000e6dd43e16c274752263b703746ad8174072033819e03"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_sha256(path: Path, expected: str, *, name: str) -> str:
    if not path.is_file():
        msg = f"missing pinned {name}: {path}"
        raise FileNotFoundError(msg)
    actual = file_sha256(path)
    if actual != expected:
        msg = f"{name} SHA-256 mismatch: expected {expected}, got {actual}"
        raise ValueError(msg)
    return actual


def enable_reference_modules() -> None:
    for name, source in (
        ("material-reference", MATERIAL_REFERENCE_SRC),
        ("runtime-reference", RUNTIME_REFERENCE_SRC),
    ):
        if not source.is_dir():
            msg = f"missing {name} source directory: {source}"
            raise FileNotFoundError(msg)
        reference = str(source)
        if reference not in sys.path:
            # The 08/17 compatibility config must precede the 06/17 runtime.
            sys.path.append(reference)


def load_pinned_module(
    path: Path, expected_sha256: str, *, module_name: str
) -> tuple[ModuleType, str]:
    identity = require_file_sha256(
        path, expected_sha256, name=f"reference module {module_name}"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load pinned reference module: {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, identity
