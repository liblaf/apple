from __future__ import annotations

import sys
from pathlib import Path

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]

MATERIAL_REFERENCE_GROUP = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep"
)
MATERIAL_REFERENCE_SRC = MATERIAL_REFERENCE_GROUP / "src"
RUNTIME_REFERENCE_GROUP = REPO_ROOT / "exp/2026/06/17/human-face-smile-prestrain-v2"
RUNTIME_REFERENCE_SRC = RUNTIME_REFERENCE_GROUP / "src"
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
