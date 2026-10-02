# Copyright 2026 liblaf
"""Download and verify the pinned Z-Anatomy inputs used by this experiment."""

# Explicit failure messages are useful when a remote or cached artifact drifts.
# ruff: noqa: EM102, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = "https://github.com/LluisV/Z-Anatomy"
COMMIT = "6c7f9016bd5899ac8edafd31b9900c151df42ed6"
MODEL_FILE_COMMIT = "4ef7d59f4a2cc002f65ebd2bbb395ae4df3a5faf"
RAW_BASE = f"https://raw.githubusercontent.com/LluisV/Z-Anatomy/{COMMIT}"


@dataclass(frozen=True)
class Artifact:
    source_path: str
    output_name: str
    size: int
    git_blob_sha1: str
    sha256: str

    @property
    def url(self) -> str:
        return f"{RAW_BASE}/{self.source_path}"


ARTIFACTS = (
    Artifact(
        source_path="Resources/Models/FBX/MuscularSystem100.fbx",
        output_name="MuscularSystem100.fbx",
        size=37_343_180,
        git_blob_sha1="2477e1b6caf97d7174762b5eec6cf1c80db64eed",
        sha256="4c19df534d5d84aabbce08604306aa0485b43e8a2483c72a95b569e1dfea2279",
    ),
    Artifact(
        source_path="Resources/Models/FBX/SkeletalSystem100.fbx",
        output_name="SkeletalSystem100.fbx",
        size=41_339_660,
        git_blob_sha1="7c62e45211bf3992bb7239170343e06e2b073865",
        sha256="294a649765cd060a62a4095da52b9c8ef2d97769aa447e196448aa5f7d596dea",
    ),
    Artifact(
        source_path="README.md",
        output_name="README.md",
        size=1_045,
        git_blob_sha1="4d6b763114ab33530786b016c2cc2334797b3370",
        sha256="b8a64f65dce34b6d464bae7c7775fa88c496204ccad25f0d286957014f282d02",
    ),
    Artifact(
        source_path="LICENSE",
        output_name="LICENSE",
        size=20_559,
        git_blob_sha1="383217194dc713bed27d55fbfaeb89dc78bd50d0",
        sha256="5e7dd512c01cfb822e3253f8f8df923103a64e269deb9bb5303f23b2376cad46",
    ),
    Artifact(
        source_path="Resources/Models/License.txt",
        output_name="Models-License.txt",
        size=1_514,
        git_blob_sha1="701da302cc856da40984a019a86cbad48effc53b",
        sha256="af62c06f620b9da20138e4c22a3f56565482dd058266540994ace97a4e24b693",
    ),
    Artifact(
        source_path="Resources/Models/Readme.txt",
        output_name="Models-Readme.txt",
        size=151,
        git_blob_sha1="f2a53fea185800cb75f06e2fa5431f8b98b5484e",
        sha256="43eab2cd13ad8be51d20227cad78d427e6078cd91e66a24ae7f947811524c810",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def hashes(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(usedforsecurity=False)
    sha1.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)
            sha1.update(block)
    return sha256.hexdigest(), sha1.hexdigest()


def verify(path: Path, artifact: Artifact) -> None:
    if path.stat().st_size != artifact.size:
        raise ValueError(
            f"{path} size mismatch: {path.stat().st_size} != {artifact.size}"
        )
    actual_sha256, actual_blob = hashes(path)
    if actual_sha256 != artifact.sha256:
        raise ValueError(
            f"{path} SHA-256 mismatch: {actual_sha256} != {artifact.sha256}"
        )
    if actual_blob != artifact.git_blob_sha1:
        raise ValueError(
            f"{path} Git blob mismatch: {actual_blob} != {artifact.git_blob_sha1}"
        )


def fetch(output_dir: Path, artifact: Artifact) -> Path:
    destination = output_dir / artifact.output_name
    if destination.exists():
        verify(destination, artifact)
        print(f"verified cached {destination}")
        return destination

    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(
        artifact.url,
        headers={"User-Agent": "apple-public-head-anatomy-experiment"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as file,
        ):
            shutil.copyfileobj(response, file, length=1024 * 1024)
        verify(temporary, artifact)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"downloaded {destination}")
    return destination


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for artifact in ARTIFACTS:
        fetch(args.output_dir, artifact)

    manifest = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "commit": COMMIT,
        "model_files_last_changed_commit": MODEL_FILE_COMMIT,
        "license": {
            "z_anatomy": "CC BY-SA 4.0",
            "upstream_bodyparts3d": "CC BY-SA 2.1 Japan",
            "required_attribution": [
                "BodyParts3D - The Database Center for Life Science - CC-BY-SA 2.1 Japan",
                "Z-Anatomy - The open source atlas of anatomy - CC-BY-SA 4.0",
            ],
        },
        "limitations": [
            "Resources/Models/Readme.txt says these repository models may not be up to date."
        ],
        "files": [
            {
                "source_path": artifact.source_path,
                "output_name": artifact.output_name,
                "url": artifact.url,
                "size": artifact.size,
                "git_blob_sha1": artifact.git_blob_sha1,
                "sha256": artifact.sha256,
            }
            for artifact in ARTIFACTS
        ],
    }
    manifest_path = args.output_dir / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
