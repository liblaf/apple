# Copyright 2026 liblaf
"""Extract a named facial reference subset from pinned Z-Anatomy FBX files.

Run this module with Blender, not the project's Python interpreter::

    blender --background --factory-startup --python zanatomy_source.py -- \
      --muscular-fbx MuscularSystem100.fbx \
      --skeletal-fbx SkeletalSystem100.fbx \
      --output-dir extracted

The extraction preserves the FBX atlas frame.  It deliberately performs no
registration to the current Apple/Melon anatomy.
"""

# Blender operators use positional booleans, and explicit failure messages make
# this standalone provenance tool easier to diagnose outside the project runtime.
# ruff: noqa: C901, EM101, EM102, FBT003, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix

REPOSITORY_COMMIT = "6c7f9016bd5899ac8edafd31b9900c151df42ed6"
MODEL_FILE_COMMIT = "4ef7d59f4a2cc002f65ebd2bbb395ae4df3a5faf"
REPOSITORY_URL = "https://github.com/LluisV/Z-Anatomy"
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/LluisV/Z-Anatomy/"
    f"{REPOSITORY_COMMIT}/Resources/Models"
)


@dataclass(frozen=True)
class Source:
    key: str
    filename: str
    git_blob_sha1: str
    sha256: str

    @property
    def url(self) -> str:
        return f"{RAW_BASE_URL}/FBX/{self.filename}"


SOURCES = {
    "muscular": Source(
        key="muscular",
        filename="MuscularSystem100.fbx",
        git_blob_sha1="2477e1b6caf97d7174762b5eec6cf1c80db64eed",
        sha256="4c19df534d5d84aabbce08604306aa0485b43e8a2483c72a95b569e1dfea2279",
    ),
    "skeletal": Source(
        key="skeletal",
        filename="SkeletalSystem100.fbx",
        git_blob_sha1="7c62e45211bf3992bb7239170343e06e2b073865",
        sha256="294a649765cd060a62a4095da52b9c8ef2d97769aa447e196448aa5f7d596dea",
    ),
}

FACIAL_EXPRESSION_BASE_NAMES = (
    "Frontalis muscle",
    "Occipitalis muscle",
    "Temporoparietalis muscle",
    "Nasalis muscle",
    "Orbital part of orbicularis oculi",
    "Palpebral part of orbicularis oculi",
    "Orbicularis oris muscle",
    "Zygomaticus major muscle",
    "Corrugator supercilii",
    "Depressor anguli oris",
    "Depressor labii inferioris",
    "Levator anguli oris",
    "Procerus muscle",
    "Risorius muscle",
    "Zygomaticus minor muscle",
    "Levator labii superioris",
    "Depressor septi nasi",
    "Mentalis muscle",
    "Levator nasolabialis",
    # This is the spelling in the source FBX.
    "Bucinator",
    "Platysma",
)

FACIAL_EXPRESSION_NAMES = tuple(
    f"{name}.{side}" for name in FACIAL_EXPRESSION_BASE_NAMES for side in ("r", "l")
)

HEAD_FASCIA_NAMES = (
    "Epicranial aponeurosis.r",
    "Epicranial aponeurosis.l",
    "Superficial layer of temporal fascia.r",
    "Superficial layer of temporal fascia.l",
    "Masseteric fascia.r",
    "Masseteric fascia.l",
    "Superficial investing cervical fascia.r",
    "Superficial investing cervical fascia.l",
)

REFERENCE_BONE_NAMES = (
    "Frontal bone",
    "Parietal bone.r",
    "Parietal bone.l",
    "Temporal bone.r",
    "Temporal bone.l",
    "Occipital bone",
    "Maxilla.r",
    "Maxilla.l",
    "Zygomatic bone.r",
    "Zygomatic bone.l",
    "Mandible",
)


@dataclass(frozen=True)
class Selection:
    source: str
    role: str
    name: str


SELECTIONS = (
    *(
        Selection("muscular", "facial_expression_muscle", name)
        for name in FACIAL_EXPRESSION_NAMES
    ),
    *(
        Selection("muscular", "head_fascia_or_aponeurosis", name)
        for name in HEAD_FASCIA_NAMES
    ),
    *(Selection("skeletal", "reference_bone", name) for name in REFERENCE_BONE_NAMES),
)


def parse_args() -> argparse.Namespace:
    if "--" not in sys.argv:
        raise RuntimeError("pass extractor arguments after Blender's '--' separator")
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscular-fbx", required=True, type=Path)
    parser.add_argument("--skeletal-fbx", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path: Path, source: Source) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = sha256(path)
    if actual_sha256 != source.sha256:
        raise ValueError(
            f"{source.filename} SHA-256 mismatch: {actual_sha256} != {source.sha256}"
        )
    actual_blob = git_blob_sha1(path)
    if actual_blob != source.git_blob_sha1:
        raise ValueError(
            f"{source.filename} Git blob mismatch: {actual_blob} != {source.git_blob_sha1}"
        )


def image_texture_names(obj: bpy.types.Object) -> list[str]:
    names: set[str] = set()
    for material in obj.data.materials:
        if material is None or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                names.add(node.image.name)
    return sorted(names)


def import_source(
    path: Path, key: str
) -> tuple[dict[str, bpy.types.Object], dict[str, Any]]:
    before = set(bpy.context.scene.objects)
    result = bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    if "FINISHED" not in result:
        raise RuntimeError(f"failed to import {path}: {result}")
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    by_name: dict[str, bpy.types.Object] = {}
    for obj in meshes:
        if obj.name in by_name:
            raise ValueError(f"duplicate mesh object name in {path}: {obj.name}")
        by_name[obj.name] = obj
    return by_name, {
        "key": key,
        "imported_object_count": len(imported),
        "imported_mesh_count": len(meshes),
        "imported_vertex_count": sum(len(obj.data.vertices) for obj in meshes),
        "imported_polygon_count": sum(len(obj.data.polygons) for obj in meshes),
        "image_textures_referenced_by_mesh_materials": sorted(
            {name for obj in meshes for name in image_texture_names(obj)}
        ),
    }


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError(f"cannot create filename from object name {name!r}")
    return slug


def bake_world_transform(obj: bpy.types.Object) -> list[list[float]]:
    source_matrix = obj.matrix_world.copy()
    obj.data = obj.data.copy()
    obj.data.transform(source_matrix)
    if source_matrix.to_3x3().determinant() < 0:
        obj.data.flip_normals()
    obj.parent = None
    obj.matrix_world = Matrix.Identity(4)
    obj.data.update()
    return [list(row) for row in source_matrix]


def mesh_points(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [tuple(vertex.co) for vertex in obj.data.vertices]


def mesh_bounds(points: list[tuple[float, float, float]]) -> list[list[float]]:
    return [
        [min(point[axis] for point in points) for axis in range(3)],
        [max(point[axis] for point in points) for axis in range(3)],
    ]


def mesh_centroid(points: list[tuple[float, float, float]]) -> list[float]:
    return [sum(point[axis] for point in points) / len(points) for axis in range(3)]


def write_binary_ply(obj: bpy.types.Object, path: Path) -> int:
    mesh = obj.data
    mesh.calc_loop_triangles()
    triangles = [tuple(triangle.vertices) for triangle in mesh.loop_triangles]
    header = "\n".join(
        (
            "ply",
            "format binary_little_endian 1.0",
            "comment Z-Anatomy FBX mesh with Blender world transform baked",
            f"element vertex {len(mesh.vertices)}",
            "property float x",
            "property float y",
            "property float z",
            f"element face {len(triangles)}",
            "property list uchar uint vertex_indices",
            "end_header",
            "",
        )
    ).encode()
    with path.open("wb") as file:
        file.write(header)
        for vertex in mesh.vertices:
            file.write(struct.pack("<fff", *vertex.co))
        for triangle in triangles:
            file.write(struct.pack("<BIII", 3, *triangle))
    return len(triangles)


def output_record(
    selection: Selection,
    obj: bpy.types.Object,
    output_dir: Path,
    source_matrix: list[list[float]],
) -> dict[str, Any]:
    points = mesh_points(obj)
    if not points:
        raise ValueError(f"selected mesh contains no vertices: {selection.name}")
    output_name = f"{slugify(selection.name)}.ply"
    output_path = output_dir / "meshes" / output_name
    triangle_count = write_binary_ply(obj, output_path)
    side = selection.name.rsplit(".", maxsplit=1)[-1]
    if side not in {"r", "l"}:
        side = "midline_or_bilateral"
    return {
        "source": selection.source,
        "source_object_name": selection.name,
        "role": selection.role,
        "side_label": side,
        "output_ply": f"meshes/{output_name}",
        "output_ply_sha256": sha256(output_path),
        "vertex_count": len(obj.data.vertices),
        "polygon_count": len(obj.data.polygons),
        "triangle_count": triangle_count,
        "bounds_m": mesh_bounds(points),
        "vertex_centroid_m": mesh_centroid(points),
        "source_matrix_world": source_matrix,
        "materials": [material.name for material in obj.data.materials if material],
        "image_textures": image_texture_names(obj),
    }


def main() -> None:
    args = parse_args()
    paths = {"muscular": args.muscular_fbx, "skeletal": args.skeletal_fbx}
    for key, source in SOURCES.items():
        verify_source(paths[key], source)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = args.output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for old_ply in mesh_dir.glob("*.ply"):
        old_ply.unlink()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    imported_by_source: dict[str, dict[str, bpy.types.Object]] = {}
    import_summaries = []
    for key in ("muscular", "skeletal"):
        objects, summary = import_source(paths[key], key)
        imported_by_source[key] = objects
        import_summaries.append(summary)

    selected: list[tuple[Selection, bpy.types.Object]] = []
    for selection in SELECTIONS:
        try:
            obj = imported_by_source[selection.source][selection.name]
        except KeyError as error:
            raise KeyError(
                f"expected {selection.source} mesh object is absent: {selection.name}"
            ) from error
        selected.append((selection, obj))

    selected_objects = {obj for _, obj in selected}
    for obj in list(bpy.context.scene.objects):
        if obj not in selected_objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    records = []
    for selection, obj in selected:
        source_matrix = bake_world_transform(obj)
        obj["source"] = selection.source
        obj["source_object_name"] = selection.name
        obj["role"] = selection.role
        records.append(output_record(selection, obj, args.output_dir, source_matrix))

    bpy.ops.object.select_all(action="DESELECT")
    for _, obj in selected:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = selected[0][1]
    glb_path = args.output_dir / "facial-reference.glb"
    result = bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_yup=True,
        export_animations=False,
        export_extras=True,
        use_selection=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"failed to export {glb_path}: {result}")

    all_bounds = [record["bounds_m"] for record in records]
    manifest = {
        "schema_version": 1,
        "generator": {
            "script": str(Path(__file__).resolve()),
            "blender_version": bpy.app.version_string,
        },
        "provenance": {
            "repository": REPOSITORY_URL,
            "repository_snapshot_commit": REPOSITORY_COMMIT,
            "model_files_last_changed_commit": MODEL_FILE_COMMIT,
            "repository_readme_url": (
                f"{REPOSITORY_URL}/blob/{REPOSITORY_COMMIT}/README.md"
            ),
            "model_readme_url": f"{RAW_BASE_URL}/Readme.txt",
            "model_license_url": f"{RAW_BASE_URL}/License.txt",
            "root_license_url": (
                "https://raw.githubusercontent.com/LluisV/Z-Anatomy/"
                f"{REPOSITORY_COMMIT}/LICENSE"
            ),
            "license": "CC BY-SA 4.0, with upstream BodyParts3D attribution and CC BY-SA 2.1 Japan terms stated by Resources/Models/License.txt",
            "required_attribution": [
                "BodyParts3D - The Database Center for Life Science - CC-BY-SA 2.1 Japan",
                "Z-Anatomy - The open source atlas of anatomy - CC-BY-SA 4.0",
            ],
            "documents": [
                {
                    "path": "README.md",
                    "git_blob_sha1": "4d6b763114ab33530786b016c2cc2334797b3370",
                    "sha256": "b8a64f65dce34b6d464bae7c7775fa88c496204ccad25f0d286957014f282d02",
                },
                {
                    "path": "LICENSE",
                    "git_blob_sha1": "383217194dc713bed27d55fbfaeb89dc78bd50d0",
                    "sha256": "5e7dd512c01cfb822e3253f8f8df923103a64e269deb9bb5303f23b2376cad46",
                },
                {
                    "path": "Resources/Models/License.txt",
                    "git_blob_sha1": "701da302cc856da40984a019a86cbad48effc53b",
                    "sha256": "af62c06f620b9da20138e4c22a3f56565482dd058266540994ace97a4e24b693",
                },
                {
                    "path": "Resources/Models/Readme.txt",
                    "git_blob_sha1": "f2a53fea185800cb75f06e2fa5431f8b98b5484e",
                    "sha256": "43eab2cd13ad8be51d20227cad78d427e6078cd91e66a24ae7f947811524c810",
                },
            ],
            "source_files": [
                {
                    "key": source.key,
                    "filename": source.filename,
                    "url": source.url,
                    "git_blob_sha1": source.git_blob_sha1,
                    "sha256": source.sha256,
                }
                for source in SOURCES.values()
            ],
        },
        "coordinate_frames": {
            "ply": {
                "name": "Blender FBX import world frame",
                "units": "meters",
                "axes_observed_from_named bilateral anatomy": {
                    "x": "lateral; source .r objects have negative x",
                    "y": "posterior-positive",
                    "z": "superior-positive",
                },
                "transform": "Each FBX object's Blender matrix_world is baked into its vertices; no cross-atlas registration is applied.",
            },
            "glb": {
                "name": "standard glTF Y-up frame",
                "units": "meters",
                "from_ply_xyz_to_glb_xyz": [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                ],
            },
        },
        "selection": {
            "facial_expression_mesh_count": len(FACIAL_EXPRESSION_NAMES),
            "head_fascia_or_aponeurosis_mesh_count": len(HEAD_FASCIA_NAMES),
            "reference_bone_mesh_count": len(REFERENCE_BONE_NAMES),
            "skin_mesh_in_pinned_fbx_release": False,
        },
        "limitations": [
            "The pinned Resources/Models/Readme.txt says these FBX models may not be up to date.",
            "Object names and geometry prove separable meshes, not anatomical measurement accuracy, attachment semantics, fibers, material laws, or simulation readiness.",
            "No skin FBX exists in the pinned repository Resources/Models/FBX tree.",
            "This extraction performs no registration to the current Apple/Melon anatomy.",
            "The side interpretation follows the source .r/.l names; it was not independently validated against specimen metadata.",
        ],
        "import_summaries": import_summaries,
        "outputs": {
            "glb": "facial-reference.glb",
            "glb_sha256": sha256(glb_path),
            "ply_directory": "meshes",
            "mesh_count": len(records),
            "bounds_m_in_ply_frame": [
                [min(bounds[0][axis] for bounds in all_bounds) for axis in range(3)],
                [max(bounds[1][axis] for bounds in all_bounds) for axis in range(3)],
            ],
        },
        "objects": records,
    }
    manifest_path = args.output_dir / "zanatomy-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Exported {len(records)} named meshes to {args.output_dir}; "
        f"manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
