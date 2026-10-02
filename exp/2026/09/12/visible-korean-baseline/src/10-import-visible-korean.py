# Copyright 2026 liblaf
"""Make a traceable, surface-only Visible Korean facial anatomy baseline.

This intentionally does not repair, close, smooth, decimate, register, or
tetrahedralize source meshes.  It removes only exact duplicate triangles and
face-connected components below a declared face-count threshold.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import trimesh as tm
from anatomy_common import ProfileCometNoCommit, sha256, write_json
from PIL import Image, ImageDraw, ImageFont

from liblaf import cherries

logger = logging.getLogger(__name__)

SOURCE_SHA256 = "93f9c523c077d05df2e0b88306cd30dd34d85860d45e8eb2793f945a10e04eea"
SOURCE_MD5 = "9eeca57bbe289be6b5bcf6c142c20219"
DOI = "10.5281/zenodo.20151689"
RECORD_URL = "https://zenodo.org/records/20151689"
CREATORS = (
    {
        "name": "Kim, Chung Yoh",
        "affiliation": "Dongguk University School of Medicine",
        "orcid": "0000-0001-8074-076X",
        "displayed_role": "Distributor",
    },
    {
        "name": "Park, Jin Seo",
        "affiliation": "Ajou university School of Medicine",
        "orcid": "0000-0001-7956-4148",
        "displayed_role": "Data manager",
    },
)

MIMETIC_MUSCLES = (
    "Buccinator muscle.stl",
    "Corrugator supercilii muscle.stl",
    "Depressor aguli oris muscle.stl",
    "Depressor labii inferioris muscle.stl",
    "Frontalis muscle.stl",
    "Levator anguli oris muscle.stl",
    "Levator labii superioris alaeque nasi muscle.stl",
    "Levator labii superioris muscle.stl",
    "Mentalis muscle.stl",
    "Nasalis muscle.stl",
    "Occipitalis muscle.stl",
    "Orbicularis oculi muscle.stl",
    "Orbicularis oris muscle.stl",
    "Platysma muscle.stl",
    "Procerus muscle.stl",
    "Risorius muscle.stl",
    "Zygomaticus major muscle.stl",
    "Zygomaticus minor muscle.stl",
)
TARGETS = {
    "skin": ("Skin.stl",),
    "bones": ("cranium.stl", "Frontal bone.stl", "Mandible.stl", "Zygomatic bone.stl"),
    "mimetic_muscles": MIMETIC_MUSCLES,
    "palpebral_ligaments": (
        "Medial palpebral ligament.stl",
        "Lateral palpebral ligament.stl",
    ),
}
COLORS = {
    "skin": "#9aa4ab",
    "bones": "#d9d0bd",
    "mimetic_muscles": "#c85d50",
    "palpebral_ligaments": "#3d719c",
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    source: Path = cherries.input("source/Surface-models.zip")
    output: Path = cherries.output("10-baseline", mkdir=True)
    minimum_component_faces: int = 100


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def source_md5(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def selected_members(names: Iterable[str]) -> dict[str, str]:
    stl_names = [name for name in names if name.lower().endswith(".stl")]
    found: dict[str, str] = {}
    missing: list[str] = []
    for group, basenames in TARGETS.items():
        for basename in basenames:
            matches = [
                name for name in stl_names if name.rsplit("/", 1)[-1] == basename
            ]
            if len(matches) != 1:
                missing.append(f"{group}: {basename} ({len(matches)} matches)")
            else:
                found[matches[0]] = group
    if missing:
        raise RuntimeError("Source member selection failed: " + "; ".join(missing))
    return found


def exact_deduplicate(mesh: tm.Trimesh) -> tuple[tm.Trimesh, int]:
    """Return a mesh with coordinate-identical duplicate triangles dropped.

    Winding differences are deliberately ignored only for duplicate detection;
    the retained face is the first source face and keeps its original winding.
    """
    vertices, inverse = np.unique(
        np.asarray(mesh.vertices), axis=0, return_inverse=True
    )
    faces = inverse[np.asarray(mesh.faces)]
    keys = np.sort(faces, axis=1)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep.sort()
    clean = tm.Trimesh(
        vertices=vertices, faces=faces[keep], process=False, validate=False
    )
    return clean, int(len(faces) - len(keep))


def edge_counts(mesh: tm.Trimesh) -> dict[str, int | bool]:
    incidence = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    return {
        "boundary_edges": int(np.count_nonzero(incidence == 1)),
        "nonmanifold_edges": int(np.count_nonzero(incidence > 2)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def polydata(mesh: tm.Trimesh) -> pv.PolyData:
    faces = np.column_stack(
        (np.full(len(mesh.faces), 3), np.asarray(mesh.faces))
    ).ravel()
    return pv.PolyData(np.asarray(mesh.vertices), faces)


def component_meshes(
    mesh: tm.Trimesh, minimum_faces: int
) -> tuple[list[tm.Trimesh], list[tm.Trimesh]]:
    components = sorted(
        mesh.split(only_watertight=False),
        key=lambda item: len(item.faces),
        reverse=True,
    )
    kept = [
        component for component in components if len(component.faces) >= minimum_faces
    ]
    removed = [
        component for component in components if len(component.faces) < minimum_faces
    ]
    return kept, removed


def add_caption(path: Path) -> None:
    source = Image.open(path).convert("RGB")
    band = 142
    canvas = Image.new("RGB", (source.width, source.height + band), "#fbfbfa")
    canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font_root = Path("/usr/share/fonts/truetype/dejavu")
    regular = ImageFont.truetype(font_root / "DejaVuSans.ttf", 21)
    bold = ImageFont.truetype(font_root / "DejaVuSans-Bold.ttf", 27)
    draw.text(
        (36, source.height + 15),
        "Visible Korean same-donor facial anatomy baseline",
        fill="#242424",
        font=bold,
    )
    draw.text(
        (36, source.height + 56),
        "Native STL coordinates, selected and minimally cleaned; cameras are deliberately named View A/B because the archive does not document axes.",
        fill="#424242",
        font=regular,
    )
    draw.text(
        (36, source.height + 91),
        f"Source {DOI} | coordinates interpreted as mm from source-image documentation | no fibers, SMAS, attachments, registration, or volume mesh.",
        fill="#606060",
        font=regular,
    )
    canvas.save(path)


def render_preview(output: Path, blocks: list[tuple[str, str, pv.PolyData]]) -> Path:
    path = output / "visible-korean-baseline-preview.png"
    skin = next(mesh for _, group, mesh in blocks if group == "skin")
    center = skin.center
    span = max(skin.length, 1.0)
    plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 900), off_screen=True)
    plotter.set_background("#f7f7f5")
    for index, location in enumerate(((0, 0), (0, 1))):
        plotter.subplot(*location)
        for _, group, mesh in blocks:
            opacity = {
                "skin": 0.10,
                "bones": 0.12,
                "mimetic_muscles": 0.95,
                "palpebral_ligaments": 1.0,
            }[group]
            plotter.add_mesh(
                mesh, color=COLORS[group], opacity=opacity, smooth_shading=False
            )
        plotter.add_text(
            f"Native View {'A' if index == 0 else 'B'}",
            position="upper_left",
            font_size=13,
            color="#303030",
        )
        if index == 0:
            position = (center[0], center[1] - 1.55 * span, center[2] + 0.02 * span)
        else:
            position = (
                center[0] + 1.10 * span,
                center[1] - 1.40 * span,
                center[2] + 0.10 * span,
            )
        plotter.camera_position = [position, center, (0.0, 0.0, 1.0)]
        plotter.camera.zoom(1.23 if index == 0 else 1.28)
    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(path, return_img=False)
    plotter.close()
    add_caption(path)
    return path


def main(cfg: Config) -> None:
    assert cfg.minimum_component_faces > 0
    assert sha256(cfg.source) == SOURCE_SHA256
    assert source_md5(cfg.source) == SOURCE_MD5
    output = cfg.output
    surfaces = output / "surfaces"
    components_dir = output / "components"
    surfaces.mkdir(parents=True, exist_ok=True)
    components_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(cfg.source) as archive:
        assert archive.testzip() is None
        selected = selected_members(archive.namelist())
        rows: list[dict[str, object]] = []
        render_blocks: list[tuple[str, str, pv.PolyData]] = []
        package = pv.MultiBlock()
        for member, group in sorted(selected.items()):
            raw_bytes = archive.read(member)
            raw = tm.load_mesh(io.BytesIO(raw_bytes), file_type="stl", process=False)
            assert isinstance(raw, tm.Trimesh)
            clean, duplicate_faces_removed = exact_deduplicate(raw)
            kept, removed = component_meshes(clean, cfg.minimum_component_faces)
            if not kept:
                message = f"No component reached threshold for {member}"
                raise RuntimeError(message)
            object_block = pv.MultiBlock()
            component_rows = []
            for index, component in enumerate(kept, start=1):
                component_poly = polydata(component)
                component_path = (
                    components_dir / f"{slug(member[:-4])}-component-{index:02}.vtp"
                )
                component_poly.save(component_path)
                object_block[f"component-{index:02}"] = component_poly
                component_rows.append(
                    {
                        "index": index,
                        "faces": len(component.faces),
                        "vertices": len(component.vertices),
                        "path": str(component_path.relative_to(output)),
                        "sha256": sha256(component_path),
                        **edge_counts(component),
                    }
                )
            merged = tm.util.concatenate(kept)
            surface = polydata(merged)
            surface_path = surfaces / f"{slug(member[:-4])}.vtp"
            surface.save(surface_path)
            package[slug(member[:-4])] = object_block
            render_blocks.append((member[:-4], group, surface))
            rows.append(
                {
                    "source_member": member,
                    "group": group,
                    "raw_bytes": len(raw_bytes),
                    "raw_triangles": len(raw.faces),
                    "raw_vertices": len(raw.vertices),
                    "exact_duplicate_faces_removed": duplicate_faces_removed,
                    "minimum_component_faces": cfg.minimum_component_faces,
                    "removed_small_component_count": len(removed),
                    "removed_small_component_faces": int(
                        sum(len(item.faces) for item in removed)
                    ),
                    "kept_component_count": len(kept),
                    "kept_triangles": len(merged.faces),
                    "kept_vertices": len(merged.vertices),
                    "surface_path": str(surface_path.relative_to(output)),
                    "surface_sha256": sha256(surface_path),
                    "components": component_rows,
                    "whole_kept_surface": edge_counts(merged),
                }
            )
            logger.info(
                "Imported %s: %d -> %d faces", member, len(raw.faces), len(merged.faces)
            )
    package_path = output / "visible-korean-selected-anatomy.vtm"
    package.save(package_path)
    preview_path = render_preview(output, render_blocks)
    manifest = {
        "schema_version": 1,
        "dataset": "Visible Korean male head dataset: surface models",
        "dataset_doi": DOI,
        "record_url": RECORD_URL,
        "license": "CC-BY-NC-4.0",
        "citation_and_credit": {
            "creators": CREATORS,
            "role_receipt": {
                "source_url": RECORD_URL,
                "checked_date": "2026-09-12",
                "source_location": "Zenodo record Authors/Creators display, lines 8-9",
                "displayed_roles": {
                    "Kim, Chung Yoh": "Distributor",
                    "Park, Jin Seo": "Data manager",
                },
            },
            "citation_instruction": "Cite this Zenodo record and its associated Data Descriptor.",
        },
        "source_archive": {
            "path": str(cfg.source),
            "sha256": SOURCE_SHA256,
            "md5": SOURCE_MD5,
            "bytes": cfg.source.stat().st_size,
            "zip_integrity": "verified with zipfile.ZipFile.testzip()",
        },
        "coordinate_system": {
            "source_stl_units": "not encoded by STL",
            "working_interpretation": "millimetres, inferred from source-image documentation and anatomical extent",
            "axis_names_and_origin": "not documented by the archive; no anatomical orientation is claimed",
            "registration": "none",
        },
        "selection": {
            "groups": {key: list(value) for key, value in TARGETS.items()},
            "selection_rule": "each listed basename must occur exactly once in the archive",
            "source_naming_note": "The source filename is Depressor aguli oris muscle.stl; its spelling is retained verbatim.",
        },
        "cleanup": {
            "performed": [
                "merge coordinate-identical vertices for duplicate-face detection",
                "drop duplicate triangles with the same unordered vertex triple, retaining first source winding",
                "drop face-connected components with fewer than minimum_component_faces faces",
            ],
            "not_performed": [
                "hole filling",
                "nonmanifold repair",
                "normal reorientation",
                "smoothing",
                "decimation",
                "registration",
                "interface matching",
                "tetrahedralization",
            ],
            "important_limit": "Outputs remain surface references, not FEM-ready domain meshes.",
        },
        "missing_or_unrepresented": [
            "SMAS",
            "galea or other aponeuroses",
            "retaining ligaments or dermal attachments",
            "measured muscle fiber directions",
            "muscle origins and insertions as explicit data",
            "material parameters",
            "tetrahedral volume mesh",
            "solver state",
        ],
        "outputs": {
            "surface_package": {
                "path": package_path.name,
                "sha256": sha256(package_path),
            },
            "preview": {"path": preview_path.name, "sha256": sha256(preview_path)},
            "objects": rows,
        },
    }
    manifest_path = output / "visible-korean-manifest.json"
    write_json(manifest_path, manifest)
    groups = {
        group: sum(row["kept_triangles"] for row in rows if row["group"] == group)
        for group in TARGETS
    }
    cherries.log_metrics(
        {
            "visible_korean/objects": len(rows),
            "visible_korean/raw_triangles": sum(
                int(row["raw_triangles"]) for row in rows
            ),
            "visible_korean/kept_triangles": sum(
                int(row["kept_triangles"]) for row in rows
            ),
            "visible_korean/duplicate_faces_removed": sum(
                int(row["exact_duplicate_faces_removed"]) for row in rows
            ),
            "visible_korean/small_components_removed": sum(
                int(row["removed_small_component_count"]) for row in rows
            ),
            **{
                f"visible_korean/{group}_triangles": total
                for group, total in groups.items()
            },
        }
    )
    logger.info("Wrote manifest %s", manifest_path)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
