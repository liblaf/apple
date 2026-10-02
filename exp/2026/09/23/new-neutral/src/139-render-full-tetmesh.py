"""Render the complete corrected neutral tetmesh and a whole-cell cutaway."""

# ruff: noqa: PLR0915

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402

RIGID_COLORS = {"cranium": "#e5d8bb", "mandible": "#cdb787", "eyes": "#91c9da"}


class Config(cherries.BaseConfig):
    review_dir: Path = GROUP / "data/review-isfixed-001"


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified(item: dict[str, str]) -> Path:
    path = Path(item["path"])
    assert sha256(path) == item["sha256"], path
    return path


def render(
    path: Path,
    surfaces: tuple[pv.PolyData, pv.PolyData],
    anatomy: dict[str, pv.PolyData],
    *,
    view: str,
    edges: bool = False,
    cutaway: bool = False,
) -> None:
    bounds = np.asarray([mesh.bounds for mesh in (*surfaces, *anatomy.values())])
    low = bounds[:, ::2].min(axis=0)
    high = bounds[:, 1::2].max(axis=0)
    center, span = (low + high) / 2, float(max(high - low))
    directions = {
        "front": (0, 0, 2.7),
        "side": (2.7, 0, 0),
        "oblique": (2.4, 0.25, 2.0),
    }
    camera = [
        (center + span * np.asarray(directions[view])).tolist(),
        center.tolist(),
        [0, 1, 0],
    ]
    plot = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2000, 1000))
    plot.set_background("#f7f7f5")
    for column, (title, surface, color) in enumerate(
        zip(
            ("Constitutive reference", "Converged neutral"),
            surfaces,
            ("#737b86", "#c75f42"),
            strict=True,
        )
    ):
        plot.subplot(0, column)
        for name, rigid in anatomy.items():
            plot.add_mesh(rigid, color=RIGID_COLORS[name], smooth_shading=True)
        plot.add_mesh(
            surface,
            color=color,
            scalars=None,
            smooth_shading=not edges,
            show_edges=edges,
            edge_color="#3c3734",
            line_width=0.6,
            ambient=0.25,
            diffuse=0.7,
            specular=0,
        )
        label = "Whole-tetrahedron cutaway" if cutaway else "Complete tetmesh boundary"
        plot.add_text(
            f"{title}\n{label} · {view} · true scale",
            position="upper_left",
            font_size=15,
            color="#202124",
        )
        plot.camera_position = camera
        plot.camera.parallel_projection = True
        plot.camera.parallel_scale = 0.58 * span
        plot.reset_camera_clipping_range()
    plot.show(screenshot=path, auto_close=True)


def main(cfg: Config) -> None:
    review = cfg.review_dir.resolve()
    output = review / "full-tetmesh"
    assert not output.exists(), output
    parent_path = review / "receipt.json"
    parent = json.loads(parent_path.read_text())
    assert parent["result"]["valid_forward"]
    reference_path = verified(parent["inputs"]["constitutive_volume"])
    endpoint_path = verified(parent["run"]["endpoint"])
    reference = pv.read(reference_path)
    assert isinstance(reference, pv.UnstructuredGrid)
    assert np.all(reference.celltypes == pv.CellType.TETRA)
    with np.load(endpoint_path, allow_pickle=False) as archive:
        displacement = archive["displacement_m"]
    assert displacement.shape == reference.points.shape
    solved = reference.copy(deep=True)
    solved.points = reference.points + displacement
    published_path = review / "neutral-volume.vtu"
    published = pv.read(published_path)
    assert np.array_equal(solved.cells, published.cells)
    assert np.array_equal(solved.points, published.points)
    assert np.array_equal(
        reference.point_data["FixedMask"],
        np.repeat(reference.point_data["IsFixed"][:, None], 3, axis=1),
    )
    # Every tetrahedron enters boundary extraction; no anatomical masks or decimation.
    surfaces = tuple(
        mesh.extract_surface(algorithm=None) for mesh in (reference, solved)
    )
    assert np.array_equal(surfaces[0].faces, surfaces[1].faces)
    assert surfaces[0].n_cells == surfaces[1].n_cells
    anatomy_receipt_path = review / "anatomy/receipt.json"
    anatomy_receipt = json.loads(anatomy_receipt_path.read_text())
    assert anatomy_receipt["inputs"]["endpoint"] == parent["run"]["endpoint"]
    anatomy_paths = {
        name: verified(anatomy_receipt["assets"][f"{name}.vtp"])
        for name in RIGID_COLORS
    }
    anatomy = {name: pv.read(path) for name, path in anatomy_paths.items()}
    output.mkdir()
    source_copy = output / Path(__file__).name
    shutil.copyfile(__file__, source_copy)
    images = []
    for view in ("front", "side"):
        name = f"reference-vs-solved-{view}.png"
        render(output / name, surfaces, anatomy, view=view)
        images.append((name, f"Complete tetmesh boundary · {view} · opaque surface"))
    render(output / "mesh-edges-front.png", surfaces, anatomy, view="front", edges=True)
    images.append(
        ("mesh-edges-front.png", "Complete tetmesh boundary with triangle edges")
    )
    # Select identical original whole cells in both states; never clip individual tets.
    cut_x = float((reference.bounds[0] + reference.bounds[1]) / 2)
    cells = np.asarray(reference.cells).reshape(-1, 5)[:, 1:]
    cell_ids = np.flatnonzero(
        np.asarray(reference.points)[cells, 0].mean(axis=1) <= cut_x
    )
    assert 0 < len(cell_ids) < reference.n_cells
    cut_surfaces = tuple(
        mesh.extract_cells(cell_ids).extract_surface(algorithm=None)
        for mesh in (reference, solved)
    )
    cut_anatomy = {
        name: mesh.clip(normal=(1, 0, 0), origin=(cut_x, 0, 0), invert=True)
        for name, mesh in anatomy.items()
    }
    render(
        output / "tetmesh-cutaway.png",
        cut_surfaces,
        cut_anatomy,
        view="oblique",
        edges=True,
        cutaway=True,
    )
    images.append(
        (
            "tetmesh-cutaway.png",
            "Whole-cell half-mesh cutaway; matching reference cell IDs in both panels. Bones and eyes clipped at the midplane.",
        )
    )
    for mesh, name in zip(
        surfaces,
        ("reference-full-boundary.vtp", "neutral-full-boundary.vtp"),
        strict=True,
    ):
        mesh.save(output / name)
    figures = "".join(
        f'<figure><a href="{name}"><img src="{name}" alt="{caption}"></a><figcaption>{caption}</figcaption></figure>'
        for name, caption in images
    )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full neutral tetmesh</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#202124;background:#f7f7f5}}img{{width:100%;height:auto}}figure{{margin:2rem 0}}a{{color:#236595}}.status{{background:#d9f2e6;padding:1rem}}</style>
<h1>Full neutral tetmesh</h1><p><a href="../">Back to equilibrium report</a></p>
<p class="status">Corrected IsFixed equilibrium · converged · zero inverted tetrahedra · contact checks passed</p>
<p>{reference.n_points:,} FEM vertices · {reference.n_cells:,} tetrahedra · {surfaces[0].n_cells:,} complete boundary triangles. Bones and eyeballs are included at their saved neutral pose. Displacements are shown at true scale.</p>
<p>The opaque views show the complete tetrahedral mesh boundary. The cutaway exposes interior elements by displaying {len(cell_ids):,} whole tetrahedra from one half of the mesh.</p>
{figures}<p>Download full tetrahedral volumes: <a href="../reference-volume.vtu">reference</a> · <a href="../neutral-volume.vtu">converged neutral</a>. Full boundary meshes: <a href="reference-full-boundary.vtp">reference</a> · <a href="neutral-full-boundary.vtp">neutral</a>. <a href="receipt.json">Render provenance</a>.</p></html>"""
    (output / "index.html").write_text(html)
    index = review / "index.html"
    previous_index = record(index)
    root_html = index.read_text()
    assert 'id="full-tetmesh-detail"' not in root_html
    for view in ("front", "side"):
        old = f'"anatomy/reference-vs-solved-{view}.png"'
        assert old in root_html
        root_html = root_html.replace(
            old, f'"full-tetmesh/reference-vs-solved-{view}.png"'
        )
    root_html = root_html.replace(
        "</h1>",
        '</h1><p id="full-tetmesh-detail">The comparisons below show the complete tetmesh boundary. <a href="full-tetmesh/">View full tetmesh, mesh edges, and interior cutaway</a>.</p>',
        1,
    )
    index.write_text(root_html)
    receipt = {
        "schema": "corrected-neutral-full-tetmesh-render-v1",
        "inputs": {
            "review": record(parent_path),
            "reference_volume": record(reference_path),
            "endpoint": record(endpoint_path),
            "published_volume": record(published_path),
            "anatomy_receipt": record(anatomy_receipt_path),
            **{name: record(path) for name, path in anatomy_paths.items()},
        },
        "vertices": reference.n_points,
        "tetrahedra": reference.n_cells,
        "boundary_vertices": surfaces[0].n_points,
        "boundary_triangles": surfaces[0].n_cells,
        "cutaway": {
            "reference_centroid_x_max_m": cut_x,
            "whole_tetrahedra": len(cell_ids),
            "identical_cell_ids_in_both_states": True,
        },
        "full_boundary_anatomical_filter": None,
        "decimation": False,
        "displacement_scale": 1,
        "points_equal_reference_plus_endpoint_exactly": True,
        "published_volume_matches_exactly": True,
        "solver_rerun": False,
        "root_index_before": previous_index,
        "root_index_after": record(index),
        "assets": {
            path.name: record(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    write_json(output / "receipt.json", receipt)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "render/vertices": reference.n_points,
            "render/tetrahedra": reference.n_cells,
            "render/full_boundary_triangles": surfaces[0].n_cells,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
