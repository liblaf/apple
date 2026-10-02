"""Add exact rigid anatomy to saved neutral views without changing the solve."""

from __future__ import annotations

import importlib.util
import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/forward-repaired-reference-004"
    review_dir: Path = GROUP / "data/review-repaired-reference-004"


def load_script(filename: str):
    spec = importlib.util.spec_from_file_location(
        Path(filename).stem, GROUP / "src" / filename
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified(item: dict) -> Path:
    path = Path(item["path"])
    assert sha256(path) == item["sha256"], path
    return path


def page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#202124;background:#f7f7f5}}img{{width:100%;height:auto}}figure{{margin:1.5rem 0}}.status{{padding:1rem;background:#fff1c7}}a{{color:#236595}}</style>
<h1>{title}</h1>{body}</html>"""


def main(cfg: Config) -> None:  # noqa: PLR0915
    run, review = cfg.run_dir.resolve(), cfg.review_dir.resolve()
    output = review / "anatomy"
    assert not output.exists(), output
    parent = json.loads((review / "receipt.json").read_text())
    protocol_path = verified(parent["run"]["protocol"])
    assert protocol_path == run / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["fixture"]["jaw_rotation_rad"] == 0
    neutral_path = verified(protocol["fixture"]["neutral_manifest"])
    neutral = json.loads(neutral_path.read_text())
    geometry_path = verified(neutral["sources"]["geometry"])
    eyes_manifest_path = verified(protocol["fixture"]["eyes_manifest"])
    eyes_manifest = json.loads(eyes_manifest_path.read_text())
    eyes_path = verified(eyes_manifest["artifacts"]["eyes.vtp"])
    renderer = load_script("20-review-neutral.py")
    motion = load_script("40-review-motion.py")
    cranium, mandible = renderer._HELPERS._geometry(geometry_path)  # noqa: SLF001
    rigid = {"cranium": cranium, "mandible": mandible, "eyes": pv.read(eyes_path)}
    initial_rigid = {
        name: np.asarray(mesh.points).copy() for name, mesh in rigid.items()
    }
    reference_path = verified(parent["inputs"]["constitutive_skin"])
    reference = pv.read(reference_path)
    endpoint_path = verified(parent["run"]["endpoint"])
    with np.load(endpoint_path, allow_pickle=False) as archive:
        displacement = archive["displacement_m"]
    ids = np.asarray(reference.point_data["GlobalPointId"], dtype=np.int64)
    u = displacement[ids]
    solved = pv.read(review / "neutral-skin.vtp")
    np.testing.assert_allclose(solved.points, reference.points + u, rtol=0, atol=5e-16)
    output.mkdir()
    (output / "transparent").mkdir()
    (output / "motion").mkdir()
    images = []
    for directory, opacity in ((output, 1.0), (output / "transparent", 0.18)):
        names = renderer._comparison(  # noqa: SLF001
            directory,
            reference,
            solved,
            endpoint_status=parent["state_label"],
            material_label="Active-strain",
            rigid_meshes=rigid,
            skin_opacity=opacity,
        )
        images.extend(str((directory / name).relative_to(output)) for name in names)
    motion_images = [
        motion.render(
            output / "motion",
            reference,
            u,
            view,
            "Repaired constitutive reference",
            rigid_meshes=rigid,
        )
        for view in ("front", "side")
    ]
    images.extend(f"motion/{name}" for name in motion_images)
    for name, mesh in rigid.items():
        assert np.array_equal(mesh.points, initial_rigid[name])
        mesh.save(output / f"{name}.vtp")
    rigid_description = "Exact cranium and mandible in ivory; registered eyeballs in blue. Bones and eyeballs stay at their neutral pose."
    status = f'<p class="status">{escape(parent["state_label"])}. Force residual {parent["terminal_force_n"]:.6g} N; required {parent["force_threshold_n"]:.6g} N.</p>'
    figures = "".join(
        f'<figure><a href="{name}"><img src="{name}" alt="Reference and endpoint with bones and eyeballs"></a><figcaption>{"Transparent skin (18% opacity)" if name.startswith("transparent/") else "Opaque skin"} · {"front" if "front" in name else "side"} · identical true scale</figcaption></figure>'
        for name in images
        if not name.startswith("motion/")
    )
    (output / "index.html").write_text(
        page(
            "Neutral face, bones and eyeballs",
            f'<p><a href="../">Back to neutral review</a> · <a href="motion/">Motion with anatomy</a></p><p>{rigid_description} The transparent views reveal the internal anatomy.</p>{status}{figures}<p>Mesh downloads: <a href="cranium.vtp">cranium</a> · <a href="mandible.vtp">mandible</a> · <a href="eyes.vtp">eyeballs</a>. <a href="receipt.json">Provenance</a>.</p>',
        )
    )
    motion_figures = "".join(
        f'<figure><img src="{name}" alt="Reference, actual motion, and magnified motion with fixed bones and eyeballs"></figure>'
        for name in motion_images
    )
    (output / "motion/index.html").write_text(
        page(
            "Neutral motion with bones and eyeballs",
            f'<p><a href="../../">Back to neutral review</a> · <a href="../">Transparent anatomy</a></p><p>{rigid_description} Only skin displacement is magnified in the 10x panel; it is a display aid.</p>{status}{motion_figures}<p><a href="../../motion/motion-summary.json">Original motion measurements</a></p>',
        )
    )
    index = review / "index.html"
    old_index = record(index)
    html = index.read_text()
    assert 'id="anatomy-detail"' not in html
    for view in ("front", "side"):
        name = f"reference-vs-solved-{view}.png"
        html = html.replace(f'"{name}"', f'"anatomy/{name}"')
    html = html.replace('href="motion/"', 'href="anatomy/motion/"')
    html = html.replace(
        "</h1>",
        '</h1><p id="anatomy-detail">Bones and eyeballs are included. <a href="anatomy/">View anatomy through transparent skin</a>.</p>',
        1,
    )
    index.write_text(html)
    receipt = {
        "schema": "saved-neutral-anatomy-review-v1",
        "inputs": {
            "protocol": record(protocol_path),
            "geometry": record(geometry_path),
            "eyes_manifest": record(eyes_manifest_path),
            "eyes": record(eyes_path),
            "skin_reference": record(reference_path),
            "endpoint": record(endpoint_path),
        },
        "rigid_geometry": {
            name: {
                "vertices": mesh.n_points,
                "triangles": mesh.n_cells,
                "coordinates_preserved_exactly": True,
            }
            for name, mesh in rigid.items()
        },
        "rigid_pose": "unchanged registered neutral pose; jaw rotation zero; eyes fixed",
        "skin_opacities": [1.0, 0.18],
        "display_displacement_scales": [0, 1, 10],
        "solver_rerun": False,
        "solver_converged": parent["result"]["success"],
        "root_index_before": old_index,
        "root_index_after": record(index),
        "render_sources": {
            name: record(GROUP / "src" / name)
            for name in (
                "20-review-neutral.py",
                "40-review-motion.py",
                "80-add-anatomy.py",
            )
        },
        "assets": {
            str(path.relative_to(output)): record(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    write_json(output / "receipt.json", receipt)
    cherries.log_output(output)
    cherries.log_metrics(
        {f"anatomy/{name}_triangles": mesh.n_cells for name, mesh in rigid.items()}
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
