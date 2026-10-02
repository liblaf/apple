"""Render the complete source soft surface at the saved MouthOpen trial."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, write_json  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "rigid_review", GROUP / "src/131-review-rigid-inverse.py"
)
assert spec is not None
assert spec.loader is not None
review = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = review
spec.loader.exec_module(review)


class Config(cherries.BaseConfig):
    review_dir: Path = GROUP / "data/review-repaired-reference-005/rigid-inverse"
    neutral_protocol: Path = GROUP / "data/forward-repaired-reference-005/protocol.json"


def main(cfg: Config) -> None:  # noqa: PLR0915
    page = cfg.review_dir.resolve()
    output = page / "full-surface"
    assert not output.exists(), output
    parent_path = page / "receipt.json"
    parent = json.loads(parent_path.read_text())
    inputs = {"parent_review": review.record(parent_path)}
    inputs.update(parent["snapshot"])
    protocol = json.loads(review.verified(inputs["protocol"]).read_text())
    neutral_path = cfg.neutral_protocol.resolve()
    neutral = json.loads(neutral_path.read_text())
    source = json.loads(review.verified(protocol["source_run"]["protocol"]).read_text())
    assert (
        neutral["reference_configuration"]["repair"]
        == source["sources"]["reference_repair"]
    )
    inputs["neutral_protocol"] = review.record(neutral_path)
    geometry_binding = neutral["reference_configuration"]["rigid_geometry"][
        "full_skull"
    ]
    inputs["geometry"] = {
        "path": geometry_binding["geometry_path"],
        "sha256": geometry_binding["geometry_sha256"],
    }
    inputs["volume"] = neutral["reference_configuration"]["constitutive_volume"]
    with np.load(review.verified(inputs["rendering"]), allow_pickle=False) as data:
        fields = {key: data[key] for key in data.files}
    with np.load(review.verified(inputs["endpoint"]), allow_pickle=False) as data:
        displacement = data["displacement_m"]
    reference = fields["full_reference_points_m"]
    assert displacement.shape == reference.shape
    assert np.isfinite(displacement).all()
    volume = pv.read(review.verified(inputs["volume"]))
    np.testing.assert_array_equal(volume.points, reference[: volume.n_points])
    with np.load(review.verified(inputs["geometry"]), allow_pickle=False) as data:
        soft_ids = data["soft_global_ids"]
        soft_faces = data["soft_faces"]
    assert soft_ids.max() < volume.n_points
    fit = review.surface(reference[soft_ids], soft_ids, soft_faces, displacement)
    components = {"soft": (soft_ids, soft_faces)}
    anatomy = {}
    for key, label in (
        ("cranium", "cranium"),
        ("mandible", "mandible"),
        ("eye", "eyes"),
    ):
        ids, faces = fields[f"{key}_global_ids"], fields[f"{key}_triangles"]
        anatomy[label] = review.surface(reference[ids], ids, faces, displacement)
        components[label] = (ids, faces)
    with np.load(review.verified(inputs["blendshapes"]), allow_pickle=False) as data:
        index = [str(name) for name in data["expression_names"]].index("MouthOpen")
        np.testing.assert_array_equal(
            data["skin_global_ids"], fields["skin_global_ids"]
        )
        target = pv.PolyData(
            data["target_points_m"][index],
            np.column_stack(
                (np.full(len(fields["skin_triangles"]), 3), fields["skin_triangles"])
            ).ravel(),
        )
    output.mkdir()
    fit.point_data["GlobalPointId"] = soft_ids
    fit.save(output / "fit-soft-surface.vtp")
    ids = np.concatenate([item[0] for item in components.values()])
    assert len(ids) == len(np.unique(ids))
    triangles, labels, offset = [], [], 0
    for component, (component_ids, faces) in enumerate(components.values()):
        triangles.append(faces + offset)
        labels.extend([component] * len(faces))
        offset += len(component_ids)
    full = review.surface(reference[ids], ids, np.concatenate(triangles), displacement)
    full.point_data["GlobalPointId"] = ids
    full.cell_data["SurfaceComponent"] = np.asarray(labels, dtype=np.int32)
    full.field_data["SurfaceComponentName"] = np.asarray(list(components))
    full.save(output / "fit-full-surface.vtp")
    images = [
        review.render(
            output, target, fit, anatomy, "full surface; adjoint failed", view
        )
        for view in ("front", "side")
    ]
    for item in inputs.values():
        review.verified(item)
    receipt = {
        "schema": "mouthopen-full-source-surface-review-v1",
        "inputs": inputs,
        "scope": "Complete source soft-tissue surface plus complete cranium, mandible and eyeballs; saved trial coordinates only; no physics solve.",
        "target_scope": "Original observed face patch; fitting objective and metrics unchanged.",
        "components": {
            key: {"vertices": len(ids), "triangles": len(faces)}
            for key, (ids, faces) in components.items()
        },
        "total_vertices": full.n_points,
        "total_triangles": full.n_cells,
        "summary": parent["summary"],
        "assets": {
            name: review.record(output / name)
            for name in [*images, "fit-soft-surface.vtp", "fit-full-surface.vtp"]
        },
    }
    write_json(output / "receipt.json", receipt)
    index_path = page / "index.html"
    html = index_path.read_text()
    for name in images:
        old = f'src="{name}"'
        assert old in html
        html = html.replace(old, f'src="full-surface/{name}"')
    description = '<p id="full-surface">The fitted result shows the complete soft-tissue surface with bones and eyeballs. The target shows its available face patch. <a href="full-surface/fit-full-surface.vtp">Download full fitted surface</a> · <a href="full-surface/receipt.json">Full-surface rendering receipt</a>.</p>'
    assert 'id="full-surface"' not in html
    html = html.replace("</h1>", "</h1>" + description, 1)
    index_path.write_text(html)
    cherries.log_output(output)
    cherries.log_metrics(
        {"render/soft_triangles": fit.n_cells, "render/total_triangles": full.n_cells}
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
