"""Package the five saved volume deformations beside their matching skins."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    comparison: Path = cherries.input("98-aligned-errors/summary.json")
    surfaces: Path = cherries.input("five-deformed-meshes/manifest.json")
    initialization: Path = cherries.input("42-fixed-directions-400/initialization.npz")
    output_dir: Path = cherries.output("five-deformed-meshes", mkdir=True)


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": sha, "bytes": path.stat().st_size}


def verify(item: dict) -> Path:
    path = Path(item["path"])
    assert record(path) == item, path
    return path


def main(cfg: Config) -> None:
    comparison = json.loads(cfg.comparison.read_text())
    surfaces = json.loads(cfg.surfaces.read_text())
    assert len(surfaces) == len(comparison["states"]) == 5
    fixture_record = next(
        item for item in comparison["inputs"] if Path(item["path"]).name == "volume.vtu"
    )
    template = pv.read(verify(fixture_record))
    assert np.all(template.celltypes == pv.CellType.TETRA)
    rest = np.asarray(template.points).copy()
    cells = np.asarray(template.cells).reshape(-1, 5)
    assert np.all(cells[:, 0] == 4)
    tets = cells[:, 1:]
    rest_edges = (rest[tets[:, 1:]] - rest[tets[:, :1]]).swapaxes(1, 2)
    rest_det = np.linalg.det(rest_edges)
    assert np.all(rest_det > 0)
    rest_inverse = np.linalg.inv(rest_edges)
    active = np.flatnonzero(template.cell_data["ActivationMask"])
    assert np.array_equal(template.point_data["GlobalPointId"], np.arange(len(rest)))
    with np.load(cfg.initialization, allow_pickle=False) as init:
        assert np.array_equal(init["rest_points"], rest)
        axes, axes_sha = init["axes"], str(init["axes_sha256"])
        assert np.array_equal(init["active_ids"], active)
    outputs = []
    for index, (state, surface) in enumerate(
        zip(comparison["states"], surfaces, strict=True)
    ):
        cherries.set_step(index)
        assert state["title"] == surface["state"]
        assert state["source"] == surface["source_state"]
        source = verify(state["source"])
        output = cfg.output_dir / Path(surface["file"]).with_suffix(".vtu")
        assert not output.exists(), output
        with np.load(source, allow_pickle=False) as saved:
            assert bool(saved["solver_valid"])
            assert np.array_equal(saved["active_ids"], active)
            u = saved["u"]
            if "rest_points" in saved:
                assert np.array_equal(saved["rest_points"], rest)
            if state["id"] == "fixed":
                assert str(saved["axes_sha256"]) == axes_sha
                assert np.all(saved["s"] >= 0)
                b = (
                    np.eye(3)
                    + saved["s"][:, None, None] * axes[:, :, None] * axes[:, None, :]
                )
            elif state["id"] in {"released", "scratch"}:
                v = saved["v"] if state["id"] == "released" else saved["q"]
                b = np.eye(3) + v[:, :, None] * v[:, None, :]
                if "B" in saved:
                    assert np.allclose(b, saved["B"], rtol=2e-14, atol=3e-13)
            else:
                b = saved["B"]
        assert u.shape == rest.shape and np.isfinite(u).all()
        assert b.shape == (len(active), 3, 3) and np.isfinite(b).all()
        deformed = rest + u
        det = np.linalg.det(
            (deformed[tets[:, 1:]] - deformed[tets[:, :1]]).swapaxes(1, 2)
            @ rest_inverse
        )
        inverted = int(np.sum(det <= 0))
        assert inverted == state["metrics"]["inverted_all_cells"]
        assert np.isclose(
            det.min(), state["metrics"]["detF_min"], rtol=1e-12, atol=1e-12
        )
        skin_path = cfg.output_dir / surface["file"]
        assert record(skin_path)["sha256"] == surface["sha256"]
        skin = pv.read(skin_path)
        skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
        assert np.array_equal(deformed[skin_ids], skin.points)
        mesh = template.copy(deep=True)
        mesh.points = deformed
        # Distinguish fixture attributes from the saved state's actual activation.
        renamed = {
            "Volume": "ReferenceVolume",
            "Activation": "FixtureActivation",
            "ActivationInv": "FixtureActivationInv",
            "ActivationFiber": "FixtureActivationFiber",
            "ActivationFiberConfidence": "FixtureActivationFiberConfidence",
        }
        for before, after in renamed.items():
            mesh.cell_data[after] = mesh.cell_data.pop(before)
        mesh.point_data["RestPosition"] = rest
        mesh.point_data["Displacement"] = u
        mesh.cell_data["DetF"] = det
        mesh.cell_data["IsInverted"] = (det <= 0).astype(np.uint8)
        full_b = np.broadcast_to(np.eye(3), (mesh.n_cells, 3, 3)).copy()
        full_b[active] = b
        mesh.cell_data["ActivationInverseMatrix"] = full_b.reshape(-1, 9)
        # PyVista adds private serialization metadata while saving. Compare the
        # public attributes that existed before that mutation with the reload.
        expected_data = {
            attr: {key: np.asarray(value) for key, value in getattr(mesh, attr).items()}
            for attr in ("point_data", "cell_data", "field_data")
        }
        LOG.info(
            "Writing %s (%s points, %s tetrahedra)",
            output.name,
            mesh.n_points,
            mesh.n_cells,
        )
        mesh.save(output, binary=True)
        reopened = pv.read(output)
        assert np.array_equal(reopened.points, deformed)
        assert np.array_equal(reopened.cells, template.cells)
        assert np.array_equal(reopened.celltypes, template.celltypes)
        assert np.array_equal(reopened.points[skin_ids], skin.points)
        for attr in ("point_data", "cell_data", "field_data"):
            expected, actual = expected_data[attr], getattr(reopened, attr)
            assert set(expected) == set(actual)
            for key in expected:
                expected_array, actual_array = (
                    np.asarray(expected[key]),
                    np.asarray(actual[key]),
                )
                if np.issubdtype(expected_array.dtype, np.inexact):
                    assert np.array_equal(
                        expected_array, actual_array, equal_nan=True
                    ), key
                else:
                    assert np.array_equal(expected_array, actual_array), key
        metrics = {
            "points": mesh.n_points,
            "tetrahedra": mesh.n_cells,
            "inverted": inverted,
        }
        cherries.log_metrics(
            {f"{state['id']}/{key}": value for key, value in metrics.items()}
        )
        outputs.append(
            {
                "state": state["title"],
                "file": output.name,
                "output": record(output),
                "surface_file": surface["file"],
                "source_state": state["source"],
                "metrics": metrics,
                "skin_points_match_exactly": True,
                "connectivity_matches_fixture_exactly": True,
            }
        )
        del mesh, reopened, full_b
    source_copy = cfg.output_dir / "100-export-tetmeshes.py"
    shutil.copyfile(Path(__file__), source_copy)
    summary = {
        "status": "completed",
        "outputs": outputs,
        "fixture": fixture_record,
        "inputs": [
            record(cfg.comparison),
            record(cfg.surfaces),
            record(cfg.initialization),
        ],
        "source": record(source_copy),
        "renamed_fixture_arrays": renamed,
        "units": {
            "points": "m",
            "RestPosition": "m",
            "Displacement": "m",
            "DetF": "dimensionless",
        },
        "activation": "ActivationInverseMatrix is actual saved B=A^-1, row-major 3x3, identity on inactive cells",
        "geometry": "X+u from saved checkpoint; original tetrahedral connectivity, no remeshing or new solve",
    }
    manifest = cfg.output_dir / "tetmesh-manifest.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n")
    cherries.log_output(manifest)
    LOG.info("Completed all five tetrahedral exports")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
