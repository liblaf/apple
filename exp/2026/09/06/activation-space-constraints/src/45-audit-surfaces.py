"""Audit optimized block surfaces for self-intersection with IPC Toolkit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DEFAULT_MATRIX_DIR = EXPERIMENT / "data" / "20-matrix"
DEFAULT_OUTPUT_DIR = EXPERIMENT / "data" / "45-audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Also audit every saved step-*.npz checkpoint.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boundary_surface(
    points: npt.NDArray[np.float64], tets: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int64]]:
    """Extract and compact all tetrahedral faces incident on exactly one tet."""
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert tets.ndim == 2
    assert tets.shape[1] == 4
    assert np.all((tets >= 0) & (tets < len(points)))

    faces = np.concatenate(
        (
            tets[:, (0, 1, 2)],
            tets[:, (0, 1, 3)],
            tets[:, (0, 2, 3)],
            tets[:, (1, 2, 3)],
        )
    )
    canonical = np.sort(faces, axis=1)
    _, first_indices, counts = np.unique(
        canonical, axis=0, return_index=True, return_counts=True
    )
    assert np.all((counts == 1) | (counts == 2)), "Non-manifold tet face found"
    boundary = faces[first_indices[counts == 1]]

    point_ids, inverse = np.unique(boundary, return_inverse=True)
    compact_faces = inverse.reshape((-1, 3)).astype(np.int32)
    return (
        np.asfortranarray(points[point_ids], dtype=np.float64),
        np.asfortranarray(compact_faces),
        point_ids,
    )


def audit_positions(
    mesh: ipctk.CollisionMesh, positions: npt.NDArray[np.float64]
) -> bool:
    assert positions.ndim == 2
    assert positions.shape[1] == 3
    return bool(
        ipctk.has_intersections(mesh, np.asfortranarray(positions, dtype=np.float64))
    )


def completed_cases(matrix_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    cases = []
    for summary_path in matrix_dir.glob("*/*/summary.json"):
        case_dir = summary_path.parent
        if not (case_dir / "final.npz").is_file():
            continue
        summary = json.loads(summary_path.read_text())
        assert isinstance(summary, dict), f"Expected JSON object in {summary_path}"
        cases.append((case_dir, summary))
    return sorted(cases, key=lambda item: item[0].relative_to(matrix_dir).as_posix())


def result_paths(case_dir: Path, *, include_checkpoints: bool) -> list[Path]:
    paths = [case_dir / "final.npz"]
    if include_checkpoints:
        paths.extend(case_dir.glob("step-*.npz"))
    return sorted(paths, key=lambda path: (path.name != "final.npz", path.name))


def audit_result(
    path: Path,
    matrix_dir: Path,
    points: npt.NDArray[np.float64],
    boundary_point_ids: npt.NDArray[np.int64],
    mesh: ipctk.CollisionMesh,
) -> dict[str, Any]:
    with np.load(path) as data:
        assert "u" in data.files, f"Missing u in {path}"
        displacement = np.asarray(data["u"], dtype=np.float64)
    assert displacement.shape == points.shape, (
        f"Expected displacement shape {points.shape}, got {displacement.shape} in {path}"
    )
    assert np.all(np.isfinite(displacement)), f"Non-finite displacement in {path}"

    positions = points[boundary_point_ids] + displacement[boundary_point_ids]
    return {
        "path": path.relative_to(matrix_dir).as_posix(),
        "sha256": sha256(path),
        "has_intersections": audit_positions(mesh, positions),
    }


def main() -> None:
    args = parse_args()
    matrix_dir = args.matrix_dir.resolve()
    output_dir = args.output_dir.resolve()
    fixture_path = matrix_dir / "fixture.npz"

    with np.load(fixture_path) as fixture:
        points = np.asarray(fixture["points"], dtype=np.float64)
        tets = np.asarray(fixture["tets"], dtype=np.int64)

    rest_positions, faces, boundary_point_ids = boundary_surface(points, tets)
    edges = np.asfortranarray(ipctk.edges(faces), dtype=np.int32)
    mesh = ipctk.CollisionMesh(rest_positions, edges, faces)
    rest_has_intersections = audit_positions(mesh, rest_positions)
    assert not rest_has_intersections, "Rest boundary has a self-intersection"

    cases = completed_cases(matrix_dir)
    results = []
    case_summaries = []
    for case_dir, case_summary in cases:
        case_path = case_dir.relative_to(matrix_dir).as_posix()
        case_results = [
            audit_result(path, matrix_dir, points, boundary_point_ids, mesh)
            for path in result_paths(
                case_dir, include_checkpoints=args.include_checkpoints
            )
        ]
        results.extend(case_results)
        case_summaries.append(
            {
                "path": case_path,
                "optimization_status": case_summary.get("status"),
                "num_audited_states": len(case_results),
                "all_states_intersection_free": all(
                    not result["has_intersections"] for result in case_results
                ),
            }
        )
    summary = {
        "method": "ipctk.has_intersections on the complete compact tet boundary",
        "ipctk_version": ipctk.__version__,
        "fixture": {
            "path": fixture_path.relative_to(EXPERIMENT).as_posix(),
            "sha256": sha256(fixture_path),
            "num_volume_vertices": len(points),
            "num_tets": len(tets),
            "num_boundary_vertices": len(rest_positions),
            "num_boundary_edges": len(edges),
            "num_boundary_faces": len(faces),
            "rest_has_intersections": rest_has_intersections,
        },
        "include_checkpoints": bool(args.include_checkpoints),
        "num_completed_cases": len(cases),
        "cases": case_summaries,
        "num_results": len(results),
        "num_with_intersections": sum(
            result["has_intersections"] for result in results
        ),
        "all_results_intersection_free": all(
            not result["has_intersections"] for result in results
        ),
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
