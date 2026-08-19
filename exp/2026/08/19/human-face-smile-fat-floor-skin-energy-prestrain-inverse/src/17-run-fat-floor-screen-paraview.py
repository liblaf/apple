from __future__ import annotations

# The wrapper prepares audited IsFace surfaces and delegates every rendered pixel
# and every saved view state to the pinned ParaView 6.1.1 pvbatch executable.
# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
import hashlib
import json
import logging
import math
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonExecutionModel import (
    vtkStreamingDemandDrivenPipeline as StreamingPipeline,
)

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DESIGN = "fat-floor-fixed-activation-paraview-screen-v1"
EXPECTED_PARAVIEW_VERSION = "6.1.1"
EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_EXPANDING_TRIANGLES = 16_723
BRANCH_ORDER = ("P0", "P1")
STATE_ORDER = ("target", "old-e0", "new-efat-zero", "new-efat-old-seed")
VIEW_ORDER = ("front", "30-degree", "mouth", "eye-cheek+x")
MODE_ORDER = ("geometry", "normal-residual")
IMAGE_RESOLUTION = (4000, 3000)

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_DIR = GROUP_DIR / "data"
OLD_DATA_DIR = (
    REPO_ROOT
    / "exp/2026/08/19/human-face-smile-selective-skin-energy-prestrain-inverse/data"
)
SUMMARY = DATA_DIR / "15-fixed-activation-fat-floor-screen-summary.json"
SCREEN_PRODUCER = Path(__file__).with_name("15-fixed-activation-fat-floor-screen.py")
RENDERER = Path(__file__).with_name("16-render-fat-floor-screen-paraview.py")
PVBATCH = Path("/usr/bin/pvbatch")
BUNDLE_ROOT = DATA_DIR / "17-paraview-fat-floor-screen"
INPUT_ROOT = BUNDLE_ROOT / "inputs"
PLATE_ROOT = BUNDLE_ROOT / "plates"
CONTRACT = BUNDLE_ROOT / "contract.json"
RECEIPT = DATA_DIR / "17-paraview-fat-floor-screen-receipt.json"

# There is no CLI bypass.  Static review must finish before an isolated approval
# edit changes this boolean and the corresponding renderer boolean to True.
PARAVIEW_EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False

EXPECTED_RENDERER_PREAPPROVAL_SIZE_BYTES = 11_585
EXPECTED_RENDERER_PREAPPROVAL_SHA256 = (
    "133efe01c56024dcc4feb7592e612bee08384d93d22460996ea1250552e33eb4"
)
EXPECTED_RENDERER_EXECUTABLE_SIZE_BYTES = 11_584
EXPECTED_RENDERER_EXECUTABLE_SHA256 = (
    "20063cc87733a92258802f5771614ef49f3fec2a4b9127436b68f3ac3f260444"
)
EXPECTED_PVBATCH_SIZE_BYTES = 18_608
EXPECTED_PVBATCH_SHA256 = (
    "be482a75b1e52a8b5d9df6c5687c743cc0b5312e30916622d54652a998eb8871"
)


@dataclass(frozen=True)
class IdentitySpec:
    path: Path
    size_bytes: int
    sha256: str


INPUTS = {
    "summary": IdentitySpec(
        SUMMARY,
        79_002,
        "69b01c2b20a1ff9bef89ed5ebe34e27416f6796494448a5ede2b4e4e7efdb43e",
    ),
    "screen_producer": IdentitySpec(
        SCREEN_PRODUCER,
        100_982,
        "19b451ce3c51842ca6735805e0016ef4a2aa360b47ae8e4d74e5953357aaf791",
    ),
    "skin_hfp0": IdentitySpec(
        DATA_DIR / "10-prepared-material-cases-v2/skin-hfp0-selective-efat-p000.vtp",
        1_611_312,
        "8aff20c6ecad328bb436213c7751c25546df3b12a01e3f6e748c24e4b9941f23",
    ),
    "skin_hfp1": IdentitySpec(
        DATA_DIR / "10-prepared-material-cases-v2/skin-hfp1-selective-efat-c020.vtp",
        2_072_304,
        "89e0b349b1ba8002bc654325ba2f025c492b6e096c242c4c194ddede72cd117d",
    ),
    "h1p0_history": IdentitySpec(
        OLD_DATA_DIR / "20-h1p0-steps.vtkhdf",
        2_066_227_811,
        "abd81c7c51480ccc10c35488f22a7734e82c171093fc4ef5443762d18507c93c",
    ),
    "h1p1_history": IdentitySpec(
        OLD_DATA_DIR / "20-h1p1-steps.vtkhdf",
        2_072_876_568,
        "c32ede23f73ae27a301d845e6f8598494f03e1927dce043118e1a52b49092b8b",
    ),
    "hfp0_zero": IdentitySpec(
        DATA_DIR
        / "15-fixed-activation-fat-floor-screen/HFP0-from-H1P0-frame014-zero/result.vtu",
        148_128_352,
        "503a49dfcb9ed99ff665e7e146a8d659c6f138f22e11273cc3e03136e1980c82",
    ),
    "hfp0_old_seed": IdentitySpec(
        DATA_DIR
        / "15-fixed-activation-fat-floor-screen/HFP0-from-H1P0-frame014-old-equilibrium/result.vtu",
        148_128_466,
        "210d19af4dbdbae37a9b2540ae4506e2aeb5237ff7e7b683bf284252c6712325",
    ),
    "hfp1_zero": IdentitySpec(
        DATA_DIR
        / "15-fixed-activation-fat-floor-screen/HFP1-from-H1P1-frame012-zero/result.vtu",
        148_111_582,
        "6269e130d0971ba62166d7605003eab4aa3d9dfc7f2ce8d09c22f41fb7707fb4",
    ),
    "hfp1_old_seed": IdentitySpec(
        DATA_DIR
        / "15-fixed-activation-fat-floor-screen/HFP1-from-H1P1-frame012-old-equilibrium/result.vtu",
        148_110_456,
        "b93a8f49a66010999b86899afe2f930d0d6880ca606caa715f7f6a2062ba0361",
    ),
}

CASE_IDS = {
    ("P0", "new-efat-zero"): "HFP0-from-H1P0-frame014-zero",
    ("P0", "new-efat-old-seed"): "HFP0-from-H1P0-frame014-old-equilibrium",
    ("P1", "new-efat-zero"): "HFP1-from-H1P1-frame012-zero",
    ("P1", "new-efat-old-seed"): "HFP1-from-H1P1-frame012-old-equilibrium",
}


class Config(cherries.BaseConfig):
    # The two histories intentionally remain ordinary pinned Paths: declaring or
    # logging them as Cherries inputs would copy about 4.1 GB into the run snapshot.
    input_summary: Path = cherries.input(SUMMARY)
    output_receipt: Path = cherries.output(RECEIPT, mkdir=True)


@dataclass
class TemporalHistory:
    label: str
    path: Path
    reader: Any
    times: np.ndarray

    @classmethod
    def open(cls, label: str, path: Path) -> TemporalHistory:
        reader = pv.get_reader(path)
        vtk_reader = reader.reader
        vtk_reader.UpdateInformation()
        information = vtk_reader.GetOutputInformation(0)
        key = StreamingPipeline.TIME_STEPS()
        if not information.Has(key):
            raise ValueError(f"{label} history has no TIME_STEPS: {path}")
        times = np.asarray(
            [information.Get(key, index) for index in range(information.Length(key))],
            dtype=np.float64,
        )
        if not np.array_equal(times, np.arange(41, dtype=np.float64)):
            raise ValueError(f"{label} history TIME_STEPS are not exactly 0..40")
        return cls(label=label, path=path, reader=reader, times=times)

    def frame(self, step: int) -> pv.UnstructuredGrid:
        if not 0 <= step < self.times.size:
            raise IndexError(f"invalid {self.label} history step {step}")
        vtk_reader = self.reader.reader
        vtk_reader.UpdateTimeStep(float(self.times[step]))
        frame = pv.wrap(vtk_reader.GetOutputDataObject(0)).copy(deep=True)
        if not isinstance(frame, pv.UnstructuredGrid):
            raise TypeError(f"{self.label}@{step} is not an UnstructuredGrid")
        return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_identity(data: bytes) -> dict[str, Any]:
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _require_identity(label: str, spec: IdentitySpec) -> dict[str, Any]:
    actual = _identity(spec.path)
    expected = {
        "path": str(spec.path.resolve()),
        "size_bytes": spec.size_bytes,
        "sha256": spec.sha256,
    }
    if actual != expected:
        raise ValueError(f"{label} identity changed: {actual} != {expected}")
    return actual


def _raw_sha256(values: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _file_identity_matches(item: dict[str, Any], spec: IdentitySpec) -> bool:
    return (
        Path(str(item.get("path", ""))).resolve() == spec.path.resolve()
        and int(item.get("size_bytes", -1)) == spec.size_bytes
        and str(item.get("sha256", "")) == spec.sha256
    )


def _validate_summary(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if summary.get("schema_version") != 1 or summary.get("complete") is not True:
        raise ValueError("forward-screen summary is incomplete")
    if summary.get("status") != "ok" or summary.get("n_forward_solves") != 4:
        raise ValueError("forward-screen completion status changed")
    execution = summary.get("execution")
    if not isinstance(execution, dict) or execution != {
        "fixed_activation_forward_only": True,
        "inverse_executed": False,
        "adjoint_executed": False,
        "backward_executed": False,
        "fresh_forward_per_case": True,
        "seeds": ["exact-zero", "exact-old-corresponding-equilibrium"],
    }:
        raise ValueError("forward-screen execution contract changed")
    expected_case_order = [CASE_IDS[key] for key in CASE_IDS]
    if summary.get("case_order") != expected_case_order:
        raise ValueError("forward-screen case order changed")
    if not _file_identity_matches(summary["producer"], INPUTS["screen_producer"]):
        raise ValueError("summary producer identity changed")
    if not _file_identity_matches(
        summary["skin_contracts"]["HFP0"]["identity"], INPUTS["skin_hfp0"]
    ) or not _file_identity_matches(
        summary["skin_contracts"]["HFP1"]["identity"], INPUTS["skin_hfp1"]
    ):
        raise ValueError("summary clean-v2 skin identities changed")
    source_frames = summary["source_frames"]
    if not _file_identity_matches(
        source_frames["H1P0"]["history"], INPUTS["h1p0_history"]
    ) or not _file_identity_matches(
        source_frames["H1P1"]["history"], INPUTS["h1p1_history"]
    ):
        raise ValueError("summary source-history identities changed")
    if (
        source_frames["H1P0"].get("step") != 14
        or source_frames["H1P1"].get("step") != 12
    ):
        raise ValueError("summary source-frame steps changed")
    if (
        source_frames["H1P0"].get("source_case") != "H1P0"
        or source_frames["H1P1"].get("source_case") != "H1P1"
    ):
        raise ValueError("summary source-frame case identifiers changed")
    if summary.get("old_aggregate") != {
        "source_case_order": ["H0P1", "H1P1", "H1P0"],
        "verified": True,
    }:
        raise ValueError("summary no longer certifies the old E=0 H1 references")
    rows = summary.get("cases")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("forward-screen case rows changed")
    by_case = {str(row["case_id"]): row for row in rows}
    if list(by_case) != expected_case_order:
        raise ValueError("forward-screen case-row order changed")
    result_specs = {
        CASE_IDS[("P0", "new-efat-zero")]: INPUTS["hfp0_zero"],
        CASE_IDS[("P0", "new-efat-old-seed")]: INPUTS["hfp0_old_seed"],
        CASE_IDS[("P1", "new-efat-zero")]: INPUTS["hfp1_zero"],
        CASE_IDS[("P1", "new-efat-old-seed")]: INPUTS["hfp1_old_seed"],
    }
    for case_id, spec in result_specs.items():
        row = by_case[case_id]
        if row.get("status") != "ok" or row.get("forward/success") is not True:
            raise ValueError(f"{case_id} is not a successful forward result")
        if (
            row.get("fixed_activation") is not True
            or row.get("new_inverse") is not False
        ):
            raise ValueError(f"{case_id} is not fixed-activation forward-only")
        if not _file_identity_matches(
            row["artifact"]["file_identity"] | {"path": row["artifact"]["path"]}, spec
        ):
            raise ValueError(f"{case_id} artifact identity changed")
        for key, value in row.items():
            if isinstance(value, int | float) and not math.isfinite(float(value)):
                raise ValueError(f"{case_id} contains non-finite metric {key}")
    expected_rows = {
        CASE_IDS[("P0", "new-efat-zero")]: ("HFP0", "H1P0", 14, "zero"),
        CASE_IDS[("P0", "new-efat-old-seed")]: (
            "HFP0",
            "H1P0",
            14,
            "old-equilibrium",
        ),
        CASE_IDS[("P1", "new-efat-zero")]: ("HFP1", "H1P1", 12, "zero"),
        CASE_IDS[("P1", "new-efat-old-seed")]: (
            "HFP1",
            "H1P1",
            12,
            "old-equilibrium",
        ),
    }
    for case_id, expected in expected_rows.items():
        row = by_case[case_id]
        actual = (
            row.get("material_case"),
            row.get("source_case"),
            row.get("source_step"),
            row.get("seed"),
        )
        if actual != expected:
            raise ValueError(f"{case_id} branch/seed contract changed")
    return by_case


def _validate_config(cfg: Config) -> None:
    if Path(cfg.input_summary).resolve() != SUMMARY.resolve():
        raise ValueError("summary input cannot be overridden")
    if Path(cfg.output_receipt).resolve() != RECEIPT.resolve():
        raise ValueError("receipt output cannot be overridden")
    stale = [
        path
        for path in (BUNDLE_ROOT, RECEIPT, _temporary_path(RECEIPT))
        if path.exists()
    ]
    if stale:
        raise FileExistsError(f"refusing stale ParaView outputs: {stale}")
    if os.environ.get("DEBUG") != "1":
        raise RuntimeError(
            "NO-GO: this visualization must run with DEBUG=1/profile=debug"
        )
    if not PARAVIEW_EXECUTION_APPROVED_AFTER_STATIC_REVIEW:
        raise RuntimeError(
            "NO-GO: ParaView visualization awaits static review and isolated source approval"
        )


def _renderer_provenance() -> dict[str, Any]:
    data = RENDERER.read_bytes()
    false_marker = b"PARAVIEW_RENDER_EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False"
    true_marker = b"PARAVIEW_RENDER_EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True"
    if data.count(true_marker) != 1 or data.count(false_marker) != 0:
        raise RuntimeError("reviewed ParaView renderer is not uniquely source-approved")
    live = _bytes_identity(data)
    expected_live = {
        "size_bytes": EXPECTED_RENDERER_EXECUTABLE_SIZE_BYTES,
        "sha256": EXPECTED_RENDERER_EXECUTABLE_SHA256,
    }
    if live != expected_live:
        raise ValueError("approved ParaView renderer identity changed")
    reconstructed = _bytes_identity(data.replace(true_marker, false_marker))
    expected_reconstructed = {
        "size_bytes": EXPECTED_RENDERER_PREAPPROVAL_SIZE_BYTES,
        "sha256": EXPECTED_RENDERER_PREAPPROVAL_SHA256,
    }
    if reconstructed != expected_reconstructed:
        raise ValueError("renderer approval-only reconstruction failed")
    return {
        "path": str(RENDERER.resolve()),
        "live": live,
        "statically_reviewed_preapproval": expected_reconstructed,
        "approval_only_reconstruction": True,
    }


def _paraview_version() -> tuple[str, dict[str, Any]]:
    identity = _identity(PVBATCH)
    expected = {
        "path": str(PVBATCH.resolve()),
        "size_bytes": EXPECTED_PVBATCH_SIZE_BYTES,
        "sha256": EXPECTED_PVBATCH_SHA256,
    }
    if identity != expected:
        raise ValueError("pvbatch executable identity changed")
    completed = subprocess.run(
        [str(PVBATCH), "--version"], check=True, capture_output=True, text=True
    )
    combined = f"{completed.stdout}\n{completed.stderr}".strip()
    if not combined.endswith(EXPECTED_PARAVIEW_VERSION):
        raise RuntimeError(
            f"requires ParaView {EXPECTED_PARAVIEW_VERSION}; got {combined!r}"
        )
    return EXPECTED_PARAVIEW_VERSION, identity


def _snapshot_inputs(phase: str) -> dict[str, dict[str, Any]]:
    return {
        label: {"phase": phase, **_require_identity(label, spec)}
        for label, spec in INPUTS.items()
    }


def _require_array_hash(
    label: str, values: np.ndarray, expected: str, *, dtype: str
) -> None:
    actual = _raw_sha256(values, dtype=dtype)
    if actual != expected:
        raise ValueError(f"{label} hash changed: {actual} != {expected}")


def _load_result_fields(
    *,
    label: str,
    row: dict[str, Any],
    spec: IdentitySpec,
    canonical: dict[str, np.ndarray] | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mesh = pv.read(spec.path)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError(f"{label} is not an UnstructuredGrid")
    if mesh.n_points != EXPECTED_POINTS or mesh.n_cells != EXPECTED_TETS:
        raise ValueError(f"{label} dimensions changed")
    required = {
        "GlobalPointId",
        "Displacement",
        "TargetDisplacement",
        "LossMask",
        "IsFixed",
        "ArtificialCutIncident",
        "IsLip",
    }
    if not required <= set(mesh.point_data):
        raise KeyError(f"{label} misses required point arrays")
    fields = {
        "points": np.asarray(mesh.points, dtype=np.float64).copy(),
        "global_ids": np.asarray(
            mesh.point_data["GlobalPointId"], dtype=np.int64
        ).copy(),
        "displacement": np.asarray(
            mesh.point_data["Displacement"], dtype=np.float64
        ).copy(),
        "target": np.asarray(
            mesh.point_data["TargetDisplacement"], dtype=np.float64
        ).copy(),
        "loss_mask": np.asarray(mesh.point_data["LossMask"], dtype=bool).copy(),
        "is_fixed": np.asarray(mesh.point_data["IsFixed"], dtype=bool).copy(),
        "cut": np.asarray(mesh.point_data["ArtificialCutIncident"], dtype=bool).copy(),
        "is_lip": np.asarray(mesh.point_data["IsLip"], dtype=bool).copy(),
    }
    for name, values in fields.items():
        if values.shape[0] != EXPECTED_POINTS or not np.isfinite(values).all():
            raise ValueError(f"{label} {name} shape or finiteness changed")
    hashes = row["artifact"]["array_hashes"]
    _require_array_hash(
        label + " displacement",
        fields["displacement"],
        hashes["point/Displacement"],
        dtype="<f8",
    )
    _require_array_hash(
        label + " global ids",
        fields["global_ids"],
        hashes["point/GlobalPointId"],
        dtype="<i8",
    )
    _require_array_hash(
        label + " target",
        fields["target"],
        hashes["point/TargetDisplacement"],
        dtype="<f8",
    )
    _require_array_hash(
        label + " loss mask", fields["loss_mask"], hashes["point/LossMask"], dtype="u1"
    )
    _require_array_hash(
        label + " fixed", fields["is_fixed"], hashes["point/IsFixed"], dtype="u1"
    )
    _require_array_hash(
        label + " cut", fields["cut"], hashes["point/ArtificialCutIncident"], dtype="u1"
    )
    fixed_displacement = fields["displacement"][fields["is_fixed"]]
    if not np.array_equal(fixed_displacement, np.zeros_like(fixed_displacement)):
        raise ValueError(f"{label} violates exact fixed zero")
    cut_displacement = fields["displacement"][fields["cut"]]
    if not np.array_equal(cut_displacement, np.zeros_like(cut_displacement)):
        raise ValueError(f"{label} violates exact cut zero")
    if canonical is not None:
        for name in (
            "points",
            "global_ids",
            "target",
            "loss_mask",
            "is_fixed",
            "cut",
            "is_lip",
        ):
            if not np.array_equal(fields[name], canonical[name]):
                raise ValueError(f"{label} canonical {name} changed")
    return fields, {
        "displacement": fields["displacement"],
    }


def _load_old_frame(
    *,
    label: str,
    spec: IdentitySpec,
    step: int,
    expected_displacement_sha256: str,
    expected_activation_sha256: str,
    canonical: dict[str, np.ndarray],
) -> np.ndarray:
    frame = TemporalHistory.open(label, spec.path).frame(step)
    if frame.n_points != EXPECTED_POINTS or frame.n_cells != EXPECTED_TETS:
        raise ValueError(f"{label}@{step} dimensions changed")
    if not np.array_equal(np.asarray(frame.points), canonical["points"]):
        raise ValueError(f"{label}@{step} rest points changed")
    for name, expected in (
        ("GlobalPointId", canonical["global_ids"]),
        ("TargetDisplacement", canonical["target"]),
        ("LossMask", canonical["loss_mask"]),
        ("IsFixed", canonical["is_fixed"]),
        ("ArtificialCutIncident", canonical["cut"]),
    ):
        if not np.array_equal(np.asarray(frame.point_data[name]), expected):
            raise ValueError(f"{label}@{step} {name} changed")
    displacement = np.asarray(frame.point_data["Displacement"], dtype=np.float64).copy()
    activation = np.asarray(frame.cell_data["ActivationInv"], dtype=np.float64)
    _require_array_hash(
        label + " old displacement",
        displacement,
        expected_displacement_sha256,
        dtype="<f8",
    )
    _require_array_hash(
        label + " old activation", activation, expected_activation_sha256, dtype="<f8"
    )
    if not np.isfinite(displacement).all() or not np.isfinite(activation).all():
        raise ValueError(f"{label}@{step} is non-finite")
    fixed_displacement = displacement[canonical["is_fixed"]]
    if not np.array_equal(fixed_displacement, np.zeros_like(fixed_displacement)):
        raise ValueError(f"{label}@{step} violates exact fixed zero")
    cut_displacement = displacement[canonical["cut"]]
    if not np.array_equal(cut_displacement, np.zeros_like(cut_displacement)):
        raise ValueError(f"{label}@{step} violates exact cut zero")
    return displacement


def _skin_mapping(skin: pv.PolyData, canonical_ids: np.ndarray) -> np.ndarray:
    if skin.n_points != EXPECTED_SKIN_POINTS or skin.n_cells != EXPECTED_SKIN_TRIANGLES:
        raise ValueError("IsFace skin dimensions changed")
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    if np.unique(skin_ids).size != EXPECTED_SKIN_POINTS:
        raise ValueError("IsFace GlobalPointId values are not unique")
    order = np.argsort(canonical_ids)
    sorted_ids = canonical_ids[order]
    positions = np.searchsorted(sorted_ids, skin_ids)
    if np.any(positions >= sorted_ids.size) or not np.array_equal(
        sorted_ids[positions], skin_ids
    ):
        raise ValueError("IsFace points do not map exactly into the anatomy mesh")
    return order[positions]


def _validate_skin_materials(
    *, skin_hfp0: pv.PolyData, skin_hfp1: pv.PolyData
) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "ExpandingTriangle",
        "TargetRestAreaRatio",
        "ClippedTargetRestAreaRatio",
        "SkinYoungModulusMPa",
        "SkinPoissonRatio",
        "C020PrestrainEnabled",
        "StressFreeAreaRatio",
        "SkinActivationInvDiag",
        "ActivationInv",
    }
    for label, skin in (("HFP0", skin_hfp0), ("HFP1", skin_hfp1)):
        if not required <= set(skin.cell_data):
            raise KeyError(f"{label} clean-v2 skin material arrays changed")
    expansion = np.asarray(skin_hfp0.cell_data["ExpandingTriangle"], dtype=bool)
    raw_ratio = np.asarray(skin_hfp0.cell_data["TargetRestAreaRatio"], dtype=np.float64)
    clipped = np.clip(raw_ratio, 0.5, 1.0)
    expected_young = np.where(expansion, 0.003, 0.2)
    expected_nu = np.full(EXPECTED_SKIN_TRIANGLES, 0.49, dtype=np.float64)
    for label, skin in (("HFP0", skin_hfp0), ("HFP1", skin_hfp1)):
        if not np.array_equal(
            np.asarray(skin.cell_data["ExpandingTriangle"], dtype=bool), expansion
        ) or not np.array_equal(
            np.asarray(skin.cell_data["TargetRestAreaRatio"], dtype=np.float64),
            raw_ratio,
        ):
            raise ValueError(f"{label} expansion or raw area-ratio field changed")
        if not np.array_equal(
            np.asarray(skin.cell_data["ClippedTargetRestAreaRatio"], dtype=np.float64),
            clipped,
        ):
            raise ValueError(f"{label} clipped area-ratio formula changed")
        if not np.array_equal(
            np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64),
            expected_young,
        ):
            raise ValueError(f"{label} no longer uses the exact .003 MPa fat-E floor")
        if not np.array_equal(
            np.asarray(skin.cell_data["SkinPoissonRatio"], dtype=np.float64),
            expected_nu,
        ):
            raise ValueError(f"{label} skin Poisson ratio changed")
    zeros = np.zeros(EXPECTED_SKIN_TRIANGLES, dtype=np.float64)
    if np.any(
        np.asarray(skin_hfp0.cell_data["C020PrestrainEnabled"], dtype=bool)
    ) or not np.array_equal(
        np.asarray(skin_hfp0.cell_data["StressFreeAreaRatio"], dtype=np.float64),
        np.ones(EXPECTED_SKIN_TRIANGLES, dtype=np.float64),
    ):
        raise ValueError("HFP0 is no longer the exact p000 material")
    if not np.array_equal(
        np.asarray(skin_hfp0.cell_data["SkinActivationInvDiag"], dtype=np.float64),
        zeros,
    ) or not np.array_equal(
        np.asarray(skin_hfp0.cell_data["ActivationInv"], dtype=np.float64),
        np.zeros((EXPECTED_SKIN_TRIANGLES, 3), dtype=np.float64),
    ):
        raise ValueError("HFP0 has nonzero skin prestrain")
    stress_free = np.square(0.98) * clipped
    diagonal = np.reciprocal(np.sqrt(stress_free)) - 1.0
    activation = np.column_stack((diagonal, diagonal, zeros))
    if not np.all(
        np.asarray(skin_hfp1.cell_data["C020PrestrainEnabled"], dtype=bool)
    ) or not np.array_equal(
        np.asarray(skin_hfp1.cell_data["StressFreeAreaRatio"], dtype=np.float64),
        stress_free,
    ):
        raise ValueError("HFP1 is no longer the exact c020 stress-free-area material")
    if not np.array_equal(
        np.asarray(skin_hfp1.cell_data["SkinActivationInvDiag"], dtype=np.float64),
        diagonal,
    ) or not np.array_equal(
        np.asarray(skin_hfp1.cell_data["ActivationInv"], dtype=np.float64),
        activation,
    ):
        raise ValueError("HFP1 c020 ActivationInv formula changed")
    return expansion, raw_ratio


def _triangles(skin: pv.PolyData) -> np.ndarray:
    faces = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    if faces.shape != (EXPECTED_SKIN_TRIANGLES, 4) or not np.all(faces[:, 0] == 3):
        raise ValueError("IsFace skin is not exactly canonical triangles")
    return faces[:, 1:].copy()


def _target_vertex_normals(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    vectors = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    if not np.isfinite(vectors).all() or np.any(
        np.linalg.norm(vectors, axis=1) <= np.finfo(np.float64).eps
    ):
        raise ValueError("target IsFace geometry has a degenerate triangle")
    normals = np.zeros_like(points)
    for local in range(3):
        np.add.at(normals, triangles[:, local], vectors)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= np.finfo(np.float64).eps):
        raise ValueError("target IsFace geometry has an undefined vertex normal")
    return normals / lengths[:, None]


def _bounds_camera(
    points: np.ndarray, *, padding: float = 1.12, aspect: float = 1.35
) -> tuple[list[float], float]:
    low, high = points.min(axis=0), points.max(axis=0)
    focus = 0.5 * (low + high)
    extent = high - low
    scale = padding * 0.5 * max(float(extent[1]), float(extent[0]) / aspect)
    if not np.isfinite(focus).all() or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid fixed ParaView camera")
    return focus.tolist(), scale


def _views(
    *, skin: pv.PolyData, skin_points: np.ndarray, is_lip: np.ndarray
) -> dict[str, dict[str, Any]]:
    face_focus, face_scale = _bounds_camera(skin_points)
    if not np.any(is_lip):
        raise ValueError("mapped IsLip camera mask is empty")
    mouth_focus, mouth_scale = _bounds_camera(skin_points[is_lip], padding=1.25)
    names = tuple(str(value) for value in skin.field_data["GroupName"])
    group_ids = np.asarray(skin.point_data["GroupId"], dtype=np.int64)
    eyelid_names = {"EyelidTop", "EyelidBottom", "EyelidOuterTop", "EyelidOuterBottom"}
    eyelid_ids = [index for index, name in enumerate(names) if name in eyelid_names]
    eyelid = np.isin(group_ids, eyelid_ids)
    if not np.any(eyelid):
        raise ValueError("eyelid camera mask is empty")
    one_eye = eyelid & (skin_points[:, 0] >= np.median(skin_points[eyelid, 0]))
    eye_focus, _ = _bounds_camera(skin_points[one_eye])
    eye_focus[1] -= 0.08 * float(np.ptp(skin_points[:, 1]))
    return {
        "front": {
            "direction": [0.0, 0.0, 1.0],
            "focus": face_focus,
            "parallel_scale": face_scale,
        },
        "30-degree": {
            "direction": [0.5, 0.0, math.sqrt(3.0) / 2.0],
            "focus": face_focus,
            "parallel_scale": face_scale,
        },
        "mouth": {
            "direction": [0.0, 0.0, 1.0],
            "focus": mouth_focus,
            "parallel_scale": mouth_scale,
        },
        "eye-cheek+x": {
            "direction": [0.0, 0.0, 1.0],
            "focus": eye_focus,
            "parallel_scale": 0.24 * float(np.ptp(skin_points[:, 1])),
        },
    }


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _write_surface_atomic(
    *,
    path: Path,
    rest_skin: pv.PolyData,
    triangles: np.ndarray,
    skin_ids: np.ndarray,
    displacement: np.ndarray,
    target: np.ndarray,
    target_normals: np.ndarray,
    expansion: np.ndarray,
    raw_ratio: np.ndarray,
) -> dict[str, Any]:
    skin_displacement = displacement[skin_ids]
    skin_target = target[skin_ids]
    residual = skin_displacement - skin_target
    normal_residual_mm = 1.0e3 * np.einsum("ij,ij->i", residual, target_normals)
    points = np.asarray(rest_skin.points, dtype=np.float64) + skin_displacement
    arrays = {
        "GlobalPointId": np.asarray(
            rest_skin.point_data["GlobalPointId"], dtype=np.int64
        ),
        "TargetNormalResidualMM": normal_residual_mm,
        "DisplacementMM": 1.0e3 * skin_displacement,
        "TargetDisplacementMM": 1.0e3 * skin_target,
        "ResidualDisplacementMM": 1.0e3 * residual,
    }
    if not np.isfinite(points).all() or any(
        not np.isfinite(values).all() for values in arrays.values()
    ):
        raise ValueError(f"non-finite ParaView surface values for {path}")
    surface = pv.PolyData(
        points, np.column_stack((np.full(triangles.shape[0], 3), triangles))
    )
    for name, values in arrays.items():
        surface.point_data[name] = values
    surface.cell_data["ExpansionRegion"] = np.asarray(expansion, dtype=np.int8)
    surface.cell_data["TargetRestAreaRatio"] = np.asarray(raw_ratio, dtype=np.float64)
    temporary = _temporary_path(path)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing stale ParaView surface: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    surface.save(temporary)
    loaded = pv.read(temporary)
    if (
        not isinstance(loaded, pv.PolyData)
        or loaded.n_points != EXPECTED_SKIN_POINTS
        or loaded.n_cells != EXPECTED_SKIN_TRIANGLES
    ):
        raise ValueError(f"ParaView surface temporary readback changed: {path}")
    if not np.array_equal(np.asarray(loaded.points), points):
        raise ValueError(f"ParaView surface point readback changed: {path}")
    for name, values in arrays.items():
        if not np.array_equal(np.asarray(loaded.point_data[name]), values):
            raise ValueError(f"ParaView surface {name} readback changed: {path}")
    if not np.array_equal(np.asarray(loaded.cell_data["ExpansionRegion"]), expansion):
        raise ValueError(f"ParaView expansion mask readback changed: {path}")
    temporary.replace(path)
    final = pv.read(path)
    if not np.array_equal(np.asarray(final.points), points):
        raise ValueError(f"ParaView surface final readback changed: {path}")
    identity = _identity(path)
    return {
        **identity,
        "point_array_hashes": {
            name: _raw_sha256(values, dtype="<i8" if name == "GlobalPointId" else "<f8")
            for name, values in arrays.items()
        },
        "target_normal_residual_mm": normal_residual_mm,
    }


def _metric_label(metrics: dict[str, Any] | None) -> str:
    if metrics is None:
        return "reference geometry | target-normal residual = 0"
    return (
        f"err={float(metrics['target/error_rms_mm']):.3f} mm | "
        f"Dexp={float(metrics['metric/D_exp_deg']):.2f} deg\n"
        f"Lexp={float(metrics['metric/L_exp_mm']):.3f} mm | "
        f"Q95exp={float(metrics['metric/Q95_exp_mm']):.3f} mm | "
        f"folds={int(metrics['warning/isface_folded_triangles'])}"
    )


def _material_label(branch: str, state: str) -> str:
    prestrain = "p000 (no skin prestrain)" if branch == "P0" else "c020 skin prestrain"
    if state == "target":
        return f"target reference | {prestrain} branch"
    if state == "old-e0":
        return f"skin E=.2 MPa; expansion E=0 | {prestrain}"
    return f"skin E=.2 MPa; expansion E=.003 MPa (fat E value) | {prestrain}"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    temporary = _temporary_path(path)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing stale JSON output: {path}")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    if _read_json(temporary) != payload:
        raise RuntimeError(f"strict JSON temporary readback failed: {path}")
    temporary.replace(path)
    if _read_json(path) != payload:
        raise RuntimeError(f"strict JSON final readback failed: {path}")
    return _identity(path)


def _prepare_contract(
    *,
    summary: dict[str, Any],
    by_case: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_specs = {
        CASE_IDS[("P0", "new-efat-zero")]: INPUTS["hfp0_zero"],
        CASE_IDS[("P0", "new-efat-old-seed")]: INPUTS["hfp0_old_seed"],
        CASE_IDS[("P1", "new-efat-zero")]: INPUTS["hfp1_zero"],
        CASE_IDS[("P1", "new-efat-old-seed")]: INPUTS["hfp1_old_seed"],
    }
    fields_by_case: dict[str, dict[str, np.ndarray]] = {}
    canonical: dict[str, np.ndarray] | None = None
    for case_id in summary["case_order"]:
        fields, _ = _load_result_fields(
            label=case_id,
            row=by_case[case_id],
            spec=result_specs[case_id],
            canonical=canonical,
        )
        if canonical is None:
            canonical = {
                name: values
                for name, values in fields.items()
                if name != "displacement"
            }
        fields_by_case[case_id] = fields
    if canonical is None:
        raise RuntimeError("no canonical forward result loaded")

    source_frames = summary["source_frames"]
    old_displacements = {
        "P0": _load_old_frame(
            label="H1P0",
            spec=INPUTS["h1p0_history"],
            step=14,
            expected_displacement_sha256=source_frames["H1P0"][
                "displacement_sha256_le_f8"
            ],
            expected_activation_sha256=source_frames["H1P0"]["activation_sha256_le_f8"],
            canonical=canonical,
        ),
        "P1": _load_old_frame(
            label="H1P1",
            spec=INPUTS["h1p1_history"],
            step=12,
            expected_displacement_sha256=source_frames["H1P1"][
                "displacement_sha256_le_f8"
            ],
            expected_activation_sha256=source_frames["H1P1"]["activation_sha256_le_f8"],
            canonical=canonical,
        ),
    }

    skins = {
        "P0": pv.read(INPUTS["skin_hfp0"].path),
        "P1": pv.read(INPUTS["skin_hfp1"].path),
    }
    if not all(isinstance(skin, pv.PolyData) for skin in skins.values()):
        raise TypeError("clean-v2 IsFace inputs are not PolyData")
    skin = skins["P0"]
    if not np.array_equal(
        np.asarray(skin.points), np.asarray(skins["P1"].points)
    ) or not np.array_equal(np.asarray(skin.faces), np.asarray(skins["P1"].faces)):
        raise ValueError("HFP0/HFP1 IsFace topology changed")
    skin_ids = _skin_mapping(skin, canonical["global_ids"])
    if not np.array_equal(np.asarray(skin.points), canonical["points"][skin_ids]):
        raise ValueError("IsFace rest points differ from the anatomy mesh")
    triangles = _triangles(skin)
    expansion, raw_ratio = _validate_skin_materials(
        skin_hfp0=skins["P0"], skin_hfp1=skins["P1"]
    )
    if int(expansion.sum()) != EXPECTED_EXPANDING_TRIANGLES:
        raise ValueError("clean-v2 expansion region changed")
    target_points = (
        np.asarray(skin.points, dtype=np.float64) + canonical["target"][skin_ids]
    )
    target_normals = _target_vertex_normals(target_points, triangles)
    views = _views(
        skin=skin,
        skin_points=np.asarray(skin.points, dtype=np.float64),
        is_lip=canonical["is_lip"][skin_ids],
    )

    old_metrics = summary["old_frame_metrics"]
    state_data: dict[tuple[str, str], tuple[np.ndarray, dict[str, Any] | None]] = {}
    for branch in BRANCH_ORDER:
        state_data[(branch, "target")] = (canonical["target"], None)
        source = "H1P0" if branch == "P0" else "H1P1"
        state_data[(branch, "old-e0")] = (
            old_displacements[branch],
            old_metrics[source],
        )
        for state in ("new-efat-zero", "new-efat-old-seed"):
            case_id = CASE_IDS[(branch, state)]
            state_data[(branch, state)] = (
                fields_by_case[case_id]["displacement"],
                by_case[case_id],
            )

    inputs: dict[str, Any] = {branch: {} for branch in BRANCH_ORDER}
    residual_values: list[np.ndarray] = []
    surface_receipts: list[dict[str, Any]] = []
    for branch in BRANCH_ORDER:
        branch_dir = INPUT_ROOT / branch.lower()
        for state in STATE_ORDER:
            displacement, metrics = state_data[(branch, state)]
            output = branch_dir / f"{state}.vtp"
            surface = _write_surface_atomic(
                path=output,
                rest_skin=skin,
                triangles=triangles,
                skin_ids=skin_ids,
                displacement=displacement,
                target=canonical["target"],
                target_normals=target_normals,
                expansion=expansion,
                raw_ratio=raw_ratio,
            )
            normal_residual = surface.pop("target_normal_residual_mm")
            if state != "target":
                residual_values.append(normal_residual)
            source_label = {
                "target": "target deformation",
                "old-e0": "old E=0 inverse frame",
                "new-efat-zero": "new E=.003 forward from zero seed",
                "new-efat-old-seed": "new E=.003 forward from old-equilibrium seed",
            }[state]
            entry = {
                **surface,
                "branch": branch,
                "state": state,
                "display_label": f"{branch} | {source_label}",
                "material_label": _material_label(branch, state),
                "metric_label": _metric_label(metrics),
                "metrics": None
                if metrics is None
                else {
                    key: metrics[key]
                    for key in (
                        "target/error_rms_mm",
                        "metric/D_exp_deg",
                        "metric/L_exp_mm",
                        "metric/Q95_exp_mm",
                        "warning/isface_folded_triangles",
                        "warning/inverted_tets",
                    )
                },
            }
            inputs[branch][state] = entry
            surface_receipts.append(entry)
    residual_limit = max(
        0.25,
        float(np.quantile(np.abs(np.concatenate(residual_values)), 0.99)),
    )
    if not math.isfinite(residual_limit) or residual_limit >= 100.0:
        raise ValueError("shared residual color limit is invalid")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "branch_order": list(BRANCH_ORDER),
        "state_order": list(STATE_ORDER),
        "view_order": list(VIEW_ORDER),
        "mode_order": list(MODE_ORDER),
        "image_resolution": list(IMAGE_RESOLUTION),
        "views": views,
        "inputs": inputs,
        "normal_residual_shared_limit_mm": residual_limit,
        "normal_residual_shared_limit_definition": (
            "max(0.25 mm, pooled absolute 99th percentile) across the six non-target P0/P1 states"
        ),
        "surface_domain": {
            "name": "IsFace",
            "points": EXPECTED_SKIN_POINTS,
            "triangles": EXPECTED_SKIN_TRIANGLES,
            "expanding_triangles": EXPECTED_EXPANDING_TRIANGLES,
        },
        "comparison": (
            "P0 and P1 are separate plates; each uses target, old expansion E=0, "
            "new expansion E=.003 from zero seed, and new expansion E=.003 from old-equilibrium seed"
        ),
        "renderer": (
            "ParaView 6.1.1 native geometry and scalar rendering only; no PyVista rendering"
        ),
    }
    contract_identity = _write_json_atomic(CONTRACT, contract)
    return contract, {
        "contract": contract_identity,
        "surfaces": surface_receipts,
        "skin_mapping_sha256_le_i8": _raw_sha256(skin_ids, dtype="<i8"),
        "triangle_sha256_le_i8": _raw_sha256(triangles, dtype="<i8"),
        "target_normal_sha256_le_f8": _raw_sha256(target_normals, dtype="<f8"),
    }


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"{path} is not a valid PNG header")
    return struct.unpack(">II", header[16:24])


def _expected_plate_paths() -> list[Path]:
    return [
        PLATE_ROOT / f"17-paraview-{branch.lower()}-{mode}.{suffix}"
        for branch in BRANCH_ORDER
        for mode in MODE_ORDER
        for suffix in ("png", "pvsm")
    ]


def _validate_outputs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    expected_inputs = [
        Path(contract["inputs"][branch][state]["path"])
        for branch in BRANCH_ORDER
        for state in STATE_ORDER
    ]
    expected = sorted([CONTRACT, *expected_inputs, *_expected_plate_paths()])
    actual = sorted(path for path in BUNDLE_ROOT.rglob("*") if path.is_file())
    if actual != expected:
        raise ValueError(
            f"ParaView bundle inventory changed: expected {expected}, got {actual}"
        )
    outputs: list[dict[str, Any]] = []
    for path in _expected_plate_paths():
        if path.suffix == ".png":
            if _png_size(path) != IMAGE_RESOLUTION or path.stat().st_size < 100_000:
                raise ValueError(f"{path} PNG size or payload changed")
        else:
            head = path.read_text(encoding="utf-8", errors="strict")[:1024]
            if "ParaView" not in head and "ServerManagerState" not in head:
                raise ValueError(f"{path} is not a recognizable ParaView state")
        outputs.append(_identity(path))
    for branch in BRANCH_ORDER:
        for state in STATE_ORDER:
            item = contract["inputs"][branch][state]
            path = Path(item["path"])
            identity = _identity(path)
            if (
                identity["size_bytes"] != item["size_bytes"]
                or identity["sha256"] != item["sha256"]
            ):
                raise ValueError(f"{branch}/{state} surface changed during rendering")
    if any(".tmp" in path.name for path in actual):
        raise ValueError("ParaView bundle contains a temporary file")
    return outputs


def main(cfg: Config) -> None:
    _validate_config(cfg)
    wrapper_before = _identity(Path(__file__))
    renderer_provenance = _renderer_provenance()
    paraview_version, pvbatch_identity = _paraview_version()
    inputs_before = _snapshot_inputs("pre")
    summary = _read_json(cfg.input_summary)
    by_case = _validate_summary(summary)

    BUNDLE_ROOT.mkdir(parents=True, exist_ok=False)
    INPUT_ROOT.mkdir()
    PLATE_ROOT.mkdir()
    contract, preparation = _prepare_contract(summary=summary, by_case=by_case)
    command = [
        str(PVBATCH),
        str(RENDERER.resolve()),
        "--contract",
        str(CONTRACT.resolve()),
        "--input-root",
        str(INPUT_ROOT.resolve()),
        "--output-dir",
        str(PLATE_ROOT.resolve()),
    ]
    logger.info("Running pinned native ParaView renderer: %s", command)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=GROUP_DIR,
    )
    if completed.stdout:
        logger.info("pvbatch stdout:\n%s", completed.stdout)
    if completed.stderr:
        logger.info("pvbatch stderr:\n%s", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"pvbatch failed with exit code {completed.returncode}")

    if _read_json(CONTRACT) != contract:
        raise ValueError("ParaView contract changed during rendering")
    if _identity(CONTRACT) != preparation["contract"]:
        raise ValueError("ParaView contract byte identity changed during rendering")
    outputs = _validate_outputs(contract)
    inputs_after = _snapshot_inputs("post")
    if _identity(PVBATCH) != pvbatch_identity:
        raise RuntimeError("pvbatch executable changed during rendering")
    wrapper_after = _identity(Path(__file__))
    if wrapper_before != wrapper_after:
        raise RuntimeError("ParaView wrapper source changed during execution")
    if _renderer_provenance() != renderer_provenance:
        raise RuntimeError("ParaView renderer source changed during execution")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "status": "ok",
        "execution_profile": "debug",
        "execution": {
            "visualization_only": True,
            "forward_executed": False,
            "inverse_executed": False,
            "adjoint_executed": False,
            "backward_executed": False,
            "pyvista_rendering_executed": False,
            "native_paraview_rendering_executed": True,
        },
        "paraview_version": paraview_version,
        "pvbatch": pvbatch_identity,
        "command": command,
        "wrapper": wrapper_after,
        "renderer_provenance": renderer_provenance,
        "inputs_pre": inputs_before,
        "inputs_post": inputs_after,
        "preparation": preparation,
        "branch_order": list(BRANCH_ORDER),
        "state_order": list(STATE_ORDER),
        "view_order": list(VIEW_ORDER),
        "mode_order": list(MODE_ORDER),
        "image_resolution": list(IMAGE_RESOLUTION),
        "normal_residual_shared_limit_mm": contract["normal_residual_shared_limit_mm"],
        "outputs": outputs,
        "authority": (
            "all geometry and target-normal-residual plates and all PVSM states were generated by ParaView 6.1.1; "
            "PyVista was used only for audited IsFace data preparation and strict readback"
        ),
    }
    _write_json_atomic(cfg.output_receipt, receipt)
    for path in sorted(path for path in BUNDLE_ROOT.rglob("*") if path.is_file()):
        cherries.log_output(path)
    logger.info("Wrote native ParaView receipt to %s", cfg.output_receipt)


if __name__ == "__main__":
    # DEBUG=1 in the reviewed command selects the non-committing debug profile.
    if os.environ.get("DEBUG") != "1":
        raise RuntimeError(
            "NO-GO: set DEBUG=1 before Cherries starts; default profile is forbidden"
        )
    cherries.main(main)
