# ruff: noqa: C901, EM101, EM102, TRY003
"""A reproducible within-muscle Raw6 activation-variation regularizer."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch

ROOT = Path(__file__).resolve().parents[6]
DEFAULT_FIXTURE = (
    ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
)
RAW6_FROBENIUS_WEIGHTS = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _rms(values: np.ndarray) -> float:
    value = float(np.sqrt(np.mean(np.square(values))))
    if not math.isfinite(value):
        raise FloatingPointError("non-finite RMS")
    return value


class ActivationRegularization:
    """Mean squared Ainv jump across same-MuscleId shared active-tet faces.

    The fixture has no ``MuscleRegionId`` cell array.  Its active cells carry
    ``MuscleId`` and a contiguous ``ActivationControlId``.  The constructor
    proves their one-to-one correspondence, then retains only shared faces
    whose endpoints have the same ``MuscleId``.  Therefore the prior does not
    couple different muscles, even where their tetrahedra touch.

    For an edge e=(i,j), Raw6 stores the symmetric Ainv-I components in the
    order (00, 11, 22, 01, 12, 02).  Since I cancels in a difference,

        R(q) = mean_e ||Ainv_i - Ainv_j||_F^2
             = mean_e sum_k w_k (q_i,k - q_j,k)^2,

    with w=(1,1,1,2,2,2).  Dividing by the edge count, rather than mesh size
    or physical conductance, makes R a dimensionless mean squared neighboring
    Ainv difference.  Constant Ainv fields on each connected component have
    zero loss; isolated active cells receive no penalty.
    """

    def __init__(self, fixture: Path = DEFAULT_FIXTURE) -> None:
        fixture = Path(fixture)
        if fixture.is_dir():
            fixture = fixture / "volume.vtu"
        self.fixture_path = fixture.resolve()
        mesh = pv.read(self.fixture_path)
        active = np.flatnonzero(
            np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
        )
        muscle_id = np.asarray(mesh.cell_data["MuscleId"], dtype=np.int64)
        control_id = np.asarray(mesh.cell_data["ActivationControlId"], dtype=np.int64)
        if len(active) != 288_235:
            raise ValueError(f"expected 288235 active cells, got {len(active)}")
        if np.any(muscle_id[active] < 0) or np.any(control_id[active] < 0):
            raise ValueError(
                "active cells require nonnegative MuscleId and ActivationControlId"
            )
        expected_controls = np.arange(
            len(np.unique(control_id[active])), dtype=np.int64
        )
        if not np.array_equal(np.unique(control_id[active]), expected_controls):
            raise ValueError("active ActivationControlId values must be contiguous")
        # This fixture defines controls by active MuscleId, one control per muscle.
        pairs = np.unique(np.c_[muscle_id[active], control_id[active]], axis=0)
        if len(pairs) != len(expected_controls) or len(np.unique(pairs[:, 0])) != len(
            pairs
        ):
            raise ValueError(
                "active MuscleId and ActivationControlId are not one-to-one"
            )
        region_muscles = np.asarray(
            mesh.field_data["ActivationRegionMuscleId"], dtype=np.int64
        )
        if not np.array_equal(region_muscles, pairs[np.argsort(pairs[:, 1]), 0]):
            raise ValueError(
                "ActivationRegionMuscleId does not match ActivationControlId"
            )

        tets = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)
        if np.any(tets[:, 0] != 4):
            raise ValueError("fixture is expected to contain only tetrahedra")
        active_tets = tets[active, 1:]
        pattern = np.array(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
        faces = np.sort(active_tets[:, pattern].reshape(-1, 3), axis=1)
        owners = np.repeat(np.arange(len(active), dtype=np.int64), 4)
        order = np.lexsort(faces.T[::-1])
        faces = faces[order]
        owners = owners[order]
        paired = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
        if np.any(np.diff(paired) == 1):
            raise ValueError("active tetrahedral shared-face graph is nonmanifold")
        all_i, all_j = owners[paired], owners[paired + 1]
        local_muscle = muscle_id[active]
        same = local_muscle[all_i] == local_muscle[all_j]
        self.i = all_i[same]
        self.j = all_j[same]
        if len(self.i) == 0:
            raise ValueError("same-MuscleId graph has no shared-face edges")
        if not np.all(local_muscle[self.i] == local_muscle[self.j]):
            raise AssertionError("cross-muscle edge escaped graph filter")
        self.active_cell_ids = active
        self.muscle_id = local_muscle
        self.control_id = control_id[active]
        self.region_muscles = region_muscles
        self.shared_active_faces = len(all_i)
        self.cross_muscle_shared_faces = int((~same).sum())
        self._index_cache: dict[
            tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _indices(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (q.device.type, q.device.index)
        result = self._index_cache.get(key)
        if result is None:
            result = (
                torch.as_tensor(self.i, dtype=torch.long, device=q.device),
                torch.as_tensor(self.j, dtype=torch.long, device=q.device),
            )
            self._index_cache[key] = result
        return result

    def loss(self, q: torch.Tensor) -> torch.Tensor:
        """Return R(q), preserving autograd and q's device/dtype."""
        if q.ndim != 2 or q.shape != (len(self.active_cell_ids), 6):
            raise ValueError(
                f"q has shape {tuple(q.shape)}; expected ({len(self.active_cell_ids)}, 6)"
            )
        if not q.is_floating_point():
            raise TypeError("q must have a floating dtype")
        i, j = self._indices(q)
        weights = torch.as_tensor(
            RAW6_FROBENIUS_WEIGHTS, dtype=q.dtype, device=q.device
        )
        return ((q[i] - q[j]).square() * weights).sum(dim=1).mean()

    def metrics(self, q_numpy: np.ndarray) -> dict[str, float]:
        """Return CPU-only scalar roughness diagnostics for a saved Raw6 field."""
        q = np.asarray(q_numpy, dtype=np.float64)
        if q.shape != (len(self.active_cell_ids), 6) or not np.isfinite(q).all():
            raise ValueError("q_numpy must be finite with shape (active_tetrahedra, 6)")
        delta = q[self.i] - q[self.j]
        weighted_squared = np.sum(delta**2 * np.asarray(RAW6_FROBENIUS_WEIGHTS), axis=1)
        # The identity part of Ainv cancels, so this is also Ainv jump squared.
        return {
            "regularizer_mean_neighbor_ainv_frobenius_squared": float(
                weighted_squared.mean()
            ),
            "regularizer_neighbor_ainv_frobenius_rms": float(
                np.sqrt(weighted_squared.mean())
            ),
            "regularizer_raw6_component_rms": _rms(delta),
            "regularizer_edge_count": float(len(self.i)),
        }

    def calibrate_lambda(
        self, q_numpy: np.ndarray, data_gradient: np.ndarray | torch.Tensor
    ) -> dict[str, float] | None:
        """Choose lambda giving the regularizer a 1% initial gradient RMS.

        Returns ``None`` when either gradient has no finite, nonzero RMS, so a
        caller cannot mistake a meaningless zero-gradient ratio for a scale.
        ``data_gradient`` must be the data-objective gradient in the same Raw6
        coordinates and units as q.
        """
        q = np.asarray(q_numpy, dtype=np.float64)
        data = np.asarray(
            data_gradient.detach().cpu().numpy()
            if isinstance(data_gradient, torch.Tensor)
            else data_gradient,
            dtype=np.float64,
        )
        if q.shape != data.shape or q.shape != (len(self.active_cell_ids), 6):
            raise ValueError(
                "q_numpy and data_gradient must share the active Raw6 shape"
            )
        if not np.isfinite(q).all() or not np.isfinite(data).all():
            return None
        q_t = torch.tensor(q, dtype=torch.float64, device="cpu", requires_grad=True)
        regularizer = self.loss(q_t)
        (gradient,) = torch.autograd.grad(regularizer, q_t)
        regularizer_rms = _rms(gradient.detach().numpy())
        data_rms = _rms(data)
        if regularizer_rms == 0.0 or data_rms == 0.0:
            return None
        value = 0.01 * data_rms / regularizer_rms
        if not math.isfinite(value) or value <= 0.0:
            return None
        return {
            "lambda": float(value),
            "target_regularizer_gradient_fraction": 0.01,
            "data_gradient_rms": data_rms,
            "regularizer_gradient_rms_at_lambda_1": regularizer_rms,
            "scaled_regularizer_gradient_rms": value * regularizer_rms,
        }

    @property
    def provenance(self) -> dict[str, Any]:
        """JSON-compatible graph receipts and the precise discretization."""
        return {
            "fixture": {
                "path": str(self.fixture_path),
                "sha256": _sha256(self.fixture_path),
            },
            "source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__)),
            },
            "support": {
                "active_cell_count": len(self.active_cell_ids),
                "active_cell_ids_sha256": _array_sha256(self.active_cell_ids),
                "label_field": "MuscleId",
                "MuscleRegionId_present": False,
                "activation_control_field": "ActivationControlId",
                "active_muscle_count": len(np.unique(self.muscle_id)),
                "active_control_count": len(np.unique(self.control_id)),
                "MuscleId_sha256": _array_sha256(self.muscle_id),
                "ActivationControlId_sha256": _array_sha256(self.control_id),
                "ActivationRegionMuscleId_sha256": _array_sha256(self.region_muscles),
            },
            "graph": {
                "construction": "unique active tetrahedron shared faces, retained only when endpoint MuscleId values match",
                "shared_active_face_count_before_muscle_filter": self.shared_active_faces,
                "within_same_MuscleId_shared_face_edge_count": len(self.i),
                "cross_MuscleId_shared_face_count_excluded": self.cross_muscle_shared_faces,
                "edge_i_sha256": _array_sha256(self.i),
                "edge_j_sha256": _array_sha256(self.j),
            },
            "loss": {
                "name": "mean squared neighboring Ainv Frobenius difference",
                "raw6_order": ["00", "11", "22", "01", "12", "02"],
                "raw6_frobenius_weights": list(RAW6_FROBENIUS_WEIGHTS),
                "normalization": "sum over the six weighted squared differences, then arithmetic mean over retained edges",
                "units": "dimensionless",
                "null_space": "constant Ainv per graph connected component; isolated active cells are unpenalized",
            },
        }
