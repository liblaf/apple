# ruff: noqa: EM101, EM102, SLF001, TRY003
"""Minimal ctypes binding for a single-GPU cuDSS direct solve.

The constants and signatures here are transcribed from the staged NVIDIA cuDSS
0.7 public header distributed by ``nvidia-cudss-cu13==0.7.0.20``. The wrapper
deliberately leaves timing and residual acceptance to its caller.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import torch

# cudss_data_types.h, with cudaDataType_t values from CUDA library_types.h.
CUDSS_R_64F = 1
CUDSS_R_32I = 10
CUDSS_R_64I = 24
CUDSS_CONFIG_PIVOT_EPSILON = 10
CUDSS_DATA_INFO = 0
CUDSS_DATA_LU_NNZ = 1
CUDSS_DATA_NPIVOTS = 2
CUDSS_DATA_INERTIA = 3
CUDSS_DATA_MEMORY_ESTIMATES = 12
CUDSS_PHASE_ANALYSIS = 3
CUDSS_PHASE_FACTORIZATION = 4
CUDSS_PHASE_REFACTORIZATION = 8
CUDSS_PHASE_SOLVE = 1008
CUDSS_MTYPE_SYMMETRIC = 1
CUDSS_MTYPE_SPD = 3
CUDSS_MVIEW_LOWER = 1
CUDSS_BASE_ZERO = 0
CUDSS_LAYOUT_COL_MAJOR = 0
CUDSS_STATUS_SUCCESS = 0

_VOID = ctypes.c_void_p
_SIZE = ctypes.c_size_t
_I32 = ctypes.c_int
_I64 = ctypes.c_int64


class CudssError(RuntimeError):
    """A synchronous cuDSS API status failure."""

    def __init__(self, function: str, status: int) -> None:
        super().__init__(f"{function} returned cuDSS status {status}")
        self.function = function
        self.status = status


def _library_path() -> str:
    explicit = os.environ.get("CUDSS_LIBRARY")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    found = ctypes.util.find_library("cudss")
    if found:
        return found
    candidates = []
    for root in map(Path, sys.path):
        candidates.extend((root / "nvidia" / "cu13" / "lib").glob("libcudss.so*"))
        candidates.extend((root / "nvidia" / "cudss" / "lib").glob("libcudss.so*"))
    if candidates:
        return str(min(candidates))
    message = "cuDSS not found; set CUDSS_LIBRARY to libcudss.so.0"
    raise FileNotFoundError(message)


def _pointer(value: torch.Tensor) -> _VOID:
    return _VOID(value.data_ptr())


def _index_type(value: torch.Tensor) -> int:
    if value.dtype == torch.int32:
        return CUDSS_R_32I
    if value.dtype == torch.int64:
        return CUDSS_R_64I
    message = f"cuDSS CSR indices must be int32 or int64, got {value.dtype}"
    raise TypeError(message)


def _load() -> ctypes.CDLL:
    library = ctypes.CDLL(_library_path())
    library.cudssCreate.argtypes = [ctypes.POINTER(_VOID)]
    library.cudssCreate.restype = _I32
    library.cudssDestroy.argtypes = [_VOID]
    library.cudssDestroy.restype = _I32
    library.cudssGetProperty.argtypes = [_I32, ctypes.POINTER(_I32)]
    library.cudssGetProperty.restype = _I32
    library.cudssConfigCreate.argtypes = [ctypes.POINTER(_VOID)]
    library.cudssConfigCreate.restype = _I32
    library.cudssConfigDestroy.argtypes = [_VOID]
    library.cudssConfigDestroy.restype = _I32
    library.cudssConfigSet.argtypes = [_VOID, _I32, _VOID, _SIZE]
    library.cudssConfigSet.restype = _I32
    library.cudssDataCreate.argtypes = [_VOID, ctypes.POINTER(_VOID)]
    library.cudssDataCreate.restype = _I32
    library.cudssDataDestroy.argtypes = [_VOID, _VOID]
    library.cudssDataDestroy.restype = _I32
    library.cudssDataGet.argtypes = [
        _VOID,
        _VOID,
        _I32,
        _VOID,
        _SIZE,
        ctypes.POINTER(_SIZE),
    ]
    library.cudssDataGet.restype = _I32
    library.cudssSetStream.argtypes = [_VOID, _VOID]
    library.cudssSetStream.restype = _I32
    library.cudssMatrixCreateDn.argtypes = [
        ctypes.POINTER(_VOID),
        _I64,
        _I64,
        _I64,
        _VOID,
        _I32,
        _I32,
    ]
    library.cudssMatrixCreateDn.restype = _I32
    library.cudssMatrixDestroy.argtypes = [_VOID]
    library.cudssMatrixDestroy.restype = _I32
    library.cudssMatrixSetValues.argtypes = [_VOID, _VOID]
    library.cudssMatrixSetValues.restype = _I32
    library.cudssExecute.argtypes = [_VOID, _I32, _VOID, _VOID, _VOID, _VOID, _VOID]
    library.cudssExecute.restype = _I32
    return library


def _version(library: ctypes.CDLL) -> tuple[int, int, int]:
    values = []
    for property_type in range(3):  # CUDA libraryPropertyType: major, minor, patch.
        value = _I32()
        status = library.cudssGetProperty(property_type, ctypes.byref(value))
        if status != CUDSS_STATUS_SUCCESS:
            raise CudssError("cudssGetProperty", int(status))
        values.append(int(value.value))
    return tuple(values)  # type: ignore[return-value]


def _bind_csr_create(library: ctypes.CDLL) -> None:
    common = [ctypes.POINTER(_VOID), _I64, _I64, _I64, _VOID, _VOID, _VOID, _VOID]
    library.cudssMatrixCreateCsr.argtypes = [*common, _I32, _I32, _I32, _I32, _I32]
    library.cudssMatrixCreateCsr.restype = _I32


@dataclass(slots=True)
class _Dense:
    tensor: torch.Tensor
    handle: _VOID


class CudssDirect:
    """cuDSS factorization for a lower-triangle CUDA CSR FP64 matrix.

    ``mode='spd'`` uses Cholesky and reports a nonzero ``info`` for a detected
    non-positive minor. ``mode='symmetric'`` uses symmetric-indefinite LDLT and
    additionally reports inertia.  Neither mode modifies matrix values: pivot
    epsilon is explicitly set to zero.
    """

    def __init__(
        self, matrix_lower: torch.Tensor, mode: Literal["spd", "symmetric"] = "spd"
    ) -> None:
        self.matrix = self._validate(matrix_lower)
        self.mode = mode
        self._lib = _load()
        self._version = _version(self._lib)
        if self._version[:2] != (0, 7):
            message = f"this experiment wrapper supports cuDSS 0.7 only, found {self._version}"
            raise RuntimeError(message)
        _bind_csr_create(self._lib)
        self._handle = _VOID()
        self._config = _VOID()
        self._data = _VOID()
        self._matrix = _VOID()
        self._factor_dense: _Dense | None = None
        self._closed = False
        self._analyzed = False
        self._factorized = False
        try:
            self._check(
                "cudssCreate", self._lib.cudssCreate(ctypes.byref(self._handle))
            )
            self._set_current_stream()
            self._check(
                "cudssConfigCreate",
                self._lib.cudssConfigCreate(ctypes.byref(self._config)),
            )
            pivot_epsilon = ctypes.c_double(0.0)
            self._check(
                "cudssConfigSet(CUDSS_CONFIG_PIVOT_EPSILON)",
                self._lib.cudssConfigSet(
                    self._config,
                    CUDSS_CONFIG_PIVOT_EPSILON,
                    ctypes.byref(pivot_epsilon),
                    ctypes.sizeof(pivot_epsilon),
                ),
            )
            self._check(
                "cudssDataCreate",
                self._lib.cudssDataCreate(self._handle, ctypes.byref(self._data)),
            )
            crow, col, values = (
                self.matrix.crow_indices(),
                self.matrix.col_indices(),
                self.matrix.values(),
            )
            csr_args = [
                ctypes.byref(self._matrix),
                self.matrix.shape[0],
                self.matrix.shape[1],
                self.matrix._nnz(),
                _pointer(crow),
                None,
                _pointer(col),
                _pointer(values),
                _index_type(crow),
                CUDSS_R_64F,
                CUDSS_MTYPE_SPD if mode == "spd" else CUDSS_MTYPE_SYMMETRIC,
                CUDSS_MVIEW_LOWER,
                CUDSS_BASE_ZERO,
            ]
            self._check(
                "cudssMatrixCreateCsr", self._lib.cudssMatrixCreateCsr(*csr_args)
            )
            probe = torch.empty(
                (self.matrix.shape[0],), dtype=torch.float64, device=self.matrix.device
            )
            self._factor_dense = self._make_dense(probe)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _validate(matrix: torch.Tensor) -> torch.Tensor:
        if matrix.layout != torch.sparse_csr:
            raise TypeError("matrix_lower must be a torch.sparse_csr_tensor")
        if not matrix.is_cuda or matrix.dtype != torch.float64:
            raise TypeError("matrix_lower must be CUDA FP64")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix_lower must be square")
        if (
            not matrix.values().is_contiguous()
            or not matrix.crow_indices().is_contiguous()
            or not matrix.col_indices().is_contiguous()
        ):
            raise ValueError("CSR values and indices must be contiguous")
        _index_type(matrix.crow_indices())
        _index_type(matrix.col_indices())
        return matrix

    def _check(self, function: str, status: int) -> None:
        if status != CUDSS_STATUS_SUCCESS:
            raise CudssError(function, status)

    def _set_current_stream(self) -> None:
        stream = torch.cuda.current_stream(self.matrix.device)
        self._check(
            "cudssSetStream",
            self._lib.cudssSetStream(self._handle, _VOID(stream.cuda_stream)),
        )

    def _data_get(self, param: int, value: Any) -> tuple[int, int]:
        size = _SIZE()
        status = int(
            self._lib.cudssDataGet(
                self._handle,
                self._data,
                param,
                ctypes.byref(value),
                ctypes.sizeof(value),
                ctypes.byref(size),
            )
        )
        return status, int(size.value)

    def _make_dense(self, tensor: torch.Tensor) -> _Dense:
        handle = _VOID()
        self._check(
            "cudssMatrixCreateDn",
            self._lib.cudssMatrixCreateDn(
                ctypes.byref(handle),
                tensor.numel(),
                1,
                tensor.numel(),
                _pointer(tensor),
                CUDSS_R_64F,
                CUDSS_LAYOUT_COL_MAJOR,
            ),
        )
        return _Dense(tensor, handle)

    def analyze(self) -> dict[str, Any]:
        self._require_open()
        self._set_current_stream()
        assert self._factor_dense is not None
        self._check(
            "cudssExecute(analysis)",
            self._lib.cudssExecute(
                self._handle,
                CUDSS_PHASE_ANALYSIS,
                self._config,
                self._data,
                self._matrix,
                self._factor_dense.handle,
                self._factor_dense.handle,
            ),
        )
        torch.cuda.current_stream(self.matrix.device).synchronize()
        estimates = (_I64 * 16)()
        memory_status, memory_size = self._data_get(
            CUDSS_DATA_MEMORY_ESTIMATES, estimates
        )
        self._analyzed = True
        return {
            "memory_estimates": list(estimates),
            "memory_estimates_status": memory_status,
            "memory_estimates_size": memory_size,
            "config": self.config_metadata,
        }

    @property
    def config_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "matrix_type": "CUDSS_MTYPE_SPD"
            if self.mode == "spd"
            else "CUDSS_MTYPE_SYMMETRIC",
            "matrix_view": "CUDSS_MVIEW_LOWER",
            "index_base": "CUDSS_BASE_ZERO",
            "pivot_epsilon": 0.0,
            "pivot_perturbation": "disabled explicitly; cuDSS never receives altered matrix values",
            "library": _library_path(),
            "api_version": self._version,
        }

    def factorize(self, *, refactor: bool = False) -> dict[str, Any]:
        self._require_open()
        if not self._analyzed:
            raise RuntimeError("call analyze() before factorize()")
        self._set_current_stream()
        phase = CUDSS_PHASE_REFACTORIZATION if refactor else CUDSS_PHASE_FACTORIZATION
        assert self._factor_dense is not None
        self._check(
            "cudssExecute(refactorization)"
            if refactor
            else "cudssExecute(factorization)",
            self._lib.cudssExecute(
                self._handle,
                phase,
                self._config,
                self._data,
                self._matrix,
                self._factor_dense.handle,
                self._factor_dense.handle,
            ),
        )
        torch.cuda.current_stream(self.matrix.device).synchronize()
        info = _I32()
        info_status, _ = self._data_get(CUDSS_DATA_INFO, info)
        nnz = _I64()
        nnz_status, _ = self._data_get(CUDSS_DATA_LU_NNZ, nnz)
        index = _I32 if self.matrix.crow_indices().dtype == torch.int32 else _I64
        pivots = index()
        pivots_status, _ = self._data_get(CUDSS_DATA_NPIVOTS, pivots)
        inertia = (index * 2)()
        inertia_status, _ = self._data_get(CUDSS_DATA_INERTIA, inertia)
        self._factorized = info.value == 0
        return {
            "phase": "refactorization" if refactor else "factorization",
            "info": int(info.value),
            "info_status": info_status,
            "factorized": self._factorized,
            "lu_nnz": int(nnz.value),
            "lu_nnz_status": nnz_status,
            "pivots": int(pivots.value),
            "pivots_status": pivots_status,
            "inertia": list(inertia)
            if self.mode == "symmetric" and inertia_status == CUDSS_STATUS_SUCCESS
            else None,
            "inertia_status": inertia_status,
        }

    def update_values(self, values: torch.Tensor) -> None:
        self._require_open()
        if (
            values.device != self.matrix.device
            or values.dtype != torch.float64
            or not values.is_contiguous()
            or values.shape != self.matrix.values().shape
        ):
            raise ValueError(
                "values must be contiguous FP64 on the matrix device with identical CSR value shape"
            )
        self._check(
            "cudssMatrixSetValues",
            self._lib.cudssMatrixSetValues(self._matrix, _pointer(values)),
        )
        self.matrix = torch.sparse_csr_tensor(
            self.matrix.crow_indices(),
            self.matrix.col_indices(),
            values,
            size=self.matrix.shape,
            device=values.device,
        )
        self._factorized = False

    def refactorize(self, values: torch.Tensor | None = None) -> dict[str, Any]:
        if values is not None:
            self.update_values(values)
        return self.factorize(refactor=True)

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        self._require_open()
        if not self._factorized:
            raise RuntimeError(
                "factorization has nonzero CUDSS_DATA_INFO; solve is rejected"
            )
        if (
            rhs.ndim != 1
            or rhs.shape[0] != self.matrix.shape[0]
            or rhs.device != self.matrix.device
            or rhs.dtype != torch.float64
            or not rhs.is_contiguous()
        ):
            raise ValueError(
                "rhs must be contiguous CUDA FP64 vector with matrix dimension"
            )
        self._set_current_stream()
        solution = torch.empty_like(rhs)
        x, b = self._make_dense(solution), self._make_dense(rhs)
        try:
            self._check(
                "cudssExecute(solve)",
                self._lib.cudssExecute(
                    self._handle,
                    CUDSS_PHASE_SOLVE,
                    self._config,
                    self._data,
                    self._matrix,
                    x.handle,
                    b.handle,
                ),
            )
            torch.cuda.current_stream(self.matrix.device).synchronize()
        finally:
            self._check(
                "cudssMatrixDestroy(solution)", self._lib.cudssMatrixDestroy(x.handle)
            )
            self._check(
                "cudssMatrixDestroy(rhs)", self._lib.cudssMatrixDestroy(b.handle)
            )
        info = _I32()
        info_status, _ = self._data_get(CUDSS_DATA_INFO, info)
        if info_status != CUDSS_STATUS_SUCCESS or info.value != 0:
            raise RuntimeError(
                f"cuDSS solve device info={info.value}, status={info_status}"
            )
        return solution

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("CudssDirect is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for function, args, active in (
            (
                "cudssMatrixDestroy",
                (self._factor_dense.handle,) if self._factor_dense else (),
                self._factor_dense is not None,
            ),
            ("cudssMatrixDestroy", (self._matrix,), bool(self._matrix.value)),
            (
                "cudssDataDestroy",
                (self._handle, self._data),
                bool(self._data.value and self._handle.value),
            ),
            ("cudssConfigDestroy", (self._config,), bool(self._config.value)),
            ("cudssDestroy", (self._handle,), bool(self._handle.value)),
        ):
            if active:
                status = getattr(self._lib, function)(*args)
                if status != CUDSS_STATUS_SUCCESS:
                    # Destruction must not hide an earlier primary exception.
                    self._close_status = {"function": function, "status": int(status)}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


__all__ = ["CudssDirect", "CudssError"]
