# ruff: noqa: A001, A002, PYI001, UP047

from typing import TypeVar, overload

from .types import Matrix, MeshQueryPoint, Vector, float32, int32, uint64, vec3f

Cols = TypeVar("Cols", bound=int)
Length = TypeVar("Length", bound=int)
Rows = TypeVar("Rows", bound=int)
Scalar = TypeVar("Scalar", bound=int | float)
Float = TypeVar("Float", bound=float)

def tid() -> int: ...
@overload
def cw_mul[V: Vector](a: V, b: V, /) -> V: ...
@overload
def cw_mul[M: Matrix](a: M, b: M, /) -> M: ...
def ddot(a: Matrix[Rows, Cols, Scalar], b: Matrix[Rows, Cols, Scalar], /) -> Scalar: ...
def determinant(a: Matrix[Rows, Rows, Float], /) -> Float: ...
def dot(a: Vector[Length, Scalar], b: Vector[Length, Scalar], /) -> Scalar: ...
def length_sq(a: Vector[Length, Scalar], /) -> Scalar: ...
def length(a: Vector[Length, Float], /) -> Float: ...
def normalize[V: Vector](a: V, /) -> V: ...
def pow(x: Float, y: Float, /) -> Float: ...
@overload
def sign(x: Scalar, /) -> Scalar: ...
@overload
def sign(x: Vector[Length, Scalar], /) -> Vector[Length, Scalar]: ...
def trace(a: Matrix[Rows, Cols, Scalar], /) -> Scalar: ...

# region Geometry

def mesh_eval_face_normal(id: uint64, face: int32, /) -> vec3f: ...
def mesh_eval_position(
    id: uint64, face: int32, bary_u: float32, bary_v: float32, /
) -> vec3f: ...
def mesh_query_point_sign_normal(
    id: uint64, point: vec3f, max_dist: float32, epsilon: float32 = ...
) -> MeshQueryPoint: ...

# endregion Geometry

__all__ = [
    "cw_mul",
    "ddot",
    "determinant",
    "dot",
    "length",
    "length_sq",
    "mesh_eval_face_normal",
    "mesh_eval_position",
    "mesh_query_point_sign_normal",
    "normalize",
    "pow",
    "sign",
    "tid",
    "trace",
]
