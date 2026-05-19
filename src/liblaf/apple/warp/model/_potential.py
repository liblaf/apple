import functools
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, Self

import attrs
import pyvista as pv
import torch
import warp as wp
from torch import Tensor

from liblaf.apple.common import DEFAULT_POTENTIAL_NAME
from liblaf.apple.torch.fem import Region
from liblaf.apple.warp.utils import warp_default_dtype

from ._material import (
    ArrayAnnotation,
    MaterialField,
    MaterialVar,
    Struct,
    StructInstance,
    make_struct,
)


@attrs.define
class WarpPotential:
    class Materials(Protocol):
        pass

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {}
    materials: Materials = attrs.field(default=None, kw_only=True)
    name: str = attrs.field(default=DEFAULT_POTENTIAL_NAME, kw_only=True)

    @functools.cached_property
    def material_vars(self) -> Mapping[str, MaterialVar]:
        dtype: Any = warp_default_dtype()
        return {name: field.make(dtype) for name, field in self.MATERIAL_FIELDS.items()}

    @functools.cached_property
    def material_struct(self) -> Struct:
        cls: type = type(self)
        return make_struct(
            tuple(self.material_vars.values()),
            module=cls.__module__,
            qualname=cls.__qualname__,
        )

    def material_from_region(
        self, region: Region, requires_grad: Sequence[str] = ()
    ) -> Any:
        struct: Struct = self.material_struct
        materials: StructInstance = struct()
        for field in self.material_vars.values():
            value: wp.array = field.from_region(region)
            setattr(materials, field.name, value)
        for name in requires_grad:
            arr: wp.array = getattr(materials, name)
            arr.requires_grad = True
        return materials

    @classmethod
    def from_region(
        cls, region: Region, requires_grad: Sequence[str] = (), **kwargs
    ) -> Self:
        self: Self = cls(**kwargs)
        self.materials = self.material_from_region(region, requires_grad=requires_grad)
        return self

    @classmethod
    def from_pyvista(
        cls, obj: pv.DataObject, requires_grad: Sequence[str] = (), **kwargs
    ) -> Self:
        region: Region = Region.from_pyvista(obj)
        return cls.from_region(region, requires_grad=requires_grad, **kwargs)

    def get_materials(self) -> dict[str, wp.array]:
        return {name: getattr(self.materials, name) for name in self.MATERIAL_FIELDS}

    def set_materials(self, materials: Mapping[str, wp.array | Tensor]) -> None:
        for name, value in materials.items():
            if torch.is_tensor(value):
                annotation: ArrayAnnotation = self.material_vars[name].annotation
                value: wp.array = wp.from_torch(value, annotation.dtype)  # noqa: PLW2901
            setattr(self.materials, name, value)

    def require_grad(self, materials: Mapping[str, bool]) -> None:
        for name, requires_grad in materials.items():
            arr: wp.array = getattr(self.materials, name)
            arr.requires_grad = requires_grad

    def fun(self, u: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def grad(self, u: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        raise NotImplementedError
