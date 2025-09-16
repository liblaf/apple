from typing import Protocol

import attrs
import warp.codegen
import warp.types as wpt


class ParamsFactory(Protocol):
    def __call__(
        self, floating: type[wpt.float_base], /
    ) -> tuple[warp.codegen.Struct, warp.codegen.Struct]: ...


@attrs.define
class ParamsStructs:
    Params32: warp.codegen.Struct
    Params64: warp.codegen.Struct
    ParamsElem32: warp.codegen.Struct
    ParamsElem64: warp.codegen.Struct


def define_params(factory: ParamsFactory) -> ParamsStructs:
    Params32: warp.codegen.Struct
    ParamsElem32: warp.codegen.Struct
    Params32, ParamsElem32 = factory(wpt.float32)
    Params64: warp.codegen.Struct
    ParamsElem64: warp.codegen.Struct
    Params64, ParamsElem64 = factory(wpt.float64)
    return ParamsStructs(
        Params32=Params32,
        Params64=Params64,
        ParamsElem32=ParamsElem32,
        ParamsElem64=ParamsElem64,
    )
