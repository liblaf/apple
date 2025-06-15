from typing import ClassVar

from ._region import Region


class SubRegion(Region):
    is_view: ClassVar[bool] = True
