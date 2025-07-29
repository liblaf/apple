from .field import DiscreteField as DiscreteField
from _typeshed import Incomplete
from warp.fem.space import SpaceRestriction as SpaceRestriction

class FieldRestriction:
    space_restriction: Incomplete
    domain: Incomplete
    field: Incomplete
    space: Incomplete
    def __init__(self, space_restriction: SpaceRestriction, field: DiscreteField) -> None: ...
