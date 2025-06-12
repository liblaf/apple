from ._element import Element


class ElementTriangle(Element):
    @property
    def n_points(self) -> int:
        return 3
