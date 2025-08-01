from .cube_shape_function import CubeNedelecFirstKindShapeFunctions as CubeNedelecFirstKindShapeFunctions, CubeNonConformingPolynomialShapeFunctions as CubeNonConformingPolynomialShapeFunctions, CubeRaviartThomasShapeFunctions as CubeRaviartThomasShapeFunctions, CubeSerendipityShapeFunctions as CubeSerendipityShapeFunctions, CubeShapeFunction as CubeShapeFunction, CubeTripolynomialShapeFunctions as CubeTripolynomialShapeFunctions
from .shape_function import ConstantShapeFunction as ConstantShapeFunction, ShapeFunction as ShapeFunction
from .square_shape_function import SquareBipolynomialShapeFunctions as SquareBipolynomialShapeFunctions, SquareNedelecFirstKindShapeFunctions as SquareNedelecFirstKindShapeFunctions, SquareNonConformingPolynomialShapeFunctions as SquareNonConformingPolynomialShapeFunctions, SquareRaviartThomasShapeFunctions as SquareRaviartThomasShapeFunctions, SquareSerendipityShapeFunctions as SquareSerendipityShapeFunctions, SquareShapeFunction as SquareShapeFunction
from .tet_shape_function import TetrahedronNedelecFirstKindShapeFunctions as TetrahedronNedelecFirstKindShapeFunctions, TetrahedronNonConformingPolynomialShapeFunctions as TetrahedronNonConformingPolynomialShapeFunctions, TetrahedronPolynomialShapeFunctions as TetrahedronPolynomialShapeFunctions, TetrahedronRaviartThomasShapeFunctions as TetrahedronRaviartThomasShapeFunctions, TetrahedronShapeFunction as TetrahedronShapeFunction
from .triangle_shape_function import TriangleNedelecFirstKindShapeFunctions as TriangleNedelecFirstKindShapeFunctions, TriangleNonConformingPolynomialShapeFunctions as TriangleNonConformingPolynomialShapeFunctions, TrianglePolynomialShapeFunctions as TrianglePolynomialShapeFunctions, TriangleRaviartThomasShapeFunctions as TriangleRaviartThomasShapeFunctions, TriangleShapeFunction as TriangleShapeFunction
from enum import Enum
from warp.fem.polynomial import Polynomial as Polynomial

class ElementBasis(Enum):
    LAGRANGE = 'P'
    SERENDIPITY = 'S'
    NONCONFORMING_POLYNOMIAL = 'dP'
    NEDELEC_FIRST_KIND = 'N1'
    RAVIART_THOMAS = 'RT'

def get_shape_function(element_class: type, space_dimension: int, degree: int, element_basis: ElementBasis, family: Polynomial | None = None): ...
