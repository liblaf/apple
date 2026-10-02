import attrs
import torch
from jaxtyping import Float
from torch import Tensor

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


@attrs.define
class DirectionUpdate:
    """Dai-Kou nonlinear conjugate-gradient direction update."""

    def __call__(
        self,
        g: Vector,
        g_prev: Vector,
        P: Vector,
        p_prev: Vector,
        *,
        restart: bool = False,
    ) -> Vector:
        """Compute a descent direction, restarting when requested."""
        if restart:
            return -P * g
        beta: Scalar = dai_kou_plus(g=g, g_prev=g_prev, P=P, p_prev=p_prev)
        Pg: Vector = -P * g
        p: Vector = Pg + beta * p_prev
        return torch.where(torch.dot(p, g) < 0.0, p, Pg)


def dai_kou(g: Vector, g_prev: Vector, P: Vector, p_prev: Vector) -> Scalar:
    """Compute the Dai-Kou conjugacy coefficient."""
    y: Vector = g - g_prev
    Py: Vector = P * y
    yTp: Scalar = torch.dot(y, p_prev)
    beta: Scalar = (
        torch.dot(g, Py) - torch.dot(y, Py) * torch.dot(p_prev, g) / yTp
    ) / yTp
    return beta


def dai_kou_plus(g: Vector, g_prev: Vector, P: Vector, p_prev: Vector) -> Scalar:
    """Compute the safeguarded nonnegative Dai-Kou coefficient."""
    beta: Scalar = dai_kou(g=g, g_prev=g_prev, P=P, p_prev=p_prev)
    beta: Scalar = torch.maximum(beta, torch.zeros_like(beta))
    beta: Scalar = torch.where(beta > 10.0, 0.0, beta)
    return beta
