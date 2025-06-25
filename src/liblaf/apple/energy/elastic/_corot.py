from .elastic import Elastic


class CoRot(Elastic):
    r"""Co-rotational.

    $$
    \begin{split}
        \Psi_{\text{CoRot}}
        & = \frac{\mu}{2} \|F - R\|_F^2 + \frac{\lambda}{2} \operatorname{tr}^2(S - I) \\
        & = \frac{\mu}{2} \|F - R\|_F^2 + \frac{\lambda}{2} (I_1^2 - 6 I_1 + 9)
    \end{split}
    $$
    """
