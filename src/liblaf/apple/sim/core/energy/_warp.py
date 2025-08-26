import warp as wp

from liblaf.apple import struct


@struct.pytree
class EnergyWarp:
    def fun(self, u: wp.array, output: wp.array) -> None:
        """Compute energy.

        $$
        E
        $$

        Args:
            u: (P,) vec3f -- displacement
            output: (1,) float -- energy
        """
        raise NotImplementedError

    def jac(self, u: wp.array, output: wp.array) -> None:
        r"""Compute Jacobian.

        $$
        \vb{J} = \pdv{E}{\vb{u}}
        $$

        Args:
            u: (P,) vec3f -- displacement
            output: (P,) vec3f -- gradient
        """
        raise NotImplementedError

    def hess(self, u: wp.array, output: wp.array) -> None:
        r"""Compute Hessian.

        $$
        \vb{H} = \pdv[2]{E}{\vb{u}}
        $$

        Args:
            u: (P,) vec3f -- displacement
            output: NotImplemented
        """
        raise NotImplementedError

    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        r"""Compute diagonal of Hessian.

        $$
        \operatorname{diag}(\vb{H})
        $$

        Args:
            u: (P,) vec3f -- displacement
            output: (P,) vec3f -- diagonal of Hessian
        """
        raise NotImplementedError

    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        r"""Compute product of Hessian with a vector.

        $$
        \vb{H} \vb{p}
        $$

        Args:
            u: (P,) vec3f -- displacement
            p: (P,) vec3f -- arbitrary vector
            output: (P,) vec3f -- product of Hessian and vector
        """
        raise NotImplementedError

    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        r"""Compute quadratic form of Hessian with a vector.

        $$
        \vb{p}^T \vb{H} \vb{p}
        $$

        Args:
            u: (P,) vec3f -- displacement
            p: (P,) vec3f -- arbitrary vector
            output: (P,) vec3f -- product of Hessian and vector
        """
        raise NotImplementedError
