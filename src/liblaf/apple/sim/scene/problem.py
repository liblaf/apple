from collections.abc import Callable

from liblaf.apple import optim, struct

from .protocol import SceneProtocol, X, Y


class SceneProblem(struct.PyTreeMutable):
    scene: SceneProtocol = struct.field()
    _callback: Callable | None = struct.field(default=None)

    def callback(self, intermediate_result: optim.OptimizeResult) -> None:
        result: optim.OptimizeResult = intermediate_result
        x: X = result["x"]
        self.scene = self.scene.pre_optim_iter(x)
        if self._callback is not None:
            self._callback(result, self.scene)

    def fun(self, x: X, /) -> Y:
        return self.scene.fun(x)

    def jac(self, x: X, /) -> X:
        return self.scene.jac(x)

    def hessp(self, x: X, p: X, /) -> X:
        return self.scene.hessp(x, p)

    def hess_diag(self, x: X, /) -> X:
        return self.scene.hess_diag(x)

    def hess_quad(self, x: X, p: X, /) -> Y:
        return self.scene.hess_quad(x, p)

    def fun_and_jac(self, x: X, /) -> tuple[Y, X]:
        return self.scene.fun_and_jac(x)

    def jac_and_hess_diag(self, x: X, /) -> tuple[X, X]:
        return self.scene.jac_and_hess_diag(x)
