import warp as wp
from _typeshed import Incomplete

@wp.kernel
def sgd_step_kernel(g: None, b: None, lr: float, weight_decay: float, momentum: float, damping: float, nesterov: int, t: int, params: None): ...

class SGD:
    b: Incomplete
    lr: Incomplete
    momentum: Incomplete
    dampening: Incomplete
    weight_decay: Incomplete
    nesterov: Incomplete
    t: int
    def __init__(self, params=None, lr: float = 0.001, momentum: float = 0.0, dampening: float = 0.0, weight_decay: float = 0.0, nesterov: bool = False) -> None: ...
    params: Incomplete
    def set_params(self, params) -> None: ...
    def reset_internal_state(self) -> None: ...
    def step(self, grad) -> None: ...
    @staticmethod
    def step_detail(g, b, lr, momentum, dampening, weight_decay, nesterov, t, params) -> None: ...
