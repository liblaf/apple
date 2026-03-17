import numpy as np
import warp as wp

wp.init()


@wp.func
def svd_rv(F: wp.mat33):
    U, sigma, V = wp.svd3(F)
    return U @ wp.transpose(V)


@wp.kernel
def test_grad(A: wp.array(dtype=wp.mat33), out: wp.array(dtype=float)):
    R = svd_rv(A[0])
    out[0] = wp.trace(R)


A = wp.array(
    np.array([[[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]]], dtype=np.float32),
    requires_grad=True,
)
out = wp.array(np.zeros(1, dtype=np.float32), requires_grad=True)

with wp.Tape() as tape:
    wp.launch(test_grad, dim=1, inputs=[A, out], outputs=[])

tape.backward()
print("Grad:", A.grad.numpy())
