# Design (Next)

## Info

- The overhead of some seemingly simple operations are not negligible in a fast loop. For example:
  - `jax.tree.flatten()` before each call to a JIT compiled function
  - Creating new `eqx.Module` instances using `dataclasses.replace()` or `eqx.tree_at()`
- [attrs](https://github.com/python-attrs/attrs) is much faster (~100x) than `eqx.Module` when creating new instances.

## References

- [Newton Physics](https://github.com/newton-physics/newton) - An open-source, GPU-accelerated physics simulation engine built upon NVIDIA Warp, specifically targeting roboticists and simulation researchers.
- [PNCG](https://github.com/Xingbaji/PNCG_IPC) - Preconditioned Nonlinear Conjugate Gradient Method for Real-time Interior-point Hyperelasticity
