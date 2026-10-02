# Positive IPC derivative precision validation

The first synthetic validation failed only its energy/gradient finite
difference at `h=3e-5`; its HVP check was already accurate to `2.80e-7`.
The cause was precision mismatch in the validation process. `Collision.fun`
wraps IPCTK's Python-float energy with `torch.as_tensor`, which used the global
float32 default, while IPCTK's NumPy gradient and Hessian remained float64.

Validation 002 sets the Torch default to float64 explicitly and records both
rebuilt collision states and a frozen central collision stencil for six step
sizes from `3e-4` to `1e-6`. Rebuilt and frozen results agree to numerical
precision and retain ten contacts throughout. At `h=3e-5`, the gradient and
HVP relative errors are `5.43e-8` and `2.80e-7`; at the declared `h=1e-4`
gate they are `6.04e-7` and `3.11e-6`. Relative force imbalance is
`1.36e-16`.

```bash
CHERRIES_NAME='Positive standard IPC contact derivative precision validation' \
CHERRIES_TAGS='joint-inverse,contact,validation,cpu' \
uv run python src/76-validate-positive-ipc-contact.py \
  --output-dir data/positive-ipc-contact-validation-002
```

The receipt is
[`summary.json`](../data/positive-ipc-contact-validation-002/summary.json)
(SHA-256
`d59cf2de2ed4c93e865cfa4bcd45be73fa532bd32979e93f6b486ca55e20511d`).
The script SHA-256 is
`b767d2180ee507d1361ec092d3e277d99725a6cc59231e84cc11c223cc3428d2`.
The [Comet run](https://www.comet.com/liblaf/apple/72b0b1b009254fdaaf63b1902c1a0c4f)
completed normally. The production forward calls `configure_cuda`, which sets
the Torch default to float64, so this validation-only mismatch did not affect
the simple-forward energies.
