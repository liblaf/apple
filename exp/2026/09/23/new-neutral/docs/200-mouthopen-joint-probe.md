# Frozen MouthOpen joint-tangent probe

`src/200-diagnose-mouthopen-joint-probe.py` rebuilt the corrected IsFixed
physics and all-fixed-tetrahedron exclusion, then used the saved iteration-61
state from `inverse-mouthopen-coupled-017`. It bound the run checkpoint,
endpoint, final summary, progress row, independent audit, rejected projection
cache, and all five cached determinant gradients by SHA-256. The checkpoint
and endpoint matched the cached activation, normalized pose, and displacement
exactly. The independently audited endpoint was valid before the diagnostic.

It ran from the experiment directory with:

```bash
CHERRIES_NAME=mouthopen-joint-probe-epsilon-001 \
CHERRIES_TAGS=new-neutral,mouthopen,diagnostic,frozen-state,derivative \
TMPDIR=tmp/200-mouthopen-joint-probe-runtime \
.venv/bin/python -u \
  src/200-diagnose-mouthopen-joint-probe.py
```

The original acceptance rule stayed `abs(adjoint - probe) <= 1e-7 +
0.005 * max(abs(adjoint), abs(probe))`. The probe held the old collision state
and shifted operator fixed, finite-differencing only the material force and
prescribed jaw boundary. It therefore does not test an IPC active-set change.
No optimizer state or candidate was adopted.

At the rejected original tetrahedron 18514, the archived directional derivative
is the cancellation `-1.6667079449e-3 + 1.7878172099e-3 =
1.2110926494e-4` from strain and jaw terms. The original `1e-7` shifted linear
solves fail the unchanged comparison at every tested signed epsilon:

| Epsilon | Difference | Limit |
| ---: | ---: | ---: |
| +5e-5 | -3.3798180e-6 | 7.2244541e-7 |
| +1e-5 | -3.8271346e-6 | 7.2468200e-7 |
| +1e-6 | -2.9652547e-6 | 7.2037260e-7 |
| -5e-5 | -2.5275839e-6 | 7.1818424e-7 |
| -1e-5 | -2.5454478e-6 | 7.1827356e-7 |
| -1e-6 | -3.5578790e-6 | 7.2333572e-7 |

With only the shifted solver tolerance tightened to `1e-9`, all six cases
pass the same rule. Their differences range from `-1.3021812e-7` to
`+2.8279471e-8`, against limits from `7.0554632e-7` to `7.0619742e-7`.
The strict shifted residuals are `9.56e-10` through `9.96e-10`. The strict
row-18514 adjoint is `1.2110478841e-4`, differing from the archived ordinary
adjoint by `-4.4765300e-9`, with shifted residual `8.3490161e-10`.

This isolates the rejection to the `1e-7` shifted linear-solve accuracy. The
finite-probe epsilon is not the cause, so it stays at `5e-5`; changing only the
coupled-tangent/adjoint solve tolerance is the smallest supported correction.
The original unshifted residual remains a separate quantity because the
projection comparison intentionally uses the same shifted derivative model.

The diagnostic completed normally in 87.76 seconds. Its receipt is
[`receipt.json`](../data/mouthopen-joint-probe-epsilon-001/receipt.json), and
the Comet run is [mouthopen-joint-probe-epsilon-001](https://www.comet.com/liblaf/apple/0dac63fd278d4cf0b12fd82110dc8376).
