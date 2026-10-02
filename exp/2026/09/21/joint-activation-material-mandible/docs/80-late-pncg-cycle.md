# Late PNCG cycle diagnostic

This bounded diagnostic cloned the accepted run-009 checkpoint at step 8,000. It
did not modify runs 009 or 010 and did not change the physical model. The main
receipt is `data/late-pncg-cycle-diagnostic-002/summary.json` (SHA-256
`af0f31315f934d8a637bb406abbf203b513a6ebc2c71798027448a07a3029ef0`).
The Comet run is
<https://www.comet.com/liblaf/apple/ac7dcd37ad3842b18419f2a56f54d5e6>.

The frozen-state gradient and exact Hessian-vector product are consistent with
central finite differences. The best relative gradient error was
`1.06e-5`; the exact HVP error was `1.92e-7`. The PNCG quadratic approximation
was not the exact directional curvature:

| quantity | value |
| --- | ---: |
| approximate `p^T H p` used by PNCG | `1.105295e-15` |
| exact HVP `p^T H p` | `2.774587e-15` |
| relative underestimate | `60.16%` |

This explains the late two-cycle's overlong Newton steps. Along the sampled
direction, a 1 micrometre trial had actual energy change `-1.49e-17`, while the
approximate damped quadratic predicted `-8.17e-17`. At 2 micrometres the true
energy increased although the approximate model still predicted a decrease.
The mismatch is in the deliberately approximate, positive/Gauss-Newton-style
`hess_quad`, rather than the physical energy gradient or exact HVP. This probe
did not decompose the difference among bulk, skin, and contact terms, so it does
not attribute the underestimate to contact.

A 20-step clone using exact-HVP directional curvature and Armijo `c=0.25`
reduced the free-force norm from `8.4093e-10` to `8.2989e-10` code units, with a
minimum of `8.2766e-10`. It needed no backtracking; actual/predicted decrease
ratios stayed near one and damping fell from `1e-3` to `1e-6`. For comparison,
the independently running approximate-curvature run 010 reached
`1.0136e-9` after its first 20 accepted steps and used seven backtracks, though
it decreased energy more. Therefore exact curvature repairs the local model
agreement, but this short probe alone does not prove faster force convergence.

The earlier receipt `data/late-pncg-cycle-diagnostic-001/summary.json` preserves
an old-Armijo (`1e-4`) 20-step control from the same checkpoint. Its force grew
to `4.3852e-9`; large Dai-Kou coefficients and directions poorly aligned with
preconditioned steepest descent appeared after the restart. That control used
the source archived with receipt 001. Receipt 002 used the current source only
to add the exact-curvature diagnostic.

Reproduction command from the experiment directory:

```bash
CHERRIES_NAME='Late PNCG exact-curvature diagnostic' \
CHERRIES_TAGS='joint-inverse,pncg,diagnostic,gpu,exact-hvp' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/80-diagnose-late-pncg-cycle.py \
  --output-dir data/late-pncg-cycle-diagnostic-002
```

The justified next PNCG-only test is a bounded continuation using the exact HVP
for directional curvature while retaining strict Armijo, IPC contact, the
force threshold, and all material parameters. This is a numerical solver change
only. A longer matched continuation is still required before calling it an
acceleration or a convergence fix.
