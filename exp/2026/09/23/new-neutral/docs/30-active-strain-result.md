# New neutral with active strain

The neutral was recomputed with active-strain bulk and skin materials using the
optimized hybrid PNCG → GPU sparse Newton-CG solver. The run met its force
tolerance in **79.007 s**, but **13 tetrahedra are inverted**. It remains a
diagnostic endpoint rather than a valid neutral equilibrium. Bone/eye contact
passed the intersection and minimum-gap gates.

## Material formulation

All three bulk tissues now use the library `StableNeoHookeanActive`, with
`B = A⁻¹ = I` for this neutral configuration. Its six stored symmetric
increments are zero. The new experiment-local
`StableNeoHookeanActiveMembrane` uses an explicit 2 × 2 tangential B and contains
no additive `baseline_stress` field. None of the installed potentials has an
`active_stress` or `baseline_stress` material field.

The existing prescribed skin tension T is converted once to

```text
B = sqrt(I + T / (h μ))
A = inverse(B)
```

The principal values of B span **1.264579–1.478075**; those of A span
**0.676556–0.790777**. These are equivalent prestretches derived from the
previous load calibration, not independently measured tissue prestrains.
The [saved field arrays](../data/forward-active-strain-001/active-strain-fields.npz)
contain B, A, the source tension, μ, and thickness.

The material follows the repository's corrected physical-volume convention:
the activated bulk norm is ‖FB‖² and its determinant terms use physical
`J = det(F)`. The membrane's activated tangential norm is `tr(Bᵀ C B)`;
its relaxed normal stretch and physical area ratio are independent of B.
Normal activation is one. Skin thickness, moduli, jaw pose, constitutive
reference, contact geometry, and geometric initializer retain their existing
values.

For this law, the calibrated active-strain and stress formulations have
identical force and tangent operators in exact arithmetic; their energies
differ by a displacement-independent constant. At the actual full-face seed,
relative force error was **7.88e-17** and relative HVP error **1.59e-18**.
The energy-offset residual was **1.36e-20 MPa m³**. See the
[mapping receipt](../data/forward-active-strain-001/active-strain-mapping.json).
The changed nonlinear trajectory and inversion count must not be interpreted
as evidence that active strain improves this model's physical validity.

## Forward result

| Measurement | Result |
| --- | ---: |
| PNCG updates | 353 |
| Newton updates | 3 |
| Forward-loop time | 79.007 s |
| Terminal free-force norm | 0.526903 N |
| Force threshold | 0.576415 N |
| Numerical convergence | true |
| Inverted tetrahedra | 13 |
| Minimum det(F) | -0.876689576 |
| Minimum active contact distance | 65.0557 µm |
| Required contact distance | 0.01 µm |
| Contact validity | true |
| Valid forward endpoint | false |
| Observed-skin area-weighted displacement RMS | 0.334034 mm |
| Final IPC stiffness | 5.4176 MPa |

Timing includes synchronized profiling, tracing, adaptive contact updates, and
the first sparse Newton construction/JIT. Model construction, equivalence
validation, operator prewarm, and final review are outside the forward timer.
This is not a matched performance comparison with the earlier stress run.

## Validation

The existing library bulk active-strain tests passed: **6 passed** with
`python -m pytest tests/warp/test_stable_neo_hookean_active.py -q -o addopts=''`.

The new membrane was checked at an anisotropic SPD B against independent
PyTorch autograd and centered finite differences, including identity activation,
stress/strain force equivalence, Hessian products, the exact diagonal, and the
clamped PNCG quadratic. Its assembled local sparse Hessian exactly matched all
analytic HVP columns. The [derivative receipt](../data/forward-active-strain-001/active-strain-check.json)
records the tests. The complete GPU sparse Hessian also matched the full physical
HVP during the solve to relative error **3.06e-16**.

The saved-result [independent audit](../data/forward-active-strain-001/independent-audit.json)
recomputes det(F) from the original constitutive coordinates and checks archived
source/input hashes. The [rendered review](../data/review-active-strain-001/index.html)
includes another CPU geometry calculation, soft-rigid triangle intersection
checks, front/side views, inversion markers, mesh downloads, and material
evidence.

## Reproduction and run evidence

Working directory: `exp/2026/09/23/new-neutral`.

```bash
CHERRIES_NAME='New neutral active strain hybrid forward' \
CHERRIES_TAGS='neutral,active-strain,hybrid,pncg,newton,gpu-sparse,adaptive-ipc' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u src/30-forward-active-strain.py
```

The run wrote `data/forward-active-strain-001`; use a fresh `--output-dir` to
reproduce. It ran on the local RTX 4090 with FP64, four IPC/OMP threads, IPC
Toolkit 1.6.0, PyTorch 2.12.0+cu130, and Python 3.14.6. Git HEAD was
`d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` with pre-existing uncommitted work.
The executed sources are archived and hashed in the run directory.

- [Summary](../data/forward-active-strain-001/summary.json),
  [protocol](../data/forward-active-strain-001/protocol.json),
  [trace](../data/forward-active-strain-001/trace.jsonl), and
  [timing](../data/forward-active-strain-001/timing.json).
- [Endpoint](../data/forward-active-strain-001/endpoint.npz),
  [material source provenance](../data/forward-active-strain-001/active-strain-source-provenance.json),
  [base source provenance](../data/forward-active-strain-001/provenance.json), and
  [guard receipt](../data/forward-active-strain-001/forward-only-guard.json).
- [Completed log](../data/forward-active-strain-001/forward.log) and
  [Comet experiment](https://www.comet.com/liblaf/apple/f1c14fc5818a4c259a6d45f31792a380).

The forward-only guard pins the reviewed inverse-wrapper source and makes its
differentiable/adjoint entry points fail if called. No inverse update or
reconstruction of a fitted expression was performed.

Completed Comet summary excerpt:

```text
name                   : New neutral active strain hybrid forward
forward/seconds        : 79.00742461098707
forward/stiffness_mpa   : 5.4176
forward/success        : 1.0
forward/terminal_force : 5.269027804419614e-07
forward/valid          : 0.0
```

## Tailnet review

The active-strain review replaces the served page at
<PRIVATE_PREVIEW_URL>. The earlier stress result remains saved under
`data/review-001`. Serving uses a transient user service bound to the Tailscale
address only; it is not persistent across reboot.

All **15** linked page assets returned HTTP 200 over the Tailscale address and
matched their saved SHA-256 hashes. See the
[HTTP verification](../data/http-verification-active-strain.json) and
[serving receipt](../data/serve-active-strain.json). The transient unit is
`apple-neutral-active-strain-preview.service`; stop it with
`systemctl --user stop apple-neutral-active-strain-preview.service`.
