# New neutral with the optimized hybrid forward solver

The new forward solve met its force tolerance in **55.752 s**, but ended with
**21 inverted tetrahedra**. It is a diagnostic endpoint, not a valid neutral
equilibrium. Bone/eye contact passed the configured intersection and gap checks.

## Problem and method

The solve starts from the repaired constitutive-reference face in
`exp/2026/09/22/neutral-newton/data/reference-seed-002`. This geometric initializer
has no prior equilibrium displacement. It uses the material/reference data from
`frozen-neutral-004`, zero bulk activation, zero prescribed jaw rotation, the
existing heterogeneous skin tension, and complete cranium, mandible, and fixed
eyeball contact.

The current optimized hybrid uses clamped PNCG directional curvature, followed
by exact GPU sparse Newton-CG when PNCG stalls. Adaptive physical IPC stiffness
starts at 0.1693 MPa and finishes at 5.4176 MPa. The original constitutive
reference is retained. The force tolerance is anchored at the initial stiffness.

| Measurement | Result |
| --- | ---: |
| PNCG updates | 161 |
| Newton updates | 62 |
| Forward-loop time | 55.752 s |
| Terminal force norm | 5.745305055620925e-7 MPa m² (0.574531 N) |
| Force threshold | 5.764153171969013e-7 MPa m² (0.576415 N) |
| Inverted tetrahedra | 21 |
| Minimum det(F) | -0.97757094474634 |
| Minimum active contact distance | 63.9181 µm |
| Required contact distance | 0.01 µm |
| Observed skin displacement RMS | 0.217028 mm |
| Numerical convergence | true |
| Geometry validity | false |
| Contact validity | true |
| Valid forward endpoint | false |

Forward time includes synchronized profiling, recorded trace work, adaptive
stiffness updates, and the initial sparse Newton construction. It excludes model
construction, operator prewarm, and subsequent endpoint rendering/auditing. This
single run does not establish a performance improvement or a usable equilibrium.

## Command

Working directory: `exp/2026/09/23/new-neutral`.

```bash
CHERRIES_NAME='New neutral optimized hybrid forward' \
CHERRIES_TAGS='neutral,hybrid,pncg,newton,gpu-sparse,adaptive-ipc' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u src/10-forward-neutral.py
```

The default output was `data/forward-002`. Reproduction requires a fresh
`--output-dir`, because the runner refuses to overwrite an existing result.

## Evidence and reproducibility

- [Forward summary](../data/forward-002/summary.json),
  [protocol](../data/forward-002/protocol.json), and
  [trace](../data/forward-002/trace.jsonl).
- [Endpoint displacement](../data/forward-002/endpoint.npz),
  [timing](../data/forward-002/timing.json), and
  [stiffness history](../data/forward-002/stiffness.json).
- [Runtime binding](../data/forward-002/profile-input-binding.json),
  [archived source hashes](../data/forward-002/provenance.json), and
  [forward-only guard](../data/forward-002/forward-only-guard.json).
- [Independent audit](../data/forward-002/independent-audit.json) recomputes
  det(F) from the original constitutive coordinates on the CPU and confirms all
  21 inverted tetrahedra. The sparse Hessian product agrees with the physical
  reference product to relative error 3.07e-16.
- [Rendered review](../data/review-001/index.html) and
  [independent review receipt](../data/review-001/receipt.json).
- [Cherries log](../logs/10-forward-neutral.log) and
  [Comet experiment](https://www.comet.com/liblaf/apple/5f60d561528e4b4e87055b0809d4d5ad).

Runtime: local RTX 4090, FP64, four IPC/OMP threads, PyTorch and IPC versions
recorded in the protocol. Git HEAD was
`d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` with pre-existing uncommitted work;
archived sources, rather than Git HEAD alone, identify the executed solver.

`forward-001` stopped during setup with zero iterations because the historical
runtime checker rejected the changed differentiable/inverse wrapper. The retry
pins the reviewed inverse-wrapper hash, records its literal diff, and fails if
its differentiable, forward-step, receipt, or adjoint APIs are called. All
historical input arrays and material/contact binding checks still apply. The
first failure and log are retained in `data/forward-001`.

The top-level wrapper and runtime binder received formatting/lint cleanup after
the solve. The run directory retains the versions used by this run; the Cherries
snapshot also retains the entrypoint. The forward source bodies and numerical
parameters were not changed for the retry.

Comet summary excerpt from the completed run:

```text
name                   : New neutral optimized hybrid forward
forward/seconds        : 55.75225274899276
forward/stiffness_mpa   : 5.4176
forward/success        : 1.0
forward/terminal_force : 5.745305055620925e-07
forward/valid          : 0.0
```

The next solver investigation should locate when these elements first invert;
force convergence and collision feasibility alone did not preserve volume
orientation on this run.

## Tailnet preview

The review is served at <PRIVATE_PREVIEW_URL> from `data/review-001`.
All 12 linked pages, images, receipts, and mesh files returned HTTP 200 and
matched their local SHA-256 hashes when fetched through the tailnet address on
PC07. The [HTTP receipt](../data/http-verification.json) records those checks.
The final [rendering run](https://www.comet.com/liblaf/apple/4ac35931f4d84565a546a0a52d30444b)
also independently reproduced 21 inversions and zero soft-rigid triangle
intersection pairs.
The server binds only the Tailscale address and runs as the transient user unit
`apple-neutral-preview.service`; its unit file is under `/run/user/1000`, so it
is not persistent across reboot. Stop it with:

```bash
systemctl --user stop apple-neutral-preview.service
```
