# Contact with the existing solved activation

The forward-only contact implementation preserves the saved MouthOpen and Smile activation tensors. The numerical attempt did not produce a new animation: it exhausted its 1,200-second continuation budget while equilibrating the neutral initializer, before reaching the MouthOpen activation or prescribed jaw pose. No activation optimization or inverse refit was performed.

## What was implemented

[`23-contact-existing-activation.py`](../src/23-contact-existing-activation.py) reads the hash-bound endpoint tensors and mesh from the completed parent transition. The intended expression path remains `S(beta) = (1-beta) S_MouthOpen + beta S_Smile`, with the saved jaw pose scaled by `1-beta`. Only displacement is solved. The original input files and earlier animation are preserved.

Frictionless standard IPC contact uses all 126,648 triangles of the extracted tetrahedral boundary. Contacts involving free vertices are enabled; pairs involving only prescribed vertices are exempt. The physical barrier uses stiffness 0.0012 MPa, activation distance 0.783672 mm, and 10 nm CCD minimum separation. Contact stencil Hessians are PSD-projected for the search direction; energy, gradient, bulk material derivatives and the strict force acceptance threshold are unchanged.

Two initialization restrictions matter:

- The saved expression displacements already intersect within the free-tissue contact scope. A collision-prevention barrier cannot start directly from them. [Saved-state audit](19-deformable-contact-preflight.md).
- Prescribed-only boundary faces intersect near 5.1603% of the saved jaw motion. Free-tissue forces cannot resolve those prescribed intersections. They are reported separately, so this contact scope cannot establish collision freedom of the entire boundary. [Prescribed-boundary audit](18-prescribed-boundary.md), [diagnostic figure](../data/30-prescribed-boundary-002/prescribed-boundary.png).

The attempted initialization therefore starts from the feasible rest mesh, equilibrates with contact, then would ramp the existing MouthOpen tensor and jaw together. The ramp was never reached. Scaling the tensor during this warm continuation is not an activation refit.

## Measured outcome

The process exited normally after saving an explicit `blocked` result and completing Cherries shutdown. A zero process exit code here means that failure handling and artifact writing completed; it does not mean equilibrium converged.

| Quantity | Result |
| --- | ---: |
| Phase reached | Neutral initialization, zero activation and zero jaw pose |
| Accepted equilibria / animation frames | 0 / 0 |
| Logged accepted Newton search steps | 70 |
| Continuation elapsed time | 1,200.749 s |
| Initial free-force norm | 6.40042e-7 |
| Final free-force norm | 1.90027e-8 |
| Required free-force norm | ≤ 1e-10 |
| Scoped boundary intersections at final state | None detected |
| Unfiltered boundary intersections at final neutral state | None detected |
| Active contact stencils | 140,670 |
| Minimum active-stencil distance | 0.496075 µm |
| Inverted cells | 21 / 1,144,268 |
| Minimum det(F) | -1.501433 |

The terminal state passes the scoped contact checks but misses the force gate by about 190 times. Its 21 inverted cells lie below the user-permitted 0.1% cap, but it is neither an accepted equilibrium nor mechanically valid. The active-stencil distance is not a global minimum over every primitive pair. The independently detected intersections are a separate check.

This is a bounded convergence failure, not proof that a contact equilibrium is unreachable. The GPU was shared with another numerical experiment, and additional bounded operator diagnostics also used the GPU; no isolated throughput claim is made. Increasing the unshifted linear-solver allowance alone did not resolve the rest system: a separate probe exhausted 20,000 PCG iterations without reaching relative residual 1e-3. [Linear-solver diagnostic](14-unshifted-pcg-probe.md).

## Verification and artifacts

The runner verified every recorded input and imported source hash at shutdown. Copied endpoint tensors and mesh retain their parent bytes. The independent [CPU audit](25-existing-contact-audit.md) verified all seven inputs and 118 frozen/live source records, exact neutral prescribed displacements, and the terminal inversion/intersection measurements. Reference tetrahedra have positive signed volume; the inversions above concern the final deformed state.

The focused contact tests passed: `python -m pytest -q tests/forward/test_gpu_contact_hessian.py tests/forward/test_collision_hess_quad.py` — 9 passed, with two existing PyTorch sparse warnings. This checks the operator behavior; it does not establish convergence of the full experiment.

The local alternative bulk BSR implementation matched the existing Hessian products to about 3.6e-16 relative error on three directions at rest and at nonzero saved activation, using the corrected contact filter. It showed no useful measured product-time improvement, so the main run retained `gpu_contact`. [Operator check](13-active-bulk-contact-check.md). The attempted full sparse backend and earlier superseded diagnostics remain preserved in their numbered reports.

Useful artifacts:

- [Terminal summary](../data/23-contact-existing-activation/summary.json), [failed solver state](../data/23-contact-existing-activation/failed-solver-state.npz), [process receipt](../data/23-contact-existing-activation/job.json).
- [Input and numerical source manifest](../data/23-contact-existing-activation/source-manifest.json), [complete terminal log](../logs/23-contact-existing-activation-terminal.log).
- [Completed-only full-tet animation renderer](../src/31-render-contact-transition.py). It was not run because the required 121 accepted frames do not exist. The previously delivered animation remains the original no-contact playback.

## Reproduction

Working directory: `exp/2026/09/30/mouthopen-smile-collisions`.

```bash
CHERRIES_NAME='Existing activation MouthOpen to Smile with scoped contact' \
CHERRIES_TAGS='mouthopen,smile,existing-activation,self-contact,standard-ipc,forward-only' \
.venv/bin/python -u \
  src/23-contact-existing-activation.py \
  --hessian-backend gpu_contact \
  --initialization neutral-continuation \
  > logs/23-contact-existing-activation-terminal.log 2>&1
```

This command records the completed attempt. Choose a new output and terminal-log path for any future rerun; the script refuses to overwrite an existing summary.

Cherries/Comet summary: name **Existing activation MouthOpen to Smile with scoped contact**; Git revision `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` with pre-existing local changes; start 2026-09-30 01:53:03 and end 02:13:13 Asia/Shanghai. [Comet run](https://www.comet.com/liblaf/apple/0c642137da2a43f98010b5a6b49df77c). The profile disables automatic commits. No commit or push was made.
