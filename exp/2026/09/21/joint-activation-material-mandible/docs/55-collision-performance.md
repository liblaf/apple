# Collision performance measurements

The latest benchmark uses the repaired, volume-valid initialization:
[valid-seed operation timing](63-valid-seed-operation-performance.md).
It measures force evaluation at 4.02 times the contact-off cost, first Hessian
product at 4.53 times, and cached products at 1.12 times on a shared GPU.
There is still no converged complete-source forward/adjoint comparison.

Four earlier measurements are preserved below. They answer different questions and must
not be combined into one overhead claim.

## Legacy partial-FEM contact: matched forward and adjoint

The successful run is
[`data/collision-benchmark-legacy-contended-002/summary.json`](../data/collision-benchmark-legacy-contended-002/summary.json)
(SHA-256 `e618d951b4a0270e2d654db830061936dcac6e68b4c6309f8c8a37db7cf21e24`,
[Comet](https://www.comet.com/liblaf/apple/b3a341bf8ab24d75a869fee44c4c546f)).
It compares collision off against the legacy pure-soft-versus-pure-bone FEM
boundary collider. Both arms use the accepted Spatial80 update-111 displacement,
identical coefficients except for a predeclared 1% increase of the fixed skin
resultant from 20.15 to 20.3515 N/m, zero jaw pose, Newton-CG tolerances
`rtol=1e-6`, `atol=1e-12`, linear `rtol=1e-3`, and adjoint `rtol=1e-7`.
One warmup precedes five measured ABBA-ordered repeats per arm; the displacement
seed and coefficients are reset and the adjoint warm start is cleared every time.

| Warmed wall time | Collision off | Legacy partial-FEM IPC | IPC / off |
| --- | ---: | ---: | ---: |
| Forward median | 9.904 s | 6.324 s | 0.639 |
| Adjoint median | 6.773 s | 8.551 s | 1.263 |
| Forward + adjoint median per repeat | 16.582 s | 14.050 s | 0.847 |

The forward ranges were 8.899–10.292 s off and 5.498–6.896 s with IPC. Adjoint
ranges were 5.322–8.551 s off and 6.525–9.047 s with IPC. The comparison was
GPU-contended: utilization snapshots ranged from 40% to 99%, and another Python
process was present throughout. These are provisional absolute times and should be
repeated once idle.

The shorter IPC forward is caused by different nonlinear work, not evidence that
contact accelerates the solver. The common seed came from the contact equilibrium.
After the skin perturbation, the off arm needed five Newton steps, 1,505–1,521
linear HVP/matvec calls, and six line-search trials. The IPC arm needed two Newton
steps, 879–881 linear calls, and two trials. By contrast, the matched adjoint was
26.3% slower with legacy IPC. All adjoint relative residuals were below
`9.96e-8`; every state had zero inverted tetrahedra. The IPC state had 141 active
pairs and a minimum active distance of about 50.35 µm.

The legacy contact-only medians were 12.28 ms for collision-state rebuilding,
1.14 ms for energy, 3.42 ms for gradient, 10.03 ms for first HVP assembly plus
matvec, 2.85 ms for a cached HVP, and 10.96 ms for CCD. Cold physics construction
was 3.324 s off versus 5.292 s with legacy IPC; Spatial80 construction was timed
separately.

The first launch in `data/collision-benchmark-legacy-contended-001/` failed before
any solve because an exact binary-float equality rejected
`20.351499999999998`. It is preserved; run 002 uses the declared `1e-12`
absolute contract.

## Complete source skull: contact-only CPU diagnostic

The complete-source contact-only run is
[`data/full-skull-contact-microbenchmark-invalid-candidate-001/summary.json`](../data/full-skull-contact-microbenchmark-invalid-candidate-001/summary.json)
(SHA-256 `8a40897f8a3029e4d71f03c1a616e09b10bad4df7d456569e626e89e63e14627`,
[Comet](https://www.comet.com/liblaf/apple/b47030198e8242119e59a7a653cc52ae)).
It retains all 35,162 cranium and 18,948 mandible source triangles with unchanged
source coordinates. OMP, MKL, OpenBLAS, NumExpr, and Torch intra/inter-op threads
were fixed to one. One warmup preceded five repeats.

Cold contact construction took 0.170 s. Warm medians were 15.52 ms for state
rebuilding, 0.875 ms for energy, 4.642 ms for gradient, 17.85 ms for first HVP
assembly plus matvec, 1.201 ms for cached HVP, and 11.19 ms for CCD. The state had
6,178 active pairs, positive minimum active distance 0.122 µm, and a finite valid
barrier.

This is deliberately an invalid-volume diagnostic. Its collision-free geometric
candidate still has 13 inverted tetrahedra (`detF_min=-0.3863`), so no FEM
equilibrium, forward, or adjoint was run and no production admission is claimed.
A later repaired seed now passes the collision and FEM-volume initialization gates;
its bounded equilibrium attempts are reported below. This diagnostic itself still
contains inverted tetrahedra, and the legacy timing above is not evidence of
complete-source skull overhead.

## Complete source skull: fixed-state full-model operations

The GPU operation benchmark is
[`data/full-skull-model-operation-benchmark-invalid-contended-002/summary.json`](../data/full-skull-model-operation-benchmark-invalid-contended-002/summary.json)
(SHA-256 `2facfc985f306e905dbf21497668db3a0f57956d93ad2d51f7ec0252a7569dd2`,
[Comet](https://www.comet.com/liblaf/apple/d65571afc5b74c208f8495f8272916b8)).
It uses the same extended `Model` and `DofMap` assembly as the complete-source
wrapper and compares contact off against complete-source soft-versus-bone contact
at an identical fixed displacement, material state, and HVP direction. One warmup
precedes five ABBA-ordered repeats per arm, with CUDA synchronization around each
operation.

| Warmed operation median | Contact off | Complete source | Source / off |
| --- | ---: | ---: | ---: |
| State construction | 0.212 ms | 13.469 ms | 63.55 |
| Full-model energy | 4.921 ms | 5.950 ms | 1.209 |
| Full-model gradient | 1.259 ms | 4.901 ms | 3.893 |
| First HVP assembly + matvec | 2.592 ms | 21.738 ms | 8.388 |
| Cached HVP | 2.823 ms | 3.505 ms | 1.242 |
| CCD | 0.047 ms | 11.727 ms | — |

The off-arm CCD entry is a no-op, so its 251.6× ratio has no useful physical
meaning; the complete-source absolute cost is the relevant quantity. The contact
state was numerically valid with 6,174 active pairs and a minimum active distance
of 0.122 µm.

This run was GPU-contended at launch (96% utilization with the unrelated Python
process present), and it deliberately made no equilibrium call. It also used the
same candidate with 13 inverted tetrahedra as the CPU diagnostic. The measurements
therefore isolate fixed-state kernel and transfer costs; they do not establish
complete-source forward, adjoint, or optimization overhead. Run 001 is preserved
as a failed validation attempt: its bitwise HVP-repeat assertion was too strict for
GPU reductions, and run 002 instead enforces a `1e-12` relative repeat contract.

## Volume-valid common-seed equilibrium attempts

The repaired initialization in
[`data/full-skull-initialization-candidate-002/admission.json`](../data/full-skull-initialization-candidate-002/admission.json)
is collision-free and has `detF` in `[0.25010, 1.99990]`. It is explicitly an
initialization rather than an equilibrium. Two bounded attempts used its exact FEM
displacement, unchanged Spatial80 update-111 coefficients, zero jaw pose, the same
extended source-node `DofMap`, and frozen Newton-CG settings (`rtol=1e-6`,
`atol=1e-12`, linear `rtol=1e-3`, at most 12 Newton steps). Only the collision
object differed.

The contact-off attempt is preserved in
[`data/full-skull-forward-benchmark-contended-001/summary.json`](../data/full-skull-forward-benchmark-contended-001/summary.json)
(SHA-256 `e99ef41dbd8bda8ecd11006f0cc6192a475d7455fa548d17f5d889306f0a1d42`,
[Comet](https://www.comet.com/liblaf/apple/7eaa6b4ab4d14f179e30314331d70632)).
It ran for 242.90 s. The initial force norm was `6.7841e-6` and the force
threshold was `6.7841e-12`. Six Newton steps were accepted, using 30,208 recorded
linear HVP/matvec calls. The seventh linear solve exhausted its 10,000-iteration
cap with relative residual `2.482e-3`, above the declared `1e-3` tolerance.

The complete-source soft-bone attempt is preserved in
[`data/full-skull-forward-first-attempt-contended-002/summary.json`](../data/full-skull-forward-first-attempt-contended-002/summary.json)
(SHA-256 `36e9357e981075b3dee471db0349564c8b1148530b13f62eca7a7290ba054105`,
[Comet](https://www.comet.com/liblaf/apple/650cbf8c958b428fb6989b0b46777f8d)).
It ran for 76.18 s. Its initial force norm was `1.0236e-5`; the first Newton
linear solve exhausted 10,000 iterations with relative residual `0.84095`, far
above `1e-3`, so no Newton step was accepted.

Both runs failed before the adjoint and before any warm repeat. Consequently,
there is no complete-source forward-plus-adjoint timing ratio. They failed under
the current Newton-CG protocol; the evidence does not distinguish conditioning,
indefiniteness, or another linear-solver limitation. They do not measure collision
overhead and do not establish model infeasibility. The complete-source policy is
soft-versus-bone only. Source cranium-mandible contact remains excluded, and the
separate bone-bone audit found genuine source-bone intersections, so neither run
supports jaw-domain or final-launch readiness.
