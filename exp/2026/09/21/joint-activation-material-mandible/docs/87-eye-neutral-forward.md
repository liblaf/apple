# Pre-stressed neutral with fixed rigid eyes

The eye-inclusive neutral forward converged in **2,980 accepted PNCG steps / 424.95 s**. Its independently recomputed free-force norm is **0.149980 mN**, below the original **0.151920 mN** threshold. The endpoint has **zero inverted tetrahedra**, no IPC soft/rigid intersections, and exactly fixed eye and support degrees of freedom.

## Physical model and inputs

- Both eyes are fixed rigid obstacles in the registered source frame. All **1,298 source vertices and 2,560 triangles** from Melon's `20-eye.ply` are retained. They have no motion parameters and do not follow the mandible.
- Contact also retains the full source cranium and mandible. The same pure-soft boundary participates in frictionless IPC contact against all rigid obstacles.
- The original FEM constitutive reference and frozen material fields are preserved. Bulk tissues use passive polynomial Stable Neo-Hookean elasticity with zero additive bulk stress and no activation. Skin retains its prescribed heterogeneous plane-stress elasticity, thickness, and baseline stress resultant. All tissue Poisson ratios remain **0.49**.
- The mandible is fixed at zero pose. This is a forward equilibrium solve, without inverse or adjoint optimization.
- `frozen-neutral-004` is the preserved prior loaded face and material bundle. The start is a repaired copy of its displacement, not a stress-free rebase.

The previous loaded face intersected the eyes in 687 triangle pairs, with a deepest soft-node penetration of 0.502 mm. `eye-initialization-003` clears the source eyes by local free-surface projection and whole-face separating planes, with a harmonic extension into the volume. It moves 452 soft-boundary nodes, has a maximum repair displacement of 2.454 mm, and begins with seven inverted tetrahedra. All seven resolve during the forward relaxation. No rigid or fixed attachment coordinates change. Independent geometry checks found no eye, cranium, or mandible intersections in the repaired start; exact-welded closed-eye containment checks also found no nonrigid FEM nodes inside either eye.

## Result

| Quantity | Eye-inclusive endpoint |
| --- | ---: |
| Initial free-force norm | 106.030760 N |
| Final free-force norm | 0.000149980232 N |
| Required force threshold | 0.000151920035 N |
| Accepted PNCG steps | 2,980 |
| Timed forward and terminal checks | 424.95 s |
| Inverted tetrahedra | 0 / 1,146,517 |
| det(F) range | 0.374199813–1.600054622 |
| Active IPC contacts | 5,195 |
| Minimum active contact gap | 17.066746 μm |
| Surface motion RMS from constitutive reference | 1.398591 mm |
| IPC soft/rigid intersections | 0 |
| Fixed-eye displacement | exactly zero |
| Rejected contact-buffer trials in successful run | 0 |

The reported force is the exact gradient at the saved accepted state. Code force units are MPa·m²; values are multiplied by 1e6 to report newtons. Tissue and contact free-gradient norms are each approximately 0.269 N; their vector sum is the residual above. An independent component recomputation matches the reported norm to relative error 1.90e-14.

Relative to the previous adopted neutral, the 15,299 observed surface nodes move **0.033334 mm RMS** (unweighted), with a maximum change of **0.232678 mm**. The maximum change across all FEM nodes is 0.680581 mm. The saved-endpoint review independently found zero source-eye, source-cranium, and source-mandible triangle intersections, and zero nonrigid FEM nodes inside either watertight eye proxy (199,059 nodes checked). Minimum vertex-to-eye clearances are approximately 92.01 and 92.43 μm; these are vertex-clearance statistics, distinct from the global minimum active IPC primitive-pair gap in the table.

## Numerical correction

Preserved attempt `eye-neutral-forward-001` stopped after 50 accepted steps. One near-contact gap reached 7.90 nm, below the existing 10 nm CCD buffer, after which CCD returned zero step lengths. There were no intersections, and all seven initial inverted cells had already resolved.

The original TightInclusionCCD default tolerance was 1 μm, larger than that buffer. The successful run explicitly uses **0.1 nm CCD tolerance**, a 100,000-iteration CCD budget, a 0.95 safety factor only for collision-limited steps, and a pre-acceptance check rejecting any trial whose minimum active gap violates the original 10 nm buffer. These are collision-path numerical controls; the barrier energy, forces, barrier distance, contact stiffness, geometry, material fields, and convergence threshold are unchanged. The extra trial rejection was not needed in the successful run.

PNCG otherwise retains the successful prior settings: strict Armijo 0.25, maximum proposed displacement component 0.5 mm, initial Hessian damping 0.001, conjugate-direction restart every 200 steps, and no inversion step guard. Inversions remain reported, consistent with the user's allowance.

## Reproduction and evidence

Run from `exp/2026/09/21/joint-activation-material-mandible`:

```bash
CHERRIES_NAME='Fixed rigid eyes neutral forward precise CCD' \
CHERRIES_TAGS='joint,forward,pncg,rigid-eyes,nu049,precise-ccd' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/87-run-eye-neutral-forward.py \
  --initialization-dir data/eye-initialization-003 \
  --output-dir data/eye-neutral-forward-002
```

Use a new output directory when reproducing; completed runs are never overwritten.

- [Cherries/Comet run](https://www.comet.com/liblaf/apple/56f927e06a3a49fab57ac3af120202dc): `Fixed rigid eyes neutral forward precise CCD`; process and shutdown hooks exited successfully.
- Result: `data/eye-neutral-forward-002/summary.json`.
- Exact run inputs and effective numerical settings: `data/eye-neutral-forward-002/protocol.json`.
- Per-step history: `data/eye-neutral-forward-002/trace.jsonl`.
- Endpoint: `data/eye-neutral-forward-002/checkpoint-terminal-step-02980.npz`, SHA-256 `e522b484b05e21ee3d2150876bffaf9ea1470cf0dd44e4871131219760583f18`.
- Preserved console, including Comet summary: `tmp/87-eye-neutral-forward-002.console.log`.
- Eye adapter validation: `data/rigid-eye-contact-validation-004/summary.json`; gradient and HVP finite-difference relative errors approximately 9.0e-9 and 2.1e-9, with exact material identity and eyes fixed under a nonzero mandible pose.
- Independent endpoint geometry audit and visual/ParaView exports are generated by `src/86-review-eye-neutral.py` and described in `docs/86-eye-neutral-review.md`.
- Final audited visual bundle: `data/eye-neutral-forward-review-006`. The mobile tailnet review (private preview omitted) serves all six comparison/contact/convergence figures. HTTP responses were checked against the published status and figure hashes; the page continues to mark the final joint inverse stage as pending.

The run used the existing dirty workspace at Git base `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; exact runtime sources are archived and hash-bound in the run. No commit was created. The original neutral bundle and earlier attempts remain preserved. This result establishes forward force convergence under the stated contact model; it is not a completed joint inverse optimization or a proof of anatomical accuracy/global mechanical stability.
