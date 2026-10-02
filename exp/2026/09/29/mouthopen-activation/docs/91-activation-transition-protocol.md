# Smile to MouthOpen activation transition

The requested animation uses the learned-axis endpoints of the Smile and MouthOpen four-stage chains. It renders the full boundary of the pruned MouthOpen tetrahedral mesh, following the user's full-tetmesh rendering preference.

## Activation and jaw path

For each active tetrahedron, interpolate the full symmetric tensor:

`S(alpha) = (1 - alpha) S_Smile + alpha S_MouthOpen`, with `B = I + S`.

The endpoints are contraction-only rank-one tensors. Their convex combination is positive semidefinite but may have two nonzero modes. We do not interpolate signed eigenvectors or project the blend back to rank one. The activation panel displays the strongest positive principal mode; the full tensor drives the solve.

Smile uses the neutral mandible pose. MouthOpen includes the chin-derived prescribed mandible pose from `data/10-mandible/prepared.npz`. The rotation is `exp(alpha * rotation_vector)` about the saved pivot and the translation is `alpha * translation`. This is a joint activation and prescribed jaw transition. The jaw is not predicted by activation. Both changes are necessary to reproduce these experiment endpoints.

The 121 requested states use `alpha = (1 - cos(pi * i / 120)) / 2` to ease in and out. Each state is independently checked for equilibrium after a warm start from the preceding state; positions are not blended for the animation. A harmonic rigid-motion carry supplies an initial guess for changing jaw constraints. It does not replace the forward solve.

## Exact mesh transfer and endpoint checks

Smile source: `exp/2026/09/21/stress-activation-loss/data/51-visualization-checkpoints-002/l2-normal/l2-normal-rankone_learned/last.npz`, SHA-256 `736577531f98eb9765eec0d2d13782e8486c68bbb5f62cc54896ca04ed6521b6`.

MouthOpen source: `data/70-mouthopen-four-stage/rankone_learned/last.npz`, SHA-256 `be53b6a0617c2ba7b32d4142b02f50541380e8dd17be9516a2a8f4683d35cd8e`.

The original Smile mesh has 288,235 active cells. The pruned mesh retains 288,172. Transfer uses original cell IDs, verifies every retained tetrahedron and rest vertex by exact identity, and removes only the 63 active cells among the previously removed fully fixed cells. The original Smile displacement is mapped only as an initial guess for a strict equilibrium solve. Its saved historical `solver_valid=false` flag is retained in the new provenance. The saved strict MouthOpen state is also replayed, and the final continuation state is compared with it rather than silently replaced by it.

## Numerical acceptance

Use the existing MouthOpen bulk material, corrected physical `J=det(F)`, fixed vertex mask, no skin membrane, no contact, and Newton search-shift reuse. Require free-force norm at most `1e-10`, with at most 100 Newton iterations and 3,000 CG iterations per ordinary frame solve. The initial historical Smile replay has a separate maximum of 1,000 Newton iterations because its saved displacement is approximate. No optimization or refitting is performed and no approximate-solve context is entered.

The 1,000-step replay also stopped above tolerance (`2.85444e-10`). Its trace showed that the historical near-tolerance shift reset repeatedly rejected the unshifted search system and accepted a large stabilization shift. The next attempt sets `reuse_shift_force_ratio=0` in a process-local wrapper, retaining shift reuse through convergence. This changes only the Newton search policy; energy, force, physical Hessian, Armijo rule, and force tolerance are unchanged. It restarts from the finite last iterate of the failed replay, verifies its provenance and fixed boundary, and requires a fresh strict solve before accepting frame zero.

As previously authorized, finite inversions are allowed up to 0.1% of retained cells (1,144 cells). Store the minimum physical J, inversion count, solver receipt, and full-boundary self-intersection diagnostic for every accepted frame. These are exploratory equilibrium states; neither the absence of contact nor a small force residual establishes mechanical validity or dynamical realism. This is a quasi-static animation, without inertial or timing calibration.

A failed transition interval may be bisected up to six times, with minimum alpha step `1e-5`; intermediate accepted states and failed solver receipts are saved. If the strict solve or inversion limit still fails, the pipeline stops visibly. No frame is fabricated from a failed solve.

## Artifacts and execution

Solver: `src/91-solve-activation-transition.py`; current outputs: `data/91-smile-mouthopen-transition-003/`. The first two attempts in `data/91-smile-mouthopen-transition/` and `data/91-smile-mouthopen-transition-002/` hit their initial iteration limits before meeting the force gate; their failures and frozen sources are preserved. Save endpoint tensors and mapping, full mesh, per-frame displacement/alpha/pose, input hashes, imported numerical source snapshots, and summary diagnostics. Renderer: `src/92-render-activation-transition.py`, with a common camera and activation scale, MP4, and keyframe contact sheet. Endpoint holds repeat already solved states.

```bash
CHERRIES_NAME='Smile to MouthOpen transition with continuous shift reuse' \
CHERRIES_TAGS='mouthopen,smile,activation-transition,forward-equilibrium' \
.venv/bin/python \
  -u src/91-solve-activation-transition.py \
  --output 91-smile-mouthopen-transition-003 \
  --initial-seed data/91-smile-mouthopen-transition-002/failed-solver-state.npz \
  > logs/91-activation-transition-003-terminal.log 2>&1
```

Run from this experiment group. The Cherries profile disables automatic commits. The result report will record the completed run, observed checks, runtime, and visual inspection.
