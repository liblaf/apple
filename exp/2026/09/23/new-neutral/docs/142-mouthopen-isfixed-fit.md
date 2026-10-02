# MouthOpen fit from the corrected neutral

The trial completed 20 accepted optimizer updates with the corrected `IsFixed`
map, skin pre-strain, and soft–rigid IPC collision. The final state is a verified
equilibrium, but remains a poor MouthOpen fit. Inverse convergence is not claimed.

| Quantity | Initial neutral | Final fit |
| --- | ---: | ---: |
| Skin-area-weighted target RMS | 7.625211 mm | 7.429377 mm |
| Free-force residual | 0.009646 N | 0.009180 N |
| Mandible rotation magnitude | 0 degrees | 0.072607 degrees |
| Mandible translation magnitude | 0 mm | 0.073529 mm |
| Minimum physical det(F) | 0.483974 | 0.100000004 |
| Inverted tetrahedra | 0 | 0 |

The RMS reduction is 2.568%. The final unweighted skin-vertex RMS is 7.6047 mm;
it is a different statistic from the area-weighted fitting objective above.

## Physical model and optimization

- Initialization is the independently verified `forward-isfixed-001` endpoint,
  with zero mandible pose and zero bulk activation. Its loaded-equilibrium
  coordinates are not substituted for the constitutive reference.
- The expression displacements were freshly transferred onto that neutral in
  `blendshapes-isfixed-001`. The target was not inherited from the invalidated
  old neutral.
- Original FEM constraints use `IsFixed` alone: 27,036 fixed vertices. All
  3,408 lip vertices and all 15,299 fitting-surface vertices are free.
- Six unrestricted Raw6 active-strain components are optimized per active
  muscle tetrahedron (288,235 cells), jointly with six mandible pose parameters.
  Skin prestretch, skin modulus, and skin thickness are preserved exactly.
- Frictionless IPC includes soft tissue against cranium, mandible, and eyes.
  Soft–soft and rigid–rigid contact remain disabled in the inherited model.
  The neutral's final adaptive barrier stiffness is held at 1.3544 MPa during
  fitting to define a consistent differentiable objective.
- Harmonic carry is a numerical seed, admitted by CCD. Every physical forward
  solve retains collision. No collision-off relaxation is used in this trial.
- Each accepted equilibrium must satisfy the 1e-8 MPa m² (0.01 N) force gate,
  contact feasibility, no enabled-pair intersections, and positive det(F).
- Proposed pose steps are capped at 1 degree and 1 mm. Fully prescribed
  tetrahedra must additionally retain det(F) >= 0.1. If no jaw step is admitted,
  its update is zero while the independent strain block continues.
- The adjoint uses an explicit relative shift of 0.001 times the mean absolute
  physical Hessian diagonal. This is an approximate gradient; it does not
  change the forward energy or convergence tolerance. The shifted residual
  gate is 1e-7; physical unshifted residuals are recorded as damping bias.
- Actual resolved objective decrease is required by the outer line search.

## Why jaw motion is limited

The CPU pose audit found 974 tetrahedra whose four vertices are prescribed but
span moving mandible and stationary cranium nodes. Their compositions are
444 cells with one jaw vertex, 304 with two, and 226 with three. The full fresh
chin pose (10.023 degrees and 5.968 mm) would invert 168 such cells. No free-DOF
carry or equilibrium solve can repair a tetrahedron whose four coordinates
are prescribed. The optimizer's proposed jaw path reached the det(F)=0.1
quality limit and its pose stopped advancing.

This establishes a concrete restriction on the tested pose path, not a proof
that no other pose or unrestricted strain field could improve the target fit.
The finite iteration trial is not evidence of a global inverse optimum.

## Runs, checks, and reproducibility

Commands run from `exp/2026/09/23/new-neutral`, using the repository's Python,
`OMP_NUM_THREADS=4`, and human-readable `CHERRIES_NAME` plus tags
`mouthopen,isfixed,inverse,active-strain,collision,skin-prestrain`.

```bash
.venv/bin/python -u \
  src/142-fit-mouthopen-isfixed.py
```

Each output includes its exact executed source archive and configuration.

1. `inverse-mouthopen-isfixed-001` stopped before optimization at a fixed-state
   pullback check. Reducing its finite-difference step from 1e-5 to 1e-6 made
   the jaw components agree within the same 1e-3 relative threshold; no
   validation tolerance was loosened.
2. `inverse-mouthopen-isfixed-002` accepted 11 updates with strain learning
   rate 0.005, then stopped when its finite pose backtracking budget was
   exhausted near the quality boundary. Its saved valid endpoint was retained
   and its stale running status was finalized with the exception evidence.
   Comet: <https://www.comet.com/liblaf/apple/06cd2729f6624c319c1ee9c6f601de28>
3. `inverse-mouthopen-isfixed-003` restarted from that checkpoint with reset
   optimizer moments, strain learning rate 0.02, and the active-constraint
   handling above. It accepted nine more updates, completed normal Cherries
   shutdown with exit code 0, and stopped at the declared iteration budget.
   Comet: <https://www.comet.com/liblaf/apple/b31973fecdfb43248bffb5ff923af16f>

There were two rejected geometry trials: one inverted harmonic carry in phase
002, and one inverted terminal forward candidate in phase 003. Neither became
an accepted endpoint. The saved accepted-step elapsed times sum to 617.2 s,
excluding initialization, process shutdown, target transfer, and audits.

`144-audit-mouthopen-isfixed.py` independently rebuilt the final physical model,
restored the saved pose and strain, and recomputed force, contact, CPU det(F),
and target RMS. All gates passed, including an 83.913789 micrometre active
contact gap and zero enabled-pair intersections. The recomputed quantities
agree with the saved summary. The audit explicitly checks the IsFixed map,
free lips, and unchanged skin prestretch arrays.

Audit: <https://www.comet.com/liblaf/apple/09744254ced24ebd967406ad4f603c15>

Static Ruff checks passed for the modified runner, wrapper, seed helper, and
gradient checker. The fit and curve images were visually inspected. Existing
unrelated working-tree changes were preserved and no commit was made.

The full-tetmesh comparison, transferred target, fit/force curves, and render
receipt are in `data/review-mouthopen-isfixed-002`. The final physical audit is
`data/inverse-mouthopen-isfixed-003/independent-audit.json`. See also reports
140 (target transfer), 141 (pose feasibility), and 144 (independent endpoint).

## Publication

The prior transient preview service was absent. A new runtime-only user unit
`apple-neutral-isfixed-preview.service` serves the saved review at
PRIVATE_PREVIEW_URL . The main page, images,
render receipt, and independent audit returned HTTP 200 and matched local
files byte for byte. Neutral and full-tetmesh pages were also rechecked.
All render-receipt asset hashes passed after adding the audit link.
No persistent service was installed.
