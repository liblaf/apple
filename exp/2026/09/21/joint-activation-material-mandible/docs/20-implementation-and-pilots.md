# Implementation and neutral-pilot results

**Historical contact-free pilots; not converged preparation. The multi-expression joint trajectory has not run.** The original oral classifier incorrectly counted shared FEM edges and vertices as penetration. The corrected audit passes these saved states' numerical geometry checks; no provisional-exclusion decision is needed. Separate source-lip defects remain anatomical limitations. Contact-enabled convergence and its visual evidence are tracked in the live review (private preview omitted).

## Observed neutral trends

| Skin baseline | Updates | Surface drift | Final muscle drift | Final minimum det(F) | Inversions |
| --- | ---: | --- | --- | ---: | ---: |
| 0.806 N/m | 8 | 0.023 → 0.019 mm | 0.012 mm | 0.986 | 0 |
| 8.06 N/m | 12 | 0.219 → 0.188 mm | 0.108 mm | 0.835 | 0 |

![Neutral balance trends](../data/progress-report/neutral-trends.png)

Both pilots retained zero activation and met the proposed **numerical deformation budgets**: surface RMS ≤0.25 mm and muscle-centroid RMS ≤0.5 mm. Surface fit improved while muscle-centroid drift increased slightly. All shared bulk stress tensors and skin stiffness were optimized; the prescribed skin stress stayed fixed during each continuation stage. Both pass the corrected numerical oral audit; neither established inverse convergence or included a bone-contact force law.

These skin stresses are **1% and 10% of an 80.6 N/m literature-derived proxy**, not full recovery of that proxy or a measured prestress map. The second stage starts from the first stage's best numerical-budget checkpoint, with fresh Adam moments. The saved filename `best-admissible.pt` predates the oral audit and only denotes the original numerical budget; it does not certify oral validity. The shared basis is spatially constant: its spatial smoothness is identically zero. No activation-smoothness result can be inferred from a zero-activation pilot.

The original post-hoc audit reported these raw pair counts on the actual deformed FEM surface. The corrected classifier identifies all these new/worsened pairs as intersections confined to shared vertices or edges, rather than free-surface penetration. Mandible support remains exactly at zero pose.

| Pilot | New upper-oral pairs | New lower-oral pairs | Worsened inherited lower-oral pairs |
| --- | ---: | ---: | ---: |
| neutral-prestress-001 | 1 | 371 | 96 |
| neutral-prestress-010 | 3 | 362 | 98 |

The corrected audit finds zero new nonadjacent penetrations in these states. Pair counts and intersection-segment lengths are diagnostic proxies, not penetration depths or a contact law. The source/FEM oral correspondence is incomplete; these tests cannot certify anatomical accuracy. The primary joint runner requires new, converged contact-enabled preparation; these historical pilots do not meet that requirement.

## Implemented pieces

- Signed additive bulk stress and exact plane-stress Stable Neo-Hookean membrane with signed tangential stress.
- Twenty shared coefficients; dense six-component activation per active muscle tetrahedron; six jaw pose coordinates per expression.
- Owned equilibrium snapshots and direct plus implicit Dirichlet-pose gradients.
- Fixed input hashes, complete recovered mandibular support, four training targets and two reserved targets, and a same-muscle conductance graph.
- A gated joint/control runner with checkpointing. It remains unvalidated as a complete optimization workflow until the admission gates pass and it is executed.

## Readiness and limitation

The recovered fixture has 228,660 points, 1,146,517 tetrahedra and 288,235 active cells. There are 7,510 mapped mandibular boundary nodes. The source template contains 17 upper/lower-lip intersections after excluding shared seam vertices. Bone contacts include posterior joint-region contacts and anterior contacts that change with opening. A sampled hinge or pose box does not establish a valid deforming oral-contact model.

Material, field, implicit-boundary, coupled, and full-face derivative checks passed. The coupled test uses an independent explicit three-coordinate Newton solve for finite differences. The full-face test covers all shared baseline-stress families, skin stiffness, dense activation, and jaw rotation/translation at two step sizes (16 checks). Its maximum relative error is **0.0346%**, below the unchanged 2% criterion; 33 forward solves were used. An earlier attempt at absolute force tolerance 1e-13 stopped on strict Armijo near roundoff. The successful run used 1e-12 absolute and 1e-6 relative forward tolerances and 1e-7 adjoint tolerance. Its source snapshot and the failed-attempt receipt are preserved.

The equilibrium adapter recognizes an initially balanced state using the declared absolute free-force tolerance and clears stale solver state. Fresh synthetic equilibrium/coupled regressions pass, including exact neutral at 2.25e-24 free-force norm and zero iterations. A new full-face derivative validation with real cranium and mandible contact is running; its receipt is separate from the earlier contact-free validation.

The current machine-readable validation and oral-audit status is in [status.json](../data/progress-report/status.json). Strong-smoothness calibration, a control trajectory, a joint trajectory, and a complete joint checkpoint-resume run remain unexecuted. The new runner checks artifact lineage and reconstructs objective components, freezes shared optimizer state in the control, and requires repeated smoothness calibration at tighter solver tolerance before fitting. The first pilot's activation neighbor-RMS budget is frozen at 0.05 normalized units (0.616 kPa); calibration must pass within it. The current neutral checkpoint also needs a new replay under the final manifest because the oral audit changed the manifest after the pilot.

The normal pilots produced local Cherries outputs; Comet warned that environment/Git-patch metadata did not finish logging. The first full-face validation also hit a Local-plugin log-copy error after writing its successful scientific receipt. The experiment profile now initializes Logging before Local so its reset does not remove Local's log handler; report generation verified that the snapshot log is written. A CLI help invocation also initialized a metadata-only Comet entry before it was interrupted; it executed no physics and is excluded from results. Local source snapshots, manifests, traces, checkpoints and scientific receipts remain available. Timing was collected while another GPU experiment was active and is not a clean joint-epoch benchmark.

## Commands and evidence

Working directory: `exp/2026/09/21/joint-activation-material-mandible`.

```bash
CHERRIES_NAME="Joint inverse nonzero neutral prestress pilot" CHERRIES_TAGS="joint-inverse,neutral,prestress,pilot" uv run --frozen python src/20-neutral-pilot.py --output-dir data/neutral-prestress-001 --updates 8 --skin-prestress-fraction 0.01
CHERRIES_NAME="Joint inverse ten-percent prestress continuation" CHERRIES_TAGS="joint-inverse,neutral,prestress,continuation" uv run --frozen python src/20-neutral-pilot.py --output-dir data/neutral-prestress-010 --initial-checkpoint data/neutral-prestress-001/best-admissible.pt --updates 12 --skin-prestress-fraction 0.1 --learning-rate 0.003
```

Each run directory contains `summary.json`, `trace.json`, `protocol.json`, source hashes/snapshots and terminal/best-admissible checkpoints. Existing working-tree edits were preserved; no Git commit or push was made.
