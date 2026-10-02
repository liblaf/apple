# MouthOpen after removal of fully prescribed tetrahedra

The user authorized a MouthOpen trial after discussing removal of every tetrahedron whose four vertices are fixed by the original `IsFixed` mask. This derived experiment retains the original historical fixture, target transfer and chin-based rigid jaw estimate. No previous modified-neutral state or fitted smile activation is inherited.

## Derived domain

`src/30-build-pruned-fixture.py` removes precisely 2,249 fully prescribed cells and compacts 760 unused fixed vertices. It retains all free vertices and every original fitting-surface vertex. Point and cell maps preserve provenance; skin `GlobalPointId` is remapped. The removed set contains 63 active cells, so activation weights and the smoothness graph are rebuilt on the retained cells. Materials, reference coordinates, and surviving boundary classifications remain unchanged.

## First numerical trial

`src/35-forward-pruned-mouthopen.py` runs the historical no-skin bulk model with physical `J=det(F)` and zero active strain. Only surviving `IsFixed` mandible vertices receive the fresh estimated rigid motion; other surviving fixed vertices remain at rest. The runtime DOF map is asserted against that rule, including free lip vertices.

The pose path scales the saved rotation vector and pivot translation from zero to one. A graph-harmonic scalar weight carries nearby tissue toward each rigid increment as an initialization; it is not an equilibrium claim. Each increment is relaxed by safeguarded Newton-CG. The experiment adds a conservative volume step bound and complete FEM boundary CCD as numerical feasibility guards; it adds no contact force or new tissue energy. The step bound uses `alpha * ||F^-1 dF||_F <= 0.8` for every retained element. It preserves positive determinant throughout each accepted linear update.

Frozen first-trial budgets and criteria:

- Absolute free-force norm at most `1e-10` in the existing MPa/metre code units; relative force tolerance zero.
- CG relative tolerance `1e-3`, at most 3,000 iterations per linear solve; at most 100 Newton steps per pose attempt.
- Pose increment initially 0.025, grows by 1.5 after acceptance up to 0.1, and halves after a failed attempt; stop below `1e-5`.
- 1,200 seconds for the pose continuation after initialization.
- Every accepted pose must have positive determinant in every retained tetrahedron, no detected complete FEM boundary intersections, and pass the original free-force criterion.
- A rejected attempt rolls back to the last accepted displacement and prescribed boundary values. Iteration or time exhaustion is a failed gate, not a converged result.

The derived boundary can have nonmanifold edges after cell deletion. Endpoint intersection checks and continuous collision detection do not establish anatomical validity or cover separately registered bone/eye obstacle meshes. The historical contact-off constitutive model remains an exploratory model. Any success is explicitly limited to the declared bulk model and FEM boundary checks; total reaction forces and the original tissue-domain mechanics are changed by deletion.

## Stop and follow-up

Activation fitting starts only after reaching the full target pose and satisfying the above gates. If the attempt fails, preserve the attempted states, diagnostics and final accepted state, report the specific failure, and do not compensate for it with activation or silently relax the criteria. Any additional solver attempt must have its own output folder and settings receipt.

Run from this experiment group with Cherries name and tags, using the repository Python environment. Each stage uses the metadata profile with automatic Git commits disabled. Sources and input hashes accompany the numerical output; reports use actual saved results after Cherries shutdown.
