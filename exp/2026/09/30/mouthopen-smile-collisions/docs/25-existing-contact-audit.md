# Independent audit of fixed-activation contact trial

The contact trial **blocked during neutral relaxation** after its declared 1,200-second wall budget. It accepted no equilibrium and wrote no animation frames. The existing MouthOpen and Smile activation tensors were copied exactly from the parent transition, but the trial never reached a state using the full MouthOpen tensor or jaw pose. The saved failed solver displacement is a diagnostic state, not an accepted result.

## Command and receipts

The numerical run was `src/23-contact-existing-activation.py` with `initialization=neutral-continuation`, standard `IPC` collision stencils, and the full extracted FEM boundary with fixed–fixed primitive pairs exempt. Its terminal [`summary.json`](../data/23-contact-existing-activation/summary.json) reports `status=blocked`, `phase=neutral`, `elapsed_seconds=1200.7487602559995`, `provenance_verified=true`, zero accepted checkpoints, and zero frames. The failure message is “declared forward wall budget exhausted.” The last logged Newton iteration was 69 with a reported force around `1.81e-8`, above the `1e-10` acceptance threshold; the failed-state file does not carry an independent accepted-force receipt.

The CPU audit was run from `exp/2026/09/30/mouthopen-smile-collisions`:

```bash
CHERRIES_NAME='Audit existing activation contact results' CHERRIES_TAGS='collision,ipc,activation,audit,cpu' .venv/bin/python -u src/25-audit-existing-contact.py > logs/25-audit-existing-contact-terminal.log 2>&1
```

It passed Ruff and compilation, completed Cherries shutdown, and recorded [Comet 2e620fba](https://www.comet.com/liblaf/apple/2e620fbabdf54c49975a8ccb3641fa87). The [audit summary](../data/25-existing-contact-audit/summary.json) records the complete checks and failed-state hash. The noncommitting Cherries profile recorded Git SHA `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`.

## Verified evidence

The audit rehashed all seven declared numerical inputs and all 118 frozen source records, checked each frozen record against its live source, and verified that the copied `mesh.npz` and `endpoints.npz` are byte-identical to the completed parent transition outputs. It matched mesh points, tetrahedra, and active-cell IDs to the pruned volume; `FixedMask` and `IsFixed` agree. The saved endpoint tensor arrays remain available for the declared formulas `S=alpha*S_MouthOpen` during initialization and `S=(1-beta)*S_MouthOpen+beta*S_Smile` during the transition. No accepted checkpoint exists on which to verify an applied nonzero activation field.

The saved `failed-solver-state.npz` has exactly zero displacement error on every prescribed vertex at neutral pose. Independently recomputed deformation Jacobians found **21 inverted tetrahedra** and minimum `J=-1.5014334659516642`. Fresh IPC geometry checks found **no scoped or unfiltered boundary self-intersections** in this failed state. Those geometry observations do not satisfy the missing free-force convergence gate. The output explicitly marks `experiment_completed=false`, `accepted_count=0`, `frame_count=0`, and `accepted_equilibrium=false`.

The trial therefore provides a collision-clear, partially relaxed neutral diagnostic state, but no contact-equilibrated MouthOpen or Smile pose and no collision-enabled animation. It does not show that the saved activation tensors can or cannot reach a feasible full-pose contact equilibrium under this model.
