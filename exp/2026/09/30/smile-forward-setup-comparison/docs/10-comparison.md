# Stage 3 Smile forward setup comparison

The exact fixed-direction Stage 3 activation was held constant in both forwards. The historical no-collision/no-skin setup reached its force tolerance. The collision/skin/prestrain solve was interrupted for the weekly-report priority and did not reach the tolerance.

| Setup | Recorded terminal energy (J) | Free force residual (N) | Inverted cells | Inverted cells (%) | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical inverse setup | 5.76042292 | 8.04232675e-05 | 89 | 0.00776264111 | Force converged |
| Collision + skin + skin prestrain | 8.30934668 | 0.0395108627 | 89 | 0.00776264111 | Interrupted; not converged |

The common force tolerance is 0.0001 N. Inversion means det(F) <= 0 over all 1,146,517 original tetrahedra. Both terminal recorded states have inverted cells. The two setups have different energy offsets; compare energy change within each setup. The interrupted new state is not an equilibrium endpoint.

The new setup uses the repaired full reference, the corrected neutral's heterogeneous StableNeoHookeanActiveMembrane, exact saved tangential prestrain, and complete source cranium/mandible/eye IPC at 1.3544 MPa. The original setup uses the exact inverse reference. See [protocol](00-protocol.md) for the full material, boundary, and solver definitions. Both forwards use safeguarded Newton at the full fixed activation.

## Saved evidence

- Requested comparison curves: `data/20-comparison/` (total energy, force, inversion, and relative-energy figures).
- Historical endpoint and complete trace: `data/10-inverse-setup-newton/final.npz`, `trace.csv`, `summary.json`.
- Longer interrupted collision/skin trace: `data/10-new-setup-newton/trace.csv` and `summary.json`; this run did not export its displacement endpoint.
- A later restart was stopped at the report chat's request. Its periodic displacement/activation checkpoint remains at `data/10-new-setup-newton-checkpointed/latest.npz`, with its own `trace.csv` and `summary.json`. Its different trajectory is not merged into the reported longer trace.
- Terminal logs and source/input SHA-256 receipts are retained for each attempt. Baseline independent CPU checks confirmed source identity, fixed nodes, and 89 inversions with minimum det(F) -1.9070241407769235. A comparable CPU endpoint audit of the longer interrupted new run is unavailable because its endpoint was not exported.

## Commands

Run directory: `exp/2026/09/30/smile-forward-setup-comparison`. Interpreter: `.venv/bin/python`.

```bash
CHERRIES_NAME='Smile Stage 3 safeguarded forward in inverse setup' CHERRIES_TAGS='smile,stage3,fixed-activation,forward,newton,no-collision,no-skin,setup-comparison' .venv/bin/python src/10-run-forward.py --branch inverse-setup --output 10-inverse-setup-newton
CHERRIES_NAME='Smile Stage 3 safeguarded forward with collision skin and prestretch' CHERRIES_TAGS='smile,stage3,fixed-activation,forward,newton,collision,skin,prestrain,setup-comparison' .venv/bin/python src/10-run-forward.py --branch new-setup --output 10-new-setup-newton
DEBUG=1 CHERRIES_NAME='Stopped Smile forward comparison plots' CHERRIES_TAGS='smile,stage3,forward,interrupted,energy,force,inversion' .venv/bin/python src/20-plot-comparison.py --new-setup data/10-new-setup-newton --output data/20-comparison
```

The original forward runs used normal Cherries profiles with automatic Git commits disabled. The [baseline Comet run](https://www.comet.com/liblaf/apple/130cfa41d34d4fd7994e283ef0dbe991) and [interrupted contact Comet run](https://www.comet.com/liblaf/apple/6258ba40e62b4f868b8ae455bbb3ca6a) retain their metadata. Plot finalization uses the local debug profile to finish promptly after the stop request. No further solve was launched after that request.
