# Fixed-control MouthOpen equilibrium refinement

Run007 reached its 100 retained-tet inversion allowance while its free force
approached the original 1e-8 stopping threshold. Tiny accepted steps performed
no correction; slightly larger trials triggered nonlinear correction and
finished with 101 inversions. See `156-mouthopen-descent-projection.md` and
run007's compact `force-threshold-diagnostic.json`.

Run007 was stopped with a verified-PID SIGINT after local update 23, both Adam
counters 237. Process and Cherries shutdown completed with exit 130. Saved
endpoint, checkpoint and summary arrays/hashes/counters agreed. The subsequent
independent 146 audit passed under the original policy: RMS
2.106035576781888 mm, force 0.009999352723935217 N, 100 retained inversions.
Its full original tet boundary with bones/eyes and fit/force history was rendered
with 147; fifteen published assets were byte-verified and the front image viewed.

The additive 157 diagnostic holds each source's active strain and jaw pose exactly
fixed and runs the original collision-on physical corrector directly. It changes
only the internal stopping target to raw force 1e-9 (0.001 N). Original admission
is reported separately: force at most 1e-8 (0.01 N), at most 100 retained inversions,
inverted rest-volume fraction at most 1e-4, and the original collision checks.
Exact skin arrays, IsFixed-only DOFs, all-fixed-tet exclusion and IPC stiffness
1.3544 MPa are verified in a fresh rebuild. No inverse checkpoint is created or
adopted, and no moments or counters are changed.

Each diagnostic case has a 600-second corrector budget. Budget exhaustion is a
failed diagnostic, not convergence. Rebuild and output audit time are separate.
The script first tests audited run007; it tests audited run006 only if the first
refinement does not meet both the tighter target and original admission gates.
All retained old/new determinant values and changed original cell IDs are saved,
along with full candidate displacement, force/contact receipts, objective and
displacement-change metrics. A failed corrector state is labelled unadmitted.

Command from `exp/2026/09/23/new-neutral`:

```bash
TMPDIR=exp/2026/09/23/new-neutral/tmp/run007-runtime \
CHERRIES_NAME='Diagnose fixed-control MouthOpen equilibrium refinement' \
CHERRIES_TAGS='mouthopen,isfixed,collision,skin-prestrain,force-threshold,diagnostic' \
OMP_NUM_THREADS=4 .venv/bin/python -u \
  src/157-polish-mouthopen-equilibrium.py \
  > tmp/157-polish-mouthopen-equilibrium-001.log 2>&1
```

Outputs: `data/mouthopen-equilibrium-polish-001/`. Source snapshots and job
identity are saved there. The authoritative continuation pointer identifies this
as a fixed-control diagnostic. The live page retains stopped run007 receipts and
links diagnostic status. Do not launch another GPU solver while it runs.

CPU compilation and Ruff checks passed before launch. No commits or pushes.
Results must be read from completed per-case receipts; neither tighter residual
convergence nor fixed-state differentiation establishes inverse stationarity.

## Completed diagnostic

Both numerical cases and Cherries shutdown finished normally, exit 0.
[Comet diagnostic](https://www.comet.com/liblaf/apple/9cf04cc8562b4a13a927e215e9601e56).

| Source | Refined RMS (mm) | Refined force (N) | Inversions | Corrector time (s) | Original gates |
| --- | ---: | ---: | ---: | ---: | --- |
| run007, counters 237/237 | 2.0918896331022063 | 0.0009833345925498126 | 101 | 46.2529 | Fail inversion count |
| run006, counters 214/214 | 2.1417899088454013 | 0.0009667553070054522 | 97 | 51.6234 | Pass |

Run007 newly inverted only original tet 674827. Its determinant changed from
+1.1339821360e-6 to -1.0499400799e-4. This fat tet has three fixed vertices and
one free vertex, so it is not eligible for all-fixed exclusion. The fixed face
stayed fixed while its free apex moved approximately 0.270 micrometers across
that face. The rejected refinement remains diagnostic; it was not adopted.

Run006 retained exactly the same set of 97 inverted cells. Its maximum nodal
correction was 0.112683 mm, RMS nodal correction 0.0268543 mm, and inverted
rest-volume fraction remained 1.2549996711732493e-5. This supplies a feasible
starting displacement at the tighter internal tolerance, with original controls
and optimizer state unchanged.

The proposed continuation uses the original run006 checkpoint for q, pose,
moments and counters, and a separately hash-bound refinement endpoint for its
initial displacement. It records the initial RMS change as fixed-control
equilibrium refinement, not as an accepted inverse update. Subsequent solves
use internal atol 1e-9 while the physical acceptance contract remains 1e-8.
The descent-aware pose QP and all collision/inversion gates remain in force.

The stopped run007 endpoint is still the latest independently audited inverse
preview; its audit and review are recorded at
[audit Comet](https://www.comet.com/liblaf/apple/1a86ec6621554921a5cd6aad303d2b95)
and [review Comet](https://www.comet.com/liblaf/apple/3399f895d6904ce6886f72030a0c7058).

## Refined continuation running (run008)

`src/158-continue-mouthopen-refined-equilibrium.py` now runs in additive directory
`data/inverse-mouthopen-coupled-008`. Its initialization binds the original
run006 checkpoint and the successful run006 refinement by SHA-256. The original
q, pose, Adam moments and counters 214/214 are restored; only the initial
displacement comes from the refinement. Startup revalidated RMS
2.141789908845402 mm, force 0.0009667553070054522 N and 97 inversions.

The first four accepted updates used alpha 0.25, 0.5, 1 and 1. Update 4 reached
RMS 2.1329402242604956 mm with force 0.0009898191307541105 N, 98 inversions and
inverted rest-volume fraction 1.2028741624839149e-5. Jaw pose magnitudes were
8.597245942953176 degrees and 4.421212119647133 mm. These are live solver
receipts, not an independently audited finished endpoint or an inverse
stationarity certificate. The tighter corrector has restored useful steps at
startup; sustained progress remains to be checked.

Command:

```bash
TMPDIR=exp/2026/09/23/new-neutral/tmp/run008-runtime \
CHERRIES_NAME='MouthOpen continuation from refined equilibrium' \
CHERRIES_TAGS='mouthopen,isfixed,active-strain,skin-prestrain,collision,descent-projection,internal-equilibrium-refinement,continuation' \
OMP_NUM_THREADS=4 .venv/bin/python -u \
  src/158-continue-mouthopen-refined-equilibrium.py \
  > tmp/158-continue-mouthopen-coupled-008.log 2>&1
```

[Comet run008](https://www.comet.com/liblaf/apple/0eb9c66fb3274c4388c83da394517722).
The active state pointer and heartbeat identify run008. Its exact PID, start
ticks, boot ID and TMPDIR are saved in `job.json`. The live tailnet page binds
run008 summary/progress; both assets were checked against local bytes. The
audited full-surface page continues to show run007.

The review script now checks refinement lineage and shows the fixed-control
RMS correction at the same optimizer iteration, rather than counting it as an
inverse update. Targeted optimizer, projection, convergence and tet-policy tests
passed (11 tests); changed Python files passed Ruff and syntax checks. No loaded
solver sources are edited during run008.
