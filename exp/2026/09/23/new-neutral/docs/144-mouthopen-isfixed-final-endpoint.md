# Corrected MouthOpen inverse endpoint audit

## Purpose

Independently rebuild the corrected `IsFixed` inverse physics and evaluate the
saved `inverse-mouthopen-isfixed-003` endpoint. The audit does not run a
forward or inverse solve.

## Command

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME='Independently audit continued corrected MouthOpen endpoint' \
CHERRIES_TAGS='mouthopen,isfixed,inverse,independent-audit,active-strain,collision,continuation' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u \
  src/144-audit-mouthopen-isfixed.py \
  --run-dir data/inverse-mouthopen-isfixed-003 \
  > tmp/144-audit-mouthopen-isfixed-003.log 2>&1
```

Comet: <https://www.comet.com/liblaf/apple/09744254ced24ebd967406ad4f603c15>.

## Verification

The rebuilt model matches the runtime `IsFixed` boundary: 27,036 of 228,660
FEM vertices are fixed, with 166,155 fixed and 604,872 free degrees of
freedom. All 3,408 lip vertices are free. Endpoint prescribed values match
the rebuilt boundary exactly.

The endpoint restores 288,235 active Raw6 rows and retains the saved
multiplicative active-strain skin prestretch arrays. Its fixed IPC stiffness
is 1.3544 MPa.

The free-force norm is 0.009180265 N, below the 0.01 N absolute gate.
Contact is numerically valid, has no intersections, and has an 83.913789 µm
minimum active gap. The independent float64 CPU check finds zero inverted
tetrahedra, with minimum det(F) 0.1000000036.

The area-weighted MouthOpen skin RMS is 7.429377 mm. Independent force,
RMS, and minimum-det(F) values match the saved final row.

The continuation ended at its configured nine accepted updates, recorded as
`finite_budget_exhausted`; the endpoint is a valid forward state but not an
inverse-converged optimum. Its mandible pose is constrained at the all-fixed
tetrahedron quality floor, so the final accepted pose increment is zero while
the active-strain update continues.

## Artifacts

The audit receipt with input hashes is
`data/inverse-mouthopen-isfixed-003/independent-audit.json`. The audit source
is `src/144-audit-mouthopen-isfixed.py` and its log is
`tmp/144-audit-mouthopen-isfixed-003.log`.
