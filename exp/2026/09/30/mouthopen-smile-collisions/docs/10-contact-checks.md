# Full FEM boundary contact checks

## Purpose

This CPU check exercises the new frictionless IPC contact module on two
approaching tetrahedral boundary surfaces. It verifies positive barrier energy
and force, continuous collision detection, a complete-boundary intersection
audit, and the analytic Hessian-vector product against a central finite
difference away from a collision-feature change.

## Command

Run from this experiment group:

```bash
CHERRIES_NAME='Full FEM boundary self-contact checks' \
CHERRIES_TAGS='mouthopen,smile,contact,ipc,cpu,validation' \
uv run python src/10-check-contact.py
```

The Cherries Profile used `Git(commit=False)`. The completed directional
derivative run is [31d4d74889a149638e5b8d201bf61e38](https://www.comet.com/liblaf/apple/31d4d74889a149638e5b8d201bf61e38).

## Results

The contact surface contained all 8 boundary vertices, 8 boundary triangles,
and 12 boundary edges. No face classes were excluded. The mean boundary-edge
length was 1.8917418554 m, giving `dhat = 0.9458709277 m`; barrier stiffness
was 0.0012 MPa, `dmin = 0`, and CCD's minimum distance was `1e-8 m`.

At the separated initial state, the barrier energy was 0.01447358099 and the
force norm was 0.05193742135. The complete-boundary intersection audit found
no intersections and 18 active IPC pairs, with minimum active distance 0.3 m.
The directed approach was limited to a CCD fraction of 0.7497253418.

The energy directional derivative and the assembled gradient agreed to a
relative error of `7.663575758e-13`. Moving the upper facing triangle toward
the lower one increased energy; the resulting force was downward on the lower
facing triangle and upward on the upper base triangle. For a deterministic,
generic perturbation with an active set held fixed through the central
difference stencil, the analytic IPC HVP relative error was
`1.300763552e-10`.

The machine-readable receipt and measurements are in
[`data/10-contact-checks-005/summary.json`](../data/10-contact-checks-005/summary.json).

## Limits

This validates the contact implementation on a small CPU fixture. It does not
establish convergence or physical validity for the full facial transition; the
transition solver must still enforce its force, CCD, fixed-boundary, inversion,
and complete-boundary-intersection gates at every accepted state.
