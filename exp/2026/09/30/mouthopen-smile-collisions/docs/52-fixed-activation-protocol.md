# Repaired reference with saved activation

The user requested collision using the existing solved Smile and MouthOpen activation. This adaptation transfers the earlier clearance repair through exact point/cell identity maps, rebuilds the constitutive reference, and solves displacements with the saved endpoint tensors. It does not fit activation again.

## Model

- `data/50-fixed-reference`: 227,900 physical points, 1,144,268 retained tetrahedra, and 288,172 active cells. Both endpoint S arrays are byte-preserved in `endpoints.npz`. The fixed point coordinates are identical to the original fixture; free reference coordinates change by at most 0.904 mm.
- Keep the original fat, aponeurosis and muscle materials, without a skin membrane. The repaired coordinates change bulk rest gradients and integration volumes. Keeping S components exactly preserves the requested controls but does not claim equivalent stress under the changed reference.
- Preserve the original IsFixed mask. Apply the saved mandible pose only to IsFixed mandible points and complete source mandible geometry. Complete source cranium and eyes remain fixed.
- Contact comprises pure-soft FEM boundary triangles against complete source cranium, mandible and eyes. Mixed attachment faces and duplicate FEM bone faces are excluded. Tissue self-contact and rigid-rigid contact are outside this experiment's scope.
- Standard IPC, physical area-weighted barrier, constant stiffness 1.3544 MPa, activation distance 100 micrometers, minimum admissible distance 10 nanometers, TightInclusion tolerance 0.1 nanometers, and 100,000 CCD iterations. When CCD limits a solver step, PNCG takes 0.9 of that step and Newton takes 0.95; a full CCD step remains unchanged. The prescribed jaw carry requires a full CCD step. The constant stiffness is the successful corrected-neutral terminal coefficient.

## Forward solve and acceptance

Start from zero displacement and zero activation on the repaired reference. Continue activation and prescribed jaw to the MouthOpen endpoint, then solve 121 cosine-spaced states from MouthOpen to Smile with S(beta) = (1-beta) S_MouthOpen + beta S_Smile and the jaw pose scaled by (1-beta). Existing harmonic weights supply only an initialization predictor; prescribed coordinates, CCD, and fresh equilibrium checks determine acceptance.

Each forward solve uses PNCG until the existing force-window handoff, followed by safeguarded Newton with exact bulk curvature and PSD contact search curvature. The GPU contact operator avoids assembling the complete bulk/contact sparse matrix. Every accepted state requires absolute free force at most 1e-8 MPa m^2 (0.01 N), prescribed coordinates exact within 1e-12 m, valid scoped contact, and finite determinants. The user's allowance of a few inverted cells is represented by a cap of 0.1% of retained cells; inverted states remain explicitly exploratory, with no claim of mechanical validity.

The run stops at the declared one-hour compute budget or if interval subdivision reaches its stated limit. It saves all accepted checkpoints and any terminal diagnostic state. An original 52 resume requires matching inputs, contact policy, numerical configuration, endpoint tensors and source hashes; only output and resume paths and the per-invocation wall budget may differ. The [75 continuation](75-continuation-protocol.md) separately permits the documented CG iteration cap change from 1,000 to 3,000 while retaining the physical model and acceptance gates. No animation is accepted unless all 121 states pass the gates.

## Execution and verification

From this experiment directory:

```bash
CHERRIES_NAME="Repaired reference saved activation with bone eye contact" CHERRIES_TAGS="mouthopen,smile,fixed-activation,repaired-reference,contact" .venv/bin/python src/52-run-fixed-activation-contact.py
```

Read `data/50-fixed-reference/summary.json` and `data/51-fixed-reference-contact-check/summary.json` for mapping and zero-rest-contact evidence. Run a bounded GPU operator check before the continuation. Numerical inputs and imported sources are frozen and rehashed. After completion, audit checkpoint hashes, S interpolation, free force, prescribed values, contact clearance, intersections and inverted-cell counts, then render the full tetmesh boundary and inspect the video. A partial run may produce a clearly labeled diagnostic image only.
