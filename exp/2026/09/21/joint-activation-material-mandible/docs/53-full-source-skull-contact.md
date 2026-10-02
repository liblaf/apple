# Complete source-skull contact adapter

The versioned adapter in `src/joint_full_skull_contact.py` adds every registered
source cranium and mandible vertex to the mechanics displacement vector as a
fixed node without adding FEM cells. Original FEM node IDs, tetrahedra,
constitutive potentials, material arrays, and reported displacements remain
unchanged. Source cranium displacement is zero. Source mandible displacement is
the same differentiable six-DoF rigid transform used for the original fixed
mandible support.

This representation reuses the existing IPC implementation without losing jaw
derivatives. The collision Hessian contains the soft-to-source-mandible blocks;
the existing implicit backward returns their fixed-DOF reaction, and PyTorch
chains that reaction through the rigid pose map. The accepted seed pose is an
explicit `solve` input, so boundary CCD moves the original mandible support and
the complete source mandible together from the accepted seed to the proposal.
The source cranium remains stationary.

The collision mesh contains the audited pure-soft FEM surface and every source
bone triangle. A binary vertex-patch filter permits only soft-bone pairs. It
disables soft-soft, source self-contact, and cranium-mandible pairs. No source
triangle or attachment-neighborhood exclusion is applied. Consequently, this
adapter does not claim that the source cranium and mandible are mutually
intersection-free or provide a bone-bone contact law; those are separate
geometry/guard requirements.

`FullSkullJointPhysics` calls the frozen `JointPhysics` constructor without its
partial FEM contact, replaces its model with an extended `DofMap`, and reuses
the same `WarpModelAdapter`. All appended coordinates are fixed, while the
original free/fixed index sets are preserved. Losses, target fitting, metrics,
and returned solve results use only the original FEM node slice.

Construction and production admission are separate. The loader accepts
`data/full-skull-initialization-audit-001/geometry.npz` as exact completeness
evidence and binds SHA-256
`a706952109c4ad67a72692202412a827bc8d57ef1ccbd393dfa70a4d4190a6aa`.
That audit reports `full_skull_contact_admitted=false`; its reference surface
still intersects the complete bones. Production physics therefore additionally
requires a successful `joint-full-skull-contact-admission-v1` receipt bound to
the same geometry and audit. The receipt must prove an intersection-free
all-FEM-node initialization, unchanged original fixed nodes, unchanged source
coordinates, all source triangles retained, and zero excluded source triangles.

## CPU validation

Run from the experiment directory:

```bash
DEBUG=1 \
CHERRIES_NAME='Full source skull contact adapter CPU validation v2' \
CHERRIES_TAGS='joint-inverse,full-skull,contact,cpu,validation' \
uv run --frozen python src/53-validate-full-skull-contact.py \
  --output-dir data/full-skull-contact-adapter-validation-002
```

The synthetic fixture is deliberately nondegenerate and contact-active. The
validation passed:

- two central energy directional checks, with maximum relative error `0.1511%`;
- exact IPC Hessian-vector products, with maximum relative error `8.73e-6`;
- complete contact-force translation balance, relative residual `2.49e-17`;
- the appended-mandible fixed-Hessian jaw pullback, relative error `7.18e-10`;
- a prescribed source-mandible crossing rejected at CCD fraction `0.47375`;
- exact soft-bone-only filtering, unchanged original free indices, appended
  fixed coordinates, and fail-fast rejection of malformed admission or any
  source-triangle exclusion.

The inherited PNCG `hess_quad` is a documented Gauss-Newton approximation; the
checked Hessian-vector product used by Newton and adjoints is the exact IPC
potential Hessian. This CPU check does not run an FEM equilibrium, admit the
real intersecting reference geometry, or validate anatomical source-bone
placement.

[Receipt](../data/full-skull-contact-adapter-validation-002/summary.json) has
SHA-256 `27bb2c4ce619355fe7c73ee3f699753a227664be455f5e9943790a002432795b`.
Run 001 is preserved as a failed strict synthetic derivative fixture; it is not
success evidence.
