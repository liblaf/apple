# IsFixed runtime audit

`137-audit-isfixed-runtime.py` completed an actual corrected FEM/IPC model
construction. It did not run a new equilibrium.

The original FEM has 228,660 nodes. The authoritative `IsFixed` set has 27,036
nodes, replacing the legacy 29,601-node support union. This releases 2,565
original FEM nodes: 1,200 carrying the Cranium group label and 1,365 carrying
the Mandible label. The model retains 28,349 appended rigid obstacle nodes, and
the jaw prescription applies to the 6,145-node intersection of `IsFixed` and
Mandible. All 3,408 `IsLip` nodes are free; 366 released nodes carry `IsLip`.

The rebuilt reference passed the 100 micrometre clearance target. Its active
minimum distance was 100.093782531 micrometres, with zero intersections and a
numerically valid IPC state.

The saved endpoints were evaluated under the corrected boundary map only. They
are not inherited equilibria. The old neutral's free-force norm is 0.2761954 N,
and the old MouthOpen trial's is 4.706661 N. The prior 0.0092505 N and
0.0099602 N values omitted reactions on nodes that are now released, so they do
not establish equilibrium for the corrected map. The released-node terms are
0.2760405 N and 4.7066505 N, respectively.

Archived source hashes were verified against the receipt. They preserve the
audited implementation independently of later edits: `137-audit-isfixed-runtime.py`, `joint_physics.py`,
`joint_full_skull_contact.py`, `reference_rebase.py`, and
`profile_input_binding.py`.

The focused CPU regression coverage is
[`test_joint_isfixed_boundary.py`](../../../../../../tests/experiments/test_joint_isfixed_boundary.py):
two tests verify that only `IsFixed` selects original FEM DOFs, the jaw uses the
fixed Mandible subset, and full-skull rigid obstacle nodes remain appended and
prescribed.

The completed command recorded in
[`tmp/isfixed-runtime-audit-001.log`](../tmp/isfixed-runtime-audit-001.log) was:

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME='Verify IsFixed constraints on actual new neutral and MouthOpen states' \
CHERRIES_TAGS='neutral,isfixed,boundary-correction,audit' OMP_NUM_THREADS=4 \
.venv/bin/python -u src/137-audit-isfixed-runtime.py \
  > tmp/isfixed-runtime-audit-001.log 2>&1
```

The receipt is [receipt.json](../data/isfixed-runtime-audit-001/receipt.json).
The completed Comet record is
[14e2183061384366b6eaecf4d52fba60](https://www.comet.com/liblaf/apple/14e2183061384366b6eaecf4d52fba60).
