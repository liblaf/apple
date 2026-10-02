# Independent fixed-reference contact audit

Numerical status: **blocked**. Independently verified 82 accepted checkpoints and 28 frames.

For each saved state, this audit rechecks the saved full displacement, prescribed physical and appended rigid nodes, tetrahedron Jacobians, and scoped IPC surface intersections on CPU. It verifies every input and frozen source hash and confirms the Smile and MouthOpen activation arrays are byte-identical to the prepared source. Fresh GPU free-force residuals were recomputed for every saved state with the exact tensor blend, fixed boundary, and contact model.

Result: `verified_incomplete`. See `data/54-fixed-contact-audit/summary.json` for per-state hashes and measurements.
