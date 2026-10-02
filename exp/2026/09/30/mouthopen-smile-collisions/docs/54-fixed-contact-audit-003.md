# Independent fixed-reference contact audit

Numerical status: **completed**. Independently verified 56 accepted checkpoints and 121 frames.

For each saved state, this audit rechecks the saved full displacement, prescribed physical and appended rigid nodes, tetrahedron Jacobians, and scoped IPC surface intersections on CPU. It verifies every input and frozen source hash and confirms the Smile and MouthOpen activation arrays are byte-identical to the prepared source. Fresh GPU free-force residuals were recomputed for every saved state with the exact tensor blend, fixed boundary, and contact model.

Result: `verified_completed`. See `exp/2026/09/30/mouthopen-smile-collisions/data/54-fixed-contact-audit-003/summary.json` for per-state hashes and measurements.
