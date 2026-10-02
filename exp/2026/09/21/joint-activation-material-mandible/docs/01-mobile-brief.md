# Joint inverse experiment

**Tuesday: inspect a valid final joint-optimization trend.** Preparation states
must be converged and visually reviewed; the final inverse need not converge.

The selected neutral is the converged full-skull, heterogeneous-skin forward
state. Its terminal displacement and the prescribed per-face skin resultant are
frozen. This is an adopted equilibrium state, not a stress-free mesh rebase:
the original FEM reference, deformation gradients, membrane tangent frames, and
complete-source skull contact remain the physical reference.

For each transferred expression displacement `d_e`, the target is
`x_neutral + d_e`. Equilibrium continues to solve absolute displacement against
the original FEM reference and reports `u - u_neutral` as expression-relative
motion. Baseline stress is no longer fitted. The remaining proposed variables
are one global skin-stiffness multiplier, six PSD activation-stress coordinates
per active muscle tetrahedron per expression, and six mandible-pose coordinates
per expression: **6,917,665 scalars** for four expressions.

The strong within-muscle activation smoothness remains mandatory. The global
skin multiplier has no spatial degree of freedom, so it receives its scalar
prior rather than a fictitious spatial smoothness force.

The frozen-neutral bundle must pass its zero-increment force, full-skull IPC,
CCD, fixed-boundary, and target-transfer checks before an inverse runner uses
it. Changing skin stiffness while keeping the baseline resultant fixed can
relax the adopted neutral; that is an expression solve effect, not a new
baseline-prestress optimization.

The legacy Shared20/Spatial80 baseline-stress pipelines are superseded. Joint-
runner integration, expression/jaw derivative checks, fixed-parameter control,
and the final simultaneous trajectory remain pending.

[Current results and visual evidence](/progress/#adopted-neutral) · [Technical design](/full/)
