# IsFixed boundary-policy correction

The published new-neutral results used the wrong Dirichlet boundary rule. They
clamped the union of the original `IsFixed` mask and every `GroupId=Cranium`
or `GroupId=Mandible` node. The authoritative clamp is the original
`IsFixed` mask alone.

The frozen input contains 27,036 `IsFixed` nodes. The old policy clamped
29,601 nodes, adding 2,565 nodes that are `IsFixed=false`: 1,200 labelled
Cranium and 1,365 labelled Mandible. Every one of the 366 lip-labelled nodes
that was added by the old policy is `IsFixed=false`; no lip-labelled node is
`IsFixed=true`.

`GroupId` remains useful only after applying `IsFixed`: `IsFixed ∩ Mandible`
receives the rigid jaw displacement, while the remaining `IsFixed` FEM nodes
stay at zero. Appended source cranium, mandible and eye vertices remain rigid
obstacle degrees of freedom. This corrects the boundary assignment without
changing material or contact parameters.

Earlier anatomical justification for clamping all group-labelled bone nodes is
retracted. The old numerical artifacts and receipts are preserved unchanged,
but their results are invalid under the requested `IsFixed` policy. The
invalidation covers forward, repaired-reference, rigid-pose, inverse-trial and
damped-adjoint outputs listed in
[`receipt.json`](../data/isfixed-policy-invalidation-001/receipt.json).

The correction must rebuild the model with the intended fixed map and
revalidate the geometrical reference before rerunning any result that needs a
physical claim. The existing coordinate-clearance evidence may be reused only
when it is revalidated under that corrected model. Historical numerical
receipts remain unchanged.
The published old-policy pages carry the same visible invalidation notice.

## Corrected fixed-boundary kinematic audit

A CPU-only audit applied the intended `IsFixed` mask to the archived
`chin-rigid-pose-001` pose. It used 27,036 intended FEM fixed nodes, with
6,145 jaw-routed nodes from `IsFixed ∩ Mandible`, instead of the archived
29,601-node `FixedMask`. Of 2,249 tetrahedra whose four vertices are intended
fixed nodes, 168 invert under that archived pose and the minimum ratio is
`J = -79.28334659163696`.

This confirms that the earlier count of 168 forced tetrahedra was not created
by adding the 2,565 group-labelled nodes. It remains a kinematic finding about
the archived pose and repaired-reference coordinates, not a forward solve or
an endorsement of the invalidated old policy. The receipt also records the
old-mask comparison separately:
[`isfixed-fixed-boundary-audit-002`](../data/isfixed-fixed-boundary-audit-002/receipt.json).

The superseding audit derives `IsFixed` from the volume field and separately
poses all Mandible-labelled nodes only for the archived old-policy comparison.
It completed after Cherries shutdown:
[Comet run](https://www.comet.com/liblaf/apple/3564013571b94654941b724224e6cce1).
