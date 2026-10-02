# Fixed rigid-eye IPC validation

## Purpose

Validate the source-eye extension before the eye-inclusive neutral PNCG run. The
2,560 complete source-eye triangles are appended as fixed obstacle nodes in the
same IPC mesh as the complete cranium and mandible. Only soft-versus-rigid
pairs are enabled.

## Command

```bash
DEBUG=1 CHERRIES_NAME='Rigid eye IPC validation' CHERRIES_TAGS='eyes,contact,validation' \
  uv run python src/85-validate-rigid-eye-contact.py \
  --output-dir data/rigid-eye-contact-validation-004
```

## Results

The CPU synthetic contact check passed with barrier energy
`4.7493675316979015e-05`, nine active pairs, directional-gradient relative error
`9.006979956932976e-09`, and exact-IPC HVP relative error
`2.0737743721991885e-09`. A trial that crosses the eye was limited by CCD to
`0.3323974609375`; rigid-rigid pairs were excluded and all appended eye DOFs
were fixed.

The real CUDA model rebuilt successfully from frozen-neutral-004 and retained
every baseline material tensor bit-for-bit. It contains 228,660 FEM nodes,
27,051 full-skull obstacle nodes, and 1,298 fixed eye nodes (257,009 total).
The eye displacement remained exactly zero under a nonzero six-DoF mandible
pose.
The bound eye source is melon `20-eye.ply`, SHA-256
`99b1859beeac17f4490998de103e06bdd864c218041c92f37944779432f8c202`; its
prepared NPZ SHA-256 is
`786769a3450b869aab81358be6e90d92d52132545ed966cf622996e6c625b9ce`.

Outputs: [summary.json](../data/rigid-eye-contact-validation-004/summary.json).

## Limitation

This is an adapter admission check. The separate eye-initialization artifact
must clear the real neutral soft-eye overlap before the forward solve; the
forward run is the equilibrium and real-contact admission evidence.
