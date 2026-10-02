# Saved face render contract

`src/50-render-face-comparisons.py` renders **only supplied saved VTU states**. Each case must name a rest/reference VTU and a final endpoint VTU with identical point and cell topology. The optional history is either a literal ordered VTU list or a VTK `.series` JSON file. It produces a video only from those explicit states, one frame per state; it never creates intermediate geometry.

The manifest may name a cell-data material array and an optional plane. The renderer produces full-head, mouth-closeup, and actual cutaway images, plus an MP4 for an explicit history. It computes a shared orthographic camera/scale from all reference and endpoint bounds. A receipt records source hashes, topology, camera, cut plane, and output hashes.

The generated `viewer.html` is a Tailnet-friendly asset browser for the exact static renders, video, PVSM, receipt, and VTU sources. It does not present itself as browser-native mesh interaction: no local Three.js or vtk.js dependency is available. A later interactive viewer can consume the same receipt and files after a 3D library is deliberately bundled.
