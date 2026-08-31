# Layered 2D unreachable-target inverse results

| case | target RMS fraction | top range | high-pass RMS | control jump RMS | reference gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per-cell tensor, no regularization | 94.245% | 0.032247 | 0.000921 | 0.657484 | 0.010% |
| Per-cell tensor + neighbor smoothing | 94.424% | 0.027297 | 0.000699 | 0.033941 | -0.000% |
| One tensor shared by the muscle | 94.454% | 0.028319 | 0.000726 | 0.000000 | -0.000% |

The entire free top targets +0.1 in y. The muscle occupies only the left part of the stiff middle layer. The unrestricted projection residual is 72.11% of the target RMS, which certifies that the uniform target is outside the linear response span.

All geometry and field images are rendered separately by ParaView. This runner writes only VTK inputs, traces, and numerical summaries.
