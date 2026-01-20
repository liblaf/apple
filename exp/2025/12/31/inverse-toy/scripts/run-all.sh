#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

export DEBUG=1

# LR=0.02 CONFORM=false python src/10-gen.py
# LR=0.02 CONFORM=true python src/10-gen.py
# LR=0.03 CONFORM=false python src/10-gen.py
# LR=0.03 CONFORM=true python src/10-gen.py
# LR=0.05 CONFORM=false python src/10-gen.py
# LR=0.05 CONFORM=true python src/10-gen.py

# JAX_ENABLE_X64=true SUFFIX=-102k-conform python src/20-forward.py
# SUFFIX=-102k-conform python src/20-forward.py
# SUFFIX=-121k python src/20-forward.py
# SUFFIX=-26k python src/20-forward.py
# SUFFIX=-30k-conform python src/20-forward.py
# SUFFIX=-7k python src/20-forward.py
# SUFFIX=-7k-conform python src/20-forward.py

# LR=0.05 COARSEN=true ACTIVATION=2 VOLUME_PRESERVE=false python src/20-forward-muscle.py
# LR=0.05 COARSEN=true ACTIVATION=2 VOLUME_PRESERVE=true python src/20-forward-muscle.py
# LR=0.05 COARSEN=false ACTIVATION=2 VOLUME_PRESERVE=false python src/20-forward-muscle.py
# LR=0.05 COARSEN=false ACTIVATION=2 VOLUME_PRESERVE=true python src/20-forward-muscle.py
# LR=0.03 COARSEN=false ACTIVATION=2 VOLUME_PRESERVE=false python src/20-forward-muscle.py
# LR=0.03 COARSEN=false ACTIVATION=2 VOLUME_PRESERVE=true python src/20-forward-muscle.py

# LR=0.05 COARSEN=true ACTIVATION=5 VOLUME_PRESERVE=false python src/20-forward-muscle.py
# LR=0.05 COARSEN=true ACTIVATION=5 VOLUME_PRESERVE=true python src/20-forward-muscle.py
# LR=0.05 COARSEN=false ACTIVATION=5 VOLUME_PRESERVE=false python src/20-forward-muscle.py
# LR=0.05 COARSEN=false ACTIVATION=5 VOLUME_PRESERVE=true python src/20-forward-muscle.py
# LR=0.03 COARSEN=false ACTIVATION=5 VOLUME_PRESERVE=false python src/20-forward-muscle.py
# LR=0.03 COARSEN=false ACTIVATION=5 VOLUME_PRESERVE=true python src/20-forward-muscle.py

SUFFIX=-7k-conform python src/20-forward.py
