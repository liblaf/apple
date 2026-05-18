#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

pybind11-stubgen \
  --output-dir "$MISE_CONFIG_ROOT/typings/" \
  --ignore-all-errors \
  'ipctk'
