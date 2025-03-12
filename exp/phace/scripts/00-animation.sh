#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

readarray -t ACTIVATIONS < <(seq 1.0 1.0 10.0)
DATA_DIR="./data"

for activation in "${ACTIVATIONS[@]}"; do
  input="$DATA_DIR/inputs/$activation.vtu"
  output="$DATA_DIR/outputs/$activation.vtu"
  python src/00-gen-activation.py --activation "$activation" --output "$input"
  python src/01-simulate.py --input "$input" --output "$output"
done
