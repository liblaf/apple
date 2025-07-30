#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

stubgen --output typings/ --package warp
py_file="$(python -c 'import warp; print(warp.__file__)')"
pyi_file="${py_file}i"
awk '
39 <= NR && NR <= 150 { next }
{ print $0 }
' "$pyi_file" >> typings/warp/__init__.pyi
