#!/bin/bash

set -euo pipefail

export CARGO_TERM_QUIET=true

for package in derive linear-algebra simulation utility rand vector gsd manifold spatial \
               geometry microstate interaction mc bevy
do
  echo '------------------------------------------------------'
  echo "hoomd-${package}:"
  cargo public-api diff latest -p "hoomd-${package}" -sss
done
