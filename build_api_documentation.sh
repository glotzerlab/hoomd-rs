#!/bin/bash

# Build documentation with `--no-deps` so that the sidebard is not polluted
# with hundreds of extra crates and the build time is kept reasonable.
# HOWEVER: `cargo doc` fails to build packages in the correct order when
# `--no-deps` is set (e.g. it may build `hoomd-mc` before `hoomd-geometry`).
# rustdoc works by examining the current target/doc directory and adding new
# crates to what is already present. The solution is to build one crate at
# a time, buiding all dependents first.

set -euo pipefail

export RUSTDOCFLAGS="--html-in-header google_analytics.html --html-in-header katex.html"
for package in derive linear-algebra simulation utility rand vector gsd manifold spatial \
               geometry microstate interaction mc bevy md
do
  cargo doc --package hoomd-$package --lib --no-deps
  cp katex.html hoomd-$package/
  cp README.md hoomd-$package/
done
