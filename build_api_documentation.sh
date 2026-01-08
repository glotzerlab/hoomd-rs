#!/bin/bash

# Build documentation with `--no-deps` so that the sidebard is not polluted
# with hundreds of extra crates and the build time is kept reasonable.
# HOWEVER: `cargo doc` fails to build packages in the correct order when
# `--no-deps` is set (e.g. it may build `hoomd-mc` before `hoomd-geometry`).
# rustdoc works by examining the current target/doc directory and adding new
# crates to what is already present. The solution is to build one crate at
# a time, buiding all dependents first.

set -euo pipefail

export RUSTDOCFLAGS="--html-in-header katex.html"
for package in gsd linear-algebra simulation utility rand vector manifold spatial \
               geometry microstate interaction mc bevy
do
  cargo doc --package hoomd-$package --lib --no-deps
done
