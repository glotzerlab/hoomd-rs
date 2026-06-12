// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Build script for hoomd-microstate.

fn main() {
    println!("cargo:rerun-if-env-changed=HOOMD_MAX_GHOSTS");
}
