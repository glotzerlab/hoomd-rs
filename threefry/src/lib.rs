// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Asdf.

/// asdf
pub(crate) mod backends;
pub use backends::C240;

mod threefry2x64;
pub use threefry2x64::ThreeFry2x64Rng;

mod squares;
pub use squares::{Squares64, Squares128};

mod sfc;
pub use sfc::SFC64Rng;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "aes"
))]
mod aesrand;
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "aes"
))]
pub use aesrand::AESRandRng;
