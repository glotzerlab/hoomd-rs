// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Provides a collection of fast, high-quality random number generators (RNGs) for the HOOMD ecosystem.
//!
//! This crate offers implementations of several modern RNG algorithms, including:
//!
//! - [`SFC64`]: The "Small Fast Chaotic" counter based RNG, which should used by default
//!   in most cases. It is extremely fast, has low latency, and is very
//!   statistically sound -- we have validated that streams are independent and
//!   uncorrelated for >2TB of data per seed.
//! - [`AESRand`]: An AES-based RNG (currently only available on some aarch64
//!   platforms). This can be even faster than SFC64, but has a smaller state that
//!   makes it less suitable for highly parallel applications.

/// Utility functions for random number generation.
pub(crate) mod util;

#[cfg(feature = "extras")]
pub mod threefry2x64;

#[cfg(feature = "extras")]
/// Squares PRNGs, which have good performance but worst statistics than SFC64.
///
/// Requires the feature `extras`.
pub mod squares;

/// Structs and implementations for the SFC64 PRNG.
mod sfc;
pub use sfc::SFC64;

mod counter;
pub use counter::Counter;

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
pub use aesrand::AESRand;
