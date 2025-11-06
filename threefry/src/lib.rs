// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Provides a collection of fast, high-quality random number generators (RNGs) for the HOOMD ecosystem.
//!
//! This crate offers implementations of several modern RNG algorithms, including:
//!
//! - `SFC64Rng`: The "Small Fast Chaotic" counter based RNG, which should be considered
//! the default in most cases. It is extremely fast, has low latency, and is very
//! statistically sound -- we have validated that streams are independent and
//! uncorrelated for >2TB of data *per seed*.
//! - `AESRandRng`: An AES-based RNG (currently only available on some aarch64 platforms). This is even faster than SFC64, but has a smaller state that makes it less suitable
//! for highly parallel applications.

pub(crate) mod backends;

#[cfg(feature = "bench")]
pub mod threefry2x64;
#[cfg(not(feature = "bench"))]
mod threefry2x64;

#[cfg(feature = "bench")]
pub mod squares;
#[cfg(not(feature = "bench"))]
mod squares;

/// Structs and implementations for the SFC64 PRNG.
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
