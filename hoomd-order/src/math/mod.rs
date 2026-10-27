// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Mathematical operations used to compute order parameters.
//!
//! When available, *hoomd-rs* uses existing crates that implement
//! mathematical operations. [`hoomd-order::math`] contains implementations
//! of operations where there are no crates or their existing implementations
//! are not suitable to compute order parameters of soft-matter simulations.
//!
//! [`hoomd-order::math`]: crate::math

mod spherical_harmonic;
pub use spherical_harmonic::{SphericalHarmonic, SphericalHarmonicOutputs};
